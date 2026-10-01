"""Provider for OpenAI-compatible Chat Completions APIs (xAI, OpenAI, vLLM, Ollama, ...)."""

import asyncio
import random
from typing import Any

import httpx2

from costless.errors import ProviderError
from costless.models import Usage
from costless.providers.base import Completion, CompletionRequest, Provider

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_BACKOFF_S = 30.0


class OpenAICompatibleProvider(Provider):
    """Calls ``POST {base_url}/chat/completions``.

    Retries 408/409/429/5xx and transport errors with exponential backoff and
    jitter, honouring ``Retry-After`` when the server sends it.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        max_retries: int = 3,
        timeout_s: float = 120.0,
        client: httpx2.AsyncClient | None = None,
        backoff_base_s: float = 0.5,
    ) -> None:
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._client = client or httpx2.AsyncClient(timeout=timeout_s)

    async def _complete(self, request: CompletionRequest) -> Completion:
        body = _request_body(request)
        response = await self._post_with_retries(body)
        try:
            doc = response.json()
            choice = doc["choices"][0]
            text = choice["message"].get("content") or ""
            usage = _usage(self.name, doc.get("model", request.model), doc["usage"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            msg = f"{self.name}: unexpected response shape: {exc!r}"
            raise ProviderError(msg) from exc
        return Completion(
            text=text, usage=usage, latency_ms=0, stop_reason=choice.get("finish_reason")
        )

    async def _post_with_retries(self, body: dict[str, Any]) -> httpx2.Response:
        attempt = 0
        while True:
            try:
                response = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx2.TransportError as exc:
                if attempt >= self._max_retries:
                    msg = f"{self.name} API unreachable after {attempt + 1} attempts: {exc}"
                    raise ProviderError(msg) from exc
                await asyncio.sleep(self._backoff(attempt, None))
                attempt += 1
                continue

            if response.status_code < 400:
                return response
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                await asyncio.sleep(self._backoff(attempt, response.headers.get("retry-after")))
                attempt += 1
                continue
            msg = f"{self.name} API error {response.status_code}: {_error_message(response)}"
            raise ProviderError(msg)

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(float(retry_after), _MAX_BACKOFF_S)
            except ValueError:
                pass  # HTTP-date form; fall back to exponential backoff
        delay: float = self._backoff_base_s * (2**attempt)
        return float(min(delay + random.uniform(0, delay / 2), _MAX_BACKOFF_S))  # noqa: S311


def _request_body(request: CompletionRequest) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if request.system is not None:
        messages.append({"role": "system", "content": request.system})
    messages.extend({"role": m.role, "content": m.content} for m in request.messages)
    body: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.stop:
        body["stop"] = list(request.stop)
    return body


def _usage(provider: str, model: str, raw: dict[str, Any]) -> Usage:
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    total = int(raw.get("total_tokens") or 0)
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    # Providers disagree on whether completion_tokens includes reasoning tokens.
    # Both are billed as output, and total_tokens always covers them, so take
    # whichever is larger: this never under-counts billed output.
    output = max(completion, total - prompt)
    return Usage(
        provider=provider,
        model=model,
        # prompt_tokens includes cached tokens; report them separately, as Anthropic does.
        input_tokens=max(prompt - cached, 0),
        output_tokens=output,
        cache_read_tokens=cached,
    )


def _error_message(response: httpx2.Response) -> str:
    try:
        doc = response.json()
    except ValueError:
        return response.text[:300] or "(empty body)"
    error = doc.get("error") if isinstance(doc, dict) else None
    if isinstance(error, dict) and "message" in error:
        return str(error["message"])
    if isinstance(error, str):
        return error
    return str(doc)[:300]
