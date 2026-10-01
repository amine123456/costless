"""Provider for OpenAI-compatible Chat Completions APIs (xAI, OpenAI, vLLM, Ollama, ...)."""

import asyncio
import random
from typing import Any

import httpx2

from costless.errors import ProviderError
from costless.models import Usage
from costless.providers.base import Completion, CompletionRequest, Provider

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_BACKOFF_S = 60.0
_GOOGLE_RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"


class OpenAICompatibleProvider(Provider):
    """Calls ``POST {base_url}/chat/completions``.

    Retries 408/409/429/5xx and transport errors with exponential backoff and
    jitter. A server-provided delay (``Retry-After``, or Google's ``RetryInfo``)
    takes precedence, capped at 60 seconds.
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
                await asyncio.sleep(self._backoff(attempt, _server_retry_delay(response)))
                attempt += 1
                continue
            msg = f"{self.name} API error {response.status_code}: {_error_message(response)}"
            raise ProviderError(msg)

    def _backoff(self, attempt: int, server_delay_s: float | None) -> float:
        if server_delay_s is not None:
            return min(server_delay_s, _MAX_BACKOFF_S)
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


def _server_retry_delay(response: httpx2.Response) -> float | None:
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return max(float(header), 0.0)
        except ValueError:
            pass  # HTTP-date form: not worth parsing, fall back to backoff
    # Google APIs put the delay in the body: error.details[].retryDelay = "37s".
    error = _error_object(response)
    details = error.get("details") if error else None
    for detail in details if isinstance(details, list) else ():
        if isinstance(detail, dict) and detail.get("@type") == _GOOGLE_RETRY_INFO:
            raw = str(detail.get("retryDelay", ""))
            try:
                return max(float(raw.removesuffix("s")), 0.0)
            except ValueError:
                return None
    return None


def _error_object(response: httpx2.Response) -> dict[str, Any] | None:
    try:
        doc = response.json()
    except ValueError:
        return None
    if isinstance(doc, list) and doc:  # Google wraps the error object in a list
        doc = doc[0]
    error = doc.get("error") if isinstance(doc, dict) else None
    return error if isinstance(error, dict) else None


def _error_message(response: httpx2.Response) -> str:
    error = _error_object(response)
    if error is not None and "message" in error:
        return str(error["message"]).split("\n")[0][:300]
    try:
        doc = response.json()
    except ValueError:
        return response.text[:300] or "(empty body)"
    if isinstance(doc, list) and doc:
        doc = doc[0]
    error = doc.get("error") if isinstance(doc, dict) else None
    if isinstance(error, dict) and "message" in error:
        return str(error["message"])
    if isinstance(error, str):
        return error
    return str(doc)[:300]
