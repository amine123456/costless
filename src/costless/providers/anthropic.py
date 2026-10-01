"""Anthropic Messages API provider (the default)."""

import anthropic
from anthropic.types import MessageParam

from costless.errors import ProviderError
from costless.models import Usage
from costless.providers.base import Completion, CompletionRequest, Provider


class AnthropicProvider(Provider):
    """Calls ``POST /v1/messages`` through the official SDK.

    Credentials and endpoint follow the SDK's own resolution (``ANTHROPIC_API_KEY``,
    ``ANTHROPIC_BASE_URL``, ...). The SDK retries 408/409/429/5xx and connection
    errors with exponential backoff.

    Server-side refusal fallbacks are deliberately not enabled: in an evaluation,
    a request silently answered by a different model would corrupt the comparison.
    A refusal is recorded as ``stop_reason="refusal"`` and scored like any output.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        max_retries: int = 3,
        timeout_s: float = 120.0,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(
            max_retries=max_retries, timeout=timeout_s
        )

    async def _complete(self, request: CompletionRequest) -> Completion:
        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        # This SDK version no longer exposes sampling parameters (current models
        # reject them); a requested temperature is passed through for older models.
        extra_body = (
            {"temperature": request.temperature} if request.temperature is not None else None
        )
        try:
            response = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=messages,
                system=request.system if request.system is not None else anthropic.omit,
                stop_sequences=list(request.stop) if request.stop else anthropic.omit,
                extra_body=extra_body,
            )
        except anthropic.APIStatusError as exc:
            msg = f"anthropic API error {exc.status_code}: {exc.message}"
            raise ProviderError(msg) from exc
        except anthropic.APIConnectionError as exc:
            msg = f"anthropic API unreachable: {exc}"
            raise ProviderError(msg) from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        u = response.usage
        return Completion(
            text=text,
            usage=Usage(
                provider=self.name,
                model=response.model,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.cache_read_input_tokens or 0,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
            ),
            latency_ms=0,
            stop_reason=response.stop_reason,
        )
