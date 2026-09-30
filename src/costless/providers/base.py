"""Provider-agnostic model interface.

Applications under test call ``provider.complete(request)``. Concrete providers
implement ``_complete``; the base class takes care of timing and usage reporting,
so every provider is metered the same way.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from costless.context import record_usage
from costless.models import Usage


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Model):
    role: Literal["user", "assistant"]
    content: str


class CompletionRequest(_Model):
    model: str
    messages: tuple[Message, ...] = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        """Stable hash of the request, used to key recordings."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Completion(_Model):
    text: str
    usage: Usage
    latency_ms: float = Field(ge=0)
    stop_reason: str | None = None


class Provider(ABC):
    """Base class for model providers."""

    name: str = "unknown"

    async def complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        completion = await self._complete(request)
        if completion.latency_ms == 0:
            elapsed = (time.perf_counter() - started) * 1000
            completion = completion.model_copy(update={"latency_ms": elapsed})
        record_usage(completion.usage)
        return completion

    @abstractmethod
    async def _complete(self, request: CompletionRequest) -> Completion: ...
