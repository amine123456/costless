"""The system under test, seen from costless: a case input goes in, an output comes out."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from costless.models import Case, Usage


class TargetResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output: str
    usage: tuple[Usage, ...] = ()


class Target(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def timeout_s(self) -> float: ...

    async def invoke(self, case: Case) -> TargetResponse: ...
