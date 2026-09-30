"""Per-attempt context shared between the runner and model providers.

The runner opens an attempt scope around every call to the system under test.
Providers report token usage into it, so costless sees every model call the
application makes without the application having to return usage explicitly.
Because this is a ContextVar, concurrent attempts never see each other's data.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from costless.models import Usage


@dataclass
class AttemptScope:
    case_id: str
    repeat: int
    usage: list[Usage] = field(default_factory=list)


_current: ContextVar[AttemptScope | None] = ContextVar("costless_attempt", default=None)


def current_attempt() -> AttemptScope | None:
    return _current.get()


@contextmanager
def attempt_scope(case_id: str, repeat: int) -> Iterator[AttemptScope]:
    scope = AttemptScope(case_id=case_id, repeat=repeat)
    token = _current.set(scope)
    try:
        yield scope
    finally:
        _current.reset(token)


def record_usage(usage: Usage) -> None:
    """Attach ``usage`` to the running attempt. A no-op outside an eval run."""
    scope = _current.get()
    if scope is not None:
        scope.usage.append(usage)
