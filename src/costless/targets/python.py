"""In-process Python target: ``package.module:function``.

The function receives ``case.input`` and may return a ``str``, a
:class:`TargetResponse`, or any JSON-serialisable value (serialised as the
output). It may be sync or async; sync functions run in a worker thread.
Model calls made through costless providers are metered automatically.
"""

import asyncio
import importlib
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from costless.errors import ConfigError, TargetError
from costless.models import Case
from costless.targets.base import TargetResponse


class PythonTarget:
    def __init__(self, ref: str, *, timeout_s: float, search_path: Path | None = None) -> None:
        self._ref = ref
        self._timeout_s = timeout_s
        self._fn = _import_callable(ref, search_path)

    @property
    def name(self) -> str:
        return f"python:{self._ref}"

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    async def invoke(self, case: Case) -> TargetResponse:
        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(case.input)
        else:
            result = await asyncio.to_thread(self._fn, case.input)
        return _normalise(result)


def _import_callable(ref: str, search_path: Path | None) -> Callable[..., Any]:
    module_name, _, attr_path = ref.partition(":")
    if search_path is not None and str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"cannot import target module {module_name!r}: {exc}"
        raise ConfigError(msg) from exc
    for attr in attr_path.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            msg = f"target {ref!r}: {module_name} has no attribute {attr_path!r}"
            raise ConfigError(msg) from exc
    if not callable(obj):
        msg = f"target {ref!r} is not callable"
        raise ConfigError(msg)
    return obj  # type: ignore[no-any-return]


def _normalise(result: Any) -> TargetResponse:  # noqa: ANN401
    if isinstance(result, TargetResponse):
        return result
    if isinstance(result, str):
        return TargetResponse(output=result)
    try:
        return TargetResponse(output=json.dumps(result))
    except TypeError as exc:
        msg = f"target returned a non-serialisable {type(result).__name__}"
        raise TargetError(msg) from exc
