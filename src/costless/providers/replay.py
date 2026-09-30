"""Record real model responses once, replay them deterministically afterwards.

Recordings are JSONL, one line per (request, repeat)::

    {"key": "<sha256>", "repeat": 0, "request": {...}, "completion": {...}}

Replay picks the recording for the current attempt's repeat index, falling back
to ``repeat % recorded``, so N-repeat runs replay the recorded variance rather
than N copies of one answer. Unknown requests raise ReplayMissError: a replayed
run never silently calls a real model.
"""

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from costless.context import current_attempt
from costless.errors import ReplayMissError
from costless.providers.base import Completion, CompletionRequest, Provider


class ReplayProvider(Provider):
    name = "replay"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._recordings: dict[str, dict[int, Completion]] = defaultdict(dict)
        self._calls: dict[str, int] = defaultdict(int)
        if not path.is_file():
            msg = f"recording file not found: {path}"
            raise ReplayMissError(msg)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                completion = Completion.model_validate(entry["completion"])
                self._recordings[entry["key"]][int(entry["repeat"])] = completion
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                msg = f"{path}:{lineno}: malformed recording: {exc}"
                raise ReplayMissError(msg) from exc

    async def _complete(self, request: CompletionRequest) -> Completion:
        key = request.fingerprint()
        recorded = self._recordings.get(key)
        if not recorded:
            msg = f"no recording for request {key[:12]} (model={request.model}) in {self._path}"
            raise ReplayMissError(msg)

        scope = current_attempt()
        if scope is not None:
            repeat = scope.repeat
        else:
            repeat = self._calls[key]
            self._calls[key] += 1

        if repeat in recorded:
            return recorded[repeat]
        ordered = [recorded[r] for r in sorted(recorded)]
        return ordered[repeat % len(ordered)]


class RecordingProvider(Provider):
    """Wraps a real provider and appends every call to a JSONL recording file."""

    def __init__(self, inner: Provider, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._lock = asyncio.Lock()
        self._calls: dict[str, int] = defaultdict(int)
        self.name = inner.name

    async def _complete(self, request: CompletionRequest) -> Completion:
        started = time.perf_counter()
        completion = await self._inner._complete(request)  # noqa: SLF001 - metered by our complete()
        if completion.latency_ms == 0:
            elapsed = (time.perf_counter() - started) * 1000
            completion = completion.model_copy(update={"latency_ms": elapsed})
        key = request.fingerprint()
        scope = current_attempt()
        async with self._lock:
            if scope is not None:
                repeat = scope.repeat
            else:
                repeat = self._calls[key]
                self._calls[key] += 1
            entry = {
                "key": key,
                "repeat": repeat,
                "request": request.model_dump(mode="json"),
                "completion": completion.model_dump(mode="json"),
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return completion
