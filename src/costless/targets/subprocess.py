"""Language-agnostic target: an executable speaking JSON over stdin/stdout.

For every attempt costless starts the command, writes one JSON object to stdin::

    {"case_id": "inc-001", "input": ...}

and expects one JSON object on stdout::

    {"output": "..." | {...}, "usage": [{"model": "...", "input_tokens": 1, "output_tokens": 2}]}

``usage`` is optional. A non-zero exit status is an attempt error; stderr is
kept (truncated) in the error message. This is how non-Python systems, such as
a Node.js agent, are evaluated.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from costless.errors import TargetError
from costless.models import Case, Usage
from costless.targets.base import TargetResponse

_STDERR_TAIL = 500


class SubprocessTarget:
    def __init__(
        self,
        command: tuple[str, ...],
        *,
        timeout_s: float,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._timeout_s = timeout_s
        self._cwd = cwd
        self._env = {**os.environ, **(env or {})}

    @property
    def name(self) -> str:
        return "subprocess:" + " ".join(self._command)

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    async def invoke(self, case: Case) -> TargetResponse:
        payload = json.dumps({"case_id": case.id, "input": case.input}).encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                env=self._env,
            )
        except OSError as exc:
            msg = f"cannot start {self._command[0]!r}: {exc.strerror}"
            raise TargetError(msg) from exc

        try:
            stdout, stderr = await proc.communicate(payload)
        except asyncio.CancelledError:
            # Timeouts arrive as cancellation: never leave the child running.
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            tail = stderr.decode("utf-8", "replace").strip()[-_STDERR_TAIL:]
            msg = f"exited with status {proc.returncode}: {tail or '(no stderr)'}"
            raise TargetError(msg)
        return _parse(stdout)


def _parse(stdout: bytes) -> TargetResponse:
    try:
        doc: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        msg = f"stdout is not a JSON object: {exc.msg}"
        raise TargetError(msg) from exc
    if not isinstance(doc, dict) or "output" not in doc:
        msg = "stdout JSON must be an object with an 'output' field"
        raise TargetError(msg)
    output = doc["output"]
    try:
        usage = tuple(Usage.model_validate(u) for u in doc.get("usage") or ())
    except ValidationError as exc:
        msg = f"invalid 'usage' in target output: {exc}"
        raise TargetError(msg) from exc
    return TargetResponse(
        output=output if isinstance(output, str) else json.dumps(output), usage=usage
    )
