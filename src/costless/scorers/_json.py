"""Helpers for scoring structured (JSON) outputs."""

import json
import re
from typing import Any

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n(?P<body>.*?)\n?```$", re.DOTALL)
_MISSING = object()


class PathNotFoundError(LookupError):
    pass


def parse_json_output(output: str) -> Any:  # noqa: ANN401
    """Parse a model output as JSON, tolerating a single surrounding Markdown code fence.

    Raises ValueError when the output is not valid JSON.
    """
    text = output.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    return json.loads(text)


def get_path(value: Any, path: str) -> Any:  # noqa: ANN401
    """Follow a dot path such as ``actions.0.owner`` through dicts and lists."""
    current = value
    for part in path.split("."):
        nxt: Any = _MISSING
        if isinstance(current, dict):
            nxt = current.get(part, _MISSING)
        elif isinstance(current, list) and part.lstrip("-").isdigit():
            index = int(part)
            if -len(current) <= index < len(current):
                nxt = current[index]
        if nxt is _MISSING:
            msg = f"path {path!r} not found (stopped at {part!r})"
            raise PathNotFoundError(msg)
        current = nxt
    return current


def as_text(value: Any) -> str:  # noqa: ANN401
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
