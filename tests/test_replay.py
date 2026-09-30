import asyncio
import json
from pathlib import Path

import pytest

from costless.context import attempt_scope
from costless.errors import ReplayMissError
from costless.models import Usage
from costless.providers import (
    Completion,
    CompletionRequest,
    Message,
    Provider,
    RecordingProvider,
    ReplayProvider,
)

REQUEST = CompletionRequest(model="m", messages=(Message(role="user", content="hi"),))


class CountingProvider(Provider):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def _complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        return Completion(
            text=f"answer {self.calls}",
            usage=Usage(
                provider="fake", model=request.model, input_tokens=10, output_tokens=self.calls
            ),
            latency_ms=0,
        )


def write_recordings(path: Path, texts: list[str]) -> None:
    lines = [
        json.dumps(
            {
                "key": REQUEST.fingerprint(),
                "repeat": i,
                "request": REQUEST.model_dump(mode="json"),
                "completion": {
                    "text": text,
                    "usage": {"model": "m", "input_tokens": 5, "output_tokens": 7},
                    "latency_ms": 12.5,
                },
            }
        )
        for i, text in enumerate(texts)
    ]
    path.write_text("\n".join(lines) + "\n")


def test_fingerprint_is_stable_and_sensitive() -> None:
    same = CompletionRequest(model="m", messages=(Message(role="user", content="hi"),))
    other = same.model_copy(update={"temperature": 0.5})
    assert REQUEST.fingerprint() == same.fingerprint()
    assert REQUEST.fingerprint() != other.fingerprint()


def test_replay_follows_attempt_repeat_index(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recordings(path, ["a", "b"])
    provider = ReplayProvider(path)

    async def text_for(repeat: int) -> str:
        with attempt_scope("case", repeat):
            return (await provider.complete(REQUEST)).text

    assert [asyncio.run(text_for(r)) for r in range(4)] == ["a", "b", "a", "b"]


def test_replay_outside_a_run_cycles_through_recordings(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recordings(path, ["a", "b"])
    provider = ReplayProvider(path)

    texts = [asyncio.run(provider.complete(REQUEST)).text for _ in range(3)]
    assert texts == ["a", "b", "a"]


def test_replay_keeps_recorded_latency_and_reports_usage(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recordings(path, ["a"])
    provider = ReplayProvider(path)

    async def go() -> tuple[Completion, list[Usage]]:
        with attempt_scope("case", 0) as scope:
            completion = await provider.complete(REQUEST)
            return completion, scope.usage

    completion, usage = asyncio.run(go())
    assert completion.latency_ms == 12.5
    assert [u.output_tokens for u in usage] == [7]


def test_replay_miss_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    write_recordings(path, ["a"])
    other = CompletionRequest(model="m", messages=(Message(role="user", content="bye"),))
    with pytest.raises(ReplayMissError, match="no recording"):
        asyncio.run(ReplayProvider(path).complete(other))


@pytest.mark.parametrize("content", ["{not json\n", '{"key": "k"}\n'])
def test_malformed_recording_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "rec.jsonl"
    path.write_text(content)
    with pytest.raises(ReplayMissError, match="malformed recording"):
        ReplayProvider(path)


def test_missing_recording_file(tmp_path: Path) -> None:
    with pytest.raises(ReplayMissError, match="recording file not found"):
        ReplayProvider(tmp_path / "nope.jsonl")


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rec" / "calls.jsonl"
    inner = CountingProvider()
    recorder = RecordingProvider(inner, path)

    async def record() -> list[str]:
        texts = []
        for repeat in range(3):
            with attempt_scope("case", repeat):
                texts.append((await recorder.complete(REQUEST)).text)
        return texts

    recorded = asyncio.run(record())
    assert recorded == ["answer 1", "answer 2", "answer 3"]
    assert inner.calls == 3

    replay = ReplayProvider(path)

    async def replayed(repeat: int) -> str:
        with attempt_scope("case", repeat):
            return (await replay.complete(REQUEST)).text

    assert [asyncio.run(replayed(r)) for r in range(3)] == recorded
    assert all(
        json.loads(line)["completion"]["latency_ms"] > 0 for line in path.read_text().splitlines()
    )
