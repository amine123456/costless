"""Test doubles shared by several test modules."""

from collections.abc import Callable

from costless.models import Usage
from costless.providers import Completion, CompletionRequest, Provider


class ScriptedProvider(Provider):
    """Answers every request with ``reply(prompt_text)``; meters a fixed usage."""

    name = "scripted"

    def __init__(self, reply: Callable[[str], str], model: str = "judge-model") -> None:
        self._reply = reply
        self._model = model
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    async def _complete(self, request: CompletionRequest) -> Completion:
        prompt = request.messages[-1].content
        self.prompts.append(prompt)
        self.systems.append(request.system)
        return Completion(
            text=self._reply(prompt),
            usage=Usage(provider=self.name, model=self._model, input_tokens=100, output_tokens=20),
            latency_ms=1.0,
        )


def verdict(score: int, reasoning: str = "ok") -> str:
    return f'{{"reasoning": "{reasoning}", "score": {score}}}'
