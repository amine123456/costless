import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import pytest

from costless.context import attempt_scope
from costless.errors import ConfigError, ProviderError
from costless.models import Usage
from costless.providers import (
    CompletionRequest,
    Message,
    RecordingProvider,
    ReplayProvider,
    model_from_env,
    provider_from_env,
)
from costless.providers.anthropic import AnthropicProvider
from costless.providers.openai_compat import OpenAICompatibleProvider

Handler = Callable[[httpx2.Request], httpx2.Response]

REQUEST = CompletionRequest(
    model="claude-opus-5-5",
    system="You summarise incidents.",
    messages=(Message(role="user", content="db down"),),
    max_tokens=256,
)


def anthropic_provider(handler: Handler) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="test",
        base_url="https://anthropic.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(client=client)


def anthropic_message(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": '{"severity": '},
            {"type": "text", "text": '"SEV1"}'},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 50,
        },
    }
    body.update(overrides)
    return body


class TestAnthropic:
    def test_request_and_response_mapping(self) -> None:
        seen: list[dict[str, object]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            assert request.url.path == "/v1/messages"
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json=anthropic_message())

        provider = anthropic_provider(handler)

        async def go() -> tuple[str, list[Usage], str | None]:
            with attempt_scope("c", 0) as scope:
                completion = await provider.complete(REQUEST)
                return completion.text, scope.usage, completion.stop_reason

        text, usage, stop_reason = asyncio.run(go())

        assert seen[0]["system"] == "You summarise incidents."
        assert seen[0]["messages"] == [{"role": "user", "content": "db down"}]
        assert seen[0]["max_tokens"] == 256
        assert "temperature" not in seen[0]  # never sent unless asked for
        assert "stop_sequences" not in seen[0]
        assert text == '{"severity": "SEV1"}'  # text blocks only, thinking ignored
        assert stop_reason == "end_turn"
        assert usage == [
            Usage(
                provider="anthropic",
                model="claude-opus-5-5",
                input_tokens=120,
                output_tokens=30,
                cache_read_tokens=1000,
                cache_write_tokens=50,
            )
        ]

    def test_optional_parameters_are_forwarded_when_set(self) -> None:
        seen: list[dict[str, object]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json=anthropic_message())

        request = REQUEST.model_copy(update={"temperature": 0.0, "stop": ("END",)})
        asyncio.run(anthropic_provider(handler).complete(request))
        assert seen[0]["temperature"] == 0.0
        assert seen[0]["stop_sequences"] == ["END"]

    def test_refusal_is_recorded(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=anthropic_message(content=[], stop_reason="refusal"))

        completion = asyncio.run(anthropic_provider(handler).complete(REQUEST))
        assert (completion.text, completion.stop_reason) == ("", "refusal")

    def test_api_errors_become_provider_errors(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                400,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad"},
                },
            )

        with pytest.raises(ProviderError, match="anthropic API error 400"):
            asyncio.run(anthropic_provider(handler).complete(REQUEST))


def openai_provider(handler: Handler, max_retries: int = 2) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="xai",
        base_url="https://api.example.test/v1/",
        api_key="secret",
        max_retries=max_retries,
        backoff_base_s=0.001,
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def chat_completion(**usage: int) -> dict[str, object]:
    return {
        "model": "grok-4",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 7},
    }


class TestOpenAICompatible:
    def test_request_and_response_mapping(self) -> None:
        seen: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append(request)
            return httpx2.Response(
                200,
                json=chat_completion(
                    prompt_tokens=100,
                    completion_tokens=7,
                    prompt_tokens_details={"cached_tokens": 60},  # type: ignore[arg-type]
                ),
            )

        completion = asyncio.run(openai_provider(handler).complete(REQUEST))
        body = json.loads(seen[0].content)

        assert str(seen[0].url) == "https://api.example.test/v1/chat/completions"
        assert seen[0].headers["authorization"] == "Bearer secret"
        assert body["messages"][0] == {"role": "system", "content": "You summarise incidents."}
        assert body["messages"][1] == {"role": "user", "content": "db down"}
        assert "temperature" not in body
        assert completion.text == "hi"
        assert completion.stop_reason == "stop"
        assert completion.usage == Usage(
            provider="xai",
            model="grok-4",
            input_tokens=40,
            output_tokens=7,
            cache_read_tokens=60,
        )

    @pytest.mark.parametrize(
        "raw",
        [
            # reasoning counted inside completion_tokens (OpenAI convention)
            {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
            # reasoning reported outside completion_tokens, but inside total_tokens
            {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
        ],
    )
    def test_reasoning_tokens_are_billed_as_output(self, raw: dict[str, object]) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=chat_completion(**raw))  # type: ignore[arg-type]

        completion = asyncio.run(openai_provider(handler).complete(REQUEST))
        assert completion.usage.output_tokens == 50

    def test_retries_rate_limits_then_succeeds(self) -> None:
        responses = [
            httpx2.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"}),
            httpx2.Response(503, json={"error": {"message": "overloaded"}}),
            httpx2.Response(200, json=chat_completion()),
        ]

        def handler(request: httpx2.Request) -> httpx2.Response:
            return responses.pop(0)

        assert asyncio.run(openai_provider(handler).complete(REQUEST)).text == "hi"
        assert responses == []

    def test_google_retry_info_and_list_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        google_429 = [
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota.\n* Quota exceeded for metric",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"}
                    ],
                }
            }
        ]
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("costless.providers.openai_compat.asyncio.sleep", fake_sleep)
        google_429[0]["error"]["details"][0]["retryDelay"] = "37s"  # type: ignore[index]
        provider = openai_provider(
            lambda request: httpx2.Response(429, json=google_429), max_retries=1
        )
        with pytest.raises(ProviderError, match=r"429: You exceeded your current quota\.$"):
            asyncio.run(provider.complete(REQUEST))
        assert sleeps == [37.0]  # the server's delay, not exponential backoff

    def test_gives_up_after_max_retries(self) -> None:
        calls = 0

        def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal calls
            calls += 1
            return httpx2.Response(503, json={"error": {"message": "overloaded"}})

        with pytest.raises(ProviderError, match="xai API error 503: overloaded"):
            asyncio.run(openai_provider(handler, max_retries=2).complete(REQUEST))
        assert calls == 3

    def test_client_errors_are_not_retried(self) -> None:
        calls = 0

        def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal calls
            calls += 1
            return httpx2.Response(401, text="unauthorized")

        with pytest.raises(ProviderError, match="401: unauthorized"):
            asyncio.run(openai_provider(handler).complete(REQUEST))
        assert calls == 1

    def test_transport_errors_are_retried(self) -> None:
        calls = 0

        def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx2.ConnectError("refused", request=request)
            return httpx2.Response(200, json=chat_completion())

        assert asyncio.run(openai_provider(handler).complete(REQUEST)).text == "hi"

        def always_down(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("refused", request=request)

        with pytest.raises(ProviderError, match="unreachable after 3 attempts"):
            asyncio.run(openai_provider(always_down).complete(REQUEST))

    def test_malformed_response(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json={"choices": []})

        with pytest.raises(ProviderError, match="unexpected response shape"):
            asyncio.run(openai_provider(handler).complete(REQUEST))


class TestFactory:
    def test_default_is_anthropic(self) -> None:
        assert isinstance(provider_from_env({"ANTHROPIC_API_KEY": "k"}), AnthropicProvider)
        assert model_from_env({}) == "claude-opus-5-5"

    def test_xai(self) -> None:
        env = {"COSTLESS_PROVIDER": "xai", "XAI_API_KEY": "k"}
        provider = provider_from_env(env)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.name == "xai"
        assert model_from_env(env) == "grok-4.7"
        assert model_from_env({**env, "COSTLESS_MODEL": "grok-other"}) == "grok-other"

    def test_gemini(self) -> None:
        env = {"COSTLESS_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}
        provider = provider_from_env(env)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.name == "gemini"
        assert model_from_env(env) == "gemini-3.5-flash"

    def test_openai_needs_explicit_model(self) -> None:
        env = {"COSTLESS_PROVIDER": "openai", "OPENAI_API_KEY": "k"}
        assert isinstance(provider_from_env(env), OpenAICompatibleProvider)
        with pytest.raises(ConfigError, match="set COSTLESS_MODEL"):
            model_from_env(env)

    def test_missing_key(self) -> None:
        with pytest.raises(ConfigError, match="needs XAI_API_KEY"):
            provider_from_env({"COSTLESS_PROVIDER": "xai"})

    def test_unknown_provider(self) -> None:
        with pytest.raises(ConfigError, match="unknown COSTLESS_PROVIDER 'bard'"):
            provider_from_env({"COSTLESS_PROVIDER": "bard"})

    def test_replay_and_record(self, tmp_path: Path) -> None:
        recording = tmp_path / "rec.jsonl"
        recording.write_text("")
        replay = provider_from_env(
            {"COSTLESS_PROVIDER": "replay", "COSTLESS_REPLAY_FILE": str(recording)}
        )
        assert isinstance(replay, ReplayProvider)
        with pytest.raises(ConfigError, match="needs COSTLESS_REPLAY_FILE"):
            provider_from_env({"COSTLESS_PROVIDER": "replay"})

        recorder = provider_from_env(
            {"ANTHROPIC_API_KEY": "k", "COSTLESS_RECORD_FILE": str(tmp_path / "out.jsonl")}
        )
        assert isinstance(recorder, RecordingProvider)
        assert recorder.name == "anthropic"

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("COSTLESS_MAX_RETRIES", "lots", "must be an integer"),
            ("COSTLESS_MAX_RETRIES", "-1", "must be >= 0"),
            ("COSTLESS_TIMEOUT_S", "soon", "must be a number"),
            ("COSTLESS_TIMEOUT_S", "0", "must be > 0"),
        ],
    )
    def test_invalid_numeric_settings(self, key: str, value: str, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            provider_from_env({"ANTHROPIC_API_KEY": "k", key: value})
