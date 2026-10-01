import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from costless import tracing
from costless.config import load_config
from costless.models import Usage
from costless.pricing import PricingTable
from costless.providers import Completion, CompletionRequest, Message, Provider
from costless.runner import run_suite
from tests.conftest import WriteFile

SPANS = InMemorySpanExporter()
METRICS = InMemoryMetricReader()


@pytest.fixture(scope="module", autouse=True)
def _sdk() -> None:
    # The global providers can be set once per process; costless' module-level
    # tracer and meter are proxies that start delegating as soon as they are.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SPANS))
    trace.set_tracer_provider(provider)
    metrics.set_meter_provider(MeterProvider(metric_readers=[METRICS]))


@pytest.fixture(autouse=True)
def _clear() -> Iterator[None]:
    SPANS.clear()
    yield


class FakeAnthropic(Provider):
    name = "anthropic"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def _complete(self, request: CompletionRequest) -> Completion:
        if self.fail:
            msg = "overloaded"
            raise RuntimeError(msg)
        return Completion(
            text='{"severity": "SEV1"}',
            usage=Usage(
                provider="anthropic",
                model="claude-haiku-4-5",
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=1000,
                cache_write_tokens=10,
            ),
            latency_ms=5,
            stop_reason="end_turn",
        )


REQUEST = CompletionRequest(
    model="claude-haiku-4-5", messages=(Message(role="user", content="hi"),), max_tokens=64
)


def spans_named(prefix: str) -> list[ReadableSpan]:
    return [s for s in SPANS.get_finished_spans() if s.name.startswith(prefix)]


def metric_points(name: str) -> list[Any]:
    data = METRICS.get_metrics_data()
    points: list[Any] = []
    for resource in data.resource_metrics if data else ():
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


def test_model_call_follows_genai_conventions() -> None:
    with tracing.pricing_scope(PricingTable.default()):
        asyncio.run(FakeAnthropic().complete(REQUEST))

    (span,) = spans_named("chat ")
    assert span.name == "chat claude-haiku-4-5"
    assert span.kind == SpanKind.CLIENT
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-haiku-4-5"
    assert attrs["gen_ai.request.max_tokens"] == 64
    assert "gen_ai.request.temperature" not in attrs
    assert attrs["gen_ai.response.model"] == "claude-haiku-4-5"
    assert attrs["gen_ai.usage.input_tokens"] == 1110  # cached tokens included, per the spec
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 1000
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["gen_ai.response.finish_reasons"] == ("end_turn",)
    # Haiku 4.5: 100 x $1 + 20 x $5 + 1000 x $0.10 + 10 x $1.25 per MTok
    assert attrs["costless.cost_usd"] == pytest.approx(0.0003125)

    durations = metric_points("gen_ai.client.operation.duration")
    assert any(
        dict(p.attributes).get("gen_ai.response.model") == "claude-haiku-4-5" for p in durations
    )


def test_provider_names_are_mapped_to_semconv_values() -> None:
    assert tracing.SEMCONV_PROVIDER["gemini"] == "gcp.gemini"
    assert tracing.SEMCONV_PROVIDER["xai"] == "x_ai"


def test_failed_call_records_error_type() -> None:
    with pytest.raises(RuntimeError):
        asyncio.run(FakeAnthropic(fail=True).complete(REQUEST))
    (span,) = spans_named("chat ")
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "RuntimeError"
    assert any(
        dict(p.attributes).get("error.type") == "RuntimeError"
        for p in metric_points("gen_ai.client.operation.duration")
    )


APP = """
from costless.models import Usage
from costless.providers import CompletionRequest, Message
from tests.test_tracing import FakeAnthropic

provider = FakeAnthropic()

async def run(text):
    c = await provider.complete(CompletionRequest(
        model="claude-haiku-4-5", messages=(Message(role="user", content=text),)))
    return c.text
"""


def test_run_produces_a_span_tree_and_run_metrics(write: WriteFile, module_name: str) -> None:
    write(f"{module_name}.py", APP)
    write("cases.yaml", "- {id: a, input: x, expected: {severity: SEV1}}\n")
    config = write(
        "costless.yaml",
        f"""
        version: 1
        target: {{type: python, callable: "{module_name}:run"}}
        datasets: [{{path: cases.yaml}}]
        run: {{repeats: 2}}
        scorers: [{{type: exact_match, path: severity}}]
        """,
    )
    result = asyncio.run(run_suite(load_config(config)))

    (run_span,) = spans_named("costless run")
    attempts = spans_named("costless attempt")
    scores = spans_named("costless score")
    calls = spans_named("chat ")
    assert len(attempts) == 2
    assert len(scores) == 2
    assert len(calls) == 2
    trace_ids = {s.context.trace_id for s in [run_span, *attempts, *scores, *calls]}
    assert len(trace_ids) == 1  # one trace per run
    assert {a.parent.span_id for a in attempts if a.parent} == {run_span.context.span_id}
    attempt_ids = {a.context.span_id for a in attempts}
    assert {c.parent.span_id for c in calls if c.parent} <= attempt_ids

    run_attrs = dict(run_span.attributes or {})
    assert run_attrs["costless.run_id"] == result.metadata.run_id
    assert run_attrs["costless.quality"] == 1.0
    assert run_attrs["costless.repeats"] == 2
    attempt_attrs = dict(attempts[0].attributes or {})
    assert attempt_attrs["costless.case_id"] == "cases/a"
    assert attempt_attrs["costless.passed"] is True
    assert attempt_attrs["costless.cost_usd"] == pytest.approx(0.0003125)

    quality = metric_points("costless.run.quality")
    assert any(p.value == 1.0 for p in quality)


def test_configure_is_a_no_op_without_otel_settings() -> None:
    assert tracing.configure({}) is None
    assert (
        tracing.configure({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x", "OTEL_SDK_DISABLED": "true"})
        is None
    )
