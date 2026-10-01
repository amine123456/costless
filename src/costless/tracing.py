"""OpenTelemetry instrumentation.

Spans
-----
``costless run``        one per run, with the run's metadata and summary
``costless attempt``    one per (case, repeat), with the outcome and cost
``costless score``      the scorers of one attempt (LLM-judge calls nest here)
``chat {model}``        every model call, following the OpenTelemetry GenAI
                        semantic conventions (``gen_ai.*`` attributes), plus
                        ``costless.cost_usd`` when the call can be priced

Metrics
-------
``gen_ai.client.operation.duration``  histogram (s), GenAI semantic conventions
``costless.run.*``                    gauges recorded once per run: quality,
                                      failure rate, cost, eval cost, p95 latency
``costless.gate.passed``              1 or 0, recorded by ``compare`` / ``ci check``

Prompts and completions are never recorded: eval inputs often contain
production data, and the GenAI conventions make content capture opt-in.

Nothing is exported unless tracing is configured (``OTEL_EXPORTER_OTLP_ENDPOINT``
or ``COSTLESS_OTEL=1``); without that, the OpenTelemetry API is a no-op.
Exporting needs the optional dependencies: ``pip install "costless[otel]"``.
"""

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from costless import __version__
from costless.errors import ConfigError

if TYPE_CHECKING:
    from costless.models import RunResult, Usage
    from costless.pricing import PricingTable

tracer = trace.get_tracer("costless", __version__)
meter = metrics.get_meter("costless", __version__)

_operation_duration = meter.create_histogram(
    "gen_ai.client.operation.duration",
    unit="s",
    description="GenAI operation duration.",
    explicit_bucket_boundaries_advisory=[
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ],
)
_run_gauges = {
    name: meter.create_gauge(f"costless.run.{name}", description=description)
    for name, description in {
        "quality": "Mean quality score of the run (0-1).",
        "failure_rate": "Share of attempts that did not pass (0-1).",
        "cost_usd": "Cost of the system under test for the whole run, in USD.",
        "cost_per_case_usd": "Mean cost of one attempt, in USD.",
        "eval_cost_usd": "Cost of LLM-judge calls for the whole run, in USD.",
        "latency_p95_ms": "95th percentile latency of the target, in milliseconds.",
        "attempts": "Number of attempts in the run.",
    }.items()
}
_gate_gauge = meter.create_gauge(
    "costless.gate.passed", description="1 if the quality gate passed, else 0."
)

# Provider names as defined by the GenAI semantic conventions.
SEMCONV_PROVIDER = {
    "anthropic": "anthropic",
    "gemini": "gcp.gemini",
    "xai": "x_ai",
    "openai": "openai",
}

_pricing: ContextVar["PricingTable | None"] = ContextVar("costless_pricing", default=None)


@contextmanager
def pricing_scope(table: "PricingTable") -> Iterator[None]:
    """Make ``table`` available to provider spans, so they can carry the call's cost."""
    token = _pricing.set(table)
    try:
        yield
    finally:
        _pricing.reset(token)


# ------------------------------------------------------------------ model calls


class ModelCall:
    """The span of one model call; ``record`` adds the response details."""

    def __init__(self, span: Span, metric_attributes: dict[str, Any]) -> None:
        self.span = span
        self.metric_attributes = metric_attributes

    def record(self, usage: "Usage", stop_reason: str | None) -> None:
        span = self.span
        # The conventions count cached tokens as part of the input tokens.
        span.set_attribute(
            "gen_ai.usage.input_tokens",
            usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        )
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        if usage.cache_read_tokens:
            span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.cache_read_tokens)
        if usage.cache_write_tokens:
            span.set_attribute("gen_ai.usage.cache_write.input_tokens", usage.cache_write_tokens)
        span.set_attribute("gen_ai.response.model", usage.model)
        self.metric_attributes["gen_ai.response.model"] = usage.model
        if stop_reason is not None:
            span.set_attribute("gen_ai.response.finish_reasons", [stop_reason])
        table = _pricing.get()
        if table is not None:
            breakdown = table.cost([usage])
            if breakdown.complete:
                span.set_attribute("costless.cost_usd", float(breakdown.usd))


@contextmanager
def model_call(
    provider: str, model: str, *, max_tokens: int, temperature: float | None
) -> Iterator[ModelCall]:
    """Span + duration metric for one model call, per the GenAI conventions."""
    metric_attributes: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": SEMCONV_PROVIDER.get(provider, provider),
        "gen_ai.request.model": model,
    }
    span_attributes = {**metric_attributes, "gen_ai.request.max_tokens": max_tokens}
    if temperature is not None:
        span_attributes["gen_ai.request.temperature"] = temperature
    started = time.perf_counter()
    with tracer.start_as_current_span(
        f"chat {model}",
        kind=SpanKind.CLIENT,
        attributes=span_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        call = ModelCall(span, metric_attributes)
        try:
            yield call
        except Exception as exc:
            error_type = type(exc).__qualname__
            span.set_attribute("error.type", error_type)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            metric_attributes["error.type"] = error_type
            raise
        finally:
            _operation_duration.record(time.perf_counter() - started, metric_attributes)


# ------------------------------------------------------------------ runs


def run_attributes(target: str, repeats: int, datasets: list[str]) -> dict[str, Any]:
    return {
        "costless.target": target,
        "costless.repeats": repeats,
        "costless.datasets": datasets,
    }


def record_run(result: "RunResult") -> None:
    """Attach the summary to the current span and record the run gauges."""
    s = result.summary
    m = result.metadata
    labels = {
        "costless.target": m.target,
        "costless.git.ref": m.git_ref or "unknown",
        "costless.datasets": ",".join(d.name for d in m.datasets),
    }
    values: dict[str, float | None] = {
        "quality": s.quality_mean,
        "failure_rate": s.failure_rate,
        "cost_usd": s.cost_total_usd,
        "cost_per_case_usd": s.cost_per_case_usd,
        "eval_cost_usd": s.eval_cost_total_usd,
        "latency_p95_ms": s.latency_p95_ms,
        "attempts": float(s.attempts),
    }
    span = trace.get_current_span()
    span.set_attribute("costless.run_id", m.run_id)
    if m.git_sha:
        span.set_attribute("costless.git.sha", m.git_sha)
    for name, value in values.items():
        if value is None:
            continue
        span.set_attribute(f"costless.{name}", value)
        _run_gauges[name].set(value, labels)


def record_gate(passed: bool, labels: Mapping[str, str]) -> None:
    _gate_gauge.set(1 if passed else 0, dict(labels))


# ------------------------------------------------------------------ setup


@dataclass
class Telemetry:
    """Handle on configured exporters; ``shutdown`` flushes buffered data."""

    _shutdown: list[Any]

    def shutdown(self) -> None:
        for provider in self._shutdown:
            provider.shutdown()


def configure(env: Mapping[str, str] | None = None) -> Telemetry | None:
    """Set up OTLP/HTTP export if the environment asks for it.

    Standard OTEL_* variables (endpoint, headers, service name, resource
    attributes) are honoured by the SDK itself.
    """
    env = os.environ if env is None else env
    wanted = env.get("COSTLESS_OTEL", "").lower() in {"1", "true", "yes"} or bool(
        env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if not wanted or env.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: PLC0415
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # noqa: PLC0415
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError as exc:
        msg = 'tracing is enabled but the exporter is not installed: pip install "costless[otel]"'
        raise ConfigError(msg) from exc

    resource = Resource.create(
        {SERVICE_NAME: env.get("OTEL_SERVICE_NAME", "costless"), "service.version": __version__}
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    return Telemetry([tracer_provider, meter_provider])
