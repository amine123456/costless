# Observability

costless emits OpenTelemetry traces and metrics. It follows the
[GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
for model calls, so any OTLP backend can store and explore them.

## Quick start (local stack)

```bash
docker compose -f deploy/compose/docker-compose.yml up -d
pip install "costless[otel]"                      # OTLP exporter
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
costless run -c costless.yaml
open http://localhost:3000                        # Grafana: Explore → Tempo / Prometheus
```

The stack has four containers, all pinned versions and all bound to localhost:

| Service | Image | Port | Role |
|---|---|---|---|
| OpenTelemetry Collector | `otel/opentelemetry-collector-contrib:0.161.0` | 4317 / 4318 | receives OTLP; sends traces to Tempo and metrics to Prometheus |
| Tempo | `grafana/tempo:3.1.0` | 3200 | trace storage, 30-day retention |
| Prometheus | `prom/prometheus:v3.15.0` | 9090 | metrics storage, 90-day retention, remote-write receiver |
| Grafana | `grafana/grafana:13.2.3` | 3000 | UI, with Tempo and Prometheus already provisioned |

Grafana runs with anonymous admin access. That is convenient on a laptop and
must never be exposed. The AWS deployment in `deploy/terraform` is the hardened
version.

## Why Tempo + Prometheus + Grafana rather than Langfuse

Langfuse has the nicer LLM-specific UI: prompt management, a playground, and
per-generation views. This stack was chosen for the following reasons:

- **Operational weight.** Langfuse v3 needs Postgres, ClickHouse, Redis and S3.
  Tempo is a single binary backed by object storage, which is realistic to run
  on ECS Fargate with least-privilege IAM.
- **No lock-in.** costless only speaks OTLP. Pointing
  `OTEL_EXPORTER_OTLP_ENDPOINT` elsewhere sends the same data to Langfuse,
  Honeycomb, Datadog or Grafana Cloud. All of them accept OTLP, and the GenAI
  conventions are what they understand.
- **One place for trends.** Run-level quality, cost and latency are metrics, so
  they belong in Prometheus next to the rest of a platform's metrics, with
  Grafana alerting available.

## Traces

Each run produces one trace:

```
costless run                      run metadata and summary
├── costless attempt              one per (case, repeat): outcome, quality, cost
│   ├── chat claude-haiku-4-5     model calls of the system under test
│   └── costless score            scorers
│       └── chat gemini-3.5-flash-lite   LLM-judge calls, kept apart from the target's
└── ...
```

**Model-call spans** follow the GenAI conventions:

| Attribute | Notes |
|---|---|
| span name `chat {model}`, kind `CLIENT` | |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | `anthropic`, `gcp.gemini`, `x_ai`, `openai` |
| `gen_ai.request.model`, `gen_ai.response.model` | |
| `gen_ai.request.max_tokens`, `gen_ai.request.temperature` | temperature only when set |
| `gen_ai.usage.input_tokens` | **includes** cached tokens, as the conventions require |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | when non-zero |
| `gen_ai.usage.output_tokens` | includes reasoning tokens |
| `gen_ai.response.finish_reasons` | |
| `error.type` | on failure, with span status `ERROR` |
| `costless.cost_usd` | not part of the conventions: priced with the run's pricing table |

**Prompts and completions are not recorded.** Eval inputs often contain
production data, and the conventions make content capture opt-in. The full
outputs are in `run.json`.

**costless spans** carry these `costless.*` attributes:

| Span | Attributes |
|---|---|
| run | `run_id`, `git.sha`, `target`, `datasets`, `repeats`, plus the summary (`quality`, `failure_rate`, `cost_usd`, ...) |
| attempt | `case_id`, `repeat`, `passed`, `quality`, `latency_ms`, `cost_usd`, `eval_cost_usd`, `failed_scorers`; status `ERROR` when the target failed |

## Metrics

| Metric (Prometheus name) | Type | Labels |
|---|---|---|
| `gen_ai_client_operation_duration_seconds` | histogram | `gen_ai_operation_name`, `gen_ai_provider_name`, `gen_ai_request_model`, `gen_ai_response_model`, `error_type` |
| `costless_run_quality` | gauge | `costless_target`, `costless_git_ref`, `costless_datasets`, `service_name` |
| `costless_run_failure_rate` | gauge | same |
| `costless_run_cost_usd`, `costless_run_cost_per_case_usd`, `costless_run_eval_cost_usd` | gauge | same |
| `costless_run_latency_p95_ms`, `costless_run_attempts` | gauge | same |
| `costless_gate_passed` | gauge (1/0) | `costless_git_ref` |

Run gauges are recorded once per run, when the process exits. Each run is its
own OTel service instance, so query across runs with an aggregation, for
example:

```promql
max by (costless_target) (costless_run_quality)
```

## Configuration

| Variable | Effect |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | enables export (OTLP/HTTP), e.g. `http://localhost:4318` |
| `COSTLESS_OTEL=1` | enables export with the SDK's default endpoint |
| `OTEL_SERVICE_NAME` | defaults to `costless` |
| `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_RESOURCE_ATTRIBUTES`, ... | standard SDK variables, honoured as usual |
| `OTEL_SDK_DISABLED=true` | turns everything off |

Without these, nothing is exported and the OpenTelemetry API costs nothing.
Exporting requires the `otel` extra (`pip install "costless[otel]"`).
