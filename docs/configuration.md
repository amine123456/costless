# Configuration reference

costless is driven by one file, `costless.yaml`, plus the dataset files it points to.
Relative paths are resolved against the directory containing `costless.yaml`.

```yaml
version: 1

target:                      # the system under test
  type: python
  callable: app.summarizer:run
  timeout_s: 60

datasets:
  - path: evals/incidents.yaml
    tags: [smoke]            # optional: only cases carrying one of these tags

run:
  repeats: 5                 # attempts per case (non-determinism, see below)
  concurrency: 4             # attempts in flight at once

scorers:                     # applied to every case
  - type: json_schema
    schema: evals/summary.schema.json
  - type: exact_match
    path: severity
    weight: 2

pricing:                     # optional; Anthropic prices are built in
  strict: true               # fail the run if a model call cannot be priced
  file: pricing.yaml         # optional extra table
  models:                    # inline prices, USD per million tokens
    grok-4: {input: 0.0, output: 0.0, cache_read: 0.0}

budget:                      # optional
  max_run_usd: 5.00          # hard stop: remaining attempts are skipped
  max_case_usd: 0.02         # mean cost of one attempt
```

Unknown keys are rejected everywhere, so a typo fails loudly instead of being ignored.
`costless validate` checks the config, datasets and scorers without calling the target.

## Datasets

A dataset is a YAML or JSONL file of cases, versioned in git next to the code it tests.

```yaml
dataset: incidents           # optional, defaults to the file name
version: "3"                 # optional, recorded in run.json
cases:
  - id: inc-001              # unique within the dataset: [A-Za-z0-9._-]
    input: {report: "..."}   # anything; passed to the target as-is
    expected: {severity: SEV1}
    rubric: "Names the failing database and the customer impact."
    tags: [database, sev1]
    scorers:                 # optional, added to the config-level scorers
      - {type: regex, pattern: "(?i)postgres", path: summary}
```

In JSONL, each line is one case object and the dataset name is the file name.

Each dataset's SHA-256 is recorded in `run.json`, so a comparison between two runs
can tell whether the cases themselves changed. In a run, cases are identified as
`<dataset>/<id>`.

Before anything runs, costless checks that every case can be graded: at least one
scorer, and an `expected` value wherever a scorer needs one.

## Scorers

Every scorer returns a score in [0, 1] and a pass/fail verdict, plus a short
explanation when it fails.

| type | passes when | options |
|---|---|---|
| `exact_match` | the output equals `expected` | `path`, `case_sensitive` (true), `normalize_whitespace` (true) |
| `regex` | the pattern is found in the output (or, with `negate`, is not) | `pattern`, `path`, `ignore_case`, `negate` |
| `json_schema` | the output is JSON valid against the schema (draft 2020-12) | `schema`: inline object or a file path |

All scorers also accept `name` and `weight` (default 1).

`path` is a dot path into the JSON output, such as `actions.0.owner`. JSON outputs
wrapped in a single Markdown code fence are accepted.

`exact_match` compares values as follows:

- With `path`, it compares that field of the output with the same field of `expected`.
  If `expected` is a scalar, it compares with `expected` directly.
- Without `path`, a string `expected` is compared with the raw output text. Any other
  `expected` is compared with the parsed JSON output.

## Targets

### `python`

```yaml
target: {type: python, callable: package.module:function, timeout_s: 60}
```

The function receives `case.input` and returns one of:

- a string, used as the output;
- any JSON-serialisable value, serialised as the output;
- a `costless.targets.TargetResponse`, which is an output plus explicit token usage.

The function can be sync or async. Sync functions run in a worker thread, and a timed-out
sync call cannot be interrupted: its result is discarded, but the thread keeps running
until it returns.

Model calls made through a `costless.providers.Provider` report their token usage
to costless automatically.

### `subprocess`

For systems written in any language:

```yaml
target:
  type: subprocess
  command: [node, dist/cli.js, eval-target]
  cwd: .                     # optional
  env: {LOG_LEVEL: error}    # added to the inherited environment
  timeout_s: 120
```

costless starts the command once per attempt and writes one JSON object to its stdin:

```json
{"case_id": "inc-001", "input": {"report": "..."}}
```

The process must print one JSON object to stdout and exit 0:

```json
{"output": "... or an object ...", "usage": [{"model": "claude-sonnet-5-5", "input_tokens": 812, "output_tokens": 190}]}
```

`usage` is optional; each entry may also set `provider`, `cache_read_tokens` and
`cache_write_tokens`.

A non-zero exit status is an attempt error, and the last 500 characters of stderr
are kept in the error. When a timeout is hit, the process is killed.

## Model providers

An application under test calls models through a costless provider. That way
every call is metered, and switching providers is just a matter of setting
environment variables:

```python
from costless.providers import CompletionRequest, Message, model_from_env, provider_from_env

provider = provider_from_env()
completion = await provider.complete(
    CompletionRequest(
        model=model_from_env(),
        system="...",
        messages=(Message(role="user", content=text),),
    )
)
```

| Variable | Meaning |
|---|---|
| `COSTLESS_PROVIDER` | `anthropic` (default), `xai`, `openai` or `replay` |
| `COSTLESS_MODEL` | model id; defaults to `claude-opus-5-5` (anthropic) and `grok-4` (xai) |
| `COSTLESS_BASE_URL` | endpoint override for `xai` / `openai`, e.g. a local vLLM or Ollama server |
| `COSTLESS_RECORD_FILE` | append every real call to this JSONL recording |
| `COSTLESS_REPLAY_FILE` | recording to replay with `COSTLESS_PROVIDER=replay` |
| `COSTLESS_MAX_RETRIES` | retries on 408/409/429/5xx and connection errors (default 3) |
| `COSTLESS_TIMEOUT_S` | per-request timeout in seconds (default 120) |

Credentials:

- `ANTHROPIC_API_KEY`, or any credential the Anthropic SDK resolves.
- `XAI_API_KEY`.
- `OPENAI_API_KEY`.

The `anthropic` provider uses the official SDK; `xai` and `openai` use the Chat
Completions API.

Each provider handles a few details:

- **Sampling parameters:** only sent when they are set, because several current
  Claude models reject them.
- **Cached prompt tokens:** reported separately from regular input on every
  provider, so they can be priced correctly.
- **Refusals:** server-side refusal fallbacks are not enabled. Having a refused
  request silently answered by a different model would invalidate the comparison.
  Instead, a refusal is recorded as `stop_reason: refusal` and scored like any
  other output.

A recording made with `COSTLESS_RECORD_FILE` can be replayed with
`COSTLESS_PROVIDER=replay`. A replay makes no network calls and fails on any
request it has no recording for.

## Cost and budget

Each attempt's token usage is priced with a table of USD prices per million tokens.

- **Token types priced:** input, output, cache read and cache write.
- **Built-in prices:** first-party Anthropic rates, copied from the
  [Anthropic pricing page](https://platform.claude.com/docs/en/about-claude/pricing).
  The date they were retrieved is recorded in `src/costless/data/pricing.yaml`.
- **Not modelled:** the batch discount, fast mode, the US data-residency multiplier,
  and 1-hour cache writes. Cache writes are priced at the 5-minute rate.
- **Other models:** add them from the provider's own pricing page, under
  `pricing.models` or in a `pricing.file`. Keys can be globs such as `grok-4*`.
  The order of precedence is exact name, then the longest matching glob, then
  inline prices over file prices over built-in prices.

`run.json` records cost at three levels:

- per attempt (`cost_usd`)
- per case (`cost_mean_usd`)
- per run (`cost_total_usd`, plus `cost_per_case_usd`, the mean cost of one attempt)

A cost is `null` when any model involved has no price. Those models are listed
in `unpriced_models`.

`costless run` exits with status 1 when any of these is true:

- `pricing.strict` is on (the default) and a model call could not be priced.
  This prevents a missing price from making a run look free.
- The total cost exceeds `budget.max_run_usd`.
- The mean cost per attempt exceeds `budget.max_case_usd`.

`max_run_usd` is also enforced while the run is in progress. Once that much has
been spent, attempts that haven't started are skipped and recorded as skipped,
and the run fails.

## Outcomes and metrics

Each attempt ends in one of three outcomes:

- **Error:** the target raised an exception, timed out, or returned unusable output.
  The attempt scores 0 and does not pass.
- **Pass:** the attempt did not error and every scorer passed.
- **Fail:** everything else.

The `quality` of an attempt is the weighted mean of its scorer scores.

Per case, `run.json` records:

- the mean and sample variance of quality across repeats
- the pass rate
- the error count
- the mean latency
- the mean token counts
- a `flaky` flag, set when the case passed on some repeats and failed on others

Per run, it records:

- **Quality mean:** the mean of the case means, so every case carries the same weight.
- **Failure rate:** the share of attempts that did not pass.
- **Error rate.**
- **p50 and p95 latency** over all attempts, using linear interpolation.
- **Total tokens.**

Latency is the wall-clock time of the target call as seen by costless.

## run.json

`costless run` writes every attempt, not only aggregates, so any run can be
re-analysed later. The file also carries metadata:

- run id and timestamps
- costless version
- git commit and ref, taken from CI variables when available
- config hash
- target
- repeats
- dataset names, versions and hashes

`costless schema` prints the JSON Schema of the document. The file is versioned with
`schema_version`, and readers reject versions they do not understand.
