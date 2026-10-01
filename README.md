# costless

**A CI quality gate for LLM applications.** costless runs your eval suite on every merge request, compares quality, cost and latency against the main-branch baseline with statistics that separate real regressions from model noise, and blocks the merge when a threshold is breached.

> Status: early development. The sections below are filled in as each component lands; nothing here describes a feature that does not exist yet.

## What works today

- Versioned eval datasets in YAML or JSONL, with content hashes and tags.
- Deterministic scorers: exact match, regex and JSON Schema validation, with weights.
- An LLM-as-judge scorer with explicit rubrics and score anchors. Its prompt is
  hardened against injection from the output being graded, and its cost is
  tracked separately from the application's.
- Judge calibration against human-labelled examples: agreement rates with 95%
  Wilson intervals, Cohen's kappa and weighted kappa, plus judge bias and
  self-consistency. It can fail CI when the judge disagrees with humans.
- Two kinds of system under test: an in-process Python callable, or any executable
  that speaks JSON over stdin/stdout.
- A runner that executes every case N times with bounded concurrency, timeouts and
  per-attempt error capture. It reports the mean and variance per case and flags
  flaky cases.
- A provider-agnostic model interface, configured with environment variables:
  Anthropic (default), Google Gemini, xAI, any OpenAI-compatible endpoint, or a
  replay file.
  Every call is metered, with retries and backoff.
- Cost tracking: token counts priced from a configurable table (Anthropic, Gemini and
  xAI list prices built in), cost per attempt, per case and per run, and a budget gate.
  The budget also acts as a hard stop during a run.
- Record and replay of model responses, for deterministic offline tests.
- `run.json`, a versioned results document that keeps every attempt.

```bash
costless validate -c costless.yaml    # check config, datasets and scorers
costless calibrate -c costless.yaml   # how well does the LLM judge agree with humans?
costless run -c costless.yaml -n 5    # run every case 5 times, write .costless/run.json
                                      # exit 1 if the budget gate fails, 2 on config errors
```

The config file, dataset format, scorers and target protocol are described in
[docs/configuration.md](docs/configuration.md).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install   # create .venv and install locked dependencies
make check     # lint, type check and tests (what CI runs)
```

## License

[MIT](./LICENSE)
