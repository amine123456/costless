# costless

**A CI quality gate for LLM applications.** costless runs your eval suite on every merge request, compares quality, cost and latency against the main-branch baseline with statistics that separate real regressions from model noise, and blocks the merge when a threshold is breached.

> Status: early development. The sections below are filled in as each component lands; nothing here describes a feature that does not exist yet.

## What works today

- Versioned eval datasets in YAML or JSONL, with content hashes and tags.
- Deterministic scorers: exact match, regex and JSON Schema validation, with weights.
- Two kinds of system under test: an in-process Python callable, or any executable
  that speaks JSON over stdin/stdout.
- A runner that executes every case N times with bounded concurrency, timeouts and
  per-attempt error capture. It reports the mean and variance per case and flags
  flaky cases.
- A replay provider that records model responses once and replays them
  deterministically, for offline tests.
- `run.json`, a versioned results document that keeps every attempt.

```bash
costless validate -c costless.yaml    # check config, datasets and scorers
costless run -c costless.yaml -n 5    # run every case 5 times, write .costless/run.json
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
