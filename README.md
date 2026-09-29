# costless

**A CI quality gate for LLM applications.** costless runs your eval suite on every merge request, compares quality, cost and latency against the main-branch baseline with statistics that separate real regressions from model noise, and blocks the merge when a threshold is breached.

> Status: early development. The sections below are filled in as each component lands; nothing here describes a feature that does not exist yet.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install   # create .venv and install locked dependencies
make check     # lint, type check and tests (what CI runs)
```

## License

[MIT](./LICENSE)
