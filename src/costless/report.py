"""Render a comparison as Markdown, for merge request / pull request comments."""

from collections.abc import Sequence

from costless import __version__
from costless.compare import CaseChange, Comparison, MetricComparison, Status

MARKER = "<!-- costless-report -->"
DOCS_URL = "https://github.com/amine123456/costless/blob/main/docs/statistics.md"

_ICON = {
    Status.OK: "✅ ok",
    Status.IMPROVED: "✅ improved",
    Status.INCONCLUSIVE: "⚠️ inconclusive",
    Status.REGRESSED: "❌ regressed",
    Status.BREACHED: "❌ limit breached",
    Status.UNAVAILABLE: "n/a",
}
_LABEL = {
    "quality": "Quality (mean score)",
    "failure_rate": "Failure rate",
    "cost_per_case": "Cost per case",
    "latency_p95_ms": "p95 latency",
}


def render_markdown(comparison: Comparison, *, budget_failures: Sequence[str] = ()) -> str:
    c = comparison
    passed = c.gate_passed and not budget_failures
    verdict = "✅ costless: quality gate passed" if passed else "❌ costless: quality gate failed"
    lines = [MARKER, f"## {verdict}", "", _context(c), ""]

    lines += [
        "| Metric | Baseline | This change | Change | 95% CI | Tolerance | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines += [_metric_row(m) for m in c.metrics]

    if c.reasons or budget_failures:
        lines += ["", "**Blocking:**"]
        lines += [f"- {reason}" for reason in c.reasons]
        lines += [f"- budget: {failure}" for failure in budget_failures]
    if any(m.status == Status.INCONCLUSIVE for m in c.metrics):
        lines += [
            "",
            "> ⚠️ A metric moved past its tolerance but the change is within run-to-run noise. "
            "More repeats (`run.repeats`) or more cases narrow the interval.",
        ]

    if c.regressed_cases:
        lines += ["", f"### What got worse ({len(c.regressed_cases)} cases)", ""]
        lines += _case_table(c.regressed_cases, with_failure=True)
    if c.fixed_cases:
        lines += [
            "",
            f"<details><summary>What got better ({len(c.fixed_cases)} cases)</summary>",
            "",
        ]
        lines += _case_table(c.fixed_cases, with_failure=False)
        lines += ["", "</details>"]

    notes = _notes(c)
    if notes:
        lines += ["", *notes]
    lines += [
        "",
        f"<sub>costless {__version__} · paired hierarchical bootstrap, "
        f"{c.bootstrap_samples:,} resamples · [how gating works]({DOCS_URL})</sub>",
    ]
    return "\n".join(lines) + "\n"


def _context(c: Comparison) -> str:
    candidate = _ref(c.candidate_ref, c.candidate_sha) or "this change"
    if c.baseline_ref is None and c.baseline_sha is None and c.paired_cases == 0:
        return f"No baseline run available for {candidate}: only absolute limits were checked."
    baseline = _ref(c.baseline_ref, c.baseline_sha) or "the baseline"
    base_repeats, cand_repeats = c.repeats
    repeats = (
        f"{cand_repeats} repeats"
        if base_repeats in (None, cand_repeats)
        else f"{base_repeats} vs {cand_repeats} repeats"
    )
    return (
        f"Compared {candidate} with {baseline}: {c.paired_cases} cases, {repeats} each, "
        f"{c.confidence:.0%} confidence."
    )


def _ref(ref: str | None, sha: str | None) -> str | None:
    if ref is None and sha is None:
        return None
    if sha is None:
        return f"`{ref}`"
    short = sha[:8]
    return f"`{ref}` ({short})" if ref else f"`{short}`"


def _metric_row(m: MetricComparison) -> str:
    label = _LABEL.get(m.metric, m.metric)
    return (
        f"| {label} | {_value(m.metric, m.baseline)} | {_value(m.metric, m.candidate)} "
        f"| {_change(m)} | {_ci(m)} | {_tolerance(m)} | {_ICON[m.status]} |"
    )


def _value(metric: str, value: float | None) -> str:
    if value is None:
        return "-"
    if metric == "quality":
        return f"{value:.3f}"
    if metric == "failure_rate":
        return f"{value:.1%}"
    if metric == "cost_per_case":
        return f"${value:.5f}"
    return f"{value / 1000:.2f} s"


def _change(m: MetricComparison) -> str:
    if m.change is None:
        return "-"
    if m.kind == "ratio":
        return f"{(m.change - 1) * 100:+.1f}%"
    if m.metric == "failure_rate":
        return f"{m.change * 100:+.1f} pp"
    return f"{m.change:+.3f}"


def _ci(m: MetricComparison) -> str:
    if m.ci is None:
        return "-"
    low, high = m.ci
    if m.kind == "ratio":
        return f"[{(low - 1) * 100:+.1f}%, {(high - 1) * 100:+.1f}%]"
    if m.metric == "failure_rate":
        return f"[{low * 100:+.1f}, {high * 100:+.1f}] pp"
    return f"[{low:+.3f}, {high:+.3f}]"


def _tolerance(m: MetricComparison) -> str:
    parts = []
    if m.tolerance is not None:
        if m.kind == "ratio":
            parts.append(f"+{m.tolerance * 100:g}%")
        elif m.metric == "failure_rate":
            parts.append(f"+{m.tolerance * 100:g} pp")
        else:
            parts.append(f"-{m.tolerance:g}")
    if m.limit is not None:
        bound = "≥" if m.better == "higher" else "≤"
        parts.append(f"{bound} {_value(m.metric, m.limit)}")
    return ", ".join(parts) or "-"


def _case_table(cases: tuple[CaseChange, ...], *, with_failure: bool) -> list[str]:
    header = "| Case | Quality | Pass rate |" + (" Example failure |" if with_failure else "")
    rule = "|---|---|---|" + ("---|" if with_failure else "")
    rows = []
    for case in cases:
        row = (
            f"| `{case.case_id}` | {case.baseline_quality:.2f} → {case.candidate_quality:.2f} "
            f"| {case.baseline_pass_rate:.0%} → {case.candidate_pass_rate:.0%} |"
        )
        if with_failure:
            row += f" {_cell(case.example_failure)} |"
        rows.append(row)
    return [header, rule, *rows]


def _cell(text: str | None, limit: int = 160) -> str:
    if not text:
        return "-"
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _notes(c: Comparison) -> list[str]:
    notes = []
    if c.datasets_changed:
        notes.append(
            "- The datasets differ between the two runs; only cases present in both are compared."
        )
    if c.only_in_candidate:
        notes.append(f"- New cases, not compared: {_ids(c.only_in_candidate)}")
    if c.only_in_baseline:
        notes.append(f"- Cases missing from this run: {_ids(c.only_in_baseline)}")
    return notes


def _ids(ids: tuple[str, ...], limit: int = 10) -> str:
    shown = ", ".join(f"`{i}`" for i in ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")
