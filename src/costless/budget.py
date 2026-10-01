"""Budget gate: fail a run that cost too much or could not be priced."""

from dataclasses import dataclass

from costless.config import BudgetSettings, PricingSettings
from costless.models import RunSummary


@dataclass(frozen=True)
class BudgetViolation:
    rule: str
    message: str


def check_budget(
    summary: RunSummary, budget: BudgetSettings, pricing: PricingSettings
) -> list[BudgetViolation]:
    violations: list[BudgetViolation] = []
    if summary.unpriced_models and pricing.strict:
        violations.append(
            BudgetViolation(
                "pricing",
                "no price for model(s) "
                + ", ".join(summary.unpriced_models)
                + "; add them under pricing.models in costless.yaml",
            )
        )
    if summary.skipped_attempts and budget.max_run_usd is not None:
        violations.append(
            BudgetViolation(
                "max_run_usd",
                f"run budget of ${budget.max_run_usd:g} exhausted; "
                f"{summary.skipped_attempts} attempt(s) were not executed",
            )
        )
    elif (
        budget.max_run_usd is not None
        and summary.cost_total_usd is not None
        and summary.cost_total_usd > budget.max_run_usd
    ):
        violations.append(
            BudgetViolation(
                "max_run_usd",
                f"run cost ${summary.cost_total_usd:.4f} exceeds ${budget.max_run_usd:g}",
            )
        )
    if (
        budget.max_case_usd is not None
        and summary.cost_per_case_usd is not None
        and summary.cost_per_case_usd > budget.max_case_usd
    ):
        violations.append(
            BudgetViolation(
                "max_case_usd",
                f"mean cost per case ${summary.cost_per_case_usd:.6f} "
                f"exceeds ${budget.max_case_usd:g}",
            )
        )
    return violations
