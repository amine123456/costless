import numpy as np
import pytest

from costless.stats import paired_bootstrap, point_statistic


def arrays(*rows: list[float] | list[int]) -> list[np.ndarray]:
    return [np.asarray(r, dtype=np.float64) for r in rows]


def test_reproducible_with_seed() -> None:
    base = arrays([1, 0, 1], [0.5, 0.5, 1], [0, 0, 1])
    cand = arrays([1, 1, 1], [0.5, 0, 1], [0, 1, 1])
    a = paired_bootstrap(base, cand, statistic="mean", samples=2000, seed=7)
    b = paired_bootstrap(base, cand, statistic="mean", samples=2000, seed=7)
    assert np.array_equal(a.candidate, b.candidate)
    assert a.difference_ci(0.95) == b.difference_ci(0.95)


def test_detects_a_clear_shift() -> None:
    base = arrays(*[[0.9, 0.8, 1.0, 0.9, 0.85]] * 20)
    cand = arrays(*[[0.6, 0.5, 0.7, 0.6, 0.55]] * 20)
    low, high = paired_bootstrap(base, cand, statistic="mean", samples=2000, seed=0).difference_ci(
        0.95
    )
    assert high < 0
    assert low == pytest.approx(-0.3, abs=0.05)


def test_no_change_intervals_usually_contain_zero() -> None:
    # Two independent runs of the same noisy system, over many datasets:
    # a 95% interval should contain the true difference (0) about 95% of the time.
    rng = np.random.default_rng(1)
    covered = 0
    trials = 100
    for seed in range(trials):
        p = rng.random(30)
        base = [(rng.random(5) < p[i]).astype(float) for i in range(30)]
        cand = [(rng.random(5) < p[i]).astype(float) for i in range(30)]
        result = paired_bootstrap(base, cand, statistic="mean", samples=1000, seed=seed)
        low, high = result.difference_ci(0.95)
        covered += low <= 0 <= high
    assert covered / trials >= 0.88


def test_paired_identical_runs_differ_by_exactly_zero() -> None:
    rows = arrays([1, 0, 1], [0.5, 0.25, 1])
    low, high = paired_bootstrap(rows, rows, statistic="mean", samples=500, seed=0).difference_ci(
        0.95
    )
    assert low == high == 0


def test_identical_constant_runs_have_zero_width_interval() -> None:
    rows = arrays([1, 1], [0, 0], [1, 1])
    result = paired_bootstrap(rows, rows, statistic="mean", samples=500, seed=0)
    low, high = result.difference_ci(0.95)
    assert low == high == 0


def test_uneven_repeat_counts_are_supported() -> None:
    base = arrays([1.0], [0.0, 1.0, 1.0], [0.5, 0.5])
    result = paired_bootstrap(base, base, statistic="mean", samples=500, seed=0)
    assert np.all(np.isfinite(result.base))


def test_p95_and_ratio() -> None:
    base = arrays(*[[100, 110, 120, 130, 140]] * 10)
    cand = arrays(*[[200, 220, 240, 260, 280]] * 10)
    assert point_statistic(base, "p95") == pytest.approx(
        np.percentile([100, 110, 120, 130, 140] * 10, 95)
    )
    ci = paired_bootstrap(base, cand, statistic="p95", samples=1000, seed=0).ratio_ci(0.95)
    assert ci is not None
    assert ci[0] > 1.5


def test_ratio_undefined_when_baseline_is_zero() -> None:
    zeros = arrays([0, 0], [0, 0])
    assert (
        paired_bootstrap(zeros, zeros, statistic="mean", samples=500, seed=0).ratio_ci(0.95) is None
    )


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="same, non-zero number of cases"):
        paired_bootstrap([], [], statistic="mean", samples=500, seed=0)
    with pytest.raises(ValueError, match="at least one attempt"):
        paired_bootstrap(arrays([1]), [np.asarray([])], statistic="mean", samples=500, seed=0)
