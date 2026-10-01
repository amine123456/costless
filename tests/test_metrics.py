import pytest

from costless.metrics import (
    cohens_kappa,
    mean,
    percentile,
    sample_variance,
    weighted_kappa,
    wilson_interval,
)


def test_mean() -> None:
    assert mean([1, 2, 3, 4]) == 2.5
    assert mean([]) == 0.0


def test_sample_variance_is_unbiased() -> None:
    assert sample_variance([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(32 / 7)
    assert sample_variance([3.0]) == 0.0


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        # Reference values from numpy.percentile (linear interpolation).
        (list(range(1, 11)), 95, 9.55),
        (list(range(1, 11)), 50, 5.5),
        ([10, 20, 30, 40], 95, 38.5),
        ([7], 95, 7),
        ([3, 1, 2], 0, 1),
        ([3, 1, 2], 100, 3),
    ],
)
def test_percentile_matches_numpy(values: list[float], q: float, expected: float) -> None:
    assert percentile(values, q) == pytest.approx(expected)


def test_percentile_edge_cases() -> None:
    assert percentile([], 95) == 0.0
    with pytest.raises(ValueError, match="percentile must be in"):
        percentile([1], 101)


def test_wilson_interval_reference_value() -> None:
    # 8/10: reference interval (0.490, 0.943).
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_cohens_kappa_hand_computed() -> None:
    # po = 0.75, pe = 0.5*0.25 + 0.5*0.75 = 0.5  =>  kappa = 0.5
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohens_kappa([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    assert cohens_kappa([1, 1], [1, 1]) is None  # no variety: undefined, not 1.0


def _quadratic_kappa_reference(a: list[int], b: list[int]) -> float:
    # Independent closed form for quadratic weights:
    # 1 - mean((a_i - b_i)^2) / mean over all i, j of (a_i - b_j)^2
    observed = sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a)
    expected = sum((x - y) ** 2 for x in a for y in b) / len(a) ** 2
    return 1 - observed / expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([1, 2, 3], [1, 3, 3]),
        ([5, 4, 4, 2, 1, 3, 5, 2], [4, 4, 5, 2, 2, 3, 5, 1]),
        ([1, 5, 1, 5], [5, 1, 5, 1]),
    ],
)
def test_weighted_kappa_matches_closed_form(a: list[int], b: list[int]) -> None:
    got = weighted_kappa(a, b, categories=range(1, 6))
    assert got == pytest.approx(_quadratic_kappa_reference(a, b))


def test_weighted_kappa_values() -> None:
    assert weighted_kappa([1, 2, 3], [1, 3, 3], categories=range(1, 4)) == pytest.approx(0.8)
    linear = weighted_kappa([1, 2, 3], [1, 3, 3], categories=range(1, 4), weights="linear")
    assert linear is not None
    assert 0 < linear < 1
    with pytest.raises(ValueError, match="same length"):
        weighted_kappa([1], [1, 2], categories=[1, 2])
    with pytest.raises(ValueError, match="unknown weights"):
        weighted_kappa([1, 2], [2, 1], categories=[1, 2], weights="cubic")
