import pytest

from costless.metrics import mean, percentile, sample_variance


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
