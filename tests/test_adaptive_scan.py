import pytest

from mcsirius.scan import (
    AdaptiveScanConfig,
    maximize_adaptive_1d,
)


def test_adaptive_scan_finds_peak_with_small_budget():
    result = maximize_adaptive_1d(
        lambda x: 10.0 - (x - 3.25) ** 2,
        start=2.0,
        lower=0.0,
        upper=5.0,
        config=AdaptiveScanConfig(
            initial_step=0.5,
            min_step=0.05,
            max_evaluations=12,
        ),
    )

    assert result.best_position == pytest.approx(
        3.25
    )

    assert result.best_score == pytest.approx(
        10.0
    )


def test_adaptive_scan_respects_measurement_budget():
    calls = []

    result = maximize_adaptive_1d(
        lambda x: calls.append(x) or (
            100.0 - (x - 5.0) ** 2
        ),
        start=2.0,
        lower=0.0,
        upper=10.0,
        config=AdaptiveScanConfig(
            initial_step=1.0,
            min_step=0.1,
            max_evaluations=7,
        ),
    )

    assert len(calls) <= 7
    assert len(result.samples) <= 7


def test_adaptive_scan_never_leaves_bounds():
    calls = []

    maximize_adaptive_1d(
        lambda x: calls.append(x) or x,
        start=0.1,
        lower=0.0,
        upper=1.0,
        config=AdaptiveScanConfig(
            initial_step=0.5,
            min_step=0.1,
            max_evaluations=7,
        ),
    )

    assert min(calls) >= 0.0
    assert max(calls) <= 1.0


def test_adaptive_config_rejects_tiny_budget():
    with pytest.raises(ValueError):
        AdaptiveScanConfig(
            initial_step=1.0,
            min_step=0.1,
            max_evaluations=2,
        ).validate()
