import math

import pytest

from mcsirius.scan import LocalScanConfig, maximize_1d


def test_coarse_to_fine_scan_finds_peak():
    result = maximize_1d(
        lambda x: 10.0 - (x - 3.2) ** 2,
        start=2.0,
        lower=0.0,
        upper=5.0,
        config=LocalScanConfig(
            coarse_radius=2.0,
            coarse_step=0.5,
            fine_radius=0.5,
            fine_step=0.1,
        ),
    )

    assert result.coarse_best_position == pytest.approx(3.0)
    assert result.best_position == pytest.approx(3.2)
    assert result.best_score == pytest.approx(10.0)

    assert {
        sample.phase
        for sample in result.samples
    } == {"coarse", "fine"}


def test_scan_is_clipped_to_bounds():
    measured = []

    result = maximize_1d(
        lambda x: measured.append(x) or x,
        start=0.2,
        lower=0.0,
        upper=1.0,
        config=LocalScanConfig(
            coarse_radius=2.0,
            coarse_step=0.5,
            fine_radius=0.4,
            fine_step=0.1,
        ),
    )

    assert min(measured) >= 0.0
    assert max(measured) <= 1.0
    assert result.best_position == pytest.approx(1.0)


def test_scan_prefers_start_on_flat_coarse_response():
    result = maximize_1d(
        lambda _: 1.0,
        start=5.0,
        lower=0.0,
        upper=10.0,
        config=LocalScanConfig(
            coarse_radius=2.0,
            coarse_step=1.0,
            fine_radius=0.5,
            fine_step=0.1,
        ),
    )

    assert result.coarse_best_position == pytest.approx(5.0)
    assert result.best_position == pytest.approx(5.0)


def test_non_finite_measurement_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        maximize_1d(
            lambda _: math.nan,
            start=1.0,
            lower=0.0,
            upper=2.0,
            config=LocalScanConfig(
                1.0,
                0.5,
                0.2,
                0.1,
            ),
        )


@pytest.mark.parametrize(
    "config",
    [
        LocalScanConfig(0.0, 0.5, 0.2, 0.1),
        LocalScanConfig(1.0, 0.0, 0.2, 0.1),
        LocalScanConfig(1.0, 0.5, 0.0, 0.1),
        LocalScanConfig(1.0, 0.5, 0.2, 0.0),
        LocalScanConfig(1.0, 0.5, 0.2, 0.5),
    ],
)
def test_invalid_scan_config_is_rejected(config):
    with pytest.raises(ValueError):
        maximize_1d(
            lambda x: x,
            start=1.0,
            lower=0.0,
            upper=2.0,
            config=config,
        )


def test_start_outside_bounds_is_rejected():
    with pytest.raises(ValueError, match="inside"):
        maximize_1d(
            lambda x: x,
            start=3.0,
            lower=0.0,
            upper=2.0,
            config=LocalScanConfig(
                1.0,
                0.5,
                0.2,
                0.1,
            ),
        )