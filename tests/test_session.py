from types import SimpleNamespace

import pytest

from mcsirius.machine import OperatingPoint
from mcsirius.scan import LocalScanConfig
from mcsirius.session import (
    ConvergenceConfig,
    run_until_converged,
)


SCAN = LocalScanConfig(
    coarse_radius=1.0,
    coarse_step=0.5,
    fine_radius=0.2,
    fine_step=0.1,
)


class DummyHardware:
    pass


def make_runner(
    scores,
    points=None,
):
    remaining_scores = list(scores)

    if points is None:
        points = [
            OperatingPoint(
                4.0,
                14.0,
                14.0,
            )
            for _ in remaining_scores
        ]

    remaining_points = list(points)

    calls = []

    def runner(
        hardware,
        **kwargs,
    ):
        calls.append(
            kwargs["operating_point"]
        )

        return SimpleNamespace(
            final_operating_point=(
                remaining_points.pop(0)
            ),
            final_cup1_score=(
                remaining_scores.pop(0)
            ),
        )

    runner.calls = calls
    return runner


def run_test_session(
    runner,
    *,
    convergence,
):
    return run_until_converged(
        DummyHardware(),
        mass_u=27.0,
        operating_point=OperatingPoint(
            4.0,
            14.0,
            14.0,
        ),
        magnet_lower_a=0.0,
        magnet_upper_a=50.0,
        magnet_scan=SCAN,
        einzel_scan=SCAN,
        extraction_scan=SCAN,
        sputter_scan=SCAN,
        convergence=convergence,
        cycle_runner=runner,
    )


def test_stops_when_improvement_is_below_threshold():
    runner = make_runner(
        [10.0, 10.05, 20.0]
    )

    result = run_test_session(
        runner,
        convergence=ConvergenceConfig(
            max_cycles=3,
            min_relative_improvement=0.01,
        ),
    )

    assert result.converged is True
    assert len(result.cycles) == 2

    assert (
        result.cycles[1].relative_improvement
        == pytest.approx(0.005)
    )


def test_runs_to_max_cycles_when_still_improving():
    runner = make_runner(
        [10.0, 12.0, 14.0]
    )

    result = run_test_session(
        runner,
        convergence=ConvergenceConfig(
            max_cycles=3,
            min_relative_improvement=0.01,
        ),
    )

    assert result.converged is False
    assert len(result.cycles) == 3
    assert result.final_cup1_score == pytest.approx(
        14.0
    )


def test_zero_initial_score_can_improve():
    runner = make_runner(
        [0.0, 5.0, 5.0]
    )

    result = run_test_session(
        runner,
        convergence=ConvergenceConfig(
            max_cycles=3,
            min_relative_improvement=0.01,
        ),
    )

    assert len(result.cycles) == 3

    assert (
        result.cycles[1].relative_improvement
        == float("inf")
    )

    assert result.converged is True


def test_each_cycle_starts_from_previous_final_point():
    points = [
        OperatingPoint(
            5.0,
            15.0,
            15.0,
        ),
        OperatingPoint(
            6.0,
            16.0,
            16.0,
        ),
    ]

    runner = make_runner(
        [10.0, 10.0],
        points=points,
    )

    run_test_session(
        runner,
        convergence=ConvergenceConfig(
            max_cycles=2,
            min_relative_improvement=0.01,
        ),
    )

    assert runner.calls[0] == OperatingPoint(
        4.0,
        14.0,
        14.0,
    )

    assert runner.calls[1] == OperatingPoint(
        5.0,
        15.0,
        15.0,
    )


@pytest.mark.parametrize(
    "config",
    [
        ConvergenceConfig(max_cycles=0),
        ConvergenceConfig(max_cycles=-1),
        ConvergenceConfig(
            min_relative_improvement=-0.1
        ),
    ],
)
def test_invalid_convergence_config_is_rejected(
    config,
):
    runner = make_runner([10.0])

    with pytest.raises(ValueError):
        run_test_session(
            runner,
            convergence=config,
        )


def test_invalid_cycle_score_is_rejected():
    runner = make_runner([-1.0])

    with pytest.raises(
        ValueError,
        match="invalid Cup-1 score",
    ):
        run_test_session(
            runner,
            convergence=ConvergenceConfig(
                max_cycles=1,
            ),
        )