import pytest

from mcsirius.runtime import run_simulation


def test_complete_simulation_reaches_known_optimum():
    result = run_simulation(27.0)

    final = (
        result.session.final_operating_point
    )

    assert final.sputter_kv == pytest.approx(
        6.0
    )

    assert final.extraction_kv == pytest.approx(
        17.0
    )

    assert final.einzel_kv == pytest.approx(
        17.5
    )

    assert (
        result.session.final_cup1_score
        == pytest.approx(
            100.0e-9,
            rel=1e-8,
        )
    )


def test_simulation_exercises_startup_ramp():
    result = run_simulation(27.0)

    assert (
        result.startup.ramp.duration_s
        == pytest.approx(14.0)
    )

    assert (
        result.startup.final_readback.sputter_kv
        == pytest.approx(4.0)
    )

    assert (
        result.startup.final_readback.extraction_kv
        == pytest.approx(14.0)
    )


def test_simulation_uses_robust_measurement_layer():
    result = run_simulation(27.0)

    assert result.measurement_count > 0


def test_simulation_converges():
    result = run_simulation(27.0)

    assert result.session.converged is True

    assert len(result.session.cycles) >= 2


def test_different_mass_uses_same_source_optimizer():
    result = run_simulation(40.0)

    final = (
        result.session.final_operating_point
    )

    assert final.sputter_kv == pytest.approx(
        6.0
    )

    assert final.extraction_kv == pytest.approx(
        17.0
    )

    assert final.einzel_kv == pytest.approx(
        17.5
    )


@pytest.mark.parametrize(
    "mass",
    [
        0.0,
        -1.0,
    ],
)
def test_invalid_simulation_mass_is_rejected(
    mass,
):
    with pytest.raises(ValueError):
        run_simulation(mass)