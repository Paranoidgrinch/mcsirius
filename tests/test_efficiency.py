import pytest

from mcsirius.runtime import run_simulation


def test_complete_run_uses_reasonable_measurement_count():
    result = run_simulation(27.0)

    assert result.session.converged is True

    assert (
        result.session.final_cup1_score
        == pytest.approx(
            100.0e-9,
            rel=1e-8,
        )
    )

    # Guard against accidentally restoring the old nested
    # full-magnet-scan behaviour, which used 895 measurements.
    assert result.measurement_count < 300