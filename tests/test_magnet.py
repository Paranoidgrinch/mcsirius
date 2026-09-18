import math

import pytest

from mcsirius.magnet import calculate_magnet_setpoint


def test_mass_27_at_default_source_voltages():
    result = calculate_magnet_setpoint(
        mass_u=27.0,
        sputter_kv=4.0,
        extraction_kv=14.0,
    )

    assert result.total_energy_ev == pytest.approx(18000.0)
    assert result.field_kg == pytest.approx(2.007391800382444)
    assert result.current_a == pytest.approx(18.760500646540915)


def test_mass_40_at_upper_source_voltages():
    result = calculate_magnet_setpoint(
        mass_u=40.0,
        sputter_kv=9.0,
        extraction_kv=25.0,
    )

    assert result.total_energy_ev == pytest.approx(34000.0)
    assert result.field_kg == pytest.approx(3.3580214457785558)
    assert result.current_a == pytest.approx(31.16753394983057)


def test_field_scales_with_square_root_of_mass():
    light = calculate_magnet_setpoint(
        mass_u=25.0,
        sputter_kv=4.0,
        extraction_kv=14.0,
    )
    heavy = calculate_magnet_setpoint(
        mass_u=100.0,
        sputter_kv=4.0,
        extraction_kv=14.0,
    )

    assert heavy.field_kg == pytest.approx(2.0 * light.field_kg)


def test_field_scales_with_square_root_of_energy():
    low = calculate_magnet_setpoint(
        mass_u=27.0,
        sputter_kv=4.0,
        extraction_kv=14.0,
    )
    high = calculate_magnet_setpoint(
        mass_u=27.0,
        sputter_kv=8.0,
        extraction_kv=64.0,
    )

    assert high.total_energy_ev == pytest.approx(
        4.0 * low.total_energy_ev
    )
    assert high.field_kg == pytest.approx(
        2.0 * low.field_kg
    )


@pytest.mark.parametrize(
    "mass_u",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_invalid_mass_is_rejected(mass_u):
    with pytest.raises(ValueError):
        calculate_magnet_setpoint(
            mass_u=mass_u,
            sputter_kv=4.0,
            extraction_kv=14.0,
        )


@pytest.mark.parametrize(
    ("sputter_kv", "extraction_kv"),
    [
        (-0.1, 14.0),
        (4.0, -0.1),
    ],
)
def test_negative_source_voltage_is_rejected(
    sputter_kv,
    extraction_kv,
):
    with pytest.raises(ValueError):
        calculate_magnet_setpoint(
            mass_u=27.0,
            sputter_kv=sputter_kv,
            extraction_kv=extraction_kv,
        )