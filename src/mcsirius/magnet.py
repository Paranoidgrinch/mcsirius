"""Physics model for the FLAVIA analyzing magnet."""

from __future__ import annotations

from dataclasses import dataclass
import math


ATOMIC_MASS_UNIT_KG = 1.66054e-27
ELEMENTARY_CHARGE_C = 1.60218e-19


@dataclass(frozen=True)
class MagnetCalibration:
    """Calibration constants used by the FLAVIA magnet calculator."""

    bend_radius_m: float = 0.5
    field_per_amp_kg: float = 0.10886
    current_fit_offset_kg: float = 0.0348763


@dataclass(frozen=True)
class MagnetSetpoint:
    """Calculated magnet setpoint for one ion-beam operating point."""

    mass_u: float
    total_energy_ev: float
    field_kg: float
    current_a: float


DEFAULT_MAGNET_CALIBRATION = MagnetCalibration()


def calculate_magnet_setpoint(
    mass_u: float,
    sputter_kv: float,
    extraction_kv: float,
    calibration: MagnetCalibration = DEFAULT_MAGNET_CALIBRATION,
) -> MagnetSetpoint:
    """
    Calculate the expected analyzing-magnet current.

    The model mirrors the current FLAVIA magnet calculator:

        E = q * (U_sputter + U_extraction)
        B = sqrt(2 m E) / (q r)
        I = (B_kG + 0.0348763) / 0.10886

    Input voltages are expressed in kV because mcsirius uses kV throughout
    its source-optimization model.
    """

    mass_u = float(mass_u)
    sputter_kv = float(sputter_kv)
    extraction_kv = float(extraction_kv)

    if not math.isfinite(mass_u) or mass_u <= 0.0:
        raise ValueError("Ion mass must be a finite value greater than zero.")

    for name, value in (
        ("sputter", sputter_kv),
        ("extraction", extraction_kv),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name.capitalize()} voltage must be a finite non-negative value."
            )

    if calibration.bend_radius_m <= 0.0:
        raise ValueError("Magnet bend radius must be greater than zero.")

    if calibration.field_per_amp_kg <= 0.0:
        raise ValueError("Magnet field-per-amp calibration must be greater than zero.")

    total_energy_ev = (sputter_kv + extraction_kv) * 1000.0
    mass_kg = mass_u * ATOMIC_MASS_UNIT_KG

    if total_energy_ev == 0.0:
        field_tesla = 0.0
    else:
        field_tesla = math.sqrt(
            2.0
            * total_energy_ev
            * ELEMENTARY_CHARGE_C
            * mass_kg
        ) / (
            ELEMENTARY_CHARGE_C
            * calibration.bend_radius_m
        )

    field_kg = field_tesla * 10.0

    current_a = (
        field_kg + calibration.current_fit_offset_kg
    ) / calibration.field_per_amp_kg

    return MagnetSetpoint(
        mass_u=mass_u,
        total_energy_ev=total_energy_ev,
        field_kg=field_kg,
        current_a=current_a,
    )