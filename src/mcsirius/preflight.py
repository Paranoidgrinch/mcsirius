"""Preflight checks before mcsirius is allowed to optimize FLAVIA."""

from __future__ import annotations

from dataclasses import dataclass

from .flavia import FlaviaHardware
from .machine import (
    DEFAULT_LIMITS,
    MachineLimits,
    OperatingPoint,
)


@dataclass(frozen=True)
class PreflightReport:
    operating_point: OperatingPoint
    magnet_current_a: float
    cup_current_a: float
    selected_cup: int


def run_live_preflight(
    hardware: FlaviaHardware,
    *,
    limits: MachineLimits = DEFAULT_LIMITS,
) -> PreflightReport:
    """
    Verify that FLAVIA is connected, Cup 1 is selected and the
    source is already inside the mcsirius optimization domain.

    This function performs no source-voltage or magnet changes.
    """

    hardware.prepare_for_optimization()

    operating_point = (
        hardware.read_operating_point()
    )

    limits.validate(
        operating_point
    )

    magnet_current = (
        hardware.read_magnet_current()
    )

    cup_current = (
        hardware.read_cup1_current()
    )

    selected_cup = hardware.selected_cup

    if selected_cup is None:
        raise RuntimeError(
            "Cup selection disappeared during preflight."
        )

    return PreflightReport(
        operating_point=operating_point,
        magnet_current_a=magnet_current,
        cup_current_a=cup_current,
        selected_cup=selected_cup,
    )