"""Deterministic optimization chain for the source-to-Cup-1 section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .machine import (
    DEFAULT_LIMITS,
    MachineLimits,
    OperatingPoint,
)
from .magnet import (
    MagnetSetpoint,
    calculate_magnet_setpoint,
)
from .scan import (
    ScanConfig,
    ScanResult,
    maximize_parameter,
)


class Cup1Hardware(Protocol):
    """
    Minimal hardware interface needed for the first
    mcsirius optimization stage.
    """

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        ...

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        ...

    def read_cup1_current(self) -> float:
        ...


@dataclass(frozen=True)
class FrontendOptimizationResult:
    magnet_seed: MagnetSetpoint
    magnet_scan: ScanResult
    einzel_scan: ScanResult
    final_cup1_score: float


def _cup1_score(
    hardware: Cup1Hardware,
) -> float:
    """
    Negative-ion currents can be reported with negative sign.

    For optimization only the current magnitude is relevant.
    """

    return abs(
        float(hardware.read_cup1_current())
    )


def optimize_magnet_and_einzel(
    hardware: Cup1Hardware,
    *,
    mass_u: float,
    operating_point: OperatingPoint,
    magnet_lower_a: float,
    magnet_upper_a: float,
    magnet_scan: ScanConfig,
    einzel_scan: ScanConfig,
    limits: MachineLimits = DEFAULT_LIMITS,
) -> FrontendOptimizationResult:
    """
    Find the mass peak around the calculated magnet seed,
    then focus the beam on Cup 1 with the Einzel lens.

    Machine-specific scan widths and step sizes are supplied
    by the caller instead of being guessed inside the core.
    """

    limits.validate(operating_point)

    magnet_seed = calculate_magnet_setpoint(
        mass_u=mass_u,
        sputter_kv=operating_point.sputter_kv,
        extraction_kv=operating_point.extraction_kv,
    )

    if magnet_lower_a >= magnet_upper_a:
        raise ValueError(
            "Magnet lower bound must be smaller "
            "than upper bound."
        )

    if not (
        magnet_lower_a
        <= magnet_seed.current_a
        <= magnet_upper_a
    ):
        raise ValueError(
            "Predicted magnet current lies outside "
            "the allowed bounds."
        )

    def measure_magnet(
        current_a: float,
    ) -> float:
        hardware.set_magnet_current(current_a)
        return _cup1_score(hardware)

    magnet_result = maximize_parameter(
        measure_magnet,
        start=magnet_seed.current_a,
        lower=magnet_lower_a,
        upper=magnet_upper_a,
        config=magnet_scan,
    )

    # A scan ends at its final sampled point, not necessarily
    # at its optimum. Explicitly restore the best setting.
    hardware.set_magnet_current(
        magnet_result.best_position
    )

    einzel_lower, einzel_upper = (
        limits.einzel_window(
            operating_point.extraction_kv
        )
    )

    def measure_einzel(
        voltage_kv: float,
    ) -> float:
        hardware.set_einzel_voltage(voltage_kv)
        return _cup1_score(hardware)

    einzel_result = maximize_parameter(
        measure_einzel,
        start=operating_point.einzel_kv,
        lower=einzel_lower,
        upper=einzel_upper,
        config=einzel_scan,
    )

    # Finish at the actual best machine settings.
    hardware.set_einzel_voltage(
        einzel_result.best_position
    )
    hardware.set_magnet_current(
        magnet_result.best_position
    )

    return FrontendOptimizationResult(
        magnet_seed=magnet_seed,
        magnet_scan=magnet_result,
        einzel_scan=einzel_result,
        final_cup1_score=_cup1_score(hardware),
    )


def optimize_einzel_at_fixed_magnet(
    hardware: Cup1Hardware,
    *,
    operating_point: OperatingPoint,
    magnet_seed: MagnetSetpoint,
    magnet_result: ScanResult,
    einzel_scan: ScanConfig,
    limits: MachineLimits = DEFAULT_LIMITS,
) -> FrontendOptimizationResult:
    """
    Re-focus the Einzel lens while keeping the already refined
    magnet setting fixed.
    """

    limits.validate(operating_point)

    hardware.set_magnet_current(
        magnet_result.best_position
    )

    einzel_lower, einzel_upper = (
        limits.einzel_window(
            operating_point.extraction_kv
        )
    )

    def measure_einzel(
        voltage_kv: float,
    ) -> float:
        hardware.set_einzel_voltage(
            voltage_kv
        )
        return _cup1_score(hardware)

    einzel_result = maximize_parameter(
        measure_einzel,
        start=operating_point.einzel_kv,
        lower=einzel_lower,
        upper=einzel_upper,
        config=einzel_scan,
    )

    hardware.set_einzel_voltage(
        einzel_result.best_position
    )
    hardware.set_magnet_current(
        magnet_result.best_position
    )

    return FrontendOptimizationResult(
        magnet_seed=magnet_seed,
        magnet_scan=magnet_result,
        einzel_scan=einzel_result,
        final_cup1_score=_cup1_score(hardware),
    )
