"""Prepare FLAVIA source voltages for automatic optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol

from .machine import (
    DEFAULT_LIMITS,
    DEFAULT_OPERATING_POINT,
    MachineLimits,
    OperatingPoint,
)
from .ramp import (
    RampConfig,
    RampResult,
    ramp_source_voltages,
)


class StartupHardware(Protocol):
    def prepare_for_optimization(
        self,
    ) -> None:
        ...

    def read_operating_point(
        self,
    ) -> OperatingPoint:
        ...

    def set_sputter_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        ...

    def set_extraction_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        ...

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        ...


@dataclass(frozen=True)
class StartupConfig:
    ramp: RampConfig = RampConfig(
        rate_kv_per_s=1.0,
        update_period_s=0.25,
        max_duration_s=15.0,
    )

    total_time_limit_s: float = 15.0
    final_settle_s: float = 0.25
    readback_tolerance_kv: float = 0.25

    def validate(self) -> None:
        self.ramp.validate()

        if (
            not math.isfinite(
                self.total_time_limit_s
            )
            or self.total_time_limit_s <= 0.0
        ):
            raise ValueError(
                "Total startup time limit must "
                "be finite and greater than zero."
            )

        if (
            not math.isfinite(
                self.final_settle_s
            )
            or self.final_settle_s < 0.0
        ):
            raise ValueError(
                "Final settling time must be "
                "finite and non-negative."
            )

        if (
            not math.isfinite(
                self.readback_tolerance_kv
            )
            or self.readback_tolerance_kv < 0.0
        ):
            raise ValueError(
                "Readback tolerance must be "
                "finite and non-negative."
            )


@dataclass(frozen=True)
class StartupResult:
    ramp: RampResult
    final_readback: OperatingPoint
    target_operating_point: OperatingPoint
    total_planned_time_s: float


def _assert_close(
    *,
    actual: float,
    expected: float,
    tolerance: float,
    label: str,
) -> None:
    difference = abs(
        actual - expected
    )

    if difference > tolerance:
        raise RuntimeError(
            f"{label} readback differs from target "
            f"by {difference:.3f} kV."
        )


def prepare_source_for_optimization(
    hardware: StartupHardware,
    *,
    target: OperatingPoint = DEFAULT_OPERATING_POINT,
    config: StartupConfig = StartupConfig(),
    limits: MachineLimits = DEFAULT_LIMITS,
    sleep: Callable[[float], None] = time.sleep,
) -> StartupResult:
    """
    Select/prepare Cup 1 hardware, ramp the source to its
    deterministic starting voltages and verify the readbacks.
    """

    config.validate()
    limits.validate(target)

    hardware.prepare_for_optimization()

    ramp_result = ramp_source_voltages(
        hardware,
        target=target,
        config=config.ramp,
        limits=limits,
        sleep=sleep,
    )

    total_planned_time = (
        ramp_result.duration_s
        + config.final_settle_s
    )

    if (
        total_planned_time
        > config.total_time_limit_s
        + 1e-12
    ):
        raise RuntimeError(
            "Source preparation would exceed "
            f"{config.total_time_limit_s:.3f} s."
        )

    if config.final_settle_s > 0.0:
        sleep(
            config.final_settle_s
        )

    final = hardware.read_operating_point()

    _assert_close(
        actual=final.sputter_kv,
        expected=target.sputter_kv,
        tolerance=(
            config.readback_tolerance_kv
        ),
        label="Sputter",
    )

    _assert_close(
        actual=final.extraction_kv,
        expected=target.extraction_kv,
        tolerance=(
            config.readback_tolerance_kv
        ),
        label="Extraction",
    )

    _assert_close(
        actual=final.einzel_kv,
        expected=target.einzel_kv,
        tolerance=(
            config.readback_tolerance_kv
        ),
        label="Einzel",
    )

    return StartupResult(
        ramp=ramp_result,
        final_readback=final,
        target_operating_point=target,
        total_planned_time_s=(
            total_planned_time
        ),
    )