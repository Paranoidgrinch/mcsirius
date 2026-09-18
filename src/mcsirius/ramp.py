"""Controlled source-voltage ramping."""

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


class RampHardware(Protocol):
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
class RampConfig:
    rate_kv_per_s: float = 1.0
    update_period_s: float = 0.25
    max_duration_s: float = 15.0

    def validate(self) -> None:
        for name, value in (
            (
                "rate_kv_per_s",
                self.rate_kv_per_s,
            ),
            (
                "update_period_s",
                self.update_period_s,
            ),
            (
                "max_duration_s",
                self.max_duration_s,
            ),
        ):
            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and greater than zero."
                )


@dataclass(frozen=True)
class RampResult:
    start_operating_point: OperatingPoint
    target_operating_point: OperatingPoint
    duration_s: float
    step_count: int


def validate_startup_operating_point(
    point: OperatingPoint,
    *,
    limits: MachineLimits = DEFAULT_LIMITS,
) -> None:
    """
    Validate a source state from which mcsirius may ramp.

    Unlike the optimization domain, startup is allowed from
    zero voltage.
    """

    values = (
        point.sputter_kv,
        point.extraction_kv,
        point.einzel_kv,
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Startup voltages must be finite."
        )

    if not (
        0.0
        <= point.sputter_kv
        <= limits.sputter_max_kv
    ):
        raise ValueError(
            "Startup sputter voltage is outside "
            "the safe range."
        )

    if not (
        0.0
        <= point.extraction_kv
        <= limits.extraction_max_kv
    ):
        raise ValueError(
            "Startup extraction voltage is outside "
            "the safe range."
        )

    if not (
        0.0
        <= point.einzel_kv
        <= limits.einzel_max_kv
    ):
        raise ValueError(
            "Startup Einzel voltage is outside "
            "the safe range."
        )

    if (
        abs(
            point.einzel_kv
            - point.extraction_kv
        )
        > limits.max_einzel_delta_kv
    ):
        raise ValueError(
            "Startup Einzel/extraction difference "
            "exceeds the allowed machine limit."
        )


def _write_safe_extraction_einzel_step(
    hardware: RampHardware,
    *,
    current_extraction_kv: float,
    current_einzel_kv: float,
    next_extraction_kv: float,
    next_einzel_kv: float,
    limits: MachineLimits,
) -> tuple[float, float]:
    """
    Apply one pair step without temporarily exceeding
    the Einzel/extraction separation limit.
    """

    tolerance = 1e-12

    extraction_first_safe = (
        abs(
            current_einzel_kv
            - next_extraction_kv
        )
        <= limits.max_einzel_delta_kv
        + tolerance
    )

    einzel_first_safe = (
        abs(
            next_einzel_kv
            - current_extraction_kv
        )
        <= limits.max_einzel_delta_kv
        + tolerance
    )

    if extraction_first_safe:
        if (
            next_extraction_kv
            != current_extraction_kv
        ):
            hardware.set_extraction_voltage(
                next_extraction_kv
            )

        if (
            next_einzel_kv
            != current_einzel_kv
        ):
            hardware.set_einzel_voltage(
                next_einzel_kv
            )

    elif einzel_first_safe:
        if (
            next_einzel_kv
            != current_einzel_kv
        ):
            hardware.set_einzel_voltage(
                next_einzel_kv
            )

        if (
            next_extraction_kv
            != current_extraction_kv
        ):
            hardware.set_extraction_voltage(
                next_extraction_kv
            )

    else:
        raise RuntimeError(
            "No safe Einzel/extraction write order "
            "exists for this ramp step."
        )

    return (
        next_extraction_kv,
        next_einzel_kv,
    )


def ramp_source_voltages(
    hardware: RampHardware,
    *,
    target: OperatingPoint = DEFAULT_OPERATING_POINT,
    config: RampConfig = RampConfig(),
    limits: MachineLimits = DEFAULT_LIMITS,
    sleep: Callable[[float], None] = time.sleep,
) -> RampResult:
    """
    Ramp sputter, extraction and Einzel simultaneously.

    The slowest required voltage change determines the total
    duration. No channel changes faster than rate_kv_per_s.
    """

    config.validate()

    start = hardware.read_operating_point()

    validate_startup_operating_point(
        start,
        limits=limits,
    )

    limits.validate(target)

    deltas = (
        target.sputter_kv
        - start.sputter_kv,
        target.extraction_kv
        - start.extraction_kv,
        target.einzel_kv
        - start.einzel_kv,
    )

    largest_change = max(
        abs(delta)
        for delta in deltas
    )

    if largest_change == 0.0:
        return RampResult(
            start_operating_point=start,
            target_operating_point=target,
            duration_s=0.0,
            step_count=0,
        )

    duration_s = (
        largest_change
        / config.rate_kv_per_s
    )

    if (
        duration_s
        > config.max_duration_s
        + 1e-12
    ):
        raise ValueError(
            "Required source ramp would take "
            f"{duration_s:.3f} s, exceeding "
            f"the {config.max_duration_s:.3f} s limit."
        )

    step_count = max(
        1,
        math.ceil(
            duration_s
            / config.update_period_s
        ),
    )

    step_duration_s = (
        duration_s / step_count
    )

    current_sputter = start.sputter_kv
    current_extraction = start.extraction_kv
    current_einzel = start.einzel_kv

    for index in range(
        1,
        step_count + 1,
    ):
        sleep(step_duration_s)

        fraction = index / step_count

        next_sputter = (
            start.sputter_kv
            + deltas[0] * fraction
        )

        next_extraction = (
            start.extraction_kv
            + deltas[1] * fraction
        )

        next_einzel = (
            start.einzel_kv
            + deltas[2] * fraction
        )

        intermediate = OperatingPoint(
            sputter_kv=next_sputter,
            extraction_kv=next_extraction,
            einzel_kv=next_einzel,
        )

        validate_startup_operating_point(
            intermediate,
            limits=limits,
        )

        if next_sputter != current_sputter:
            hardware.set_sputter_voltage(
                next_sputter
            )
            current_sputter = next_sputter

        (
            current_extraction,
            current_einzel,
        ) = _write_safe_extraction_einzel_step(
            hardware,
            current_extraction_kv=(
                current_extraction
            ),
            current_einzel_kv=current_einzel,
            next_extraction_kv=(
                next_extraction
            ),
            next_einzel_kv=next_einzel,
            limits=limits,
        )

    return RampResult(
        start_operating_point=start,
        target_operating_point=target,
        duration_s=duration_s,
        step_count=step_count,
    )