"""Robust Cup-1 current acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Callable, Protocol


class RawCup1Hardware(Protocol):
    """Minimal source-to-Cup-1 hardware interface."""

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

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        ...

    def read_cup1_current(self) -> float:
        ...


@dataclass(frozen=True)
class MeasurementConfig:
    """
    Configuration for one robust Cup-1 measurement.

    Timing values are deliberately configurable because the
    final useful values depend on the real Keithley/FLAVIA
    update behaviour.
    """

    sample_count: int = 5
    settle_seconds: float = 0.20
    inter_sample_seconds: float = 0.05

    def validate(self) -> None:
        if (
            type(self.sample_count) is not int
            or self.sample_count < 1
        ):
            raise ValueError(
                "Sample count must be a positive integer."
            )

        if (
            not math.isfinite(self.settle_seconds)
            or self.settle_seconds < 0.0
        ):
            raise ValueError(
                "Settling time must be finite and non-negative."
            )

        if (
            not math.isfinite(
                self.inter_sample_seconds
            )
            or self.inter_sample_seconds < 0.0
        ):
            raise ValueError(
                "Inter-sample time must be finite "
                "and non-negative."
            )


@dataclass(frozen=True)
class Cup1Measurement:
    samples: tuple[float, ...]
    median_current: float
    mad: float

    @property
    def magnitude(self) -> float:
        return abs(self.median_current)


def acquire_cup1_measurement(
    read_current: Callable[[], float],
    *,
    config: MeasurementConfig,
    sleep: Callable[[float], None] = time.sleep,
    settle: bool = True,
) -> Cup1Measurement:
    """
    Acquire several Cup-1 samples and reduce them robustly.

    The central value is the median. Noise is represented by
    the median absolute deviation (MAD).
    """

    config.validate()

    if settle and config.settle_seconds > 0.0:
        sleep(config.settle_seconds)

    samples: list[float] = []

    for index in range(config.sample_count):
        value = float(read_current())

        if not math.isfinite(value):
            raise ValueError(
                "Cup-1 measurement returned "
                "a non-finite value."
            )

        samples.append(value)

        if (
            index + 1 < config.sample_count
            and config.inter_sample_seconds > 0.0
        ):
            sleep(config.inter_sample_seconds)

    center = float(median(samples))

    deviations = tuple(
        abs(value - center)
        for value in samples
    )

    mad = float(median(deviations))

    return Cup1Measurement(
        samples=tuple(samples),
        median_current=center,
        mad=mad,
    )


class RobustCup1Hardware:
    """
    Hardware proxy that turns each Cup-1 read into a robust
    multi-sample measurement.

    Any hardware write marks the beam state as changed. The
    next measurement therefore waits for the configured
    settling time. Repeated reads without a preceding write
    do not unnecessarily wait for settling again.
    """

    def __init__(
        self,
        hardware: RawCup1Hardware,
        *,
        config: MeasurementConfig = MeasurementConfig(),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()

        self.hardware = hardware
        self.config = config
        self.sleep = sleep

        self.last_measurement: Cup1Measurement | None = None
        self.measurement_history: list[
            Cup1Measurement
        ] = []

        self._dirty = True

    def _mark_changed(self) -> None:
        self._dirty = True

    def set_sputter_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.hardware.set_sputter_voltage(
            voltage_kv
        )
        self._mark_changed()

    def set_extraction_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.hardware.set_extraction_voltage(
            voltage_kv
        )
        self._mark_changed()

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.hardware.set_einzel_voltage(
            voltage_kv
        )
        self._mark_changed()

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        self.hardware.set_magnet_current(
            current_a
        )
        self._mark_changed()

    def read_cup1_current(self) -> float:
        result = acquire_cup1_measurement(
            self.hardware.read_cup1_current,
            config=self.config,
            sleep=self.sleep,
            settle=self._dirty,
        )

        self._dirty = False

        self.last_measurement = result
        self.measurement_history.append(result)

        return result.median_current