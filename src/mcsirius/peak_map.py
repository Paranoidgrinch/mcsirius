"""Diagnostic live mass-peak mapping around a predicted mass."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .flavia import (
    FlaviaHardware,
)
from .live_backend import (
    StandaloneLiveBackend,
)
from .machine import (
    DEFAULT_OPERATING_POINT,
    OperatingPoint,
)
from .magnet import (
    calculate_magnet_setpoint,
)
from .measurement import (
    MeasurementConfig,
    RobustCup1Hardware,
)


class PeakMapHardware(Protocol):
    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        ...

    def read_cup1_current(
        self,
    ) -> float:
        ...


@dataclass(frozen=True)
class PeakMapConfig:
    half_width_a: float = 0.45
    step_a: float = 0.03
    neighbor_masses: int = 2
    source_tolerance_fraction: float = 0.10

    def validate(self) -> None:
        if (
            not math.isfinite(self.half_width_a)
            or self.half_width_a <= 0.0
        ):
            raise ValueError(
                "Peak-map half width must be finite "
                "and greater than zero."
            )

        if (
            not math.isfinite(self.step_a)
            or self.step_a <= 0.0
        ):
            raise ValueError(
                "Peak-map step must be finite "
                "and greater than zero."
            )

        if self.step_a > self.half_width_a:
            raise ValueError(
                "Peak-map step cannot exceed half width."
            )

        if (
            type(self.neighbor_masses) is not int
            or self.neighbor_masses < 0
        ):
            raise ValueError(
                "neighbor_masses must be a "
                "non-negative integer."
            )

        if (
            not math.isfinite(
                self.source_tolerance_fraction
            )
            or self.source_tolerance_fraction < 0.0
        ):
            raise ValueError(
                "Source tolerance must be finite "
                "and non-negative."
            )


@dataclass(frozen=True)
class PredictedMassPosition:
    mass_u: float
    magnet_current_a: float


@dataclass(frozen=True)
class PeakMapSample:
    magnet_current_a: float
    cup_current_a: float

    @property
    def magnitude_a(self) -> float:
        return abs(
            self.cup_current_a
        )


@dataclass(frozen=True)
class ObservedPeak:
    magnet_current_a: float
    cup_current_a: float

    @property
    def magnitude_a(self) -> float:
        return abs(
            self.cup_current_a
        )


@dataclass(frozen=True)
class PeakMapResult:
    target_mass_u: float
    predicted_target_a: float
    predicted_masses: tuple[
        PredictedMassPosition,
        ...
    ]
    samples: tuple[
        PeakMapSample,
        ...
    ]
    observed_peaks: tuple[
        ObservedPeak,
        ...
    ]


@dataclass(frozen=True)
class LivePeakMapReport:
    source_readback: OperatingPoint
    starting_magnet_a: float
    restored_magnet_a: float
    peak_map: PeakMapResult


def _assert_source_near_start(
    point: OperatingPoint,
    *,
    tolerance_fraction: float,
) -> None:
    target = DEFAULT_OPERATING_POINT

    for label, actual, expected in (
        (
            "Sputter",
            point.sputter_kv,
            target.sputter_kv,
        ),
        (
            "Extraction",
            point.extraction_kv,
            target.extraction_kv,
        ),
        (
            "Einzel",
            point.einzel_kv,
            target.einzel_kv,
        ),
    ):
        allowed = (
            abs(expected)
            * tolerance_fraction
        )

        difference = abs(
            actual - expected
        )

        if difference > allowed:
            percent = (
                100.0
                * difference
                / abs(expected)
            )

            raise RuntimeError(
                f"{label} is not near the prepared "
                f"source point: {actual:.3f} kV "
                f"vs {expected:.3f} kV "
                f"({percent:.2f}% difference)."
            )


def predicted_mass_positions(
    mass_u: float,
    *,
    neighbor_masses: int,
) -> tuple[
    PredictedMassPosition,
    ...
]:
    """
    Predict neighboring mass positions using the deterministic
    4/14 kV source setpoints, not the calibrated HV readbacks.
    """

    mass_u = float(mass_u)

    if (
        not math.isfinite(mass_u)
        or mass_u <= 0.0
    ):
        raise ValueError(
            "Mass must be finite and greater than zero."
        )

    positions = []

    for offset in range(
        -neighbor_masses,
        neighbor_masses + 1,
    ):
        mass = mass_u + offset

        if mass <= 0.0:
            continue

        setpoint = calculate_magnet_setpoint(
            mass_u=mass,
            sputter_kv=(
                DEFAULT_OPERATING_POINT.sputter_kv
            ),
            extraction_kv=(
                DEFAULT_OPERATING_POINT.extraction_kv
            ),
        )

        positions.append(
            PredictedMassPosition(
                mass_u=mass,
                magnet_current_a=(
                    setpoint.current_a
                ),
            )
        )

    return tuple(positions)


def _scan_positions(
    center_a: float,
    *,
    half_width_a: float,
    step_a: float,
) -> tuple[float, ...]:
    steps_each_side = int(
        round(
            half_width_a / step_a
        )
    )

    if steps_each_side < 1:
        raise ValueError(
            "Peak map needs at least one step "
            "on each side."
        )

    actual_half_width = (
        steps_each_side * step_a
    )

    if (
        abs(
            actual_half_width
            - half_width_a
        )
        > 1e-9
    ):
        raise ValueError(
            "Peak-map half width must be an "
            "integer multiple of the step."
        )

    return tuple(
        center_a + index * step_a
        for index in range(
            -steps_each_side,
            steps_each_side + 1,
        )
    )


def _find_local_peaks(
    samples: tuple[
        PeakMapSample,
        ...
    ],
) -> tuple[
    ObservedPeak,
    ...
]:
    """
    Find simple measured local maxima.

    No mass label is assigned here. That is deliberate:
    measured maxima and predicted masses remain separate until
    an explicit identification rule has been validated.
    """

    peaks = []

    for index in range(
        1,
        len(samples) - 1,
    ):
        left = samples[index - 1]
        center = samples[index]
        right = samples[index + 1]

        center_score = (
            center.magnitude_a
        )

        if (
            center_score
            >= left.magnitude_a
            and center_score
            >= right.magnitude_a
            and (
                center_score
                > left.magnitude_a
                or center_score
                > right.magnitude_a
            )
        ):
            peaks.append(
                ObservedPeak(
                    magnet_current_a=(
                        center.magnet_current_a
                    ),
                    cup_current_a=(
                        center.cup_current_a
                    ),
                )
            )

    return tuple(peaks)


def acquire_peak_map(
    hardware: PeakMapHardware,
    *,
    mass_u: float,
    config: PeakMapConfig = PeakMapConfig(),
) -> PeakMapResult:
    """
    Scan monotonically from low to high current.

    The hardware magnet setter is responsible for waiting for
    the real magnet readback to settle before this function
    reads Cup 1.
    """

    config.validate()

    target = calculate_magnet_setpoint(
        mass_u=float(mass_u),
        sputter_kv=(
            DEFAULT_OPERATING_POINT.sputter_kv
        ),
        extraction_kv=(
            DEFAULT_OPERATING_POINT.extraction_kv
        ),
    )

    positions = _scan_positions(
        target.current_a,
        half_width_a=config.half_width_a,
        step_a=config.step_a,
    )

    if positions[0] < 0.0:
        raise ValueError(
            "Peak-map current would fall below 0 A."
        )

    if positions[-1] > 120.0:
        raise ValueError(
            "Peak-map current would exceed 120 A."
        )

    samples = []

    for current_a in positions:
        hardware.set_magnet_current(
            current_a
        )

        cup_current = float(
            hardware.read_cup1_current()
        )

        if not math.isfinite(
            cup_current
        ):
            raise RuntimeError(
                "Cup-1 current is non-finite."
            )

        samples.append(
            PeakMapSample(
                magnet_current_a=(
                    current_a
                ),
                cup_current_a=(
                    cup_current
                ),
            )
        )

    sample_tuple = tuple(samples)

    return PeakMapResult(
        target_mass_u=float(mass_u),
        predicted_target_a=(
            target.current_a
        ),
        predicted_masses=(
            predicted_mass_positions(
                mass_u,
                neighbor_masses=(
                    config.neighbor_masses
                ),
            )
        ),
        samples=sample_tuple,
        observed_peaks=(
            _find_local_peaks(
                sample_tuple
            )
        ),
    )


def run_live_peak_map(
    mass_u: float,
    *,
    config: PeakMapConfig = PeakMapConfig(),
) -> LivePeakMapReport:
    """
    Measure the real magnet spectrum around the predicted mass.

    Source voltages are not changed. The starting magnet current
    is restored after the scan, including when an exception
    occurs.
    """

    config.validate()

    backend = StandaloneLiveBackend(
        configure_keithley=True
    )

    backend.start()

    try:
        backend.wait_until_ready()

        hardware = FlaviaHardware(
            backend
        )

        hardware.prepare_for_optimization()

        source = (
            hardware.read_operating_point()
        )

        _assert_source_near_start(
            source,
            tolerance_fraction=(
                config.source_tolerance_fraction
            ),
        )

        starting_magnet = (
            hardware.read_magnet_current()
        )

        robust = RobustCup1Hardware(
            hardware,
            config=MeasurementConfig(
                sample_count=5,
                settle_seconds=0.15,
                inter_sample_seconds=0.05,
            ),
        )

        try:
            result = acquire_peak_map(
                robust,
                mass_u=mass_u,
                config=config,
            )

        finally:
            hardware.set_magnet_current(
                starting_magnet
            )

        restored = (
            hardware.read_magnet_current()
        )

        return LivePeakMapReport(
            source_readback=source,
            starting_magnet_a=(
                starting_magnet
            ),
            restored_magnet_a=restored,
            peak_map=result,
        )

    finally:
        backend.stop()
