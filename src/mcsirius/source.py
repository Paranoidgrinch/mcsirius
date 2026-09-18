"""Coupled source-voltage optimization with magnet tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    LocalScanConfig,
    ScanResult,
    maximize_1d,
)


class SourceHardware(Protocol):
    """Hardware operations required for source optimization."""

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
class SourceOptimizationResult:
    start_operating_point: OperatingPoint
    extraction_scan: ScanResult
    sputter_scan: ScanResult
    final_operating_point: OperatingPoint
    final_magnet_seed: MagnetSetpoint
    final_magnet_scan: ScanResult
    final_cup1_score: float


def _cup1_score(
    hardware: SourceHardware,
) -> float:
    return abs(float(hardware.read_cup1_current()))


def _validate_voltage_pair(
    *,
    extraction_kv: float,
    einzel_kv: float,
    limits: MachineLimits,
) -> None:
    if not math.isfinite(extraction_kv):
        raise ValueError("Extraction voltage must be finite.")

    if not math.isfinite(einzel_kv):
        raise ValueError("Einzel voltage must be finite.")

    if not (
        limits.extraction_min_kv
        <= extraction_kv
        <= limits.extraction_max_kv
    ):
        raise ValueError(
            "Extraction voltage is outside machine limits."
        )

    if not (
        limits.einzel_min_kv
        <= einzel_kv
        <= limits.einzel_max_kv
    ):
        raise ValueError(
            "Einzel voltage is outside machine limits."
        )

    if (
        abs(einzel_kv - extraction_kv)
        > limits.max_einzel_delta_kv
    ):
        raise ValueError(
            "Einzel/extraction difference exceeds "
            "the allowed machine limit."
        )


def move_extraction_einzel(
    hardware: SourceHardware,
    *,
    current_extraction_kv: float,
    current_einzel_kv: float,
    target_extraction_kv: float,
    target_einzel_kv: float,
    limits: MachineLimits = DEFAULT_LIMITS,
    max_step_kv: float = 0.5,
) -> tuple[float, float]:
    """
    Move extraction and Einzel together without violating
    their maximum allowed separation at an intermediate write.
    """

    _validate_voltage_pair(
        extraction_kv=current_extraction_kv,
        einzel_kv=current_einzel_kv,
        limits=limits,
    )

    _validate_voltage_pair(
        extraction_kv=target_extraction_kv,
        einzel_kv=target_einzel_kv,
        limits=limits,
    )

    if (
        not math.isfinite(max_step_kv)
        or max_step_kv <= 0.0
    ):
        raise ValueError(
            "Maximum voltage-pair step must be "
            "finite and greater than zero."
        )

    extraction_delta = (
        target_extraction_kv
        - current_extraction_kv
    )
    einzel_delta = (
        target_einzel_kv
        - current_einzel_kv
    )

    largest_delta = max(
        abs(extraction_delta),
        abs(einzel_delta),
    )

    if largest_delta == 0.0:
        return (
            current_extraction_kv,
            current_einzel_kv,
        )

    step_count = max(
        1,
        math.ceil(
            largest_delta / max_step_kv
        ),
    )

    start_extraction = current_extraction_kv
    start_einzel = current_einzel_kv

    current_extraction = current_extraction_kv
    current_einzel = current_einzel_kv

    tolerance = 1e-12

    for index in range(1, step_count + 1):
        fraction = index / step_count

        next_extraction = (
            start_extraction
            + extraction_delta * fraction
        )
        next_einzel = (
            start_einzel
            + einzel_delta * fraction
        )

        extraction_first_safe = (
            abs(
                current_einzel
                - next_extraction
            )
            <= limits.max_einzel_delta_kv
            + tolerance
        )

        einzel_first_safe = (
            abs(
                next_einzel
                - current_extraction
            )
            <= limits.max_einzel_delta_kv
            + tolerance
        )

        if extraction_first_safe:
            hardware.set_extraction_voltage(
                next_extraction
            )
            current_extraction = next_extraction

            hardware.set_einzel_voltage(
                next_einzel
            )
            current_einzel = next_einzel

        elif einzel_first_safe:
            hardware.set_einzel_voltage(
                next_einzel
            )
            current_einzel = next_einzel

            hardware.set_extraction_voltage(
                next_extraction
            )
            current_extraction = next_extraction

        else:
            raise RuntimeError(
                "Could not find a safe voltage-pair "
                "transition step."
            )

    return (
        current_extraction,
        current_einzel,
    )


def _scan_magnet_for_operating_point(
    hardware: SourceHardware,
    *,
    mass_u: float,
    operating_point: OperatingPoint,
    magnet_lower_a: float,
    magnet_upper_a: float,
    magnet_scan: LocalScanConfig,
) -> tuple[MagnetSetpoint, ScanResult]:
    seed = calculate_magnet_setpoint(
        mass_u=mass_u,
        sputter_kv=operating_point.sputter_kv,
        extraction_kv=operating_point.extraction_kv,
    )

    if not (
        magnet_lower_a
        <= seed.current_a
        <= magnet_upper_a
    ):
        raise ValueError(
            "Predicted magnet current lies outside "
            "the allowed magnet bounds."
        )

    def measure(
        current_a: float,
    ) -> float:
        hardware.set_magnet_current(current_a)
        return _cup1_score(hardware)

    result = maximize_1d(
        measure,
        start=seed.current_a,
        lower=magnet_lower_a,
        upper=magnet_upper_a,
        config=magnet_scan,
    )

    hardware.set_magnet_current(
        result.best_position
    )

    return seed, result


def optimize_source_voltages(
    hardware: SourceHardware,
    *,
    mass_u: float,
    operating_point: OperatingPoint,
    magnet_lower_a: float,
    magnet_upper_a: float,
    magnet_scan: LocalScanConfig,
    extraction_scan: LocalScanConfig,
    sputter_scan: LocalScanConfig,
    limits: MachineLimits = DEFAULT_LIMITS,
    voltage_pair_step_kv: float = 0.5,
) -> SourceOptimizationResult:
    """
    Optimize extraction and sputter one dimension at a time.

    Every source-energy change is followed by a local magnet
    search around the newly calculated mass/energy prediction.

    During extraction tuning, the current Einzel/extraction
    offset is preserved where possible.
    """

    limits.validate(operating_point)

    current_extraction = (
        operating_point.extraction_kv
    )
    current_einzel = (
        operating_point.einzel_kv
    )
    current_sputter = (
        operating_point.sputter_kv
    )

    einzel_offset = (
        operating_point.einzel_kv
        - operating_point.extraction_kv
    )

    latest_seed: MagnetSetpoint | None = None
    latest_magnet_scan: ScanResult | None = None

    def measure_extraction(
        extraction_kv: float,
    ) -> float:
        nonlocal current_extraction
        nonlocal current_einzel
        nonlocal latest_seed
        nonlocal latest_magnet_scan

        einzel_lower, einzel_upper = (
            limits.einzel_window(
                extraction_kv
            )
        )

        target_einzel = min(
            einzel_upper,
            max(
                einzel_lower,
                extraction_kv
                + einzel_offset,
            ),
        )

        (
            current_extraction,
            current_einzel,
        ) = move_extraction_einzel(
            hardware,
            current_extraction_kv=(
                current_extraction
            ),
            current_einzel_kv=current_einzel,
            target_extraction_kv=(
                extraction_kv
            ),
            target_einzel_kv=target_einzel,
            limits=limits,
            max_step_kv=voltage_pair_step_kv,
        )

        point = OperatingPoint(
            sputter_kv=current_sputter,
            extraction_kv=current_extraction,
            einzel_kv=current_einzel,
        )

        (
            latest_seed,
            latest_magnet_scan,
        ) = _scan_magnet_for_operating_point(
            hardware,
            mass_u=mass_u,
            operating_point=point,
            magnet_lower_a=magnet_lower_a,
            magnet_upper_a=magnet_upper_a,
            magnet_scan=magnet_scan,
        )

        return latest_magnet_scan.best_score

    extraction_result = maximize_1d(
        measure_extraction,
        start=operating_point.extraction_kv,
        lower=limits.extraction_min_kv,
        upper=limits.extraction_max_kv,
        config=extraction_scan,
    )

    # Restore and re-evaluate the best extraction point.
    measure_extraction(
        extraction_result.best_position
    )

    extraction_best = current_extraction
    einzel_after_extraction = current_einzel

    def measure_sputter(
        sputter_kv: float,
    ) -> float:
        nonlocal current_sputter
        nonlocal latest_seed
        nonlocal latest_magnet_scan

        if not (
            limits.sputter_min_kv
            <= sputter_kv
            <= limits.sputter_max_kv
        ):
            raise ValueError(
                "Sputter voltage is outside "
                "machine limits."
            )

        hardware.set_sputter_voltage(
            sputter_kv
        )
        current_sputter = sputter_kv

        point = OperatingPoint(
            sputter_kv=current_sputter,
            extraction_kv=extraction_best,
            einzel_kv=einzel_after_extraction,
        )

        (
            latest_seed,
            latest_magnet_scan,
        ) = _scan_magnet_for_operating_point(
            hardware,
            mass_u=mass_u,
            operating_point=point,
            magnet_lower_a=magnet_lower_a,
            magnet_upper_a=magnet_upper_a,
            magnet_scan=magnet_scan,
        )

        return latest_magnet_scan.best_score

    sputter_result = maximize_1d(
        measure_sputter,
        start=operating_point.sputter_kv,
        lower=limits.sputter_min_kv,
        upper=limits.sputter_max_kv,
        config=sputter_scan,
    )

    # Restore and re-evaluate the best sputter point.
    measure_sputter(
        sputter_result.best_position
    )

    if (
        latest_seed is None
        or latest_magnet_scan is None
    ):
        raise RuntimeError(
            "Source optimization completed without "
            "a magnet result."
        )

    final_point = OperatingPoint(
        sputter_kv=current_sputter,
        extraction_kv=extraction_best,
        einzel_kv=einzel_after_extraction,
    )

    limits.validate(final_point)

    return SourceOptimizationResult(
        start_operating_point=operating_point,
        extraction_scan=extraction_result,
        sputter_scan=sputter_result,
        final_operating_point=final_point,
        final_magnet_seed=latest_seed,
        final_magnet_scan=latest_magnet_scan,
        final_cup1_score=_cup1_score(hardware),
    )