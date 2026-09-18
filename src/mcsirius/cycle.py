"""Complete deterministic source-to-Cup-1 optimization cycle."""

from __future__ import annotations

from dataclasses import dataclass

from .machine import (
    DEFAULT_LIMITS,
    MachineLimits,
    OperatingPoint,
)
from .optimizer import (
    FrontendOptimizationResult,
    optimize_einzel_at_fixed_magnet,
    optimize_magnet_and_einzel,
)
from .scan import ScanConfig
from .source import (
    SourceHardware,
    SourceOptimizationResult,
    optimize_source_voltages,
)


@dataclass(frozen=True)
class OptimizationCycleResult:
    start_operating_point: OperatingPoint
    initial_focus: FrontendOptimizationResult
    source_tuning: SourceOptimizationResult
    final_focus: FrontendOptimizationResult
    final_operating_point: OperatingPoint
    final_cup1_score: float


def run_optimization_cycle(
    hardware: SourceHardware,
    *,
    mass_u: float,
    operating_point: OperatingPoint,
    magnet_lower_a: float,
    magnet_upper_a: float,
    magnet_scan: ScanConfig,
    einzel_scan: ScanConfig,
    extraction_scan: ScanConfig,
    sputter_scan: ScanConfig,
    limits: MachineLimits = DEFAULT_LIMITS,
    voltage_pair_step_kv: float = 0.5,
) -> OptimizationCycleResult:
    """
    Run one complete deterministic optimization pass:

        magnet
        -> Einzel
        -> extraction
        -> sputter
        -> magnet
        -> Einzel

    Extraction and sputter changes always trigger a new
    mass/energy-based magnet prediction and local magnet scan.
    """

    limits.validate(operating_point)

    initial_focus = (
        optimize_magnet_and_einzel(
            hardware,
            mass_u=mass_u,
            operating_point=operating_point,
            magnet_lower_a=magnet_lower_a,
            magnet_upper_a=magnet_upper_a,
            magnet_scan=magnet_scan,
            einzel_scan=einzel_scan,
            limits=limits,
        )
    )

    focused_point = OperatingPoint(
        sputter_kv=operating_point.sputter_kv,
        extraction_kv=(
            operating_point.extraction_kv
        ),
        einzel_kv=(
            initial_focus.einzel_scan.best_position
        ),
    )

    source_tuning = optimize_source_voltages(
        hardware,
        mass_u=mass_u,
        operating_point=focused_point,
        magnet_correction_a=(
            initial_focus.magnet_scan.best_position
            - initial_focus.magnet_seed.current_a
        ),
        magnet_lower_a=magnet_lower_a,
        magnet_upper_a=magnet_upper_a,
        magnet_scan=magnet_scan,
        extraction_scan=extraction_scan,
        sputter_scan=sputter_scan,
        limits=limits,
        voltage_pair_step_kv=(
            voltage_pair_step_kv
        ),
    )

    source_point = (
        source_tuning.final_operating_point
    )

    final_focus = optimize_einzel_at_fixed_magnet(
        hardware,
        operating_point=source_point,
        magnet_seed=(
            source_tuning.final_magnet_seed
        ),
        magnet_result=(
            source_tuning.final_magnet_scan
        ),
        einzel_scan=einzel_scan,
        limits=limits,
    )

    final_point = OperatingPoint(
        sputter_kv=source_point.sputter_kv,
        extraction_kv=source_point.extraction_kv,
        einzel_kv=(
            final_focus.einzel_scan.best_position
        ),
    )

    limits.validate(final_point)

    return OptimizationCycleResult(
        start_operating_point=operating_point,
        initial_focus=initial_focus,
        source_tuning=source_tuning,
        final_focus=final_focus,
        final_operating_point=final_point,
        final_cup1_score=(
            final_focus.final_cup1_score
        ),
    )