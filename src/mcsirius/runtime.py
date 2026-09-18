"""High-level mcsirius execution profiles."""

from __future__ import annotations

from dataclasses import dataclass

from .machine import OperatingPoint
from .measurement import (
    MeasurementConfig,
    RobustCup1Hardware,
)
from .scan import (
    AdaptiveScanConfig,
    ScanConfig,
)
from .session import (
    ConvergenceConfig,
    OptimizationSessionResult,
    run_single_pass,
)
from .simulation import SimulatedHardware
from .startup import (
    StartupConfig,
    StartupResult,
    prepare_source_for_optimization,
)


@dataclass(frozen=True)
class OptimizationProfile:
    magnet_lower_a: float
    magnet_upper_a: float

    magnet_scan: ScanConfig
    einzel_scan: ScanConfig
    extraction_scan: ScanConfig
    sputter_scan: ScanConfig

    measurement: MeasurementConfig
    convergence: ConvergenceConfig
    startup: StartupConfig


@dataclass(frozen=True)
class RuntimeResult:
    mass_u: float
    startup: StartupResult
    session: OptimizationSessionResult
    measurement_count: int


SIMULATION_PROFILE = OptimizationProfile(
    magnet_lower_a=0.0,
    magnet_upper_a=120.0,

    magnet_scan=AdaptiveScanConfig(
        initial_step=0.20,
        min_step=0.05,
        max_evaluations=7,
    ),

    einzel_scan=AdaptiveScanConfig(
        initial_step=0.50,
        min_step=0.10,
        max_evaluations=7,
    ),

    extraction_scan=AdaptiveScanConfig(
        initial_step=1.00,
        min_step=0.25,
        max_evaluations=9,
    ),

    sputter_scan=AdaptiveScanConfig(
        initial_step=1.00,
        min_step=0.25,
        max_evaluations=8,
    ),

    measurement=MeasurementConfig(
        sample_count=5,
        settle_seconds=0.20,
        inter_sample_seconds=0.05,
    ),

    convergence=ConvergenceConfig(
        max_cycles=3,
        min_relative_improvement=0.01,
    ),

    startup=StartupConfig(),
)


def run_simulation(
    mass_u: float,
    *,
    profile: OptimizationProfile = SIMULATION_PROFILE,
) -> RuntimeResult:
    """
    Exercise the complete mcsirius algorithm without touching
    any real hardware.
    """

    raw = SimulatedHardware(
        mass_u=mass_u
    )

    startup = prepare_source_for_optimization(
        raw,
        config=profile.startup,
        sleep=lambda _: None,
    )

    hardware = RobustCup1Hardware(
        raw,
        config=profile.measurement,
        sleep=lambda _: None,
    )

    session = run_single_pass(
        hardware,
        mass_u=mass_u,
        operating_point=(
            startup.final_readback
        ),
        magnet_lower_a=(
            profile.magnet_lower_a
        ),
        magnet_upper_a=(
            profile.magnet_upper_a
        ),
        magnet_scan=profile.magnet_scan,
        einzel_scan=profile.einzel_scan,
        extraction_scan=(
            profile.extraction_scan
        ),
        sputter_scan=profile.sputter_scan,
    )

    return RuntimeResult(
        mass_u=float(mass_u),
        startup=startup,
        session=session,
        measurement_count=len(
            hardware.measurement_history
        ),
    )