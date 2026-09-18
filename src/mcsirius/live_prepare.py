"""Controlled first live preparation of FLAVIA."""

from __future__ import annotations

from dataclasses import dataclass

from .flavia import (
    FlaviaAdapterConfig,
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
    MagnetSetpoint,
    calculate_magnet_setpoint,
)
from .startup import (
    StartupResult,
    prepare_source_for_optimization,
)


@dataclass(frozen=True)
class LivePrepareResult:
    mass_u: float
    source: StartupResult
    magnet: MagnetSetpoint
    final_source_readback: OperatingPoint
    final_magnet_readback_a: float
    cup_current_a: float


def prepare_live_machine(
    mass_u: float,
) -> LivePrepareResult:
    """
    Prepare the real source for optimization, but do not start
    any optimization scans.

    The machine is left at the deterministic starting point.
    """

    backend = StandaloneLiveBackend(
        configure_keithley=True
    )

    backend.start()

    try:
        backend.wait_until_ready()

        hardware = FlaviaHardware(
            backend,
            config=FlaviaAdapterConfig(
                magnet_rate_a_per_s=1.0,
                magnet_update_period_s=0.25,
            ),
        )

        source = prepare_source_for_optimization(
            hardware
        )

        magnet = calculate_magnet_setpoint(
            mass_u=float(mass_u),
            sputter_kv=(
                DEFAULT_OPERATING_POINT.sputter_kv
            ),
            extraction_kv=(
                DEFAULT_OPERATING_POINT.extraction_kv
            ),
        )

        hardware.set_magnet_current(
            magnet.current_a
        )

        final_source = (
            hardware.read_operating_point()
        )

        final_magnet = (
            hardware.read_magnet_current()
        )

        cup_current = (
            hardware.read_cup1_current()
        )

        return LivePrepareResult(
            mass_u=float(mass_u),
            source=source,
            magnet=magnet,
            final_source_readback=final_source,
            final_magnet_readback_a=final_magnet,
            cup_current_a=cup_current,
        )

    finally:
        backend.stop()
