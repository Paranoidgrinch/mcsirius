"""Deterministic simulated source-to-Cup-1 hardware."""

from __future__ import annotations

import math

from .machine import OperatingPoint
from .magnet import calculate_magnet_setpoint


class SimulatedHardware:
    """
    Smooth synthetic FLAVIA front end for end-to-end testing.

    The optimizer sees exactly the same hardware methods that
    real hardware will provide later.
    """

    def __init__(
        self,
        *,
        mass_u: float,
        operating_point: OperatingPoint = OperatingPoint(
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        mass_u = float(mass_u)

        if (
            not math.isfinite(mass_u)
            or mass_u <= 0.0
        ):
            raise ValueError(
                "Ion mass must be finite and greater than zero."
            )

        self.mass_u = mass_u

        self.sputter_kv = operating_point.sputter_kv
        self.extraction_kv = operating_point.extraction_kv
        self.einzel_kv = operating_point.einzel_kv

        self.magnet_a = (
            calculate_magnet_setpoint(
                mass_u,
                self.sputter_kv,
                self.extraction_kv,
            ).current_a
        )

        self.selected_cup = 1
        self.prepared = False

    def prepare_for_optimization(self) -> None:
        self.selected_cup = 1
        self.prepared = True

    def read_operating_point(
        self,
    ) -> OperatingPoint:
        return OperatingPoint(
            sputter_kv=self.sputter_kv,
            extraction_kv=self.extraction_kv,
            einzel_kv=self.einzel_kv,
        )

    def set_sputter_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.sputter_kv = float(voltage_kv)

    def set_extraction_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.extraction_kv = float(voltage_kv)

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.einzel_kv = float(voltage_kv)

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        self.magnet_a = float(current_a)

    def read_cup1_current(self) -> float:
        """
        Synthetic negative-ion current.

        Optimum:
            sputter   = 6.0 kV
            extraction = 17.0 kV
            Einzel    = extraction + 0.5 kV
            magnet    = calculated seed + 0.2 A

        Peak current magnitude is 100 nA.
        """

        seed = calculate_magnet_setpoint(
            self.mass_u,
            self.sputter_kv,
            self.extraction_kv,
        ).current_a

        magnet_peak = seed + 0.20
        einzel_peak = self.extraction_kv + 0.50

        penalty = (
            (
                (self.sputter_kv - 6.0)
                / 1.3
            ) ** 2
            + (
                (self.extraction_kv - 17.0)
                / 2.0
            ) ** 2
            + (
                (self.einzel_kv - einzel_peak)
                / 0.7
            ) ** 2
            + (
                (self.magnet_a - magnet_peak)
                / 0.25
            ) ** 2
        )

        magnitude_a = (
            100.0e-9
            * math.exp(-0.5 * penalty)
        )

        return -magnitude_a