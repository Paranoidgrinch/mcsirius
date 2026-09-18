"""Adapter between mcsirius and the existing FLAVIA backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

from .machine import OperatingPoint


SPUTTER_SET_CHANNEL = "cs/sputter/set_u_v"
SPUTTER_MEAS_CHANNEL = "cs/sputter/meas_u_v"

EXTRACTION_SET_CHANNEL = "cs/extraction/set_u_v"
EXTRACTION_MEAS_CHANNEL = "cs/extraction/meas_u_v"

EINZEL_SET_CHANNEL = "cs/einzellens/set_u_v"
EINZEL_MEAS_CHANNEL = "cs/einzellens/meas_u_v"

MAGNET_MEAS_CHANNEL = "magnet_current_meas"

KEITHLEY_CURRENT_CHANNEL = "keithley/current_A"

MQTT_CONNECTED_CHANNEL = "mqtt_connected"
MAGNET_CONNECTED_CHANNEL = "magnet_connected"
KEITHLEY_CONNECTED_CHANNEL = "keithley/connected"
CUP_CONNECTED_CHANNEL = "cup/connected"
CUP_SELECTED_CHANNEL = "cup/selected"


class FlaviaHardwareError(RuntimeError):
    """Raised when the FLAVIA backend is not ready or valid."""


@dataclass(frozen=True)
class FlaviaAdapterConfig:
    cup_number: int = 1
    current_max_age_s: float = 2.0
    cup_select_timeout_s: float = 5.0
    cup_poll_interval_s: float = 0.05

    magnet_rate_a_per_s: float = 1.0
    magnet_update_period_s: float = 0.25

    def validate(self) -> None:
        if (
            type(self.cup_number) is not int
            or self.cup_number < 1
        ):
            raise ValueError(
                "Cup number must be a positive integer."
            )

        for name, value in (
            (
                "current_max_age_s",
                self.current_max_age_s,
            ),
            (
                "cup_select_timeout_s",
                self.cup_select_timeout_s,
            ),
            (
                "cup_poll_interval_s",
                self.cup_poll_interval_s,
            ),
            (
                "magnet_rate_a_per_s",
                self.magnet_rate_a_per_s,
            ),
            (
                "magnet_update_period_s",
                self.magnet_update_period_s,
            ),
        ):
            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and greater than zero."
                )


class FlaviaHardware:
    """
    Thin adapter over an already running FLAVIA Backend.

    mcsirius uses kV for source voltages.
    FLAVIA source channels use V.
    """

    def __init__(
        self,
        backend: Any,
        *,
        config: FlaviaAdapterConfig = FlaviaAdapterConfig(),
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()

        self.backend = backend
        self.config = config
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        self.sleep = sleep

    def _channel(
        self,
        name: str,
    ) -> Any:
        channel = self.backend.model.get(name)

        if channel is None:
            raise FlaviaHardwareError(
                f"FLAVIA channel is unavailable: {name}"
            )

        return channel

    def _numeric_channel(
        self,
        name: str,
        *,
        reject_bad_quality: bool = True,
    ) -> float:
        channel = self._channel(name)

        if channel.value is None:
            raise FlaviaHardwareError(
                f"FLAVIA channel has no value: {name}"
            )

        if (
            reject_bad_quality
            and getattr(
                channel,
                "quality",
                "unknown",
            ) == "bad"
        ):
            raise FlaviaHardwareError(
                f"FLAVIA channel quality is bad: {name}"
            )

        try:
            value = float(channel.value)
        except (TypeError, ValueError) as exc:
            raise FlaviaHardwareError(
                f"FLAVIA channel is not numeric: {name}"
            ) from exc

        if not math.isfinite(value):
            raise FlaviaHardwareError(
                f"FLAVIA channel is not finite: {name}"
            )

        return value

    def _connected(
        self,
        name: str,
    ) -> bool:
        channel = self.backend.model.get(name)

        if channel is None:
            return False

        return bool(channel.value)

    def assert_ready(self) -> None:
        required = (
            (
                MQTT_CONNECTED_CHANNEL,
                "MQTT",
            ),
            (
                MAGNET_CONNECTED_CHANNEL,
                "magnet",
            ),
            (
                KEITHLEY_CONNECTED_CHANNEL,
                "Keithley",
            ),
            (
                CUP_CONNECTED_CHANNEL,
                "cup switch",
            ),
        )

        missing = [
            label
            for channel, label in required
            if not self._connected(channel)
        ]

        if missing:
            raise FlaviaHardwareError(
                "FLAVIA hardware is not ready: "
                + ", ".join(missing)
            )

    @property
    def selected_cup(self) -> int | None:
        channel = self.backend.model.get(
            CUP_SELECTED_CHANNEL
        )

        if (
            channel is None
            or channel.value is None
        ):
            return None

        try:
            return int(float(channel.value))
        except (TypeError, ValueError):
            return None

    def prepare_for_optimization(self) -> None:
        """
        Verify connections and ensure that Cup 1 is selected.
        """

        self.assert_ready()

        if self.selected_cup == self.config.cup_number:
            return

        self.backend.cup.select_cup(
            self.config.cup_number
        )

        deadline = (
            self.monotonic()
            + self.config.cup_select_timeout_s
        )

        while True:
            if (
                self.selected_cup
                == self.config.cup_number
            ):
                return

            if self.monotonic() >= deadline:
                raise FlaviaHardwareError(
                    "Timed out while selecting "
                    f"Cup {self.config.cup_number}."
                )

            self.sleep(
                self.config.cup_poll_interval_s
            )

    def set_sputter_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        voltage_kv = float(voltage_kv)

        if (
            not math.isfinite(voltage_kv)
            or voltage_kv < 0.0
        ):
            raise ValueError(
                "Sputter voltage must be finite "
                "and non-negative."
            )

        self.backend.set_channel(
            SPUTTER_SET_CHANNEL,
            voltage_kv * 1000.0,
        )

    def set_extraction_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        voltage_kv = float(voltage_kv)

        if (
            not math.isfinite(voltage_kv)
            or voltage_kv < 0.0
        ):
            raise ValueError(
                "Extraction voltage must be finite "
                "and non-negative."
            )

        self.backend.set_channel(
            EXTRACTION_SET_CHANNEL,
            voltage_kv * 1000.0,
        )

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        voltage_kv = float(voltage_kv)

        if (
            not math.isfinite(voltage_kv)
            or voltage_kv < 0.0
        ):
            raise ValueError(
                "Einzel voltage must be finite "
                "and non-negative."
            )

        self.backend.set_channel(
            EINZEL_SET_CHANNEL,
            voltage_kv * 1000.0,
        )

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        """
        Move the analyzing magnet with a bounded software slew.

        Large current changes are therefore never sent as one
        instantaneous setpoint. Small optimizer corrections use
        exactly the same path.
        """

        target_a = float(current_a)

        if (
            not math.isfinite(target_a)
            or target_a < 0.0
        ):
            raise ValueError(
                "Magnet current must be finite "
                "and non-negative."
            )

        start_a = self.read_magnet_current()

        delta_a = target_a - start_a

        if delta_a == 0.0:
            return

        duration_s = (
            abs(delta_a)
            / self.config.magnet_rate_a_per_s
        )

        step_count = max(
            1,
            math.ceil(
                duration_s
                / self.config.magnet_update_period_s
            ),
        )

        step_duration_s = (
            duration_s / step_count
        )

        for index in range(
            1,
            step_count + 1,
        ):
            fraction = index / step_count

            next_a = (
                start_a
                + delta_a * fraction
            )

            self.backend.set_magnet_current(
                next_a
            )

            self.sleep(
                step_duration_s
            )

    def read_operating_point(
        self,
    ) -> OperatingPoint:
        return OperatingPoint(
            sputter_kv=(
                self._numeric_channel(
                    SPUTTER_MEAS_CHANNEL
                )
                / 1000.0
            ),
            extraction_kv=(
                self._numeric_channel(
                    EXTRACTION_MEAS_CHANNEL
                )
                / 1000.0
            ),
            einzel_kv=(
                self._numeric_channel(
                    EINZEL_MEAS_CHANNEL
                )
                / 1000.0
            ),
        )

    def read_magnet_current(
        self,
    ) -> float:
        return self._numeric_channel(
            MAGNET_MEAS_CHANNEL
        )

    def read_cup1_current(
        self,
    ) -> float:
        if (
            self.selected_cup
            != self.config.cup_number
        ):
            raise FlaviaHardwareError(
                f"Cup {self.config.cup_number} "
                "is not selected."
            )

        channel = self._channel(
            KEITHLEY_CURRENT_CHANNEL
        )

        if getattr(
            channel,
            "quality",
            "unknown",
        ) == "bad":
            raise FlaviaHardwareError(
                "Keithley current quality is bad."
            )

        if channel.value is None:
            raise FlaviaHardwareError(
                "Keithley current is unavailable."
            )

        try:
            current = float(channel.value)
        except (TypeError, ValueError) as exc:
            raise FlaviaHardwareError(
                "Keithley current is not numeric."
            ) from exc

        if not math.isfinite(current):
            raise FlaviaHardwareError(
                "Keithley current is not finite."
            )

        try:
            timestamp = float(channel.timestamp)
        except (TypeError, ValueError) as exc:
            raise FlaviaHardwareError(
                "Keithley timestamp is invalid."
            ) from exc

        age_s = self.wall_clock() - timestamp

        if age_s > self.config.current_max_age_s:
            raise FlaviaHardwareError(
                "Keithley current is stale "
                f"({age_s:.3f} s old)."
            )

        return current
