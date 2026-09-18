from dataclasses import dataclass

import pytest

from mcsirius.flavia import (
    FlaviaAdapterConfig,
    FlaviaHardware,
    FlaviaHardwareError,
)
from mcsirius.machine import OperatingPoint


@dataclass
class FakeChannel:
    value: object
    timestamp: float = 100.0
    quality: str = "good"


class FakeModel:
    def __init__(self):
        self.channels = {
            "mqtt_connected": FakeChannel(True),
            "magnet_connected": FakeChannel(True),
            "keithley/connected": FakeChannel(True),
            "cup/connected": FakeChannel(True),
            "cup/selected": FakeChannel(1),
            "keithley/current_A": FakeChannel(
                -2.5e-9,
                timestamp=100.0,
            ),
            "cs/sputter/meas_u_v": FakeChannel(
                4000.0
            ),
            "cs/extraction/meas_u_v": FakeChannel(
                14000.0
            ),
            "cs/einzellens/meas_u_v": FakeChannel(
                14500.0
            ),
            "magnet_current_meas": FakeChannel(
                18.75
            ),
        }

    def get(self, name):
        return self.channels.get(name)


class FakeCup:
    def __init__(self, model):
        self.model = model
        self.commands = []

    def select_cup(self, number):
        self.commands.append(number)
        self.model.channels[
            "cup/selected"
        ].value = number


class FakeBackend:
    def __init__(self):
        self.model = FakeModel()
        self.cup = FakeCup(self.model)

        self.channel_writes = []
        self.magnet_writes = []

    def set_channel(
        self,
        channel,
        value,
    ):
        self.channel_writes.append(
            (channel, value)
        )

    def set_magnet_current(
        self,
        value,
    ):
        self.magnet_writes.append(value)


def make_hardware(
    backend=None,
):
    if backend is None:
        backend = FakeBackend()

    hardware = FlaviaHardware(
        backend,
        wall_clock=lambda: 100.5,
        monotonic=lambda: 10.0,
        sleep=lambda _: None,
    )

    return backend, hardware


def test_source_voltage_writes_convert_kv_to_v():
    backend, hardware = make_hardware()

    hardware.set_sputter_voltage(4.0)
    hardware.set_extraction_voltage(14.0)
    hardware.set_einzel_voltage(14.5)

    assert backend.channel_writes == [
        (
            "cs/sputter/set_u_v",
            4000.0,
        ),
        (
            "cs/extraction/set_u_v",
            14000.0,
        ),
        (
            "cs/einzellens/set_u_v",
            14500.0,
        ),
    ]


def test_magnet_write_uses_flavia_backend_api():
    backend, hardware = make_hardware()

    hardware.set_magnet_current(19.0)

    assert backend.magnet_writes == [
        19.0
    ]


def test_prepare_selects_cup_one():
    backend = FakeBackend()
    backend.model.channels[
        "cup/selected"
    ].value = 3

    _, hardware = make_hardware(
        backend
    )

    hardware.prepare_for_optimization()

    assert backend.cup.commands == [1]
    assert hardware.selected_cup == 1


def test_prepare_does_not_reselect_cup_one():
    backend, hardware = make_hardware()

    hardware.prepare_for_optimization()

    assert backend.cup.commands == []


def test_read_cup_current_requires_fresh_value():
    _, hardware = make_hardware()

    assert (
        hardware.read_cup1_current()
        == pytest.approx(-2.5e-9)
    )


def test_stale_keithley_value_is_rejected():
    backend = FakeBackend()

    backend.model.channels[
        "keithley/current_A"
    ].timestamp = 90.0

    hardware = FlaviaHardware(
        backend,
        config=FlaviaAdapterConfig(
            current_max_age_s=2.0,
        ),
        wall_clock=lambda: 100.0,
        monotonic=lambda: 10.0,
        sleep=lambda _: None,
    )

    with pytest.raises(
        FlaviaHardwareError,
        match="stale",
    ):
        hardware.read_cup1_current()


def test_current_read_requires_correct_cup():
    backend, hardware = make_hardware()

    backend.model.channels[
        "cup/selected"
    ].value = 2

    with pytest.raises(
        FlaviaHardwareError,
        match="not selected",
    ):
        hardware.read_cup1_current()


def test_disconnected_hardware_is_rejected():
    backend, hardware = make_hardware()

    backend.model.channels[
        "keithley/connected"
    ].value = False

    with pytest.raises(
        FlaviaHardwareError,
        match="Keithley",
    ):
        hardware.assert_ready()


def test_source_readbacks_are_converted_to_kv():
    _, hardware = make_hardware()

    point = hardware.read_operating_point()

    assert point == OperatingPoint(
        sputter_kv=4.0,
        extraction_kv=14.0,
        einzel_kv=14.5,
    )


def test_magnet_readback_is_available():
    _, hardware = make_hardware()

    assert (
        hardware.read_magnet_current()
        == pytest.approx(18.75)
    )