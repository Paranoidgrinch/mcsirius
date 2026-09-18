from dataclasses import dataclass

import pytest

from mcsirius.flavia import FlaviaHardware
from mcsirius.machine import OperatingPoint
from mcsirius.preflight import (
    run_live_preflight,
)


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
                -3.0e-9,
                timestamp=100.0,
            ),

            "cs/sputter/meas_u_v": FakeChannel(
                4000.0
            ),
            "cs/extraction/meas_u_v": FakeChannel(
                14000.0
            ),
            "cs/einzellens/meas_u_v": FakeChannel(
                14000.0
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

    def select_cup(self, number):
        self.model.channels[
            "cup/selected"
        ].value = number


class FakeBackend:
    def __init__(self):
        self.model = FakeModel()
        self.cup = FakeCup(self.model)

    def set_channel(self, name, value):
        raise AssertionError(
            "Preflight must not write source channels."
        )

    def set_magnet_current(self, value):
        raise AssertionError(
            "Preflight must not write magnet current."
        )


def make_hardware():
    backend = FakeBackend()

    hardware = FlaviaHardware(
        backend,
        wall_clock=lambda: 100.5,
        monotonic=lambda: 10.0,
        sleep=lambda _: None,
    )

    return backend, hardware


def test_preflight_reads_valid_machine_state():
    _, hardware = make_hardware()

    report = run_live_preflight(
        hardware
    )

    assert report.operating_point == OperatingPoint(
        4.0,
        14.0,
        14.0,
    )

    assert report.magnet_current_a == pytest.approx(
        18.75
    )

    assert report.cup_current_a == pytest.approx(
        -3.0e-9
    )

    assert report.selected_cup == 1


def test_preflight_rejects_source_below_search_domain():
    backend, hardware = make_hardware()

    backend.model.channels[
        "cs/sputter/meas_u_v"
    ].value = 3000.0

    with pytest.raises(ValueError):
        run_live_preflight(
            hardware
        )


def test_preflight_rejects_excessive_einzel_offset():
    backend, hardware = make_hardware()

    backend.model.channels[
        "cs/extraction/meas_u_v"
    ].value = 16000.0

    backend.model.channels[
        "cs/einzellens/meas_u_v"
    ].value = 19000.0

    with pytest.raises(
        ValueError,
        match="difference",
    ):
        run_live_preflight(
            hardware
        )


def test_preflight_selects_cup_one_without_source_writes():
    backend, hardware = make_hardware()

    backend.model.channels[
        "cup/selected"
    ].value = 4

    report = run_live_preflight(
        hardware
    )

    assert report.selected_cup == 1