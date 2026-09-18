import pytest

from mcsirius.machine import OperatingPoint
from mcsirius.startup import (
    StartupConfig,
    prepare_source_for_optimization,
)


class FakeStartupHardware:
    def __init__(
        self,
        point,
        *,
        follow_commands=True,
    ):
        self.sputter_kv = point.sputter_kv
        self.extraction_kv = (
            point.extraction_kv
        )
        self.einzel_kv = point.einzel_kv

        self.follow_commands = (
            follow_commands
        )

        self.prepared = False
        self.time_s = 0.0

    def prepare_for_optimization(self):
        self.prepared = True

    def sleep(self, seconds):
        self.time_s += seconds

    def read_operating_point(self):
        return OperatingPoint(
            self.sputter_kv,
            self.extraction_kv,
            self.einzel_kv,
        )

    def set_sputter_voltage(self, value):
        if self.follow_commands:
            self.sputter_kv = value

    def set_extraction_voltage(self, value):
        if self.follow_commands:
            self.extraction_kv = value

    def set_einzel_voltage(self, value):
        if self.follow_commands:
            self.einzel_kv = value


def test_zero_source_reaches_start_in_under_fifteen_seconds():
    hardware = FakeStartupHardware(
        OperatingPoint(
            0.0,
            0.0,
            0.0,
        )
    )

    result = prepare_source_for_optimization(
        hardware,
        sleep=hardware.sleep,
    )

    assert hardware.prepared is True

    assert (
        result.ramp.duration_s
        == pytest.approx(14.0)
    )

    assert (
        result.total_planned_time_s
        == pytest.approx(14.25)
    )

    assert hardware.time_s == pytest.approx(
        14.25
    )

    assert result.final_readback == (
        OperatingPoint(
            4.0,
            14.0,
            14.0,
        )
    )


def test_high_source_reaches_start_quickly():
    hardware = FakeStartupHardware(
        OperatingPoint(
            9.0,
            25.0,
            25.0,
        )
    )

    result = prepare_source_for_optimization(
        hardware,
        sleep=hardware.sleep,
    )

    assert (
        result.ramp.duration_s
        == pytest.approx(11.0)
    )

    assert (
        result.total_planned_time_s
        == pytest.approx(11.25)
    )


def test_failed_readback_is_detected():
    hardware = FakeStartupHardware(
        OperatingPoint(
            0.0,
            0.0,
            0.0,
        ),
        follow_commands=False,
    )

    with pytest.raises(
        RuntimeError,
        match="readback",
    ):
        prepare_source_for_optimization(
            hardware,
            sleep=hardware.sleep,
        )


def test_readback_tolerance_is_configurable():
    hardware = FakeStartupHardware(
        OperatingPoint(
            4.0,
            14.0,
            14.0,
        )
    )

    hardware.sputter_kv = 3.9

    result = prepare_source_for_optimization(
        hardware,
        config=StartupConfig(
            final_settle_s=0.0,
            readback_tolerance_kv=0.2,
        ),
        sleep=hardware.sleep,
    )

    assert (
        result.final_readback.sputter_kv
        == pytest.approx(4.0)
    )