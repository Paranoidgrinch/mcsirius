import pytest

from mcsirius.machine import OperatingPoint
from mcsirius.ramp import (
    RampConfig,
    ramp_source_voltages,
    validate_startup_operating_point,
)


class FakeRampHardware:
    def __init__(
        self,
        point,
    ):
        self.sputter_kv = point.sputter_kv
        self.extraction_kv = (
            point.extraction_kv
        )
        self.einzel_kv = point.einzel_kv

        self.time_s = 0.0
        self.writes = []

    def sleep(self, duration_s):
        self.time_s += duration_s

    def read_operating_point(self):
        return OperatingPoint(
            self.sputter_kv,
            self.extraction_kv,
            self.einzel_kv,
        )

    def _record(
        self,
        name,
        old,
        new,
    ):
        self.writes.append(
            (
                self.time_s,
                name,
                old,
                new,
            )
        )

    def _check_pair(self):
        assert (
            abs(
                self.einzel_kv
                - self.extraction_kv
            )
            <= 2.0 + 1e-9
        )

    def set_sputter_voltage(
        self,
        value,
    ):
        old = self.sputter_kv
        self.sputter_kv = value

        self._record(
            "sputter",
            old,
            value,
        )

    def set_extraction_voltage(
        self,
        value,
    ):
        old = self.extraction_kv
        self.extraction_kv = value

        self._record(
            "extraction",
            old,
            value,
        )

        self._check_pair()

    def set_einzel_voltage(
        self,
        value,
    ):
        old = self.einzel_kv
        self.einzel_kv = value

        self._record(
            "einzel",
            old,
            value,
        )

        self._check_pair()


def test_zero_to_default_takes_fourteen_seconds():
    hardware = FakeRampHardware(
        OperatingPoint(
            0.0,
            0.0,
            0.0,
        )
    )

    result = ramp_source_voltages(
        hardware,
        sleep=hardware.sleep,
    )

    assert result.duration_s == pytest.approx(
        14.0
    )

    assert hardware.time_s == pytest.approx(
        14.0
    )

    assert hardware.read_operating_point() == (
        OperatingPoint(
            4.0,
            14.0,
            14.0,
        )
    )


def test_upper_values_to_default_take_eleven_seconds():
    hardware = FakeRampHardware(
        OperatingPoint(
            9.0,
            25.0,
            25.0,
        )
    )

    result = ramp_source_voltages(
        hardware,
        sleep=hardware.sleep,
    )

    assert result.duration_s == pytest.approx(
        11.0
    )

    assert hardware.read_operating_point() == (
        OperatingPoint(
            4.0,
            14.0,
            14.0,
        )
    )


def test_voltage_rate_never_exceeds_one_kv_per_second():
    hardware = FakeRampHardware(
        OperatingPoint(
            0.0,
            0.0,
            0.0,
        )
    )

    ramp_source_voltages(
        hardware,
        config=RampConfig(
            rate_kv_per_s=1.0,
            update_period_s=0.25,
            max_duration_s=15.0,
        ),
        sleep=hardware.sleep,
    )

    previous = {
        "sputter": (0.0, 0.0),
        "extraction": (0.0, 0.0),
        "einzel": (0.0, 0.0),
    }

    for (
        timestamp,
        name,
        _,
        value,
    ) in hardware.writes:
        previous_time, previous_value = (
            previous[name]
        )

        delta_t = (
            timestamp - previous_time
        )
        delta_v = abs(
            value - previous_value
        )

        assert delta_t > 0.0

        assert (
            delta_v / delta_t
            <= 1.0 + 1e-9
        )

        previous[name] = (
            timestamp,
            value,
        )


def test_boundary_offset_stays_safe_during_ramp():
    hardware = FakeRampHardware(
        OperatingPoint(
            4.0,
            14.0,
            16.0,
        )
    )

    ramp_source_voltages(
        hardware,
        sleep=hardware.sleep,
    )

    assert hardware.read_operating_point() == (
        OperatingPoint(
            4.0,
            14.0,
            14.0,
        )
    )


def test_ramp_longer_than_limit_is_rejected_before_write():
    hardware = FakeRampHardware(
        OperatingPoint(
            0.0,
            0.0,
            0.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="exceeding",
    ):
        ramp_source_voltages(
            hardware,
            target=OperatingPoint(
                4.0,
                20.0,
                20.0,
            ),
            config=RampConfig(
                rate_kv_per_s=1.0,
                update_period_s=0.25,
                max_duration_s=15.0,
            ),
            sleep=hardware.sleep,
        )

    assert hardware.writes == []


def test_unsafe_initial_pair_is_rejected():
    point = OperatingPoint(
        0.0,
        10.0,
        0.0,
    )

    with pytest.raises(
        ValueError,
        match="difference",
    ):
        validate_startup_operating_point(
            point
        )


def test_already_at_target_performs_no_writes():
    hardware = FakeRampHardware(
        OperatingPoint(
            4.0,
            14.0,
            14.0,
        )
    )

    result = ramp_source_voltages(
        hardware,
        sleep=hardware.sleep,
    )

    assert result.duration_s == 0.0
    assert result.step_count == 0
    assert hardware.writes == []