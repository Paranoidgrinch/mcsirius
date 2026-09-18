import math

import pytest

from mcsirius.measurement import (
    MeasurementConfig,
    RobustCup1Hardware,
    acquire_cup1_measurement,
)


def test_measurement_uses_median_against_outlier():
    values = iter(
        [
            -10.0,
            -10.1,
            -999.0,
            -9.9,
            -10.0,
        ]
    )

    sleeps = []

    result = acquire_cup1_measurement(
        lambda: next(values),
        config=MeasurementConfig(
            sample_count=5,
            settle_seconds=0.2,
            inter_sample_seconds=0.05,
        ),
        sleep=sleeps.append,
    )

    assert result.median_current == pytest.approx(
        -10.0
    )
    assert result.magnitude == pytest.approx(
        10.0
    )
    assert result.mad == pytest.approx(
        0.1
    )

    assert sleeps == pytest.approx(
        [
            0.2,
            0.05,
            0.05,
            0.05,
            0.05,
        ]
    )


def test_non_finite_current_is_rejected():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        acquire_cup1_measurement(
            lambda: math.nan,
            config=MeasurementConfig(
                sample_count=1,
                settle_seconds=0.0,
                inter_sample_seconds=0.0,
            ),
            sleep=lambda _: None,
        )


@pytest.mark.parametrize(
    "config",
    [
        MeasurementConfig(sample_count=0),
        MeasurementConfig(sample_count=-1),
        MeasurementConfig(
            sample_count=1,
            settle_seconds=-0.1,
        ),
        MeasurementConfig(
            sample_count=1,
            inter_sample_seconds=-0.1,
        ),
    ],
)
def test_invalid_measurement_config_is_rejected(
    config,
):
    with pytest.raises(ValueError):
        config.validate()


class FakeHardware:
    def __init__(self):
        self.sputter = 4.0
        self.extraction = 14.0
        self.einzel = 14.0
        self.magnet = 18.0

        self.read_values = []
        self.writes = []

    def set_sputter_voltage(self, value):
        self.sputter = value
        self.writes.append(
            ("sputter", value)
        )

    def set_extraction_voltage(self, value):
        self.extraction = value
        self.writes.append(
            ("extraction", value)
        )

    def set_einzel_voltage(self, value):
        self.einzel = value
        self.writes.append(
            ("einzel", value)
        )

    def set_magnet_current(self, value):
        self.magnet = value
        self.writes.append(
            ("magnet", value)
        )

    def read_cup1_current(self):
        if self.read_values:
            return self.read_values.pop(0)

        return -10.0


def test_wrapper_forwards_hardware_writes():
    raw = FakeHardware()

    wrapper = RobustCup1Hardware(
        raw,
        config=MeasurementConfig(
            sample_count=1,
            settle_seconds=0.0,
            inter_sample_seconds=0.0,
        ),
        sleep=lambda _: None,
    )

    wrapper.set_sputter_voltage(5.0)
    wrapper.set_extraction_voltage(15.0)
    wrapper.set_einzel_voltage(15.5)
    wrapper.set_magnet_current(20.0)

    assert raw.writes == [
        ("sputter", 5.0),
        ("extraction", 15.0),
        ("einzel", 15.5),
        ("magnet", 20.0),
    ]


def test_wrapper_settles_only_after_change():
    raw = FakeHardware()

    sleeps = []

    wrapper = RobustCup1Hardware(
        raw,
        config=MeasurementConfig(
            sample_count=1,
            settle_seconds=0.3,
            inter_sample_seconds=0.0,
        ),
        sleep=sleeps.append,
    )

    wrapper.read_cup1_current()
    wrapper.read_cup1_current()

    assert sleeps == pytest.approx([0.3])

    wrapper.set_magnet_current(19.0)
    wrapper.read_cup1_current()

    assert sleeps == pytest.approx(
        [0.3, 0.3]
    )


def test_wrapper_records_measurement_history():
    raw = FakeHardware()

    raw.read_values = [
        -1.0,
        -1.1,
        -0.9,
        -2.0,
        -2.1,
        -1.9,
    ]

    wrapper = RobustCup1Hardware(
        raw,
        config=MeasurementConfig(
            sample_count=3,
            settle_seconds=0.0,
            inter_sample_seconds=0.0,
        ),
        sleep=lambda _: None,
    )

    first = wrapper.read_cup1_current()
    second = wrapper.read_cup1_current()

    assert first == pytest.approx(-1.0)
    assert second == pytest.approx(-2.0)

    assert len(
        wrapper.measurement_history
    ) == 2

    assert (
        wrapper.last_measurement.median_current
        == pytest.approx(-2.0)
    )