import pytest

from mcsirius.flavia import (
    FlaviaAdapterConfig,
    FlaviaHardware,
    FlaviaHardwareError,
)


class Channel:
    def __init__(self, value, timestamp):
        self.value = value
        self.timestamp = timestamp
        self.quality = "good"


class SequenceModel:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def get(self, name):
        if name != "magnet_current_meas":
            return None

        index = min(
            self.index,
            len(self.values) - 1,
        )

        value = self.values[index]

        self.index += 1

        return Channel(
            value=value,
            timestamp=float(self.index),
        )


class Backend:
    def __init__(self, values):
        self.model = SequenceModel(values)


class Clock:
    def __init__(self):
        self.time = 0.0

    def monotonic(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


def test_waits_for_two_stable_magnet_readbacks():
    clock = Clock()

    backend = Backend(
        [
            43.80,
            43.30,
            43.08,
            43.06,
        ]
    )

    hardware = FlaviaHardware(
        backend,
        config=FlaviaAdapterConfig(
            magnet_settle_tolerance_a=0.05,
            magnet_settle_timeout_s=5.0,
            magnet_settle_poll_s=0.1,
            magnet_settle_samples=2,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    result = hardware.wait_for_magnet_settled(
        43.0536
    )

    assert result == pytest.approx(
        43.06
    )


def test_magnet_settle_timeout_is_detected():
    clock = Clock()

    backend = Backend(
        [
            43.80,
        ]
    )

    hardware = FlaviaHardware(
        backend,
        config=FlaviaAdapterConfig(
            magnet_settle_tolerance_a=0.05,
            magnet_settle_timeout_s=0.5,
            magnet_settle_poll_s=0.1,
            magnet_settle_samples=2,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(
        FlaviaHardwareError,
        match="did not settle",
    ):
        hardware.wait_for_magnet_settled(
            43.0536
        )
