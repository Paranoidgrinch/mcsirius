import pytest

from mcsirius.flavia import (
    FlaviaAdapterConfig,
    FlaviaHardware,
)


class Channel:
    def __init__(self, value):
        self.value = value
        self.timestamp = 0.0
        self.quality = "good"


class Model:
    def __init__(self, current):
        self.channels = {
            "magnet_current_meas":
                Channel(current),
        }

    def get(self, name):
        return self.channels.get(name)


class Backend:
    def __init__(self, current):
        self.model = Model(current)
        self.writes = []

    def set_magnet_current(self, value):
        self.writes.append(value)

        self.model.channels[
            "magnet_current_meas"
        ].value = value


def test_large_magnet_change_is_ramped():
    backend = Backend(51.0)

    sleeps = []

    hardware = FlaviaHardware(
        backend,
        config=FlaviaAdapterConfig(
            magnet_rate_a_per_s=1.0,
            magnet_update_period_s=0.25,
        ),
        sleep=sleeps.append,
    )

    hardware.set_magnet_current(49.0)

    assert len(backend.writes) == 8

    assert backend.writes[-1] == pytest.approx(
        49.0
    )

    assert sum(sleeps) == pytest.approx(
        2.0
    )

    previous = 51.0

    for value in backend.writes:
        assert abs(value - previous) <= (
            0.25 + 1e-12
        )

        previous = value


def test_small_magnet_change_uses_same_rate_limit():
    backend = Backend(20.0)

    sleeps = []

    hardware = FlaviaHardware(
        backend,
        config=FlaviaAdapterConfig(
            magnet_rate_a_per_s=1.0,
            magnet_update_period_s=0.25,
        ),
        sleep=sleeps.append,
    )

    hardware.set_magnet_current(20.2)

    assert backend.writes[-1] == pytest.approx(
        20.2
    )

    assert sum(sleeps) == pytest.approx(
        0.2
    )


def test_zero_magnet_change_does_nothing():
    backend = Backend(20.0)

    hardware = FlaviaHardware(
        backend,
        sleep=lambda _: None,
    )

    hardware.set_magnet_current(20.0)

    assert backend.writes == []
