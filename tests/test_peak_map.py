import math

import pytest

from mcsirius.machine import (
    OperatingPoint,
)
from mcsirius.peak_map import (
    PeakMapConfig,
    _assert_source_near_start,
    acquire_peak_map,
    predicted_mass_positions,
)


class FakeHardware:
    def __init__(self):
        self.magnet_a = 0.0
        self.positions = []

    def set_magnet_current(
        self,
        current_a,
    ):
        self.magnet_a = current_a
        self.positions.append(
            current_a
        )

    def read_cup1_current(self):
        # Three synthetic neighboring peaks.
        value = 0.0

        for center, amplitude, width in (
            (42.91, 2.0e-9, 0.025),
            (43.06, 5.0e-9, 0.025),
            (43.21, 3.0e-9, 0.025),
        ):
            value += (
                amplitude
                * math.exp(
                    -0.5
                    * (
                        (
                            self.magnet_a
                            - center
                        )
                        / width
                    ) ** 2
                )
            )

        return -value


def test_default_peak_map_uses_31_positions():
    hardware = FakeHardware()

    result = acquire_peak_map(
        hardware,
        mass_u=145.0,
    )

    assert len(result.samples) == 31

    assert len(
        hardware.positions
    ) == 31

    assert hardware.positions == sorted(
        hardware.positions
    )


def test_neighbor_predictions_cover_five_masses():
    positions = predicted_mass_positions(
        145.0,
        neighbor_masses=2,
    )

    assert [
        item.mass_u
        for item in positions
    ] == [
        143.0,
        144.0,
        145.0,
        146.0,
        147.0,
    ]

    currents = [
        item.magnet_current_a
        for item in positions
    ]

    assert currents == sorted(currents)


def test_peak_map_detects_real_maxima_without_mass_labels():
    hardware = FakeHardware()

    result = acquire_peak_map(
        hardware,
        mass_u=145.0,
    )

    peaks = [
        peak.magnet_current_a
        for peak in result.observed_peaks
    ]

    assert len(peaks) >= 2

    assert any(
        abs(value - 43.06) <= 0.03
        for value in peaks
    )


def test_source_gate_accepts_observed_live_prepare_values():
    _assert_source_near_start(
        OperatingPoint(
            sputter_kv=3.833,
            extraction_kv=13.231,
            einzel_kv=13.126,
        ),
        tolerance_fraction=0.10,
    )


def test_source_gate_rejects_large_deviation():
    with pytest.raises(
        RuntimeError,
        match="Extraction",
    ):
        _assert_source_near_start(
            OperatingPoint(
                sputter_kv=4.0,
                extraction_kv=12.0,
                einzel_kv=14.0,
            ),
            tolerance_fraction=0.10,
        )


def test_peak_map_width_must_match_step():
    hardware = FakeHardware()

    with pytest.raises(
        ValueError,
        match="integer multiple",
    ):
        acquire_peak_map(
            hardware,
            mass_u=145.0,
            config=PeakMapConfig(
                half_width_a=0.44,
                step_a=0.03,
            ),
        )
