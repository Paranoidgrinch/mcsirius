import pytest

from mcsirius.machine import OperatingPoint
from mcsirius.magnet import (
    calculate_magnet_setpoint,
)
from mcsirius.scan import LocalScanConfig
from mcsirius.source import (
    move_extraction_einzel,
    optimize_source_voltages,
)


class FakeSourceHardware:
    def __init__(
        self,
        *,
        mass_u: float = 27.0,
        sputter_kv: float = 4.0,
        extraction_kv: float = 14.0,
        einzel_kv: float = 14.0,
    ):
        self.mass_u = mass_u
        self.sputter_kv = sputter_kv
        self.extraction_kv = extraction_kv
        self.einzel_kv = einzel_kv

        self.magnet_a = (
            calculate_magnet_setpoint(
                mass_u,
                sputter_kv,
                extraction_kv,
            ).current_a
        )

        self.writes = []

    def _assert_pair_safe(self):
        assert (
            abs(
                self.einzel_kv
                - self.extraction_kv
            )
            <= 2.0 + 1e-9
        )

    def set_sputter_voltage(
        self,
        voltage_kv: float,
    ):
        self.sputter_kv = voltage_kv
        self.writes.append(
            ("sputter", voltage_kv)
        )

    def set_extraction_voltage(
        self,
        voltage_kv: float,
    ):
        self.extraction_kv = voltage_kv
        self.writes.append(
            ("extraction", voltage_kv)
        )
        self._assert_pair_safe()

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ):
        self.einzel_kv = voltage_kv
        self.writes.append(
            ("einzel", voltage_kv)
        )
        self._assert_pair_safe()

    def set_magnet_current(
        self,
        current_a: float,
    ):
        self.magnet_a = current_a
        self.writes.append(
            ("magnet", current_a)
        )

    def read_cup1_current(self):
        predicted = (
            calculate_magnet_setpoint(
                self.mass_u,
                self.sputter_kv,
                self.extraction_kv,
            ).current_a
        )

        magnet_peak = predicted + 0.20

        score = (
            100.0
            - 5.0
            * (
                self.sputter_kv - 6.0
            ) ** 2
            - 3.0
            * (
                self.extraction_kv - 17.0
            ) ** 2
            - 4.0
            * (
                self.einzel_kv
                - self.extraction_kv
            ) ** 2
            - 30.0
            * (
                self.magnet_a
                - magnet_peak
            ) ** 2
        )

        return -max(0.0, score)


def test_safe_voltage_pair_move_handles_large_change():
    hardware = FakeSourceHardware()

    extraction, einzel = (
        move_extraction_einzel(
            hardware,
            current_extraction_kv=14.0,
            current_einzel_kv=14.0,
            target_extraction_kv=22.0,
            target_einzel_kv=22.0,
            max_step_kv=0.5,
        )
    )

    assert extraction == pytest.approx(22.0)
    assert einzel == pytest.approx(22.0)

    assert hardware.extraction_kv == pytest.approx(
        22.0
    )
    assert hardware.einzel_kv == pytest.approx(
        22.0
    )


def test_safe_move_preserves_boundary_offset():
    hardware = FakeSourceHardware(
        extraction_kv=14.0,
        einzel_kv=16.0,
    )

    extraction, einzel = (
        move_extraction_einzel(
            hardware,
            current_extraction_kv=14.0,
            current_einzel_kv=16.0,
            target_extraction_kv=20.0,
            target_einzel_kv=22.0,
            max_step_kv=0.5,
        )
    )

    assert extraction == pytest.approx(20.0)
    assert einzel == pytest.approx(22.0)


def test_invalid_voltage_pair_is_rejected_before_write():
    hardware = FakeSourceHardware()

    with pytest.raises(ValueError):
        move_extraction_einzel(
            hardware,
            current_extraction_kv=14.0,
            current_einzel_kv=14.0,
            target_extraction_kv=20.0,
            target_einzel_kv=23.0,
        )

    assert hardware.writes == []


def test_source_optimizer_finds_voltage_peaks():
    hardware = FakeSourceHardware()

    result = optimize_source_voltages(
        hardware,
        mass_u=27.0,
        operating_point=OperatingPoint(
            sputter_kv=4.0,
            extraction_kv=14.0,
            einzel_kv=14.0,
        ),
        magnet_lower_a=0.0,
        magnet_upper_a=50.0,
        magnet_scan=LocalScanConfig(
            coarse_radius=0.6,
            coarse_step=0.2,
            fine_radius=0.2,
            fine_step=0.05,
        ),
        extraction_scan=LocalScanConfig(
            coarse_radius=4.0,
            coarse_step=1.0,
            fine_radius=0.5,
            fine_step=0.25,
        ),
        sputter_scan=LocalScanConfig(
            coarse_radius=3.0,
            coarse_step=1.0,
            fine_radius=0.5,
            fine_step=0.25,
        ),
    )

    assert (
        result.extraction_scan.best_position
        == pytest.approx(17.0)
    )

    assert (
        result.sputter_scan.best_position
        == pytest.approx(6.0)
    )

    assert (
        result.final_operating_point
        == OperatingPoint(
            sputter_kv=6.0,
            extraction_kv=17.0,
            einzel_kv=17.0,
        )
    )

    expected_seed = (
        calculate_magnet_setpoint(
            27.0,
            6.0,
            17.0,
        ).current_a
    )

    assert (
        result.final_magnet_seed.current_a
        == pytest.approx(expected_seed)
    )

    assert (
        result.final_magnet_scan.best_position
        == pytest.approx(
            expected_seed + 0.20
        )
    )

    assert result.final_cup1_score == pytest.approx(
        100.0
    )


def test_source_optimizer_keeps_einzel_offset():
    hardware = FakeSourceHardware(
        einzel_kv=14.5,
    )

    result = optimize_source_voltages(
        hardware,
        mass_u=27.0,
        operating_point=OperatingPoint(
            4.0,
            14.0,
            14.5,
        ),
        magnet_lower_a=0.0,
        magnet_upper_a=50.0,
        magnet_scan=LocalScanConfig(
            0.6,
            0.2,
            0.2,
            0.05,
        ),
        extraction_scan=LocalScanConfig(
            4.0,
            1.0,
            0.5,
            0.25,
        ),
        sputter_scan=LocalScanConfig(
            3.0,
            1.0,
            0.5,
            0.25,
        ),
    )

    assert (
        result.final_operating_point.einzel_kv
        - result.final_operating_point.extraction_kv
        == pytest.approx(0.5)
    )


def test_source_optimizer_rejects_invalid_start_point():
    hardware = FakeSourceHardware()

    with pytest.raises(ValueError):
        optimize_source_voltages(
            hardware,
            mass_u=27.0,
            operating_point=OperatingPoint(
                3.0,
                14.0,
                14.0,
            ),
            magnet_lower_a=0.0,
            magnet_upper_a=50.0,
            magnet_scan=LocalScanConfig(
                0.5,
                0.25,
                0.2,
                0.05,
            ),
            extraction_scan=LocalScanConfig(
                2.0,
                1.0,
                0.5,
                0.25,
            ),
            sputter_scan=LocalScanConfig(
                2.0,
                1.0,
                0.5,
                0.25,
            ),
        )

    assert hardware.writes == []