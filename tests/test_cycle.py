import pytest

from mcsirius.cycle import (
    run_optimization_cycle,
)
from mcsirius.machine import OperatingPoint
from mcsirius.magnet import (
    calculate_magnet_setpoint,
)
from mcsirius.scan import LocalScanConfig


class CoupledFakeHardware:
    def __init__(self):
        self.mass_u = 27.0

        self.sputter_kv = 4.0
        self.extraction_kv = 14.0
        self.einzel_kv = 14.0

        self.magnet_a = (
            calculate_magnet_setpoint(
                27.0,
                4.0,
                14.0,
            ).current_a
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
        voltage_kv,
    ):
        self.sputter_kv = voltage_kv

    def set_extraction_voltage(
        self,
        voltage_kv,
    ):
        self.extraction_kv = voltage_kv
        self._check_pair()

    def set_einzel_voltage(
        self,
        voltage_kv,
    ):
        self.einzel_kv = voltage_kv
        self._check_pair()

    def set_magnet_current(
        self,
        current_a,
    ):
        self.magnet_a = current_a

    def read_cup1_current(self):
        predicted = (
            calculate_magnet_setpoint(
                self.mass_u,
                self.sputter_kv,
                self.extraction_kv,
            ).current_a
        )

        magnet_peak = predicted + 0.20
        einzel_peak = (
            self.extraction_kv + 0.50
        )

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
                self.einzel_kv - einzel_peak
            ) ** 2
            - 30.0
            * (
                self.magnet_a - magnet_peak
            ) ** 2
        )

        return -max(0.0, score)


MAGNET_SCAN = LocalScanConfig(
    coarse_radius=0.6,
    coarse_step=0.2,
    fine_radius=0.2,
    fine_step=0.05,
)

EINZEL_SCAN = LocalScanConfig(
    coarse_radius=2.0,
    coarse_step=0.5,
    fine_radius=0.5,
    fine_step=0.1,
)

EXTRACTION_SCAN = LocalScanConfig(
    coarse_radius=4.0,
    coarse_step=1.0,
    fine_radius=0.5,
    fine_step=0.25,
)

SPUTTER_SCAN = LocalScanConfig(
    coarse_radius=3.0,
    coarse_step=1.0,
    fine_radius=0.5,
    fine_step=0.25,
)


def test_complete_cycle_finds_coupled_optimum():
    hardware = CoupledFakeHardware()

    result = run_optimization_cycle(
        hardware,
        mass_u=27.0,
        operating_point=OperatingPoint(
            4.0,
            14.0,
            14.0,
        ),
        magnet_lower_a=0.0,
        magnet_upper_a=50.0,
        magnet_scan=MAGNET_SCAN,
        einzel_scan=EINZEL_SCAN,
        extraction_scan=EXTRACTION_SCAN,
        sputter_scan=SPUTTER_SCAN,
    )

    assert (
        result.final_operating_point.sputter_kv
        == pytest.approx(6.0)
    )

    assert (
        result.final_operating_point.extraction_kv
        == pytest.approx(17.0)
    )

    assert (
        result.final_operating_point.einzel_kv
        == pytest.approx(17.5)
    )

    predicted = calculate_magnet_setpoint(
        27.0,
        6.0,
        17.0,
    ).current_a

    assert hardware.magnet_a == pytest.approx(
        predicted + 0.20
    )

    assert result.final_cup1_score == pytest.approx(
        100.0
    )


def test_complete_cycle_improves_cup1_current():
    hardware = CoupledFakeHardware()

    start_score = abs(
        hardware.read_cup1_current()
    )

    result = run_optimization_cycle(
        hardware,
        mass_u=27.0,
        operating_point=OperatingPoint(
            4.0,
            14.0,
            14.0,
        ),
        magnet_lower_a=0.0,
        magnet_upper_a=50.0,
        magnet_scan=MAGNET_SCAN,
        einzel_scan=EINZEL_SCAN,
        extraction_scan=EXTRACTION_SCAN,
        sputter_scan=SPUTTER_SCAN,
    )

    assert result.final_cup1_score > start_score


def test_cycle_result_matches_hardware_final_state():
    hardware = CoupledFakeHardware()

    result = run_optimization_cycle(
        hardware,
        mass_u=27.0,
        operating_point=OperatingPoint(
            4.0,
            14.0,
            14.0,
        ),
        magnet_lower_a=0.0,
        magnet_upper_a=50.0,
        magnet_scan=MAGNET_SCAN,
        einzel_scan=EINZEL_SCAN,
        extraction_scan=EXTRACTION_SCAN,
        sputter_scan=SPUTTER_SCAN,
    )

    assert hardware.sputter_kv == pytest.approx(
        result.final_operating_point.sputter_kv
    )

    assert hardware.extraction_kv == pytest.approx(
        result.final_operating_point.extraction_kv
    )

    assert hardware.einzel_kv == pytest.approx(
        result.final_operating_point.einzel_kv
    )