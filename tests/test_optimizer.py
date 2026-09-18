import pytest

from mcsirius.machine import OperatingPoint
from mcsirius.magnet import calculate_magnet_setpoint
from mcsirius.optimizer import (
    optimize_magnet_and_einzel,
)
from mcsirius.scan import LocalScanConfig


class FakeCup1Hardware:
    def __init__(
        self,
        *,
        magnet_a: float,
        einzel_kv: float,
        magnet_peak: float,
        einzel_peak: float,
    ):
        self.magnet_a = magnet_a
        self.einzel_kv = einzel_kv
        self.magnet_peak = magnet_peak
        self.einzel_peak = einzel_peak
        self.writes = []

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        self.magnet_a = current_a
        self.writes.append(
            ("magnet", current_a)
        )

    def set_einzel_voltage(
        self,
        voltage_kv: float,
    ) -> None:
        self.einzel_kv = voltage_kv
        self.writes.append(
            ("einzel", voltage_kv)
        )

    def read_cup1_current(self) -> float:
        score = (
            100.0
            - 8.0
            * (
                self.magnet_a
                - self.magnet_peak
            )
            ** 2
            - 5.0
            * (
                self.einzel_kv
                - self.einzel_peak
            )
            ** 2
        )

        # Simulate the sign normally seen for negative ions.
        return -score


def test_chain_uses_magnet_model_then_focuses_einzel():
    point = OperatingPoint(
        4.0,
        14.0,
        14.0,
    )

    seed = calculate_magnet_setpoint(
        27.0,
        4.0,
        14.0,
    ).current_a

    hardware = FakeCup1Hardware(
        magnet_a=seed,
        einzel_kv=14.0,
        magnet_peak=seed + 0.3,
        einzel_peak=15.2,
    )

    result = optimize_magnet_and_einzel(
        hardware,
        mass_u=27.0,
        operating_point=point,
        magnet_lower_a=10.0,
        magnet_upper_a=30.0,
        magnet_scan=LocalScanConfig(
            1.0,
            0.25,
            0.3,
            0.05,
        ),
        einzel_scan=LocalScanConfig(
            2.0,
            0.5,
            0.5,
            0.1,
        ),
    )

    assert (
        result.magnet_seed.current_a
        == pytest.approx(seed)
    )

    assert (
        result.magnet_scan.best_position
        == pytest.approx(seed + 0.3)
    )

    assert (
        result.einzel_scan.best_position
        == pytest.approx(15.2)
    )

    assert (
        hardware.magnet_a
        == pytest.approx(
            result.magnet_scan.best_position
        )
    )

    assert (
        hardware.einzel_kv
        == pytest.approx(
            result.einzel_scan.best_position
        )
    )

    assert (
        result.final_cup1_score
        == pytest.approx(100.0)
    )


def test_einzel_search_cannot_leave_extraction_window():
    point = OperatingPoint(
        4.0,
        14.0,
        14.0,
    )

    seed = calculate_magnet_setpoint(
        27.0,
        4.0,
        14.0,
    ).current_a

    hardware = FakeCup1Hardware(
        magnet_a=seed,
        einzel_kv=14.0,
        magnet_peak=seed,
        einzel_peak=17.0,
    )

    result = optimize_magnet_and_einzel(
        hardware,
        mass_u=27.0,
        operating_point=point,
        magnet_lower_a=10.0,
        magnet_upper_a=30.0,
        magnet_scan=LocalScanConfig(
            0.5,
            0.25,
            0.2,
            0.05,
        ),
        einzel_scan=LocalScanConfig(
            3.0,
            0.5,
            0.5,
            0.1,
        ),
    )

    positions = [
        sample.position
        for sample
        in result.einzel_scan.samples
    ]

    assert min(positions) >= 14.0
    assert max(positions) <= 16.0

    assert (
        result.einzel_scan.best_position
        == pytest.approx(16.0)
    )


def test_invalid_operating_point_is_rejected_before_write():
    hardware = FakeCup1Hardware(
        magnet_a=0.0,
        einzel_kv=0.0,
        magnet_peak=0.0,
        einzel_peak=0.0,
    )

    with pytest.raises(ValueError):
        optimize_magnet_and_einzel(
            hardware,
            mass_u=27.0,
            operating_point=OperatingPoint(
                3.0,
                14.0,
                14.0,
            ),
            magnet_lower_a=0.0,
            magnet_upper_a=40.0,
            magnet_scan=LocalScanConfig(
                1.0,
                0.5,
                0.2,
                0.1,
            ),
            einzel_scan=LocalScanConfig(
                1.0,
                0.5,
                0.2,
                0.1,
            ),
        )

    assert hardware.writes == []


def test_magnet_seed_outside_bounds_rejected_before_write():
    hardware = FakeCup1Hardware(
        magnet_a=0.0,
        einzel_kv=14.0,
        magnet_peak=0.0,
        einzel_peak=14.0,
    )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        optimize_magnet_and_einzel(
            hardware,
            mass_u=27.0,
            operating_point=OperatingPoint(
                4.0,
                14.0,
                14.0,
            ),
            magnet_lower_a=0.0,
            magnet_upper_a=10.0,
            magnet_scan=LocalScanConfig(
                1.0,
                0.5,
                0.2,
                0.1,
            ),
            einzel_scan=LocalScanConfig(
                1.0,
                0.5,
                0.2,
                0.1,
            ),
        )

    assert hardware.writes == []