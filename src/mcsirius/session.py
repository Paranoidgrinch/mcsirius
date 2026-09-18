"""Repeated optimization cycles with a simple convergence rule."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

from .cycle import run_optimization_cycle
from .machine import (
    DEFAULT_LIMITS,
    MachineLimits,
    OperatingPoint,
)
from .scan import ScanConfig
from .source import SourceHardware


class CycleOutcome(Protocol):
    final_operating_point: OperatingPoint
    final_cup1_score: float


CycleRunner = Callable[..., CycleOutcome]


@dataclass(frozen=True)
class ConvergenceConfig:
    """
    Stop after an optimization cycle produces no meaningful
    relative improvement.

    max_cycles limits the total machine work even when the
    convergence criterion is never reached.
    """

    max_cycles: int = 3
    min_relative_improvement: float = 0.01

    def validate(self) -> None:
        if (
            type(self.max_cycles) is not int
            or self.max_cycles < 1
        ):
            raise ValueError(
                "Maximum cycle count must be "
                "a positive integer."
            )

        if (
            not math.isfinite(
                self.min_relative_improvement
            )
            or self.min_relative_improvement < 0.0
        ):
            raise ValueError(
                "Minimum relative improvement must "
                "be finite and non-negative."
            )


@dataclass(frozen=True)
class CycleSummary:
    cycle_number: int
    start_operating_point: OperatingPoint
    final_operating_point: OperatingPoint
    final_cup1_score: float
    relative_improvement: float | None


@dataclass(frozen=True)
class OptimizationSessionResult:
    cycles: tuple[CycleSummary, ...]
    final_operating_point: OperatingPoint
    final_cup1_score: float
    converged: bool


def _relative_improvement(
    previous: float,
    current: float,
) -> float:
    if previous == 0.0:
        if current > 0.0:
            return math.inf

        return 0.0

    return (
        current - previous
    ) / abs(previous)


def run_until_converged(
    hardware: SourceHardware,
    *,
    mass_u: float,
    operating_point: OperatingPoint,
    magnet_lower_a: float,
    magnet_upper_a: float,
    magnet_scan: ScanConfig,
    einzel_scan: ScanConfig,
    extraction_scan: ScanConfig,
    sputter_scan: ScanConfig,
    convergence: ConvergenceConfig = ConvergenceConfig(),
    limits: MachineLimits = DEFAULT_LIMITS,
    voltage_pair_step_kv: float = 0.5,
    cycle_runner: CycleRunner = run_optimization_cycle,
) -> OptimizationSessionResult:
    """
    Repeat complete optimization cycles until the Cup-1 score
    stops improving significantly or max_cycles is reached.
    """

    convergence.validate()
    limits.validate(operating_point)

    current_point = operating_point

    previous_score: float | None = None
    summaries: list[CycleSummary] = []

    converged = False

    for cycle_number in range(
        1,
        convergence.max_cycles + 1,
    ):
        start_point = current_point

        result = cycle_runner(
            hardware,
            mass_u=mass_u,
            operating_point=start_point,
            magnet_lower_a=magnet_lower_a,
            magnet_upper_a=magnet_upper_a,
            magnet_scan=magnet_scan,
            einzel_scan=einzel_scan,
            extraction_scan=extraction_scan,
            sputter_scan=sputter_scan,
            limits=limits,
            voltage_pair_step_kv=(
                voltage_pair_step_kv
            ),
        )

        score = float(
            result.final_cup1_score
        )

        if (
            not math.isfinite(score)
            or score < 0.0
        ):
            raise ValueError(
                "Optimization cycle returned "
                "an invalid Cup-1 score."
            )

        improvement: float | None = None

        if previous_score is not None:
            improvement = _relative_improvement(
                previous_score,
                score,
            )

        current_point = (
            result.final_operating_point
        )

        limits.validate(current_point)

        summaries.append(
            CycleSummary(
                cycle_number=cycle_number,
                start_operating_point=start_point,
                final_operating_point=current_point,
                final_cup1_score=score,
                relative_improvement=improvement,
            )
        )

        if (
            improvement is not None
            and improvement
            <= convergence.min_relative_improvement
        ):
            converged = True
            break

        previous_score = score

    final = summaries[-1]

    return OptimizationSessionResult(
        cycles=tuple(summaries),
        final_operating_point=(
            final.final_operating_point
        ),
        final_cup1_score=(
            final.final_cup1_score
        ),
        converged=converged,
    )


def run_single_pass(
    hardware: SourceHardware,
    *,
    mass_u: float,
    operating_point: OperatingPoint,
    magnet_lower_a: float,
    magnet_upper_a: float,
    magnet_scan: ScanConfig,
    einzel_scan: ScanConfig,
    extraction_scan: ScanConfig,
    sputter_scan: ScanConfig,
    limits: MachineLimits = DEFAULT_LIMITS,
    voltage_pair_step_kv: float = 0.5,
) -> OptimizationSessionResult:
    """
    Run one complete coordinate-optimization pass.

    Each one-dimensional adaptive search itself converges to
    its configured minimum step, so the fast runtime does not
    automatically repeat the entire expensive machine cycle.
    """

    limits.validate(operating_point)

    result = run_optimization_cycle(
        hardware,
        mass_u=mass_u,
        operating_point=operating_point,
        magnet_lower_a=magnet_lower_a,
        magnet_upper_a=magnet_upper_a,
        magnet_scan=magnet_scan,
        einzel_scan=einzel_scan,
        extraction_scan=extraction_scan,
        sputter_scan=sputter_scan,
        limits=limits,
        voltage_pair_step_kv=voltage_pair_step_kv,
    )

    score = float(result.final_cup1_score)

    if (
        not math.isfinite(score)
        or score < 0.0
    ):
        raise ValueError(
            "Optimization cycle returned "
            "an invalid Cup-1 score."
        )

    summary = CycleSummary(
        cycle_number=1,
        start_operating_point=operating_point,
        final_operating_point=(
            result.final_operating_point
        ),
        final_cup1_score=score,
        relative_improvement=None,
    )

    return OptimizationSessionResult(
        cycles=(summary,),
        final_operating_point=(
            result.final_operating_point
        ),
        final_cup1_score=score,
        converged=True,
    )
