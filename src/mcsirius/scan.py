"""Deterministic one-dimensional coarse-to-fine maximization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal


Phase = Literal["coarse", "fine"]


@dataclass(frozen=True)
class LocalScanConfig:
    coarse_radius: float
    coarse_step: float
    fine_radius: float
    fine_step: float

    def validate(self) -> None:
        values = (
            self.coarse_radius,
            self.coarse_step,
            self.fine_radius,
            self.fine_step,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Scan parameters must be finite.")

        if self.coarse_radius <= 0.0 or self.fine_radius <= 0.0:
            raise ValueError("Scan radii must be greater than zero.")

        if self.coarse_step <= 0.0 or self.fine_step <= 0.0:
            raise ValueError("Scan steps must be greater than zero.")

        if self.fine_step >= self.coarse_step:
            raise ValueError("Fine step must be smaller than coarse step.")


@dataclass(frozen=True)
class ScanSample:
    position: float
    score: float
    phase: Phase


@dataclass(frozen=True)
class ScanResult:
    start_position: float
    coarse_best_position: float
    best_position: float
    best_score: float
    samples: tuple[ScanSample, ...]


def _scan_points(
    *,
    center: float,
    radius: float,
    step: float,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    local_lower = max(lower, center - radius)
    local_upper = min(upper, center + radius)

    count = math.ceil(radius / step)

    candidates = [
        local_lower,
        center,
        local_upper,
    ]

    candidates.extend(
        center + index * step
        for index in range(-count, count + 1)
    )

    clipped = [
        min(local_upper, max(local_lower, value))
        for value in candidates
        if lower <= value <= upper
    ]

    unique: dict[float, float] = {}

    for value in clipped:
        key = round(value, 12)
        unique[key] = value

    return tuple(sorted(unique.values()))


def _measure_phase(
    measure: Callable[[float], float],
    points: tuple[float, ...],
    phase: Phase,
) -> tuple[ScanSample, ...]:
    samples: list[ScanSample] = []

    for position in points:
        score = float(measure(position))

        if not math.isfinite(score):
            raise ValueError(
                f"Measurement returned a non-finite score "
                f"at {position:.12g}."
            )

        samples.append(
            ScanSample(
                position=position,
                score=score,
                phase=phase,
            )
        )

    return tuple(samples)


def _best_sample(
    samples: tuple[ScanSample, ...],
    center: float,
) -> ScanSample:
    return max(
        samples,
        key=lambda sample: (
            sample.score,
            -abs(sample.position - center),
            -sample.position,
        ),
    )


def maximize_1d(
    measure: Callable[[float], float],
    *,
    start: float,
    lower: float,
    upper: float,
    config: LocalScanConfig,
) -> ScanResult:
    """
    Maximize a scalar score with a bounded coarse scan
    followed by a fine scan.
    """

    start = float(start)
    lower = float(lower)
    upper = float(upper)

    if not all(
        math.isfinite(value)
        for value in (start, lower, upper)
    ):
        raise ValueError("Start and bounds must be finite.")

    if lower >= upper:
        raise ValueError(
            "Lower bound must be smaller than upper bound."
        )

    if not lower <= start <= upper:
        raise ValueError(
            "Start position must lie inside the scan bounds."
        )

    config.validate()

    coarse_points = _scan_points(
        center=start,
        radius=config.coarse_radius,
        step=config.coarse_step,
        lower=lower,
        upper=upper,
    )

    coarse_samples = _measure_phase(
        measure,
        coarse_points,
        "coarse",
    )

    coarse_best = _best_sample(
        coarse_samples,
        start,
    )

    fine_points = _scan_points(
        center=coarse_best.position,
        radius=config.fine_radius,
        step=config.fine_step,
        lower=lower,
        upper=upper,
    )

    fine_samples = _measure_phase(
        measure,
        fine_points,
        "fine",
    )

    fine_best = _best_sample(
        fine_samples,
        coarse_best.position,
    )

    all_samples = coarse_samples + fine_samples

    overall_best = max(
        all_samples,
        key=lambda sample: (
            sample.score,
            -abs(sample.position - fine_best.position),
            -sample.position,
        ),
    )

    return ScanResult(
        start_position=start,
        coarse_best_position=coarse_best.position,
        best_position=overall_best.position,
        best_score=overall_best.score,
        samples=all_samples,
    )

@dataclass(frozen=True)
class AdaptiveScanConfig:
    """
    Bounded hill-climb for expensive real measurements.

    The optimizer checks both directions around the current
    point, moves uphill and reduces the step size when neither
    direction improves the score.
    """

    initial_step: float
    min_step: float
    max_evaluations: int = 7

    def validate(self) -> None:
        if (
            not math.isfinite(self.initial_step)
            or self.initial_step <= 0.0
        ):
            raise ValueError(
                "Initial adaptive step must be finite "
                "and greater than zero."
            )

        if (
            not math.isfinite(self.min_step)
            or self.min_step <= 0.0
        ):
            raise ValueError(
                "Minimum adaptive step must be finite "
                "and greater than zero."
            )

        if self.min_step >= self.initial_step:
            raise ValueError(
                "Minimum adaptive step must be smaller "
                "than the initial step."
            )

        if (
            type(self.max_evaluations) is not int
            or self.max_evaluations < 3
        ):
            raise ValueError(
                "Adaptive scan needs at least three evaluations."
            )


ScanConfig = LocalScanConfig | AdaptiveScanConfig


def maximize_adaptive_1d(
    measure: Callable[[float], float],
    *,
    start: float,
    lower: float,
    upper: float,
    config: AdaptiveScanConfig,
) -> ScanResult:
    """
    Maximize a scalar score with a small bounded measurement
    budget.

    Previously measured positions are cached, so moving through
    the search interval never measures the same point twice.
    """

    start = float(start)
    lower = float(lower)
    upper = float(upper)

    if not all(
        math.isfinite(value)
        for value in (start, lower, upper)
    ):
        raise ValueError(
            "Start and bounds must be finite."
        )

    if lower >= upper:
        raise ValueError(
            "Lower bound must be smaller than upper bound."
        )

    if not lower <= start <= upper:
        raise ValueError(
            "Start position must lie inside the scan bounds."
        )

    config.validate()

    samples: list[ScanSample] = []
    cache: dict[float, ScanSample] = {}

    def evaluate(position: float) -> ScanSample:
        position = min(
            upper,
            max(lower, float(position)),
        )

        key = round(position, 12)

        if key in cache:
            return cache[key]

        if len(samples) >= config.max_evaluations:
            raise RuntimeError(
                "Adaptive measurement budget exhausted."
            )

        score = float(measure(position))

        if not math.isfinite(score):
            raise ValueError(
                "Measurement returned a non-finite score "
                f"at {position:.12g}."
            )

        sample = ScanSample(
            position=position,
            score=score,
            phase="fine",
        )

        cache[key] = sample
        samples.append(sample)

        return sample

    current = evaluate(start)
    best = current

    step = config.initial_step

    while (
        step >= config.min_step - 1e-12
        and len(samples) < config.max_evaluations
    ):
        candidates = [current]

        for position in (
            current.position - step,
            current.position + step,
        ):
            clipped = min(
                upper,
                max(lower, position),
            )

            key = round(clipped, 12)

            if key in cache:
                candidates.append(cache[key])
                continue

            if len(samples) >= config.max_evaluations:
                break

            candidates.append(
                evaluate(clipped)
            )

        candidate = max(
            candidates,
            key=lambda sample: (
                sample.score,
                -abs(
                    sample.position
                    - current.position
                ),
            ),
        )

        if candidate.score > current.score:
            current = candidate

            if current.score > best.score:
                best = current

            continue

        step /= 2.0

    if current.score > best.score:
        best = current

    return ScanResult(
        start_position=start,
        coarse_best_position=best.position,
        best_position=best.position,
        best_score=best.score,
        samples=tuple(samples),
    )


def maximize_parameter(
    measure: Callable[[float], float],
    *,
    start: float,
    lower: float,
    upper: float,
    config: ScanConfig,
) -> ScanResult:
    """Dispatch to the requested one-dimensional optimizer."""

    if isinstance(config, AdaptiveScanConfig):
        return maximize_adaptive_1d(
            measure,
            start=start,
            lower=lower,
            upper=upper,
            config=config,
        )

    return maximize_1d(
        measure,
        start=start,
        lower=lower,
        upper=upper,
        config=config,
    )

