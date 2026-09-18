from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    sputter_kv: float
    extraction_kv: float
    einzel_kv: float


@dataclass(frozen=True)
class MachineLimits:
    sputter_min_kv: float = 4.0
    sputter_max_kv: float = 9.0

    extraction_min_kv: float = 14.0
    extraction_max_kv: float = 25.0

    einzel_min_kv: float = 14.0
    einzel_max_kv: float = 25.0

    max_einzel_delta_kv: float = 2.0

    def einzel_window(self, extraction_kv: float) -> tuple[float, float]:
        if not self.extraction_min_kv <= extraction_kv <= self.extraction_max_kv:
            raise ValueError(
                f"Extraction voltage {extraction_kv:.3f} kV is outside "
                f"{self.extraction_min_kv:.3f}..{self.extraction_max_kv:.3f} kV"
            )

        lower = max(
            self.einzel_min_kv,
            extraction_kv - self.max_einzel_delta_kv,
        )
        upper = min(
            self.einzel_max_kv,
            extraction_kv + self.max_einzel_delta_kv,
        )

        return lower, upper

    def validate(self, point: OperatingPoint) -> None:
        errors: list[str] = []

        if not self.sputter_min_kv <= point.sputter_kv <= self.sputter_max_kv:
            errors.append(
                f"sputter {point.sputter_kv:.3f} kV outside "
                f"{self.sputter_min_kv:.3f}..{self.sputter_max_kv:.3f} kV"
            )

        if not self.extraction_min_kv <= point.extraction_kv <= self.extraction_max_kv:
            errors.append(
                f"extraction {point.extraction_kv:.3f} kV outside "
                f"{self.extraction_min_kv:.3f}..{self.extraction_max_kv:.3f} kV"
            )

        if not self.einzel_min_kv <= point.einzel_kv <= self.einzel_max_kv:
            errors.append(
                f"einzel {point.einzel_kv:.3f} kV outside "
                f"{self.einzel_min_kv:.3f}..{self.einzel_max_kv:.3f} kV"
            )

        delta = abs(point.einzel_kv - point.extraction_kv)

        if delta > self.max_einzel_delta_kv:
            errors.append(
                f"einzel/extraction difference {delta:.3f} kV exceeds "
                f"{self.max_einzel_delta_kv:.3f} kV"
            )

        if errors:
            raise ValueError("; ".join(errors))


DEFAULT_LIMITS = MachineLimits()
DEFAULT_OPERATING_POINT = OperatingPoint(
    sputter_kv=4.0,
    extraction_kv=14.0,
    einzel_kv=14.0,
)
