"""Command-line interface for mcsirius."""

from __future__ import annotations

import argparse
import math
from typing import Sequence

from .runtime import run_simulation
from .live_backend import probe_live_hardware


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcsirius",
        description=(
            "Optimize the FLAVIA source-to-Cup-1 section "
            "for a selected ion mass."
        ),
    )

    parser.add_argument(
        "mass",
        type=float,
        help="ion mass in atomic mass units (u)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "explicitly run the simulated hardware path "
            "(currently also the default)"
        ),
    )

    parser.add_argument(
        "--live-check",
        action="store_true",
        help=(
            "connect to source hardware and print "
            "readbacks without changing source, magnet "
            "or Cup selection"
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    mass_u = float(args.mass)

    if (
        not math.isfinite(mass_u)
        or mass_u <= 0.0
    ):
        print(
            "error: ion mass must be greater than zero"
        )
        return 2

    if args.live_check:
        try:
            report = probe_live_hardware()
        except Exception as exc:
            print()
            print("mcsirius live hardware check")
            print("============================")
            print(f"FAILED: {exc}")
            print()
            return 1

        print()
        print("mcsirius live hardware check")
        print("============================")
        print(f"Sputter     : {report.sputter_kv:.3f} kV")
        print(f"Extraction  : {report.extraction_kv:.3f} kV")
        print(f"Einzel      : {report.einzel_kv:.3f} kV")
        print(
            "Magnet      : "
            f"{report.magnet_current_a:.4f} A"
        )
        print(
            "Selected Cup: "
            f"{report.selected_cup}"
        )
        print(
            "Keithley    : "
            f"{report.keithley_current_a * 1e9:.3f} nA"
        )
        print()
        print(
            "No source-voltage, magnet-current or "
            "Cup-selection command was sent."
        )
        print()

        return 0

    # During commissioning the normal CLI deliberately defaults
    # to simulation.
    result = run_simulation(mass_u)

    point = (
        result.session.final_operating_point
    )

    print()
    print("mcsirius source optimizer")
    print("========================")
    print("MODE        : DRY RUN")
    print(f"Ion mass    : {mass_u:g} u")
    print(
        "Startup     : "
        f"{result.startup.ramp.duration_s:.2f} s"
    )
    print(
        "Sputter     : "
        f"{point.sputter_kv:.3f} kV"
    )
    print(
        "Extraction  : "
        f"{point.extraction_kv:.3f} kV"
    )
    print(
        "Einzel      : "
        f"{point.einzel_kv:.3f} kV"
    )
    print(
        "Cup 1       : "
        f"{result.session.final_cup1_score * 1e9:.3f} nA"
    )
    print(
        "Cycles      : "
        f"{len(result.session.cycles)}"
    )
    print(
        "Converged   : "
        f"{'yes' if result.session.converged else 'no'}"
    )
    print(
        "Measurements: "
        f"{result.measurement_count}"
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
