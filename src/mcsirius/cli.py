"""Command-line interface for mcsirius."""

from __future__ import annotations

import argparse
import math
from typing import Sequence

from .runtime import run_simulation
from .live_backend import probe_live_hardware
from .live_prepare import prepare_live_machine
from .peak_map import run_live_peak_map


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

    parser.add_argument(
        "--live-prepare",
        action="store_true",
        help=(
            "prepare the real machine at the deterministic "
            "source and magnet starting point without running "
            "the optimizer"
        ),
    )

    parser.add_argument(
        "--live-peak-map",
        action="store_true",
        help=(
            "scan the real analyzing magnet around the "
            "predicted mass without changing source voltages "
            "or assigning a mass label to measured peaks"
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

    if args.live_peak_map:
        try:
            report = run_live_peak_map(
                mass_u
            )
        except Exception as exc:
            print()
            print("mcsirius live peak map")
            print("======================")
            print(f"FAILED: {exc}")
            print()
            return 1

        result = report.peak_map

        print()
        print("mcsirius live peak map")
        print("======================")
        print(f"Target mass : {mass_u:g} u")
        print(
            "Model center: "
            f"{result.predicted_target_a:.4f} A"
        )
        print(
            "Source      : "
            f"{report.source_readback.sputter_kv:.3f} / "
            f"{report.source_readback.extraction_kv:.3f} / "
            f"{report.source_readback.einzel_kv:.3f} kV"
        )
        print()

        print("Predicted neighboring masses")
        print("----------------------------")

        for item in result.predicted_masses:
            print(
                f"m={item.mass_u:7.1f} u  "
                f"I={item.magnet_current_a:8.4f} A"
            )

        print()
        print("Measured map")
        print("------------")
        print(
            "Magnet [A]    Cup 1 [nA]    |Cup 1| [nA]"
        )

        for sample in result.samples:
            print(
                f"{sample.magnet_current_a:10.4f}"
                f"  {sample.cup_current_a * 1e9:11.3f}"
                f"  {sample.magnitude_a * 1e9:13.3f}"
            )

        print()
        print("Observed local maxima")
        print("---------------------")

        if not result.observed_peaks:
            print("none")
        else:
            for index, peak in enumerate(
                result.observed_peaks,
                start=1,
            ):
                print(
                    f"Peak {index:2d}: "
                    f"{peak.magnet_current_a:.4f} A  "
                    f"{peak.cup_current_a * 1e9:.3f} nA"
                )

        print()
        print(
            "No measured peak was assigned a mass label."
        )
        print(
            "Starting magnet: "
            f"{report.starting_magnet_a:.4f} A"
        )
        print(
            "Restored magnet: "
            f"{report.restored_magnet_a:.4f} A"
        )
        print()

        return 0

    if args.live_prepare:
        try:
            report = prepare_live_machine(
                mass_u
            )
        except Exception as exc:
            print()
            print("mcsirius live preparation")
            print("=========================")
            print(f"FAILED: {exc}")
            print()
            return 1

        source = (
            report.final_source_readback
        )

        print()
        print("mcsirius live preparation")
        print("=========================")
        print("MODE        : LIVE PREPARE")
        print(f"Ion mass    : {mass_u:g} u")
        print(
            "Sputter     : "
            f"{source.sputter_kv:.3f} kV"
        )
        print(
            "Extraction  : "
            f"{source.extraction_kv:.3f} kV"
        )
        print(
            "Einzel      : "
            f"{source.einzel_kv:.3f} kV"
        )
        print(
            "Magnet calc : "
            f"{report.magnet.current_a:.4f} A"
        )
        print(
            "Magnet meas : "
            f"{report.final_magnet_readback_a:.4f} A"
        )
        print(
            "Cup 1       : "
            f"{report.cup_current_a * 1e9:.3f} nA"
        )
        print()
        print(
            "Optimizer was NOT started."
        )
        print()

        return 0

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
