"""Emit the sm_120 pilot predictor-predictions artifact.

Runs ``drive_pilot_predictor_cohort`` against every cell
in the pilot lookup (``harvest_tier2_pilot._PILOT_PATH_LOOKUP``)
and writes
``tools/sol_baselines_harness/calibration/cache_prediction/sm120_pilot_predictor_predictions_v1.toml``.

CLI:

  python tools/sol_baselines_harness/calibration/cache_prediction/sm120_coverage_expansion/emit_pilot_predictor_predictions.py \\
      --captured-at-utc 2026-05-27T22:00:00+00:00 \\
      [--cells slug1,slug2]
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
)

import argparse

_REPO_ROOT = Path(__file__).resolve().parents[5]

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.drive_pilot_predictor import (  # noqa: E402
    emit_predictor_predictions,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (  # noqa: E402
    _PILOT_PATH_LOOKUP,
)


_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_pilot_predictor_predictions_v1.toml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset of pilot slugs. "
            "Default: all 12 pilot cells."
        ),
    )
    args = parser.parse_args()
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.emit_pilot_predictor_predictions.main')

    pilot_cells: list[tuple[str, str]] = sorted(
        _PILOT_PATH_LOOKUP.keys()
    )
    if args.cells:
        wanted = set(args.cells.split(","))
        pilot_cells = [
            (s, l) for (s, l) in pilot_cells if s in wanted
        ]
    if not pilot_cells:
        print("no cells selected", file=sys.stderr)
        return 1
    print(
        f"driving predictor over {len(pilot_cells)} pilot cells",
        file=sys.stderr,
    )
    out = emit_predictor_predictions(
        pilot_cells,
        output=args.output,
        captured_at_utc=args.captured_at_utc,
    )
    print(
        f"emitted {out} ({len(pilot_cells)} cells)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
