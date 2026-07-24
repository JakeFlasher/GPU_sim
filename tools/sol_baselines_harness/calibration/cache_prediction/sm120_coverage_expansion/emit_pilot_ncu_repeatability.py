"""Emit the sm_120 pilot NCU 3-run repeatability status
artifact.

Runs ``ncu_repeatability.live_ncu_run_fn`` against every
cell in the pilot lookup
(``harvest_tier2_pilot._PILOT_PATH_LOOKUP``), captures
per-cell 3-run L1 / L2 hit rates + per-metric mean + CV +
closed-vocabulary failure tokens, and writes the result to
``scripts/baselines_harness/calibration/cache_prediction/sm120_pilot_ncu_repeatability_v1.toml``.

CLAUDE.md Critical Rule 1: NCU is calibration-only on
sm_120; this driver MUST NOT ship in the deployed framework
path or B200 surface. The status TOML is the durable
evidence input for the next-round expanded ledger emission.

CLAUDE.md Critical Rule 2: every NCU invocation passes
``--clock-control none`` (pinned by the underlying
``ncu_harvester._build_invocation``).

CLI:

  python -m baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.emit_pilot_ncu_repeatability \\
      --captured-at-utc 2026-05-27T17:00:00+00:00 \\
      [--cells slug1,slug2] \\
      [--stub]                # use the safe stub (no NCU)

Default invocation runs the live NCU-wrapped sol-execbench
subprocess (3 times per cell); ``--stub`` swaps in the
``ncu_not_yet_run`` stub for byte-deterministic emission
testing on a host without the NCU toolchain.
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

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ncu_repeatability import (  # noqa: E402
    _stub_ncu_run_fn,
    emit_ncu_repeatability_status,
    live_ncu_run_fn,
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
    / "sm120_pilot_ncu_repeatability_v1.toml"
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
        help=(
            "ISO-8601 UTC timestamp stamped on every row + "
            "the [meta] block (controls byte-determinism of "
            "the emitted artifact)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Default: {_DEFAULT_OUTPUT}",
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
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use the safe stub run_fn instead of the live "
            "NCU-wrapped sol-execbench invocation."
        ),
    )
    args = parser.parse_args()
    refuse_legacy_ncu_calibration(
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_coverage_expansion.emit_pilot_ncu_repeatability.main"
    )

    pilot_cells: list[tuple[str, str]] = sorted(_PILOT_PATH_LOOKUP.keys())
    if args.cells:
        wanted = set(args.cells.split(","))
        pilot_cells = [(s, l) for (s, l) in pilot_cells if s in wanted]
    if not pilot_cells:
        print("no cells selected; nothing to do", file=sys.stderr)
        return 1

    if args.stub:
        run_fn = _stub_ncu_run_fn
        print(
            f"Stub mode: emitting status for {len(pilot_cells)} "
            f"cells (no NCU invocation)",
            file=sys.stderr,
        )
    else:
        def run_fn(*, slug, language):  # noqa: ANN001
            print(
                f"  live ncu: {slug} ({language}) 3 runs",
                file=sys.stderr,
                flush=True,
            )
            outcome = live_ncu_run_fn(slug=slug, language=language)
            print(
                f"    -> completed={outcome.completed} "
                f"failure_reason={outcome.failure_reason!r}",
                file=sys.stderr,
                flush=True,
            )
            return outcome

    out = emit_ncu_repeatability_status(
        pilot_cells,
        output=args.output,
        captured_at_utc=args.captured_at_utc,
        run_fn=run_fn,
    )
    print(f"emitted {out} ({len(pilot_cells)} cells)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
