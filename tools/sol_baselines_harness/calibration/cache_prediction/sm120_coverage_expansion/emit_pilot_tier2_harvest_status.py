"""Emit the sm_120 pilot Tier-2 NVBit harvest status artifact.

Runs ``live_tier2_harvest_fn`` against every cell in
``harvest_tier2_pilot._PILOT_PATH_LOOKUP`` (the 12-cell pilot
cohort), captures per-cell completion + per-cell sidecar
evidence + closed-vocabulary failure tokens, and writes the
result to
``tools/sol_baselines_harness/calibration/cache_prediction/sm120_pilot_tier2_harvest_status_v1.toml``.

Idempotent via resume semantics: a cell that already has a
non-empty parseable sidecar at
``<slug>__<lang>/nvbit_metrics/tier2_address_trace.t2a`` under
``.harness_traces/`` is skipped (the existing sidecar is parsed
for evidence; no re-harvest).

CLI:

  python -m baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.emit_pilot_tier2_harvest_status \\
      --captured-at-utc 2026-05-27T16:00:00+00:00 \\
      [--cells slug1,slug2] \\
      [--stub]                # use the safe stub (no NVBit)
      [--per-cell-timeout-s 120]

Default invocation runs the live NVBit-injected sol-execbench
subprocess; ``--stub`` swaps in the harvest-not-yet-wired stub
for byte-deterministic emission testing on a host without the
toolchain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[5]

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (  # noqa: E402
    _PILOT_PATH_LOOKUP,
    _stub_harvest_fn,
    emit_harvest_status,
    live_tier2_harvest_fn,
)


_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_pilot_tier2_harvest_status_v1.toml"
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
            "ISO-8601 UTC timestamp stamped on every row + the "
            "[meta] block (controls byte-determinism of the "
            "emitted artifact)."
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
            "Use the safe stub harvest_fn instead of the live "
            "NVBit-injected sol-execbench invocation."
        ),
    )
    parser.add_argument(
        "--per-cell-timeout-s",
        type=int,
        default=120,
        help=(
            "Per-cell subprocess timeout (seconds). Live "
            "harvester uses 2x this as the overall subprocess.run "
            "wall-time cap."
        ),
    )
    args = parser.parse_args()

    pilot_cells: list[tuple[str, str]] = sorted(_PILOT_PATH_LOOKUP.keys())
    if args.cells:
        wanted = set(args.cells.split(","))
        pilot_cells = [(s, l) for (s, l) in pilot_cells if s in wanted]
    if not pilot_cells:
        print("no cells selected; nothing to do", file=sys.stderr)
        return 1

    if args.stub:
        harvest_fn = _stub_harvest_fn
        print(
            f"Stub mode: emitting status for {len(pilot_cells)} "
            f"cells (no NVBit invocation)",
            file=sys.stderr,
        )
    else:
        def harvest_fn(*, slug, language, sidecar_path):  # noqa: ANN001
            print(
                f"  live harvest: {slug} ({language}) "
                f"timeout={args.per_cell_timeout_s}s",
                file=sys.stderr,
                flush=True,
            )
            outcome = live_tier2_harvest_fn(
                slug=slug,
                language=language,
                sidecar_path=sidecar_path,
                timeout_s=args.per_cell_timeout_s,
            )
            print(
                f"    -> completed={outcome.completed} "
                f"failure_reason={outcome.failure_reason!r}",
                file=sys.stderr,
                flush=True,
            )
            return outcome

    out = emit_harvest_status(
        pilot_cells,
        output=args.output,
        captured_at_utc=args.captured_at_utc,
        harvest_fn=harvest_fn,
    )
    print(f"emitted {out} ({len(pilot_cells)} cells)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
