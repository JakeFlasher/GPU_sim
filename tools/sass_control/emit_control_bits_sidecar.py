#!/usr/bin/env python3
"""Emit the per-instruction SASS control-bits sidecar for a kernel binary.

Plan §1E: the static control-bit sidecar is the simulator's dependence-stall
input — ``sim/sm`` reads stall counts / dependence-barrier indices / wait
masks / yield / reuse flags from this sidecar instead of simulating
scoreboards (the UPC MICRO'25 13-17% cycles-MAPE lever).

Wraps ``tools/sass_control/extractor.py`` (fail-closed parse contract) with:
- the pinned-toolkit gate satisfied explicitly via --toolkit-bindir
  (default /usr/local/cuda-13.2/bin — the tracer's pinned nvcc; the
  extractor contract pins V13.2 and the SAME bindir's cuobjdump is used so
  disassembly and version gate cannot drift apart);
- a summary block (decoded / unknown_encoding counts per function) so a
  sidecar with ANY unknown encodings is visible at a glance — downstream
  consumers must treat unknown_encoding instructions as typed-unavailable,
  never as stall=0.

Usage:
  .venv/bin/python tools/sass_control/emit_control_bits_sidecar.py \
      --binary workloads/portable_lcd/<slug>/<slug>_run \
      --arch sm_86 --out <out>.control_bits.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extractor import extract_for_binary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--arch", default="sm_86")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--toolkit-bindir", default="/usr/local/cuda-13.2/bin",
        help="bindir providing the PINNED nvcc + cuobjdump (default: the "
             "13.2 toolkit the extractor contract is verified against)",
    )
    args = ap.parse_args()

    bindir = Path(args.toolkit_bindir)
    # FAIL-CLOSED: the extractor resolves `cuobjdump` via PATH. If the pinned
    # bindir lacks it, PATH lookup would fall through to a DIFFERENT toolkit's
    # cuobjdump while the version gate still checks this bindir's nvcc —
    # exactly the disassembly/version drift the extractor contract forbids.
    for tool in ("nvcc", "cuobjdump"):
        if not (bindir / tool).is_file():
            print(
                f"pinned_toolkit_incomplete: {bindir / tool} missing — "
                f"refusing PATH fallback to another toolkit",
                file=sys.stderr,
            )
            return 2
    import os
    os.environ["PATH"] = f"{bindir}:{os.environ['PATH']}"

    result = extract_for_binary(
        Path(args.binary), arch=args.arch, nvcc=str(bindir / "nvcc"),
    )

    summary = {}
    total_decoded = total_unknown = 0
    for name, records in result["functions"].items():
        decoded = sum(1 for r in records if r["status"] == "decoded")
        unknown = len(records) - decoded
        total_decoded += decoded
        total_unknown += unknown
        summary[name] = {
            "instructions": len(records),
            "decoded": decoded,
            "unknown_encoding": unknown,
            "stall_cycles_static_sum": sum(
                r["control"]["stall"] for r in records
                if r["status"] == "decoded"
            ),
        }
    result["summary"] = {
        "functions": summary,
        "total_decoded": total_decoded,
        "total_unknown_encoding": total_unknown,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        f"control-bits sidecar: {out} — {len(result['functions'])} function(s), "
        f"{total_decoded} decoded, {total_unknown} unknown_encoding"
    )
    return 0 if total_unknown == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
