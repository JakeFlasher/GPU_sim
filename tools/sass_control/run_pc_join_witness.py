#!/usr/bin/env python3
"""Live PC/vpc join witness: traced NVBit instruction PCs joined to the
static SASS control-bit extraction, per corpus cell.

For every closed-identity cell (trace sidecar binary == admitted label
pin), the witness:

1. extracts the per-function control records from the CELL BINARY
   (arch-scoped cuobjdump parse under the pinned parser contract);
2. walks the cell's first-launch threadblock records and collects the
   distinct traced (function_unique_id, pc) pairs, verifying the wire
   contract's vpc consistency (low 32 bits of vpc == pc) where vpc is
   present;
3. joins each traced pc into the extracted function's record map
   (kernel-name match, ambiguity fail-closed) and reports coverage:
   every traced pc must resolve to a DECODED control record.

The witness artifact carries per-cell coverage plus sample decoded
records; cells that cannot join (identity, trace, or ambiguity) are
typed, never skipped silently.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
_LL = REPO_ROOT / "tools" / "local_labels"
if str(_LL) not in sys.path:
    sys.path.insert(0, str(_LL))

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)

from extractor import extract_for_binary  # noqa: E402

DEFAULT_WITNESS_OUT = "data/sass_control/pc_join_witness_sm_86.json"


def traced_pcs_for_cell(trace_dir: Path) -> dict:
    """Distinct (function_unique_id -> {pc}) from the FIRST launch's
    threadblocks, with the vpc low-32 consistency check."""
    threadblock_pb2 = load_dynamic_trace_bindings().threadblock_pb2

    tb_root = trace_dir / "threadblocks"

    def _num(p: Path, prefix: str) -> int:
        try:
            return int(p.name.removeprefix(prefix))
        except ValueError:
            return 1 << 30

    devices = sorted(tb_root.glob("device_*"),
                     key=lambda p: _num(p, "device_"))
    streams = sorted(devices[0].glob("stream_*"),
                     key=lambda p: _num(p, "stream_"))
    kernels = sorted(streams[0].glob("kernel_*"),
                     key=lambda p: _num(p, "kernel_"))
    first = kernels[0]
    pcs_by_function: dict[int, set[int]] = {}
    vpc_mismatches = 0
    n_records = 0
    for pb_path in sorted(first.rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(pb_path.read_bytes())
        for warp_index in sorted(tb.warps.keys()):
            for inst in tb.warps[warp_index].instructions:
                n_records += 1
                fid = inst.function_unique_id
                pcs_by_function.setdefault(fid, set()).add(inst.pc)
                if inst.HasField("vpc") and (
                        inst.vpc & 0xFFFFFFFF) != inst.pc:
                    vpc_mismatches += 1
    return {
        "pcs_by_function": pcs_by_function,
        "vpc_low32_mismatches": vpc_mismatches,
        "n_instruction_records": n_records,
        "first_launch": first.name,
    }


def kernel_names_by_function_id(trace_dir: Path) -> dict[int, str]:
    """function_unique_id -> kernel name from the top-level Trace."""
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2

    trace = trace_pb2.Trace()
    trace.ParseFromString((trace_dir / "dynamic_trace.pb").read_bytes())
    out: dict[int, str] = {}
    for device in trace.gpu_device.values():
        for stream in device.streams.values():
            for kernel in stream.kernels:
                out[kernel.function_unique_id] = kernel.name
    return out


def join_cell(cell: dict, trace_dir: Path, binary: Path) -> dict:
    extraction = extract_for_binary(binary)
    traced = traced_pcs_for_cell(trace_dir)
    names = kernel_names_by_function_id(trace_dir)
    per_function = []
    total_traced = total_joined = total_decoded = 0
    for fid, pcs in sorted(traced["pcs_by_function"].items()):
        kernel_name = names.get(fid, "")
        # The tracer appends a ___<n> uniquifier to kernel names; the
        # SASS function symbol carries the plain (possibly mangled)
        # name. Strip the suffix before matching.
        base_name = re.sub(r"___\d+$", "", kernel_name)
        matches = [
            (fname, recs)
            for fname, recs in extraction["functions"].items()
            if base_name and base_name in fname
        ]
        if len(matches) != 1:
            per_function.append({
                "function_unique_id": fid,
                "kernel_name": kernel_name,
                "status": "typed",
                "typed_reason": (
                    "kernel_sass_not_in_cell_binary" if not matches
                    else "kernel_function_match_ambiguous"),
                "detail": (
                    "vendor/library kernels are linked at runtime; "
                    "their SASS is outside the cell binary's static "
                    "extraction surface" if not matches else
                    f"{len(matches)} same-name SASS functions"),
                "n_matches": len(matches),
            })
            total_traced += len(pcs)
            continue
        fname, recs = matches[0]
        by_pc = {r["pc"]: r for r in recs}
        joined = decoded = 0
        missing = []
        for pc in sorted(pcs):
            rec = by_pc.get(pc)
            if rec is None:
                missing.append(pc)
                continue
            joined += 1
            if rec["status"] == "decoded":
                decoded += 1
        sample_pc = min(pcs)
        sample = by_pc.get(sample_pc)
        per_function.append({
            "function_unique_id": fid,
            "kernel_name": kernel_name,
            "sass_function": fname,
            "status": "joined",
            "n_traced_pcs": len(pcs),
            "n_joined": joined,
            "n_decoded": decoded,
            "missing_pcs": missing[:8],
            "sample": {
                "pc": sample_pc,
                "opcode_text": sample["opcode_text"][:60],
                "control": sample.get("control"),
            } if sample else None,
        })
        total_traced += len(pcs)
        total_joined += joined
        total_decoded += decoded
    return {
        "identity": {
            "slug": cell["slug"],
            "shape_point": cell["shape_point"],
            **({"variant_id": cell["variant_id"]}
               if cell.get("variant_id") else {}),
            "binary_sha256": cell["binary_sha256"],
            "kernel_signature_sha256": cell["kernel_signature_sha256"],
        },
        "first_launch": traced["first_launch"],
        "vpc_low32_mismatches": traced["vpc_low32_mismatches"],
        "n_instruction_records": traced["n_instruction_records"],
        "functions": per_function,
        "coverage": {
            "n_traced_pcs": total_traced,
            "n_joined": total_joined,
            "n_decoded": total_decoded,
            "full_join": total_traced == total_joined == total_decoded
            and total_traced > 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emissions",
        default="/dev/shm/sm120_rtx_5090_work/replay_count_emissions.json")
    parser.add_argument("--out", default=DEFAULT_WITNESS_OUT)
    args = parser.parse_args(argv)

    from replay_count_producer import (
        _binary_for_cell, _trace_dir_for_cell)

    emissions = json.loads(
        (REPO_ROOT / args.emissions).read_text(encoding="utf-8"))
    cells = []
    n_full = 0
    for record in emissions["cells"]:
        if record["cell_status"].get("status") != "emitted":
            continue
        ident = record["identity"]
        trace_dir = _trace_dir_for_cell(ident)
        binary = _binary_for_cell(ident)
        if trace_dir is None or binary is None:
            cells.append({
                "identity": ident,
                "status": "typed",
                "typed_reason": "binary_or_trace_unresolved",
            })
            continue
        result = join_cell(ident, trace_dir, binary)
        cells.append(result)
        if result["coverage"]["full_join"]:
            n_full += 1
        print(f"[sass_join] {ident['slug']}@{ident['shape_point']}"
              f"{('@' + ident['variant_id']) if ident.get('variant_id') else ''}: "
              f"{result['coverage']['n_joined']}/"
              f"{result['coverage']['n_traced_pcs']} joined, "
              f"full={result['coverage']['full_join']}", flush=True)

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=60).stdout.strip()
    artifact = {
        "schema_version": 1,
        "artifact_kind": "sass_control_pc_join_witness",
        "arch": "sm_86",
        "source_commit": head,
        "n_cells": len(cells),
        "n_full_join": n_full,
        "cells": cells,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[sass_join] wrote {args.out}: {n_full}/{len(cells)} cells "
          f"fully joined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
