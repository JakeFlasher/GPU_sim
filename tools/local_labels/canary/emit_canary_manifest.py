#!/usr/bin/env python3
"""Capture the address-carrying-contract canary and emit its evidence manifest.

Reproducible end-to-end flow on the current dev host (RTX 3090 Ti):

  1. Build `sm86_addr_carry_canary.cu` for sm_86 (nvcc 13.2 toolkit include).
  2. Capture it under the freshly-built new-contract tracer (address-carrying
     contract) into a trace dir — LD_LIBRARY_PATH must carry the pinned protobuf
     lib or NVBit injection fails SILENTLY (no trace); we assert the trace exists.
  3. Validate the trace with `import_trace_contract.validate_trace_dir`.
  4. Read `trace_contract_id` + `memory_payload_classes`; count admitted ATOMIC
     (class 6) and ASYNC_COPY (class 8) records + the per-address `address_space`
     histogram + how many LDGSTS/ASYNC_COPY records resolved an `allocation_id`;
     assert 0 `unsupported_memory_path` exclusions for the flipped classes.
  5. Emit a manifest JSON with the pinned provenance (git head, clean-tree, tool
     pins, clock-lock state) + the canary evidence.

The canary is proven by: new `trace_contract_id` + capability advertising ATOMIC +
ASYNC_COPY, nonzero admitted atomic/async records, 0 unexpected
`unsupported_memory_path`, and the LDGSTS global source resolving an
allocation_id (the operand-ordering adjudication confirmed end-to-end on silicon).

Targets-never-features boundary: no NCU metric is read here; this is a
trace-input generator + validator.

Usage:
  tools/nvbit_tracer/pinned_build_env.sh must be sourced (or NVBIT_TRACER_* set).
  .venv/bin/python tools/local_labels/canary/emit_canary_manifest.py \
      --out results/nvbit_capture_staging/canary/canary_manifest.json \
      [--work <scratch_dir>]
Exit: 0 all canary checks pass; 1 a check failed; 2 setup/build/capture error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmissionError,
    require_trace_admission,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_write_allowed,
)

CANARY_DIR = REPO / "tools" / "local_labels" / "canary"
CANARY_CU = CANARY_DIR / "sm86_addr_carry_canary.cu"
TRACER_SO = REPO / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
PB_LIB = REPO / ".toolchain" / "pb" / "lib"
PB_GEN = REPO / "workloads" / "simbench_bw" / "_pb_generated"
NVCC = "/usr/local/cuda-13.2/bin/nvcc"
CUDA_INC = "/usr/local/cuda-13.2/include"

ATOMIC, ASYNC_COPY, GLOBAL = 6, 8, 1
CLASS_NAME = {ATOMIC: "ATOMIC", ASYNC_COPY: "ASYNC_COPY", GLOBAL: "GLOBAL"}
EXPECTED_CONTRACT_ID = "sm86_addr_carry_v1"
STAGING_ROOT = REPO / "results" / "nvbit_capture_staging"
DEFAULT_CANARY_ROOT = STAGING_ROOT / "canary"

# The two named scorability witnesses (the plan's positive-test cells): a tensor-core HMMA
# GEMM whose cp.async operand loads (LDGSTS -> ASYNC_COPY) drive the op_ld surface,
# and a global-atomic histogram whose atomics (-> ATOMIC) drive the L2 surface.
# `surface_class` is the flipped op class that was STRIPPED pre-flip (making the
# surface `na`); the witness passes when that class is admitted into the trace AND
# into the sim's L2 consuming path (cache_replay), never left in the exclusion
# sidecar.
WITNESSES = [
    {"name": "gemm_hmma_bf16", "cu": "ac4_gemm_hmma_bf16.cu",
     "surface": "op_ld", "surface_class": ASYNC_COPY},
    {"name": "ac4_global_atomic_histogram", "cu": "ac4_global_atomic_histogram.cu",
     "surface": "l2", "surface_class": ATOMIC},
]


def _assert_staging_output(path: Path | str, *, operation: str) -> Path:
    candidate = assert_write_allowed(path, repo_root=REPO, operation=operation)
    staging = STAGING_ROOT.resolve(strict=False)
    try:
        candidate.relative_to(staging)
    except ValueError as exc:
        raise ProtectedPathError(
            f"refusing to {operation} outside {staging}: {candidate}"
        ) from exc
    return candidate


def _canary_output_plan(
    *, out: Path | str | None = None, work: Path | str | None = None,
) -> dict[str, Path]:
    """Clear the complete canary output graph before any probe or build."""

    work_root = _assert_staging_output(
        work if work is not None else DEFAULT_CANARY_ROOT / "work",
        operation="select canary work root",
    )
    manifest = _assert_staging_output(
        out if out is not None else DEFAULT_CANARY_ROOT / "canary_manifest.json",
        operation="write canary manifest",
    )
    names = [CANARY_CU.stem, *(Path(w["cu"]).stem for w in WITNESSES)]
    candidates: dict[str, Path] = {
        "work": work_root,
        "out": manifest,
        "canary_binary": work_root / names[0],
        "canary_trace": work_root / "canary_trace",
    }
    for witness, stem in zip(WITNESSES, names[1:], strict=True):
        candidates[f"{witness['name']}_binary"] = work_root / stem
        candidates[f"{witness['name']}_trace"] = (
            work_root / f"{witness['name']}_trace"
        )
    return {
        name: _assert_staging_output(path, operation=f"prepare {name}")
        for name, path in candidates.items()
    }


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _git(*args) -> str:
    p = _run(["git", "-C", str(REPO), *args])
    return p.stdout.strip()


def _clock_state() -> dict:
    p = _run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,persistence_mode",
              "--format=csv,noheader"])
    return {"nvidia_smi": p.stdout.strip() if p.returncode == 0 else "unavailable"}


def build_cu(cu: Path, work: Path, *, binary: Path | None = None) -> Path:
    binary = _assert_staging_output(
        binary if binary is not None else work / cu.stem,
        operation="build canary binary",
    )
    p = _run([NVCC, "-ccbin=g++-14", "-O3", "-std=c++17",
              "-gencode", "arch=compute_86,code=sm_86", "-lineinfo",
              f"-I{CUDA_INC}", str(cu), "-o", str(binary)])
    if p.returncode != 0 or not binary.is_file():
        raise SystemExit(f"[canary] nvcc build failed for {cu.name}:\n{p.stderr}")
    return binary


def capture(binary: Path, out: Path) -> None:
    out = _assert_staging_output(out, operation="capture canary trace")
    if out.exists():
        _assert_staging_output(out, operation="remove prior canary trace")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = tracer_runtime_env(TRACER_SO, pb_lib=PB_LIB)
    env["CUDA_INJECTION64_PATH"] = str(TRACER_SO)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(out)
    p = _run([str(binary)], env=env, cwd=str(binary.parent))
    if p.returncode != 0:
        raise SystemExit(
            f"[canary] target exited {p.returncode}; stderr:\n{p.stderr}"
        )
    if not (out / "dynamic_trace.pb").is_file():
        raise SystemExit(
            f"[canary] no dynamic_trace.pb produced — NVBit injection likely "
            f"failed silently (check LD_LIBRARY_PATH / .so). stderr:\n{p.stderr}"
        )
    try:
        require_trace_admission(out)
    except TraceAdmissionError as exc:
        raise SystemExit(f"[canary] captured trace is not admissible: {exc}") from exc


def analyze(out: Path) -> dict:
    sys.path.insert(0, str(PB_GEN))
    sys.path.insert(0, str(REPO / "tools" / "nvbit_tracer"))
    import trace_pb2  # type: ignore
    import threadblock_pb2  # type: ignore
    from import_trace_contract import validate_trace_dir  # type: ignore

    v = validate_trace_dir(out)
    t = trace_pb2.Trace()
    t.ParseFromString((out / "dynamic_trace.pb").read_bytes())

    cls = Counter()
    space = Counter()
    async_with_id = 0
    for p in sorted((out / "threadblocks").rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(p.read_bytes())
        for wi in tb.warps:
            for inst in tb.warps[wi].instructions:
                if inst.HasField("memory_op_class") and inst.addresses:
                    cls[inst.memory_op_class] += 1
                    for a in inst.addresses:
                        if a.HasField("address_space"):
                            space[a.address_space] += 1
                        if (inst.memory_op_class == ASYNC_COPY
                                and a.HasField("allocation_id") and a.allocation_id):
                            async_with_id += 1
    excl = json.loads((out / "extra_info" / "unsupported_memory_path.json").read_text())
    flipped_excl = [e for e in excl if e.get("excluded_op_class") in ("ATOMIC", "ASYNC_COPY")]
    allocs = json.loads((out / "extra_info" / "allocations.json").read_text())

    return {
        "validate_trace_dir": {"ok": bool(v.ok), "typed_reason": v.typed_reason},
        "trace_contract_id": t.trace_contract_id,
        "memory_payload_classes": list(t.memory_payload_classes),
        "admitted_records_by_class": {int(k): int(n) for k, n in cls.items()},
        "admitted_atomic": cls.get(ATOMIC, 0),
        "admitted_async_copy": cls.get(ASYNC_COPY, 0),
        "admitted_global": cls.get(GLOBAL, 0),
        "address_space_histogram": {int(k): int(n) for k, n in space.items()},
        "ldgsts_source_records_with_allocation_id": async_with_id,
        "unsupported_memory_path_total": len(excl),
        "unsupported_memory_path_flipped_classes": len(flipped_excl),
        "allocations_sidecar_entries": len(allocs),
    }


def analyze_witness(out: Path, surface: str, surface_class: int) -> dict:
    """Prove a surface (op_ld / L2) went from `na` (its records stripped pre-flip)
    to scorable. The surface-relevant flipped class must be admitted into the trace
    AND into the sim's L2 consuming path (`cache_replay.decode_cell_access_stream`)
    — never left in the exclusion sidecar. Pre-flip all of these fail (records
    stripped, surface starved -> `na`); post-flip all hold (`scorable`)."""
    ev = analyze(out)
    sys.path.insert(0, str(REPO / "tools" / "cache_replay"))
    from sector_cache_replay import decode_cell_access_stream  # type: ignore

    stream = decode_cell_access_stream(out)
    admitted_surface = ev["admitted_records_by_class"].get(surface_class, 0)
    surface_excluded_from_l2 = int(stream.excluded_op_class_counts.get(surface_class, 0))
    scorable = bool(
        ev["validate_trace_dir"]["ok"]
        and ev["trace_contract_id"] == EXPECTED_CONTRACT_ID
        and admitted_surface > 0
        and ev["unsupported_memory_path_flipped_classes"] == 0
        and stream.n_global > 0
        and surface_excluded_from_l2 == 0
    )
    return {
        "surface": surface,
        "surface_class": CLASS_NAME.get(surface_class, surface_class),
        "trace_contract_id": ev["trace_contract_id"],
        "validate_ok": ev["validate_trace_dir"]["ok"],
        "admitted_surface_class_records": admitted_surface,
        "unsupported_memory_path_flipped_classes": ev["unsupported_memory_path_flipped_classes"],
        "cache_replay_l2_stream_sectors": int(stream.n_global),
        "cache_replay_surface_class_excluded_from_l2": surface_excluded_from_l2,
        "before": "na (records stripped to unsupported_memory_path pre-flip; surface starved)",
        "after": "scorable" if scorable else "na",
        "scorable": scorable,
    }


def witness_verdict(name: str, w: dict) -> list[str]:
    """Fail-closed witness conditions: the surface must be scorable."""
    fails = []
    if not w["validate_ok"]:
        fails.append(f"witness {name}: validate_trace_dir failed")
    if w["trace_contract_id"] != EXPECTED_CONTRACT_ID:
        fails.append(f"witness {name}: wrong trace_contract_id {w['trace_contract_id']!r}")
    if w["admitted_surface_class_records"] <= 0:
        fails.append(f"witness {name}: {w['surface']} surface has 0 admitted "
                     f"{w['surface_class']} records (still na)")
    if w["unsupported_memory_path_flipped_classes"] != 0:
        fails.append(f"witness {name}: flipped-class records still in unsupported_memory_path")
    if w["cache_replay_surface_class_excluded_from_l2"] != 0:
        fails.append(f"witness {name}: {w['surface_class']} still excluded from the L2 stream")
    if not w["scorable"]:
        fails.append(f"witness {name}: {w['surface']} surface not scorable (remains na)")
    return fails


def canary_verdict(ev: dict) -> tuple[bool, list[str]]:
    fails = []
    if not ev["validate_trace_dir"]["ok"]:
        fails.append(f"validate_trace_dir failed: {ev['validate_trace_dir']['typed_reason']}")
    if ev["trace_contract_id"] != EXPECTED_CONTRACT_ID:
        fails.append(f"trace_contract_id={ev['trace_contract_id']!r} != {EXPECTED_CONTRACT_ID!r}")
    if ATOMIC not in ev["memory_payload_classes"] or ASYNC_COPY not in ev["memory_payload_classes"]:
        fails.append(f"capability missing ATOMIC/ASYNC_COPY: {ev['memory_payload_classes']}")
    if ev["admitted_atomic"] <= 0:
        fails.append("no admitted ATOMIC records")
    if ev["admitted_async_copy"] <= 0:
        fails.append("no admitted ASYNC_COPY records")
    if ev["unsupported_memory_path_flipped_classes"] != 0:
        fails.append("ATOMIC/ASYNC records still address-stripped to unsupported_memory_path")
    if ev["ldgsts_source_records_with_allocation_id"] <= 0:
        fails.append("no LDGSTS global-source record resolved an allocation_id")
    return (not fails), fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    try:
        plan = _canary_output_plan(out=args.out, work=args.work)
    except ProtectedPathError as exc:
        print(f"[canary] output refusal: {exc}", file=sys.stderr)
        return 2

    if not TRACER_SO.is_file():
        raise SystemExit(
            f"[canary] tracer .so missing at {TRACER_SO}; restore the "
            "byte-exact preserved artifact before capture"
        )

    # Re-clear the directory immediately before its first mutation.  The
    # batch plan above proves the complete graph up front; this second check
    # closes the interval between planning and mkdir (including a symlink
    # substitution at the selected work path).
    work = _assert_staging_output(
        plan["work"], operation="create canary work root",
    )
    work.mkdir(parents=True, exist_ok=True)

    # The canary: exercises every flipped class; proves admission + contract id.
    binary = build_cu(CANARY_CU, work, binary=plan["canary_binary"])
    trace_dir = plan["canary_trace"]
    capture(binary, trace_dir)
    ev = analyze(trace_dir)
    ok, fails = canary_verdict(ev)

    # The two named scorability witnesses: prove op_ld / L2 na -> scorable.
    witnesses = {}
    for w in WITNESSES:
        wbin = build_cu(
            CANARY_DIR / w["cu"],
            work,
            binary=plan[f"{w['name']}_binary"],
        )
        wout = plan[f"{w['name']}_trace"]
        capture(wbin, wout)
        wev = analyze_witness(wout, w["surface"], w["surface_class"])
        witnesses[w["name"]] = wev
        wfails = witness_verdict(w["name"], wev)
        fails.extend(wfails)
    ok = not fails

    manifest = {
        "schema_version": 1,
        "artifact_id": "sm86_addr_carry_canary_v1",
        "arch": "sm_86",
        "canary_pass": ok,
        "canary_failures": fails,
        "canary_source": "tools/local_labels/canary/sm86_addr_carry_canary.cu",
        "provenance": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_clean": _git("status", "--porcelain") == "",
            "tracer_so_present": TRACER_SO.is_file(),
            "clock_state": _clock_state(),
            "cuda_toolkit": "13.2",
        },
        "evidence": ev,
        "scorability_witnesses": witnesses,
    }
    out = _assert_staging_output(plan["out"], operation="write canary manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    verdict = "PASS" if ok else "FAIL"
    for name, wev in witnesses.items():
        print(f"  witness {name}: {wev['surface']} surface {wev['before'].split()[0]}"
              f"->{wev['after']} (admitted {wev['surface_class']}="
              f"{wev['admitted_surface_class_records']}, l2_sectors={wev['cache_replay_l2_stream_sectors']})")
    print(f"canary manifest [{verdict}] -> {out} | contract={ev['trace_contract_id']} "
          f"ATOMIC={ev['admitted_atomic']} ASYNC_COPY={ev['admitted_async_copy']} "
          f"ldgsts_id={ev['ldgsts_source_records_with_allocation_id']} "
          f"flipped_exclusions={ev['unsupported_memory_path_flipped_classes']}")
    for f in fails:
        print(f"  FAIL: {f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
