#!/usr/bin/env python3
"""Roster-wide NVBit trace capture toward gate M1 (plan §M1, sim input side).

Captures one NVBit dynamic trace per repeatability-matrix roster cell at its
timing shape, then validates every capture with the offline trace-contract
validator (``validate_trace_dir``). Standard portable_lcd slugs go through
``trace_activation_cell`` (identity + admission sidecar); the ac4 multi-case
cells go through a direct-injection path with per-case output dirs (the
orchestrator derives its output dir from the slug, which would collide).

NVBit-only: never co-run with NCU capture (repo rule). Traces are
timing-free (instruction/address streams), so the clock protocol does not
gate trace validity — but the manifest records the host clock state anyway.

Usage:
  .venv/bin/python tools/local_labels/roster_trace_capture.py \
      --roster tools/local_labels/repeatability_matrix_roster_full.json \
      --output-root results/nvbit_capture_staging/roster_trace_capture
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "nvbit_tracer"))

from tools.local_labels.activation_trace_orchestrator import (  # noqa: E402
    trace_activation_cell,
    _TRACER_SO,
    _binary_sha256,
    _tracer_env,
    _activation_output_plan,
    _assert_staging_output,
    _reject_output_dir_extras,
)
from tools.local_labels.roster_path_resolver import resolve_cell_root  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmissionError,
    require_trace_admission,
)
from import_trace_contract import validate_trace_dir  # noqa: E402


_DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "nvbit_capture_staging" / "roster_trace_capture"
)


def _roster_capture_plan(
    *,
    roster: list[dict],
    sku: str,
    arch_dir: str,
    output_root: Path | str,
    manifest_out: Path | str | None,
) -> tuple[list[dict[str, Path | bool]], Path]:
    """Clear every delegated/direct output before the first capture."""

    root = _assert_staging_output(output_root, operation="select roster output root")
    manifest = _assert_staging_output(
        manifest_out if manifest_out is not None else root / "capture_manifest.json",
        operation="write roster capture manifest",
    )
    plans: list[dict[str, Path | bool]] = []
    for cell in roster:
        slug = str(cell["slug"])
        shape_point = str(cell["shape_point"])
        extras = tuple(
            arg for arg in cell.get("app_args", [])
            if arg != f"--{shape_point}"
        )
        _reject_output_dir_extras(extras)
        portable = (
            REPO_ROOT / "workloads" / "portable_lcd" / slug
        ).is_dir()
        if portable:
            activation = _activation_output_plan(
                slug=slug,
                shape_point=shape_point,
                sku=sku,
                arch_dir=arch_dir,
                fallback_source="portable_lcd",
                output_root=root,
                extra_app_args=extras,
            )
            plans.append({"portable": True, **activation})
            continue
        direct_root = root / "direct" / slug
        direct = {
            "portable": False,
            "cell_dir": direct_root,
            "traces_folder": direct_root / f"trace_{shape_point}",
            "workload_output_root": direct_root / "workload_output",
            "portable_manifest": (
                direct_root / "workload_output" / slug / "manifest.json"
            ),
        }
        plans.append({
            name: (
                value if isinstance(value, bool) else _assert_staging_output(
                    value, operation=f"prepare direct {name}",
                )
            )
            for name, value in direct.items()
        })
    return plans, manifest


def _host_clock_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=persistence_mode,clocks.sm,clocks.mem",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return {"nvidia_smi_persistence_sm_mem": out}
    except Exception as exc:  # pragma: no cover - informational only
        return {"nvidia_smi_error": str(exc)}


def _capture_direct(
    slug: str,
    binary: Path,
    app_args: list[str],
    traces_folder: Path,
    workload_output_root: Path,
) -> dict:
    """Direct NVBit injection for cells the orchestrator cannot route
    (multi-case binaries needing per-case output dirs)."""
    _reject_output_dir_extras(tuple(app_args))
    traces_folder = _assert_staging_output(
        traces_folder, operation="prepare direct trace",
    )
    workload_output_root = _assert_staging_output(
        workload_output_root, operation="prepare direct workload output",
    )
    if traces_folder.exists():
        _assert_staging_output(traces_folder, operation="remove prior direct trace")
        shutil.rmtree(traces_folder)
    traces_folder.mkdir(parents=True, exist_ok=True)
    workload_output_root.mkdir(parents=True, exist_ok=True)
    env = _tracer_env()
    env["CUDA_INJECTION64_PATH"] = str(_TRACER_SO)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(traces_folder)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary), *app_args,
            "--output_dir", str(workload_output_root),
        ], capture_output=True, text=True, env=env,
        cwd=str(binary.parent),
    )
    elapsed = time.perf_counter() - t0
    pb = traces_folder / "dynamic_trace.pb"
    if proc.returncode != 0 or not pb.is_file():
        return {
            "slug": slug, "ok": False,
            "typed_reason": "capture_failed",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-300:],
        }
    try:
        admission = require_trace_admission(traces_folder)
    except TraceAdmissionError as exc:
        return {
            "slug": slug,
            "ok": False,
            "typed_reason": exc.typed_reason,
            "detail": exc.detail,
            "returncode": proc.returncode,
        }
    expected = admission.expected_record_count
    admitted = admission.admitted_record_count
    return {
        "slug": slug, "ok": True,
        "traces_folder": str(traces_folder),
        "trace_pb_bytes": pb.stat().st_size,
        "capture_seconds": round(elapsed, 2),
        "records_expected": expected,
        "records_admitted": admitted,
        "drop_count": max((expected or 0) - (admitted or 0), 0),
        # Label<->trace binary-identity join key — without it the frozen
        # bundle cannot prove the traced binary is the one NCU labeled.
        "binary_sha256": _binary_sha256(binary),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", required=True)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--sku", default="rtx_3090_ti")
    ap.add_argument("--arch-dir", default="sm_86")
    ap.add_argument(
        "--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT,
        help="Active output root below results/nvbit_capture_staging.",
    )
    args = ap.parse_args(argv)

    roster = json.loads(Path(args.roster).read_text())
    try:
        plans, out = _roster_capture_plan(
            roster=roster,
            sku=args.sku,
            arch_dir=args.arch_dir,
            output_root=args.output_root,
            manifest_out=args.manifest_out,
        )
    except (KeyError, ValueError) as exc:
        print(f"roster_trace_capture: output refusal: {exc}", file=sys.stderr)
        return 2

    _tracer_env()  # all outputs cleared before dlopen preflight / capture
    manifest = {
        "generated_by": "roster_trace_capture",
        "host_clock_state": _host_clock_state(),
        "cells": [],
    }
    failures = 0
    for cell, plan in zip(roster, plans, strict=True):
        slug = cell["slug"]
        binary = REPO_ROOT / cell["binary"]
        slug_dir_exists = (
            REPO_ROOT / "workloads" / "portable_lcd" / slug
        ).is_dir()
        print(f"[trace] {slug} ...", flush=True)
        try:
            if slug_dir_exists:
                sidecar = trace_activation_cell(
                    slug=slug,
                    shape_point=cell["shape_point"],
                    arch_dir=args.arch_dir,
                    sku=args.sku,
                    extra_app_args=tuple(
                        a for a in cell.get("app_args", [])
                        if a != f"--{cell['shape_point']}"
                    ),
                    output_root=args.output_root,
                    _precleared_plan={
                        key: value for key, value in plan.items()
                        if key != "portable"
                    },
                )
                traces_folder = Path(plan["traces_folder"])
                rec = {
                    "slug": slug, "ok": True,
                    "traces_folder": str(traces_folder),
                    "trace_pb_bytes": sidecar.get("trace_bytes"),
                    "records_expected": sidecar.get("expected_record_count"),
                    "records_admitted": sidecar.get("admitted_record_count"),
                    "drop_rate": sidecar.get("actual_drop_rate"),
                    "binary_sha256": sidecar.get("binary_sha256"),
                    "via": "trace_activation_cell",
                }
            else:
                traces_folder = Path(plan["traces_folder"])
                rec = _capture_direct(
                    slug,
                    binary,
                    list(cell.get("app_args", [])),
                    traces_folder,
                    Path(plan["workload_output_root"]),
                )
                rec["via"] = "direct_injection"
        except Exception as exc:
            rec = {"slug": slug, "ok": False,
                   "typed_reason": "capture_exception", "detail": str(exc)[:400]}

        # FAIL-CLOSED: a cell is OK only if its trace dir exists AND the
        # contract validation ran and passed — a skipped validation must
        # never read as success.
        if rec.get("ok"):
            tf = rec.get("traces_folder")
            if not tf or not Path(tf).is_dir():
                rec["ok"] = False
                rec["typed_reason"] = "traces_folder_missing"
                rec["contract_ok"] = False
            else:
                v = validate_trace_dir(Path(tf))
                rec["contract_ok"] = bool(v.ok)
                if not v.ok:
                    rec["contract_typed_reason"] = v.typed_reason
                    rec["contract_detail"] = (v.detail or "")[:300]
        if not (rec.get("ok") and rec.get("contract_ok") is True):
            failures += 1
        manifest["cells"].append(rec)

    out = _assert_staging_output(out, operation="write roster capture manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    n_ok = sum(1 for c in manifest["cells"]
               if c.get("ok") and c.get("contract_ok"))
    print(f"trace manifest: {out} — {n_ok}/{len(manifest['cells'])} cells "
          f"captured+contract-valid, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
