#!/usr/bin/env python3
"""Collect comparison-only NCU producer-kernel measurements for the boundary probe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_task_done_boundary_ncu_20260904_v2"
MODES = ("control", "store", "store_device")
FOOTPRINTS_MB = (1, 4, 8, 16, 64)
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "dram__sectors_write.sum",
    "lts__t_sectors_srcunit_tex_lookup_miss.sum",
)
CLOCK_STABLE_SAMPLES = 3
CLOCK_MAX_ATTEMPTS = 10
CLOCK_POLL_SECONDS = 1.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def identity(path: Path) -> dict[str, Any]:
    try:
        portable = str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        portable = str(path.resolve())
    return {
        "path": portable,
        "size_bytes": path.stat().st_size,
    }


def _gpu_state_once(*, require_reference_clocks: bool = False) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,driver_version,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,persistence_mode",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = result.stdout.strip()
    if require_reference_clocks and result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed during clock preflight (return code {result.returncode})"
        )
    if require_reference_clocks:
        fields = [field.strip() for field in output.split(",")]
        if len(fields) != 10:
            raise RuntimeError("clock-required GPU snapshot has an unexpected field count")
        observed = (fields[6], fields[7], fields[8])
        if observed != ("1350", "1350", "10501"):
            raise RuntimeError(
                "live GPU clocks differ from the task-done reference: "
                f"observed graphics/sm/memory={'/'.join(observed)} MHz, expected 1350/1350/10501 MHz"
            )
    return {"command": command, "return_code": result.returncode, "output": output}


def gpu_state(*, require_reference_clocks: bool = False) -> dict[str, Any]:
    """Return a host snapshot, stabilizing the externally locked clocks."""

    if not require_reference_clocks:
        return _gpu_state_once()
    last_error: RuntimeError | None = None
    stable = 0
    for attempt in range(CLOCK_MAX_ATTEMPTS):
        try:
            snapshot = _gpu_state_once(require_reference_clocks=True)
        except RuntimeError as error:
            last_error = error
            stable = 0
        else:
            stable += 1
            if stable >= CLOCK_STABLE_SAMPLES:
                return snapshot
        if attempt + 1 < CLOCK_MAX_ATTEMPTS:
            time.sleep(CLOCK_POLL_SECONDS)
    detail = str(last_error) if last_error is not None else "no matching clock sample"
    raise RuntimeError(
        f"clock preflight did not obtain {CLOCK_STABLE_SAMPLES} consecutive "
        f"samples in {CLOCK_MAX_ATTEMPTS} attempts: {detail}"
    )


def parse_csv(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("Kernel Name", "").strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one producer row, got {len(rows)}")
    row = rows[0]
    if "producer_kernel" not in row["Kernel Name"]:
        raise RuntimeError(f"{path}: selected kernel is {row['Kernel Name']!r}")
    values: dict[str, float] = {}
    for metric in METRICS:
        try:
            values[metric] = float(row[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{path}: metric {metric} is unavailable") from error
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    ncu = args.ncu.resolve()
    for path in (binary, ncu):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing executable: {path}")
    before = gpu_state(require_reference_clocks=args.require_reference_clocks)
    records: list[dict[str, Any]] = []
    environment = {
        **os.environ,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/opt/cuda/bin:/usr/bin:/bin",
    }
    for mode in MODES:
        for footprint in FOOTPRINTS_MB:
            cell = output / f"{mode}_{footprint}MiB"
            cell.mkdir(parents=True, exist_ok=True)
            repeats: list[dict[str, Any]] = []
            for repeat in range(1, args.repeats + 1):
                rep = cell / f"rep-{repeat}"
                rep.mkdir(parents=True, exist_ok=True)
                report_base = rep / "ncu-report"
                report = rep / "ncu-report.ncu-rep"
                csv_path = rep / "ncu.csv"
                target = [str(binary), mode, str(footprint), str(args.thrash_mb), str(args.blocks), str(args.trials)]
                command = [
                    str(ncu),
                    "--force-overwrite",
                    "--export", str(report_base),
                    "--profile-from-start", "on",
                    "--launch-skip", "2",
                    "--launch-count", "1",
                    "--replay-mode", "application",
                    "--app-replay-mode", "strict",
                    "--clock-control", "none",
                    "--cache-control", "all",
                    "--kernel-name-base", "demangled",
                    "--kernel-name", "regex:producer_kernel.*",
                    "--metrics", ",".join(METRICS),
                    *target,
                ]
                if args.resume and report.is_file() and csv_path.is_file():
                    metrics = parse_csv(csv_path)
                    repeats.append({
                        "repeat": repeat,
                        "command": command,
                        "export_command": [str(ncu), "--import", str(report), "--csv", "--page", "raw", "--print-units", "base", "--log-file", str(csv_path)],
                        "metrics": metrics,
                        "report": identity(report),
                        "csv": identity(csv_path),
                    })
                    print(f"reused {mode} {footprint}MiB repeat {repeat}", flush=True)
                    continue
                result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout_seconds, env=environment, check=False)
                (rep / "ncu.stdout.txt").write_text(result.stdout, encoding="utf-8")
                (rep / "ncu.stderr.txt").write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0 or not report.is_file() or report.stat().st_size == 0:
                    raise RuntimeError(f"{mode}/{footprint}/rep-{repeat}: NCU failed ({result.returncode})")
                export = [str(ncu), "--import", str(report), "--csv", "--page", "raw", "--print-units", "base", "--log-file", str(csv_path)]
                exported = subprocess.run(export, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout_seconds, env=environment, check=False)
                (rep / "export.stdout.txt").write_text(exported.stdout, encoding="utf-8")
                (rep / "export.stderr.txt").write_text(exported.stderr, encoding="utf-8")
                if exported.returncode != 0 or not csv_path.is_file() or csv_path.stat().st_size == 0:
                    raise RuntimeError(f"{mode}/{footprint}/rep-{repeat}: NCU export failed ({exported.returncode})")
                metrics = parse_csv(csv_path)
                repeats.append({
                    "repeat": repeat,
                    "command": command,
                    "export_command": export,
                    "metrics": metrics,
                    "report": identity(report),
                    "csv": identity(csv_path),
                })
                print(f"completed {mode} {footprint}MiB repeat {repeat}", flush=True)
            records.append({"mode": mode, "footprint_mb": footprint, "repeat_count": args.repeats, "repeats": repeats})
    after = gpu_state(require_reference_clocks=args.require_reference_clocks)
    manifest = {
        "schema_version": 1,
        "campaign_id": output.name,
        "created_at": now(),
        "repository": {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        },
        "probe_binary": identity(binary),
        "ncu_binary": identity(ncu),
        "source": identity(HERE / "task_done_boundary.cu"),
        "protocol": {
            "profile_from_start": "on",
            "launch_skip": 2,
            "launch_count": 1,
            "replay_mode": "application",
            "app_replay_mode": "strict",
            "clock_control": "none",
            "cache_control": "all",
            "kernel_filter": "regex:producer_kernel.*",
            "metrics": list(METRICS),
            "repeats": args.repeats,
            "thrash_mb": args.thrash_mb,
            "blocks": args.blocks,
            "trials_per_application": args.trials,
            "simulator_runs": 0,
            "ncu_values_used_for_selection": False,
            "reference_clocks_required": args.require_reference_clocks,
            "clock_stable_samples_required": CLOCK_STABLE_SAMPLES,
            "clock_max_attempts": CLOCK_MAX_ATTEMPTS,
            "clock_poll_seconds": CLOCK_POLL_SECONDS,
        },
        "gpu_pre": before,
        "gpu_post": after,
        "records": records,
    }
    (output / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=HERE / "task_done_boundary")
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--thrash-mb", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=84)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--require-reference-clocks",
        action="store_true",
        help="fail closed unless graphics/SM/memory clocks are 1350/1350/10501 MHz",
    )
    parser.add_argument("--resume", action="store_true", help="finalize an existing complete output tree without rerunning reports")
    args = parser.parse_args()
    if args.repeats < 1 or args.trials < 1 or args.trials % 2 == 0:
        parser.error("repeats/trials must be positive; trials must be odd")
    try:
        manifest = run(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"task-done NCU campaign: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output.resolve()), "records": len(manifest["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
