#!/usr/bin/env python3
"""Run and retain the independent SM86 task-done boundary probe matrix."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_task_done_boundary_20260904_v2"
MODES = ("control", "store", "store_device")
FOOTPRINTS_MB = (1, 4, 8, 16, 64)
CLOCK_STABLE_SAMPLES = 3
CLOCK_MAX_ATTEMPTS = 10
CLOCK_POLL_SECONDS = 1.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _gpu_state_once(*, require_reference_clocks: bool = False) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,driver_version,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,persistence_mode",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.strip()
    if require_reference_clocks and completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed during clock preflight (return code {completed.returncode})"
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
    return {
        "command": command,
        "return_code": completed.returncode,
        "output": output,
    }


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


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT.resolve())),
        "size_bytes": path.stat().st_size,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"probe binary is missing: {binary}")
    before = gpu_state(require_reference_clocks=args.require_reference_clocks)
    records: list[dict[str, Any]] = []
    for mode in MODES:
        for footprint in FOOTPRINTS_MB:
            command = [str(binary), mode, str(footprint), str(args.thrash_mb), str(args.blocks), str(args.trials)]
            destination = output / f"{mode}_{footprint}MiB"
            destination.mkdir()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_seconds,
                check=False,
                env={**os.environ, "LANG": "C", "LC_ALL": "C"},
            )
            (destination / "stdout.jsonl").write_text(completed.stdout, encoding="utf-8")
            (destination / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            (destination / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            lines = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            trials = [line for line in lines if line.get("record") == "trial"]
            summaries = [line for line in lines if line.get("record") == "summary"]
            if completed.returncode != 0 or len(trials) != args.trials or len(summaries) != 1:
                raise RuntimeError(f"{mode}/{footprint}: failed probe execution")
            if not all(line.get("correct") is True for line in trials + summaries):
                raise RuntimeError(f"{mode}/{footprint}: payload/checksum validation failed")
            records.append(
                {
                    "mode": mode,
                    "footprint_mb": footprint,
                    "trials": trials,
                    "summary": summaries[0],
                    "stdout": identity(destination / "stdout.jsonl"),
                    "stderr": identity(destination / "stderr.txt"),
                    "command": command,
                }
            )
            print(f"completed {mode} {footprint}MiB", flush=True)
    after = gpu_state(require_reference_clocks=args.require_reference_clocks)
    manifest = {
        "schema_version": 1,
        "campaign_id": output.name,
        "created_at": now(),
        "repository": {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        },
        "binary": identity(binary),
        "source": identity(HERE / "task_done_boundary.cu"),
        "protocol": {
            "modes": list(MODES),
            "footprints_mb": list(FOOTPRINTS_MB),
            "thrash_mb": args.thrash_mb,
            "blocks": args.blocks,
            "trials": args.trials,
            "sequence": "producer -> write-thrash -> same-stream consumer -> async D2H copy",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thrash-mb", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=84)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--require-reference-clocks",
        action="store_true",
        help="fail closed unless graphics/SM/memory clocks are 1350/1350/10501 MHz",
    )
    args = parser.parse_args()
    if args.trials < 1 or args.trials % 2 == 0:
        parser.error("--trials must be a positive odd number")
    try:
        manifest = run(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"task-done campaign: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output.resolve()), "records": len(manifest["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
