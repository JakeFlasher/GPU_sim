"""Re-harvest Tier-2 address traces for the 70 sm_120 historical-label cells.

The historical Tier-2 traces (from Rounds 1-2) lived under
``/home/jakeshea/.solbench_round15_tmp/`` which is now gone. The
associativity sweep driver (sibling file ``sweep_associativity.py``)
needs per-cell traces to feed the set-associative simulator, so this
script re-harvests them via NVBit injection on the live RTX 5060
Laptop (sm_120).

Reads the ledger at
``scripts/baselines_harness/calibration/cache_prediction/label_ledger_v1_sm120_partition.toml``,
finds every entry with a ``[entries.historical_ncu_label]`` block,
extracts the binary path from
``historical_ncu_provenance.command_argv[-1]`` (the path the NCU
harvest used), and runs the binary under
``nvbit_tracer/metric_extractor/metric_extractor.so`` with the env
tuned for sm_120 + Tier-2 dump + permissive module regex (since
the calibration binaries are NOT under a ``sol_execbench_*`` prefix).

Output layout (per cell):
  <output_root>/<cell_name>/
    nvbit_metrics/
      tier2_address_trace.t2a      (parsed by tier2_sidecar_parser)
      launches.jsonl
    stdout.log
    stderr.log
    status.json                    (accepted/skipped counts)

The script is idempotent: a cell with a non-empty
``tier2_address_trace.t2a`` of size > 0 already in place is skipped.

Usage:
  python harvest_tier2_traces.py \\
      --output-root /tmp/sm120_tier2_traces \\
      [--cells calib_coalesced_32x4B,problem_001_...]

The default cell list = every ledger entry with a
``historical_ncu_label`` block.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

# __file__ is .../scripts/baselines_harness/calibration/cache_prediction/
#   sm120_l2_associativity_probe/harvest_tier2_traces.py
# repo root is 5 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_LEDGER = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "label_ledger_v1_sm120_partition.toml"
)
_NVBIT_SO = (
    _REPO_ROOT
    / "nvbit_tracer"
    / "metric_extractor"
    / "metric_extractor.so"
)


def _ledger_historical_cells() -> list[dict]:
    """Return ledger entries that carry historical_ncu_label blocks.
    Each entry is a dict; key fields used downstream are
    ``cell_name``, ``predictor_bucket``, and the
    ``historical_ncu_provenance.command_argv`` (for binary path).
    """
    data = tomllib.loads(_LEDGER.read_text("utf-8"))
    return [
        e for e in data.get("entries", [])
        if "historical_ncu_label" in e
    ]


def _binary_path_from_entry(entry: dict) -> Path | None:
    """Extract the binary path from the NCU command_argv. The NCU
    invocation is structured as
    `[ncu, --launch-skip, N, ..., --csv, <binary_path>]`, so the
    binary is the last argv token.
    """
    prov = entry.get("historical_ncu_provenance") or {}
    argv = prov.get("command_argv") or []
    if not argv:
        return None
    return Path(argv[-1])


def harvest_one(
    cell_name: str,
    binary_path: Path,
    output_root: Path,
    *,
    max_slots: int = 1,
    tier2_budget_bytes: int = 256 * 1024 * 1024,
    per_process_budget_bytes: int = 2 * 1024 * 1024 * 1024,
    timeout_s: int = 300,
) -> dict:
    """Run NVBit on ``binary_path`` and capture the Tier-2 sidecar
    plus launches.jsonl into ``output_root / cell_name /
    nvbit_metrics/``. Returns a status dict.
    """
    cell_dir = output_root / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    nvbit_dir = cell_dir / "nvbit_metrics"
    nvbit_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = nvbit_dir / "tier2_address_trace.t2a"

    # Idempotency: if the sidecar already exists and is non-empty,
    # skip the run.
    if sidecar_path.exists() and sidecar_path.stat().st_size > 0:
        return {
            "cell_name": cell_name,
            "status": "cached",
            "sidecar_bytes": sidecar_path.stat().st_size,
        }

    if not binary_path.exists():
        return {
            "cell_name": cell_name,
            "status": "binary_missing",
            "binary_path": str(binary_path),
        }

    env = os.environ.copy()
    env["CUDA_INJECTION64_PATH"] = str(_NVBIT_SO)
    env["METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP"] = "1"
    env["METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"] = str(max_slots)
    env["METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES"] = str(
        tier2_budget_bytes,
    )
    env["METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES"] = str(
        per_process_budget_bytes,
    )
    # The metric_extractor's default user-module regex is
    # ``.*sol_execbench.*`` which excludes the calibration binaries
    # in nvbit_tracer/metric_extractor/tests/. A permissive regex
    # accepts every loaded module.
    env["METRIC_EXTRACTOR_USER_MODULE_REGEX"] = ".*"
    env["METRIC_EXTRACTOR_LAUNCHES_PATH"] = str(
        nvbit_dir / "launches.jsonl",
    )

    stdout_log = cell_dir / "stdout.log"
    stderr_log = cell_dir / "stderr.log"
    try:
        with open(stdout_log, "wb") as so, open(stderr_log, "wb") as se:
            result = subprocess.run(
                [str(binary_path)],
                env=env,
                stdout=so,
                stderr=se,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired:
        return {
            "cell_name": cell_name,
            "status": "timeout",
            "timeout_s": timeout_s,
        }
    except Exception as exc:
        return {
            "cell_name": cell_name,
            "status": "subprocess_error",
            "error": repr(exc),
        }

    # Parse the metric_extractor terminating line to confirm accepted >= 1.
    stderr_tail = stderr_log.read_text("utf-8", errors="replace")
    accepted = 0
    for line in stderr_tail.splitlines():
        if "[metric_extractor] terminating:" in line:
            for token in line.split():
                if token.startswith("accepted="):
                    try:
                        accepted = int(token.split("=", 1)[1])
                    except ValueError:
                        pass
                    break
            break

    status = {
        "cell_name": cell_name,
        "binary_path": str(binary_path),
        "exit_code": result.returncode,
        "accepted_launches": accepted,
        "sidecar_bytes": (
            sidecar_path.stat().st_size if sidecar_path.exists() else 0
        ),
    }
    if accepted < 1:
        status["status"] = "no_launches_accepted"
    elif status["sidecar_bytes"] == 0:
        status["status"] = "sidecar_empty"
    else:
        status["status"] = "ok"
    (cell_dir / "status.json").write_text(
        json.dumps(status, indent=2),
        "utf-8",
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Per-cell trace directories are created under here.",
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="",
        help="Optional comma-separated subset of cell_name values.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="If > 0, stop after this many cells (for smoke runs).",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    entries = _ledger_historical_cells()
    if args.cells:
        wanted = set(args.cells.split(","))
        entries = [e for e in entries if e["cell_name"] in wanted]
    if args.max_cells > 0:
        entries = entries[: args.max_cells]

    print(f"Harvesting {len(entries)} cells -> {args.output_root}")
    statuses: list[dict] = []
    for idx, entry in enumerate(entries, start=1):
        cell_name = entry["cell_name"]
        binary = _binary_path_from_entry(entry)
        if binary is None:
            statuses.append({
                "cell_name": cell_name,
                "status": "no_binary_in_provenance",
            })
            continue
        print(f"[{idx:>3}/{len(entries)}] {cell_name}", flush=True)
        st = harvest_one(cell_name, binary, args.output_root)
        statuses.append(st)
        print(
            f"    -> {st['status']}"
            + (
                f"  accepted={st.get('accepted_launches', 0)}"
                f"  bytes={st.get('sidecar_bytes', 0)}"
                if st["status"] in {"ok", "cached", "no_launches_accepted", "sidecar_empty"}
                else ""
            ),
            flush=True,
        )

    summary_path = args.output_root / "harvest_summary.json"
    summary = {
        "cells_total": len(entries),
        "ok": sum(1 for s in statuses if s["status"] == "ok"),
        "cached": sum(1 for s in statuses if s["status"] == "cached"),
        "binary_missing": sum(
            1 for s in statuses if s["status"] == "binary_missing"
        ),
        "no_launches_accepted": sum(
            1 for s in statuses if s["status"] == "no_launches_accepted"
        ),
        "sidecar_empty": sum(
            1 for s in statuses if s["status"] == "sidecar_empty"
        ),
        "timeout": sum(1 for s in statuses if s["status"] == "timeout"),
        "subprocess_error": sum(
            1 for s in statuses if s["status"] == "subprocess_error"
        ),
        "no_binary_in_provenance": sum(
            1 for s in statuses if s["status"] == "no_binary_in_provenance"
        ),
        "per_cell": statuses,
    }
    summary_path.write_text(json.dumps(summary, indent=2), "utf-8")
    print(f"\nSummary written to {summary_path}")
    print(
        f"  ok={summary['ok']}  cached={summary['cached']}  "
        f"binary_missing={summary['binary_missing']}  "
        f"no_launches={summary['no_launches_accepted']}  "
        f"empty_sidecar={summary['sidecar_empty']}  "
        f"timeout={summary['timeout']}  "
        f"error={summary['subprocess_error']}"
    )
    success = summary["ok"] + summary["cached"]
    return 0 if success == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())
