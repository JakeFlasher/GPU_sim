#!/usr/bin/env python3
"""Run simulator predictions from a split manifest, before the NCU join.

The manifest contains traces/cards and fixed split labels. No NCU/CUPTI input is
accepted. Predictions use the declared observer-projection lower endpoint as a
model-local point and preserve open upper bounds. The native execution path is
reported explicitly; detailed replay plus a latent sidecar is not a macro
runtime measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from prediction_contract import (
    validate_prediction_document,
    validate_prediction_row,
    validate_split_roster,
    validate_target_contract,
)


def _run(command: list[str], timeout: float) -> tuple[dict[str, Any], float, int | None]:
    if any(argument.startswith("--ncu") or argument.startswith("--cupti") for argument in command):
        raise ValueError("prediction manifest cannot pass NCU/CUPTI inputs")
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        peak_kb = 0
        while process.poll() is None:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                values = [int(match.group(1)) for match in re.finditer(r"Vm(?:HWM|RSS):\s+(\d+)\s+kB", status)]
                peak_kb = max([peak_kb, *values]) if values else peak_kb
            except (FileNotFoundError, PermissionError):
                pass
            if time.perf_counter() - started > timeout:
                process.kill()
                process.wait()
                raise TimeoutError(f"prediction timed out after {timeout}s")
            time.sleep(0.005)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode()
        stderr = stderr_file.read().decode()
    if process.returncode != 0:
        raise RuntimeError(f"prediction failed with {process.returncode}: {stderr[-2000:]}")
    return json.loads(stdout), (time.perf_counter() - started) * 1000.0, peak_kb * 1024 or None


def run_manifest(manifest: dict[str, Any], sim_run: Path, profile: Path, timeout: float) -> dict[str, Any]:
    if manifest.get("schema") != "reification_prediction_manifest_v1":
        raise ValueError("manifest schema must be reification_prediction_manifest_v1")
    target_contract = validate_target_contract(
        manifest.get("target_contract"), "prediction manifest", require_hardware=True
    )
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest rows must be nonempty")
    validate_split_roster(rows, "prediction manifest")
    if not sim_run.exists():
        raise ValueError(f"sim-run executable does not exist: {sim_run}")
    if not profile.is_file():
        raise ValueError(f"predictive service profile does not exist: {profile}")
    validated_rows: list[dict[str, Any]] = []
    required = ("workload", "family", "split", "trace", "control", "arch_card", "pchase_card")
    # Validate the entire split/family roster before launching even the first
    # simulator process. A late row must not leave a partially materialized
    # prediction document.
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be an object")
        if any(key not in row for key in required):
            raise ValueError(f"manifest row {index} is missing a required field")
        for field in required[3:]:
            path = Path(row[field])
            # Native traces are directory trees; cards and control sidecars
            # are regular files.  Validate existence without flattening the
            # trace representation into a file-only contract.
            if not path.exists():
                raise ValueError(f"manifest row {index}.{field} does not exist: {path}")
        validated_rows.append(row)
    predictions = []
    for row in validated_rows:
        command = [
            str(sim_run),
            "--trace", str(row["trace"]),
            "--control", str(row["control"]),
            "--arch-card", str(row["arch_card"]),
            "--pchase-card", str(row["pchase_card"]),
            "--mode", "flat",
            "--cache-scope", "launch",
            "--latent-envelope",
            "--predictive-service-profile", str(profile),
        ]
        document, runtime_ms, memory_bytes = _run(command, timeout)
        launches = document.get("latent_envelope", {}).get("launches", [])
        timing = document.get("launches", [{}])[0].get("timing", {})
        common_row = {
            "workload": row["workload"],
            "family": row["family"],
            "split": row["split"],
            "intervention": row.get("intervention", "default"),
        }
        if not launches:
            predictions.append({**common_row, "status": "refused", "reason": "latent_envelope_missing"})
            continue
        launch = launches[0]
        projection = launch.get("ncu_projection", {})
        lower = projection.get("predicted_lower_cycles")
        upper = projection.get("predicted_upper_cycles")
        if not isinstance(lower, int):
            predictions.append({
                **common_row,
                "status": "refused",
                "reason": "observer_projection_lower_unknown",
            })
            continue
        predictions.append({
            **common_row,
            "status": "predicted",
            "prediction": {"point": lower, "lower": lower, "upper": upper},
            "point_basis": "declared_observer_projection_lower_bound",
            "observer_projection": {
                "target_metric": projection.get("target_metric"),
                "role": projection.get("role"),
                "conversion": projection.get("conversion"),
                "simulator_lower_cycles": projection.get("simulator_lower_cycles"),
                "simulator_upper_cycles": projection.get("simulator_upper_cycles"),
                "predicted_lower_cycles": lower,
                "predicted_upper_cycles": upper,
            },
            "native_terminal_modeled_event": timing.get("terminal_modeled_event_time"),
            "runtime_ms": round(runtime_ms, 3),
            "memory_bytes": memory_bytes,
            "step_reduction": None,
        })
    document = {
        "schema": "reification_prediction_v2",
        "intervention": "declared_observer_projection",
        "prediction_stage": "pre_observation",
        "execution_path": "native_detailed_replay_with_model_local_latent_sidecar",
        "macro_execution_observed": False,
        "step_reduction_basis": "not_available_native_frontend_exact_fallback",
        "target_contract": target_contract,
        "rows": predictions,
    }
    for index, row in enumerate(predictions):
        validate_prediction_row(row, index)
    return validate_prediction_document(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sim-run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = run_manifest(json.loads(args.manifest.read_text()), args.sim_run, args.profile, args.timeout)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
