#!/usr/bin/env python3
"""Validate the durable six-cell SM86 live-campaign handoff.

The validator consumes the repository-relative manifest and materialized raw
evidence bundle.  It keeps NCU observations on the comparison side of the
boundary and independently replays the NVBit trace through the semantic
conservation checks used by the retained-corpus validator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.semantic_replay_validation import (
    validate_replay_document,
)
from workloads.research_probes.trace_invariants import derive_trace_invariants


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results/captures/sm86_simulator_live_baseline_20260829_v2/campaign_manifest.json"
)
STAGES = ("correctness", "ncu", "nvbit", "static_control", "simulator")
_CURRENT_TIMEOUTS = {
    "correctness": 180,
    "ncu": 900,
    "nvbit": 900,
    "static_control": 300,
    "simulator": 900,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_json_object(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for offset in reversed([index for index, value in enumerate(text) if value == "{"]):
        try:
            value, _end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"{path}: no JSON object in captured stdout")


def _kernel_row(path: Path) -> Mapping[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        if row.get("Kernel Name"):
            return row
    raise ValueError(f"{path}: NCU CSV has no kernel row")


def _assert_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}: missing repository-relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes manifest root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{label}: artifact is missing: {relative}")
    return path


def _check_timeout(
    stage: Mapping[str, Any], label: str, defaults: Mapping[str, Any]
) -> int:
    timeout = stage.get("timeout_seconds", defaults.get(label.rsplit("/", 1)[-1]))
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(f"{label}: finite timeout_seconds is required")
    return timeout


def _validate_cell(
    root: Path, cell: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    cell_id = cell.get("id")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell has no id")
    binary = cell.get("binary")
    if not isinstance(binary, Mapping):
        raise ValueError(f"{cell_id}: binary record is missing")
    binary_path = _assert_file(REPO_ROOT, binary.get("path"), f"{cell_id}.binary")
    if _sha256(binary_path) != binary.get("sha256"):
        raise ValueError(f"{cell_id}: binary SHA-256 does not match manifest")
    trace_binary_path = _assert_file(
        REPO_ROOT,
        binary.get("trace_path", binary.get("path")),
        f"{cell_id}.trace_binary",
    )
    trace_binary_hash = binary.get("trace_sha256", binary.get("sha256"))
    if _sha256(trace_binary_path) != trace_binary_hash:
        raise ValueError(f"{cell_id}: traced-binary SHA-256 does not match manifest")
    stages = cell.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
        raise ValueError(f"{cell_id}: stage set is incomplete")

    # The manifest lives at <repo>/results/captures/<campaign>; derive the
    # repository root explicitly so all artifact paths remain stable.
    repo = REPO_ROOT
    row: dict[str, Any] = {"cell": cell_id, "checks": {}}
    # The manifest's stage-timeout table is part of the reproducibility
    # contract; an individual stage may override it only with another finite
    # value.
    for stage_name in STAGES:
        stage = stages[stage_name]
        if not isinstance(stage, Mapping):
            raise ValueError(f"{cell_id}/{stage_name}: stage record is malformed")
        _check_timeout(stage, f"{cell_id}/{stage_name}", defaults)
        if stage.get("exit_code") != 0:
            raise ValueError(f"{cell_id}/{stage_name}: exit_code is not zero")
        for key in ("stdout", "stderr", "csv", "stats", "mapping", "control"):
            if key in stage:
                _assert_file(repo / "results/captures/sm86_simulator_live_baseline_20260829_v2", stage[key], f"{cell_id}/{stage_name}.{key}")

    correctness = stages["correctness"]
    correctness_stdout = _assert_file(
        repo / "results/captures/sm86_simulator_live_baseline_20260829_v2",
        correctness["stdout"],
        f"{cell_id}/correctness.stdout",
    )
    correctness_result = _last_json_object(correctness_stdout)
    if correctness_result.get("correctness_ok", correctness_result.get("correct")) is not True:
        raise ValueError(f"{cell_id}: correctness result is not true")
    row["checks"]["correctness"] = True

    ncu = stages["ncu"]
    ncu_argv = ncu.get("argv")
    if not isinstance(ncu_argv, list) or binary.get("path") not in ncu_argv:
        raise ValueError(f"{cell_id}: NCU argv is not bound to the cell binary")
    ncu_csv = _assert_file(
        repo / "results/captures/sm86_simulator_live_baseline_20260829_v2",
        ncu["csv"],
        f"{cell_id}/ncu.csv",
    )
    ncu_row = _kernel_row(ncu_csv)
    ncu_observed = ncu.get("observed")
    if not isinstance(ncu_observed, Mapping):
        raise ValueError(f"{cell_id}: NCU observed metrics are missing")
    for metric, expected in ncu_observed.items():
        if metric not in ncu_row:
            raise ValueError(f"{cell_id}: NCU metric is absent: {metric}")
        try:
            actual = float(str(ncu_row[metric]).replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"{cell_id}: NCU metric is not numeric: {metric}") from exc
        if abs(actual - float(expected)) > 0.5:
            raise ValueError(f"{cell_id}: NCU metric differs: {metric}")
    row["checks"]["ncu"] = True

    nvbit = stages["nvbit"]
    nvbit_argv = nvbit.get("argv")
    trace_binary = binary.get("trace_path", binary.get("path"))
    if not isinstance(nvbit_argv, list) or trace_binary not in nvbit_argv:
        raise ValueError(f"{cell_id}: NVBit argv is not bound to the traced binary")
    trace = _assert_file(
        repo / "results/captures/sm86_simulator_live_baseline_20260829_v2",
        nvbit["output"] + "/dynamic_trace.pb",
        f"{cell_id}/nvbit.dynamic_trace",
    ).parent
    stats = _assert_file(repo / "results/captures/sm86_simulator_live_baseline_20260829_v2", nvbit["stats"], f"{cell_id}/nvbit.stats")
    stat_rows = list(csv.DictReader(stats.open(encoding="utf-8", newline="")))
    if len(stat_rows) != 1:
        raise ValueError(f"{cell_id}: expected exactly one NVBit stats row")
    stat = {str(key).strip(): str(value).strip() for key, value in stat_rows[0].items()}
    if stat.get("total_insts") != stat.get("total_reported_insts"):
        raise ValueError(f"{cell_id}: NVBit total/report instruction mismatch")
    mapping = json.load(open(repo / "results/captures/sm86_simulator_live_baseline_20260829_v2" / nvbit["mapping"]))
    selection = mapping.get("selection", {})
    if selection.get("accepted_active_launches") != 1 or selection.get("exact_contract_satisfied") is not True:
        raise ValueError(f"{cell_id}: NVBit launch selection is not exact")
    row["checks"]["nvbit"] = True

    static = stages["static_control"]
    control = _assert_file(repo / "results/captures/sm86_simulator_live_baseline_20260829_v2", static["control"], f"{cell_id}/static.control")
    control_doc = json.load(open(control))
    decoded = sum(
        1
        for entries in control_doc.get("functions", {}).values()
        for entry in entries
        if entry.get("status") == "decoded"
    )
    if decoded <= 0:
        raise ValueError(f"{cell_id}: static control sidecar has no decoded instructions")
    static_binary = static.get("binary_path")
    static_hash = static.get("binary_sha256")
    if static_binary != binary.get("path") and cell_id != "cp_async_canary":
        raise ValueError(f"{cell_id}: static binary is not the capture binary")
    if not isinstance(static_hash, str) or len(static_hash) != 64:
        raise ValueError(f"{cell_id}: static binary identity is missing")
    if _sha256(repo / static_binary) != static_hash:
        raise ValueError(f"{cell_id}: static binary hash mismatch")
    row["checks"]["static_control"] = True

    simulator = stages["simulator"]
    simulator_argv = simulator.get("argv")
    if not isinstance(simulator_argv, list):
        raise ValueError(f"{cell_id}: simulator argv is missing")
    if nvbit.get("output") not in simulator_argv or static.get("control") not in simulator_argv:
        raise ValueError(f"{cell_id}: simulator argv is not bound to trace/control inputs")
    outputs = simulator.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError(f"{cell_id}: simulator twin outputs are missing")
    first = _assert_file(repo / "results/captures/sm86_simulator_live_baseline_20260829_v2", outputs[0], f"{cell_id}/simulator.replay-1")
    second = _assert_file(repo / "results/captures/sm86_simulator_live_baseline_20260829_v2", outputs[1], f"{cell_id}/simulator.replay-2")
    if first.read_bytes() != second.read_bytes():
        raise ValueError(f"{cell_id}: simulator twins differ")
    replay = json.load(open(first))
    semantic_rows, semantic_errors = validate_replay_document(replay, trace, label=cell_id)
    if semantic_errors:
        raise ValueError(f"{cell_id}: semantic replay errors: {'; '.join(semantic_errors[:3])}")
    row["checks"]["simulator_semantics"] = True
    row["semantic_launches"] = semantic_rows
    return row


def validate(manifest_path: Path, report_path: Path | None) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    root = manifest_path.parent
    document = json.load(open(manifest_path))
    if document.get("schema_version") != "sm86.live.campaign.v2":
        raise ValueError("unsupported campaign manifest schema")
    bundle = document.get("evidence_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("manifest evidence_bundle is missing")
    archive = _assert_file(root, bundle.get("path"), "evidence_bundle")
    if _sha256(archive) != bundle.get("sha256"):
        raise ValueError("evidence bundle SHA-256 does not match manifest")
    if archive.stat().st_size != bundle.get("size_bytes"):
        raise ValueError("evidence bundle size does not match manifest")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("manifest must contain exactly six cohort cells")
    timeouts = document.get("stage_timeouts_seconds")
    if not isinstance(timeouts, Mapping):
        raise ValueError("manifest stage_timeouts_seconds is missing")
    if any(
        timeouts.get(stage) != value for stage, value in _CURRENT_TIMEOUTS.items()
    ):
        raise ValueError("manifest stage timeout table differs from the validated contract")
    rows = [_validate_cell(root, cell, timeouts) for cell in cells]
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "cell_count": len(rows),
        "semantic_launch_count": sum(len(row["semantic_launches"]) for row in rows),
        "passed": True,
        "rows": rows,
        "ncu_values_used_as_simulator_inputs": False,
    }
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validated fresh SM86 cohort: {len(rows)} cells, {report['semantic_launch_count']} semantic launches")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        return validate(args.manifest, args.report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"live cohort validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
