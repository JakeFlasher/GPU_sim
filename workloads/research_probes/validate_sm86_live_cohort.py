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
from workloads.research_probes import replay_resume as replay_resume
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


def _required_int(record: Mapping[str, Any], key: str, label: str) -> int:
    value = record.get(key)
    try:
        parsed = int(str(value).replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: integer field {key} is missing or malformed") from exc
    return parsed


def _check_timeout(
    stage: Mapping[str, Any], label: str, defaults: Mapping[str, Any]
) -> int:
    timeout = stage.get("timeout_seconds", defaults.get(label.rsplit("/", 1)[-1]))
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(f"{label}: finite timeout_seconds is required")
    return timeout


def _validate_host_preflight(
    root: Path, host_record_path: Path, repository: Mapping[str, Any]
) -> None:
    try:
        host = json.load(open(host_record_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"host preflight record is not valid JSON: {host_record_path}") from exc
    if not isinstance(host, Mapping) or host.get("schema_version") != 2:
        raise ValueError("host preflight record schema is unsupported")
    binding = host.get("revision_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("host preflight revision binding is missing")
    for key in ("capture_revision", "evidence_source_revision", "candidate_revision"):
        value = binding.get(key)
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"host preflight revision binding {key} is malformed")
    if binding.get("capture_revision") != repository.get("capture_revision"):
        raise ValueError("host preflight capture revision differs from manifest")
    if binding.get("candidate_revision") != repository.get("candidate_revision"):
        raise ValueError("host preflight candidate revision differs from manifest")
    simulator_identity = repository.get("simulator_binary")
    if not isinstance(simulator_identity, Mapping):
        raise ValueError("manifest simulator binary identity is malformed")
    expected_hashes = {
        "simulator_binary_sha256": simulator_identity.get("sha256"),
        "tracer_dso_sha256": repository.get("tracer_dso_sha256"),
        "architecture_card_sha256": repository.get("architecture_card_sha256"),
        "pchase_card_sha256": repository.get("pchase_card_sha256"),
    }
    for key, expected in expected_hashes.items():
        if not isinstance(expected, str) or binding.get(key) != expected:
            raise ValueError(f"host preflight {key} is not bound to the manifest input")
    commands = host.get("commands")
    if not isinstance(commands, list):
        raise ValueError("host preflight command records are missing")
    required_names = {
        "gpu_query",
        "gpu_list",
        "nvcc_version",
        "ncu_version",
        "rust_version",
        "cargo_version",
        "sudo_noninteractive",
        "git_revision",
        "input_hashes",
    }
    seen: set[str] = set()
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError("host preflight command record is malformed")
        name = command.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("host preflight command names must be unique")
        seen.add(name)
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError(f"host preflight {name}: argv is missing")
        timeout = command.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError(f"host preflight {name}: finite timeout is missing")
        return_code = command.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise ValueError(f"host preflight {name}: return code is missing")
        _assert_file(root, command.get("output_path"), f"host preflight {name}.stdout")
        _assert_file(root, command.get("stderr_path"), f"host preflight {name}.stderr")
    if not required_names.issubset(seen):
        raise ValueError(f"host preflight command set is incomplete: {sorted(required_names - seen)}")
    git_command = next(command for command in commands if command.get("name") == "git_revision")
    revision_output = _assert_file(root, git_command["output_path"], "host preflight git revision")
    if revision_output.read_text(encoding="utf-8").strip() != repository.get("candidate_revision"):
        raise ValueError("host preflight git revision does not identify the candidate")


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
                _assert_file(root, stage[key], f"{cell_id}/{stage_name}.{key}")

    correctness = stages["correctness"]
    correctness_stdout = _assert_file(
        root,
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
        root,
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
        root,
        nvbit["output"] + "/dynamic_trace.pb",
        f"{cell_id}/nvbit.dynamic_trace",
    ).parent
    stats = _assert_file(root, nvbit["stats"], f"{cell_id}/nvbit.stats")
    stat_rows = list(csv.DictReader(stats.open(encoding="utf-8", newline="")))
    if len(stat_rows) != 1:
        raise ValueError(f"{cell_id}: expected exactly one NVBit stats row")
    stat = {str(key).strip(): str(value).strip() for key, value in stat_rows[0].items()}
    if stat.get("total_insts") != stat.get("total_reported_insts"):
        raise ValueError(f"{cell_id}: NVBit total/report instruction mismatch")
    mapping = json.load(open(_assert_file(root, nvbit["mapping"], f"{cell_id}/nvbit.mapping")))
    selection = mapping.get("selection", {})
    if selection.get("accepted_active_launches") != 1 or selection.get("exact_contract_satisfied") is not True:
        raise ValueError(f"{cell_id}: NVBit launch selection is not exact")
    if not isinstance(mapping.get("launches"), list) or len(mapping["launches"]) != 1:
        raise ValueError(f"{cell_id}: NVBit mapping must contain exactly one launch")
    if selection.get("accepted_channel_records") != _required_int(
        stat, "total_reported_insts", f"{cell_id}/nvbit.stats"
    ):
        raise ValueError(f"{cell_id}: NVBit mapping/stat record counts differ")
    row["checks"]["nvbit"] = True

    static = stages["static_control"]
    control = _assert_file(root, static["control"], f"{cell_id}/static.control")
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
    if static_binary != trace_binary:
        raise ValueError(f"{cell_id}: static binary is not the traced capture binary")
    if not isinstance(static_hash, str) or len(static_hash) != 64:
        raise ValueError(f"{cell_id}: static binary identity is missing")
    static_binary_path = _assert_file(
        REPO_ROOT, static_binary, f"{cell_id}.static_binary"
    )
    if _sha256(static_binary_path) != static_hash:
        raise ValueError(f"{cell_id}: static binary hash mismatch")
    if static_hash != trace_binary_hash:
        raise ValueError(f"{cell_id}: static and traced binary identities differ")
    if control_doc.get("binary") != static_binary:
        raise ValueError(f"{cell_id}: control sidecar binary identity differs")
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
    first = _assert_file(root, outputs[0], f"{cell_id}/simulator.replay-1")
    second = _assert_file(root, outputs[1], f"{cell_id}/simulator.replay-2")
    if first.read_bytes() != second.read_bytes():
        raise ValueError(f"{cell_id}: simulator twins differ")
    replay_document = json.load(open(first))
    expected_grid = cell.get("expected_grid")
    expected_block = cell.get("expected_block")
    if not (
        isinstance(expected_grid, list)
        and len(expected_grid) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in expected_grid)
        and isinstance(expected_block, list)
        and len(expected_block) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in expected_block)
    ):
        raise ValueError(f"{cell_id}: trusted expected grid/block geometry is missing")
    trace_stats = {key.strip(): value.strip() for key, value in stat.items()}
    expected_blocks = expected_grid[0] * expected_grid[1] * expected_grid[2]
    # NVBit's ``#threads`` column is the per-block launch width; NCU's
    # launch__thread_count below is the whole-grid population.
    expected_threads = expected_block[0] * expected_block[1] * expected_block[2]
    for field, expected_value in (
        ("grid_dimX", expected_grid[0]),
        ("grid_dimY", expected_grid[1]),
        ("grid_dimZ", expected_grid[2]),
        ("block_dimX", expected_block[0]),
        ("block_dimY", expected_block[1]),
        ("block_dimZ", expected_block[2]),
        ("#blocks", expected_blocks),
        ("#threads", expected_threads),
    ):
        if _required_int(trace_stats, field, f"{cell_id}/nvbit.stats") != expected_value:
            raise ValueError(f"{cell_id}: NVBit geometry field {field} differs from trusted manifest")
    for field, expected_value in (
        ("launch__grid_dim_x", expected_grid[0]),
        ("launch__grid_dim_y", expected_grid[1]),
        ("launch__grid_dim_z", expected_grid[2]),
        ("launch__block_dim_x", expected_block[0]),
        ("launch__block_dim_y", expected_block[1]),
        ("launch__block_dim_z", expected_block[2]),
        ("launch__thread_count", expected_blocks * expected_threads),
    ):
        if _required_int(ncu_row, field, f"{cell_id}/ncu.csv") != expected_value:
            raise ValueError(f"{cell_id}: NCU geometry differs from trusted manifest")
    expected_launches = replay_resume.expected_launches_from_trace(
        trace, grid=expected_grid, block=expected_block
    )
    trace_invariants = derive_trace_invariants(trace).by_identity()
    if len(expected_launches) != len(trace_invariants):
        raise ValueError(f"{cell_id}: trace launch/payload identity count differs")
    if len(trace_invariants) != 1:
        raise ValueError(f"{cell_id}: trace must contain exactly one launch identity")
    trace_row = next(iter(trace_invariants.values()))
    if trace_row.blocks != expected_blocks:
        raise ValueError(f"{cell_id}: trace payload block population is incomplete")
    expected_warps_per_block = (expected_block[0] * expected_block[1] * expected_block[2] + 31) // 32
    if trace_row.warps != expected_blocks * expected_warps_per_block:
        raise ValueError(f"{cell_id}: trace warp population is incomplete")
    if _required_int(trace_stats, "total_reported_insts", f"{cell_id}/nvbit.stats") <= 0:
        raise ValueError(f"{cell_id}: NVBit trace has no reported instructions")
    simulator_binary = _assert_file(REPO_ROOT, "sim/target/release/sim-run", f"{cell_id}.simulator_binary")
    simulator_path = simulator.get("binary_path")
    simulator_hash = simulator.get("binary_sha256")
    if simulator_path != "sim/target/release/sim-run" or not isinstance(simulator_hash, str):
        raise ValueError(f"{cell_id}: simulator binary path is not current sim-run")
    if simulator_hash != _sha256(simulator_binary):
        raise ValueError(f"{cell_id}: simulator binary hash differs from current sim-run")
    if simulator_argv[0] != simulator_path:
        raise ValueError(f"{cell_id}: simulator argv is not bound to current sim-run")
    dependencies = [
        simulator_binary,
        control,
        _assert_file(REPO_ROOT, "arch_profiles/cards/sm_86.json", f"{cell_id}.arch_card"),
        _assert_file(REPO_ROOT, "data/pchase/sm_86_static_card.json", f"{cell_id}.pchase_card"),
        trace,
    ]
    expectation = replay_resume.ReplayExpectation(
        launches=expected_launches, carded_dram=False
    )
    replay_error = replay_resume.replay_pair_error(
        first, second, dependencies, expectation
    )
    if replay_error is not None:
        raise ValueError(f"{cell_id}: strict direct replay validation failed: {replay_error}")
    semantic_rows, semantic_errors = validate_replay_document(
        replay_document, trace, label=cell_id
    )
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
    repository = document.get("repository")
    if not isinstance(repository, Mapping):
        raise ValueError("manifest repository identity is missing")
    for revision_key in ("capture_revision", "handoff_revision", "candidate_revision"):
        revision = repository.get(revision_key)
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"manifest repository {revision_key} is malformed")
    simulator_record = repository.get("simulator_binary")
    if not isinstance(simulator_record, Mapping):
        raise ValueError("manifest simulator binary identity is missing")
    simulator_binary = _assert_file(
        REPO_ROOT, simulator_record.get("path"), "repository.simulator_binary"
    )
    if _sha256(simulator_binary) != simulator_record.get("sha256"):
        raise ValueError("manifest simulator binary SHA-256 does not match")
    host = document.get("host")
    if not isinstance(host, Mapping):
        raise ValueError("manifest host preflight record is missing")
    host_record_path = _assert_file(root, host.get("preflight_record"), "host.preflight_record")
    _validate_host_preflight(root, host_record_path, repository)
    bundle = document.get("evidence_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("manifest evidence_bundle is missing")
    archive = _assert_file(root, bundle.get("path"), "evidence_bundle")
    if _sha256(archive) != bundle.get("sha256"):
        raise ValueError("evidence bundle SHA-256 does not match manifest")
    if archive.stat().st_size != bundle.get("size_bytes"):
        raise ValueError("evidence bundle size does not match manifest")
    closure_analysis = document.get("closure_analysis")
    if not isinstance(closure_analysis, Mapping):
        raise ValueError("manifest closure_analysis is missing")
    _check_timeout(closure_analysis, "closure_analysis", {"closure_analysis": 180})
    if closure_analysis.get("exit_code") != 0:
        raise ValueError("closure analysis did not exit zero")
    closure_report_path = _assert_file(
        root, closure_analysis.get("report"), "closure_analysis.report"
    )
    if _sha256(closure_report_path) != closure_analysis.get("report_sha256"):
        raise ValueError("closure analysis report SHA-256 does not match manifest")
    closure_report = json.load(open(closure_report_path))
    if not isinstance(closure_report, Mapping) or closure_report.get("schema_version") != 1:
        raise ValueError("closure analysis report schema is unsupported")
    analysis = closure_report.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("decision") != "no-go":
        raise ValueError("closure analysis decision is missing or not no-go")
    if analysis.get("candidate_revision") not in (None, repository.get("candidate_revision")):
        raise ValueError("closure analysis candidate revision differs from manifest")
    closure_rows = closure_report.get("rows")
    if not isinstance(closure_rows, list) or len(closure_rows) != 6:
        raise ValueError("closure analysis must contain one row per cohort cell")
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("cell"), str)
        or not isinstance(row.get("live"), Mapping)
        or row["live"].get("stable_within_five_percent") is not True
        or not isinstance(row.get("simulator"), Mapping)
        or row["simulator"].get("twin_byte_identical") is not True
        for row in closure_rows
    ):
        raise ValueError("closure analysis rows lack repeatability or simulator-twin evidence")
    if analysis.get("ncu_values_used_as_simulator_inputs") is not False:
        raise ValueError("closure analysis violates the NCU input boundary")
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
    try:
        manifest_label = str(manifest_path.relative_to(REPO_ROOT))
    except ValueError:
        manifest_label = str(manifest_path)
    report = {
        "schema_version": 1,
        "manifest": manifest_label,
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
