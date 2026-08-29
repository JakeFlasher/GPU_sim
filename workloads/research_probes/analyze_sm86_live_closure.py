#!/usr/bin/env python3
"""Build a reproducible residual-selection report for the SM86 live cohort.

This is deliberately an analysis-only tool.  It reads the already captured
NCU and simulator artifacts, compares an independent NCU repeat with the
baseline, and never writes measurements back into simulator inputs.
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
from workloads.research_probes import replay_resume
from workloads.research_probes.semantic_replay_validation import (
    validate_replay_document,
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results/captures/sm86_simulator_live_baseline_20260829_v2/campaign_manifest.json"
)
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__inst_executed.sum",
)
TRUSTED_NCU_IDENTITY = {
    "path": "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu",
    "version": "2026.1.0.0",
    "sha256": "a44ff2c735c4dcf66c78f3f430155805d6a0b1f74f26331f98ecec03bfbf3f1a",
}
CAUSE_NAMES = (
    "control_stall",
    "dependence_counter_read",
    "dependence_counter_non_memory_write",
    "dependence_counter_shared_write",
    "dependence_counter_l1tex",
    "dependence_counter_unknown_origin",
    "async_source",
    "async_destination_local",
    "async_destination_sm_service",
    "cta_barrier_rendezvous",
    "pipeline_tensor",
    "eligible_not_selected",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"input tree has no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        payload_hash = _sha256(item).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload_hash).to_bytes(8, "big"))
        digest.update(payload_hash)
    return digest.hexdigest()


def _campaign_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}: missing campaign-relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes campaign root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{label}: file is missing: {relative}")
    return path


def _campaign_dir(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}: missing campaign-relative directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes campaign root: {relative}") from exc
    if not path.is_dir():
        raise ValueError(f"{label}: directory is missing: {relative}")
    return path


def _label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _ncu_row(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        if row.get("Kernel Name"):
            return {str(key): str(value) for key, value in row.items()}
    raise ValueError(f"{path}: no kernel row")


def _metric(row: Mapping[str, str], name: str) -> int:
    value = row.get(name)
    if value is None:
        raise ValueError(f"NCU row has no metric {name}")
    try:
        return int(round(float(value.replace(",", ""))))
    except ValueError as exc:
        raise ValueError(f"NCU metric {name} is not numeric") from exc


def _top_causes(readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    limiting = readiness.get("limiting_cycles")
    if not isinstance(limiting, Mapping):
        return []
    values = []
    for name in CAUSE_NAMES:
        value = limiting.get(name, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            values.append({"cause": name, "limiting_cycles": value})
    values.sort(key=lambda item: (-item["limiting_cycles"], item["cause"]))
    return values[:3]


def _replay_summary(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise ValueError(f"{path}: expected one replay launch")
    launch = launches[0]
    if not isinstance(launch, Mapping):
        raise ValueError(f"{path}: replay launch is malformed")
    timing = launch.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError(f"{path}: replay timing is missing")
    readiness = timing.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError(f"{path}: replay readiness is missing")
    frontier = timing.get("completion_frontier")
    if not isinstance(frontier, Mapping):
        raise ValueError(f"{path}: replay completion frontier is missing")
    counters = launch.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError(f"{path}: replay counters are missing")
    tails = counters.get("unmodeled_tails")
    if not isinstance(tails, Mapping):
        tails = frontier.get("unmodeled_tails", {})
    issue = timing.get("diagnostic_last_issue_cycles")
    last_event = timing.get("last_processed_modeled_system_event_time")
    terminal = timing.get("terminal_modeled_event_time")
    if not isinstance(issue, int) or isinstance(issue, bool):
        raise ValueError(f"{path}: diagnostic last-issue frontier is malformed")
    if last_event is not None and (
        not isinstance(last_event, int) or isinstance(last_event, bool)
    ):
        raise ValueError(f"{path}: last-modeled-event frontier is malformed")
    if not isinstance(terminal, int) or isinstance(terminal, bool):
        raise ValueError(f"{path}: terminal modeled event time is malformed")
    if not isinstance(frontier.get("status"), str):
        raise ValueError(f"{path}: completion frontier status is malformed")
    lower_bound = frontier.get("lower_bound_cycles")
    if lower_bound is not None and (
        not isinstance(lower_bound, int) or isinstance(lower_bound, bool)
    ):
        raise ValueError(f"{path}: completion frontier lower bound is malformed")
    typed_tails = frontier.get("unmodeled_tails")
    if not isinstance(typed_tails, Mapping) or any(
        not isinstance(typed_tails.get(name), bool)
        for name in (
            "reduction_read_modify_write",
            "dirty_writeback",
            "coherence",
            "async_copy_shared_visibility",
        )
    ):
        raise ValueError(f"{path}: completion frontier typed tails are malformed")
    return {
        "last_issue_cycles": issue,
        "last_modeled_event_cycles": last_event,
        "terminal_lower_bound_cycles": terminal,
        "completion_status": frontier.get("status"),
        "completion_frontier": {
            "status": frontier.get("status"),
            "lower_bound_cycles": lower_bound,
            "unmodeled_tails": dict(typed_tails),
        },
        "tails": dict(typed_tails),
        "readiness_schema": readiness.get("schema_version"),
        "top_readiness_causes": _top_causes(readiness),
        "instructions_issued": counters.get("instructions_issued"),
        "records_seen": counters.get("records_seen"),
        "sha256": _sha256(path),
    }


def _validate_replay_binding(
    root: Path,
    cell: Mapping[str, Any],
    nvbit: Mapping[str, Any],
    static: Mapping[str, Any],
    simulator: Mapping[str, Any],
    outputs: list[str],
) -> None:
    slug = str(cell["id"])
    relative = simulator.get("replay_binding")
    expected_relative = f"evidence/raw/simulator/{slug}/replay_binding.json"
    if relative != expected_relative:
        raise ValueError(f"{slug}: replay binding path is not the simulator cell binding")
    binding_path = _campaign_file(root, relative, f"{slug}.replay_binding")
    if simulator.get("replay_binding_sha256") != _sha256(binding_path):
        raise ValueError(f"{slug}: replay binding hash does not match")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(binding, Mapping) or type(binding.get("schema_version")) is not int or binding.get("schema_version") != 1 or binding.get("cell") != slug:
        raise ValueError(f"{slug}: replay binding schema is malformed")
    freshness = binding.get("freshness")
    if not isinstance(freshness, Mapping) or freshness.get("mode") != "content_bound":
        raise ValueError(f"{slug}: replay binding freshness mode is unsupported")
    if binding.get("command") != simulator.get("argv"):
        raise ValueError(f"{slug}: replay binding command differs")
    argv = simulator.get("argv")
    if not isinstance(argv, list) or argv[:1] != ["sim/target/release/sim-run"]:
        raise ValueError(f"{slug}: simulator command is not the current sim-run")
    for option, expected in (
        ("--trace", nvbit.get("output")),
        ("--control", static.get("control")),
        ("--arch-card", "arch_profiles/cards/sm_86.json"),
        ("--pchase-card", "data/pchase/sm_86_static_card.json"),
        ("--mode", "l1-l2"),
    ):
        if argv.count(option) != 1:
            raise ValueError(f"{slug}: simulator command must contain one {option}")
        index = argv.index(option)
        if index + 1 >= len(argv):
            raise ValueError(f"{slug}: simulator command has no value for {option}")
        actual = argv[index + 1]
        if option in {"--trace", "--control"}:
            if not _campaign_token_matches(actual, str(expected), root):
                raise ValueError(f"{slug}: simulator command input is misbound for {option}")
        elif actual != expected:
            raise ValueError(f"{slug}: simulator command value differs for {option}")
    binary = REPO_ROOT / str(simulator.get("binary_path", ""))
    if not binary.is_file() or simulator.get("binary_sha256") != _sha256(binary):
        raise ValueError(f"{slug}: simulator binary identity is not current")
    inputs = binding.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{slug}: replay binding inputs are missing")
    trace_path = _campaign_dir(root, nvbit.get("output"), f"{slug}.trace")
    control_path = _campaign_file(root, static.get("control"), f"{slug}.control")
    input_paths = {
        "trace": (nvbit.get("output"), trace_path, _tree_sha256(trace_path)),
        "control": (static.get("control"), control_path, _sha256(control_path)),
        "architecture_card": ("arch_profiles/cards/sm_86.json", REPO_ROOT / "arch_profiles/cards/sm_86.json", _sha256(REPO_ROOT / "arch_profiles/cards/sm_86.json")),
        "pchase_card": ("data/pchase/sm_86_static_card.json", REPO_ROOT / "data/pchase/sm_86_static_card.json", _sha256(REPO_ROOT / "data/pchase/sm_86_static_card.json")),
    }
    for name, (expected_path, _path, expected_hash) in input_paths.items():
        record = inputs.get(name)
        if not isinstance(record, Mapping) or record.get("path") != expected_path or record.get("sha256") != expected_hash:
            raise ValueError(f"{slug}: replay binding input mismatch for {name}")
    output_records = binding.get("outputs")
    if not isinstance(output_records, list) or len(output_records) != 2:
        raise ValueError(f"{slug}: replay binding outputs are missing")
    if outputs[0] == outputs[1]:
        raise ValueError(f"{slug}: replay output paths must be distinct")
    for record, expected in zip(output_records, outputs):
        output_path = _campaign_file(root, expected, f"{slug}.replay")
        if not isinstance(record, Mapping) or record.get("path") != expected or record.get("sha256") != _sha256(output_path):
            raise ValueError(f"{slug}: replay binding output mismatch")


def _validate_replay_semantics(
    root: Path,
    cell: Mapping[str, Any],
    trace: Path,
    outputs: list[str],
) -> dict[str, Any]:
    slug = str(cell["id"])
    grid = cell.get("expected_grid")
    block = cell.get("expected_block")
    if not isinstance(grid, list) or not isinstance(block, list):
        raise ValueError(f"{slug}: replay semantic geometry is missing")
    try:
        expected_launches = replay_resume.expected_launches_from_trace(
            trace, grid=grid, block=block
        )
        expectation = replay_resume.ReplayExpectation(
            launches=expected_launches, carded_dram=False
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{slug}: cannot derive replay expectation: {exc}") from exc
    semantic_launches = 0
    for relative in outputs:
        path = _campaign_file(root, relative, f"{slug}.replay")
        error = replay_resume.direct_replay_schema_error(path, expectation)
        if error is not None:
            raise ValueError(f"{slug}: strict replay schema failed: {error}")
        document = replay_resume.load_json_strict(path)
        if not isinstance(document, Mapping):
            raise ValueError(f"{slug}: replay root is not an object")
        rows, errors = validate_replay_document(document, trace, label=slug)
        if errors:
            raise ValueError(f"{slug}: semantic replay errors: {'; '.join(errors[:3])}")
        semantic_launches += len(rows)
    return {"passed": True, "semantic_launches": semantic_launches}


def _campaign_token_matches(token: object, relative: str, root: Path) -> bool:
    if not isinstance(token, str):
        return False
    candidates = {
        relative,
        "results/captures/sm86_simulator_live_baseline_20260829_v2/" + relative,
        str((root / relative).resolve()),
    }
    return token in candidates


def _repeat_command_contract(
    root: Path,
    cell: Mapping[str, Any],
    stage: Mapping[str, Any],
    repeat_dir: str,
    ncu_identity: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """Return expected NCU argv and the indices whose paths may be normalized."""

    source = stage.get("argv")
    tool = stage.get("tool_path")
    if not isinstance(source, list) or any(not isinstance(item, str) for item in source):
        raise ValueError(f"{cell.get('id')}: canonical NCU argv is malformed")
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"{cell.get('id')}: canonical NCU tool path is missing")
    if tool != ncu_identity.get("path"):
        raise ValueError(f"{cell.get('id')}: canonical NCU tool path is not the trusted toolchain")
    try:
        output_index = source.index("--output")
        separator = source.index("--", output_index)
    except ValueError as exc:
        raise ValueError(f"{cell.get('id')}: canonical NCU argv lacks output/target separator") from exc
    if output_index + 1 >= separator:
        raise ValueError(f"{cell.get('id')}: canonical NCU output path is malformed")
    target = source[separator + 1 :]
    if not target or target[0] != cell.get("binary", {}).get("path"):
        raise ValueError(f"{cell.get('id')}: canonical NCU target is not the cell binary")
    def option_value(name: str) -> str | None:
        try:
            index = source.index(name)
        except ValueError:
            return None
        if index + 1 >= separator:
            raise ValueError(f"{cell.get('id')}: canonical NCU option {name} has no value")
        return source[index + 1]
    command = [tool, "--force-overwrite", "--export", repeat_dir + "/ncu-report"]
    for name in ("--launch-skip", "--launch-count", "--replay-mode", "--clock-control", "--metrics"):
        value = option_value(name)
        if value is not None:
            command.extend((name, value))
    command.extend(target)
    return command, target


def _validate_repeat_metadata(
    root: Path,
    cell: Mapping[str, Any],
    stage: Mapping[str, Any],
    repeat_dir: str,
    metadata: Mapping[str, Any],
    ncu_identity: Mapping[str, str],
) -> None:
    slug = str(cell["id"])
    expected_command, target = _repeat_command_contract(
        root, cell, stage, repeat_dir, ncu_identity
    )
    command = metadata.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError(f"{slug}: repeated NCU metadata command is missing")
    if len(command) != len(expected_command):
        raise ValueError(f"{slug}: repeated NCU command length differs from the cell contract")
    for index, (actual, expected) in enumerate(zip(command, expected_command)):
        if index == 3:  # --export value
            if not _campaign_token_matches(actual, expected, root):
                raise ValueError(f"{slug}: repeated NCU export path is misbound")
        elif actual != expected:
            raise ValueError(f"{slug}: repeated NCU command differs at argv[{index}]")
    return_code = metadata.get("return_code")
    if isinstance(return_code, bool) or not isinstance(return_code, int) or return_code != 0:
        raise ValueError(f"{slug}: repeated NCU return_code is malformed or nonzero")
    expected_csv = repeat_dir + "/ncu.csv"
    expected_report = repeat_dir + "/ncu-report.ncu-rep"
    for key, expected in (("csv", expected_csv), ("report", expected_report)):
        if not _campaign_token_matches(metadata.get(key), expected, root):
            raise ValueError(f"{slug}: repeated NCU metadata {key} path is misbound")
    _campaign_file(root, expected_report, f"{slug}.ncu_repeat.report")
    if target[0] != cell.get("binary", {}).get("path"):
        raise ValueError(f"{slug}: repeated NCU target binary differs")


def _repeat_metrics(
    root: Path,
    cell: Mapping[str, Any],
    stage: Mapping[str, Any],
    ncu_identity: Mapping[str, str],
) -> dict[str, Any]:
    slug = str(cell["id"])
    baseline_csv = _campaign_file(root, stage.get("csv"), f"{slug}.ncu")
    repeat_dir = f"evidence/raw/ncu_repeat/rep-2/{slug}"
    repeat_csv = _campaign_file(
        root, repeat_dir + "/ncu.csv", f"{slug}.ncu_repeat"
    )
    repeat_metadata = _campaign_file(
        root, repeat_dir + "/metadata.json", f"{slug}.ncu_repeat.metadata"
    )
    repeat_report = _campaign_file(
        root, repeat_dir + "/ncu-report.ncu-rep", f"{slug}.ncu_repeat.report"
    )
    metadata = json.loads(repeat_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{slug}: repeated NCU metadata is malformed")
    _validate_repeat_metadata(root, cell, stage, repeat_dir, metadata, ncu_identity)
    baseline_row = _ncu_row(baseline_csv)
    repeat_row = _ncu_row(repeat_csv)
    if any(name not in repeat_row for name in METRICS):
        raise ValueError(f"{slug}: repeated NCU CSV does not contain the canonical metric set")
    baseline = {name: _metric(baseline_row, name) for name in METRICS}
    repeat = {name: _metric(repeat_row, name) for name in METRICS}
    deltas = {name: repeat[name] - baseline[name] for name in METRICS}
    max_relative = max(
        abs(deltas[name]) / max(abs(baseline[name]), 1) for name in METRICS
    )
    return {
        "baseline": baseline,
        "repeat_2": repeat,
        "delta_repeat_minus_baseline": deltas,
        "max_relative_delta": round(max_relative, 8),
        "stable_within_five_percent": max_relative <= 0.05,
        "baseline_csv": {"path": _label(baseline_csv), "sha256": _sha256(baseline_csv)},
        "repeat_csv": {"path": _label(repeat_csv), "sha256": _sha256(repeat_csv)},
        "repeat_report": {"path": _label(repeat_report), "sha256": _sha256(repeat_report)},
        "repeat_metadata": {
            "path": _label(repeat_metadata),
            "sha256": _sha256(repeat_metadata),
            "argv": metadata.get("command"),
            "started_at": metadata.get("started_at"),
            "finished_at": metadata.get("finished_at"),
            "contract_validated": True,
        },
    }


def _derive_decision(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Derive the closure decision from validated observations.

    A no-go is only eligible when the live repeat and the simulator diagnostic
    lanes are present for every comparable row.  Missing or contradictory
    evidence is an analysis failure, never an implicit no-go.
    """

    comparable = [row for row in rows if row.get("comparable_to_simulator_timing")]
    unstable = [
        row["cell"]
        for row in rows
        if row.get("live", {}).get("stable_within_five_percent") is not True
    ]
    if unstable:
        raise ValueError(
            "insufficient_evidence: live repeat is unstable for " + ", ".join(unstable)
        )
    missing_diagnostics = [
        row["cell"]
        for row in rows
        if not isinstance(row.get("simulator", {}).get("readiness_schema"), int)
        or not row.get("simulator", {}).get("top_readiness_causes")
    ]
    if missing_diagnostics:
        raise ValueError(
            "insufficient_evidence: required readiness diagnostics are absent for "
            + ", ".join(missing_diagnostics)
        )
    tail_rows = [
        row
        for row in comparable
        if row.get("simulator", {}).get("tails", {}).get("dirty_writeback") is True
        and row.get("simulator", {}).get("tails", {}).get("coherence") is True
    ]
    if not comparable or len(tail_rows) != len(comparable):
        raise ValueError(
            "insufficient_evidence: dirty_writeback/coherence retirement tails "
            "are not observed across every comparable row"
        )
    for row in comparable:
        vector = row["residual"].get("selection_vector", {})
        row["residual"]["selection_score"] = sum(
            abs(int(value))
            for value in vector.values()
            if isinstance(value, int) and not isinstance(value, bool)
        )
    residual_score = sum(row["residual"]["selection_score"] for row in comparable)
    residuals = [row["residual"]["baseline_cycles"] for row in comparable]
    tail_cells = [row["cell"] for row in tail_rows]
    service_cells = [
        row["cell"]
        for row in rows
        if any(
            item.get("cause") == "async_destination_sm_service"
            for item in row.get("simulator", {}).get("top_readiness_causes", [])
        )
    ]
    control_cells = [
        row["cell"]
        for row in rows
        if any(
            item.get("cause") in {"control_stall", "cta_barrier_rendezvous"}
            for item in row.get("simulator", {}).get("top_readiness_causes", [])
        )
    ]
    candidate_ranking = [
        {
            "candidate": "dirty_writeback_coherence_retirement",
            "evidence_cells": tail_cells,
            "selection": "next_discriminator",
            "selection_score": residual_score,
            "reason": (
                f"Typed dirty-writeback/coherence tails occur in {len(tail_cells)}/"
                f"{len(comparable)} comparable rows; the summed repeat-stable residual "
                f"is {residual_score} across the SM-completion, GPC-completion, and "
                "GPC-launch-envelope dimensions; a scalar offset cannot separate "
                "retirement from launch or async/barrier behavior."
            ),
        },
        {
            "candidate": "async_destination_service",
            "evidence_cells": service_cells,
            "selection": "candidate_not_admitted",
            "selection_score": len(service_cells),
            "reason": "A modeled readiness cause is present, but no independent silicon observable isolates it from retirement tails.",
        },
        {
            "candidate": "control_and_barrier_readiness",
            "evidence_cells": control_cells,
            "selection": "candidate_not_admitted",
            "selection_score": len(control_cells),
            "reason": "Readiness counters are modeled diagnostics, not an independent live timing label.",
        },
        {
            "candidate": "launch_envelope",
            "evidence_cells": [row["cell"] for row in rows],
            "selection": "candidate_not_admitted",
            "selection_score": len(rows),
            "reason": "GPC elapsed includes launch work not represented by the direct event frontier; it is not a simulator-local latency.",
        },
    ]
    validation = {
        "rows": len(rows),
        "comparable_rows": len(comparable),
        "repeat_stable_rows": len(rows) - len(unstable),
        "retirement_tail_rows": len(tail_rows),
        "required_diagnostic_rows": len(rows) - len(missing_diagnostics),
        "residual_score_cycles": residual_score,
        "selection_dimensions": [
            "sm_completion_gap_cycles",
            "gpc_completion_gap_cycles",
            "gpc_launch_envelope_cycles",
        ],
        "residual_min_cycles": min(residuals),
        "residual_max_cycles": max(residuals),
        "decision_eligible": True,
    }
    decision = {
        "decision": "no-go",
        "seam": "cycle scalar completion/retirement mapping",
        "selected_diagnostic": "dirty_writeback_coherence_retirement",
            "reason": (
            f"Live residuals are repeat-stable across {len(rows)} rows, while the "
            f"simulator reports dirty-writeback/coherence tails in every comparable "
            f"row ({len(tail_rows)}/{len(comparable)}); the multi-frontier selection "
            "vector includes SM-active completion, GPC completion, and launch-envelope "
            "gaps, but the observations do not identify one architecture-local latency."
        ),
        "next_action": "Run the matched store-free/barrier-free async-source discriminator with an independent retirement/frontier observable before changing simulator timing state.",
        "claim_limits": [
            "No cycle-accuracy claim",
            "No NCU leakage into simulator inputs",
            "No cross-architecture inference",
            "No JADE/HETR or CPU-GPU co-simulation change",
        ],
    }
    return candidate_ranking, validation, decision


def analyze(manifest_path: Path, output_path: Path) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    root = manifest_path.parent
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema_version") != "sm86.live.campaign.v2":
        raise ValueError("unsupported campaign manifest")
    declared_ncu = document.get("ncu_identity")
    if not isinstance(declared_ncu, Mapping) or any(
        declared_ncu.get(key) != value for key, value in TRUSTED_NCU_IDENTITY.items()
    ):
        raise ValueError("manifest NCU identity is not the trusted toolchain")
    ncu_path = Path(TRUSTED_NCU_IDENTITY["path"])
    if not ncu_path.is_file() or _sha256(ncu_path) != TRUSTED_NCU_IDENTITY["sha256"]:
        raise ValueError("trusted NCU executable identity cannot be verified")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("closure analysis requires six cohort cells")

    rows: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("id"), str):
            raise ValueError("malformed cohort cell")
        slug = cell["id"]
        stages = cell.get("stages")
        if not isinstance(stages, Mapping):
            raise ValueError(f"{slug}: stages are missing")
        live = _repeat_metrics(root, cell, stages["ncu"], TRUSTED_NCU_IDENTITY)
        simulator_stage = stages["simulator"]
        outputs = simulator_stage.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 2 or any(
            not isinstance(item, str) for item in outputs
        ):
            raise ValueError(f"{slug}: simulator replay output list is malformed")
        if outputs[0] == outputs[1]:
            raise ValueError(f"{slug}: simulator replay output paths must be distinct")
        replay_one = _campaign_file(root, outputs[0], f"{slug}.replay")
        replay_two = _campaign_file(root, outputs[1], f"{slug}.replay_twin")
        trace = _campaign_dir(root, stages["nvbit"].get("output"), f"{slug}.trace")
        _validate_replay_binding(
            root,
            cell,
            stages["nvbit"],
            stages["static_control"],
            simulator_stage,
            outputs,
        )
        semantic_validation = _validate_replay_semantics(root, cell, trace, outputs)
        sim = _replay_summary(replay_one)
        if replay_one.read_bytes() != replay_two.read_bytes():
            raise ValueError(f"{slug}: simulator twins differ")
        sim["twin_sha256"] = _sha256(replay_two)
        sim["twin_byte_identical"] = True
        sim["semantic_validation"] = semantic_validation
        comparable = slug != "cp_async_canary"
        issue = sim["last_issue_cycles"]
        last_event = sim["last_modeled_event_cycles"]
        baseline_sm = live["baseline"]["sm__cycles_active.max"]
        repeat_sm = live["repeat_2"]["sm__cycles_active.max"]
        baseline_gpc = live["baseline"]["gpc__cycles_elapsed.max"]
        repeat_gpc = live["repeat_2"]["gpc__cycles_elapsed.max"]
        terminal = sim["terminal_lower_bound_cycles"]
        sm_gap = {
            "baseline_vs_last_issue_cycles": baseline_sm - issue,
            "repeat_2_vs_last_issue_cycles": repeat_sm - issue,
            "baseline_vs_last_modeled_event_cycles": (
                None if last_event is None else baseline_sm - last_event
            ),
            "repeat_2_vs_last_modeled_event_cycles": (
                None if last_event is None else repeat_sm - last_event
            ),
            "baseline_vs_completion_lower_bound_cycles": baseline_sm - terminal,
            "repeat_2_vs_completion_lower_bound_cycles": repeat_sm - terminal,
        }
        gpc_gap = {
            "baseline_vs_last_issue_cycles": baseline_gpc - issue,
            "repeat_2_vs_last_issue_cycles": repeat_gpc - issue,
            "baseline_vs_last_modeled_event_cycles": (
                None if last_event is None else baseline_gpc - last_event
            ),
            "repeat_2_vs_last_modeled_event_cycles": (
                None if last_event is None else repeat_gpc - last_event
            ),
            "baseline_vs_completion_lower_bound_cycles": baseline_gpc - terminal,
            "repeat_2_vs_completion_lower_bound_cycles": repeat_gpc - terminal,
            "baseline_launch_envelope_cycles": baseline_gpc - baseline_sm,
            "repeat_2_launch_envelope_cycles": repeat_gpc - repeat_sm,
        }
        selection_vector = {
            "sm_completion_gap_cycles": sm_gap[
                "baseline_vs_completion_lower_bound_cycles"
            ],
            "gpc_completion_gap_cycles": gpc_gap[
                "baseline_vs_completion_lower_bound_cycles"
            ],
            "gpc_launch_envelope_cycles": gpc_gap[
                "baseline_launch_envelope_cycles"
            ],
        }
        row: dict[str, Any] = {
            "cell": slug,
            "comparable_to_simulator_timing": comparable,
            "live": live,
            "simulator": sim,
            "residual": {
                "metric": "multi-frontier SM-active and GPC elapsed comparison",
                "baseline_cycles": sm_gap[
                    "baseline_vs_completion_lower_bound_cycles"
                ],
                "repeat_2_cycles": sm_gap[
                    "repeat_2_vs_completion_lower_bound_cycles"
                ],
                "repeat_delta_cycles": repeat_sm - baseline_sm,
                "interpretable": comparable,
                "sm_active_gaps": sm_gap,
                "gpc_elapsed_gaps": gpc_gap,
                "selection_vector": selection_vector,
            },
            "artifacts": [
                {"path": _label(replay_one), "sha256": _sha256(replay_one)},
                {"path": _label(replay_two), "sha256": _sha256(replay_two)},
            ],
        }
        rows.append(row)

    candidate_ranking, validation, decision = _derive_decision(rows)
    comparable_rows = [row for row in rows if row["comparable_to_simulator_timing"]]
    ranked = sorted(
        comparable_rows,
        key=lambda row: (-row["residual"]["selection_score"], row["cell"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["residual"]["rank_by_multi_frontier_score"] = rank
    report = {
        "schema_version": 1,
        "analysis": {
            "command": [
                ".venv/bin/python",
                "workloads/research_probes/analyze_sm86_live_closure.py",
                "--manifest",
                _label(manifest_path),
                "--output",
                _label(output_path),
            ],
            "manifest": {
                "path": _label(manifest_path),
                "campaign_id": document.get("campaign_id"),
                "candidate_revision": document.get("repository", {}).get("candidate_revision")
                if isinstance(document.get("repository"), Mapping)
                else None,
            },
            "repeat_root": "evidence/raw/ncu_repeat/rep-2",
            "decision": "no-go",
            "selected_diagnostic": decision["selected_diagnostic"],
            "next_discriminator": "matched store-free/barrier-free async-source pair with an independent retirement observable",
            "repeatability_basis": "Two independent application-replay NCU captures per cell; simulator twins are byte-identical.",
            "held_out_discriminator": "not_run; the next pair is intentionally held out to avoid selecting a timing seam from the scored cohort.",
            "ncu_values_used_as_simulator_inputs": False,
            "input_validation": validation,
        },
        "candidate_ranking": candidate_ranking,
        "rows": rows,
        "decision": decision,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote closure report: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return analyze(args.manifest, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"closure analysis: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
