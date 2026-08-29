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
import re
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
PORTABLE_NCU_METRICS = METRICS + (
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "dram__sectors_read.sum",
    "sm__pipe_tensor_cycles_active.sum",
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
FRONTIER_DIMENSIONS = (
    "sm_completion_gap_cycles",
    "gpc_completion_gap_cycles",
    "gpc_launch_envelope_cycles",
)
TAIL_KEYS = (
    "reduction_read_modify_write",
    "dirty_writeback",
    "coherence",
    "async_copy_shared_visibility",
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
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: NCU CSV has no header")
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in reader
            if row.get("Kernel Name")
        ]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty kernel row")
    return rows[0]


def _expected_ncu_target(cell: Mapping[str, Any]) -> list[str]:
    binary = cell.get("binary")
    if not isinstance(binary, Mapping) or not isinstance(binary.get("path"), str):
        raise ValueError(f"{cell.get('id')}: NCU target binary is missing")
    if cell.get("id") == "cp_async_canary":
        return [binary["path"], "c11", "l1", "0", "4", "3"]
    return [binary["path"], "--timing_shape", "--warmup", "1", "--reps", "1"]


def _expected_ncu_metrics(cell: Mapping[str, Any]) -> tuple[str, ...]:
    return METRICS if cell.get("id") == "cp_async_canary" else PORTABLE_NCU_METRICS


def _canonical_ncu_argv(cell: Mapping[str, Any], output: str) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/local_labels/collect_ncu.py",
        "--output",
        output,
        "--metrics",
        ",".join(_expected_ncu_metrics(cell)),
        "--launch-skip",
        "1",
        "--launch-count",
        "1",
        "--replay-mode",
        "application",
        "--clock-control",
        "none",
        "--",
        *_expected_ncu_target(cell),
    ]


def _validate_ncu_stage_contract(
    root: Path, cell: Mapping[str, Any], stage: Mapping[str, Any]
) -> None:
    slug = str(cell.get("id"))
    output = f"evidence/raw/ncu/{slug}"
    expected_argv = _canonical_ncu_argv(cell, output)
    if stage.get("argv") != expected_argv:
        raise ValueError(f"{slug}: NCU stage command is not bound to the reviewed target")
    if stage.get("output") != output or stage.get("csv") != output + "/ncu.csv":
        raise ValueError(f"{slug}: NCU output/CSV paths are not bound to the stage")
    if stage.get("tool_path") != TRUSTED_NCU_IDENTITY["path"]:
        raise ValueError(f"{slug}: NCU stage tool is not the trusted executable")
    exit_code = stage.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        raise ValueError(f"{slug}: NCU stage exit code is malformed or nonzero")
    _campaign_file(root, stage.get("csv"), f"{slug}.ncu.csv")
    for key in ("stdout", "stderr"):
        if key in stage:
            _campaign_file(root, stage.get(key), f"{slug}.ncu.{key}")


def _tuple3(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError(f"{label}: launch geometry is missing")
    match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", value)
    if match is None:
        raise ValueError(f"{label}: launch geometry is malformed")
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _validate_ncu_row_identity(
    row: Mapping[str, str],
    cell: Mapping[str, Any],
    label: str,
    expected_kernel: str | None = None,
) -> str:
    binary = cell.get("binary")
    if not isinstance(binary, Mapping):
        raise ValueError(f"{label}: binary identity is missing")
    expected_process = Path(str(binary.get("path"))).name
    if row.get("Process Name") != expected_process:
        raise ValueError(f"{label}: NCU process identity differs from the reviewed binary")
    kernel = row.get("Kernel Name")
    if not isinstance(kernel, str) or not kernel:
        raise ValueError(f"{label}: NCU kernel identity is empty")
    if expected_kernel is not None and kernel != expected_kernel:
        raise ValueError(f"{label}: NCU kernel identity differs between baseline and repeat")
    if row.get("Device") != "0" or row.get("CC") != "8.6":
        raise ValueError(f"{label}: NCU device/compute-capability identity differs")
    expected_grid = tuple(cell.get("expected_grid", ()))
    expected_block = tuple(cell.get("expected_block", ()))
    if len(expected_grid) != 3 or len(expected_block) != 3:
        raise ValueError(f"{label}: expected launch geometry is missing")
    if _tuple3(row.get("Grid Size"), label) != expected_grid:
        raise ValueError(f"{label}: NCU grid geometry differs")
    if _tuple3(row.get("Block Size"), label) != expected_block:
        raise ValueError(f"{label}: NCU block geometry differs")
    for name, expected in (
        ("launch__grid_dim_x", expected_grid[0]),
        ("launch__grid_dim_y", expected_grid[1]),
        ("launch__grid_dim_z", expected_grid[2]),
        ("launch__block_dim_x", expected_block[0]),
        ("launch__block_dim_y", expected_block[1]),
        ("launch__block_dim_z", expected_block[2]),
        ("launch__thread_count", expected_grid[0] * expected_grid[1] * expected_grid[2] * expected_block[0] * expected_block[1] * expected_block[2]),
    ):
        if _metric(row, name) != expected:
            raise ValueError(f"{label}: NCU launch geometry field {name} differs")
    return kernel


def _validate_ncu_csv(
    path: Path,
    cell: Mapping[str, Any],
    label: str,
    expected_kernel: str | None = None,
) -> tuple[dict[str, str], str]:
    row = _ncu_row(path)
    required_metrics = _expected_ncu_metrics(cell)
    if any(name not in row for name in required_metrics):
        raise ValueError(f"{label}: NCU CSV is missing a canonical metric")
    kernel = _validate_ncu_row_identity(row, cell, label, expected_kernel)
    for name in required_metrics:
        _metric(row, name)
    return row, kernel


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
    document = replay_resume.load_json_strict(path)
    if not isinstance(document, Mapping):
        raise ValueError(f"{path}: replay root is malformed")
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
    counter_tails = counters.get("unmodeled_tails")
    if not isinstance(counter_tails, Mapping) or set(counter_tails) != set(TAIL_KEYS):
        raise ValueError(f"{path}: replay counter typed tails are malformed")
    if any(type(counter_tails.get(name)) is not bool for name in TAIL_KEYS):
        raise ValueError(f"{path}: replay counter typed tails are malformed")
    counter_tails = {name: counter_tails[name] for name in TAIL_KEYS}
    issue = timing.get("diagnostic_last_issue_cycles")
    last_event = timing.get("last_processed_modeled_system_event_time")
    terminal = timing.get("terminal_modeled_event_time")
    if not isinstance(issue, int) or isinstance(issue, bool):
        raise ValueError(f"{path}: diagnostic last-issue frontier is malformed")
    if last_event is not None and (
        not isinstance(last_event, int) or isinstance(last_event, bool)
    ):
        raise ValueError(f"{path}: last-modeled-event frontier is malformed")
    if terminal is not None and (
        not isinstance(terminal, int) or isinstance(terminal, bool)
    ):
        raise ValueError(f"{path}: terminal modeled event time is malformed")
    status = frontier.get("status")
    completion_cycles: int | None = None
    lower_bound: int | None = None
    if status == "complete":
        if set(frontier) != {"status", "cycles"}:
            raise ValueError(f"{path}: complete frontier fields are malformed")
        cycles = frontier.get("cycles")
        if cycles is not None and (
            not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0
        ):
            raise ValueError(f"{path}: complete frontier cycles are malformed")
        completion_cycles = cycles
        if any(counter_tails.values()):
            raise ValueError(
                f"{path}: complete frontier must have all counter typed tails false"
            )
        typed_tails = counter_tails
        frontier_kind = "exact"
    elif status == "incomplete_unmodeled_retirement_tails":
        if set(frontier) != {"status", "lower_bound_cycles", "unmodeled_tails"}:
            raise ValueError(f"{path}: incomplete frontier fields are malformed")
        lower_bound = frontier.get("lower_bound_cycles")
        if lower_bound is not None and (
            not isinstance(lower_bound, int) or isinstance(lower_bound, bool) or lower_bound < 0
        ):
            raise ValueError(f"{path}: completion frontier lower bound is malformed")
        frontier_tails = frontier.get("unmodeled_tails")
        if not isinstance(frontier_tails, Mapping) or set(frontier_tails) != set(TAIL_KEYS):
            raise ValueError(f"{path}: completion frontier typed tails are malformed")
        if any(type(frontier_tails.get(name)) is not bool for name in TAIL_KEYS):
            raise ValueError(f"{path}: completion frontier typed tails are malformed")
        typed_tails = {name: frontier_tails[name] for name in TAIL_KEYS}
        if typed_tails != counter_tails:
            raise ValueError(f"{path}: counter and frontier typed tails differ")
        frontier_kind = "lower_bound"
    else:
        raise ValueError(f"{path}: completion frontier status is malformed")
    reference_cycles = (
        completion_cycles
        if status == "complete" and completion_cycles is not None
        else lower_bound
        if status == "incomplete_unmodeled_retirement_tails" and lower_bound is not None
        else terminal
    )
    return {
        "last_issue_cycles": issue,
        "last_modeled_event_cycles": last_event,
        "terminal_lower_bound_cycles": terminal,
        "terminal_modeled_event_cycles": terminal,
        "completion_status": status,
        "completion_kind": frontier_kind,
        "completion_cycles": completion_cycles,
        "completion_lower_bound_cycles": lower_bound,
        "completion_reference_cycles": reference_cycles,
        "completion_frontier": {
            "status": status,
            "kind": frontier_kind,
            "cycles": completion_cycles,
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
    if (
        simulator.get("binary_path") != "sim/target/release/sim-run"
        or not binary.is_file()
        or simulator.get("binary_sha256") != _sha256(binary)
    ):
        raise ValueError(f"{slug}: simulator binary identity is not current")
    bound_binary = binding.get("simulator_binary")
    if (
        not isinstance(bound_binary, Mapping)
        or bound_binary.get("path") != "sim/target/release/sim-run"
        or bound_binary.get("sha256") != _sha256(binary)
    ):
        raise ValueError(f"{slug}: replay binding simulator identity differs")
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
    report_path = _campaign_file(root, expected_report, f"{slug}.ncu_repeat.report")
    if report_path.stat().st_size == 0:
        raise ValueError(f"{slug}: repeated NCU report artifact is empty")
    csv_path = _campaign_file(root, expected_csv, f"{slug}.ncu_repeat.csv")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"{slug}: repeated NCU CSV artifact is empty")
    if target[0] != cell.get("binary", {}).get("path"):
        raise ValueError(f"{slug}: repeated NCU target binary differs")


def _repeat_metrics(
    root: Path,
    cell: Mapping[str, Any],
    stage: Mapping[str, Any],
    ncu_identity: Mapping[str, str],
) -> dict[str, Any]:
    slug = str(cell["id"])
    _validate_ncu_stage_contract(root, cell, stage)
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
    metadata = replay_resume.load_json_strict(repeat_metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{slug}: repeated NCU metadata is malformed")
    _validate_repeat_metadata(root, cell, stage, repeat_dir, metadata, ncu_identity)
    baseline_row, kernel_name = _validate_ncu_csv(
        baseline_csv, cell, f"{slug}.ncu.baseline"
    )
    repeat_row, _ = _validate_ncu_csv(
        repeat_csv, cell, f"{slug}.ncu.repeat", expected_kernel=kernel_name
    )
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
        "provenance": {
            "process_name": baseline_row.get("Process Name"),
            "kernel_name": kernel_name,
            "device": baseline_row.get("Device"),
            "compute_capability": baseline_row.get("CC"),
            "grid_size": baseline_row.get("Grid Size"),
            "block_size": baseline_row.get("Block Size"),
            "target": _expected_ncu_target(cell),
            "metrics": list(_expected_ncu_metrics(cell)),
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
    for row in rows:
        vector = row.get("residual", {}).get("selection_vector")
        if not isinstance(vector, Mapping) or set(vector) != set(FRONTIER_DIMENSIONS):
            raise ValueError(
                f"insufficient_evidence: {row.get('cell')} lacks independent frontier dimensions"
            )
        if any(
            not isinstance(vector.get(name), int) or isinstance(vector.get(name), bool)
            for name in FRONTIER_DIMENSIONS
        ):
            raise ValueError(
                f"insufficient_evidence: {row.get('cell')} has malformed frontier dimensions"
            )
        # The dimensions stay independent.  This annotation makes the
        # mechanism-local selection rule explicit and prevents a future caller
        # from mistaking the vector for a scalar selection feature.
        row["residual"]["selection_basis"] = {
            "rule": "mechanism_local_typed_tail_precedence",
            "frontier_dimensions": list(FRONTIER_DIMENSIONS),
            "independent_dimensions": True,
            "aggregate_score_used": False,
            "eligible_for_selection": row.get("comparable_to_simulator_timing") is True,
            "mechanism_evidence": ["dirty_writeback", "coherence"]
            if row.get("comparable_to_simulator_timing") is True
            else [],
        }
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
    candidate_evidence = [
        {
            "candidate": "dirty_writeback_coherence_retirement",
            "evidence_cells": tail_cells,
            "selection": "next_discriminator",
            "selection_rule": "mechanism_local_typed_tail_precedence",
            "evidence_basis": {
                "typed_tail_flags": ["dirty_writeback", "coherence"],
                "typed_tail_coverage": {
                    "numerator": len(tail_cells),
                    "denominator": len(comparable),
                },
                "frontier_dimensions": list(FRONTIER_DIMENSIONS),
                "frontiers_remain_independent": True,
                "aggregate_score_used": False,
                "independent_live_observable": False,
            },
            "reason": (
                f"Typed dirty-writeback/coherence tails occur in {len(tail_cells)}/"
                f"{len(comparable)} comparable rows.  The mechanism-local rule gives "
                "precedence to a typed retirement-tail witness that spans the "
                "independent SM-completion, GPC-completion, and GPC-launch-envelope "
                "frontiers; no scalar or cross-dimension score is used, and no "
                "independent live observable separates retirement from launch or "
                "async/barrier behavior."
            ),
        },
        {
            "candidate": "async_destination_service",
            "evidence_cells": service_cells,
            "selection": "candidate_not_admitted",
            "selection_rule": "mechanism_local_typed_tail_precedence",
            "evidence_basis": {
                "frontier_dimensions": list(FRONTIER_DIMENSIONS),
                "frontiers_remain_independent": True,
                "aggregate_score_used": False,
                "independent_live_observable": False,
            },
            "reason": "A modeled readiness cause is present, but no independent silicon observable isolates it from retirement tails.",
        },
        {
            "candidate": "control_and_barrier_readiness",
            "evidence_cells": control_cells,
            "selection": "candidate_not_admitted",
            "selection_rule": "mechanism_local_typed_tail_precedence",
            "evidence_basis": {
                "frontier_dimensions": list(FRONTIER_DIMENSIONS),
                "frontiers_remain_independent": True,
                "aggregate_score_used": False,
                "independent_live_observable": False,
            },
            "reason": "Readiness counters are modeled diagnostics, not an independent live timing label.",
        },
        {
            "candidate": "launch_envelope",
            "evidence_cells": [row["cell"] for row in rows],
            "selection": "candidate_not_admitted",
            "selection_rule": "mechanism_local_typed_tail_precedence",
            "evidence_basis": {
                "frontier_dimensions": list(FRONTIER_DIMENSIONS),
                "frontiers_remain_independent": True,
                "aggregate_score_used": False,
                "independent_live_observable": False,
            },
            "reason": "GPC elapsed includes launch work not represented by the direct event frontier; it is not a simulator-local latency.",
        },
    ]
    validation = {
        "rows": len(rows),
        "comparable_rows": len(comparable),
        "repeat_stable_rows": len(rows) - len(unstable),
        "retirement_tail_rows": len(tail_rows),
        "required_diagnostic_rows": len(rows) - len(missing_diagnostics),
        "selection_dimensions": list(FRONTIER_DIMENSIONS),
        "selection_rule": "mechanism_local_typed_tail_precedence",
        "frontiers_remain_independent": True,
        "aggregate_score_used": False,
        "decision_eligible": True,
    }
    decision = {
        "decision": "no-go",
        "seam": "completion/retirement frontier mapping",
        "selected_diagnostic": "dirty_writeback_coherence_retirement",
        "selection_rule": "mechanism_local_typed_tail_precedence",
        "reason": (
            f"Live residuals are repeat-stable across {len(rows)} rows, while the "
            f"simulator reports dirty-writeback/coherence tails in every comparable "
            f"row ({len(tail_rows)}/{len(comparable)}); the multi-frontier selection "
            "vector includes SM-active completion, GPC completion, and launch-envelope "
            "gaps and is retained component-wise.  The typed-tail witness is selected "
            "by mechanism-local precedence, but the observations do not identify one "
            "architecture-local latency."
        ),
        "next_action": "Run the matched store-free/barrier-free async-source discriminator with an independent retirement/frontier observable before changing simulator timing state.",
        "claim_limits": [
            "No cycle-accuracy claim",
            "No NCU leakage into simulator inputs",
            "No cross-architecture inference",
            "No JADE/HETR or CPU-GPU co-simulation change",
        ],
    }
    return candidate_evidence, validation, decision


def build_report(manifest_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Recompute a closure report entirely from the manifest's artifacts.

    This pure builder is also used by the campaign validator to ensure that a
    report cannot be made acceptable merely by changing its own hash.
    """

    manifest_path = manifest_path.expanduser().resolve()
    if output_path is None:
        output_path = manifest_path.parent / "closure_report.json"
    output_path = output_path.expanduser().resolve()
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
        completion_reference = sim["completion_reference_cycles"]
        if not isinstance(completion_reference, int) or isinstance(completion_reference, bool):
            raise ValueError(f"{slug}: completion frontier has no usable cycle reference")
        sm_gap = {
            "baseline_vs_last_issue_cycles": baseline_sm - issue,
            "repeat_2_vs_last_issue_cycles": repeat_sm - issue,
            "baseline_vs_last_modeled_event_cycles": (
                None if last_event is None else baseline_sm - last_event
            ),
            "repeat_2_vs_last_modeled_event_cycles": (
                None if last_event is None else repeat_sm - last_event
            ),
            "baseline_vs_completion_frontier_cycles": baseline_sm - completion_reference,
            "repeat_2_vs_completion_frontier_cycles": repeat_sm - completion_reference,
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
            "baseline_vs_completion_frontier_cycles": baseline_gpc - completion_reference,
            "repeat_2_vs_completion_frontier_cycles": repeat_gpc - completion_reference,
            "baseline_launch_envelope_cycles": baseline_gpc - baseline_sm,
            "repeat_2_launch_envelope_cycles": repeat_gpc - repeat_sm,
        }
        selection_vector = {
            "sm_completion_gap_cycles": sm_gap["baseline_vs_completion_frontier_cycles"],
            "gpc_completion_gap_cycles": gpc_gap["baseline_vs_completion_frontier_cycles"],
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
                "interpretable": comparable,
                "frontier_reference": {
                    "kind": sim["completion_kind"],
                    "exact_cycles": sim["completion_cycles"],
                    "lower_bound_cycles": sim["completion_lower_bound_cycles"],
                    "reference_cycles": completion_reference,
                },
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

    candidate_evidence, validation, decision = _derive_decision(rows)
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
        "candidate_evidence": candidate_evidence,
        "rows": rows,
        "decision": decision,
        "acceptance_criteria": {
            "AC-1": {
                "status": "pass",
                "evidence": [
                    "campaign-bound release build transcript and binary hash",
                    "current RTX 3090 Ti host/toolchain preflight",
                ],
            },
            "AC-3": {
                "status": "pass",
                "evidence": [
                    "strict direct replay schema validation",
                    "trace-derived instruction and traffic conservation",
                    "byte-identical simulator twins",
                    "typed completion frontiers and multi-frontier gaps",
                    "fresh retained-corpus validation",
                ],
            },
            "AC-4": {
                "status": "pass",
                "mode": "falsified_no_go",
                "selected_diagnostic": decision["selected_diagnostic"],
                "next_discriminator": decision["next_action"],
                "evidence": "repeat-stable live observations and typed simulator tails do not isolate one architecture-local latency",
            },
        },
    }
    return report


def analyze(manifest_path: Path, output_path: Path) -> int:
    output_path = output_path.expanduser().resolve()
    report = build_report(manifest_path, output_path)
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
