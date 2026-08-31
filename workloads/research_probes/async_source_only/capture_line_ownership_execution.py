#!/usr/bin/env python3
"""Capture static, direct, NVBit, and replay evidence for line ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import capture_multiline_execution as support


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_line_ownership_20260831_v1"
)
AGGREGATE_REPORT = REPO_ROOT / "data/local_labels/sm86_async_line_ownership_report.json"
BINARY = HERE / "async_source_line_ownership"
SOURCE = HERE / "async_source_line_ownership.cu"
MAKEFILE = HERE / "Makefile.line_ownership"
NVBIT_CAPTURE = (
    REPO_ROOT / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
TRACER = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"

ORIENTATIONS = ("first", "second")
VARIANTS = ("early_fanout", "late_fanout")
REPETITIONS = tuple(range(1, 11))
TRACE_GROUPS = 64
DIRECT_GROUPS = 8192

# The accepted schema-2 static implementation is intentionally shared. Its
# build command derives the make target from BINARY.name, so binding these two
# inputs gives this probe the same stale-source/tool behavior without copying
# another manifest framework.
support.BINARY = BINARY
support.SOURCE = SOURCE
support.MAKEFILE = MAKEFILE

CaptureError = support.CaptureError


@dataclass(frozen=True)
class Cell:
    orientation: str

    def __post_init__(self) -> None:
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"unsupported orientation: {self.orientation}")

    @property
    def orientation_name(self) -> str:
        return f"{self.orientation}_line_early"

    @property
    def short(self) -> str:
        return "first" if self.orientation == "first" else "second"

    @property
    def direct_name(self) -> str:
        return f"async_source_line_ownership_{self.orientation_name}_g{DIRECT_GROUPS}"

    def endpoint_name(self, variant: str) -> str:
        if variant not in VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        return (
            f"async_source_line_ownership_{variant}_{self.orientation_name}_"
            f"g{TRACE_GROUPS}"
        )


def retained_binary(root: Path) -> dict[str, Any]:
    return support.retained_binary(root)


def require_full_campaign_authorization(
    root: Path,
    binary: Mapping[str, Any],
    report_path: Path = AGGREGATE_REPORT,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise CaptureError(
            f"full campaign requires a smoke-pass aggregate: {report_path}"
        )
    report = support.read_json(report_path)
    expected_scalars = {
        "schema_version": 1,
        "assay": "sm86_async_source_line_ownership",
        "mode": "smoke_only",
        "status": "source_line_ownership_smoke_passed",
        "root": str(root.resolve()),
        "binary": dict(binary),
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "missing_artifact_count": 0,
        "validation_error_count": 0,
    }
    mismatches = {
        field: {"observed": report.get(field), "expected": value}
        for field, value in expected_scalars.items()
        if report.get(field) != value
    }
    smoke_gates = report.get("smoke_gates")
    criteria = report.get("criteria")
    if (
        not isinstance(smoke_gates, dict)
        or not smoke_gates
        or not all(value is True for value in smoke_gates.values())
    ):
        mismatches["smoke_gates"] = {
            "observed": smoke_gates,
            "expected": "all true",
        }
    if not isinstance(criteria, dict) or criteria.get("smoke_passed") is not True:
        mismatches["criteria.smoke_passed"] = {
            "observed": None if not isinstance(criteria, dict) else criteria.get("smoke_passed"),
            "expected": True,
        }
    if mismatches:
        raise CaptureError(
            f"full campaign is not authorized by this root/binary smoke: {mismatches}"
        )
    return report


def trace_payload_manifest_path(destination: Path) -> Path:
    return destination / "payload_manifest.json"


def trace_payload_files(destination: Path) -> list[Path]:
    paths = [destination / "dynamic_trace.pb", destination / "stats.csv"]
    paths.extend(sorted((destination / "threadblocks").rglob("*.pb")))
    if (destination / "extra_info").is_dir():
        paths.extend(
            sorted(
                path
                for path in (destination / "extra_info").rglob("*")
                if path.is_file()
            )
        )
    unique = sorted(set(paths), key=lambda path: path.relative_to(destination).as_posix())
    if not unique:
        raise CaptureError(f"{destination}: no trace payload files")
    for path in unique:
        if not path.is_file() or path.is_symlink():
            raise CaptureError(f"{destination}: invalid trace payload {path}")
    return unique


def current_trace_payload_manifest(destination: Path) -> dict[str, Any]:
    entries = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": support.sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in trace_payload_files(destination)
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "kind": "line_ownership_trace_payload_manifest",
        "entries": entries,
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_trace_payload_manifest(destination: Path) -> Path:
    path = trace_payload_manifest_path(destination)
    support.atomic_write_json(path, current_trace_payload_manifest(destination))
    return path


def validate_trace_payload_manifest(destination: Path) -> dict[str, Any]:
    path = trace_payload_manifest_path(destination)
    observed = support.read_json(path)
    expected = current_trace_payload_manifest(destination)
    if observed != expected:
        raise CaptureError(f"{path}: trace payload identity drift")
    return observed


def add_run_manifest_output(
    manifest_path: Path, name: str, output_path: Path
) -> None:
    document = support.read_json(manifest_path)
    outputs = document.get("outputs")
    if not isinstance(outputs, dict):
        raise CaptureError(f"{manifest_path}: missing outputs")
    outputs[name] = support.file_identity(output_path)
    support.atomic_write_json(manifest_path, document)


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def cells(smoke: bool, selector: str | None) -> list[Cell]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --cells are mutually exclusive")
        return [Cell("first")]
    if selector is None:
        return [Cell(value) for value in ORIENTATIONS]
    result: list[Cell] = []
    for raw in selector.split(","):
        value = raw.strip().removesuffix("_line_early")
        if value not in ORIENTATIONS:
            raise CaptureError("--cells accepts first,second")
        cell = Cell(value)
        if cell not in result:
            result.append(cell)
    if not result:
        raise CaptureError("--cells selected no cells")
    return result


def repetitions(smoke: bool, selector: str | None) -> list[int]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --repeats are mutually exclusive")
        return [1]
    if selector is None:
        return list(REPETITIONS)
    result: list[int] = []
    for raw in selector.split(","):
        try:
            value = int(raw.strip())
        except ValueError as error:
            raise CaptureError(f"invalid repetition: {raw!r}") from error
        if value not in REPETITIONS:
            raise CaptureError("repetitions must be in 1..10")
        if value not in result:
            result.append(value)
    if not result:
        raise CaptureError("--repeats selected no repetitions")
    return result


def target_arguments(
    cell: Cell,
    *,
    groups: int,
    markers: str,
    order: str | None = None,
    variant: str | None = None,
) -> list[str]:
    if (order is None) == (variant is None):
        raise ValueError("select exactly one paired order or endpoint variant")
    result = [
        "--groups",
        str(groups),
        "--orientation",
        cell.orientation_name,
    ]
    if order is not None:
        result.extend(("--order", order))
    else:
        result.extend(("--variant", str(variant)))
    result.extend(("--markers", markers))
    return result


def common_target_contract(cell: Cell, groups: int, mode: str) -> dict[str, Any]:
    return {
        "workload": "async_source_line_ownership",
        "route": "l2",
        "mode": mode,
        "orientation": cell.orientation_name,
        "groups": groups,
        "warmup_groups": 64,
        "lead_clock_samples": 16,
        "source_cohorts_per_copy": 8,
        "aligned_source_cohorts_per_copy": 4,
        "split_source_cohorts_per_copy": 4,
        "source_sectors_per_cohort": 4,
        "payload_bytes_per_cohort": 64,
        "scalar_ldg_instructions_per_copy": 1,
        "scalar_ldg_sectors_per_copy": 24,
        "scalar_preload_lines_per_copy": 8,
        "ldgsts_instructions_per_copy": 1,
        "ldgsts_sectors_per_copy": 32,
        "unique_ldgsts_lines_per_copy": 12,
        "payload_wavefronts_per_copy": 32,
        "projected_bypass_wavefronts_per_copy": 32,
        "destination_lines_per_copy": 20,
        "local_device_lts_lookups_per_copy": 56,
        "source_pool_single_allocation": True,
        "independent_source_arms": True,
        "source_vector_preserved": True,
        "preload_vector_preserved": True,
        "destination_multiset_preserved": True,
        "ownership_permutation_preserved": True,
        "static_ldgsts_positions_preserved": True,
        "result_store_after_intervals": True,
        "device": "NVIDIA GeForce RTX 3090 Ti",
        "compute_capability": "8.6",
        "correct": True,
    }


def validate_arm(value: Any, groups: int, context: str) -> None:
    if not isinstance(value, dict) or value.get("correct") is not True:
        raise CaptureError(f"{context}: invalid result arm")
    if value.get("mismatches") != 0 or value.get("sum_consistent") is not True:
        raise CaptureError(f"{context}: correctness/sum invariant failed")
    for field in ("issue_cycles", "wait_cycles", "total_cycles"):
        observed = value.get(field)
        if not isinstance(observed, int) or observed < 0:
            raise CaptureError(f"{context}: invalid {field}")
    if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
        raise CaptureError(f"{context}: cycle sum mismatch")
    if groups <= 0:
        raise CaptureError(f"{context}: invalid group count")


def validate_target(
    payload: Mapping[str, Any],
    cell: Cell,
    *,
    groups: int,
    mode: str,
    order: str = "ab",
    variant: str | None = None,
) -> None:
    expected = common_target_contract(cell, groups, mode)
    mismatches = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise CaptureError(f"target contract mismatch: {mismatches}")
    if mode == "pair":
        if payload.get("order") != order:
            raise CaptureError("target pair order mismatch")
        for arm in VARIANTS:
            validate_arm(payload.get(arm), groups, f"target.{arm}")
    else:
        if payload.get("variant") != variant:
            raise CaptureError("target endpoint variant mismatch")
        validate_arm(payload.get("result"), groups, f"target.{variant}")


def direct_command(cell: Cell, repetition: int) -> list[str]:
    return [
        str(BINARY),
        *target_arguments(
            cell,
            groups=DIRECT_GROUPS,
            markers="off",
            order=expected_order(repetition),
        ),
    ]


def validate_direct_provenance(
    destination: Path,
    cell: Cell,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = destination / "result.json"
    command_path = destination / "command.json"
    command_document = support.read_json(command_path)
    timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(timeout, int) or timeout < 1:
        raise CaptureError(f"{command_path}: invalid timeout")
    manifest = destination / "direct.run.json"
    support.validate_run_manifest(
        manifest,
        command=direct_command(cell, repetition),
        timeout=timeout,
        stdout_path=result_path,
        stderr_path=destination / "stderr.txt",
        input_paths={"probe_binary": BINARY},
        output_paths={"result": result_path},
        environment_value=support.environment(),
    )
    expected = {
        "schema_version": 1,
        "cell": cell.orientation_name,
        "repetition": repetition,
        "order": expected_order(repetition),
        "configured_timeout_seconds": timeout,
        "command": direct_command(cell, repetition),
        "binary": dict(binary),
        "run_manifest": support.file_identity(manifest),
    }
    if command_document != expected:
        raise CaptureError(f"{command_path}: direct binding mismatch")
    return {
        "result": support.file_identity(result_path),
        "command": support.file_identity(command_path),
        "run_manifest": support.file_identity(manifest),
        "stderr": support.file_identity(destination / "stderr.txt"),
    }


def validate_direct(
    destination: Path,
    cell: Cell,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    validate_direct_provenance(destination, cell, repetition, binary)
    payload = support.read_json(destination / "result.json")
    validate_target(
        payload,
        cell,
        groups=DIRECT_GROUPS,
        mode="pair",
        order=expected_order(repetition),
    )
    return payload


def capture_direct(
    root: Path,
    selected_cells: Sequence[Cell],
    selected_repetitions: Sequence[int],
    timeout: int,
) -> None:
    binary = retained_binary(root)
    for cell in selected_cells:
        for repetition in selected_repetitions:
            destination = root / "direct" / cell.direct_name / f"repeat-{repetition}"
            result_path = destination / "result.json"
            if result_path.is_file():
                validate_direct_provenance(destination, cell, repetition, binary)
                print(f"direct existing: {cell.short} r{repetition}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = direct_command(cell, repetition)
            manifest = destination / "direct.run.json"
            support.recorded_run(
                command,
                timeout=timeout,
                manifest_path=manifest,
                stdout_path=result_path,
                stderr_path=destination / "stderr.txt",
                input_paths={"probe_binary": BINARY},
                output_paths={"result": result_path},
                environment_value=support.environment(),
            )
            support.atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "cell": cell.orientation_name,
                    "repetition": repetition,
                    "order": expected_order(repetition),
                    "configured_timeout_seconds": timeout,
                    "command": command,
                    "binary": binary,
                    "run_manifest": support.file_identity(manifest),
                },
            )
            validate_direct_provenance(destination, cell, repetition, binary)
            print(f"direct: {cell.short} r{repetition}", flush=True)


def nvbit_commands(
    destination: Path, cell: Cell, variant: str, target_timeout: int
) -> tuple[list[str], list[str]]:
    capture = [
        sys.executable,
        str(NVBIT_CAPTURE),
        "--output",
        str(destination),
        "--cell",
        cell.endpoint_name(variant),
        "--binary",
        str(BINARY),
        "--mode",
        "ncu",
        "--gpu",
        "0",
        "--timeout",
        str(target_timeout),
        "--target-args",
        *target_arguments(
            cell,
            groups=TRACE_GROUPS,
            markers="on",
            variant=variant,
        ),
    ]
    wrapper = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "Delegate=yes",
        *capture,
    ]
    return capture, wrapper


def validate_nvbit(
    root: Path,
    destination: Path,
    cell: Cell,
    variant: str,
    binary: Mapping[str, Any],
) -> None:
    name = cell.endpoint_name(variant)
    complete = support.read_json(destination / "complete.json")
    binding = support.read_json(destination / "campaign_binding.json")
    if complete.get("complete") is not True or complete.get("key") != name:
        raise CaptureError(f"{destination}: invalid NVBit completion")
    if complete.get("binary_sha256") != binary["sha256"]:
        raise CaptureError(f"{destination}: NVBit binary mismatch")
    if binding.get("cell") != name or binding.get("output_contract", {}).get("passed") is not True:
        raise CaptureError(f"{destination}: invalid NVBit binding")
    run_root = root / "runs/nvbit" / name
    command_path = run_root / "command.json"
    command_document = support.read_json(command_path)
    target_timeout = command_document.get("configured_target_timeout_seconds")
    outer_timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(target_timeout, int) or target_timeout < 1:
        raise CaptureError(f"{command_path}: invalid target timeout")
    if not isinstance(outer_timeout, int) or outer_timeout < target_timeout:
        raise CaptureError(f"{command_path}: invalid wrapper timeout")
    capture, wrapper = nvbit_commands(destination, cell, variant, target_timeout)
    manifest = run_root / "wrapper.run.json"
    payload_manifest = trace_payload_manifest_path(destination)
    validate_trace_payload_manifest(destination)
    outputs = {
        "binding": destination / "campaign_binding.json",
        "complete": destination / "complete.json",
        "payload_manifest": payload_manifest,
        "stats": destination / "stats.csv",
        "trace": destination / "dynamic_trace.pb",
    }
    support.validate_run_manifest(
        manifest,
        command=wrapper,
        timeout=outer_timeout,
        stdout_path=run_root / "wrapper.stdout.txt",
        stderr_path=run_root / "wrapper.stderr.txt",
        input_paths={
            "capture_helper": NVBIT_CAPTURE,
            "probe_binary": BINARY,
            "tracer": TRACER,
        },
        output_paths=outputs,
        environment_value=support.environment(),
    )
    expected = {
        "schema_version": 1,
        "cell": name,
        "variant": variant,
        "orientation": cell.orientation,
        "configured_target_timeout_seconds": target_timeout,
        "configured_timeout_seconds": outer_timeout,
        "capture_command": capture,
        "command": wrapper,
        "binary": dict(binary),
        "run_manifest": support.file_identity(manifest),
        "outputs": support.identities(outputs),
    }
    if command_document != expected:
        raise CaptureError(f"{command_path}: NVBit wrapper binding mismatch")
    stdout_payload = None
    for line in reversed((destination / "stdout.txt").read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            stdout_payload = value
            break
    if stdout_payload is None:
        raise CaptureError(f"{destination}: no target JSON")
    validate_target(
        stdout_payload,
        cell,
        groups=TRACE_GROUPS,
        mode="endpoint",
        variant=variant,
    )


def capture_nvbit(root: Path, selected_cells: Sequence[Cell], timeout: int) -> None:
    binary = retained_binary(root)
    for cell in selected_cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            destination = root / "nvbit" / name
            if (destination / "complete.json").is_file():
                validate_nvbit(root, destination, cell, variant, binary)
                print(f"nvbit existing: {name}")
                continue
            run_root = root / "runs/nvbit" / name
            run_root.mkdir(parents=True, exist_ok=True)
            target_timeout = timeout
            outer_timeout = timeout + 120
            capture, wrapper = nvbit_commands(destination, cell, variant, target_timeout)
            manifest = run_root / "wrapper.run.json"
            outputs = {
                "binding": destination / "campaign_binding.json",
                "complete": destination / "complete.json",
                "stats": destination / "stats.csv",
                "trace": destination / "dynamic_trace.pb",
            }
            support.recorded_run(
                wrapper,
                timeout=outer_timeout,
                manifest_path=manifest,
                stdout_path=run_root / "wrapper.stdout.txt",
                stderr_path=run_root / "wrapper.stderr.txt",
                input_paths={
                    "capture_helper": NVBIT_CAPTURE,
                    "probe_binary": BINARY,
                    "tracer": TRACER,
                },
                output_paths=outputs,
                environment_value=support.environment(),
            )
            payload_manifest = write_trace_payload_manifest(destination)
            add_run_manifest_output(
                manifest, "payload_manifest", payload_manifest
            )
            outputs["payload_manifest"] = payload_manifest
            support.atomic_write_json(
                run_root / "command.json",
                {
                    "schema_version": 1,
                    "cell": name,
                    "variant": variant,
                    "orientation": cell.orientation,
                    "configured_target_timeout_seconds": target_timeout,
                    "configured_timeout_seconds": outer_timeout,
                    "capture_command": capture,
                    "command": wrapper,
                    "binary": binary,
                    "run_manifest": support.file_identity(manifest),
                    "outputs": support.identities(outputs),
                },
            )
            validate_nvbit(root, destination, cell, variant, binary)
            print(f"nvbit: {name}", flush=True)


def simulator_build_command() -> list[str]:
    return [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "sim/Cargo.toml"),
        "-p",
        "sim-run",
    ]


def capture_simulator_build(root: Path, timeout: int) -> None:
    destination = root / "simulator-build"
    manifest = destination / "build.run.json"
    stdout = destination / "stdout.txt"
    stderr = destination / "stderr.txt"
    command = simulator_build_command()
    inputs = {
        "cargo_manifest": REPO_ROOT / "sim/Cargo.toml",
        "cargo_lock": REPO_ROOT / "sim/Cargo.lock",
    }
    outputs = {"sim_run": SIM_RUN}
    if manifest.is_file():
        validate_simulator_build(root)
        return
    destination.mkdir(parents=True, exist_ok=True)
    support.recorded_run(
        command,
        timeout=timeout,
        manifest_path=manifest,
        stdout_path=stdout,
        stderr_path=stderr,
        input_paths=inputs,
        output_paths=outputs,
        environment_value=support.environment(),
    )


def validate_simulator_build(root: Path) -> None:
    destination = root / "simulator-build"
    manifest = destination / "build.run.json"
    configured = support.read_json(manifest).get("configured_timeout_seconds")
    if not isinstance(configured, int) or configured < 1:
        raise CaptureError(f"{manifest}: invalid timeout")
    support.validate_run_manifest(
        manifest,
        command=simulator_build_command(),
        timeout=configured,
        stdout_path=destination / "stdout.txt",
        stderr_path=destination / "stderr.txt",
        input_paths={
            "cargo_manifest": REPO_ROOT / "sim/Cargo.toml",
            "cargo_lock": REPO_ROOT / "sim/Cargo.lock",
        },
        output_paths={"sim_run": SIM_RUN},
        environment_value=support.environment(),
    )


def replay_command(root: Path, cell: Cell, variant: str) -> list[str]:
    return [
        str(SIM_RUN),
        "--trace",
        str(root / "nvbit" / cell.endpoint_name(variant)),
        "--control",
        str(root / "static/control_bits.json"),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
    ]


def validate_replay(root: Path, destination: Path, cell: Cell, variant: str) -> None:
    first = destination / "replay-1.json"
    second = destination / "replay-2.json"
    if not first.is_file() or not second.is_file() or first.read_bytes() != second.read_bytes():
        raise CaptureError(f"{destination}: incomplete or non-deterministic replay twins")
    support.read_json(first)
    command_path = destination / "command.json"
    command_document = support.read_json(command_path)
    timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(timeout, int) or timeout < 1:
        raise CaptureError(f"{command_path}: invalid timeout")
    command = replay_command(root, cell, variant)
    trace_directory = root / "nvbit" / cell.endpoint_name(variant)
    validate_trace_payload_manifest(trace_directory)
    inputs = {
        "arch_card": ARCH_CARD,
        "control": root / "static/control_bits.json",
        "pchase_card": PCHASE_CARD,
        "sim_run": SIM_RUN,
        "trace": trace_directory / "dynamic_trace.pb",
        "trace_payload_manifest": trace_payload_manifest_path(trace_directory),
    }
    manifests: dict[str, Any] = {}
    for repeat in (1, 2):
        manifest = destination / f"replay-{repeat}.run.json"
        support.validate_run_manifest(
            manifest,
            command=command,
            timeout=timeout,
            stdout_path=destination / f"replay-{repeat}.json",
            stderr_path=destination / f"stderr-{repeat}.txt",
            input_paths=inputs,
            output_paths={"replay": destination / f"replay-{repeat}.json"},
            environment_value=support.environment(),
        )
        manifests[f"replay_{repeat}"] = support.file_identity(manifest)
    expected = {
        "schema_version": 1,
        "cell": cell.endpoint_name(variant),
        "variant": variant,
        "orientation": cell.orientation,
        "configured_timeout_seconds": timeout,
        "command": command,
        "inputs": support.identities(inputs),
        "outputs": support.identities({"replay_1": first, "replay_2": second}),
        "run_manifests": manifests,
    }
    if command_document != expected:
        raise CaptureError(f"{command_path}: replay binding mismatch")


def capture_replay(root: Path, selected_cells: Sequence[Cell], timeout: int) -> None:
    capture_simulator_build(root, timeout)
    for cell in selected_cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            trace = root / "nvbit" / name / "dynamic_trace.pb"
            if not trace.is_file():
                raise CaptureError(f"missing trace: {trace}")
            validate_trace_payload_manifest(root / "nvbit" / name)
            destination = root / "simulator" / name
            if (destination / "command.json").is_file():
                validate_replay(root, destination, cell, variant)
                print(f"replay existing: {name}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = replay_command(root, cell, variant)
            inputs = {
                "arch_card": ARCH_CARD,
                "control": root / "static/control_bits.json",
                "pchase_card": PCHASE_CARD,
                "sim_run": SIM_RUN,
                "trace": trace,
                "trace_payload_manifest": trace_payload_manifest_path(
                    root / "nvbit" / name
                ),
            }
            manifests: dict[str, Any] = {}
            for repeat in (1, 2):
                manifest = destination / f"replay-{repeat}.run.json"
                support.recorded_run(
                    command,
                    timeout=timeout,
                    manifest_path=manifest,
                    stdout_path=destination / f"replay-{repeat}.json",
                    stderr_path=destination / f"stderr-{repeat}.txt",
                    input_paths=inputs,
                    output_paths={"replay": destination / f"replay-{repeat}.json"},
                    environment_value=support.environment(),
                )
                manifests[f"replay_{repeat}"] = support.file_identity(manifest)
            support.atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "cell": name,
                    "variant": variant,
                    "orientation": cell.orientation,
                    "configured_timeout_seconds": timeout,
                    "command": command,
                    "inputs": support.identities(inputs),
                    "outputs": support.identities(
                        {
                            "replay_1": destination / "replay-1.json",
                            "replay_2": destination / "replay-2.json",
                        }
                    ),
                    "run_manifests": manifests,
                },
            )
            validate_replay(root, destination, cell, variant)
            print(f"replay: {name}", flush=True)


def validate_selected(
    root: Path,
    selected_cells: Sequence[Cell],
    selected_repetitions: Sequence[int],
) -> None:
    binary = retained_binary(root)
    validate_simulator_build(root)
    for cell in selected_cells:
        for repetition in selected_repetitions:
            validate_direct_provenance(
                root / "direct" / cell.direct_name / f"repeat-{repetition}",
                cell,
                repetition,
                binary,
            )
        for variant in VARIANTS:
            validate_nvbit(
                root,
                root / "nvbit" / cell.endpoint_name(variant),
                cell,
                variant,
                binary,
            )
            validate_replay(
                root,
                root / "simulator" / cell.endpoint_name(variant),
                cell,
                variant,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--phase",
        choices=("all", "static", "direct", "nvbit", "replay"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cells", help="first,second")
    parser.add_argument("--repeats", help="comma-separated subset in 1..10")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.timeout < 1:
            raise CaptureError("--timeout must be positive")
        selected_cells = cells(args.smoke, args.cells)
        selected_repetitions = repetitions(args.smoke, args.repeats)
        root = args.root.resolve()
        full_execution = not args.smoke and (
            args.validate_only
            or args.phase in {"all", "direct", "nvbit", "replay"}
        )
        if full_execution:
            require_full_campaign_authorization(root, retained_binary(root))
        if args.validate_only:
            validate_selected(root, selected_cells, selected_repetitions)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "root": str(root),
                        "cells": [cell.orientation_name for cell in selected_cells],
                        "repetitions": selected_repetitions,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.phase in {"all", "static"}:
            support.capture_static(root, args.timeout)
        if args.phase in {"all", "direct"}:
            capture_direct(root, selected_cells, selected_repetitions, args.timeout)
        if args.phase in {"all", "nvbit"}:
            capture_nvbit(root, selected_cells, args.timeout)
        if args.phase in {"all", "replay"}:
            capture_replay(root, selected_cells, args.timeout)
        return 0
    except (CaptureError, OSError, ValueError) as error:
        print(f"line ownership execution capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
