"""Sealed receipt contract for the direct golden-cell NCU action."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from pathlib import Path
from typing import Any

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor


class GoldenReceiptError(ValueError):
    """A golden receipt model, binding, or seal failed."""


def _fail(message: str) -> None:
    raise GoldenReceiptError(message)


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: Any) -> str:
    try:
        return immutable.sha256_bytes(immutable.canonical_json_bytes(value))
    except immutable.ImmutableArtifactError as exc:
        raise GoldenReceiptError(str(exc)) from exc


def _identity(expected: ncu_action_executor.ExpectedNcuIdentity) -> dict[str, str]:
    try:
        ncu_action_executor.validate_expected_ncu_identity(expected)
    except ncu_action_executor.NcuActionExecutorError as exc:
        raise GoldenReceiptError(str(exc)) from exc
    return {
        "resolved_path": expected.resolved_path,
        "executable_sha256": expected.executable_sha256,
        "version_line": expected.version_line,
    }


def _command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        _fail("golden command is not a sequence")
    values = list(command)
    if not values or any(type(value) is not str for value in values):
        _fail("golden command contains a non-string item")
    if any("\x00" in value for value in values):
        _fail("golden command contains a NUL-bearing item")
    expected_prefix = (
        "--metrics", "lts__t_sector_hit_rate.pct",
        "--clock-control", "none",
        "--cache-control", "all",
        "--replay-mode", "kernel",
        "--kernel-name-base", "mangled",
        "--launch-count", "1", "--csv",
    )
    if tuple(values[:-2]) != expected_prefix or values[-2] != "--":
        _fail("golden command differs from the closed direct-NCU arguments")
    if re.fullmatch(r"/proc/self/fd/[0-9]+", values[-1]) is None:
        _fail("golden command target is not a retained binary descriptor")
    return values


def _environment(environment: Mapping[str, str]) -> dict[str, str]:
    if type(environment) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in environment.items()
    ):
        _fail("golden environment is not an exact string dictionary")
    return dict(sorted(environment.items()))


def _absolute_path(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.startswith("/")
        or "\x00" in value
        or value != str(Path(value))
        or "//" in value
        or any(part in {"", ".", ".."} for part in Path(value).parts)
    ):
        _fail(f"{label} is not canonical absolute path")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or Path(value).as_posix() != value
        or any(part in {"", ".", ".."} for part in Path(value).parts)
    ):
        _fail(f"{label} is not canonical repo-relative path")
    return value


def _topology(
    target_binary: Mapping[str, Any],
    trace_relative_path: object,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    target_path = Path(_relative_path(
        target_binary["relative_path"], label="golden target binary path",
    ))
    trace_path = Path(_relative_path(
        trace_relative_path, label="golden trace path",
    ))
    if (
        target_path.name != f"{target_path.parent.name}.bin"
        or trace_path.name != "trace"
        or trace_path.parent != target_path.parent
    ):
        _fail("golden receipt target and trace paths differ in cell")
    expected_names = {
        "ncu_report": "ncu_l2.csv",
        "ncu_stderr": "ncu_l2.stderr",
        "label_run_manifest": "ncu_run_manifest.json",
        "ncu_launch_observation": "ncu_launch_observation.json",
        "label_capture_manifest": "ncu_capture_manifest.json",
    }
    for artifact in artifacts:
        path = Path(artifact["relative_path"])
        if path.parent != trace_path.parent or path.name != expected_names[artifact["role"]]:
            _fail("golden receipt artifact is outside the selected cell")


def _artifacts(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_roles = (
        "ncu_report",
        "ncu_stderr",
        "label_run_manifest",
        "ncu_launch_observation",
        "label_capture_manifest",
    )
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        _fail("golden artifacts are not a sequence")
    if len(artifacts) != len(expected_roles):
        _fail("golden artifact role count differs")
    result: list[dict[str, Any]] = []
    for expected_role, item in zip(expected_roles, artifacts, strict=True):
        if not isinstance(item, Mapping) or set(item) != {
            "role", "relative_path", "sha256", "size_bytes",
        }:
            _fail(f"{expected_role} artifact fields differ")
        relative = item["relative_path"]
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or Path(relative).as_posix() != relative
        ):
            _fail(f"{expected_role} artifact path is not canonical")
        if item["role"] != expected_role or not _sha256(item["sha256"]):
            _fail(f"{expected_role} artifact identity differs")
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
            _fail(f"{expected_role} artifact size is invalid")
        result.append(dict(item))
    return result


def _dim3(value: object, *, label: str) -> list[int]:
    if (
        type(value) is not list
        or len(value) != 3
        or any(type(component) is not int or component <= 0 for component in value)
    ):
        _fail(f"{label} is not an exact positive dim3")
    return list(value)


def _trace_launch(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "launch_index",
        "device_id",
        "stream_id",
        "kernel_id",
        "kernel_name",
        "grid_dim",
        "block_dim",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _fail("golden admitted trace launch fields differ")
    for name in ("launch_index", "device_id", "stream_id", "kernel_id"):
        if type(value[name]) is not int or value[name] < 0:
            _fail(f"golden admitted trace {name} is invalid")
    kernel_name = value["kernel_name"]
    if (
        type(kernel_name) is not str
        or not kernel_name.strip()
        or kernel_name != kernel_name.strip()
        or "\x00" in kernel_name
    ):
        _fail("golden admitted trace kernel name is invalid")
    return {
        "launch_index": value["launch_index"],
        "device_id": value["device_id"],
        "stream_id": value["stream_id"],
        "kernel_id": value["kernel_id"],
        "kernel_name": kernel_name,
        "grid_dim": _dim3(value["grid_dim"], label="golden trace grid_dim"),
        "block_dim": _dim3(value["block_dim"], label="golden trace block_dim"),
    }


def _trace_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    counter_fields = {
        "launch_count", "threadblock_count", "instruction_count",
        "expected_record_count", "admitted_record_count",
    }
    expected_fields = counter_fields | {"launches"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _fail("golden trace admission fields differ")
    if any(
        type(value[name]) is not int or value[name] <= 0
        for name in counter_fields
    ):
        _fail("golden trace admission counters are invalid")
    if value["admitted_record_count"] > value["expected_record_count"]:
        _fail("golden trace admission counters are inconsistent")
    launches = value["launches"]
    if (
        type(launches) is not list
        or value["launch_count"] != 1
        or len(launches) != 1
    ):
        _fail("golden trace admission must bind exactly one launch")
    launch = _trace_launch(launches[0])
    if launch["launch_index"] != 0:
        _fail("golden trace admission launch ordinal is not zero")
    return {
        **{name: value[name] for name in counter_fields},
        "launches": [launch],
    }


def _nvbit_kernel_base(value: object) -> str:
    if type(value) is not str:
        _fail("golden admitted trace kernel name is not a string")
    match = re.fullmatch(r"(.+)___(0|[1-9][0-9]*)", value)
    if match is None:
        _fail("golden admitted trace kernel lacks its producer variant suffix")
    if match.group(2) != "0":
        _fail("golden admitted trace kernel variant is ambiguous")
    return match.group(1)


def _launch_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "kind",
        "cell_slug",
        "ncu_observation",
        "admitted_trace_launch",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _fail("golden NCU/trace launch binding fields differ")
    slug = value["cell_slug"]
    if (
        type(slug) is not str
        or not slug
        or "/" in slug
        or "\\" in slug
        or value["schema_version"] != 1
        or value["kind"] != "golden_ncu_trace_launch_binding_v1"
    ):
        _fail("golden NCU/trace launch binding identity differs")
    ncu = value["ncu_observation"]
    ncu_fields = {
        "report_launch_id", "device_id", "kernel_name", "grid_dim", "block_dim",
    }
    if not isinstance(ncu, Mapping) or set(ncu) != ncu_fields:
        _fail("golden NCU launch binding fields differ")
    if (
        type(ncu["report_launch_id"]) is not str
        or not ncu["report_launch_id"].strip()
        or ncu["report_launch_id"] != ncu["report_launch_id"].strip()
        or type(ncu["device_id"]) is not int
        or ncu["device_id"] < 0
        or type(ncu["kernel_name"]) is not str
        or not ncu["kernel_name"].strip()
        or ncu["kernel_name"] != ncu["kernel_name"].strip()
    ):
        _fail("golden NCU launch binding identity is invalid")
    ncu_grid_dim = _dim3(
        ncu["grid_dim"], label="golden NCU observation grid_dim",
    )
    ncu_block_dim = _dim3(
        ncu["block_dim"], label="golden NCU observation block_dim",
    )
    trace_launch = _trace_launch(value["admitted_trace_launch"])
    if (
        ncu["device_id"] != trace_launch["device_id"]
        or ncu["kernel_name"] != _nvbit_kernel_base(trace_launch["kernel_name"])
        or ncu_grid_dim != trace_launch["grid_dim"]
        or ncu_block_dim != trace_launch["block_dim"]
    ):
        _fail("golden NCU observation differs from admitted trace launch")
    return {
        "schema_version": 1,
        "kind": "golden_ncu_trace_launch_binding_v1",
        "cell_slug": slug,
        "ncu_observation": {
            **dict(ncu),
            "grid_dim": ncu_grid_dim,
            "block_dim": ncu_block_dim,
        },
        "admitted_trace_launch": trace_launch,
    }


def _target_binary(target_binary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(target_binary, Mapping) or set(target_binary) != {
        "relative_path", "sha256", "size_bytes", "mode", "executable",
    }:
        _fail("golden target binary binding fields differ")
    relative_path = _relative_path(
        target_binary["relative_path"], label="golden target binary path",
    )
    mode = target_binary["mode"]
    if (
        not _sha256(target_binary["sha256"])
        or type(target_binary["size_bytes"]) is not int
        or target_binary["size_bytes"] <= 0
        or type(mode) is not str
        or re.fullmatch(r"0[0-7]{3}", mode) is None
        or int(mode, 8) & 0o222
        or int(mode, 8) & 0o111 == 0
        or target_binary["executable"] is not True
    ):
        _fail("golden target binary binding is invalid")
    result = dict(target_binary)
    result["relative_path"] = relative_path
    return result


def _target_argument_binding(
    command: Sequence[str], target_binary: Mapping[str, Any],
) -> dict[str, Any]:
    command_model = _command(command)
    target_model = _target_binary(target_binary)
    return {
        "role": "target_binary",
        "descriptor_path": command_model[-1],
        "target_relative_path": target_model["relative_path"],
        "target_sha256": target_model["sha256"],
        "target_size_bytes": target_model["size_bytes"],
        "target_mode": target_model["mode"],
    }


def _controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "clock_control": "none",
        "cache_control": "all",
        "replay_mode": "kernel",
        "launch_skip": 0,
        "launch_count": 1,
        "metric": "lts__t_sector_hit_rate.pct",
    }
    if type(controls) is not dict or controls != expected:
        _fail("golden launch controls differ from the closed contract")
    return dict(expected)


def validate_ncu_identity_against_preflight(
    document: Mapping[str, Any], manifest: Mapping[str, Any],
) -> None:
    """Join the sealed receipt identity to the retained preflight authority."""
    if not isinstance(document, Mapping) or not isinstance(manifest, Mapping):
        _fail("golden receipt/preflight identity inputs are not objects")
    try:
        ncu = manifest["checks"]["ncu"]
        expected = ncu_action_executor.build_expected_ncu_identity(
            resolved_path=ncu["resolved_path"],
            executable_sha256=ncu["executable_sha256"],
            version_line=ncu["version_line"],
        )
    except (KeyError, TypeError, ncu_action_executor.NcuActionExecutorError) as exc:
        raise GoldenReceiptError(
            f"preflight NCU identity is not admitted: {exc}"
        ) from exc
    if document.get("ncu_identity") != _identity(expected):
        _fail("golden receipt NCU identity differs from preflight authority")


def validate_intrinsic_receipt(document: Any) -> None:
    """Validate the fixed receipt contract without trusting caller expectations."""
    if not isinstance(document, dict):
        _fail("golden receipt is not an object")
    try:
        immutable.verify_model_seal(document, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise GoldenReceiptError(str(exc)) from exc
    expected_fields = {
        "schema_version", "kind", "trust_scope", "status", "supervision",
        "preflight_sha256", "preflight_path", "ncu_identity", "command",
        "target_argument_role", "target_argument_binding", "command_sha256",
        "environment", "environment_sha256",
        "working_directory", "exit_status", "target_binary",
        "trace_relative_path", "trace_admission", "trace_tree_sha256",
        "trace_tree_bytes", "launch_controls", "artifacts",
        "launch_observation_sha256", "launch_binding",
    }
    if set(document) != expected_fields | {"receipt_seal_sha256"}:
        _fail("golden receipt fields differ")
    if (
        document["schema_version"] != 2
        or document["kind"] != "golden_ncu_collection_receipt_v2"
        or document["trust_scope"] != "capture_process_integrity_v1"
        or document["status"] != "pass"
        or document["target_argument_role"]
        != "retained_target_binary_descriptor_v1"
        or document["supervision"] != {
            "execution_mode": "bounded_direct_ncu_v1",
            "ncu_binding": "retained_descriptor_singleton_argv_v1",
            "command_derivation": "execute_ncu_action_direct_argument_vector_v1",
            "residual_process_policy": "kill_process_group_before_return_v1",
        }
        or document["working_directory"] != "/"
        or document["exit_status"] != 0
    ):
        _fail("golden receipt fixed identity or execution fields differ")
    if not _sha256(document["preflight_sha256"]):
        _fail("golden receipt preflight authority hash is invalid")
    if not _sha256(document["launch_observation_sha256"]):
        _fail("golden receipt observation hash is invalid")
    preflight_path = _absolute_path(
        document["preflight_path"], label="golden preflight path",
    )
    try:
        expected_identity = ncu_action_executor.build_expected_ncu_identity(
            resolved_path=document["ncu_identity"]["resolved_path"],
            executable_sha256=document["ncu_identity"]["executable_sha256"],
            version_line=document["ncu_identity"]["version_line"],
        )
    except (KeyError, TypeError, ncu_action_executor.NcuActionExecutorError) as exc:
        raise GoldenReceiptError(f"golden NCU identity is invalid: {exc}") from exc
    if document["ncu_identity"] != _identity(expected_identity):
        _fail("golden NCU identity is not canonical")
    command = _command(document["command"])
    if document["command_sha256"] != _digest(command):
        _fail("golden command hash does not match command")
    if document["target_argument_role"] != "retained_target_binary_descriptor_v1":
        _fail("golden target argument role is not canonical")
    environment = _environment(document["environment"])
    if environment != ncu_action_executor.ncu_action_environment():
        _fail("golden direct-NCU environment differs from the closed contract")
    if document["environment_sha256"] != _digest(environment):
        _fail("golden environment hash does not match environment")
    target = _target_binary(document["target_binary"])
    if document["target_argument_binding"] != _target_argument_binding(
        command, target,
    ):
        _fail("golden target argument is not bound to target binary")
    trace = _relative_path(
        document["trace_relative_path"], label="golden trace path",
    )
    trace_admission = _trace_admission(document["trace_admission"])
    launch_binding = _launch_binding(document["launch_binding"])
    if launch_binding["admitted_trace_launch"] != trace_admission["launches"][0]:
        _fail("golden receipt launch binding differs from trace admission")
    target_slug = Path(target["relative_path"]).parent.name
    if launch_binding["cell_slug"] != target_slug:
        _fail("golden receipt launch binding differs from target cell")
    if (
        not _sha256(document["trace_tree_sha256"])
        or type(document["trace_tree_bytes"]) is not int
        or document["trace_tree_bytes"] <= 0
    ):
        _fail("golden trace tree identity is invalid")
    artifacts = _artifacts(document["artifacts"])
    _topology(target, trace, artifacts)
    _controls(document["launch_controls"])
    _absolute_path(preflight_path, label="golden preflight path")


def build_receipt(
    *,
    preflight_sha256: str,
    preflight_path: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    command: Sequence[str],
    environment: Mapping[str, str],
    working_directory: str,
    target_binary: Mapping[str, Any],
    trace_relative_path: str,
    trace_admission: Mapping[str, Any],
    trace_tree_sha256: str,
    trace_tree_bytes: int,
    exit_status: int,
    artifacts: Sequence[Mapping[str, Any]],
    launch_observation_sha256: str,
    launch_binding: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    if not _sha256(preflight_sha256) or not _sha256(launch_observation_sha256):
        _fail("golden receipt authority hash is invalid")
    preflight_path_model = _absolute_path(
        preflight_path, label="golden preflight path",
    )
    command_model = _command(command)
    environment_model = _environment(environment)
    target_model = _target_binary(target_binary)
    trace_model = _relative_path(trace_relative_path, label="golden trace path")
    trace_admission_model = _trace_admission(trace_admission)
    launch_binding_model = _launch_binding(launch_binding)
    if (
        launch_binding_model["admitted_trace_launch"]
        != trace_admission_model["launches"][0]
    ):
        _fail("golden launch binding differs from trace admission")
    if (
        launch_binding_model["cell_slug"]
        != Path(target_model["relative_path"]).parent.name
    ):
        _fail("golden launch binding differs from target cell")
    if (
        not _sha256(trace_tree_sha256)
        or type(trace_tree_bytes) is not int
        or trace_tree_bytes <= 0
    ):
        _fail("golden trace tree identity is invalid")
    artifact_model = _artifacts(artifacts)
    _topology(target_model, trace_model, artifact_model)
    controls_model = _controls(controls)
    if working_directory != "/":
        _fail("golden direct action working directory must be /")
    if type(exit_status) is not int or exit_status != 0:
        _fail("golden receipt accepts only successful exit status 0")
    if not isinstance(controls, Mapping):
        _fail("golden controls are not an object")
    unsigned = {
        "schema_version": 2,
        "kind": "golden_ncu_collection_receipt_v2",
        "trust_scope": "capture_process_integrity_v1",
        "status": "pass",
        "supervision": {
            "execution_mode": "bounded_direct_ncu_v1",
            "ncu_binding": "retained_descriptor_singleton_argv_v1",
            "command_derivation": "execute_ncu_action_direct_argument_vector_v1",
            "residual_process_policy": "kill_process_group_before_return_v1",
        },
        "preflight_sha256": preflight_sha256,
        "preflight_path": preflight_path_model,
        "ncu_identity": _identity(expected_ncu_identity),
        "command": command_model,
        "target_argument_role": "retained_target_binary_descriptor_v1",
        "target_argument_binding": _target_argument_binding(
            command_model, target_model,
        ),
        "command_sha256": _digest(command_model),
        "environment": environment_model,
        "environment_sha256": _digest(environment_model),
        "working_directory": working_directory,
        "exit_status": exit_status,
        "target_binary": target_model,
        "trace_relative_path": trace_model,
        "trace_admission": trace_admission_model,
        "trace_tree_sha256": trace_tree_sha256,
        "trace_tree_bytes": trace_tree_bytes,
        "launch_controls": controls_model,
        "artifacts": artifact_model,
        "launch_observation_sha256": launch_observation_sha256,
        "launch_binding": launch_binding_model,
    }
    try:
        receipt = immutable.seal_model(unsigned, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise GoldenReceiptError(str(exc)) from exc
    validate_receipt(
        receipt,
        expected_preflight_sha256=preflight_sha256,
        expected_preflight_path=preflight_path_model,
        expected_ncu_identity=expected_ncu_identity,
        expected_command=command_model,
        expected_environment=environment_model,
        expected_working_directory=working_directory,
        expected_target_binary=target_model,
        expected_trace_relative_path=trace_model,
        expected_trace_admission=trace_admission_model,
        expected_trace_tree_sha256=trace_tree_sha256,
        expected_trace_tree_bytes=trace_tree_bytes,
        expected_exit_status=exit_status,
        expected_artifacts=artifact_model,
        expected_launch_observation_sha256=launch_observation_sha256,
        expected_launch_binding=launch_binding_model,
        expected_controls=controls,
    )
    return receipt


def validate_receipt(
    document: Any,
    *,
    expected_preflight_sha256: str,
    expected_preflight_path: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    expected_command: Sequence[str],
    expected_environment: Mapping[str, str],
    expected_working_directory: str,
    expected_target_binary: Mapping[str, Any],
    expected_trace_relative_path: str,
    expected_trace_admission: Mapping[str, Any],
    expected_trace_tree_sha256: str,
    expected_trace_tree_bytes: int,
    expected_exit_status: int,
    expected_artifacts: Sequence[Mapping[str, Any]],
    expected_launch_observation_sha256: str,
    expected_launch_binding: Mapping[str, Any],
    expected_controls: Mapping[str, Any],
) -> None:
    if not isinstance(document, dict):
        _fail("golden receipt is not an object")
    validate_intrinsic_receipt(document)
    try:
        immutable.verify_model_seal(document, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise GoldenReceiptError(str(exc)) from exc
    target_model = _target_binary(expected_target_binary)
    trace_model = _relative_path(
        expected_trace_relative_path, label="golden trace path",
    )
    artifact_model = _artifacts(expected_artifacts)
    _topology(target_model, trace_model, artifact_model)
    trace_admission_model = _trace_admission(expected_trace_admission)
    launch_binding_model = _launch_binding(expected_launch_binding)
    if (
        launch_binding_model["admitted_trace_launch"]
        != trace_admission_model["launches"][0]
        or launch_binding_model["cell_slug"]
        != Path(target_model["relative_path"]).parent.name
    ):
        _fail("expected golden launch binding differs from trace or target")
    expected = {
        "schema_version": 2,
        "kind": "golden_ncu_collection_receipt_v2",
        "trust_scope": "capture_process_integrity_v1",
        "status": "pass",
        "supervision": {
            "execution_mode": "bounded_direct_ncu_v1",
            "ncu_binding": "retained_descriptor_singleton_argv_v1",
            "command_derivation": "execute_ncu_action_direct_argument_vector_v1",
            "residual_process_policy": "kill_process_group_before_return_v1",
        },
        "preflight_sha256": expected_preflight_sha256,
        "preflight_path": _absolute_path(
            expected_preflight_path, label="golden preflight path",
        ),
        "ncu_identity": _identity(expected_ncu_identity),
        "command": _command(expected_command),
        "target_argument_role": "retained_target_binary_descriptor_v1",
        "target_argument_binding": _target_argument_binding(
            expected_command, expected_target_binary,
        ),
        "command_sha256": _digest(_command(expected_command)),
        "environment": _environment(expected_environment),
        "environment_sha256": _digest(_environment(expected_environment)),
        "working_directory": expected_working_directory,
        "exit_status": expected_exit_status,
        "target_binary": target_model,
        "trace_relative_path": trace_model,
        "trace_admission": trace_admission_model,
        "trace_tree_sha256": expected_trace_tree_sha256,
        "trace_tree_bytes": expected_trace_tree_bytes,
        "launch_controls": _controls(expected_controls),
        "artifacts": artifact_model,
        "launch_observation_sha256": expected_launch_observation_sha256,
        "launch_binding": launch_binding_model,
    }
    for name, value in expected.items():
        if document.get(name) != value:
            _fail(f"golden receipt field mismatch: {name}")
    if set(document) != set(expected) | {"receipt_seal_sha256"}:
        _fail("golden receipt fields differ")
