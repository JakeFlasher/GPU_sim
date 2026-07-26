"""Pure contract for one supervisor-authored NCU collection receipt."""

from __future__ import annotations

import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "ncu_collection_execution_receipt.schema.json"
)
SCHEMA_ID = "simbench_bw/schemas/ncu_collection_execution_receipt.schema.json"
SCHEMA_SHA256 = "013d0c4d467f8395122d2c08c6be8ac441e8ceb7a86905ecc02f8c34c9642253"
_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_ARTIFACT_ROLES = (
    "ncu_report",
    "label_run_manifest",
    "ncu_launch_observation",
    "label_capture_manifest",
)


class CollectionReceiptError(ValueError):
    """A collection receipt schema, model, seal, or binding failed."""


def _fail(message: str) -> None:
    raise CollectionReceiptError(message)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: Any) -> str:
    try:
        return immutable.sha256_bytes(immutable.canonical_json_bytes(value))
    except immutable.ImmutableArtifactError as exc:
        raise CollectionReceiptError(str(exc)) from exc


def load_schema(
    *, schema_path: Path | None = None, payload: bytes | None = None,
) -> dict[str, Any]:
    """Load the byte-pinned active collection-receipt schema."""

    path = SCHEMA_PATH if schema_path is None else Path(schema_path)
    if path != SCHEMA_PATH:
        _fail(f"collection receipt schema path is not canonical: {path}")
    if payload is None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CollectionReceiptError(
                f"collection receipt schema authority is unavailable: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            _fail("collection receipt schema authority is not a regular file")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise CollectionReceiptError(
                f"collection receipt schema authority is unreadable: {exc}"
            ) from exc
    if immutable.sha256_bytes(payload) != SCHEMA_SHA256:
        _fail("collection receipt schema SHA-256 mismatch")
    try:
        schema = immutable.strict_json_loads(
            payload, label="NCU collection receipt schema",
        )
    except immutable.ImmutableArtifactError as exc:
        raise CollectionReceiptError(str(exc)) from exc
    if (
        type(schema) is not dict
        or schema.get("$schema") != _SCHEMA_DRAFT
        or schema.get("$id") != SCHEMA_ID
        or schema.get("type") != "object"
        or schema.get("properties", {}).get("schema_version", {}).get("const") != 1
        or schema.get("properties", {}).get("kind", {}).get("const")
        != "ncu_collection_execution_receipt_v1"
    ):
        _fail("collection receipt schema authority shape mismatch")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise CollectionReceiptError(
            f"collection receipt schema is invalid: {exc}"
        ) from exc
    return schema


def validate_schema(value: Any) -> None:
    validator = Draft202012Validator(load_schema())
    errors = sorted(
        validator.iter_errors(value), key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/" + "/".join(
            str(component) for component in error.absolute_path
        )
        _fail(
            f"collection receipt schema validation failed at {location}: "
            f"{error.message}"
        )


def artifact_binding(
    *, role: str, relative_path: str, sha256: str, size_bytes: int,
) -> dict[str, Any]:
    """Build one role-tagged artifact identity without reading a pathname."""

    if role not in _ARTIFACT_ROLES:
        _fail(f"unsupported collection receipt artifact role: {role!r}")
    if (
        type(relative_path) is not str
        or not relative_path
        or "\x00" in relative_path
        or relative_path.splitlines() != [relative_path]
        or "\\" in relative_path
        or Path(relative_path).is_absolute()
        or Path(relative_path).as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in Path(relative_path).parts)
    ):
        _fail(f"invalid collection receipt artifact path: {relative_path!r}")
    if not _is_sha256(sha256):
        _fail(f"invalid {role} SHA-256")
    if type(size_bytes) is not int or size_bytes <= 0:
        _fail(f"invalid {role} size")
    return {
        "role": role,
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _identity_model(
    expected: ncu_action_executor.ExpectedNcuIdentity,
) -> dict[str, str]:
    try:
        ncu_action_executor.validate_expected_ncu_identity(expected)
    except ncu_action_executor.NcuActionExecutorError as exc:
        raise CollectionReceiptError(str(exc)) from exc
    return {
        "resolved_path": expected.resolved_path,
        "executable_sha256": expected.executable_sha256,
        "version_line": expected.version_line,
    }


def _environment_model(
    environment: ncu_action_executor.NcuCollectionEnvironment,
) -> dict[str, str]:
    if type(environment) is not ncu_action_executor.NcuCollectionEnvironment:
        _fail("collection environment has the wrong authority type")
    return environment.as_dict()


def _canonical_command_model(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        _fail("collection command is not a sequence")
    command_model = list(command)
    if not command_model or any(type(item) is not str for item in command_model):
        _fail("collection command contains non-string items")
    if any("\x00" in item for item in command_model):
        _fail("collection command contains an impossible NUL-bearing argument")
    try:
        separator = command_model.index("--")
    except ValueError as exc:
        raise CollectionReceiptError(
            "canonical collection command lacks the target separator"
        ) from exc
    if any(
        item == "--ncu-exec-path" or item.startswith("--ncu-exec-path=")
        for item in command_model[:separator]
    ):
        _fail("canonical collection command contains an ephemeral NCU execution path")
    return command_model


def _artifact_models(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        _fail("collection artifacts are not a sequence")
    if len(artifacts) != len(_ARTIFACT_ROLES):
        _fail("collection artifact roles or order differ")
    models: list[dict[str, Any]] = []
    required_fields = {"role", "relative_path", "sha256", "size_bytes"}
    for expected_role, item in zip(_ARTIFACT_ROLES, artifacts, strict=True):
        if not isinstance(item, Mapping) or set(item) != required_fields:
            _fail(f"{expected_role} artifact binding fields differ")
        if item["role"] != expected_role:
            _fail("collection artifact roles or order differ")
        models.append(artifact_binding(
            role=item["role"],
            relative_path=item["relative_path"],
            sha256=item["sha256"],
            size_bytes=item["size_bytes"],
        ))
    return models


def build_receipt(
    *,
    preflight_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    command: Sequence[str],
    environment: ncu_action_executor.NcuCollectionEnvironment,
    working_directory: Path,
    exit_status: int,
    artifacts: Sequence[Mapping[str, Any]],
    launch_observation_sha256: str,
) -> dict[str, Any]:
    """Build and validate one sealed successful collection receipt."""

    if not _is_sha256(preflight_sha256):
        _fail("preflight_sha256 is not lowercase 64-hex")
    command_model = _canonical_command_model(command)
    environment_model = _environment_model(environment)
    if not isinstance(working_directory, Path):
        _fail("working directory is not a path")
    working_directory_text = str(working_directory)
    if environment_model.get("PWD") != working_directory_text:
        _fail("collection environment PWD differs from working directory")
    if type(exit_status) is not int or exit_status != 0:
        _fail("collection receipt can record only successful exit status 0")
    artifact_model = _artifact_models(artifacts)
    if not _is_sha256(launch_observation_sha256):
        _fail("launch observation seal is not lowercase 64-hex")

    unsigned = {
        "schema_version": 1,
        "kind": "ncu_collection_execution_receipt_v1",
        "trust_scope": "capture_process_integrity_only_unsigned",
        "status": "pass",
        "supervision": {
            "execution_mode": "bounded_supervised_child_v1",
            "ncu_binding": "retained_descriptor_singleton_argv_v1",
            "command_derivation": (
                "insert_authorized_ncu_proc_path_option_before_"
                "target_separator_v1"
            ),
            "residual_process_policy": "kill_process_group_before_return_v1",
        },
        "preflight_sha256": preflight_sha256,
        "ncu_identity": _identity_model(expected_ncu_identity),
        "command": command_model,
        "command_sha256": _digest(command_model),
        "environment": environment_model,
        "environment_sha256": _digest(environment_model),
        "working_directory": working_directory_text,
        "exit_status": exit_status,
        "artifacts": artifact_model,
        "launch_observation_sha256": launch_observation_sha256,
    }
    try:
        receipt = immutable.seal_model(unsigned, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise CollectionReceiptError(str(exc)) from exc
    validate_receipt(
        receipt,
        expected_preflight_sha256=preflight_sha256,
        expected_ncu_identity=expected_ncu_identity,
        expected_command=command_model,
        expected_environment=environment,
        expected_working_directory=working_directory,
        expected_exit_status=exit_status,
        expected_artifacts=artifact_model,
        expected_launch_observation_sha256=launch_observation_sha256,
    )
    return receipt


def validate_receipt(
    document: Any,
    *,
    expected_preflight_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    expected_command: Sequence[str],
    expected_environment: ncu_action_executor.NcuCollectionEnvironment,
    expected_working_directory: Path,
    expected_exit_status: int,
    expected_artifacts: Sequence[Mapping[str, Any]],
    expected_launch_observation_sha256: str,
) -> None:
    """Validate a receipt against independently supplied collection bindings."""

    validate_schema(document)
    if type(document) is not dict:
        _fail("collection receipt root is not an exact object")
    try:
        immutable.verify_model_seal(document, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise CollectionReceiptError(str(exc)) from exc

    command_model = _canonical_command_model(expected_command)
    environment_model = _environment_model(expected_environment)
    artifact_model = _artifact_models(expected_artifacts)
    if not _is_sha256(expected_preflight_sha256):
        _fail("expected preflight_sha256 is not lowercase 64-hex")
    if not isinstance(expected_working_directory, Path):
        _fail("expected working directory is not a path")
    expected_working_directory_text = str(expected_working_directory)
    if environment_model.get("PWD") != expected_working_directory_text:
        _fail("expected collection environment PWD differs from working directory")
    if type(expected_exit_status) is not int or expected_exit_status != 0:
        _fail("expected collection exit status is not exact integer zero")
    if not _is_sha256(expected_launch_observation_sha256):
        _fail("expected launch observation seal is not lowercase 64-hex")
    expected = {
        "preflight_sha256": expected_preflight_sha256,
        "ncu_identity": _identity_model(expected_ncu_identity),
        "command": command_model,
        "command_sha256": _digest(command_model),
        "environment": environment_model,
        "environment_sha256": _digest(environment_model),
        "working_directory": expected_working_directory_text,
        "exit_status": expected_exit_status,
        "artifacts": artifact_model,
        "launch_observation_sha256": expected_launch_observation_sha256,
    }
    mismatches = [
        name for name, value in expected.items() if document.get(name) != value
    ]
    if mismatches:
        _fail(f"collection receipt external bindings differ: {mismatches}")
