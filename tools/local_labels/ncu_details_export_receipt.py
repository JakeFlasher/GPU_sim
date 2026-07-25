"""Pure contract for one descriptor-bound NCU Details export receipt."""

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
    / "ncu_details_export_receipt.schema.json"
)
SCHEMA_ID = "simbench_bw/schemas/ncu_details_export_receipt.schema.json"
SCHEMA_SHA256 = "330aa7e45d8812af919586205559c7b7d839b7c00bc632b1aee3d53ea06311e8"
_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
RETAINED_REPORT_ARGUMENT = "<RETAINED_REPORT_FD>"
EXPECTED_IMPORT_ARGUMENTS = (
    "--import",
    RETAINED_REPORT_ARGUMENT,
    "--page",
    "details",
    "--csv",
    "--print-kernel-base",
    "mangled",
    "--print-metric-name",
    "name",
    "--print-units",
    "base",
    "--print-fp",
)
EXPECTED_ACTION_ENVIRONMENT = (
    ("LANG", "C"),
    ("LANGUAGE", "C"),
    ("LC_ALL", "C"),
    ("PATH", "/bin:/usr/bin"),
)
_PRODUCT_ROLES = (
    "ncu_report",
    "details_csv",
    "details_stderr",
    "launch_inventory",
    "launch_metrics",
)
_PRODUCT_FIELDS = {"role", "relative_path", "sha256", "size_bytes"}


class DetailsExportReceiptError(ValueError):
    """An export receipt schema, seal, or external binding failed."""


def _fail(message: str) -> None:
    raise DetailsExportReceiptError(message)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha40(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: Any) -> str:
    try:
        return immutable.sha256_bytes(immutable.canonical_json_bytes(value))
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportReceiptError(str(exc)) from exc


def load_schema(
    *, schema_path: Path | None = None, payload: bytes | None = None,
) -> dict[str, Any]:
    """Load the byte-pinned active Details-export receipt schema."""

    path = SCHEMA_PATH if schema_path is None else Path(schema_path)
    if path != SCHEMA_PATH:
        _fail(f"Details export receipt schema path is not canonical: {path}")
    if payload is None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise DetailsExportReceiptError(
                f"Details export receipt schema authority is unavailable: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            _fail("Details export receipt schema authority is not a regular file")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise DetailsExportReceiptError(
                f"Details export receipt schema authority is unreadable: {exc}"
            ) from exc
    if immutable.sha256_bytes(payload) != SCHEMA_SHA256:
        _fail("Details export receipt schema SHA-256 mismatch")
    try:
        schema = immutable.strict_json_loads(
            payload, label="NCU Details export receipt schema",
        )
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportReceiptError(str(exc)) from exc
    if (
        type(schema) is not dict
        or schema.get("$schema") != _SCHEMA_DRAFT
        or schema.get("$id") != SCHEMA_ID
        or schema.get("type") != "object"
        or schema.get("properties", {}).get("schema_version", {}).get("const") != 1
        or schema.get("properties", {}).get("kind", {}).get("const")
        != "ncu_details_export_receipt_v1"
    ):
        _fail("Details export receipt schema authority shape mismatch")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise DetailsExportReceiptError(
            f"Details export receipt schema is invalid: {exc}"
        ) from exc
    return schema


def validate_schema(value: Any) -> None:
    errors = sorted(
        Draft202012Validator(load_schema()).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/" + "/".join(
            str(component) for component in error.absolute_path
        )
        _fail(
            f"Details export receipt schema validation failed at {location}: "
            f"{error.message}"
        )


def _validate_relative_path(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or value.splitlines() != [value]
        or "\\" in value
        or Path(value).is_absolute()
        or Path(value).as_posix() != value
        or any(part in {"", ".", ".."} for part in Path(value).parts)
    ):
        _fail(f"invalid Details export product path: {value!r}")
    return value


def product_binding(
    *, role: str, relative_path: str, sha256: str, size_bytes: int,
) -> dict[str, Any]:
    """Build one role-tagged byte identity for an export input or product."""

    if role not in _PRODUCT_ROLES:
        _fail(f"unsupported Details export product role: {role!r}")
    path = _validate_relative_path(relative_path)
    if not _is_sha256(sha256):
        _fail(f"invalid {role} SHA-256")
    minimum = 0 if role == "details_stderr" else 1
    if type(size_bytes) is not int or size_bytes < minimum:
        _fail(f"invalid {role} size")
    return {
        "role": role,
        "relative_path": path,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def product_from_bytes(
    *, role: str, relative_path: str, payload: bytes,
) -> dict[str, Any]:
    if type(payload) is not bytes:
        _fail(f"{role} payload is not exact bytes")
    return product_binding(
        role=role,
        relative_path=relative_path,
        sha256=immutable.sha256_bytes(payload),
        size_bytes=len(payload),
    )


def _product_models(
    products: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(products, (str, bytes)) or not isinstance(products, Sequence):
        _fail("Details export products are not a sequence")
    if len(products) != len(_PRODUCT_ROLES):
        _fail("Details export product roles or order differ")
    models: list[dict[str, Any]] = []
    for expected_role, item in zip(_PRODUCT_ROLES, products, strict=True):
        if not isinstance(item, Mapping) or set(item) != _PRODUCT_FIELDS:
            _fail(f"{expected_role} product binding fields differ")
        if item["role"] != expected_role:
            _fail("Details export product roles or order differ")
        models.append(product_binding(
            role=item["role"],
            relative_path=item["relative_path"],
            sha256=item["sha256"],
            size_bytes=item["size_bytes"],
        ))
    paths = [item["relative_path"] for item in models]
    if len(set(paths)) != len(paths):
        _fail("Details export product paths are not unique")
    return models


def _identity_model(
    expected: ncu_action_executor.ExpectedNcuIdentity,
) -> dict[str, str]:
    try:
        ncu_action_executor.validate_expected_ncu_identity(expected)
    except ncu_action_executor.NcuActionExecutorError as exc:
        raise DetailsExportReceiptError(str(exc)) from exc
    return {
        "resolved_path": expected.resolved_path,
        "executable_sha256": expected.executable_sha256,
        "version_line": expected.version_line,
    }


def _arguments_model(arguments: Sequence[str]) -> list[str]:
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        _fail("Details export arguments are not a sequence")
    model = list(arguments)
    if any(type(item) is not str or not item or "\x00" in item for item in model):
        _fail("Details export arguments contain an invalid item")
    if tuple(model) != EXPECTED_IMPORT_ARGUMENTS:
        _fail("Details export arguments differ from the normalized authority")
    if any("/proc/self/fd/" in item for item in model):
        _fail("Details export arguments leak an ephemeral descriptor path")
    return model


def _environment_model(environment: Mapping[str, str]) -> dict[str, str]:
    if type(environment) is not dict:
        _fail("Details export environment is not an exact dictionary")
    expected = dict(EXPECTED_ACTION_ENVIRONMENT)
    if environment != expected:
        _fail("Details export environment differs from the closed action authority")
    return expected


def _validate_scalar_authority(
    *, preflight_sha256: str, source_commit: str,
    capture_identity_sha256: str, working_directory: str, exit_status: int,
) -> None:
    if not _is_sha256(preflight_sha256):
        _fail("preflight_sha256 is not lowercase 64-hex")
    if not _is_sha40(source_commit):
        _fail("source_commit is not lowercase 40-hex")
    if not _is_sha256(capture_identity_sha256):
        _fail("capture_identity_sha256 is not lowercase 64-hex")
    if type(working_directory) is not str or working_directory != "/":
        _fail("Details export working directory is not exact root")
    if type(exit_status) is not int or exit_status != 0:
        _fail("Details export exit status is not exact integer zero")


def build_receipt(
    *,
    preflight_sha256: str,
    source_commit: str,
    capture_identity_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    import_arguments: Sequence[str],
    environment: Mapping[str, str],
    working_directory: str,
    exit_status: int,
    products: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate one sealed successful Details-export receipt."""

    _validate_scalar_authority(
        preflight_sha256=preflight_sha256,
        source_commit=source_commit,
        capture_identity_sha256=capture_identity_sha256,
        working_directory=working_directory,
        exit_status=exit_status,
    )
    arguments_model = _arguments_model(import_arguments)
    environment_model = _environment_model(environment)
    product_models = _product_models(products)
    unsigned = {
        "schema_version": 1,
        "kind": "ncu_details_export_receipt_v1",
        "trust_scope": "capture_process_integrity_only_unsigned",
        "status": "pass",
        "preflight_sha256": preflight_sha256,
        "source_commit": source_commit,
        "capture_identity_sha256": capture_identity_sha256,
        "ncu_identity": _identity_model(expected_ncu_identity),
        "import_arguments": arguments_model,
        "import_arguments_sha256": _digest(arguments_model),
        "environment": environment_model,
        "environment_sha256": _digest(environment_model),
        "working_directory": working_directory,
        "exit_status": exit_status,
        "products": product_models,
    }
    try:
        receipt = immutable.seal_model(unsigned, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportReceiptError(str(exc)) from exc
    validate_receipt(
        receipt,
        expected_preflight_sha256=preflight_sha256,
        expected_source_commit=source_commit,
        expected_capture_identity_sha256=capture_identity_sha256,
        expected_ncu_identity=expected_ncu_identity,
        expected_import_arguments=arguments_model,
        expected_environment=environment_model,
        expected_working_directory=working_directory,
        expected_exit_status=exit_status,
        expected_products=product_models,
    )
    return receipt


def validate_receipt(
    document: Any,
    *,
    expected_preflight_sha256: str,
    expected_source_commit: str,
    expected_capture_identity_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    expected_import_arguments: Sequence[str],
    expected_environment: Mapping[str, str],
    expected_working_directory: str,
    expected_exit_status: int,
    expected_products: Sequence[Mapping[str, Any]],
) -> None:
    """Validate one receipt against independent exporter and product bindings."""

    validate_schema(document)
    if type(document) is not dict:
        _fail("Details export receipt root is not an exact object")
    if type(document.get("schema_version")) is not int:
        _fail("Details export receipt schema_version is not an exact integer")
    try:
        immutable.verify_model_seal(document, "receipt_seal_sha256")
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportReceiptError(str(exc)) from exc
    _validate_scalar_authority(
        preflight_sha256=expected_preflight_sha256,
        source_commit=expected_source_commit,
        capture_identity_sha256=expected_capture_identity_sha256,
        working_directory=expected_working_directory,
        exit_status=expected_exit_status,
    )
    arguments_model = _arguments_model(expected_import_arguments)
    environment_model = _environment_model(expected_environment)
    product_models = _product_models(expected_products)
    expected = {
        "preflight_sha256": expected_preflight_sha256,
        "source_commit": expected_source_commit,
        "capture_identity_sha256": expected_capture_identity_sha256,
        "ncu_identity": _identity_model(expected_ncu_identity),
        "import_arguments": arguments_model,
        "import_arguments_sha256": _digest(arguments_model),
        "environment": environment_model,
        "environment_sha256": _digest(environment_model),
        "working_directory": expected_working_directory,
        "exit_status": expected_exit_status,
        "products": product_models,
    }
    mismatches = [
        name for name, value in expected.items()
        if immutable.canonical_json_bytes(document.get(name))
        != immutable.canonical_json_bytes(value)
    ]
    if mismatches:
        _fail(f"Details export receipt external bindings differ: {mismatches}")
