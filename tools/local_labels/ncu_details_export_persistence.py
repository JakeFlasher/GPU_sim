"""Durable no-replace publication for one NCU Details export product set."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
from tools.local_labels import ncu_details_export_receipt as receipt_contract
from tools.local_labels import ncu_launch_documents


_EXPECTED_BASENAMES = {
    "ncu_report": "run_0_launch_set.ncu-rep",
    "details_csv": "run_0_launch_set.details.csv",
    "details_stderr": "run_0_launch_set.details.stderr",
    "launch_inventory": "run_0_launch_set.launch_inventory.json",
    "launch_metrics": "run_0_launch_set.launch_metrics.json",
}
_RECEIPT_BASENAME = "run_0_launch_set.details_export_receipt.json"
_PARTIAL_SUFFIX = ".ncu-details-partial"
_GENERATED_ROLES = (
    "details_csv",
    "details_stderr",
    "launch_inventory",
    "launch_metrics",
)


class DetailsExportPersistenceError(RuntimeError):
    """A Details export product set could not be safely published."""


def _fail(message: str) -> None:
    raise DetailsExportPersistenceError(message)


def _canonical_root(path: Path) -> Path:
    if not isinstance(path, Path):
        _fail("artifact_root is not a path")
    text = str(path)
    if (
        not path.is_absolute()
        or text.startswith("//")
        or os.path.normpath(text) != text
    ):
        _fail(f"artifact_root is not a canonical absolute path: {text!r}")
    return path


def _relative(value: Any, *, label: str) -> PurePosixPath:
    try:
        return immutable.safe_relative_path(value)
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportPersistenceError(
            f"{label} is not a safe relative path: {exc}"
        ) from exc


def _require_retained_identity(
    retained: immutable.RetainedRegular,
    *,
    binding: Mapping[str, Any],
    expected_bytes: bytes | None,
    require_read_only: bool,
) -> bytes:
    retained.verify_unchanged()
    if (
        retained.sha256 != binding.get("sha256")
        or retained.size_bytes != binding.get("size_bytes")
    ):
        _fail(f"retained {binding.get('role')} identity differs from receipt")
    metadata = os.fstat(retained.descriptor)
    if require_read_only and metadata.st_mode & 0o222:
        _fail(f"published {binding.get('role')} remains writable")
    if expected_bytes is None:
        return b""
    offset = 0
    while offset < retained.size_bytes:
        try:
            block = os.pread(
                retained.descriptor,
                min(1 << 20, retained.size_bytes - offset),
                offset,
            )
        except OSError as exc:
            raise DetailsExportPersistenceError(
                f"cannot read retained export product {retained.path}: {exc}"
            ) from exc
        if not block:
            _fail(f"short retained export product read: {retained.path}")
        if block != expected_bytes[offset:offset + len(block)]:
            _fail(f"retained {binding.get('role')} bytes differ from producer bytes")
        offset += len(block)
    if offset != len(expected_bytes):
        _fail(f"retained {binding.get('role')} bytes differ from producer bytes")
    retained.verify_unchanged()
    return expected_bytes


def _strict_model(payload: bytes, *, label: str) -> Any:
    try:
        return immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportPersistenceError(str(exc)) from exc


def _validated_plan(
    *,
    artifact_root: Path,
    receipt_relative_path: str,
    products: ncu_launch_documents.DetailsExportProducts,
    expected_preflight_sha256: str,
    expected_source_commit: str,
    expected_capture_identity_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    expected_launch_count: int,
    protocol: Mapping[str, Any],
    registry: Any,
    percentage_metric_names: Iterable[str],
    unit_metric_names: Iterable[str],
) -> tuple[
    Path,
    PurePosixPath,
    dict[str, dict[str, Any]],
    dict[str, bytes],
]:
    root = _canonical_root(artifact_root)
    if type(products) is not ncu_launch_documents.DetailsExportProducts:
        _fail("Details export products have the wrong authority type")
    try:
        expected_inventory_bytes = immutable.pretty_json_bytes(
            products.inventory_document
        )
        expected_metrics_bytes = immutable.pretty_json_bytes(
            products.metrics_document
        )
        expected_receipt_bytes = immutable.pretty_json_bytes(
            products.receipt_document
        )
    except immutable.ImmutableArtifactError as exc:
        raise DetailsExportPersistenceError(str(exc)) from exc
    if (
        type(products.csv_bytes) is not bytes
        or not products.csv_bytes
        or type(products.stderr_bytes) is not bytes
        or products.inventory_bytes != expected_inventory_bytes
        or products.metrics_bytes != expected_metrics_bytes
        or products.receipt_bytes != expected_receipt_bytes
    ):
        _fail("Details export producer bytes are internally inconsistent")
    try:
        csv_text = products.csv_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DetailsExportPersistenceError(
            f"Details CSV is not strict UTF-8: {exc}"
        ) from exc
    if _strict_model(
        products.inventory_bytes, label="launch inventory",
    ) != products.inventory_document:
        _fail("launch inventory bytes differ from its exact model")
    if _strict_model(
        products.metrics_bytes, label="launch metrics",
    ) != products.metrics_document:
        _fail("launch metrics bytes differ from its exact model")
    if _strict_model(
        products.receipt_bytes, label="Details export receipt",
    ) != products.receipt_document:
        _fail("Details export receipt bytes differ from its exact model")
    try:
        receipt_contract.validate_schema(products.receipt_document)
        immutable.verify_model_seal(
            products.receipt_document, "receipt_seal_sha256"
        )
        receipt_contract.validate_receipt(
            products.receipt_document,
            expected_preflight_sha256=expected_preflight_sha256,
            expected_source_commit=expected_source_commit,
            expected_capture_identity_sha256=(
                expected_capture_identity_sha256
            ),
            expected_ncu_identity=expected_ncu_identity,
            expected_import_arguments=(
                receipt_contract.EXPECTED_IMPORT_ARGUMENTS
            ),
            expected_environment=dict(
                receipt_contract.EXPECTED_ACTION_ENVIRONMENT
            ),
            expected_working_directory="/",
            expected_exit_status=0,
            expected_products=products.receipt_document.get("products"),
        )
    except (
        KeyError,
        TypeError,
        immutable.ImmutableArtifactError,
        ncu_action_executor.NcuActionExecutorError,
        receipt_contract.DetailsExportReceiptError,
    ) as exc:
        raise DetailsExportPersistenceError(
            f"Details export receipt semantic contract is invalid: {exc}"
        ) from exc

    raw_bindings = products.receipt_document.get("products")
    if type(raw_bindings) is not list or len(raw_bindings) != 5:
        _fail("Details export receipt product bindings are incomplete")
    bindings: dict[str, dict[str, Any]] = {}
    relative_paths: list[PurePosixPath] = []
    for expected_role, raw_binding in zip(
        _EXPECTED_BASENAMES, raw_bindings, strict=True,
    ):
        if type(raw_binding) is not dict or raw_binding.get("role") != expected_role:
            _fail("Details export receipt product role order differs")
        relative = _relative(
            raw_binding.get("relative_path"), label=f"{expected_role} path",
        )
        if relative.name != _EXPECTED_BASENAMES[expected_role]:
            _fail(f"{expected_role} basename differs from the launch-set contract")
        bindings[expected_role] = raw_binding
        relative_paths.append(relative)
    receipt_relative = _relative(
        receipt_relative_path, label="Details export receipt path",
    )
    if receipt_relative.name != _RECEIPT_BASENAME:
        _fail("Details export receipt basename differs from the launch-set contract")
    all_paths = [*relative_paths, receipt_relative]
    if len({path.as_posix() for path in all_paths}) != len(all_paths):
        _fail("Details export publication paths are not unique")
    parents = {path.parent.as_posix() for path in all_paths}
    if len(parents) != 1 or next(iter(parents)) == ".":
        _fail("Details export products do not share one non-root relative parent")

    payloads = {
        "details_csv": products.csv_bytes,
        "details_stderr": products.stderr_bytes,
        "launch_inventory": products.inventory_bytes,
        "launch_metrics": products.metrics_bytes,
    }
    for role, payload in payloads.items():
        binding = bindings[role]
        if (
            immutable.sha256_bytes(payload) != binding.get("sha256")
            or len(payload) != binding.get("size_bytes")
        ):
            _fail(f"{role} producer bytes differ from the receipt binding")
    try:
        ncu_launch_documents.validate_launch_inventory_document(
            products.inventory_document,
            expected_launch_count=expected_launch_count,
            report_sha256=bindings["ncu_report"]["sha256"],
            protocol=protocol,
        )
        ncu_launch_documents.validate_launch_metrics_document(
            products.metrics_document,
            csv_text=csv_text,
            inventory=products.inventory_document,
            registry=registry,
            percentage_metric_names=percentage_metric_names,
            unit_metric_names=unit_metric_names,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        immutable.ImmutableArtifactError,
    ) as exc:
        raise DetailsExportPersistenceError(
            f"Details export launch semantics are invalid: {exc}"
        ) from exc
    return root, receipt_relative, bindings, payloads


def persist_details_export_products(
    *,
    artifact_root: Path,
    receipt_relative_path: str,
    products: ncu_launch_documents.DetailsExportProducts,
    expected_preflight_sha256: str,
    expected_source_commit: str,
    expected_capture_identity_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    expected_launch_count: int,
    protocol: Mapping[str, Any],
    registry: Any,
    percentage_metric_names: Iterable[str],
    unit_metric_names: Iterable[str],
) -> Path:
    """Publish and immediately revalidate one Details export product set.

    Each leaf is committed atomically without replacement. An interrupted prefix
    and every uncertain partial remain as evidence. The receipt is published last
    and is not a campaign-DONE admission claim.
    """

    try:
        root, receipt_relative, bindings, payloads = _validated_plan(
            artifact_root=artifact_root,
            receipt_relative_path=receipt_relative_path,
            products=products,
            expected_preflight_sha256=expected_preflight_sha256,
            expected_source_commit=expected_source_commit,
            expected_capture_identity_sha256=(
                expected_capture_identity_sha256
            ),
            expected_ncu_identity=expected_ncu_identity,
            expected_launch_count=expected_launch_count,
            protocol=protocol,
            registry=registry,
            percentage_metric_names=percentage_metric_names,
            unit_metric_names=unit_metric_names,
        )
        report_relative = _relative(
            bindings["ncu_report"]["relative_path"], label="NCU report path",
        )
        parent_relative = report_relative.parent
        parent = root.joinpath(*parent_relative.parts)
        report_path = root.joinpath(*report_relative.parts)
        with contextlib.ExitStack() as retained_stack:
            pinned_parent = retained_stack.enter_context(
                immutable.PinnedDirectoryReader(parent)
            )
            report = retained_stack.enter_context(
                immutable.open_retained_regular(report_path)
            )
            _require_retained_identity(
                report,
                binding=bindings["ncu_report"],
                expected_bytes=None,
                require_read_only=False,
            )
            retained_outputs: dict[str, immutable.RetainedRegular] = {}
            for role in _GENERATED_ROLES:
                binding = bindings[role]
                relative = _relative(
                    binding["relative_path"], label=f"{role} path",
                )
                if relative.parent != parent_relative:
                    _fail(f"{role} escaped the pinned publication parent")
                published = immutable.publish_regular_file(
                    parent,
                    relative.name,
                    payloads[role],
                    partial_suffix=_PARTIAL_SUFFIX,
                    pinned_parent=pinned_parent,
                )
                retained = retained_stack.enter_context(
                    immutable.open_retained_regular(published)
                )
                _require_retained_identity(
                    retained,
                    binding=binding,
                    expected_bytes=payloads[role],
                    require_read_only=True,
                )
                retained_outputs[role] = retained

            for retained in (report, *retained_outputs.values()):
                retained.verify_unchanged()
            receipt_path = immutable.publish_regular_file(
                parent,
                receipt_relative.name,
                products.receipt_bytes,
                partial_suffix=_PARTIAL_SUFFIX,
                pinned_parent=pinned_parent,
            )
            retained_receipt = retained_stack.enter_context(
                immutable.open_retained_regular(receipt_path)
            )
            receipt_binding = {
                "role": "details_export_receipt",
                "sha256": immutable.sha256_bytes(products.receipt_bytes),
                "size_bytes": len(products.receipt_bytes),
            }
            retained_receipt_bytes = _require_retained_identity(
                retained_receipt,
                binding=receipt_binding,
                expected_bytes=products.receipt_bytes,
                require_read_only=True,
            )
            retained_receipt_model = _strict_model(
                retained_receipt_bytes, label="retained Details export receipt",
            )
            if retained_receipt_model != products.receipt_document:
                _fail("retained Details export receipt model differs")

            rebuilt = [
                receipt_contract.product_binding(
                    role="ncu_report",
                    relative_path=bindings["ncu_report"]["relative_path"],
                    sha256=report.sha256,
                    size_bytes=report.size_bytes,
                )
            ]
            for role in _GENERATED_ROLES:
                retained = retained_outputs[role]
                rebuilt.append(receipt_contract.product_binding(
                    role=role,
                    relative_path=bindings[role]["relative_path"],
                    sha256=retained.sha256,
                    size_bytes=retained.size_bytes,
                ))
            if rebuilt != products.receipt_document["products"]:
                _fail("retained Details export product set differs from receipt")
            for retained in (
                report,
                *retained_outputs.values(),
                retained_receipt,
            ):
                retained.verify_unchanged()
            pinned_parent.verify_path()
            return receipt_path
    except RecursionError as exc:
        raise DetailsExportPersistenceError(
            "Details export model is cyclic or excessively nested"
        ) from exc
    except (
        OSError,
        immutable.ImmutableArtifactError,
        receipt_contract.DetailsExportReceiptError,
    ) as exc:
        raise DetailsExportPersistenceError(str(exc)) from exc
