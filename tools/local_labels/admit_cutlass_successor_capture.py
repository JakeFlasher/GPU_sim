#!/usr/bin/env python3
"""Validate external CUTLASS successor receipts and emit an additive overlay.

This tool never edits the pinned successor cohort.  It reads the external
receipt directory, validates the immutable 237f17c bundle, and writes one
explicitly requested overlay.  A row is capture-ready only after an exact NCU
and NVBit join, including the runner's seven-field identity and launch identity.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import build_cutlass_successor_cohort as cohort  # noqa: E402
from tools.local_labels import campaign_preflight  # noqa: E402
from tools.local_labels import delegated_cgroup  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import label_driver_authority  # noqa: E402
from tools.local_labels import ncu_action_executor as ncu_executor  # noqa: E402
from tools.local_labels.capture_toolchain_contract import (  # noqa: E402
    ncu_version_matches,
)
from tools.nvbit_tracer import external_tracer_build as external_build  # noqa: E402
from tools.nvbit_tracer.protobuf_wire_preflight import (  # noqa: E402
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    preflight_trace_protobuf,
)


PINNED_BUNDLE_COMMIT = "237f17cd34728b31ea752062e7ea7843a3a2603a"
RECEIPT_SCHEMA_PATHS_BY_VERSION = {
    2: Path(
        "workloads/simbench_bw/schemas/"
        "cutlass_successor_capture_receipt.schema.json"
    ),
    3: Path(
        "workloads/simbench_bw/schemas/"
        "cutlass_successor_capture_receipt.v3.schema.json"
    ),
}
RECEIPT_SCHEMA_IDS_BY_VERSION = {
    2: "cutlass_successor_capture_receipt.schema.json",
    3: "cutlass_successor_capture_receipt.v3.schema.json",
}
RECEIPT_SCHEMA_SHA256_BY_VERSION = {
    2: "1833c68061a4e0d52f372ec61c5a445235071d918bf14dc2b2ddaec44770d486",
    3: "9ff5b0d86fd95b908ce05adac3f96c1ea5aac397d9fbc6b729c3c512586b8865",
}
CAPTURE_PRODUCER_SHA256_BY_RECEIPT_VERSION = {
    # v2 receipts were emitted by the final repository-tracer producer before
    # retained external NVBit phase tools became part of the receipt contract.
    2: "a67f4a0f84f5ee73cd3f5998e4da79927ac2bee05148cecd08216ddab5694e86",
    3: "bad7125c3b77e8228952408a211cabb1f8124b48e6dfbe1b614d042c1474236a",
}
ACTIVE_RECEIPT_SCHEMA_VERSION = 3
RECEIPT_SCHEMA_REL = RECEIPT_SCHEMA_PATHS_BY_VERSION[
    ACTIVE_RECEIPT_SCHEMA_VERSION
]
OVERLAY_SCHEMA_PATHS_BY_VERSION = {
    1: Path(
        "workloads/simbench_bw/schemas/"
        "cutlass_successor_capture_admission.schema.json"
    ),
    2: Path(
        "workloads/simbench_bw/schemas/"
        "cutlass_successor_capture_admission.v2.schema.json"
    ),
}
OVERLAY_SCHEMA_IDS_BY_VERSION = {
    1: "cutlass_successor_capture_admission.schema.json",
    2: "cutlass_successor_capture_admission.v2.schema.json",
}
OVERLAY_SCHEMA_SHA256_BY_VERSION = {
    1: "a5b8f636e32ee2979fe0d577c16c4610f3faa259b9ac687331dc6438031cb064",
    2: "7f8f7368bbb080f7fd26c6e2dc8ab173fce455e353b2dec179dda13a8943e27b",
}
OVERLAY_IDS_BY_VERSION = {
    1: "cutlass_successor_capture_admission_sm86_v1",
    2: "cutlass_successor_capture_admission_sm86_v2",
}
ACTIVE_OVERLAY_SCHEMA_VERSION = 2
OVERLAY_SCHEMA_REL = OVERLAY_SCHEMA_PATHS_BY_VERSION[
    ACTIVE_OVERLAY_SCHEMA_VERSION
]
OVERLAY_ID = OVERLAY_IDS_BY_VERSION[ACTIVE_OVERLAY_SCHEMA_VERSION]
DEFAULT_METRIC_VERSION = "ncu_2026.1.x"
RECEIPT_SUFFIX = ".capture_receipt.json"
LOCAL_LABEL_SIDECAR_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/local_label_sidecar.schema.json"
)
TRACE_CONTRACT_IDS_BY_RECEIPT_VERSION = {
    2: "sm86_addr_carry_v1",
    3: "sm86_addr_carry_shmem_split_v2",
}
TRACE_CONTRACT_ID = TRACE_CONTRACT_IDS_BY_RECEIPT_VERSION[
    ACTIVE_RECEIPT_SCHEMA_VERSION
]
TRACE_MEMORY_PAYLOAD_CLASSES = [1, 2, 3, 6, 8]
NVBIT_TRACE_BINARY_VERSION = 86
NVBIT_VERSION = "1.8"
NVBIT_TRACE_SUMMARY_PRODUCER = "nvbit_tracer"
NVBIT_CAPTURE_STDERR_PROTOCOL = (
    "nvbit_1_8_sm86_empty_or_startup_diagnostics_v1"
)
NVBIT_CAPTURE_STDOUT_NAME = "capture_process.stdout.bin"
NVBIT_CAPTURE_STDERR_NAME = "capture_process.stderr.bin"
_NVBIT_CAPTURE_STDERR_PATTERN = re.compile(
    rb"WARNING: Do not call CUDA memory allocation in nvbit_at_ctx_init\(\)\. "
    rb"It will cause deadlocks\. Do them in nvbit_tool_init\(\)\. If you "
    rb"encounter deadlocks, remove CUDA API calls to debug\.\n"
    rb"INFO: tracer binary_version initialised per-context from device compute "
    rb"capability at tool-init: binary_version=86 \(ctx=0x[1-9a-f][0-9a-f]*\)\n"
    rb"INFO: cuFuncGetAttribute\(BINARY_VERSION\) returned attr_bv=0 "
    rb"\(CUresult=400\); falling back to device compute capability 8\.6 -> "
    rb"binary_version=86\n"
)
GPU_NAME = "NVIDIA GeForce RTX 3090 Ti"
GPU_UUID_PATTERN = (
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
LABEL_DRIVER_REL = label_driver_authority.LABEL_DRIVER_REL
LABEL_DRIVER_SHA256 = label_driver_authority.LABEL_DRIVER_SHA256
LABEL_DRIVER_MODE = label_driver_authority.LABEL_DRIVER_MODE
NCU_ACTION_EXECUTOR_SHA256_BY_RECEIPT_VERSION = {
    2: "3515165fafd2084e7fb1d24402b0d8be751f18714f7827cabb1a603344b6c8ac",
    3: "ae2e0a23babfc0789e9297815633171506a8a9be1108e2c62eeb21c0b0013ce4",
}
NCU_ACTION_EXECUTOR_SHA256 = NCU_ACTION_EXECUTOR_SHA256_BY_RECEIPT_VERSION[
    ACTIVE_RECEIPT_SCHEMA_VERSION
]
NVBIT_TRACER_SHA256_BY_RECEIPT_VERSION = {
    2: "de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d",
}
NVBIT_TRACER_SIZE_BYTES_BY_RECEIPT_VERSION = {
    2: 15_942_128,
}
CGROUP_CONTAINMENT_SUPERVISOR_SHA256 = (
    "1b4cbe75929eb6f8f98c55f300c964ba1592fa5547335cdcf2324fe0d4a0d452"
)
CAPTURE_PRODUCER_SELF_SHA256_AUTHORITY = (
    "d2754d7e4ef33877c0962d2ca18431981b4b50d6c2fe3b20e893e5cb9a951b31"
)
NCU_LAUNCHER_PATH = (
    "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu"
)
NCU_LAUNCHER_SHA256 = (
    "a44ff2c735c4dcf66c78f3f430155805d6a0b1f74f26331f98ecec03bfbf3f1a"
)
NCU_EXECUTABLE_PATH = (
    "/usr/local/cuda-13.2/nsight-compute-2026.1.0/"
    "target/linux-desktop-glibc_2_11_3-x64/ncu"
)
NCU_EXECUTABLE_SHA256 = (
    "c966a9a9467ae1b4a37fd6be88991a57dc3ec790249c699d68ef74df8f4d62ac"
)
NCU_VERSION_LINE = "Version 2026.1.0.0 (build 37166530) (public-release)"
NVBIT_TRACER_REL = Path("tools/nvbit_tracer/tracer_tool/tracer_tool.so")
NVBIT_PB_LIB_REL = Path(".toolchain/pb/lib")
NVBIT_COLLECTION_PATH = "/usr/local/cuda-13.2/bin:/usr/bin:/bin"
NCU_COLLECTION_REQUIRED_ENVIRONMENT_NAMES = (
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "PATH",
    "PWD",
    "PYTHONNOUSERSITE",
    "LABEL_DRIVER_GPU_INDEX",
    "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE",
)
NCU_COLLECTION_OPTIONAL_ENVIRONMENT_NAMES = (
    "CC",
    "CUDAHOSTCXX",
    "CUDA_HOME",
    "CUTLASS_DIR",
    "CXX",
    "FLASHINFER_TRACE_DIR",
    "LD_SOURCE_COMMIT_OVERRIDE",
    "NVCC_CCBIN",
    "PYTORCH_ALLOC_CONF",
    "SOL_EXECBENCH_CAPTURE_REQUEST",
    "TMPDIR",
)
RECEIPT_TREE_MAX_FILES = 32_768
RECEIPT_TREE_MAX_DIRECTORIES = 4_096
RECEIPT_TREE_MAX_TOTAL_FILE_BYTES = 1024 * 1024 * 1024
RECEIPT_TREE_MAX_TOTAL_RELATIVE_PATH_BYTES = 32 * 1024 * 1024

BUNDLE_FILES = (
    cohort.VALIDATION_REL,
    cohort.BUDGETS_REL,
    cohort.ROSTER_REL,
    cohort.MATRIX_REL,
    cohort.VALIDATION_SCHEMA_REL,
    cohort.BUDGETS_SCHEMA_REL,
    cohort.ROSTER_SCHEMA_REL,
    cohort.MATRIX_SCHEMA_REL,
)
BUNDLE_HASHES = {
    cohort.VALIDATION_REL.as_posix(): "2682e21a0c4874f104a0480673fb965ef2e1cb16a4d8a7ef92c34a7b476ccb0e",
    cohort.BUDGETS_REL.as_posix(): "374abe73648f53c5c4e249c11a84f6b0b1ab3eb8ce9e331220919a462d01f551",
    cohort.ROSTER_REL.as_posix(): "210b8678c64208ea741c91cdd6ce1dce5d55bd29a169dbb0a5a9aa2274ce7185",
    cohort.MATRIX_REL.as_posix(): "ec0dc06429299371ec08adf9132eb1f843d99a7ba846d0ada712b747bc7a1879",
}


ADMISSION_REFUSED = "capture_admission_refused"
ADMISSION_RESOURCE_UNAVAILABLE = "capture_admission_resource_unavailable"


class AdmissionError(RuntimeError):
    """External capture admission failed closed."""

    def __init__(
        self,
        detail: str,
        *,
        code: str = ADMISSION_REFUSED,
    ) -> None:
        self.code = code
        self.detail = detail
        rendered = detail if code == ADMISSION_REFUSED else f"{code}: {detail}"
        super().__init__(rendered)


def _memory_refusal(stage: str, cause: MemoryError) -> AdmissionError:
    error = AdmissionError(
        f"{stage} exhausted memory",
        code=ADMISSION_RESOURCE_UNAVAILABLE,
    )
    error.__cause__ = cause
    return error


class _ReceiptGeneration:
    """One descriptor-relative receipt-tree generation held through admission."""

    def __init__(
        self,
        root: Path,
        reader: immutable.PinnedDirectoryReader,
        snapshot: immutable.PinnedTreeSnapshot,
    ) -> None:
        self.root = root
        self.reader = reader
        try:
            self._metadata = dict(snapshot.files)
            self._payloads: dict[str, bytes] = {}
        except MemoryError as exc:
            raise _memory_refusal("receipt retained generation", exc) from exc

    @property
    def paths(self) -> list[Path]:
        try:
            return [self.root / relative for relative in self._metadata]
        except MemoryError as exc:
            raise _memory_refusal("receipt retained path inventory", exc) from exc

    def _relative(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise AdmissionError(
                f"receipt generation path escapes root: {path}"
            ) from exc
        if relative not in self._metadata:
            raise AdmissionError(
                f"receipt generation does not contain regular file: {relative}"
            )
        return relative

    def read_bytes(self, path: Path) -> bytes:
        try:
            relative = self._relative(path)
            metadata = self._metadata[relative]
            if relative not in self._payloads:
                payload = self.reader.read_snapshot(
                    relative, expected_metadata=metadata,
                ).payload
                self._payloads[relative] = payload
            else:
                self.reader.verify_regular(relative, metadata)
            return self._payloads[relative]
        except MemoryError as exc:
            raise _memory_refusal("receipt retained byte cache", exc) from exc

    def inventory(self) -> list[dict[str, Any]]:
        try:
            inventory: list[dict[str, Any]] = []
            for relative in self._metadata:
                payload = self.read_bytes(self.root / relative)
                inventory.append({
                    "path": relative,
                    "sha256": sha256_bytes(payload),
                    "size_bytes": len(payload),
                })
            return inventory
        except MemoryError as exc:
            raise _memory_refusal("receipt retained inventory", exc) from exc


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_parser_diagnostics(
    value: Any,
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    if type(value) is not dict:
        raise AdmissionError(f"{label} parser diagnostics are not an object")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for name, records in value.items():
        if type(name) is not str or not isinstance(records, list):
            raise AdmissionError(f"{label} parser diagnostics are malformed")
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            if (
                type(record) is not dict
                or set(record) != {"metric_label", "raw_value"}
                or type(record["metric_label"]) is not str
                or isinstance(record["raw_value"], bool)
                or not isinstance(record["raw_value"], (int, float))
            ):
                raise AdmissionError(f"{label} parser diagnostics are malformed")
            try:
                json.dumps(record, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise AdmissionError(
                    f"{label} parser diagnostics are non-finite"
                ) from exc
            normalized_records.append(dict(record))
        normalized[name] = sorted(
            normalized_records,
            key=lambda record: (
                record["metric_label"],
                record["raw_value"],
            ),
        )
    return normalized


_CAPTURE_PRODUCER_SELF_SEAL = re.compile(
    rb'CAPTURE_PRODUCER_SELF_SHA256 = \(\n    "([0-9a-f]{64})"\n\)'
)


def _capture_producer_self_digest(payload: bytes) -> str:
    matches = list(_CAPTURE_PRODUCER_SELF_SEAL.finditer(payload))
    if len(matches) != 1:
        raise AdmissionError(
            "successor capture producer has no unique embedded self-seal"
        )
    match = matches[0]
    declared = match.group(1).decode("ascii", errors="strict")
    start, end = match.span(1)
    normalized = payload[:start] + (b"0" * 64) + payload[end:]
    if sha256_bytes(normalized) != declared:
        raise AdmissionError("successor capture producer self-seal mismatch")
    return declared


def _capture_producer_sha256(receipt_schema_version: int) -> str:
    expected_sha256 = CAPTURE_PRODUCER_SHA256_BY_RECEIPT_VERSION.get(
        receipt_schema_version
    )
    if expected_sha256 is None:
        raise AdmissionError(
            "unsupported receipt version for capture producer authority: "
            f"{receipt_schema_version!r}"
        )
    if receipt_schema_version != ACTIVE_RECEIPT_SCHEMA_VERSION:
        # Historical receipt producers are immutable hash authorities.  They
        # are intentionally not substituted with the active producer bytes.
        return expected_sha256

    source = ROOT / "tools/local_labels/capture_cutlass_successor_receipts.py"
    try:
        with immutable.open_retained_regular(source) as retained:
            payload = retained.read_bytes()
            declared = _capture_producer_self_digest(payload)
            if declared != CAPTURE_PRODUCER_SELF_SHA256_AUTHORITY:
                raise AdmissionError(
                    "successor capture producer differs from self-seal authority"
                )
            if retained.sha256 != expected_sha256:
                raise AdmissionError(
                    "successor capture producer differs from receipt-version authority"
                )
            return retained.sha256
    except immutable.ImmutableArtifactError as exc:
        raise AdmissionError(
            f"successor capture producer immutable identity failed: {exc}"
        ) from exc


def _ncu_action_executor_sha256(receipt_schema_version: int) -> str:
    expected_sha256 = NCU_ACTION_EXECUTOR_SHA256_BY_RECEIPT_VERSION.get(
        receipt_schema_version
    )
    if expected_sha256 is None:
        raise AdmissionError(
            "unsupported receipt version for NCU executor authority: "
            f"{receipt_schema_version!r}"
        )
    if receipt_schema_version != ACTIVE_RECEIPT_SCHEMA_VERSION:
        return expected_sha256

    try:
        with immutable.open_retained_regular(
            Path(ncu_executor.__file__).resolve(strict=True)
        ) as retained:
            if retained.sha256 != expected_sha256:
                raise AdmissionError(
                    "successor NCU executor differs from receipt-version authority"
                )
            return retained.sha256
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise AdmissionError(
            f"successor NCU executor immutable identity failed: {exc}"
        ) from exc


def _historical_nvbit_tracer_authority(
    receipt_schema_version: int,
) -> dict[str, Any]:
    sha256 = NVBIT_TRACER_SHA256_BY_RECEIPT_VERSION.get(receipt_schema_version)
    size_bytes = NVBIT_TRACER_SIZE_BYTES_BY_RECEIPT_VERSION.get(
        receipt_schema_version
    )
    if sha256 is None or size_bytes is None:
        raise AdmissionError(
            "unsupported receipt version for historical NVBit tracer authority: "
            f"{receipt_schema_version!r}"
        )
    return {
        "identity_kind": "repository_tracer",
        "path": NVBIT_TRACER_REL.as_posix(),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _regular(path: Path, label: str) -> None:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise AdmissionError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink():
        raise AdmissionError(f"{label} is a symlink: {path}")
    if not path.is_file() or stat.st_nlink != 1:
        raise AdmissionError(f"{label} is not a single regular file: {path}")


def _load_json(
    path: Path,
    label: str,
    generation: _ReceiptGeneration | None = None,
) -> dict[str, Any]:
    _regular(path, label)
    try:
        payload = (
            generation.read_bytes(path)
            if generation is not None
            else path.read_bytes()
        )
        value = immutable.strict_json_loads(payload, label=label)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise AdmissionError(f"invalid JSON {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"JSON document is not an object: {label}")
    return value


def _check_dt(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise AdmissionError(f"{label} is not an ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdmissionError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise AdmissionError(f"{label} must include a timezone")


def _validate_schema(path: Path, document: dict[str, Any], label: str) -> None:
    schema = _load_json(path, label)
    try:
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(document)
    except jsonschema.ValidationError as exc:
        raise AdmissionError(
            f"{label} schema validation failed: {exc.message}"
        ) from exc


def _validate_registered_overlay_schema(
    document: dict[str, Any], label: str,
) -> None:
    version = document.get("schema_version")
    if type(version) is not int or version not in OVERLAY_SCHEMA_PATHS_BY_VERSION:
        raise AdmissionError(
            f"{label} has unsupported overlay schema_version: {version!r}"
        )
    relative = OVERLAY_SCHEMA_PATHS_BY_VERSION[version]
    try:
        with immutable.open_retained_regular(ROOT / relative) as retained:
            payload = retained.read_bytes()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise AdmissionError(
            f"{label} schema authority is unavailable: {relative}: {exc}"
        ) from exc
    if sha256_bytes(payload) != OVERLAY_SCHEMA_SHA256_BY_VERSION[version]:
        raise AdmissionError(
            f"{label} schema authority hash differs for version {version}"
        )
    try:
        schema = immutable.strict_json_loads(
            payload, label=f"capture admission overlay schema v{version}"
        )
    except immutable.ImmutableArtifactError as exc:
        raise AdmissionError(
            f"{label} schema authority is invalid: {exc}"
        ) from exc
    if (
        type(schema) is not dict
        or schema.get("$id") != OVERLAY_SCHEMA_IDS_BY_VERSION[version]
        or schema.get("properties", {}).get("schema_version", {}).get("const")
        != version
    ):
        raise AdmissionError(
            f"{label} schema authority identity differs for version {version}"
        )
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(document)
    except jsonschema.SchemaError as exc:
        raise AdmissionError(
            f"{label} schema authority is invalid: {exc.message}"
        ) from exc
    except jsonschema.ValidationError as exc:
        raise AdmissionError(
            f"{label} schema validation failed: {exc.message}"
        ) from exc


def _validate_registered_receipt_schema(
    document: dict[str, Any], label: str,
) -> dict[str, Any]:
    version = document.get("schema_version")
    if type(version) is not int or version not in RECEIPT_SCHEMA_PATHS_BY_VERSION:
        raise AdmissionError(
            f"{label} has unsupported receipt schema_version: {version!r}"
        )
    relative = RECEIPT_SCHEMA_PATHS_BY_VERSION[version]
    path = ROOT / relative
    try:
        with immutable.open_retained_regular(path) as retained:
            payload = retained.read_bytes()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise AdmissionError(
            f"{label} schema authority is unavailable: {relative}: {exc}"
        ) from exc
    if sha256_bytes(payload) != RECEIPT_SCHEMA_SHA256_BY_VERSION[version]:
        raise AdmissionError(
            f"{label} schema authority hash differs for version {version}"
        )
    try:
        schema = immutable.strict_json_loads(
            payload, label=f"receipt schema v{version}"
        )
    except immutable.ImmutableArtifactError as exc:
        raise AdmissionError(
            f"{label} schema authority is invalid: {exc}"
        ) from exc
    if (
        type(schema) is not dict
        or schema.get("$id") != RECEIPT_SCHEMA_IDS_BY_VERSION[version]
        or schema.get("properties", {}).get("schema_version", {}).get("const")
        != version
    ):
        raise AdmissionError(
            f"{label} schema authority identity differs for version {version}"
        )
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(document)
    except jsonschema.SchemaError as exc:
        raise AdmissionError(
            f"{label} schema authority is invalid: {exc.message}"
        ) from exc
    except jsonschema.ValidationError as exc:
        raise AdmissionError(
            f"{label} schema validation failed: {exc.message}"
        ) from exc
    return {
        "schema_version": version,
        "schema_id": RECEIPT_SCHEMA_IDS_BY_VERSION[version],
        "path": relative.as_posix(),
        "sha256": RECEIPT_SCHEMA_SHA256_BY_VERSION[version],
        "size_bytes": len(payload),
    }


def _git_blob(path: Path) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{PINNED_BUNDLE_COMMIT}:{path.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AdmissionError(f"pinned successor bundle is unavailable: {path}")
    return result.stdout


def validate_immutable_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Require the current four manifests and four source schemas to equal 237f17c."""
    expected_documents: dict[str, dict[str, Any]] = {}
    for path in BUNDLE_FILES:
        current = ROOT / path
        _regular(current, f"pinned bundle file {path}")
        expected = _git_blob(path)
        actual = current.read_bytes()
        if actual != expected:
            raise AdmissionError(f"pinned successor bundle drifted: {path}")
        expected_hash = BUNDLE_HASHES.get(path.as_posix())
        if expected_hash is not None:
            if sha256_bytes(actual) != expected_hash:
                raise AdmissionError(f"pinned successor manifest hash drifted: {path}")
            try:
                expected_document = json.loads(expected.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise AdmissionError(
                    f"pinned successor manifest is invalid JSON: {path}"
                ) from exc
            if not isinstance(expected_document, dict):
                raise AdmissionError(
                    f"pinned successor manifest is not an object: {path}"
                )
            expected_documents[path.as_posix()] = expected_document
    try:
        documents = cohort.validate_existing_bundle()
    except MemoryError:
        raise
    except Exception as exc:  # normalize helper failures at this boundary
        raise AdmissionError(f"pinned successor bundle validation failed: {exc}") from exc
    expected_return = tuple(
        expected_documents[path.as_posix()] for path in BUNDLE_FILES[:4]
    )
    if documents != expected_return:
        raise AdmissionError(
            "validated successor bundle documents differ from pinned Git bytes"
        )
    return documents


def _external_root(raw: Path) -> Path:
    if not raw.is_absolute():
        raise AdmissionError("receipt root must be an absolute external path")
    candidate = Path(raw.anchor)
    for component in raw.parts[1:]:
        candidate /= component
        if candidate.is_symlink():
            raise AdmissionError(f"receipt root path contains a symlink: {candidate}")
    if raw.is_symlink():
        raise AdmissionError("receipt root is a symlink")
    root = raw.resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise AdmissionError("receipt root must be outside the repository")
    if not root.is_dir():
        raise AdmissionError(f"receipt root is not a directory: {raw}")
    return root


def _safe_receipt_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or "\\" in raw:
        raise AdmissionError(f"{label} is not a safe relative path")
    lexical = root / raw
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"{label} escapes receipt root: {raw!r}") from exc
    _regular(resolved, label)
    if resolved != lexical:
        raise AdmissionError(f"{label} resolves through a symlink: {raw!r}")
    return resolved


def _receipt_ref(
    generation: _ReceiptGeneration,
    root: Path,
    relative: str,
    seal: str,
    *,
    schema_authority: dict[str, Any],
    nvbit_tool: dict[str, Any] | None,
) -> dict[str, Any]:
    path = _safe_receipt_path(root, relative, "receipt path")
    payload = generation.read_bytes(path)
    reference = {
        "path": relative,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "receipt_seal_sha256": seal,
        "schema_authority": dict(schema_authority),
    }
    if nvbit_tool is not None:
        reference["nvbit_tool"] = dict(nvbit_tool)
    return reference


def _expected_rows(matrix: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in matrix["rows"]:
        _require_declared_only_launch_contract(row)
        key = (row["matrix_key"], row["variant_id"])
        if key in rows:
            raise AdmissionError(f"duplicate matrix row identity: {key}")
        rows[key] = row
    if len(rows) != 3:
        raise AdmissionError("successor matrix must contain exactly three rows")
    return rows


def _expected_roster_rows(roster: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in roster["rows"]:
        key = (row["matrix_key"], row["variant_id"])
        if key in rows:
            raise AdmissionError(f"duplicate roster row identity: {key}")
        rows[key] = row
    return rows


def _verify_process(process: dict[str, Any], label: str) -> None:
    expected = canonical_sha({k: v for k, v in process.items() if k != "process_identity_sha256"})
    if process["process_identity_sha256"] != expected:
        raise AdmissionError(f"{label} process identity seal mismatch")
    if process["boot_id_sha256"] == "0" * 64:
        raise AdmissionError(f"{label} process boot identity is unavailable")
    containment = process["containment"]
    if containment != {
        "kind": delegated_cgroup.CGROUP_V2_KILL_V1,
        "parent_cgroup_sha256": containment["parent_cgroup_sha256"],
        "cgroup_inode": containment["cgroup_inode"],
        "terminal_populated_zero": True,
        "leaf_removed": True,
    }:
        raise AdmissionError(f"{label} process containment is not terminal")
    if containment["parent_cgroup_sha256"] == "0" * 64:
        raise AdmissionError(f"{label} parent cgroup identity is unavailable")
    _check_dt(process["started_at_utc"], f"{label} process started_at_utc")


def _require_declared_only_launch_contract(row: dict[str, Any]) -> None:
    inventory = row.get("launch_inventory")
    shape = row["shape"]
    command = [
        f"./{Path(row['static_launch_inventory']['binary_path']).name}",
        *(str(value) for value in shape["actual_args"]),
    ]
    expected = {
        "status": "declared_only",
        "expected_process_runs": cohort.RUNTIME_REPEAT_COUNT,
        "observed_process_runs": cohort.RUNTIME_REPEAT_COUNT,
        "expected_candidate_launch_count": 1,
        "observed_candidate_launch_count": None,
        "candidate_selection_status": "not_run",
        "dynamic_launch_status": "not_run",
        "nvbit_trace_status": "not_run",
        "static_launch_contract": shape["trace_shape"]["launch_config"],
        "typed_reason": "dynamic_launch_inventory_requires_nvbit_capture",
        "command": command,
        "command_sha256": canonical_sha(command),
    }
    if type(inventory) is not dict or inventory != expected:
        raise AdmissionError(
            "successor launch geometry is not a declared-only contract"
        )


def _verify_launch(
    launch: dict[str, Any], identity: dict[str, Any], label: str,
) -> None:
    expected = canonical_sha({k: v for k, v in launch.items() if k != "launch_identity_sha256"})
    if launch["launch_identity_sha256"] != expected:
        raise AdmissionError(f"{label} launch identity seal mismatch")
    if launch["kernel_signature_sha256"] != identity["kernel_signature_sha256"]:
        raise AdmissionError(f"{label} launch/kernel signature mismatch")
    if (
        launch["launch_index"] != 0
        or type(launch["kernel_name"]) is not str
        or not launch["kernel_name"]
    ):
        raise AdmissionError(f"{label} observed launch identity is invalid")
    for field in ("grid_dim", "block_dim"):
        dimensions = launch[field]
        if (
            type(dimensions) is not list
            or len(dimensions) != 3
            or any(type(value) is not int or value <= 0 for value in dimensions)
        ):
            raise AdmissionError(f"{label} observed launch geometry is invalid")


def _import_ncu_report_csv(
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    report_path: Path,
) -> tuple[str | None, str | None, Any]:
    from tools.local_labels import parse_ncu_metrics

    imported, error = parse_ncu_metrics._import_report_csv(
        expected_ncu_identity,
        report_path,
        require_cgroup_containment=True,
    )
    if error is not None or imported is None:
        return None, None, error
    return imported.text, imported.report_sha256, None


def _ncu_dimension(value: Any, label: str) -> list[int]:
    if type(value) is not str:
        raise AdmissionError(f"NCU report {label} is not a string")
    text = value.strip()
    if text.startswith("(") or text.endswith(")"):
        if not (text.startswith("(") and text.endswith(")")):
            raise AdmissionError(f"NCU report {label} parentheses are malformed")
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3 or any(
        re.fullmatch(r"0|[1-9][0-9]*", part) is None for part in parts
    ):
        raise AdmissionError(f"NCU report {label} is not a 3D dimension")
    dimensions = [int(part) for part in parts]
    if any(value <= 0 for value in dimensions):
        raise AdmissionError(f"NCU report {label} must be positive")
    return dimensions


def _ncu_device_name(value: Any) -> str:
    if value not in {GPU_NAME, "0", f"{GPU_NAME} (0)"}:
        raise AdmissionError("NCU report device differs from selected GPU 0")
    return GPU_NAME


def _verify_ncu_report(
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    report_path: Path,
    report_sha256: str,
    sidecar: dict[str, Any],
    observation: dict[str, Any],
    expected_launch_identity: dict[str, Any],
    allowed_kernel_names: set[str],
) -> dict[str, Any]:
    from tools.local_labels import metric_registry, parse_ncu_metrics

    csv_text, imported_report_sha256, import_error = _import_ncu_report_csv(
        expected_ncu_identity, report_path
    )
    if import_error is not None or csv_text is None:
        detail = getattr(import_error, "detail", repr(import_error))
        raise AdmissionError(f"NCU report import failed: {detail}")
    if imported_report_sha256 != report_sha256:
        raise AdmissionError("NCU report import used different retained bytes")
    extraction = parse_ncu_metrics.extract_unit_tagged_metrics(
        csv_text,
        metric_registry.load_registry(),
        ["gpc__cycles_elapsed.max", "gpu__time_duration.sum"],
    )
    if not extraction.ok:
        raise AdmissionError(
            f"NCU report metric extraction failed: {extraction.typed_reason}: {extraction.detail}"
        )
    expected_unit_records = [
        {
            "ncu_metric": metric,
            "unit": extraction.metrics[metric]["unit"],
            "value": extraction.metrics[metric]["value"],
        }
        for metric in (
            "gpc__cycles_elapsed.max",
            "gpu__time_duration.sum",
        )
    ]
    declared_unit_records = sidecar.get("unit_tagged_metrics")
    if isinstance(declared_unit_records, list):
        declared_metric_names = [
            record.get("ncu_metric")
            for record in declared_unit_records
            if isinstance(record, dict)
        ]
        if len(declared_metric_names) != len(set(declared_metric_names)):
            raise AdmissionError("NCU sidecar has duplicate unit-tagged metrics")
    if declared_unit_records != expected_unit_records:
        raise AdmissionError(
            "NCU sidecar/report metric disagreement: unit-tagged metric set"
        )
    percentages = parse_ncu_metrics.parse_raw_page_csv(
        csv_text,
        [
            "lts__t_sector_hit_rate.pct",
            "l1tex__t_sector_hit_rate.pct",
        ],
    )
    if not percentages.ok:
        raise AdmissionError(
            "NCU report percentage extraction failed: "
            f"{percentages.typed_reason}: {percentages.detail}"
        )
    percentage_fields = {
        "lts__t_sector_hit_rate.pct": (
            "cache_hit_rate_l2_sector",
            "confidence_interval_l2_sector",
        ),
        "l1tex__t_sector_hit_rate.pct": (
            "cache_hit_rate_l1tex_sector",
            "confidence_interval_l1tex_sector",
        ),
    }
    for metric, (field, interval_field) in percentage_fields.items():
        value = percentages.fraction_values[metric]
        expected_interval = {
            "lower": value,
            "upper": value,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        }
        if (
            sidecar.get(field) != value
            or sidecar.get(interval_field) != expected_interval
        ):
            raise AdmissionError(
                f"NCU sidecar/report percentage disagreement: {metric}"
            )
    expected_diagnostics = parse_ncu_metrics.diagnostics_to_json_shape(
        parse_ncu_metrics._merged_parser_diagnostics_result(
            percentages,
            extraction,
        )
    )
    if expected_diagnostics:
        if _normalized_parser_diagnostics(
            sidecar.get("parser_diagnostics"),
            "NCU sidecar",
        ) != _normalized_parser_diagnostics(
            expected_diagnostics,
            "NCU report",
        ):
            raise AdmissionError("NCU sidecar/report parser diagnostics disagree")
    elif "parser_diagnostics" in sidecar:
        raise AdmissionError("NCU sidecar has unreported parser diagnostics")
    parsed_observation = parse_ncu_metrics.extract_launch_observation(csv_text)
    if (
        "ID" not in parsed_observation["identity_columns"]
        or "Kernel Name" not in parsed_observation["identity_columns"]
        or not parsed_observation["identity"].get("ID")
        or not parsed_observation["identity"].get("Kernel Name")
    ):
        raise AdmissionError("NCU report launch identity is incomplete")
    if (
        observation.get("identity_columns") != parsed_observation["identity_columns"]
        or observation.get("identity") != parsed_observation["identity"]
        or observation.get("report_sha256") != report_sha256
    ):
        raise AdmissionError("NCU report launch identity disagrees with observation artifact")
    parsed_identity = parsed_observation["identity"]
    if parsed_identity.get("ID") != expected_launch_identity["ID"]:
        raise AdmissionError("NCU report launch identity differs from successor launch")
    device = _ncu_device_name(parsed_identity.get("Device"))
    block_dim = _ncu_dimension(parsed_identity.get("Block Size"), "Block Size")
    grid_dim = _ncu_dimension(parsed_identity.get("Grid Size"), "Grid Size")
    if (
        device != expected_launch_identity["Device"]
        or block_dim != expected_launch_identity["Block Size"]
        or grid_dim != expected_launch_identity["Grid Size"]
    ):
        raise AdmissionError("NCU report launch identity differs from successor launch")
    kernel_name = parsed_identity["Kernel Name"]
    if kernel_name not in allowed_kernel_names:
        raise AdmissionError("NCU report kernel is absent from pinned SASS authority")
    return {
        "kernel_name": kernel_name,
        "grid_dim": grid_dim,
        "block_dim": block_dim,
    }


def _sass_kernel_names(row: dict[str, Any]) -> set[str]:
    inventory = row["static_launch_inventory"]
    sass_path = ROOT / inventory["sass_path"]
    _regular(sass_path, "successor SASS authority")
    payload = sass_path.read_bytes()
    if sha256_bytes(payload) != inventory["sass_sha256"]:
        raise AdmissionError("successor SASS authority hash mismatch")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AdmissionError(f"successor SASS authority is not UTF-8: {exc}") from exc
    names = set(re.findall(r"^\s*Function : (\S+)\s*$", text, flags=re.MULTILINE))
    if len(names) != inventory["function_count"]:
        raise AdmissionError("successor SASS function inventory differs")
    return names


def _verify_ncu_artifacts(
    generation: _ReceiptGeneration,
    root: Path,
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    receipt: dict[str, Any],
    row: dict[str, Any],
    artifact_by_role: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Re-derive NCU identity and target-only metric semantics from payloads."""
    cell = Path(row["arch"]) / row["successor_slug"] / "trace_00"
    expected_paths = {
        "ncu_report": cell / "run_0_launch_set.ncu-rep",
        "label_sidecar": cell / "label_0_0.json",
        "label_run_manifest": cell / "label_run_manifest_0_launch_set.json",
        "ncu_launch_observation": (
            cell / "run_0_launch_set.launch_observation.json"
        ),
    }
    for role, expected_path in expected_paths.items():
        if artifact_by_role[role][0]["path"] != expected_path.as_posix():
            raise AdmissionError(
                f"NCU {role} path differs from canonical launch-set namespace"
            )
    try:
        from tools.local_labels import import_label

        sidecar_path = _safe_receipt_path(
            root, artifact_by_role["label_sidecar"][0]["path"], "label sidecar"
        )
        sidecar = _load_json(sidecar_path, "label sidecar", generation)
        result = import_label.validate(sidecar)
        if not result.ok:
            raise AdmissionError(
                f"NCU label sidecar semantic validation failed: {result.typed_reason}"
            )
    except AdmissionError:
        raise
    except MemoryError:
        raise
    except Exception as exc:
        raise AdmissionError(f"NCU label sidecar semantic validation failed: {exc}") from exc

    producer_sidecar_fields = {
        "schema_version",
        "device",
        "slug",
        "shape_point",
        "repeat_index",
        "launch_index",
        "binary_sha256",
        "profiler_family",
        "profiler_version",
        "cuda_version",
        "driver_version",
        "gpu_uuid_sha256",
        "replay_mode",
        "aggregation",
        "warmup_count",
        "cache_hit_rate_l2_sector",
        "cache_hit_rate_l1tex_sector",
        "confidence_interval_l2_sector",
        "confidence_interval_l1tex_sector",
        "measurement_protocol",
        "evidence_class",
        "unit_tagged_metrics",
    }
    observed_sidecar_fields = frozenset(sidecar)
    if observed_sidecar_fields not in {
        frozenset(producer_sidecar_fields),
        frozenset({*producer_sidecar_fields, "parser_diagnostics"}),
    }:
        raise AdmissionError(
            "NCU label sidecar fields differ from the capture producer"
        )
    expected = receipt["identity"]
    environment_path = _safe_receipt_path(
        root,
        artifact_by_role["environment"][0]["path"],
        "capture environment",
    )
    environment = _load_json(
        environment_path,
        "capture environment",
        generation,
    )
    live = environment.get("live_checks")
    if not isinstance(live, dict):
        raise AdmissionError("NCU capture environment has no live-check authority")
    binary_path = ROOT / row["static_launch_inventory"]["binary_path"]
    _regular(binary_path, "successor executable")
    if sha256_bytes(binary_path.read_bytes()) != expected["binary_sha256"]:
        raise AdmissionError("successor executable hash does not match matrix identity")
    expected_fields = {
        "device": expected["arch"],
        "slug": expected["slug"],
        "shape_point": expected["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
        "binary_sha256": expected["binary_sha256"],
        "profiler_family": "nsight_compute",
        "profiler_version": expected_ncu_identity.version_line,
        "cuda_version": "13.2",
        "driver_version": live.get("driver_version"),
        "gpu_uuid_sha256": live.get("gpu_uuid_sha256"),
        "replay_mode": "kernel",
        "aggregation": "kernel",
        "warmup_count": 0,
        "evidence_class": "campaign",
    }
    if any(sidecar.get(key) != value for key, value in expected_fields.items()):
        raise AdmissionError("NCU label sidecar identity does not match successor row")
    protocol = sidecar.get("measurement_protocol")
    expected_protocol = {
        "ncu_clock_control": "none",
        "ncu_cache_control": "all",
        "ncu_replay_mode": "kernel",
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_profile_from_start": True,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 1,
        "host_gpu_index": 0,
        "host_persistence_mode": "Enabled",
        "host_sm_clock_mhz": 1350,
        "host_mem_clock_mhz": 10501,
    }
    if not isinstance(protocol, dict) or protocol != expected_protocol:
        raise AdmissionError("NCU label sidecar controls are not pinned")
    metrics = {
        item.get("ncu_metric"): item
        for item in sidecar.get("unit_tagged_metrics", [])
        if isinstance(item, dict)
    }
    required_units = {
        "gpc__cycles_elapsed.max": "cycles",
        "gpu__time_duration.sum": "ns",
    }
    for metric, unit in required_units.items():
        item = metrics.get(metric)
        if not isinstance(item, dict) or item.get("unit") != unit:
            raise AdmissionError(f"NCU sidecar metric/unit binding is invalid: {metric}")

    run_path = _safe_receipt_path(
        root, artifact_by_role["label_run_manifest"][0]["path"], "label run manifest"
    )
    run = _load_json(run_path, "label run manifest", generation)
    run_expected = {
        "schema_version": 1,
        "source_commit": PINNED_BUNDLE_COMMIT,
        "binary_sha256": expected["binary_sha256"],
        "kernel_signature_sha256": expected["kernel_signature_sha256"],
        "kernel_signature_source": "fallback_legacy",
        "arch": expected["arch"],
        "slug": expected["slug"],
        "shape_point": expected["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }
    if run != run_expected:
        raise AdmissionError(
            "NCU label run manifest differs from the closed successor contract"
        )

    report_artifact = artifact_by_role["ncu_report"][0]
    observation_path = _safe_receipt_path(
        root, artifact_by_role["ncu_launch_observation"][0]["path"],
        "NCU launch observation",
    )
    observation = _load_json(
        observation_path, "NCU launch observation", generation,
    )
    try:
        from tools.local_labels import parse_ncu_metrics

        parse_ncu_metrics.validate_launch_observation_document(
            observation,
            requested_launch_skip=0,
            requested_launch_count=1,
            report_sha256=report_artifact["sha256"],
        )
    except MemoryError:
        raise
    except Exception as exc:
        raise AdmissionError(
            f"NCU launch observation contract is invalid: {exc}"
        ) from exc
    launch = receipt["launch_identity"]
    expected_launch_identity = {
        "ID": "0",
        "Device": GPU_NAME,
        "Kernel Name": launch["kernel_name"],
        "Block Size": launch["block_dim"],
        "Grid Size": launch["grid_dim"],
    }
    return _verify_ncu_report(
        expected_ncu_identity,
        _safe_receipt_path(root, report_artifact["path"], "NCU report"),
        report_artifact["sha256"],
        sidecar,
        observation,
        expected_launch_identity,
        _sass_kernel_names(row),
    )


def _nvbit_kernel_base(value: Any) -> str:
    if type(value) is not str:
        raise AdmissionError("NVBit admitted kernel name is not a string")
    match = re.fullmatch(r"(.+)___(0|[1-9][0-9]*)", value)
    if match is None:
        raise AdmissionError("NVBit admitted kernel lacks its producer variant suffix")
    if match.group(2) != "0":
        raise AdmissionError(
            "NVBit admitted kernel variant is ambiguous for the NCU launch join"
        )
    return match.group(1)


def _nvbit_capture_stderr_matches(stderr: bytes) -> bool:
    return stderr == b"" or _NVBIT_CAPTURE_STDERR_PATTERN.fullmatch(stderr) is not None


def _verify_nvbit_artifacts(
    generation: _ReceiptGeneration,
    root: Path,
    receipt: dict[str, Any],
    row: dict[str, Any],
    artifact_by_role: dict[str, list[dict[str, Any]]],
    *,
    receipt_schema_version: int,
) -> dict[str, Any]:
    """Re-derive the closed NVBit trace-summary contract from its JSON payload."""
    trace_roles = {
        "dynamic_trace",
        "stats",
        "threadblock",
        "trace_summary",
        "trace_auxiliary",
    }
    trace_size_bytes = sum(
        artifact["size_bytes"]
        for role in trace_roles
        for artifact in artifact_by_role.get(role, [])
    )
    if trace_size_bytes > row["trace_budget"]["max_trace_bytes"]:
        raise AdmissionError("NVBit trace artifacts exceed immutable trace byte budget")
    summary_path = _safe_receipt_path(
        root, artifact_by_role["trace_summary"][0]["path"], "NVBit trace summary"
    )
    summary = _load_json(summary_path, "NVBit trace summary", generation)
    dynamic = artifact_by_role["dynamic_trace"][0]
    stats = artifact_by_role["stats"][0]
    expected_summary_fields = {
        "schema_version",
        "producer",
        "artifact_kind",
        "trace_contract_id",
        "memory_payload_classes",
        "binary_version",
        "nvbit_version",
        "trace_contract_status",
        "expected_record_count",
        "admitted_record_count",
        "threadblock_count",
        "dynamic_trace",
        "stats",
        "capture_stdout",
        "capture_stderr",
        "capture_stderr_protocol",
    }
    if set(summary) != expected_summary_fields:
        raise AdmissionError("NVBit trace summary fields differ from closed contract")

    dynamic_path = _safe_receipt_path(
        root, dynamic["path"], "NVBit dynamic trace"
    )
    expected_trace_contract_id = TRACE_CONTRACT_IDS_BY_RECEIPT_VERSION[
        receipt_schema_version
    ]
    expected_trace_root = root / "nvbit" / row["successor_slug"]
    if dynamic_path != expected_trace_root / "dynamic_trace.pb":
        raise AdmissionError("NVBit dynamic trace path differs from canonical trace root")
    for role in trace_roles:
        for artifact in artifact_by_role.get(role, []):
            artifact_path = _safe_receipt_path(
                root, artifact["path"], f"NVBit {role} artifact"
            )
            try:
                relative = artifact_path.relative_to(expected_trace_root)
            except ValueError as exc:
                raise AdmissionError(
                    f"NVBit {role} artifact is outside canonical trace root"
                ) from exc
            if role == "stats" and relative != Path("stats.csv"):
                raise AdmissionError("NVBit stats artifact path differs from producer path")
            if role == "trace_summary" and relative != Path("trace_summary.json"):
                raise AdmissionError(
                    "NVBit trace summary artifact path differs from producer path"
                )
            if role == "threadblock" and (
                len(relative.parts) < 2
                or relative.parts[0] != "threadblocks"
                or relative.suffix != ".pb"
            ):
                raise AdmissionError(
                    "NVBit threadblock artifact path differs from producer tree"
                )
    try:
        dynamic_payload = generation.read_bytes(dynamic_path)
        if sha256_bytes(dynamic_payload) != dynamic["sha256"]:
            raise AdmissionError("NVBit dynamic trace changed before metadata validation")
        preflight_trace_protobuf(dynamic_payload)
        from tools.nvbit_tracer.proto_binding_loader import (
            load_dynamic_trace_bindings,
        )

        trace_pb2 = load_dynamic_trace_bindings().trace_pb2
        trace = trace_pb2.Trace()
        trace.ParseFromString(dynamic_payload)
    except AdmissionError:
        raise
    except (
        ProtobufDecodedResourceLimitExceeded,
        ProtobufWireMalformed,
    ) as exc:
        raise AdmissionError(
            f"NVBit trace producer metadata is unreadable: {exc}"
        ) from exc
    except MemoryError:
        raise
    except Exception as exc:
        raise AdmissionError(
            f"NVBit trace producer metadata is unreadable: {exc}"
        ) from exc
    if (
        type(trace.binary_version) is not int
        or trace.binary_version != NVBIT_TRACE_BINARY_VERSION
        or type(trace.nvbit_version) is not str
        or trace.nvbit_version != NVBIT_VERSION
        or trace.trace_contract_id != expected_trace_contract_id
        or list(trace.memory_payload_classes) != TRACE_MEMORY_PAYLOAD_CLASSES
    ):
        raise AdmissionError("NVBit dynamic trace producer version metadata differs")

    expected_summary = {
        "schema_version": 1,
        "producer": NVBIT_TRACE_SUMMARY_PRODUCER,
        "artifact_kind": "cutlass_successor_nvbit_trace_summary",
        "trace_contract_id": expected_trace_contract_id,
        "memory_payload_classes": TRACE_MEMORY_PAYLOAD_CLASSES,
        "binary_version": NVBIT_TRACE_BINARY_VERSION,
        "nvbit_version": NVBIT_VERSION,
        "trace_contract_status": "pass",
        "dynamic_trace": Path(dynamic["path"]).name,
        "stats": Path(stats["path"]).name,
        "capture_stderr_protocol": NVBIT_CAPTURE_STDERR_PROTOCOL,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise AdmissionError("NVBit trace summary semantics do not match payload inventory")
    auxiliary_by_relative: dict[str, dict[str, Any]] = {}
    for artifact in artifact_by_role.get("trace_auxiliary", []):
        artifact_path = _safe_receipt_path(
            root, artifact["path"], "NVBit trace auxiliary artifact"
        )
        relative = artifact_path.relative_to(expected_trace_root).as_posix()
        auxiliary_by_relative[relative] = artifact
    for field, name in (
        ("capture_stdout", NVBIT_CAPTURE_STDOUT_NAME),
        ("capture_stderr", NVBIT_CAPTURE_STDERR_NAME),
    ):
        declared = summary[field]
        artifact = auxiliary_by_relative.get(name)
        if declared is None:
            if artifact is not None:
                raise AdmissionError(
                    f"NVBit {field} artifact is present but undeclared"
                )
        elif declared != name or artifact is None:
            raise AdmissionError(
                f"NVBit {field} summary binding differs from artifact inventory"
            )
    if summary["capture_stderr"] is not None:
        stderr_artifact = auxiliary_by_relative[NVBIT_CAPTURE_STDERR_NAME]
        stderr_path = _safe_receipt_path(
            root, stderr_artifact["path"], "NVBit capture stderr"
        )
        if not _nvbit_capture_stderr_matches(
            generation.read_bytes(stderr_path)
        ):
            raise AdmissionError(
                "NVBit capture stderr differs from closed protocol"
            )
    if type(summary["binary_version"]) is not int:
        raise AdmissionError("NVBit trace summary producer version metadata differs")
    for key in ("expected_record_count", "admitted_record_count", "threadblock_count"):
        if type(summary.get(key)) is not int or summary[key] <= 0:
            raise AdmissionError(f"NVBit trace summary has invalid {key}")
    if summary["admitted_record_count"] > summary["expected_record_count"]:
        raise AdmissionError("NVBit trace summary admits more records than expected")
    try:
        from tools.nvbit_tracer.import_trace_contract import validate_trace_dir
        from tools.nvbit_tracer.trace_admission import (
            require_historical_trace_admission,
            require_trace_admission,
        )

        trace_root = dynamic_path.parent
        admission = (
            require_historical_trace_admission
            if receipt_schema_version == 2
            else require_trace_admission
        )
        admission_result = admission(trace_root)
        contract = validate_trace_dir(trace_root)
        if not contract.ok:
            raise AdmissionError(f"NVBit trace contract failed: {contract.typed_reason}")
    except AdmissionError:
        raise
    except MemoryError:
        raise
    except Exception as exc:
        raise AdmissionError(f"NVBit trace semantic validation failed: {exc}") from exc
    if admission_result.dynamic_trace_sha256 != dynamic["sha256"]:
        raise AdmissionError("NVBit trace admission used different dynamic trace bytes")
    if (
        summary["expected_record_count"] != admission_result.expected_record_count
        or summary["admitted_record_count"] != admission_result.admitted_record_count
        or summary["threadblock_count"] != admission_result.threadblock_count
    ):
        raise AdmissionError("NVBit trace summary counts do not match independent admission")
    if len(admission_result.launches) != 1:
        raise AdmissionError("NVBit trace must contain exactly one admitted launch")
    launch = admission_result.launches[0]
    if launch.launch_index != 0 or launch.device_id != 0 or not launch.kernel_name:
        raise AdmissionError("NVBit admitted launch identity is invalid")
    observed = {
        "kernel_name": _nvbit_kernel_base(launch.kernel_name),
        "grid_dim": [launch.grid_x, launch.grid_y, launch.grid_z],
        "block_dim": [launch.block_x, launch.block_y, launch.block_z],
    }
    if observed["kernel_name"] not in _sass_kernel_names(row):
        raise AdmissionError("NVBit admitted kernel is absent from pinned SASS authority")
    return observed


def _expected_preflight(
    row: dict[str, Any], roster_row: dict[str, Any]
) -> dict[str, Any]:
    static = row["static_launch_inventory"]
    return {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_preflight",
        "matrix_key": row["matrix_key"],
        "variant_id": row["variant_id"],
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "build_arch": "sm_86",
        "source_commit": PINNED_BUNDLE_COMMIT,
        "binary_path": static["binary_path"],
        "binary_sha256": static["binary_sha256"],
        "witness_path": roster_row["witness"]["path"],
        "witness_sha256": roster_row["witness"]["sha256"],
        "shape_path": row["shape"]["path"],
        "shape_sha256": row["shape"]["sha256"],
        "no_competing_processes": True,
        "competing_processes": [],
        "gpu_name": "NVIDIA GeForce RTX 3090 Ti",
        "persistence_mode": "Enabled",
        "sm_clock_mhz": 1350,
        "mem_clock_mhz": 10501,
        "cuda_release": {"major": 13, "minor": 2},
        "nvbit_version": "1.8",
        "ncu_version": "2026.1.0",
    }


def _expected_ncu_command(
    output_root: Path,
    row: dict[str, Any],
    ncu_identity: ncu_executor.ExpectedNcuIdentity,
) -> list[str]:
    binary = ROOT / row["static_launch_inventory"]["binary_path"]
    arguments = [str(value) for value in row["shape"]["actual_args"]]
    return [
        str(ROOT / LABEL_DRIVER_REL),
        "--device", "sm_86",
        "--slug", row["successor_slug"],
        "--shape-point", "trace_00",
        "--repeat", "0",
        "--launch", "0",
        "--binary", str(binary),
        "--warmup", "0",
        "--ncu", ncu_identity.resolved_path,
        "--ncu-executable-sha256", ncu_identity.executable_sha256,
        "--ncu-version-line", ncu_identity.version_line,
        "--out-dir", str(output_root),
        "--expected-cuda-release", "13.2",
        "--expected-ncu-version-prefix", "2026.1.0",
        "--publication-metric-scope", "cycles_only_binding",
        "--target-processes", "all",
        "--filter-mode", "global",
        "--replay-mode", "kernel",
        "--profile-from-start", "on",
        "--launch-skip", "0",
        "--launch-count", "1",
        "--clock-control", "none",
        "--cache-control", "all",
        "--unit-metric", "gpc__cycles_elapsed.max",
        "--unit-metric", "gpu__time_duration.sum",
        "--",
        *arguments,
    ]


def _expected_nvbit_command(row: dict[str, Any]) -> list[str]:
    binary = ROOT / row["static_launch_inventory"]["binary_path"]
    arguments = [str(value) for value in row["shape"]["actual_args"]]
    return [str(binary), *arguments]


def _expected_nvbit_environment(
    output_root: Path,
    row: dict[str, Any],
    *,
    receipt_schema_version: int = ACTIVE_RECEIPT_SCHEMA_VERSION,
) -> dict[str, str]:
    binary = ROOT / row["static_launch_inventory"]["binary_path"]
    if receipt_schema_version == 2:
        injection_path = ROOT / NVBIT_TRACER_REL
    elif receipt_schema_version == 3:
        injection_path = (
            output_root
            / "capture_tools"
            / f"{row['variant_id']}.nvbit_tracer_tool.so"
        )
    else:
        raise AdmissionError(
            "unsupported receipt version for NVBit environment authority: "
            f"{receipt_schema_version!r}"
        )
    return {
        "CUDA_INJECTION64_PATH": str(injection_path),
        "CUDA_VISIBLE_DEVICES": "0",
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": str(ROOT / NVBIT_PB_LIB_REL),
        "PATH": NVBIT_COLLECTION_PATH,
        "PWD": str(binary.parent.resolve(strict=True)),
        "TRACES_FOLDER": str(output_root / "nvbit" / row["successor_slug"]),
        "USER_DEFINED_FOLDERS": "1",
    }


def _validate_ncu_authority(
    authority: Any,
) -> ncu_executor.ExpectedNcuIdentity:
    required = {
        "status",
        "expected",
        "execution_mode",
        "host_machine",
        "launcher_resolved_path",
        "launcher_sha256",
        "launcher_target_relpath",
        "resolved_path",
        "executable_sha256",
        "version_line",
    }
    if type(authority) is not dict or set(authority) != required:
        raise AdmissionError("successor NCU authority fields differ")
    if (
        authority["status"] != campaign_preflight.STATUS_OK
        or authority["expected"] != "2026.1.0"
        or authority["execution_mode"] != campaign_preflight.NCU_EXECUTION_MODE
        or authority["host_machine"] != campaign_preflight.NCU_HOST_MACHINE
        or authority["launcher_target_relpath"]
        != campaign_preflight.NCU_LAUNCHER_TARGET_RELPATH
    ):
        raise AdmissionError("successor NCU authority pins differ")
    launcher = authority["launcher_resolved_path"]
    resolved = authority["resolved_path"]
    if (
        type(launcher) is not str
        or type(resolved) is not str
        or not Path(launcher).is_absolute()
        or not Path(resolved).is_absolute()
        or os.path.normpath(launcher) != launcher
        or os.path.normpath(resolved) != resolved
        or Path(resolved) != Path(launcher).parent / authority["launcher_target_relpath"]
    ):
        raise AdmissionError("successor NCU authority paths differ")
    if not all(
        type(authority[name]) is str
        and re.fullmatch(r"[0-9a-f]{64}", authority[name]) is not None
        for name in ("launcher_sha256", "executable_sha256")
    ):
        raise AdmissionError("successor NCU authority hashes are invalid")
    if authority["version_line"] != NCU_VERSION_LINE:
        raise AdmissionError("successor NCU authority version line differs")
    if (
        launcher != NCU_LAUNCHER_PATH
        or authority["launcher_sha256"] != NCU_LAUNCHER_SHA256
        or resolved != NCU_EXECUTABLE_PATH
        or authority["executable_sha256"] != NCU_EXECUTABLE_SHA256
    ):
        raise AdmissionError("successor NCU executable pins differ")
    try:
        return ncu_executor.build_expected_ncu_identity(
            resolved_path=resolved,
            executable_sha256=authority["executable_sha256"],
            version_line=authority["version_line"],
        )
    except ncu_executor.NcuActionExecutorError as exc:
        raise AdmissionError(
            f"successor NCU executable authority is invalid: {exc}"
        ) from exc


def _verify_capture_documents(
    generation: _ReceiptGeneration,
    root: Path,
    receipt: dict[str, Any],
    row: dict[str, Any],
    roster_row: dict[str, Any],
    artifact_by_role: dict[str, list[dict[str, Any]]],
    *,
    receipt_schema_version: int,
) -> ncu_executor.ExpectedNcuIdentity | None:
    preflight = _load_json(
        _safe_receipt_path(root, artifact_by_role["preflight"][0]["path"], "preflight"),
        "preflight",
        generation,
    )
    expected = _expected_preflight(row, roster_row)
    if preflight != expected or canonical_sha(preflight) != canonical_sha(expected):
        raise AdmissionError("successor preflight content does not match immutable pins")
    for key in ("binary_path", "witness_path", "shape_path"):
        path = ROOT / preflight[key]
        _regular(path, f"successor preflight {key}")
        if sha256_bytes(path.read_bytes()) != preflight[key.replace("_path", "_sha256")]:
            raise AdmissionError(f"successor preflight {key} hash mismatch")

    environment = _load_json(
        _safe_receipt_path(root, artifact_by_role["environment"][0]["path"], "environment"),
        "environment",
        generation,
    )
    if set(environment) != {
        "schema_version", "artifact_kind", "source_commit",
        "preflight_pins_sha256", "live_checks", "live_outputs", "environment",
    }:
        raise AdmissionError("successor environment document keys are not closed")
    if (
        environment["schema_version"] != 1
        or environment["artifact_kind"] != "cutlass_successor_capture_environment"
        or environment["source_commit"] != PINNED_BUNDLE_COMMIT
        or environment["preflight_pins_sha256"] != canonical_sha(preflight)
    ):
        raise AdmissionError("successor environment pin binding is invalid")
    selected = environment["environment"]
    if not isinstance(selected, dict) or any(
        type(name) is not str or type(value) is not str
        for name, value in selected.items()
    ):
        raise AdmissionError("successor environment variables are not closed strings")
    if receipt["capture_kind"] == "nvbit":
        expected_nvbit_environment = _expected_nvbit_environment(
            root,
            row,
            receipt_schema_version=receipt_schema_version,
        )
        if selected != expected_nvbit_environment:
            raise AdmissionError("NVBit collection environment pins differ")
        trace_root = _safe_receipt_path(
            root, artifact_by_role["dynamic_trace"][0]["path"], "NVBit dynamic trace"
        ).parent
        if Path(selected["TRACES_FOLDER"]) != trace_root:
            raise AdmissionError("NVBit trace output path differs from artifacts")
    else:
        allowed_ncu_environment_names = set(
            NCU_COLLECTION_REQUIRED_ENVIRONMENT_NAMES
        ) | set(NCU_COLLECTION_OPTIONAL_ENVIRONMENT_NAMES)
        if (
            not set(NCU_COLLECTION_REQUIRED_ENVIRONMENT_NAMES).issubset(selected)
            or not set(selected).issubset(allowed_ncu_environment_names)
        ):
            raise AdmissionError(
                "successor NCU environment names differ from producer contract"
            )
        if any(
            selected.get(name)
            for name in (
                "NVBIT_PB_PREFIX", "CUDA_INJECTION64_PATH", "TRACES_FOLDER",
                "USER_DEFINED_FOLDERS",
            )
        ):
            raise AdmissionError("NCU environment contains NVBit injection controls")
        try:
            ncu_environment = ncu_executor.NcuCollectionEnvironment(dict(selected))
        except ncu_executor.NcuActionExecutorError as exc:
            raise AdmissionError(
                f"successor NCU collection environment is invalid: {exc}"
            ) from exc
        expected_pwd = str(
            (ROOT / row["static_launch_inventory"]["binary_path"])
            .parent.resolve(strict=True)
        )
        if (
            ncu_environment.as_dict() != selected
            or selected["PWD"] != expected_pwd
            or selected.get("LD_SOURCE_COMMIT_OVERRIDE")
            != PINNED_BUNDLE_COMMIT
        ):
            raise AdmissionError("successor NCU collection environment pins differ")

    live = environment["live_checks"]
    base_live_keys = {
        "smi_command", "smi_stdout_sha256", "competing_process_command",
        "competing_process_stdout_sha256", "identity_command",
        "identity_stdout_sha256", "gpu_uuid_sha256", "driver_version",
        "nvcc_command", "nvcc_stdout_sha256", "cuda_release",
        "nvbit_version", "gpu_name", "persistence_mode", "sm_clock_mhz",
        "mem_clock_mhz",
    }
    ncu_live_keys = {
        "ncu_command", "ncu_stdout_sha256", "ncu_authority", "ncu_version",
    }
    live_keys = base_live_keys | (
        ncu_live_keys if receipt["capture_kind"] == "ncu" else set()
    )
    if not isinstance(live, dict) or set(live) != live_keys:
        raise AdmissionError("successor live-check document keys are not closed")
    if (
        live["smi_command"] != ["nvidia-smi", "--query-gpu=name,compute_cap,persistence_mode,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"]
        or live["competing_process_command"] != ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"]
        or live["identity_command"] != ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"]
        or live["driver_version"] != "610.43.03"
        or live["nvcc_command"] != ["/usr/local/cuda-13.2/bin/nvcc", "--version"]
        or live["cuda_release"] != {"major": 13, "minor": 2}
        or live["nvbit_version"] != "1.8"
        or live["gpu_name"] != "NVIDIA GeForce RTX 3090 Ti"
        or live["persistence_mode"] != "Enabled"
        or live["sm_clock_mhz"] != 1350
        or live["mem_clock_mhz"] != 10501
    ):
        raise AdmissionError("successor live-check pins are invalid")
    outputs = environment["live_outputs"]
    output_keys = {
        "smi_stdout", "competing_process_stdout", "identity_stdout",
        "nvcc_stdout",
    }
    if receipt["capture_kind"] == "ncu":
        output_keys.add("ncu_stdout")
    if not isinstance(outputs, dict) or set(outputs) != output_keys or any(
        not isinstance(value, str) for value in outputs.values()
    ):
        raise AdmissionError("successor live outputs are not closed strings")
    for output_key in output_keys:
        hash_key = f"{output_key}_sha256"
        if sha256_bytes(outputs[output_key].encode()) != live[hash_key]:
            raise AdmissionError(f"successor live output hash mismatch: {output_key}")
    smi_lines = [line.strip() for line in outputs["smi_stdout"].splitlines() if line.strip()]
    if smi_lines != ["NVIDIA GeForce RTX 3090 Ti, 8.6, Enabled, 1350, 10501"]:
        raise AdmissionError("successor nvidia-smi output is not pinned")
    identity_lines = [
        line.strip()
        for line in outputs["identity_stdout"].splitlines()
        if line.strip()
    ]
    if len(identity_lines) != 1:
        raise AdmissionError("successor selected-GPU identity output is not singular")
    identity_fields = [part.strip() for part in identity_lines[0].split(",")]
    if (
        len(identity_fields) != 2
        or re.fullmatch(GPU_UUID_PATTERN, identity_fields[0]) is None
        or identity_fields[1] != live["driver_version"]
        or sha256_bytes(identity_fields[0].encode("utf-8"))
        != live["gpu_uuid_sha256"]
    ):
        raise AdmissionError("successor selected-GPU identity output is not pinned")
    competing = outputs["competing_process_stdout"].strip().lower()
    if competing and "no running processes" not in competing:
        raise AdmissionError("successor competing-process output is not empty")
    if "release 13.2" not in outputs["nvcc_stdout"]:
        raise AdmissionError("successor nvcc output does not prove CUDA 13.2")
    ncu_identity: ncu_executor.ExpectedNcuIdentity | None = None
    if receipt["capture_kind"] == "ncu":
        ncu_identity = _validate_ncu_authority(live["ncu_authority"])
        if (
            live["ncu_command"] != [ncu_identity.resolved_path, "--version"]
            or live["ncu_version"] != "2026.1.0"
        ):
            raise AdmissionError("successor NCU live-check pins are invalid")
        expected_ncu_stdout = (
            "NVIDIA (R) Nsight Compute Command Line Profiler\n"
            "Copyright (c) 2018-2026 NVIDIA Corporation\n"
            f"{ncu_identity.version_line}\n"
        )
        if outputs["ncu_stdout"] != expected_ncu_stdout:
            raise AdmissionError("successor NCU output differs from exact authority")
        expected_command = _expected_ncu_command(root, row, ncu_identity)
        if receipt["command"] != expected_command:
            raise AdmissionError(
                "successor NCU command authority differs from the exact row"
            )
    elif receipt["command"] != _expected_nvbit_command(row):
        raise AdmissionError(
            "successor NVBit command authority differs from the exact row"
        )
    expected_host = canonical_sha({
        "gpu": live["gpu_name"],
        "smi": live["smi_stdout_sha256"],
        "gpu_uuid_sha256": live["gpu_uuid_sha256"],
        "driver_version": live["driver_version"],
        "identity_stdout_sha256": live["identity_stdout_sha256"],
    })
    if receipt["process"]["host_identity_sha256"] != expected_host:
        raise AdmissionError("successor process host identity does not match live output")
    for key in live_keys:
        if key.endswith("sha256") and (
            not isinstance(live[key], str) or len(live[key]) != 64
            or any(char not in "0123456789abcdef" for char in live[key])
        ):
            raise AdmissionError(f"successor live-check hash is invalid: {key}")
    return ncu_identity


def _verify_receipt(
    generation: _ReceiptGeneration,
    root: Path,
    path: Path,
    expected_rows: dict[tuple[str, str], dict[str, Any]],
    expected_roster_rows: dict[tuple[str, str], dict[str, Any]],
    used_artifacts: set[str],
) -> tuple[
    tuple[str, str],
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    rel = path.relative_to(root).as_posix()
    document = _load_json(path, f"receipt {rel}", generation)
    schema_authority = _validate_registered_receipt_schema(
        document, f"receipt {rel}"
    )
    schema_version = schema_authority["schema_version"]
    if schema_version != ACTIVE_RECEIPT_SCHEMA_VERSION:
        raise AdmissionError(
            "receipt schema_version "
            f"{schema_version} is historical-read-only and cannot enter "
            "active capture admission"
        )
    seal = document["receipt_seal_sha256"]
    if seal != canonical_sha({k: v for k, v in document.items() if k != "receipt_seal_sha256"}):
        raise AdmissionError(f"receipt seal mismatch: {rel}")
    receipt = document["receipt"]
    pair_key = (receipt["matrix_key"], receipt["variant_id"])
    if pair_key not in expected_rows:
        raise AdmissionError(f"receipt has unknown matrix_key+variant_id: {pair_key}")
    row = expected_rows[pair_key]
    if pair_key not in expected_roster_rows:
        raise AdmissionError(f"receipt has no immutable roster row: {pair_key}")
    roster_row = expected_roster_rows[pair_key]
    identity = receipt["identity"]
    expected_binary = row["static_launch_inventory"]["binary_sha256"]
    expected_kernel = hashlib.sha256(
        f"{row['successor_slug']}|{row['static_launch_inventory']['binary_path']}".encode("utf-8")
    ).hexdigest()
    expected_identity = {
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "binary_sha256": expected_binary,
        "source_commit": PINNED_BUNDLE_COMMIT,
        "kernel_signature_sha256": expected_kernel,
        "arch": row["arch"],
        "ncu_metric_version": DEFAULT_METRIC_VERSION,
    }
    if identity != expected_identity:
        raise AdmissionError(f"receipt identity does not match matrix: {rel}")
    _require_declared_only_launch_contract(row)
    _verify_launch(receipt["launch_identity"], identity, rel)
    _verify_process(receipt["process"], rel)
    if receipt["command_sha256"] != canonical_sha(receipt["command"]):
        raise AdmissionError(f"{rel} command digest does not match receipt command")
    if receipt["process"]["command_sha256"] != receipt["command_sha256"]:
        raise AdmissionError(f"{rel} process command does not match receipt command")
    _check_dt(receipt["captured_at_utc"], f"{rel} captured_at_utc")
    containment_sha256 = sha256_bytes(Path(delegated_cgroup.__file__).read_bytes())
    if containment_sha256 != CGROUP_CONTAINMENT_SUPERVISOR_SHA256:
        raise AdmissionError(
            "successor cgroup containment supervisor differs from pinned authority"
        )
    expected_provenance = {
        "capture_producer_sha256": _capture_producer_sha256(schema_version),
        "containment_supervisor_sha256": containment_sha256,
    }
    nvbit_tool: dict[str, Any] | None = None
    if receipt["capture_kind"] == "ncu":
        expected_tool = ROOT / "tools/local_labels/label_driver.sh"
        _regular(expected_tool, "successor capture phase tool")
        phase_tool_sha256 = sha256_bytes(expected_tool.read_bytes())
        if phase_tool_sha256 != LABEL_DRIVER_SHA256:
            raise AdmissionError("successor label driver differs from pinned authority")
        expected_provenance["phase_tool_sha256"] = phase_tool_sha256
        expected_provenance["action_executor_sha256"] = (
            _ncu_action_executor_sha256(schema_version)
        )
    elif schema_version == 2:
        # Historical v2 receipts bind an explicit immutable tracer identity.
        # Admission must not reconstruct that authority from the current ignored
        # repository tracer, which may be absent or belong to a later build.
        nvbit_tool = _historical_nvbit_tracer_authority(schema_version)
        expected_provenance["phase_tool_sha256"] = nvbit_tool["sha256"]
    expected_semantics = (
        {
            "semantic_kind": "ncu_metric_validation",
            "primary_metric": "gpc__cycles_elapsed.max",
            "secondary_metric": "gpu__time_duration.sum",
            "primary_metric_present": True,
            "secondary_metric_present": True,
            "target_only": True,
            "simulator_config_present": False,
        }
        if receipt["capture_kind"] == "ncu"
        else {
            "semantic_kind": "nvbit_trace_validation",
            "trace_contract_status": "pass",
            "dynamic_trace_role": "dynamic_trace",
            "summary_role": "trace_summary",
            "target_only": True,
            "simulator_config_present": False,
        }
    )
    if receipt["capture_kind"] == "nvbit" and schema_version == 3:
        expected_semantics["descriptor_bindings"] = {
            "command_0": "capture_target",
            "CUDA_INJECTION64_PATH": "phase_tool",
            "TRACES_FOLDER": "trace_root",
        }
    if receipt["semantic_validation"] != expected_semantics:
        raise AdmissionError(f"{rel} semantic validation is not exact")
    if receipt["capture_kind"] == "ncu":
        expected_roles = {
            "ncu_report",
            "label_sidecar",
            "label_run_manifest",
            "ncu_launch_observation",
            "preflight",
            "environment",
        }
    else:
        expected_roles = {
            "dynamic_trace",
            "stats",
            "threadblock",
            "trace_summary",
            "trace_auxiliary",
            "preflight",
            "environment",
        }
        if schema_version == 3:
            expected_roles.update({
                "phase_tool",
                "runtime_manifest",
                "external_build_source_manifest",
                "external_build_receipt",
                "external_build_artifact_identity",
            })
    mandatory_roles = expected_roles - {"threadblock", "trace_auxiliary"}
    artifact_by_role: dict[str, list[dict[str, Any]]] = {}
    seen_roles: set[str] = set()

    for artifact in receipt["artifacts"]:
        role = artifact["role"]
        if role not in expected_roles:
            raise AdmissionError(f"unexpected artifact role in receipt: {rel}:{role}")
        artifact_path = _safe_receipt_path(root, artifact["path"], "artifact path")
        artifact_rel = artifact_path.relative_to(root).as_posix()
        if receipt["capture_kind"] == "nvbit":
            canonical_stream_paths = {
                f"nvbit/{row['successor_slug']}/{NVBIT_CAPTURE_STDOUT_NAME}",
                f"nvbit/{row['successor_slug']}/{NVBIT_CAPTURE_STDERR_NAME}",
            }
            if (
                artifact_rel in canonical_stream_paths
                and role != "trace_auxiliary"
            ):
                raise AdmissionError(
                    "NVBit canonical capture stream has the wrong artifact role"
                )
        if role == "phase_tool":
            expected_phase_tool = (
                root
                / "capture_tools"
                / f"{row['variant_id']}.nvbit_tracer_tool.so"
            )
            if (
                schema_version != 3
                or receipt["capture_kind"] != "nvbit"
                or artifact_path != expected_phase_tool
            ):
                raise AdmissionError(
                    f"NVBit phase-tool artifact path differs: {artifact['path']}"
                )
        if role == "runtime_manifest":
            expected_runtime_manifest = (
                root
                / "capture_tools"
                / f"{row['variant_id']}.nvbit_runtime_manifest.json"
            )
            if (
                schema_version != 3
                or receipt["capture_kind"] != "nvbit"
                or artifact_path != expected_runtime_manifest
            ):
                raise AdmissionError(
                    "NVBit runtime-manifest artifact path differs: "
                    f"{artifact['path']}"
                )
        if role in {
            "external_build_source_manifest",
            "external_build_receipt",
            "external_build_artifact_identity",
        }:
            filename_by_role = {
                "external_build_source_manifest": (
                    "nvbit_external_build_source_manifest.json"
                ),
                "external_build_receipt": "nvbit_external_build_receipt.json",
                "external_build_artifact_identity": (
                    "nvbit_external_build_artifact_identity.json"
                ),
            }
            expected_build_document = (
                root
                / "capture_tools"
                / f"{row['variant_id']}.{filename_by_role[role]}"
            )
            if (
                schema_version != 3
                or receipt["capture_kind"] != "nvbit"
                or artifact_path != expected_build_document
            ):
                raise AdmissionError(
                    f"NVBit external-build artifact path differs: {artifact['path']}"
                )
        if role in seen_roles and role not in {"threadblock", "trace_auxiliary"}:
            raise AdmissionError(f"duplicate artifact role in receipt: {rel}:{role}")
        seen_roles.add(role)
        if artifact_rel.endswith(RECEIPT_SUFFIX):
            raise AdmissionError(f"receipt listed as its own artifact: {artifact_rel}")
        if artifact_rel in used_artifacts:
            raise AdmissionError(f"duplicate artifact path: {artifact_rel}")
        used_artifacts.add(artifact_rel)
        if receipt["capture_kind"] == "nvbit" and role in {"dynamic_trace", "stats", "threadblock", "trace_summary", "trace_auxiliary"} and not artifact["path"].startswith("nvbit/"):
            raise AdmissionError(f"NVBit payload artifact is outside nvbit/: {artifact['path']}")
        if role == "dynamic_trace" and Path(artifact["path"]).name != "dynamic_trace.pb":
            raise AdmissionError(f"dynamic trace artifact has wrong producer path: {artifact['path']}")
        if role == "stats" and Path(artifact["path"]).name != "stats.csv":
            raise AdmissionError(f"stats artifact has wrong producer path: {artifact['path']}")
        if role == "trace_summary" and Path(artifact["path"]).name != "trace_summary.json":
            raise AdmissionError(f"trace summary artifact has wrong producer path: {artifact['path']}")
        if role == "threadblock" and (
            not artifact["path"].endswith(".pb")
            or "/threadblocks/" not in f"/{artifact['path']}"
        ):
            raise AdmissionError(f"threadblock artifact is outside threadblocks/: {artifact['path']}")
        artifact_by_role.setdefault(role, []).append(artifact)
        payload = generation.read_bytes(artifact_path)
        if not payload:
            raise AdmissionError(f"zero-size artifact: {artifact_rel}")
        if sha256_bytes(payload) != artifact["sha256"] or len(payload) != artifact["size_bytes"]:
            raise AdmissionError(f"artifact hash or size drifted: {artifact_rel}")
    if not mandatory_roles.issubset(seen_roles) or (
        receipt["capture_kind"] == "nvbit"
        and not artifact_by_role.get("threadblock")
    ):
        raise AdmissionError(f"receipt artifact set incomplete: {rel}")
    if receipt["capture_kind"] == "nvbit" and schema_version == 3:
        phase_tool_artifact = artifact_by_role["phase_tool"][0]
        expected_provenance["phase_tool_sha256"] = phase_tool_artifact[
            "sha256"
        ]
        runtime_manifest_artifact = artifact_by_role["runtime_manifest"][0]
        expected_provenance["runtime_manifest_sha256"] = (
            runtime_manifest_artifact["sha256"]
        )
        source_manifest_artifact = artifact_by_role[
            "external_build_source_manifest"
        ][0]
        build_receipt_artifact = artifact_by_role["external_build_receipt"][0]
        artifact_identity_artifact = artifact_by_role[
            "external_build_artifact_identity"
        ][0]
        source_payload = generation.read_bytes(
            _safe_receipt_path(
                root,
                source_manifest_artifact["path"],
                "external build source manifest",
            )
        )
        build_payload = generation.read_bytes(
            _safe_receipt_path(
                root,
                build_receipt_artifact["path"],
                "external build receipt",
            )
        )
        artifact_payload = generation.read_bytes(
            _safe_receipt_path(
                root,
                artifact_identity_artifact["path"],
                "external build artifact identity",
            )
        )
        try:
            external_identity = external_build.verify_admitted_bundle_documents(
                source_payload,
                build_payload,
                artifact_payload,
                repo_root=ROOT,
                expected_tracer_sha256=phase_tool_artifact["sha256"],
                expected_tracer_size_bytes=phase_tool_artifact["size_bytes"],
            )
        except external_build.ExternalTracerBuildAuthorityError as exc:
            raise AdmissionError(
                f"NVBit external build authority is invalid: {exc}"
            ) from exc
        source_identity = external_identity["source_identity"]
        expected_build_authority = {
            "authority_kind": external_build.REVIEWED_AUTHORITY_KIND,
            "expected_tracer_sha256": phase_tool_artifact["sha256"],
            "source_commit_sha1": source_identity["head_commit_sha1"],
            "source_tree_sha1": source_identity["head_tree_sha1"],
            "source_inventory_sha256": source_identity["inventory_sha256"],
            "source_manifest_sha256": source_manifest_artifact["sha256"],
            "build_receipt_sha256": build_receipt_artifact["sha256"],
            "artifact_identity_sha256": artifact_identity_artifact["sha256"],
        }
        expected_provenance["external_build_authority"] = (
            expected_build_authority
        )
        nvbit_tool = {
            "identity_kind": "retained_reviewed_external_build_bundle.v1",
            "path": phase_tool_artifact["path"],
            "sha256": phase_tool_artifact["sha256"],
            "size_bytes": phase_tool_artifact["size_bytes"],
            "runtime_manifest": dict(runtime_manifest_artifact),
            "external_build_authority": expected_build_authority,
            "source_manifest": dict(source_manifest_artifact),
            "build_receipt": dict(build_receipt_artifact),
            "artifact_identity": dict(artifact_identity_artifact),
        }
    if receipt["provenance"] != expected_provenance:
        raise AdmissionError(
            f"{rel} capture tool provenance does not match current inputs"
        )
    preflight = artifact_by_role["preflight"][0]
    environment = artifact_by_role["environment"][0]
    if receipt["preflight_path"] != preflight["path"] or receipt["preflight_sha256"] != preflight["sha256"]:
        raise AdmissionError(f"{rel} preflight reference does not match typed artifact")
    if receipt["environment_sha256"] != environment["sha256"]:
        raise AdmissionError(f"{rel} environment hash does not match typed artifact")
    expected_ncu_identity = _verify_capture_documents(
        generation,
        root,
        receipt,
        row,
        roster_row,
        artifact_by_role,
        receipt_schema_version=schema_version,
    )
    if receipt["capture_kind"] == "ncu":
        if expected_ncu_identity is None:
            raise AdmissionError("NCU receipt lacks executable authority")
        observed_launch = _verify_ncu_artifacts(
            generation,
            root,
            expected_ncu_identity,
            receipt,
            row,
            artifact_by_role,
        )
    else:
        observed_launch = _verify_nvbit_artifacts(
            generation,
            root,
            receipt,
            row,
            artifact_by_role,
            receipt_schema_version=schema_version,
        )
    declared_observation = {
        "kernel_name": receipt["launch_identity"]["kernel_name"],
        "grid_dim": receipt["launch_identity"]["grid_dim"],
        "block_dim": receipt["launch_identity"]["block_dim"],
    }
    if declared_observation != observed_launch:
        raise AdmissionError(
            f"receipt launch identity differs from independently observed launch: {rel}"
        )
    return (
        pair_key,
        receipt["capture_kind"],
        receipt["identity"],
        receipt["launch_identity"],
        observed_launch,
        receipt["process"],
        _receipt_ref(
            generation,
            root,
            rel,
            seal,
            schema_authority=schema_authority,
            nvbit_tool=nvbit_tool,
        ),
    )


def _row_key(pair_key: tuple[str, str]) -> dict[str, str]:
    return {"matrix_key": pair_key[0], "variant_id": pair_key[1]}


def _process_separation(ncu: dict[str, Any], nvbit: dict[str, Any]) -> dict[str, Any]:
    if ncu["host_identity_sha256"] != nvbit["host_identity_sha256"] or ncu["boot_id_sha256"] != nvbit["boot_id_sha256"]:
        raise AdmissionError("NCU and NVBit receipts do not prove the same host boot")
    same_tuple = (ncu["pid"], ncu["started_at_utc"]) == (nvbit["pid"], nvbit["started_at_utc"])
    if ncu["process_identity_sha256"] == nvbit["process_identity_sha256"] or same_tuple:
        raise AdmissionError("NCU and NVBit receipts do not prove separate processes")
    if ncu["command_sha256"] == nvbit["command_sha256"]:
        raise AdmissionError("NCU and NVBit receipts have the same process command")
    return {
        "host_identity_sha256": ncu["host_identity_sha256"],
        "boot_id_sha256": ncu["boot_id_sha256"],
        "ncu_process_identity_sha256": ncu["process_identity_sha256"],
        "nvbit_process_identity_sha256": nvbit["process_identity_sha256"],
        "separate": True,
    }


def _bundle_overlay_meta(matrix: dict[str, Any]) -> dict[str, Any]:
    base = {
        "source_commit": PINNED_BUNDLE_COMMIT,
        "validation": {"path": cohort.VALIDATION_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.VALIDATION_REL.as_posix()]},
        "trace_budgets": {"path": cohort.BUDGETS_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.BUDGETS_REL.as_posix()]},
        "roster": {"path": cohort.ROSTER_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.ROSTER_REL.as_posix()]},
        "capture_matrix": {"path": cohort.MATRIX_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.MATRIX_REL.as_posix()]},
    }
    return {**base, "bundle_identity_sha256": canonical_sha(base)}


def _build_overlay_from_generation(
    generation: _ReceiptGeneration,
    root: Path,
    *,
    generated_at_utc: str,
    matrix: dict[str, Any],
    expected: dict[tuple[str, str], dict[str, Any]],
    expected_roster: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    files = generation.paths
    receipt_files = [path for path in files if path.name.endswith(RECEIPT_SUFFIX)]
    extras = [path for path in files if not path.name.endswith(RECEIPT_SUFFIX)]
    used_artifacts: set[str] = set()
    receipts: dict[
        tuple[str, str],
        dict[
            str,
            tuple[
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ],
        ],
    ] = {}
    for path in receipt_files:
        pair_key, kind, identity, launch, observed, process, ref = _verify_receipt(
            generation, root, path, expected, expected_roster, used_artifacts
        )
        by_kind = receipts.setdefault(pair_key, {})
        if kind in by_kind:
            raise AdmissionError(f"duplicate {kind} receipt for {pair_key}")
        by_kind[kind] = (identity, launch, observed, process, ref)
    referenced = set(used_artifacts) | {path.relative_to(root).as_posix() for path in receipt_files}
    extra_rel = [path.relative_to(root).as_posix() for path in extras if path.relative_to(root).as_posix() not in referenced]
    if extra_rel:
        raise AdmissionError(f"unreferenced extra receipt files: {extra_rel}")
    inventory = generation.inventory()
    rows: list[dict[str, Any]] = []
    ready_keys: list[tuple[str, str]] = []
    for pair_key in expected:
        by_kind = receipts.get(pair_key, {})
        ncu = by_kind.get("ncu")
        nvbit = by_kind.get("nvbit")
        if ncu is None and nvbit is None:
            rows.append({"matrix_key": pair_key[0], "variant_id": pair_key[1], "status": "pending", "typed_reason": "ncu_and_nvbit_receipts_absent", "ncu_receipt": None, "nvbit_receipt": None, "identity": None, "launch_identity": None, "process_separation": None})
            continue
        if ncu is None or nvbit is None:
            rows.append({"matrix_key": pair_key[0], "variant_id": pair_key[1], "status": "pending", "typed_reason": "ncu_receipt_absent" if ncu is None else "nvbit_receipt_absent", "ncu_receipt": ncu[4] if ncu else None, "nvbit_receipt": nvbit[4] if nvbit else None, "identity": None, "launch_identity": None, "process_separation": None})
            continue
        identity, launch, ncu_observed, ncu_process, ncu_ref = ncu
        nv_identity, nv_launch, nv_observed, nv_process, nv_ref = nvbit
        if identity != nv_identity or launch != nv_launch:
            raise AdmissionError(f"NCU/NVBit identity or launch join mismatch: {pair_key}")
        if ncu_observed != nv_observed:
            raise AdmissionError(f"NCU/NVBit observed launch join mismatch: {pair_key}")
        separation = _process_separation(ncu_process, nv_process)
        rows.append({"matrix_key": pair_key[0], "variant_id": pair_key[1], "status": "capture_ready", "typed_reason": "valid_joined_receipt_pair", "ncu_receipt": ncu_ref, "nvbit_receipt": nv_ref, "identity": identity, "launch_identity": launch, "process_separation": separation})
        ready_keys.append(pair_key)
    row_keys = [_row_key(pair_key) for pair_key in ready_keys]
    chunks = []
    if row_keys:
        chunks = [{"chunk_id": "cutlass-successor-capture-000", "rows": row_keys, "membership_sha256": canonical_sha(row_keys)}]
    if not receipt_files:
        dispatch_status, dispatch_reason = "not_admitted", "capture_receipts_absent"
    elif ready_keys and len(ready_keys) == len(expected):
        dispatch_status, dispatch_reason = "capture_ready", "all_successor_receipt_pairs_valid"
    elif ready_keys:
        dispatch_status, dispatch_reason = "partially_admitted", "capture_receipts_partially_ready"
    else:
        dispatch_status, dispatch_reason = "not_admitted", "capture_receipts_partially_ready"
    document = {
        "schema_version": ACTIVE_OVERLAY_SCHEMA_VERSION,
        "artifact_kind": "cutlass_successor_capture_admission_overlay",
        "overlay_id": OVERLAY_ID,
        "generated_at_utc": generated_at_utc,
        "base_bundle": _bundle_overlay_meta(matrix),
        "receipt_root": {"path": str(root), "inventory_sha256": canonical_sha(inventory)},
        "dispatch": {"status": dispatch_status, "m6_schedule_created": False, "typed_reason": dispatch_reason},
        "chunking": {"policy": "capture_ready_rows_only", "max_rows_per_chunk": 8, "runnable_order": row_keys, "runnable_membership_sha256": canonical_sha(row_keys)},
        "chunks": chunks,
        "rows": rows,
        "counts": {"total": len(rows), "pending": len(rows) - len(ready_keys), "capture_ready": len(ready_keys), "chunks": len(chunks)},
        "integrity": {"base_bundle_unchanged": True, "unique_matrix_keys": True, "unique_variant_ids": True, "receipt_inventory_closed": True, "only_ready_rows_chunked": True, "separate_ncu_nvbit_processes": True},
    }
    document["overlay_seal_sha256"] = canonical_sha(document)
    _validate_registered_overlay_schema(
        document, "capture admission overlay"
    )
    return document


def _receipt_generation_limits() -> dict[str, int]:
    return {
        "max_files": RECEIPT_TREE_MAX_FILES,
        "max_directories": RECEIPT_TREE_MAX_DIRECTORIES,
        "max_total_file_bytes": RECEIPT_TREE_MAX_TOTAL_FILE_BYTES,
        "max_total_relative_path_bytes": (
            RECEIPT_TREE_MAX_TOTAL_RELATIVE_PATH_BYTES
        ),
    }


def build_overlay(receipt_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    try:
        root = _external_root(receipt_root)
        _check_dt(generated_at_utc, "generated_at_utc")
        _, _, roster, matrix = validate_immutable_bundle()
        expected = _expected_rows(matrix)
        expected_roster = _expected_roster_rows(roster)
        with immutable.PinnedDirectoryReader(root) as reader:
            with reader.retain_tree_generation(
                **_receipt_generation_limits()
            ) as snapshot:
                generation = _ReceiptGeneration(root, reader, snapshot)
                return _build_overlay_from_generation(
                    generation,
                    root,
                    generated_at_utc=generated_at_utc,
                    matrix=matrix,
                    expected=expected,
                    expected_roster=expected_roster,
                )
    except AdmissionError:
        raise
    except MemoryError as exc:
        raise _memory_refusal("capture admission overlay generation", exc) from exc
    except immutable.ImmutableArtifactError as exc:
        raise AdmissionError(
            f"receipt tree immutable generation validation failed: {exc}"
        ) from exc


def _overlay_output_target(output: Path, receipt_root: Path) -> tuple[Path, Path]:
    if not output.is_absolute():
        raise AdmissionError("overlay output must be an absolute path")
    resolved_root = _external_root(receipt_root)
    target = output.resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        pass
    else:
        raise AdmissionError("overlay output must not be inside receipt root")
    return resolved_root, target


def _overlay_payload(document: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except MemoryError as exc:
        raise _memory_refusal("capture admission overlay serialization", exc) from exc


def write_overlay(document: dict[str, Any], output: Path, receipt_root: Path) -> None:
    try:
        _resolved_root, _target = _overlay_output_target(output, receipt_root)
        payload = _overlay_payload(document)
        with immutable.PinnedDirectoryReader(output.parent) as parent:
            immutable.write_regular_bytes_at(
                parent.descriptor,
                output.name,
                payload,
                mode=0o444,
            )
            parent.verify_path()
    except AdmissionError:
        raise
    except MemoryError as exc:
        raise _memory_refusal("capture admission overlay write", exc) from exc
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise AdmissionError(f"cannot exclusively write overlay {output}: {exc}") from exc


def _build_and_write_overlay(
    receipt_root: Path,
    output: Path,
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build and publish while one receipt-tree generation remains retained."""

    root, _target = _overlay_output_target(output, receipt_root)
    _check_dt(generated_at_utc, "generated_at_utc")
    _, _, roster, matrix = validate_immutable_bundle()
    expected = _expected_rows(matrix)
    expected_roster = _expected_roster_rows(roster)
    try:
        with (
            contextlib.ExitStack() as source_stack,
            contextlib.ExitStack() as output_stack,
        ):
            reader = source_stack.enter_context(
                immutable.PinnedDirectoryReader(root)
            )
            snapshot = source_stack.enter_context(
                reader.retain_tree_generation(
                    **_receipt_generation_limits()
                )
            )
            generation = _ReceiptGeneration(root, reader, snapshot)
            document = _build_overlay_from_generation(
                generation,
                root,
                generated_at_utc=generated_at_utc,
                matrix=matrix,
                expected=expected,
                expected_roster=expected_roster,
            )
            output_parent = output_stack.enter_context(
                immutable.PinnedDirectoryReader(output.parent)
            )

            def authorize_published_overlay() -> None:
                try:
                    source_stack.close()
                except immutable.ImmutableArtifactError as exc:
                    raise AdmissionError(
                        "receipt tree immutable generation validation failed: "
                        f"{exc}"
                    ) from exc
                try:
                    output_stack.close()
                except (OSError, immutable.ImmutableArtifactError) as exc:
                    raise AdmissionError(
                        f"overlay output parent closure failed: {exc}"
                    ) from exc

            immutable.publish_regular_file(
                output.parent,
                output.name,
                _overlay_payload(document),
                pinned_parent=output_parent,
                after_commit=authorize_published_overlay,
            )
            return document
    except AdmissionError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise AdmissionError(
            f"cannot transactionally publish overlay {output}: {exc}"
        ) from exc


def build_and_write_overlay(
    receipt_root: Path,
    output: Path,
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Translate every allocation failure into one typed admission refusal."""

    try:
        return _build_and_write_overlay(
            receipt_root,
            output,
            generated_at_utc=generated_at_utc,
        )
    except AdmissionError:
        raise
    except MemoryError as exc:
        raise _memory_refusal("capture admission retained overlay", exc) from exc


def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--receipt-root", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--generated-at-utc", default="2026-07-23T00:00:00Z")
        args = parser.parse_args(argv)
        build_and_write_overlay(
            args.receipt_root,
            args.output,
            generated_at_utc=args.generated_at_utc,
        )
    except MemoryError as exc:
        refusal = _memory_refusal("capture admission CLI", exc)
        print(f"CUTLASS successor capture admission failed: {refusal}", file=sys.stderr)
        return 1
    except (AdmissionError, OSError, ValueError, jsonschema.SchemaError) as exc:
        print(f"CUTLASS successor capture admission failed: {exc}", file=sys.stderr)
        return 1
    print(f"CUTLASS successor capture admission: pass=True -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
