#!/usr/bin/env python3
"""Emit and verify immutable, resumable capture-chunk DONE manifests.

A DONE manifest is a cryptographic resume boundary, not a progress counter.  It
names the exact nonempty matrix chunk, accounts for every member with one typed
outcome, pins every admitted artifact and campaign input, and semantically
revalidates current-matrix SOL native-environment attestations.  Sealing is
refused unless the repository is clean and still at the preflight/source commit.

``verify_seal`` is deliberately a consumer-side verifier: by default it re-reads
and hashes every referenced input and artifact, re-derives membership/counts,
cross-checks the capture matrix and roster authority, and re-checks git HEAD and
cleanliness.  An empty problem list is therefore safe evidence that a chunk may
be skipped during resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _import_root in (REPO_ROOT, _HERE):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
from campaign_exclusion_ledger import REASON_VOCABULARY  # noqa: E402
from tools.local_labels import capture_toolchain_contract  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import ncu_action_executor  # noqa: E402
from tools.local_labels import ncu_collection_receipt  # noqa: E402
from tools.local_labels import parse_ncu_metrics  # noqa: E402

MANIFEST_KIND = "campaign_chunk_done"
SCHEMA_VERSION = 2
SCHEMA_PATH = (REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
               / "campaign_done_manifest.schema.json")
LOCAL_LABEL_SIDECAR_SCHEMA_PATH = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "local_label_sidecar.schema.json"
)

ROSTER_SOURCES = ("portable_lcd", "simbench_bw", "sol_execbench")
CELL_DISPOSITIONS = ("admitted", "excluded", "failed")
ARTIFACT_KINDS = (
    "label_sidecar",
    "label_run_manifest",
    "ncu_report",
    "ncu_launch_observation",
    "trace_manifest",
    "trace_root",
    "trace_stats",
    "trace_threadblock",
    "trace_auxiliary",
    "trace_route_record",
    "sol_capture_evidence",
    "control_evidence",
    "control_bits",
    "control_manifest",
    "cell_result",
    "ncu_collection_receipt",
    "execution_log",
)
_ADMITTED_REQUIRED_ARTIFACT_KINDS = frozenset({
    "label_sidecar", "label_run_manifest", "ncu_report",
    "ncu_launch_observation", "ncu_collection_receipt", "trace_manifest",
    "trace_root", "trace_stats", "trace_threadblock", "control_bits",
    "control_manifest", "cell_result",
})
_TRACE_UNSUPPORTED_REQUIRED_ARTIFACT_KINDS = frozenset({
    "label_sidecar", "label_run_manifest", "ncu_report",
    "ncu_launch_observation", "trace_manifest",
    "trace_auxiliary", "trace_route_record", "control_bits",
    "control_manifest", "cell_result",
})
_INPUT_HASH_FIELDS = {
    "pins": "pins_sha256",
    "roster_authority": "roster_authority_sha256",
    "trace_budget": "trace_budget_sha256",
    "capture_matrix": "capture_matrix_sha256",
    "preflight": "preflight_sha256",
}
_SHA40_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA64_RE = re.compile(r"[0-9a-f]{64}\Z")

# This is the canonical JSON digest of the one immutable Round-27 bootstrap
# DONE manifest.  It is the sole compatibility exception to the Round-28
# admission/cell-provenance additions; all of its original seal checks still
# run.  Matching a few identity fields would make the exception forgeable, so
# compatibility is granted only to the complete manifest value.
HISTORICAL_BOOTSTRAP_CANONICAL_SHA256 = (
    "b9d22d239b6d7ce08b2a1a902ff0c2bd21a356d0b5a6eeaf6f09f955bf4c5fa8"
)
HISTORICAL_BOOTSTRAP_FILE_SHA256 = (
    "3e64d122b741d362b2945775ae73ed1200d5828e4ff273ff09f98e0bc82e2564"
)
HISTORICAL_BOOTSTRAP_REPOSITORY_ROOT = Path(
    "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite"
)

_ADMISSION_FIELDS = frozenset({
    "preflight_generated_at_utc", "capture_invocations",
})
_CAPTURE_INVOCATION_FIELDS = frozenset({"run_id", "admitted_at_utc"})
_SUPERSEDES_FIELDS = frozenset({
    "ledger_entry_id", "superseded_preflight_sha256", "superseded_path",
})
_SUPERSESSION_LEDGER_FIELDS = frozenset({
    "schema_version",
    "timestamp_utc",
    "session_id",
    "chunk_id",
    "cell_id",
    "original_preflight_sha256",
    "current_preflight_sha256",
    "source_path",
    "destination_path",
    "reason_code",
    "operator_flag",
    "source_commit",
    "matrix_seal_sha256",
    "inventory_sha256",
    "ledger_entry_id",
})
_SUPERSESSION_REASON = "PREFLIGHT_EXPIRED_BEFORE_DONE"

PREPARATION_FAILURE_ROOT_NAME = "preparation_failures"
PREPARATION_FAILURE_LEDGER_FILENAME = "preparation_failure_ledger.jsonl"
_PREPARATION_FAILURE_BACKREF_FIELDS = frozenset({
    "ledger_entry_id", "ledger_path", "archive_path",
})
CAPTURE_DEFECT_ROOT_NAME = "capture_defect_attempts"
CAPTURE_DEFECT_LEDGER_FILENAME = "capture_defect_ledger.jsonl"
_CAPTURE_DEFECT_BACKREF_FIELDS = frozenset({
    "ledger_entry_id", "ledger_path", "archive_path", "original_head",
    "reason_code",
})
SOL_NATIVE_ENV_ATTESTATION_FILENAME = "sol_native_env_fingerprint.json"
SOL_NATIVE_ENV_ATTESTATION_KIND = "sol_native_env_attestation"
SOL_NATIVE_ENV_ATTESTATION_SCHEMA_VERSION = 2
SOL_NATIVE_ENV_PIN_SECTION = "sol_native_env_fingerprint"
SOL_NATIVE_ENV_PIN_PATH = (
    "data/local_labels/campaign/sol_native_env_fingerprint_sm86.json"
)
SOL_NATIVE_ENV_COMPONENTS = (
    "python_version",
    "torch_version",
    "nvcc_version",
    "cuda_runtime_version",
    "cutlass_tree_hash",
    "cudnn_version",
    "sol_execbench_package_version",
    "nvcc_ccbin",
    "cxx_compiler",
    "c_compiler",
)


def _load_certified_sol_native_env_hash() -> str:
    """Derive the DONE verifier's hash from the tracked fingerprint authority.

    This module deliberately has no second literal copy of the environment
    identity.  A malformed, degraded, or internally stale fingerprint must make
    the verifier unavailable rather than silently certifying an obsolete host.
    """
    path = REPO_ROOT / SOL_NATIVE_ENV_PIN_PATH
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"certified SOL native fingerprint is unreadable: {path}: {exc}"
        ) from exc
    components = document.get("components") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("fingerprint_id")
        != "sm86_sol_native_env_fingerprint_v1"
        or document.get("arch") != "sm_86"
        or document.get("sku") != "rtx_3090_ti"
        or document.get("degraded") is not False
        or document.get("missing_components") != []
        or not isinstance(components, dict)
        or set(components) != set(SOL_NATIVE_ENV_COMPONENTS)
        or not all(isinstance(value, str) for value in components.values())
    ):
        raise RuntimeError("certified SOL native fingerprint contract is invalid")
    derived = hashlib.sha256(json.dumps(
        components, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")).hexdigest()
    if document.get("native_env_hash") != derived:
        raise RuntimeError(
            "certified SOL native fingerprint native_env_hash is stale"
        )
    return derived


SOL_CERTIFIED_NATIVE_ENV_HASH = _load_certified_sol_native_env_hash()
SOL_CERTIFIED_NVCC_CCBIN = "g++-14"
SOL_CERTIFIED_ENVIRONMENT = {
    "CUTLASS_DIR": "third_party/cutlass",
    "PATH_prepend": "tools/sol_execbench/.venv/bin",
    "NVCC_CCBIN": SOL_CERTIFIED_NVCC_CCBIN,
}
_SOL_ENV_ATTESTATION_FIELDS = frozenset({
    "schema_version",
    "kind",
    "identity",
    "certified_native_env_hash",
    "observed_native_env_hash",
    "certified_hash_equal",
    "degraded",
    "missing_components",
    "components",
    "components_sha256",
    "compiler_identity",
    "matrix_pin",
    "attestation_sha256",
})
_SOL_COMPILER_IDENTITY_FIELDS = frozenset({
    "requested", "resolved_realpath", "executable_sha256", "version_line",
})
_SOL_ENV_PIN_FIELDS = frozenset({"path", "sha256", "native_env_hash"})

SOL_DEFINITION_EQUIVALENCE_FILENAME = (
    "definition_equivalence_attestation.json"
)
SOL_DEFINITION_PREPARE_MANIFEST_FILENAME = "capture_prepare_manifest.json"
SOL_DEFINITION_SOURCE_FILENAME = "definition.source.json"
SOL_DEFINITION_STAGED_FILENAME = "definition.json"
SOL_DEFINITION_EQUIVALENCE_RULE_ID = "SOL_DEF_EQUIV_V2"
SOL_E387_LEGACY_SOURCE_COMMIT = (
    "e38788ca0a4457570bd562ea1389b07b3f855e22"
)
SOL_E387_LEGACY_COMPARATOR_PATH = PurePosixPath(
    "tools/local_labels/sol_definition_equivalence.py"
)
SOL_E387_LEGACY_COMPARATOR_SHA256 = (
    "573a8a84196da4c8105c1a1b742cde10bb9c1923a19fd1a056392c4eb78c9bfc"
)
SOL_E387_LEGACY_CAPTURE_ADAPTER_PATH = PurePosixPath(
    "tools/sol_execbench/src/sol_execbench/driver/capture_adapter.py"
)
SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256 = (
    "35672d28ddfcec9c9e2a9eb1021d7fc7c84b458e1865406ff0b4aa349a45d003"
)
_SOL_DEFINITION_TOP_LEVEL_FIELDS = frozenset({
    "hf_id", "op_type", "custom_inputs_entrypoint",
})

_ROUTE_RESULT_FIELDS = frozenset({
    "schema_version", "kind", "cell_id", "identity", "protocol",
    "protocol_sha256", "budget", "route_config", "full_trace", "outcome",
    "typed_reason", "method", "trace_sampling_schema_version",
    "trace_sampling_sidecar_sha256", "retained", "result_sha256",
})
_ROUTE_PROTOCOL_FIELDS = frozenset({
    "protocol_version", "source_commit", "authority_cell_sha256",
    "capture_matrix_sha256", "preflight_sha256", "pins_sha256",
    "trace_budget_sha256", "capture_manifest_sha256", "binary_sha256",
    "tracer_so_sha256", "args_hash",
})
_ROUTE_BUDGET_FIELDS = frozenset({
    "max_trace_bytes", "max_wall_time_s", "sampling_policy",
})
_ROUTE_CONFIG_FIELDS = frozenset({
    "selection_max_raw_bytes", "staging_limit_bytes", "gate_forbidden_policy",
    "reduced_shape_variant", "seed", "sample_size", "max_reconcile_attempts",
})
_ROUTE_FULL_TRACE_FIELDS = frozenset({
    "tree_sha256", "tree_bytes", "threadblock_count", "decoded_record_count",
    "expected_records", "admitted_records",
})
_ROUTE_RETAINED_FIELDS = frozenset({
    "tree_sha256", "bytes", "file_count", "artifacts",
})
_ADMITTED_ROUTE_METHODS = frozenset({
    "lossless_compressed_full",
    "wave_segment_stratified_subset",
    "reduced_shape_matched_label",
})


class DoneManifestError(Exception):
    """A typed seal refusal; callers must not write a DONE file."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sol_stage_manifest_sha256(value: Any) -> str:
    """Reproduce the unchanged SOL adapter stage-definition hash domain."""
    payload = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_historical_bootstrap(manifest: Any) -> bool:
    """Return whether ``manifest`` is the one explicitly allowlisted seal."""
    return (
        isinstance(manifest, dict)
        and _canonical_sha256(manifest)
        == HISTORICAL_BOOTSTRAP_CANONICAL_SHA256
    )


def _parse_utc_timestamp(raw: Any, where: str, problems: list[str]) -> datetime | None:
    """Parse an aware UTC RFC-3339 timestamp without normalizing bad evidence."""
    if not isinstance(raw, str) or not raw:
        problems.append(f"{where} is not a nonempty UTC timestamp: {raw!r}")
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        problems.append(f"{where} is not a valid UTC timestamp: {raw!r}")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        problems.append(f"{where} is not UTC: {raw!r}")
        return None
    return parsed


def _normalize_admission(
    admission: Any, *, problems: list[str], where: str = "admission",
) -> tuple[str | None, dict[str, datetime]]:
    """Validate the closed single-preflight invocation ledger in DONE."""
    if not isinstance(admission, dict):
        problems.append(f"{where} is not an object")
        return None, {}
    unknown = sorted(set(admission) - _ADMISSION_FIELDS)
    missing = sorted(_ADMISSION_FIELDS - set(admission))
    if unknown or missing:
        problems.append(
            f"{where} field set mismatch: missing={missing}, unknown={unknown}"
        )
    generated = admission.get("preflight_generated_at_utc")
    generated_at = _parse_utc_timestamp(
        generated, f"{where}.preflight_generated_at_utc", problems,
    )
    invocations = admission.get("capture_invocations")
    if not isinstance(invocations, list) or not invocations:
        problems.append(f"{where}.capture_invocations must be a nonempty array")
        return generated if isinstance(generated, str) else None, {}
    normalized: dict[str, datetime] = {}
    for index, row in enumerate(invocations):
        label = f"{where}.capture_invocations[{index}]"
        if not isinstance(row, dict):
            problems.append(f"{label} is not an object")
            continue
        unknown = sorted(set(row) - _CAPTURE_INVOCATION_FIELDS)
        missing = sorted(_CAPTURE_INVOCATION_FIELDS - set(row))
        if unknown or missing:
            problems.append(
                f"{label} field set mismatch: missing={missing}, unknown={unknown}"
            )
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            problems.append(f"{label}.run_id is not a nonempty string")
            continue
        admitted = _parse_utc_timestamp(
            row.get("admitted_at_utc"), f"{label}.admitted_at_utc", problems,
        )
        if admitted is not None and generated_at is not None:
            from campaign_preflight import (  # noqa: E402
                PREFLIGHT_FRESHNESS_SECONDS,
                PREFLIGHT_FUTURE_SKEW_SECONDS,
            )
            age_seconds = (admitted - generated_at).total_seconds()
            if age_seconds > PREFLIGHT_FRESHNESS_SECONDS:
                problems.append(
                    f"{label} admission exceeds the inclusive "
                    f"{PREFLIGHT_FRESHNESS_SECONDS:.3f}s preflight window"
                )
            if age_seconds < -PREFLIGHT_FUTURE_SKEW_SECONDS:
                problems.append(
                    f"{label} preflight exceeds the "
                    f"{PREFLIGHT_FUTURE_SKEW_SECONDS:.3f}s future-skew bound"
                )
        if run_id in normalized:
            problems.append(f"{where} has duplicate run_id {run_id!r}")
        elif admitted is not None:
            normalized[run_id] = admitted
    return generated if isinstance(generated, str) else None, normalized


def authority_cell_sha256(cell: dict) -> str:
    """Return the canonical identity of one complete authority cell row."""
    if not isinstance(cell, dict):
        raise DoneManifestError("authority cell is not an object")
    return _canonical_sha256(cell)


def _identity(row: Any, where: str, problems: list[str]) -> tuple[str, str] | None:
    if not isinstance(row, dict):
        problems.append(f"{where} is not an object")
        return None
    source = row.get("roster_source")
    slug = row.get("slug")
    if source not in ROSTER_SOURCES:
        problems.append(
            f"{where}.roster_source={source!r} is not one of {list(ROSTER_SOURCES)}")
    if not isinstance(slug, str) or not slug:
        problems.append(f"{where}.slug is not a nonempty string: {slug!r}")
    if source not in ROSTER_SOURCES or not isinstance(slug, str) or not slug:
        return None
    return source, slug


def chunk_membership_sha256(members: list[dict]) -> str:
    """Hash canonical, sorted ``(roster_source, slug)`` identities.

    Authority-row hashes deliberately are not part of this digest: they are
    sealed independently and the membership digest remains the exact matrix
    identity set requested by the chunk.
    """
    problems: list[str] = []
    identities = [
        identity for i, row in enumerate(members)
        if (identity := _identity(row, f"chunk_membership[{i}]", problems))
        is not None
    ]
    duplicates = sorted(k for k, n in Counter(identities).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate chunk authority identities: {duplicates}")
    if not identities:
        problems.append("chunk membership must be nonempty")
    if problems:
        raise DoneManifestError("; ".join(problems))
    canonical = [
        {"roster_source": source, "slug": slug}
        for source, slug in sorted(identities)
    ]
    return _canonical_sha256(canonical)


def current_source_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(repo_root),
        )
    except OSError:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_tree_clean(repo_root: Path = REPO_ROOT) -> bool:
    """Return the real tree state; a failed git query is never called clean."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "-uall"], capture_output=True,
            text=True, cwd=str(repo_root),
        )
    except OSError:
        return False
    return proc.returncode == 0 and not proc.stdout.strip()


def _valid_relative_artifact_path(raw: Any) -> bool:
    if (
        not isinstance(raw, str)
        or not raw
        or "\x00" in raw
        or "\\" in raw
        or raw.splitlines() != [raw]
    ):
        return False
    path = PurePosixPath(raw)
    return (not path.is_absolute() and raw == path.as_posix()
            and path.as_posix() != "." and ".." not in path.parts)


def _normalize_membership(members: list[dict]) -> list[dict]:
    if not isinstance(members, list):
        raise DoneManifestError("chunk_membership is not an array")
    problems: list[str] = []
    normalized: list[dict] = []
    identities: list[tuple[str, str]] = []
    for i, row in enumerate(members):
        identity = _identity(row, f"chunk_membership[{i}]", problems)
        authority_sha = row.get("authority_cell_sha256") if isinstance(row, dict) else None
        if not isinstance(authority_sha, str) or not _SHA64_RE.fullmatch(authority_sha):
            problems.append(
                f"chunk_membership[{i}].authority_cell_sha256 is not sha256: "
                f"{authority_sha!r}")
        if identity is not None:
            identities.append(identity)
            normalized.append({
                "roster_source": identity[0],
                "slug": identity[1],
                "authority_cell_sha256": authority_sha,
            })
    duplicates = sorted(k for k, n in Counter(identities).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate chunk authority identities: {duplicates}")
    if not normalized:
        problems.append("chunk membership must be nonempty")
    if problems:
        raise DoneManifestError("; ".join(problems))
    return sorted(normalized, key=lambda r: (r["roster_source"], r["slug"]))


def _normalize_artifacts(artifacts: Any, where: str, problems: list[str]) -> list[dict]:
    if not isinstance(artifacts, list):
        problems.append(f"{where} is not an array")
        return []
    normalized: list[dict] = []
    seen: set[str] = set()
    allowed = {"artifact_kind", "relative_path", "sha256", "size_bytes"}
    for i, artifact in enumerate(artifacts):
        label = f"{where}[{i}]"
        if not isinstance(artifact, dict):
            problems.append(f"{label} is not an object")
            continue
        unknown = sorted(set(artifact) - allowed)
        if unknown:
            problems.append(f"{label} has unknown fields {unknown}")
        kind = artifact.get("artifact_kind")
        rel = artifact.get("relative_path")
        sha = artifact.get("sha256")
        size = artifact.get("size_bytes")
        if kind not in ARTIFACT_KINDS:
            problems.append(
                f"{label}.artifact_kind={kind!r} is not one of {list(ARTIFACT_KINDS)}")
        if not _valid_relative_artifact_path(rel):
            problems.append(
                f"{label}.relative_path must stay beneath output_root: {rel!r}")
        elif rel in seen:
            problems.append(f"{where} has duplicate relative_path {rel!r}")
        else:
            seen.add(rel)
        if not isinstance(sha, str) or not _SHA64_RE.fullmatch(sha):
            problems.append(f"{label}.sha256 is not sha256: {sha!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            problems.append(f"{label}.size_bytes is not a non-negative int: {size!r}")
        elif size == 0 and kind != "execution_log":
            problems.append(
                f"{label}.size_bytes must be positive for semantic artifact {kind!r}")
        normalized.append({
            "artifact_kind": kind,
            "relative_path": rel,
            "sha256": sha,
            "size_bytes": size,
        })
    return normalized


def _is_uuid4(raw: Any) -> bool:
    """Return whether ``raw`` is the canonical lower-case text of a UUID4."""
    if not isinstance(raw, str):
        return False
    try:
        parsed = uuid.UUID(raw)
    except (AttributeError, ValueError):
        return False
    return parsed.version == 4 and str(parsed) == raw


def _normalize_prior_preparation_failures(
    value: Any, *, where: str, problems: list[str],
) -> list[dict]:
    """Validate the closed active-journal back-reference representation."""
    if not isinstance(value, list):
        problems.append(f"{where} is not an array")
        return []
    normalized: list[dict] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(value):
        label = f"{where}[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{label} is not an object")
            continue
        unknown = sorted(set(raw) - _PREPARATION_FAILURE_BACKREF_FIELDS)
        missing = sorted(_PREPARATION_FAILURE_BACKREF_FIELDS - set(raw))
        if unknown or missing:
            problems.append(
                f"{label} field set mismatch: missing={missing}, unknown={unknown}"
            )
        entry_id = raw.get("ledger_entry_id")
        ledger_raw = raw.get("ledger_path")
        archive_raw = raw.get("archive_path")
        if not _is_uuid4(entry_id):
            problems.append(f"{label}.ledger_entry_id is not a canonical UUID4")
        elif entry_id in seen_ids:
            problems.append(f"{where} has duplicate ledger_entry_id {entry_id!r}")
        else:
            seen_ids.add(entry_id)
        ledger_path = Path(ledger_raw) if isinstance(ledger_raw, str) else None
        archive_path = Path(archive_raw) if isinstance(archive_raw, str) else None
        if (
            ledger_path is None
            or not ledger_path.is_absolute()
            or str(ledger_path) != ledger_raw
            or ledger_path.name != PREPARATION_FAILURE_LEDGER_FILENAME
            or ledger_path.parent.name != PREPARATION_FAILURE_ROOT_NAME
        ):
            problems.append(
                f"{label}.ledger_path is not the canonical preparation ledger path"
            )
        if (
            archive_path is None
            or not archive_path.is_absolute()
            or str(archive_path) != archive_raw
            or archive_path.parent.name != PREPARATION_FAILURE_ROOT_NAME
        ):
            problems.append(
                f"{label}.archive_path is not a canonical preparation archive path"
            )
        if (
            ledger_path is not None
            and archive_path is not None
            and ledger_path.parent != archive_path.parent
        ):
            problems.append(
                f"{label} ledger_path/archive_path do not share one archive root"
            )
        normalized.append({
            "ledger_entry_id": entry_id,
            "ledger_path": ledger_raw,
            "archive_path": archive_raw,
        })
    return normalized


def _normalize_prior_capture_defect_attempts(
    value: Any, *, where: str, problems: list[str],
) -> list[dict]:
    """Validate the separate, closed non-authoritative attempt references."""
    if not isinstance(value, list):
        problems.append(f"{where} is not an array")
        return []
    normalized: list[dict] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(value):
        label = f"{where}[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{label} is not an object")
            continue
        unknown = sorted(set(raw) - _CAPTURE_DEFECT_BACKREF_FIELDS)
        missing = sorted(_CAPTURE_DEFECT_BACKREF_FIELDS - set(raw))
        if unknown or missing:
            problems.append(
                f"{label} field set mismatch: missing={missing}, unknown={unknown}"
            )
        entry_id = raw.get("ledger_entry_id")
        ledger_raw = raw.get("ledger_path")
        archive_raw = raw.get("archive_path")
        original_head = raw.get("original_head")
        reason_code = raw.get("reason_code")
        if not _is_uuid4(entry_id):
            problems.append(f"{label}.ledger_entry_id is not a canonical UUID4")
        elif entry_id in seen_ids:
            problems.append(f"{where} has duplicate ledger_entry_id {entry_id!r}")
        else:
            seen_ids.add(entry_id)
        ledger_path = Path(ledger_raw) if isinstance(ledger_raw, str) else None
        archive_path = Path(archive_raw) if isinstance(archive_raw, str) else None
        if (
            ledger_path is None or not ledger_path.is_absolute()
            or str(ledger_path) != ledger_raw
            or ledger_path.name != CAPTURE_DEFECT_LEDGER_FILENAME
            or ledger_path.parent.name != CAPTURE_DEFECT_ROOT_NAME
        ):
            problems.append(
                f"{label}.ledger_path is not the canonical capture-defect ledger path"
            )
        if (
            archive_path is None or not archive_path.is_absolute()
            or str(archive_path) != archive_raw
            or archive_path.parent.name != CAPTURE_DEFECT_ROOT_NAME
            or (
                isinstance(entry_id, str)
                and archive_path.name != f"capture-defect-{entry_id}"
            )
        ):
            problems.append(
                f"{label}.archive_path is not a canonical capture-defect archive path"
            )
        if (
            ledger_path is not None and archive_path is not None
            and ledger_path.parent != archive_path.parent
        ):
            problems.append(
                f"{label} ledger_path/archive_path do not share one archive root"
            )
        if not isinstance(original_head, str) or not _SHA40_RE.fullmatch(original_head):
            problems.append(f"{label}.original_head is not a commit")
        if reason_code != "CAPTURE_DRIVER_INTEGRITY_FALSE_POSITIVE_BEFORE_ROI":
            problems.append(f"{label}.reason_code is not the closed defect reason")
        normalized.append({
            "ledger_entry_id": entry_id,
            "ledger_path": ledger_raw,
            "archive_path": archive_raw,
            "original_head": original_head,
            "reason_code": reason_code,
        })
    return normalized


def _definition_attestation_structure_problems(
    value: Any, *, label: str, allow_e387_legacy_shape: bool = False,
) -> list[str]:
    """Route structural eligibility through the single certifying V2 helper."""
    try:
        import sol_definition_equivalence as sol_def  # noqa: E402
    except ImportError as exc:
        return [
            f"SOL_ATTESTATION_STATE_MISMATCH: {label}: "
            f"definition-equivalence verifier unavailable: {exc}"
        ]
    if (
        allow_e387_legacy_shape
        and sol_def.is_e387_legacy_attestation_shape(value)
    ):
        return []
    try:
        sol_def.validate_attestation_shape_and_eligibility(value)
    except sol_def.SolDefinitionEquivalenceError as exc:
        return [f"{exc.code}: {label}: {exc.detail}"]
    return []


def _is_terminal_capture_result(row: dict) -> bool:
    return (
        row.get("disposition") == "admitted"
        or (
            row.get("disposition") == "failed"
            and row.get("typed_reason") == "trace_unsupported"
        )
    )


def _is_sol_env_attestation_artifact(artifact: dict) -> bool:
    rel = artifact.get("relative_path")
    return (
        artifact.get("artifact_kind") == "sol_capture_evidence"
        and isinstance(rel, str)
        and PurePosixPath(rel).name == SOL_NATIVE_ENV_ATTESTATION_FILENAME
    )


def _normalize_results(
    results: list[dict], *, allow_historical_bootstrap: bool = False,
    allow_e387_legacy_definition_attestation: bool = False,
) -> list[dict]:
    if not isinstance(results, list):
        raise DoneManifestError("cell_results is not an array")
    problems: list[str] = []
    normalized: list[dict] = []
    identities: list[tuple[str, str]] = []
    allowed = {
        "roster_source", "slug", "authority_cell_sha256", "disposition",
        "typed_reason", "detail", "capture_preflight_sha256",
        "capture_started_at_utc", "admission_run_id", "supersedes",
        "prior_preparation_failures", "prior_capture_defect_attempts",
        "definition_equivalence",
        "definition_equivalence_attestation_sha256", "artifacts",
    }
    for i, row in enumerate(results):
        label = f"cell_results[{i}]"
        identity = _identity(row, label, problems)
        if not isinstance(row, dict):
            continue
        unknown = sorted(set(row) - allowed)
        if unknown:
            problems.append(f"{label} has unknown fields {unknown}")
        authority_sha = row.get("authority_cell_sha256")
        if not isinstance(authority_sha, str) or not _SHA64_RE.fullmatch(authority_sha):
            problems.append(f"{label}.authority_cell_sha256 is not sha256: {authority_sha!r}")
        disposition = row.get("disposition")
        if disposition not in CELL_DISPOSITIONS:
            problems.append(
                f"{label}.disposition={disposition!r} is not one of "
                f"{list(CELL_DISPOSITIONS)}")
        reason = row.get("typed_reason")
        if disposition == "admitted":
            if reason is not None:
                problems.append(f"{label} admitted outcome must have typed_reason=null")
        elif disposition in {"excluded", "failed"} and reason not in REASON_VOCABULARY:
            problems.append(
                f"{label}.typed_reason={reason!r} is not in the closed vocabulary "
                f"{list(REASON_VOCABULARY)}")
        detail = row.get("detail")
        if detail is not None and not isinstance(detail, str):
            problems.append(f"{label}.detail is not a string or null")
        provenance_fields = {
            "capture_preflight_sha256", "capture_started_at_utc",
            "admission_run_id", "supersedes", "prior_preparation_failures",
            "prior_capture_defect_attempts",
            "definition_equivalence",
            "definition_equivalence_attestation_sha256",
        }
        provenance_present = provenance_fields <= set(row)
        if not provenance_present and not allow_historical_bootstrap:
            problems.append(
                f"{label} missing required provenance fields "
                f"{sorted(provenance_fields - set(row))}"
            )
        capture_preflight = row.get("capture_preflight_sha256")
        if provenance_present and (
            not isinstance(capture_preflight, str)
            or not _SHA64_RE.fullmatch(capture_preflight)
        ):
            problems.append(f"{label}.capture_preflight_sha256 is not sha256")
        capture_started = row.get("capture_started_at_utc")
        if provenance_present:
            _parse_utc_timestamp(
                capture_started, f"{label}.capture_started_at_utc", problems,
            )
        admission_run_id = row.get("admission_run_id")
        if provenance_present and (
            not isinstance(admission_run_id, str) or not admission_run_id
        ):
            problems.append(f"{label}.admission_run_id is not a nonempty string")
        supersedes = row.get("supersedes")
        if provenance_present and supersedes is not None and (
            not isinstance(supersedes, dict)
            or set(supersedes) != _SUPERSEDES_FIELDS
        ):
            problems.append(f"{label}.supersedes is malformed")
        prior_preparation_failures = _normalize_prior_preparation_failures(
            row.get("prior_preparation_failures"),
            where=f"{label}.prior_preparation_failures", problems=problems,
        ) if provenance_present else []
        prior_capture_defect_attempts = _normalize_prior_capture_defect_attempts(
            row.get("prior_capture_defect_attempts"),
            where=f"{label}.prior_capture_defect_attempts", problems=problems,
        ) if provenance_present else []
        definition_equivalence = row.get("definition_equivalence")
        definition_equivalence_sha256 = row.get(
            "definition_equivalence_attestation_sha256"
        )
        if provenance_present and identity is not None:
            if identity[0] == "sol_execbench":
                problems.extend(_definition_attestation_structure_problems(
                    definition_equivalence,
                    label=f"{label}.definition_equivalence",
                    allow_e387_legacy_shape=(
                        allow_e387_legacy_definition_attestation
                    ),
                ))
                if (
                    not isinstance(definition_equivalence_sha256, str)
                    or not _SHA64_RE.fullmatch(definition_equivalence_sha256)
                ):
                    problems.append(
                        "SOL_ATTESTATION_STATE_MISMATCH: "
                        f"{label}.definition_equivalence_attestation_sha256 "
                        "is not sha256"
                    )
            elif (
                definition_equivalence is not None
                or definition_equivalence_sha256 is not None
            ):
                problems.append(
                    f"{label} non-SOL definition equivalence fields must be null"
                )
        artifacts = _normalize_artifacts(row.get("artifacts"), f"{label}.artifacts", problems)
        kinds = {a["artifact_kind"] for a in artifacts}
        cell_result_count = sum(
            artifact["artifact_kind"] == "cell_result" for artifact in artifacts
        )
        if cell_result_count != 1:
            problems.append(
                f"{label} requires exactly one cell_result artifact; "
                f"found {cell_result_count}"
            )
        ncu_launch_observation_count = sum(
            artifact["artifact_kind"] == "ncu_launch_observation"
            for artifact in artifacts
        )
        if not allow_historical_bootstrap and (
            disposition == "admitted"
            or (disposition == "failed" and reason == "trace_unsupported")
        ) and ncu_launch_observation_count != 1:
            problems.append(
                f"{label} terminal traced outcome requires exactly one "
                "ncu_launch_observation artifact; "
                f"found {ncu_launch_observation_count}"
            )
        for artifact in artifacts:
            rel = artifact.get("relative_path")
            if isinstance(rel, str) and "superseded" in PurePosixPath(rel).parts:
                problems.append(
                    f"{label} publishes an artifact beneath superseded/: {rel!r}"
                )
            if (
                isinstance(rel, str)
                and PREPARATION_FAILURE_ROOT_NAME in PurePosixPath(rel).parts
            ):
                problems.append(
                    "PREPARATION_ARCHIVE_LEAK: "
                    f"{label} publishes an artifact beneath "
                    f"{PREPARATION_FAILURE_ROOT_NAME}/: {rel!r}"
                )
            if (
                isinstance(rel, str)
                and CAPTURE_DEFECT_ROOT_NAME in PurePosixPath(rel).parts
            ):
                problems.append(
                    "CAPTURE_DEFECT_ARCHIVE_LEAK: "
                    f"{label} publishes an artifact beneath "
                    f"{CAPTURE_DEFECT_ROOT_NAME}/: {rel!r}"
                )
        if disposition == "admitted":
            required_kinds = _ADMITTED_REQUIRED_ARTIFACT_KINDS
            if allow_historical_bootstrap:
                required_kinds = required_kinds - {
                    "ncu_launch_observation", "ncu_collection_receipt",
                }
            missing = sorted(required_kinds - kinds)
            if missing:
                problems.append(
                    f"{label} admitted outcome missing required artifact kinds {missing}")
        elif disposition == "failed" and reason == "trace_unsupported":
            required_kinds = _TRACE_UNSUPPORTED_REQUIRED_ARTIFACT_KINDS
            if allow_historical_bootstrap:
                required_kinds = required_kinds - {"ncu_launch_observation"}
            missing = sorted(required_kinds - kinds)
            if missing:
                problems.append(
                    f"{label} trace_unsupported outcome missing required "
                    f"artifact kinds {missing}"
                )
        if (
            identity is not None
            and identity[0] == "sol_execbench"
            and _is_terminal_capture_result(row)
        ):
            attestations = [
                artifact for artifact in artifacts
                if _is_sol_env_attestation_artifact(artifact)
            ]
            if len(attestations) != 1:
                problems.append(
                    f"{label} terminal SOL outcome requires exactly one "
                    f"sol_capture_evidence/{SOL_NATIVE_ENV_ATTESTATION_FILENAME}; "
                    f"found {len(attestations)}"
                )
        if identity is not None:
            identities.append(identity)
            normalized_row = {
                "roster_source": identity[0],
                "slug": identity[1],
                "authority_cell_sha256": authority_sha,
                "disposition": disposition,
                "typed_reason": reason,
                "detail": detail,
                "artifacts": sorted(
                    artifacts, key=lambda a: str(a.get("relative_path", ""))),
            }
            if provenance_present:
                normalized_row.update({
                    "capture_preflight_sha256": capture_preflight,
                    "capture_started_at_utc": capture_started,
                    "admission_run_id": admission_run_id,
                    "supersedes": supersedes,
                    "prior_preparation_failures": prior_preparation_failures,
                    "prior_capture_defect_attempts": (
                        prior_capture_defect_attempts
                    ),
                    "definition_equivalence": definition_equivalence,
                    "definition_equivalence_attestation_sha256": (
                        definition_equivalence_sha256
                    ),
                })
            normalized.append(normalized_row)
    duplicates = sorted(k for k, n in Counter(identities).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate cell-result authority identities: {duplicates}")
    if not normalized:
        problems.append("cell_results must be nonempty")
    if problems:
        raise DoneManifestError("; ".join(problems))
    return sorted(normalized, key=lambda r: (r["roster_source"], r["slug"]))


def _identity_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["roster_source"], row["slug"]): row for row in rows}


def _validate_hash(name: str, value: Any, problems: list[str]) -> None:
    if not isinstance(value, str) or not _SHA64_RE.fullmatch(value):
        problems.append(f"{name} is not sha256: {value!r}")


def _semantic_problems(manifest: dict) -> list[str]:
    """Re-derive all relationships that JSON Schema cannot express."""
    problems: list[str] = []
    pre = manifest.get("source_commit_pre")
    post = manifest.get("source_commit_post")
    head = manifest.get("source_head_at_seal")
    for name, value in (
        ("source_commit_pre", pre), ("source_commit_post", post),
        ("source_head_at_seal", head),
    ):
        if not isinstance(value, str) or not _SHA40_RE.fullmatch(value):
            problems.append(f"{name} is not a 40-hex commit sha: {value!r}")
    if not (pre == post == head):
        problems.append(
            f"source/head drift: pre={pre!r} post={post!r} head={head!r}")
    if manifest.get("source_commit_stable") is not True:
        problems.append("source_commit_stable is not true")
    if manifest.get("working_tree_clean") is not True:
        problems.append("working_tree_clean must be true; dirty chunks cannot seal")

    historical_bootstrap = _is_historical_bootstrap(manifest)
    admission_runs: dict[str, datetime] = {}
    if not historical_bootstrap:
        _, admission_runs = _normalize_admission(
            manifest.get("admission"), problems=problems,
        )
        _validate_hash(
            "matrix_seal_sha256", manifest.get("matrix_seal_sha256"), problems,
        )

    membership = manifest.get("chunk_membership")
    results = manifest.get("cell_results")
    try:
        normalized_membership = _normalize_membership(membership)
    except DoneManifestError as exc:
        problems.append(str(exc))
        normalized_membership = []
    try:
        normalized_results = _normalize_results(
            results,
            allow_historical_bootstrap=historical_bootstrap,
            allow_e387_legacy_definition_attestation=(
                pre == post == head == SOL_E387_LEGACY_SOURCE_COMMIT
            ),
        )
    except DoneManifestError as exc:
        problems.append(str(exc))
        normalized_results = []

    member_map = _identity_map(normalized_membership)
    result_map = _identity_map(normalized_results)
    missing_results = sorted(set(member_map) - set(result_map))
    extra_results = sorted(set(result_map) - set(member_map))
    if missing_results:
        problems.append(f"chunk members missing cell results: {missing_results}")
    if extra_results:
        problems.append(f"cell results not declared by chunk membership: {extra_results}")
    for identity in sorted(set(member_map) & set(result_map)):
        if (member_map[identity].get("authority_cell_sha256")
                != result_map[identity].get("authority_cell_sha256")):
            problems.append(f"authority hash differs between membership/result for {identity}")
    if not historical_bootstrap:
        for index, row in enumerate(normalized_results):
            label = f"cell_results[{index}]"
            if row.get("capture_preflight_sha256") != manifest.get("preflight_sha256"):
                problems.append(
                    f"{label}.capture_preflight_sha256 does not equal DONE preflight"
                )
            run_id = row.get("admission_run_id")
            admitted = admission_runs.get(run_id) if isinstance(run_id, str) else None
            started_problems: list[str] = []
            started = _parse_utc_timestamp(
                row.get("capture_started_at_utc"),
                f"{label}.capture_started_at_utc", started_problems,
            )
            if admitted is None:
                problems.append(
                    f"{label}.admission_run_id={run_id!r} is absent from DONE admission"
                )
            elif started is not None and started < admitted:
                problems.append(f"{label} capture started before its admission")

    expected = manifest.get("expected_cells")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        problems.append(f"expected_cells must be a positive int: {expected!r}")
    elif expected != len(normalized_membership):
        problems.append(
            f"expected_cells={expected} != exact membership size={len(normalized_membership)}")

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        problems.append("counts is not an object")
        counts = {}
    derived = Counter(row.get("disposition") for row in normalized_results)
    expected_counts = {name: derived[name] for name in CELL_DISPOSITIONS}
    if counts != expected_counts:
        problems.append(f"counts={counts!r} != derived cell-result counts={expected_counts!r}")

    failures = manifest.get("typed_failures")
    expected_failures = [
        {
            "roster_source": row["roster_source"],
            "slug": row["slug"],
            "typed_reason": row["typed_reason"],
            "detail": row["detail"],
        }
        for row in normalized_results if row.get("disposition") == "failed"
    ]
    if failures != expected_failures:
        problems.append("typed_failures does not exactly equal failed cell results")

    if normalized_membership:
        try:
            derived_membership_sha = chunk_membership_sha256(normalized_membership)
            if manifest.get("chunk_membership_sha256") != derived_membership_sha:
                problems.append(
                    "chunk_membership_sha256 does not match canonical sorted identities")
        except DoneManifestError as exc:
            problems.append(str(exc))

    for name, field in _INPUT_HASH_FIELDS.items():
        _validate_hash(field, manifest.get(field), problems)
        paths = manifest.get("input_paths")
        if not isinstance(paths, dict) or not isinstance(paths.get(name), str) \
                or not paths.get(name):
            problems.append(f"input_paths.{name} is not a nonempty path")

    output_root = manifest.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        problems.append(f"output_root must be an absolute path: {output_root!r}")
    elif (
        not _is_historical_bootstrap(manifest)
        and "superseded" in Path(output_root).parts
    ):
        problems.append(f"published output_root is beneath superseded/: {output_root!r}")
    elif (
        not _is_historical_bootstrap(manifest)
        and PREPARATION_FAILURE_ROOT_NAME in Path(output_root).parts
    ):
        problems.append(
            "PREPARATION_ARCHIVE_LEAK: published output_root is beneath "
            f"{PREPARATION_FAILURE_ROOT_NAME}/: {output_root!r}"
        )

    global_artifacts: set[str] = set()
    for row in normalized_results:
        for artifact in row["artifacts"]:
            rel = artifact["relative_path"]
            if rel in global_artifacts:
                problems.append(f"artifact path is shared by multiple cells: {rel!r}")
            global_artifacts.add(rel)
    return problems


def _resolve_input_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if ".." in path.parts:
        raise DoneManifestError(f"relative input path escapes repo_root: {raw!r}")
    return repo_root / path


def _historical_bootstrap_repo_reference(
    path: Path, *, manifest: dict, repo_root: Path,
) -> Path:
    """Map the one hash-allowlisted bootstrap's former checkout to this clone.

    Round-27 recorded two tracked inputs with absolute paths in its source
    checkout.  Those paths are location identifiers, not mutable external
    authorities: their sealed bytes live in ``source_commit_pre``.  Only the
    exact allowlisted DONE receives this relocation; current/live manifests
    continue to require the bytes at every absolute external path verbatim.
    """
    if not _is_historical_bootstrap(manifest):
        return path
    try:
        relative = path.relative_to(HISTORICAL_BOOTSTRAP_REPOSITORY_ROOT)
    except ValueError:
        return path
    if not relative.parts or ".." in relative.parts:
        return path
    return repo_root / relative


def _repo_relative_path(path: Path, repo_root: Path) -> PurePosixPath | None:
    """Return a canonical repository-relative path without requiring existence."""
    try:
        rel = path.resolve(strict=False).relative_to(repo_root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    if not rel.parts or ".." in rel.parts:
        return None
    return PurePosixPath(*rel.parts)


def _git_blob_at_commit(
    repo_root: Path, commit: Any, rel: PurePosixPath,
) -> bytes | None:
    """Read one tracked file exactly as it existed at a sealed source commit."""
    if not isinstance(commit, str) or not _SHA40_RE.fullmatch(commit):
        return None
    try:
        proc = subprocess.run(
            ["git", "show", f"{commit}:{rel.as_posix()}"],
            cwd=repo_root, capture_output=True,
        )
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def _reference_bytes_at_seal(
    path: Path, *, expected_sha256: Any, repo_root: Path, source_commit: Any,
) -> tuple[bytes | None, str | None, str]:
    """Resolve immutable external bytes or a tracked file at its sealed commit.

    Current bytes win when they still match.  When a later clean commit changed
    a repository-relative input, historical audit uses ``git show`` at the
    DONE source commit.  External files never receive this fallback.
    """
    current: bytes | None = None
    try:
        current = path.read_bytes()
    except OSError:
        pass
    if current is not None:
        current_sha = hashlib.sha256(current).hexdigest()
        if current_sha == expected_sha256:
            return current, current_sha, "current"
    rel = _repo_relative_path(path, repo_root)
    if rel is not None:
        historical = _git_blob_at_commit(repo_root, source_commit, rel)
        if historical is not None:
            historical_sha = hashlib.sha256(historical).hexdigest()
            if historical_sha == expected_sha256:
                return historical, historical_sha, "sealed_commit"
            if current is None:
                return historical, historical_sha, "sealed_commit_mismatch"
    if current is not None:
        return current, hashlib.sha256(current).hexdigest(), "current_mismatch"
    return None, None, "unavailable"


def _load_json_bytes(
    payload: bytes, path: Path, label: str, problems: list[str],
) -> dict | None:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        problems.append(f"invalid JSON in {label} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        problems.append(f"{label} {path} is not a JSON object")
        return None
    return value


def _load_json(path: Path, label: str, problems: list[str]) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        problems.append(f"cannot read {label} {path}: {exc}")
        return None
    except (json.JSONDecodeError, UnicodeError) as exc:
        problems.append(f"invalid JSON in {label} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        problems.append(f"{label} {path} is not a JSON object")
        return None
    return value


def _load_strict_json(path: Path, label: str, problems: list[str]) -> dict | None:
    try:
        value = immutable.strict_json_loads(path.read_bytes(), label=label)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        problems.append(f"invalid {label} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        problems.append(f"{label} {path} is not a JSON object")
        return None
    return value


def _matrix_chunk_identities(matrix: dict, chunk_id: str, problems: list[str]) -> set[tuple[str, str]]:
    chunks = matrix.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        problems.append("capture matrix lacks required nonempty top-level chunks[]")
        return set()
    matches = [row for row in chunks if isinstance(row, dict)
               and row.get("chunk_id") == chunk_id]
    if len(matches) != 1:
        problems.append(
            f"capture matrix must contain exactly one chunk_id={chunk_id!r}; "
            f"found {len(matches)}")
        return set()
    chunk = matches[0]
    cells = chunk.get("cells")
    if not isinstance(cells, list) or not cells:
        problems.append(f"capture matrix chunk {chunk_id!r} has no cells")
        return set()
    identities: list[tuple[str, str]] = []
    for i, row in enumerate(cells):
        identity = _identity(row, f"capture_matrix.chunks[{chunk_id}].cells[{i}]", problems)
        if identity is not None:
            identities.append(identity)
    duplicates = sorted(k for k, n in Counter(identities).items() if n > 1)
    if duplicates:
        problems.append(f"capture matrix chunk has duplicate identities: {duplicates}")
    membership_sha = chunk.get("membership_sha256")
    if membership_sha is not None:
        try:
            actual_sha = chunk_membership_sha256(cells)
        except DoneManifestError as exc:
            problems.append(str(exc))
        else:
            if membership_sha != actual_sha:
                problems.append(
                    f"capture matrix chunk membership_sha256={membership_sha!r} "
                    f"!= canonical identities hash={actual_sha}")
    return set(identities)


def _schema_problems(manifest: dict) -> list[str]:
    """Return structural schema failures without throwing on a tampered seal."""
    if _is_historical_bootstrap(manifest):
        return []
    import jsonschema
    try:
        schema = json.loads(SCHEMA_PATH.read_text())
        validator = jsonschema.Draft202012Validator(schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        return [f"DONE schema unavailable or invalid: {exc}"]
    return [
        f"DONE schema violation at {'/'.join(str(p) for p in error.path) or '<root>'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(manifest),
            key=lambda e: tuple(str(part) for part in e.path),
        )
    ]


def _artifacts_by_kind(row: dict) -> dict[str, list[dict]]:
    indexed: dict[str, list[dict]] = {}
    artifacts = row.get("artifacts")
    for artifact in artifacts if isinstance(artifacts, list) else []:
        if isinstance(artifact, dict):
            indexed.setdefault(str(artifact.get("artifact_kind")), []).append(artifact)
    return indexed


def _resolve_sealed_artifact(
    artifact: dict, *, output_root: Path, label: str, problems: list[str],
) -> Path | None:
    rel = artifact.get("relative_path")
    if not _valid_relative_artifact_path(rel):
        problems.append(f"{label} has unsafe artifact path {rel!r}")
        return None
    candidate = output_root / rel
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        problems.append(f"{label} is unavailable/unsafe: {exc}")
        return None
    if candidate.is_symlink() or not resolved.is_file():
        problems.append(f"{label} is not a regular file")
        return None
    try:
        actual_size = resolved.stat().st_size
        actual_sha = _sha256_file(resolved)
    except OSError as exc:
        problems.append(f"cannot hash {label}: {exc}")
        return None
    if actual_size != artifact.get("size_bytes"):
        problems.append(f"{label} size seal is stale")
    if actual_sha != artifact.get("sha256"):
        problems.append(f"{label} sha256 seal is stale")
    return resolved


def _verify_superseded_inventory(
    destination: Path, *, expected_sha256: Any, label: str,
    problems: list[str],
) -> None:
    inventory_path = destination / "inventory.json"
    try:
        inventory_sha = _sha256_file(inventory_path)
        inventory = json.loads(inventory_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        problems.append(f"{label} supersession inventory is unreadable: {exc}")
        return
    if inventory_sha != expected_sha256:
        problems.append(
            f"{label} supersession inventory hash mismatch: "
            f"ledger={expected_sha256!r} actual={inventory_sha}"
        )
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"schema_version", "files"}
        or inventory.get("schema_version") != 1
        or not isinstance(inventory.get("files"), list)
        or not inventory["files"]
    ):
        problems.append(f"{label} supersession inventory document is malformed")
        return
    claimed: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(inventory["files"]):
        item_label = f"{label} supersession inventory.files[{index}]"
        if not isinstance(row, dict) or set(row) != {
            "relative_path", "sha256", "size_bytes",
        }:
            problems.append(f"{item_label} is malformed")
            continue
        rel = row.get("relative_path")
        sha = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not _valid_relative_artifact_path(rel)
            or rel == "inventory.json"
            or rel in seen
        ):
            problems.append(f"{item_label}.relative_path is invalid: {rel!r}")
            continue
        if not isinstance(sha, str) or not _SHA64_RE.fullmatch(sha):
            problems.append(f"{item_label}.sha256 is invalid: {sha!r}")
            continue
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            problems.append(f"{item_label}.size_bytes is invalid: {size!r}")
            continue
        seen.add(rel)
        claimed.append({
            "relative_path": rel, "sha256": sha, "size_bytes": size,
        })
    actual: list[dict] = []
    try:
        candidates = sorted(destination.rglob("*"))
    except OSError as exc:
        problems.append(f"{label} cannot enumerate superseded evidence: {exc}")
        return
    for path in candidates:
        rel = path.relative_to(destination).as_posix()
        if path.is_symlink():
            problems.append(f"{label} superseded evidence contains symlink: {rel!r}")
        elif path.is_file() and rel != "inventory.json":
            try:
                actual.append({
                    "relative_path": rel,
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                })
            except OSError as exc:
                problems.append(
                    f"{label} cannot hash superseded evidence {rel!r}: {exc}"
                )
    if sorted(claimed, key=lambda row: row["relative_path"]) != actual:
        problems.append(f"{label} supersession inventory does not match destination")


def _verify_superseded_journal_binding(
    destination: Path, *, entry: dict, row: dict, manifest: dict,
    ledger_entries: list[dict], label: str, problems: list[str],
) -> None:
    """Bind the ledger claim back to the immutable journal it moved."""
    start_path = destination / "journal_start.json"
    try:
        start = json.loads(start_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        problems.append(f"{label} superseded journal start is unreadable: {exc}")
        return
    if not isinstance(start, dict):
        problems.append(f"{label} superseded journal start is not an object")
        return
    start_seal = start.get("journal_start_seal_sha256")
    start_unsigned = {
        key: value for key, value in start.items()
        if key != "journal_start_seal_sha256"
    }
    if start_seal != _canonical_sha256(start_unsigned):
        problems.append(f"{label} superseded journal start seal is stale")
    expected = {
        "schema_version": 2,
        "state": "capture_started",
        "identity": {
            "roster_source": row.get("roster_source"), "slug": row.get("slug"),
        },
        "authority_cell_sha256": row.get("authority_cell_sha256"),
        "source_commit": manifest.get("source_commit_pre"),
        "capture_matrix_sha256": manifest.get("capture_matrix_sha256"),
        "matrix_seal_sha256": manifest.get("matrix_seal_sha256"),
        "capture_preflight_sha256": entry.get("original_preflight_sha256"),
    }
    mismatches = [
        f"{field}={start.get(field)!r} expected={value!r}"
        for field, value in expected.items() if start.get(field) != value
    ]
    if mismatches:
        problems.append(
            f"{label} superseded journal does not bind the ledger: {mismatches}"
        )
    prior_candidates = [
        prior for prior in ledger_entries
        if (
            prior.get("cell_id") == entry.get("cell_id")
            and prior.get("current_preflight_sha256")
            == entry.get("original_preflight_sha256")
        )
    ]
    prior_backref = start.get("supersedes")
    if prior_candidates:
        if len(prior_candidates) != 1 or not isinstance(prior_backref, dict):
            problems.append(
                f"{label} superseded journal has an inconsistent supersession chain"
            )
        else:
            prior = prior_candidates[0]
            expected_backref = {
                "ledger_entry_id": prior.get("ledger_entry_id"),
                "superseded_preflight_sha256": prior.get(
                    "original_preflight_sha256"
                ),
                "superseded_path": prior.get("destination_path"),
            }
            if prior_backref != expected_backref:
                problems.append(
                    f"{label} superseded journal back-reference does not bind "
                    "the prior ledger entry"
                )
    elif prior_backref is not None:
        problems.append(
            f"{label} superseded journal has a dangling prior back-reference"
        )
    complete_path = destination / "journal_complete.json"
    if complete_path.exists():
        try:
            complete = json.loads(complete_path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            problems.append(f"{label} superseded journal completion is unreadable: {exc}")
            return
        if not isinstance(complete, dict):
            problems.append(f"{label} superseded journal completion is not an object")
            return
        complete_seal = complete.get("journal_seal_sha256")
        complete_unsigned = {
            key: value for key, value in complete.items()
            if key != "journal_seal_sha256"
        }
        if complete_seal != _canonical_sha256(complete_unsigned):
            problems.append(f"{label} superseded journal completion seal is stale")
        complete_expected = {**expected, "state": "complete"}
        if any(
            complete.get(field) != value
            for field, value in complete_expected.items()
        ):
            problems.append(
                f"{label} superseded journal completion does not bind its start"
            )


def _verify_supersedes_backref(
    supersedes: dict, *, manifest: dict, row: dict, label: str,
    problems: list[str],
) -> None:
    unknown = sorted(set(supersedes) - _SUPERSEDES_FIELDS)
    missing = sorted(_SUPERSEDES_FIELDS - set(supersedes))
    if unknown or missing:
        problems.append(
            f"{label}.supersedes field set mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
        return
    entry_id = supersedes.get("ledger_entry_id")
    original_preflight = supersedes.get("superseded_preflight_sha256")
    destination_raw = supersedes.get("superseded_path")
    if not isinstance(entry_id, str) or not _SHA64_RE.fullmatch(entry_id):
        problems.append(f"{label}.supersedes.ledger_entry_id is not sha256")
    if (
        not isinstance(original_preflight, str)
        or not _SHA64_RE.fullmatch(original_preflight)
    ):
        problems.append(f"{label}.supersedes superseded preflight is invalid")
    elif original_preflight == manifest.get("preflight_sha256"):
        problems.append(f"{label}.supersedes does not cross a preflight identity")
    if not isinstance(destination_raw, str) or not Path(destination_raw).is_absolute():
        problems.append(f"{label}.supersedes.superseded_path is not absolute")
        return
    destination = Path(destination_raw)
    try:
        resolved_destination = destination.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        problems.append(f"{label} superseded destination is unavailable: {exc}")
        return
    if destination.is_symlink() or not resolved_destination.is_dir():
        problems.append(f"{label} superseded destination is not a regular directory")
        return
    if str(resolved_destination) != destination_raw:
        problems.append(f"{label}.supersedes.superseded_path is not canonical")
    safe_cell_id = f"{row.get('roster_source')}__{row.get('slug')}"
    expected_attempt = re.compile(
        rf".+Z__pf-{re.escape(str(original_preflight)[:8])}"
        rf"__to-pf-{re.escape(str(manifest.get('preflight_sha256'))[:8])}\Z"
    )
    if (
        resolved_destination.name != safe_cell_id
        or resolved_destination.parent.parent.name != "superseded"
        or not expected_attempt.fullmatch(resolved_destination.parent.name)
    ):
        problems.append(
            f"{label} superseded destination does not match the required hierarchy"
        )
    ledger_path = resolved_destination.parent.parent / "supersession_ledger.jsonl"
    try:
        raw_lines = ledger_path.read_text().splitlines()
    except OSError as exc:
        problems.append(f"{label} supersession ledger is unreadable: {exc}")
        return
    if not raw_lines:
        problems.append(f"{label} supersession ledger is empty")
        return
    ledger_entries: list[dict] = []
    ledger_ids: list[str] = []
    for index, raw_line in enumerate(raw_lines):
        line_label = f"{label} supersession ledger line {index + 1}"
        if not raw_line:
            problems.append(f"{line_label} is blank")
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            problems.append(f"{line_label} is invalid JSON: {exc}")
            continue
        if not isinstance(entry, dict) or set(entry) != _SUPERSESSION_LEDGER_FIELDS:
            problems.append(f"{line_label} has a non-closed field set")
            continue
        ledger_entries.append(entry)
        candidate_id = entry.get("ledger_entry_id")
        if not isinstance(candidate_id, str) or not _SHA64_RE.fullmatch(candidate_id):
            problems.append(f"{line_label}.ledger_entry_id is not sha256")
        else:
            ledger_ids.append(candidate_id)
            if candidate_id != _canonical_sha256({
                key: value for key, value in entry.items()
                if key != "ledger_entry_id"
            }):
                problems.append(f"{line_label} entry seal is stale")
        if entry.get("schema_version") != 1:
            problems.append(f"{line_label}.schema_version is not 1")
        _parse_utc_timestamp(
            entry.get("timestamp_utc"), f"{line_label}.timestamp_utc", problems,
        )
        for field in (
            "original_preflight_sha256", "current_preflight_sha256",
            "matrix_seal_sha256", "inventory_sha256",
        ):
            value = entry.get(field)
            if not isinstance(value, str) or not _SHA64_RE.fullmatch(value):
                problems.append(f"{line_label}.{field} is not sha256")
        if (
            entry.get("chunk_id") != manifest.get("chunk_id")
            or entry.get("source_commit") != manifest.get("source_commit_pre")
            or entry.get("matrix_seal_sha256")
            != manifest.get("matrix_seal_sha256")
            or entry.get("reason_code") != _SUPERSESSION_REASON
        ):
            problems.append(f"{line_label} does not bind the DONE campaign context")
    duplicates = sorted(key for key, count in Counter(ledger_ids).items() if count > 1)
    if duplicates:
        problems.append(f"{label} supersession ledger has duplicate ids: {duplicates}")
    matches = [entry for entry in ledger_entries if entry.get("ledger_entry_id") == entry_id]
    if len(matches) != 1:
        problems.append(
            f"{label} supersedes back-reference resolves to {len(matches)} ledger entries"
        )
        return
    entry = matches[0]
    derived_entry_id = _canonical_sha256({
        key: value for key, value in entry.items() if key != "ledger_entry_id"
    })
    if entry.get("ledger_entry_id") != derived_entry_id:
        problems.append(f"{label} supersession ledger entry seal is stale")
    current_candidates = [
        candidate for candidate in ledger_entries
        if (
            candidate.get("cell_id") == safe_cell_id
            and candidate.get("current_preflight_sha256")
            == manifest.get("preflight_sha256")
        )
    ]
    if len(current_candidates) != 1 or current_candidates[0] is not entry:
        problems.append(
            f"{label} supersession ledger has non-unique current-attempt authority"
        )
    expected = {
        "schema_version": 1,
        "chunk_id": manifest.get("chunk_id"),
        "cell_id": safe_cell_id,
        "original_preflight_sha256": original_preflight,
        "current_preflight_sha256": manifest.get("preflight_sha256"),
        "source_path": str(
            resolved_destination.parent.parent.parent / "journals" / safe_cell_id
        ),
        "destination_path": str(resolved_destination),
        "reason_code": _SUPERSESSION_REASON,
        "source_commit": manifest.get("source_commit_pre"),
        "matrix_seal_sha256": manifest.get("matrix_seal_sha256"),
        "ledger_entry_id": entry_id,
    }
    mismatches = [
        f"{field}={entry.get(field)!r} expected={value!r}"
        for field, value in expected.items() if entry.get(field) != value
    ]
    for field in ("session_id", "operator_flag"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            mismatches.append(f"{field} must be a nonempty string")
    if (
        isinstance(entry.get("operator_flag"), str)
        and "--supersede-stale-journals" not in entry["operator_flag"].split()
    ):
        mismatches.append("operator_flag does not record --supersede-stale-journals")
    source_path = entry.get("source_path")
    if not isinstance(source_path, str) or not Path(source_path).is_absolute():
        mismatches.append("source_path must be absolute")
    inventory_sha = entry.get("inventory_sha256")
    if not isinstance(inventory_sha, str) or not _SHA64_RE.fullmatch(inventory_sha):
        mismatches.append("inventory_sha256 must be sha256")
    if mismatches:
        problems.append(f"{label} supersession ledger is inconsistent: {mismatches}")
    _verify_superseded_inventory(
        resolved_destination, expected_sha256=inventory_sha,
        label=label, problems=problems,
    )
    _verify_superseded_journal_binding(
        resolved_destination, entry=entry, row=row, manifest=manifest,
        ledger_entries=ledger_entries, label=label, problems=problems,
    )


def _verify_cell_capture_provenance(
    row: dict, *, manifest: dict, output_root: Path,
    admission_runs: dict[str, datetime], label: str,
) -> list[str]:
    """Verify one sealed cell result against its chunk admission identity."""
    problems: list[str] = []
    cell_artifacts = _artifacts_by_kind(row).get("cell_result", [])
    if len(cell_artifacts) != 1:
        return [
            f"{label} requires exactly one cell_result artifact for provenance; "
            f"found {len(cell_artifacts)}"
        ]
    result_path = _resolve_sealed_artifact(
        cell_artifacts[0], output_root=output_root,
        label=f"{label} cell result", problems=problems,
    )
    if result_path is None:
        return problems
    try:
        cell = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return [*problems, f"{label} cell result is unreadable: {exc}"]
    identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
    expected = {
        "identity": identity,
        "authority_cell_sha256": row.get("authority_cell_sha256"),
        "disposition": row.get("disposition"),
        "typed_reason": row.get("typed_reason"),
        "capture_preflight_sha256": row.get("capture_preflight_sha256"),
        "capture_started_at_utc": row.get("capture_started_at_utc"),
        "admission_run_id": row.get("admission_run_id"),
        "supersedes": row.get("supersedes"),
        "prior_preparation_failures": row.get("prior_preparation_failures"),
        "prior_capture_defect_attempts": row.get(
            "prior_capture_defect_attempts"
        ),
        "definition_equivalence": row.get("definition_equivalence"),
        "definition_equivalence_attestation_sha256": row.get(
            "definition_equivalence_attestation_sha256"
        ),
    }
    if not isinstance(cell, dict):
        return [*problems, f"{label} cell result is not an object"]
    mismatches = [
        f"{field}={cell.get(field)!r} expected={value!r}"
        for field, value in expected.items() if cell.get(field) != value
    ]
    required = {
        "capture_preflight_sha256", "capture_started_at_utc",
        "admission_run_id", "supersedes", "prior_preparation_failures",
        "prior_capture_defect_attempts",
        "definition_equivalence", "definition_equivalence_attestation_sha256",
    }
    missing = sorted(required - set(cell))
    if missing:
        mismatches.append(f"missing required fields {missing}")
    if mismatches:
        problems.append(f"{label} cell result provenance mismatch: {mismatches}")
    run_id = cell.get("admission_run_id")
    started = _parse_utc_timestamp(
        cell.get("capture_started_at_utc"),
        f"{label} cell result capture_started_at_utc", problems,
    )
    admitted = admission_runs.get(run_id) if isinstance(run_id, str) else None
    if admitted is None:
        problems.append(
            f"{label} admission_run_id={run_id!r} is absent from DONE admission"
        )
    elif started is not None and started < admitted:
        problems.append(
            f"{label} capture started before its admission: "
            f"started={cell.get('capture_started_at_utc')!r}, "
            f"admitted={admitted.isoformat()!r}"
        )
    supersedes = cell.get("supersedes")
    if supersedes is not None:
        if not isinstance(supersedes, dict):
            problems.append(f"{label}.supersedes is not null or an object")
    return problems


def _manifest_supersession_roots(manifest: dict, rows: list[dict]) -> set[Path]:
    roots: set[Path] = set()
    for row in rows:
        supersedes = row.get("supersedes") if isinstance(row, dict) else None
        raw = supersedes.get("superseded_path") if isinstance(supersedes, dict) else None
        if isinstance(raw, str) and Path(raw).is_absolute():
            # Keep discovery lexical.  Resolution is an adversarial operation
            # (including symlink loops) and the verifier below converts it to
            # a typed problem instead of allowing an exception to escape.
            roots.add(Path(raw).parent.parent)
    output_raw = manifest.get("output_root")
    chunk_id = manifest.get("chunk_id")
    matrix_sha = manifest.get("capture_matrix_sha256")
    source_commit = manifest.get("source_commit_pre")
    if (
        isinstance(output_raw, str)
        and Path(output_raw).is_absolute()
        and Path(output_raw).name == chunk_id
        and Path(output_raw).parent.name == "chunks"
        and isinstance(matrix_sha, str)
        and isinstance(source_commit, str)
    ):
        campaign_root = Path(output_raw).parent.parent
        derived = (
            campaign_root / ".staging" / source_commit / str(chunk_id) /
            matrix_sha / "superseded"
        )
        if derived.exists() or derived.is_symlink():
            roots.add(derived)
    return roots


def _verify_supersession_root(
    root: Path, *, manifest: dict, rows: list[dict], label: str,
) -> list[str]:
    """Close the ledger in both directions against every preserved attempt."""
    problems: list[str] = []
    if root.is_symlink() or not root.is_dir():
        return [f"{label} superseded root is not a regular directory: {root}"]
    actual_destinations: set[str] = set()
    try:
        root_entries = sorted(root.iterdir())
    except OSError as exc:
        return [f"{label} superseded root is unreadable: {exc}"]
    for group in root_entries:
        if group.name == "supersession_ledger.jsonl":
            if group.is_symlink() or not group.is_file():
                problems.append(f"{label} supersession ledger is not a regular file")
            continue
        if group.is_symlink() or not group.is_dir():
            problems.append(f"{label} superseded root has unexpected entry {group}")
            continue
        try:
            cell_dirs = list(group.iterdir())
        except OSError as exc:
            problems.append(f"{label} cannot enumerate attempt group {group}: {exc}")
            continue
        if not cell_dirs:
            problems.append(f"{label} superseded attempt group is empty: {group}")
        for destination in cell_dirs:
            if destination.is_symlink() or not destination.is_dir():
                problems.append(
                    f"{label} superseded attempt has unexpected entry {destination}"
                )
                continue
            try:
                actual_destinations.add(str(destination.resolve(strict=True)))
            except (OSError, RuntimeError) as exc:
                problems.append(
                    f"{label} cannot resolve superseded destination "
                    f"{destination}: {exc}"
                )

    ledger_path = root / "supersession_ledger.jsonl"
    try:
        raw_lines = ledger_path.read_text().splitlines()
    except OSError as exc:
        if actual_destinations:
            problems.append(f"{label} superseded evidence has no ledger: {exc}")
        return problems
    ledger: list[dict] = []
    for index, raw in enumerate(raw_lines):
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            problems.append(f"{label} ledger line {index + 1} is invalid: {exc}")
            continue
        if isinstance(entry, dict):
            ledger.append(entry)
        else:
            problems.append(f"{label} ledger line {index + 1} is not an object")
    ledger_destinations = {
        entry.get("destination_path") for entry in ledger
        if isinstance(entry.get("destination_path"), str)
    }
    if ledger_destinations != actual_destinations:
        problems.append(
            f"{label} ledger/destination set mismatch: "
            f"unledgered={sorted(actual_destinations - ledger_destinations)!r}, "
            f"missing={sorted(ledger_destinations - actual_destinations)!r}"
        )
    session_ids = {
        entry.get("session_id") for entry in ledger
        if isinstance(entry.get("session_id"), str)
    }
    if len(session_ids) > 1:
        problems.append(f"{label} supersession ledger mixes session ids")

    by_cell = {
        f"{row.get('roster_source')}__{row.get('slug')}": row
        for row in rows if isinstance(row, dict)
    }
    current_expected: dict[str, dict] = {}
    for index, entry in enumerate(ledger):
        cell_id = entry.get("cell_id")
        row = by_cell.get(cell_id)
        if row is None:
            problems.append(
                f"{label} ledger line {index + 1} names a non-member cell {cell_id!r}"
            )
            continue
        backref = {
            "ledger_entry_id": entry.get("ledger_entry_id"),
            "superseded_preflight_sha256": entry.get(
                "original_preflight_sha256"
            ),
            "superseded_path": entry.get("destination_path"),
        }
        candidate_manifest = {
            **manifest,
            "preflight_sha256": entry.get("current_preflight_sha256"),
        }
        entry_problems: list[str] = []
        _verify_supersedes_backref(
            backref, manifest=candidate_manifest, row=row,
            label=f"{label} ledger line {index + 1}", problems=entry_problems,
        )
        problems.extend(entry_problems)
        if entry.get("current_preflight_sha256") == manifest.get("preflight_sha256"):
            if cell_id in current_expected:
                problems.append(
                    f"{label} has duplicate current ledger authority for {cell_id}"
                )
            current_expected[cell_id] = backref

    for cell_id, row in by_cell.items():
        observed = row.get("supersedes")
        expected = current_expected.get(cell_id)
        if observed != expected and (observed is not None or expected is not None):
            problems.append(
                f"{label} cell {cell_id} supersedes back-reference does not "
                "exactly match current ledger authority"
            )
        cell_entries = {
            entry.get("ledger_entry_id"): entry for entry in ledger
            if entry.get("cell_id") == cell_id
            and isinstance(entry.get("ledger_entry_id"), str)
        }
        reachable: set[str] = set()
        cursor = observed
        while isinstance(cursor, dict):
            cursor_id = cursor.get("ledger_entry_id")
            if not isinstance(cursor_id, str) or cursor_id in reachable:
                if cursor_id in reachable:
                    problems.append(f"{label} cell {cell_id} supersession chain cycles")
                break
            entry = cell_entries.get(cursor_id)
            if entry is None:
                break
            reachable.add(cursor_id)
            try:
                start = json.loads(
                    (Path(entry["destination_path"]) / "journal_start.json").read_text()
                )
            except (
                KeyError, OSError, RuntimeError, json.JSONDecodeError, UnicodeError,
            ):
                break
            cursor = start.get("supersedes") if isinstance(start, dict) else None
        if reachable != set(cell_entries):
            problems.append(
                f"{label} cell {cell_id} ledger contains an orphaned attempt chain"
            )
    return problems


def _manifest_preparation_ledger_paths(
    manifest: dict, rows: list[dict],
) -> set[Path]:
    """Discover the one campaign-global ledger without trusting only backrefs."""
    paths: set[Path] = set()
    for row in rows:
        refs = row.get("prior_preparation_failures") if isinstance(row, dict) else None
        for ref in refs if isinstance(refs, list) else []:
            raw = ref.get("ledger_path") if isinstance(ref, dict) else None
            if isinstance(raw, str) and Path(raw).is_absolute():
                paths.add(Path(raw))
    output_raw = manifest.get("output_root")
    chunk_id = manifest.get("chunk_id")
    if (
        isinstance(output_raw, str)
        and Path(output_raw).is_absolute()
        and Path(output_raw).name == chunk_id
        and Path(output_raw).parent.name == "chunks"
    ):
        candidate = (
            Path(output_raw).parent.parent / PREPARATION_FAILURE_ROOT_NAME
            / PREPARATION_FAILURE_LEDGER_FILENAME
        )
        # The campaign-global preparation ledger is authoritative even when no
        # active DONE row carries a back-reference.  Always admit its canonical
        # path so strict archive verification can recognize an unledgered
        # post-move directory.  A missing or genuinely empty preparation root
        # remains valid because the shared loader returns an empty row set.
        paths.add(candidate)
    return paths


def verify_manifest_preparation_failure_closure(
    manifest: dict, *,
    result_documents: list[tuple[dict, dict]] | None = None,
) -> list[str]:
    """Close preparation archives, ledger rows, DONE rows, and cell results."""
    historical_bootstrap = _is_historical_bootstrap(manifest)
    rows_raw = manifest.get("cell_results")
    rows = [row for row in rows_raw if isinstance(row, dict)] \
        if isinstance(rows_raw, list) else []
    problems: list[str] = []

    documents_by_identity: dict[tuple[Any, Any], dict] = {}
    if not historical_bootstrap:
        for row, document in result_documents or []:
            if not isinstance(row, dict) or not isinstance(document, dict):
                problems.append(
                    "PREPARATION_LEDGER_INCONSISTENT: cell result join is malformed"
                )
                continue
            key = (row.get("roster_source"), row.get("slug"))
            if key in documents_by_identity:
                problems.append(
                    "PREPARATION_LEDGER_INCONSISTENT: duplicate cell-result join "
                    f"for {key!r}"
                )
            documents_by_identity[key] = document
            if document.get("prior_preparation_failures") != row.get(
                "prior_preparation_failures"
            ):
                problems.append(
                    "PREPARATION_LEDGER_INCONSISTENT: active cell result and DONE "
                    f"back-references differ for {key!r}"
                )

    ledger_paths = _manifest_preparation_ledger_paths(manifest, rows)
    if len(ledger_paths) > 1:
        problems.append(
            "PREPARATION_LEDGER_INCONSISTENT: one DONE resolves multiple "
            f"preparation ledgers: {sorted(map(str, ledger_paths))!r}"
        )
    try:
        import archive_preparation_failure as archive  # noqa: E402
    except ImportError as exc:
        return [
            *problems,
            "PREPARATION_LEDGER_INCONSISTENT: preparation archive verifier "
            f"is unavailable: {exc}",
        ]

    entries_by_ledger: dict[Path, list[dict]] = {}
    for ledger_path in sorted(ledger_paths, key=str):
        expected_path = (
            ledger_path.parent / archive.PREPARATION_FAILURE_LEDGER_NAME
        )
        if (
            not ledger_path.is_absolute()
            or ledger_path != expected_path
            or ledger_path.parent.name != archive.PREPARATION_FAILURE_ROOT_NAME
        ):
            problems.append(
                "PREPARATION_LEDGER_INCONSISTENT: noncanonical ledger path "
                f"{ledger_path}"
            )
            continue
        try:
            entries = archive.load_preparation_failure_ledger(
                ledger_path, verify_archives=True,
            )
        except archive.PreparationArchiveRefusal as exc:
            problems.append(f"{exc.code}: {exc.detail}")
            continue
        if ledger_path.exists() and not entries:
            problems.append(
                "PREPARATION_LEDGER_INCONSISTENT: preparation ledger is empty"
            )
        entries_by_ledger[ledger_path.resolve(strict=False)] = entries

    # The historical bootstrap is exempt only from the Round-28 per-cell
    # provenance/back-reference shape.  Its certifying verifier still closes
    # the campaign-global preparation root so an FS-PRE/FS-POST archive state
    # cannot be hidden behind the compatibility allowlist.
    if historical_bootstrap:
        return problems

    for index, row in enumerate(rows):
        label = f"cell_results[{index}]"
        observed_problems: list[str] = []
        observed = _normalize_prior_preparation_failures(
            row.get("prior_preparation_failures"),
            where=f"{label}.prior_preparation_failures",
            problems=observed_problems,
        )
        problems.extend(
            f"PREPARATION_LEDGER_INCONSISTENT: {problem}"
            for problem in observed_problems
        )
        cell_id = f"{row.get('roster_source')}__{row.get('slug')}"
        expected: list[dict] = []
        for ledger_path, entries in sorted(
            entries_by_ledger.items(), key=lambda item: str(item[0]),
        ):
            expected.extend(archive.preparation_failure_backrefs(
                entries, cell_id=cell_id, ledger_path=ledger_path,
            ))
        if observed != expected:
            problems.append(
                "PREPARATION_LEDGER_INCONSISTENT: active back-reference set "
                f"differs from ledger authority for {cell_id}: "
                f"observed={observed!r}, expected={expected!r}"
            )
    return problems


def _manifest_capture_defect_ledger_paths(
    manifest: dict, rows: list[dict],
) -> set[Path]:
    paths: set[Path] = set()
    for row in rows:
        refs = row.get("prior_capture_defect_attempts") \
            if isinstance(row, dict) else None
        for ref in refs if isinstance(refs, list) else []:
            raw = ref.get("ledger_path") if isinstance(ref, dict) else None
            if isinstance(raw, str) and Path(raw).is_absolute():
                paths.add(Path(raw))
    output_raw = manifest.get("output_root")
    chunk_id = manifest.get("chunk_id")
    if (
        isinstance(output_raw, str) and Path(output_raw).is_absolute()
        and Path(output_raw).name == chunk_id
        and Path(output_raw).parent.name == "chunks"
    ):
        paths.add(
            Path(output_raw).parent.parent / CAPTURE_DEFECT_ROOT_NAME
            / CAPTURE_DEFECT_LEDGER_FILENAME
        )
    return paths


def verify_manifest_capture_defect_closure(
    manifest: dict, *,
    result_documents: list[tuple[dict, dict]] | None = None,
) -> list[str]:
    """Close defect archives and prove they contribute no published row."""
    historical_bootstrap = _is_historical_bootstrap(manifest)
    raw_rows = manifest.get("cell_results")
    rows = [row for row in raw_rows if isinstance(row, dict)] \
        if isinstance(raw_rows, list) else []
    problems: list[str] = []
    if not historical_bootstrap:
        for row, document in result_documents or []:
            if not isinstance(row, dict) or not isinstance(document, dict):
                problems.append(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT: cell-result join malformed"
                )
                continue
            if document.get("prior_capture_defect_attempts") != row.get(
                "prior_capture_defect_attempts"
            ):
                problems.append(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT: active cell result and "
                    "DONE capture-defect back-references differ"
                )
    paths = _manifest_capture_defect_ledger_paths(manifest, rows)
    if len(paths) > 1:
        problems.append(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT: one DONE resolves multiple "
            f"capture-defect ledgers: {sorted(map(str, paths))!r}"
        )
    try:
        import archive_capture_defect_attempt as archive  # noqa: E402
    except ImportError as exc:
        return [
            *problems,
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT: archive verifier unavailable: "
            f"{exc}",
        ]
    entries_by_ledger: dict[Path, list[dict]] = {}
    for ledger_path in sorted(paths, key=str):
        if (
            not ledger_path.is_absolute()
            or ledger_path.name != archive.CAPTURE_DEFECT_LEDGER_NAME
            or ledger_path.parent.name != archive.CAPTURE_DEFECT_ROOT_NAME
        ):
            problems.append(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT: noncanonical ledger path "
                f"{ledger_path}"
            )
            continue
        try:
            entries = archive.load_capture_defect_ledger(
                ledger_path, verify_archives=True,
            )
        except archive.CaptureDefectRefusal as exc:
            problems.append(f"{exc.code}: {exc.detail}")
            continue
        if any(
            row.get("campaign_rows") != 0
            or row.get("authoritative") is not False
            or row.get("reusable") is not False
            for row in entries
        ):
            problems.append(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT: archived attempt is "
                "authoritative, reusable, or contributes campaign rows"
            )
        entries_by_ledger[ledger_path.resolve(strict=False)] = entries
    if historical_bootstrap:
        return problems
    for index, row in enumerate(rows):
        label = f"cell_results[{index}]"
        observed_problems: list[str] = []
        observed = _normalize_prior_capture_defect_attempts(
            row.get("prior_capture_defect_attempts"),
            where=f"{label}.prior_capture_defect_attempts",
            problems=observed_problems,
        )
        problems.extend(
            f"CAPTURE_DEFECT_LEDGER_INCONSISTENT: {problem}"
            for problem in observed_problems
        )
        cell_id = f"{row.get('roster_source')}__{row.get('slug')}"
        expected: list[dict] = []
        for ledger_path, entries in sorted(
            entries_by_ledger.items(), key=lambda item: str(item[0]),
        ):
            expected.extend(archive.capture_defect_backrefs(
                entries, cell_id=cell_id, ledger_path=ledger_path,
            ))
        if observed != expected:
            problems.append(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT: active back-reference set "
                f"differs from ledger authority for {cell_id}: "
                f"observed={observed!r}, expected={expected!r}"
            )
    return problems


def _route_matrix_row(
    matrix: Any, identity: dict, *, label: str, problems: list[str],
) -> dict | None:
    cells = matrix.get("cells") if isinstance(matrix, dict) else None
    if not isinstance(cells, list):
        problems.append(f"{label} cannot bind route: capture matrix cells is not an array")
        return None
    matches = [
        cell for cell in cells
        if isinstance(cell, dict) and cell.get("identity") == identity
    ]
    if len(matches) != 1:
        problems.append(
            f"{label} route must bind exactly one capture-matrix row; found {len(matches)}"
        )
        return None
    return matches[0]


def _verify_trace_manifest_launch_contract(
    row: dict, *, matrix: dict | None, output_root: Path, label: str,
) -> list[str]:
    """Bind published label/trace launch selection back to the matrix row."""
    if row.get("disposition") != "admitted" or not isinstance(matrix, dict):
        return []
    problems: list[str] = []
    identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
    matrix_row = _route_matrix_row(
        matrix, identity, label=label, problems=problems,
    )
    if matrix_row is None:
        return problems
    invocation = matrix_row.get("invocation")
    launch_contract = (
        invocation.get("capture_launch_contract")
        if isinstance(invocation, dict) else None
    )
    if not isinstance(launch_contract, dict):
        # Historical/synthetic v1 matrices predate this additive execution
        # binding.  Their existing seals remain verifiable; every matrix
        # produced by the current builder carries the contract and enters the
        # strict branch below.
        return problems
    by_kind = _artifacts_by_kind(row)
    trace_rows = by_kind.get("trace_manifest", [])
    label_rows = by_kind.get("label_sidecar", [])
    observation_rows = by_kind.get("ncu_launch_observation", [])
    if (
        len(trace_rows) != 1
        or len(label_rows) != 1
        or len(observation_rows) != 1
    ):
        return [
            *problems,
            f"{label} requires one trace manifest, one label sidecar, and one "
            "NCU launch observation for launch-contract verification",
        ]
    trace_path = _resolve_sealed_artifact(
        trace_rows[0], output_root=output_root,
        label=f"{label} trace manifest", problems=problems,
    )
    label_path = _resolve_sealed_artifact(
        label_rows[0], output_root=output_root,
        label=f"{label} label sidecar", problems=problems,
    )
    observation_path = _resolve_sealed_artifact(
        observation_rows[0], output_root=output_root,
        label=f"{label} NCU launch observation", problems=problems,
    )
    if trace_path is None or label_path is None or observation_path is None:
        return problems
    trace = _load_strict_json(
        trace_path, f"{label}.trace_manifest", problems,
    )
    sidecar = _load_strict_json(
        label_path, f"{label}.label_sidecar", problems,
    )
    observation = _load_strict_json(
        observation_path, f"{label}.ncu_launch_observation", problems,
    )
    if trace is None or sidecar is None or observation is None:
        return problems
    expected_count = launch_contract.get("trace_launch_count")
    observed_launches = trace.get("launches") if isinstance(trace, dict) else None
    if trace.get("capture_launch_contract") != launch_contract:
        problems.append(f"{label} trace launch contract does not match the matrix")
    if (
        trace.get("launch_count") != expected_count
        or not isinstance(observed_launches, list)
        or len(observed_launches) != expected_count
    ):
        problems.append(f"{label} trace launch count does not match the matrix")
    expected_sequence = launch_contract.get("trace_launch_sequence")
    if isinstance(expected_sequence, list) and isinstance(observed_launches, list):
        fields = (
            "device_id", "stream_id", "kernel_id", "kernel_name",
            "grid_dim", "block_dim",
        )
        expected = [{key: launch.get(key) for key in fields} for launch in expected_sequence]
        observed = [{key: launch.get(key) for key in fields} for launch in observed_launches]
        if observed != expected:
            problems.append(f"{label} trace launch sequence does not match the matrix")
    if sidecar.get("launch_index") != launch_contract.get("label_launch_index"):
        problems.append(f"{label} label launch index does not match the matrix")
    trace_launch_index = launch_contract.get("label_trace_launch_index")
    if (
        type(trace_launch_index) is not int
        or trace_launch_index < 0
        or not isinstance(observed_launches, list)
        or trace_launch_index >= len(observed_launches)
    ):
        problems.append(f"{label} selected trace launch is invalid")
    else:
        expected_launch_key = _trace_launch_key(
            observed_launches[trace_launch_index]
        )
        observed_launch_key = _ncu_observation_launch_key(observation)
        if expected_launch_key is None or observed_launch_key != expected_launch_key:
            problems.append(
                f"{label} NCU launch observation differs from the trace launch"
            )
    return problems


def _route_retained_inventory(
    trace_root: Path, retained: Any, *, label: str, problems: list[str],
) -> list[dict]:
    if not isinstance(retained, dict) or set(retained) != _ROUTE_RETAINED_FIELDS:
        problems.append(f"{label} route retained block is malformed")
        return []
    claimed = retained.get("artifacts")
    if not isinstance(claimed, list) or not claimed:
        problems.append(f"{label} route retained inventory is empty")
        return []
    if trace_root.is_symlink() or not trace_root.is_dir():
        problems.append(f"{label} retained trace directory is absent")
        return []
    actual: list[dict] = []
    for path in sorted(trace_root.rglob("*")):
        if path.is_symlink():
            problems.append(f"{label} retained trace contains symlink: {path}")
            continue
        if path.is_file():
            actual.append({
                "relative_path": path.relative_to(trace_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
    claimed_basic: list[dict] = []
    claimed_paths: set[str] = set()
    for index, artifact in enumerate(claimed):
        if not isinstance(artifact, dict) or set(artifact) != {
            "relative_path", "size_bytes", "sha256", "artifact_kind",
        }:
            problems.append(f"{label} route retained artifact[{index}] is malformed")
            continue
        rel = artifact.get("relative_path")
        if not isinstance(rel, str) or rel in claimed_paths:
            problems.append(f"{label} route retained artifact path is invalid: {rel!r}")
            continue
        claimed_paths.add(rel)
        claimed_basic.append({
            "relative_path": rel,
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
        })
    if claimed_basic != actual:
        problems.append(f"{label} retained route inventory does not match disk")
    if (
        retained.get("file_count") != len(actual)
        or retained.get("bytes") != sum(item["size_bytes"] for item in actual)
        or retained.get("tree_sha256") != _canonical_sha256(actual)
    ):
        problems.append(f"{label} retained route aggregate seal is stale")
    return actual


def _verify_trace_route_evidence(
    row: dict, *, manifest: dict, matrix: dict | None,
    preflight: dict | None, output_root: Path, label: str,
) -> list[str]:
    """Re-derive a routed trace instead of trusting its sealed result string."""
    problems: list[str] = []
    by_kind = _artifacts_by_kind(row)
    routes = by_kind.get("trace_route_record", [])
    if not routes:
        return problems
    results = by_kind.get("cell_result", [])
    trace_manifests = by_kind.get("trace_manifest", [])
    if len(routes) != 1 or len(results) != 1 or len(trace_manifests) != 1:
        return [
            f"{label} routed trace requires exactly one route record, cell result, "
            f"and trace manifest (route={len(routes)}, result={len(results)}, "
            f"trace_manifest={len(trace_manifests)})"
        ]
    route_path = _resolve_sealed_artifact(
        routes[0], output_root=output_root,
        label=f"{label} route record", problems=problems,
    )
    result_path = _resolve_sealed_artifact(
        results[0], output_root=output_root,
        label=f"{label} cell result", problems=problems,
    )
    trace_manifest_path = _resolve_sealed_artifact(
        trace_manifests[0], output_root=output_root,
        label=f"{label} trace manifest", problems=problems,
    )
    if None in (route_path, result_path, trace_manifest_path):
        return problems
    try:
        route = json.loads(route_path.read_text())
        cell_result = json.loads(result_path.read_text())
        trace_manifest = json.loads(trace_manifest_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return [*problems, f"{label} routed evidence is unreadable: {exc}"]

    if not isinstance(route, dict) or set(route) != _ROUTE_RESULT_FIELDS:
        problems.append(f"{label} route record has a non-closed field set")
        return problems
    unsigned = {key: value for key, value in route.items() if key != "result_sha256"}
    if route.get("result_sha256") != _canonical_sha256(unsigned):
        problems.append(f"{label} route result seal is stale")
    identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
    if (
        route.get("schema_version") != 1
        or route.get("kind") != "ac9_trace_budget_route"
        or route.get("identity") != identity
        or route.get("cell_id") != row.get("slug")
    ):
        problems.append(f"{label} route identity/schema does not match the DONE row")

    matrix_row = _route_matrix_row(
        matrix, identity, label=label, problems=problems,
    )
    invocation = matrix_row.get("invocation") if isinstance(matrix_row, dict) else None
    trace_budget = matrix_row.get("trace_budget") if isinstance(matrix_row, dict) else None
    matrix_pins = matrix.get("input_pins") if isinstance(matrix, dict) else None
    budget_pin = (
        matrix_pins.get("per_cell_trace_budgets")
        if isinstance(matrix_pins, dict) else None
    )
    injection = (
        (preflight.get("checks") or {}).get("injection_runtime")
        if isinstance(preflight, dict) else None
    )
    protocol = route.get("protocol")
    if not isinstance(protocol, dict) or set(protocol) != _ROUTE_PROTOCOL_FIELDS:
        problems.append(f"{label} route protocol has a non-closed field set")
    if (
        not isinstance(invocation, dict)
        or not isinstance(trace_budget, dict)
        or trace_budget.get("status") != "declared"
        or not isinstance(budget_pin, dict)
        or trace_budget.get("source_sha256") != budget_pin.get("sha256")
        or not isinstance(injection, dict)
    ):
        problems.append(f"{label} cannot rederive route inputs from matrix/preflight")
        expected_protocol = None
    else:
        args_contract = {
            "adapter_kind": invocation.get("adapter_kind"),
            "trace_app_args": invocation.get("trace_app_args"),
            "logical_workload": invocation.get("logical_workload"),
            "trace_cardinality_contract": invocation.get(
                "trace_cardinality_contract"
            ),
        }
        for additive in ("capture_launch_contract", "matrix_capture_override"):
            if additive in invocation:
                args_contract[additive] = invocation[additive]
        args_hash = _canonical_sha256(args_contract)
        expected_protocol = {
            "protocol_version": manifest.get("protocol_version"),
            "source_commit": manifest.get("source_commit_pre"),
            "authority_cell_sha256": row.get("authority_cell_sha256"),
            "capture_matrix_sha256": manifest.get("capture_matrix_sha256"),
            "preflight_sha256": manifest.get("preflight_sha256"),
            "pins_sha256": manifest.get("pins_sha256"),
            "trace_budget_sha256": trace_budget.get("source_sha256"),
            "capture_manifest_sha256": trace_manifests[0].get("sha256"),
            "binary_sha256": invocation.get("executable_sha256"),
            "tracer_so_sha256": injection.get("tracer_so_sha256"),
            "args_hash": args_hash,
        }
        if protocol != expected_protocol:
            problems.append(f"{label} route protocol/input hashes do not rederive")
    if route.get("protocol_sha256") != _canonical_sha256(protocol):
        problems.append(f"{label} route protocol_sha256 is stale")

    expected_budget = None
    if isinstance(trace_budget, dict):
        expected_budget = {
            field: trace_budget.get(field) for field in _ROUTE_BUDGET_FIELDS
        }
    budget = route.get("budget")
    if (
        not isinstance(budget, dict)
        or set(budget) != _ROUTE_BUDGET_FIELDS
        or budget != expected_budget
        or type(budget.get("max_trace_bytes")) is not int
        or budget.get("max_trace_bytes", 0) <= 0
        or type(budget.get("max_wall_time_s")) is not int
        or budget.get("max_wall_time_s", 0) <= 0
        or budget.get("sampling_policy") != "off"
    ):
        problems.append(f"{label} route budget does not match the matrix cell")
    config = route.get("route_config")
    if (
        not isinstance(config, dict)
        or set(config) != _ROUTE_CONFIG_FIELDS
        or type(config.get("selection_max_raw_bytes")) is not int
        or config.get("selection_max_raw_bytes", 0) <= 0
        or type(config.get("staging_limit_bytes")) is not int
        or config.get("staging_limit_bytes", 0) <= 0
        or type(config.get("seed")) is not int
        or type(config.get("sample_size")) is not int
        or config.get("sample_size", 0) <= 0
        or type(config.get("max_reconcile_attempts")) is not int
        or config.get("max_reconcile_attempts", 0) <= 0
        or config.get("gate_forbidden_policy") not in {
            "typed_unavailable", "reduced_shape"
        }
        or (
            isinstance(budget, dict)
            and type(budget.get("max_trace_bytes")) is int
            and config.get("selection_max_raw_bytes", 0)
            > budget["max_trace_bytes"]
        )
        or (
            config.get("gate_forbidden_policy") == "typed_unavailable"
            and config.get("reduced_shape_variant") is not None
        )
        or (
            config.get("gate_forbidden_policy") == "reduced_shape"
            and config.get("reduced_shape_variant") not in {
                "grid_stride_footprint_preserving", "dataset_shrunk"
            }
        )
    ):
        problems.append(f"{label} route configuration is malformed")

    full = route.get("full_trace")
    if not isinstance(full, dict) or set(full) != _ROUTE_FULL_TRACE_FIELDS:
        problems.append(f"{label} full-trace seal is malformed")
    else:
        _validate_hash(f"{label}.full_trace.tree_sha256", full.get("tree_sha256"), problems)
        positive_fields = (
            "tree_bytes", "threadblock_count", "decoded_record_count",
            "expected_records", "admitted_records",
        )
        if any(type(full.get(field)) is not int or full[field] <= 0 for field in positive_fields):
            problems.append(f"{label} full-trace counts are non-positive/malformed")
        if full.get("admitted_records") != full.get("expected_records"):
            problems.append(f"{label} full-trace seal is not zero-drop")
        if (
            isinstance(budget, dict)
            and type(budget.get("max_trace_bytes")) is int
            and type(full.get("tree_bytes")) is int
            and full["tree_bytes"] <= budget["max_trace_bytes"]
        ):
            problems.append(f"{label} route was not required by the full-trace size")
        capture_bindings = {
            "identity": identity,
            "binary_sha256": protocol.get("binary_sha256") if isinstance(protocol, dict) else None,
            "tracer_so_sha256": protocol.get("tracer_so_sha256") if isinstance(protocol, dict) else None,
            "max_trace_bytes": budget.get("max_trace_bytes") if isinstance(budget, dict) else None,
            "trace_tree_bytes": full.get("tree_bytes"),
            "threadblocks_pb_count": full.get("threadblock_count"),
            "decoded_record_count": full.get("decoded_record_count"),
            "expected_records": full.get("expected_records"),
            "admitted_records": full.get("admitted_records"),
            "drop_rate": 0.0,
            "launch_count": (
                (invocation.get("capture_launch_contract") or {}).get(
                    "trace_launch_count", 1
                )
            ),
        }
        if not isinstance(trace_manifest, dict) or any(
            trace_manifest.get(field) != expected
            for field, expected in capture_bindings.items()
        ):
            problems.append(f"{label} trace manifest does not bind the full route input")

    trace_root = route_path.parent / "trace"
    actual_inventory = _route_retained_inventory(
        trace_root, route.get("retained"), label=label, problems=problems,
    )
    retained = route.get("retained") if isinstance(route.get("retained"), dict) else {}
    if (
        isinstance(budget, dict)
        and type(retained.get("bytes")) is int
        and type(budget.get("max_trace_bytes")) is int
        and retained["bytes"] > budget["max_trace_bytes"]
    ):
        problems.append(f"{label} retained route exceeds its cell budget")

    try:
        import capture_budget_route as route_verifier
        sampling_sidecar = json.loads((trace_root / "trace_sampling.json").read_text())
        route_verifier.validate_sidecar(sampling_sidecar)
    except Exception as exc:  # noqa: BLE001 - evidence decoder boundary
        reason = getattr(exc, "reason", type(exc).__name__)
        problems.append(f"{label} trace-sampling sidecar is invalid: {reason}: {exc}")
        sampling_sidecar = None
    if isinstance(sampling_sidecar, dict) and (
        sampling_sidecar.get("method") != route.get("method")
        or sampling_sidecar.get("schema_version")
        != route.get("trace_sampling_schema_version")
        or (sampling_sidecar.get("hashes") or {}).get("sidecar_sha256")
        != route.get("trace_sampling_sidecar_sha256")
    ):
        problems.append(f"{label} trace-sampling sidecar does not bind to route")

    outcome, method = route.get("outcome"), route.get("method")
    admitted = outcome == "admitted" and method in _ADMITTED_ROUTE_METHODS
    unavailable = outcome == "typed_unavailable" and method == "unavailable"
    if not (admitted or unavailable):
        problems.append(f"{label} route outcome/method combination is invalid")
    if admitted and route.get("typed_reason") is not None:
        problems.append(f"{label} admitted route must have typed_reason=null")
    if unavailable and (
        not isinstance(route.get("typed_reason"), str) or not route["typed_reason"]
    ):
        problems.append(f"{label} unavailable route lacks a typed reason")
    if (
        unavailable
        and isinstance(config, dict)
        and config.get("gate_forbidden_policy") != "typed_unavailable"
    ):
        problems.append(f"{label} unavailable outcome conflicts with route policy")
    if (
        method == "reduced_shape_matched_label"
        and isinstance(config, dict)
        and config.get("gate_forbidden_policy") != "reduced_shape"
    ):
        problems.append(f"{label} reduced-shape outcome conflicts with route policy")
    expected_disposition = "admitted" if admitted else "failed"
    expected_reason = None if admitted else "trace_unsupported"
    expected_route_outcome = "admitted" if admitted else "typed_unavailable"
    if (
        row.get("disposition") != expected_disposition
        or row.get("typed_reason") != expected_reason
    ):
        problems.append(f"{label} DONE disposition does not match route outcome")
    if (
        not isinstance(cell_result, dict)
        or cell_result.get("identity") != identity
        or cell_result.get("authority_cell_sha256") != row.get("authority_cell_sha256")
        or cell_result.get("disposition") != expected_disposition
        or cell_result.get("typed_reason") != expected_reason
        or cell_result.get("binary_sha256")
        != (protocol.get("binary_sha256") if isinstance(protocol, dict) else None)
        or cell_result.get("capture_preflight_sha256") != manifest.get("preflight_sha256")
        or cell_result.get("trace_route_outcome") != expected_route_outcome
        or cell_result.get("trace_route_result_sha256") != route.get("result_sha256")
    ):
        problems.append(f"{label} cell result is not bound to the routed trace")

    if admitted and actual_inventory:
        try:
            replay = route_verifier.verify_replay_bundle(trace_root)
        except Exception as exc:  # noqa: BLE001 - convert verifier refusal to evidence
            reason = getattr(exc, "reason", type(exc).__name__)
            problems.append(f"{label} replay bundle verification failed: {reason}: {exc}")
        else:
            replay_bindings = {
                "method": method,
                "schema_version": route.get("trace_sampling_schema_version"),
                "sidecar_sha256": route.get("trace_sampling_sidecar_sha256"),
                "retained_bytes": retained.get("bytes"),
                "retained_tree_sha256": retained.get("tree_sha256"),
                "retained_file_count": retained.get("file_count"),
            }
            if any(replay.get(field) != expected for field, expected in replay_bindings.items()):
                problems.append(f"{label} replay verifier result does not bind to route")
            if method == "lossless_compressed_full":
                try:
                    envelope = json.loads(
                        (trace_root / "ac9_replay_envelope.json").read_text()
                    )
                except (OSError, json.JSONDecodeError, UnicodeError) as exc:
                    problems.append(f"{label} replay envelope is unreadable: {exc}")
                else:
                    if (
                        not isinstance(full, dict)
                        or envelope.get("source_tree_sha256") != full.get("tree_sha256")
                        or envelope.get("source_tree_bytes") != full.get("tree_bytes")
                        or envelope.get("threadblock_count") != full.get("threadblock_count")
                    ):
                        problems.append(
                            f"{label} lossless replay source does not equal the full trace"
                        )
    return problems


def _verify_trace_unsupported_evidence(
    row: dict, *, output_root: Path, label: str,
) -> list[str]:
    """Re-derive the retained AC-9 terminal proof; a typed string is not proof."""
    problems: list[str] = []
    artifacts = row.get("artifacts") or []
    by_kind: dict[str, list[dict]] = {}
    for artifact in artifacts:
        if isinstance(artifact, dict):
            by_kind.setdefault(str(artifact.get("artifact_kind")), []).append(artifact)
    routes = by_kind.get("trace_route_record", [])
    results = by_kind.get("cell_result", [])
    if len(routes) != 1 or len(results) != 1:
        return [
            f"{label} trace_unsupported requires exactly one route record and "
            f"cell result (route={len(routes)}, result={len(results)})"
        ]
    route_path = output_root / str(routes[0].get("relative_path", ""))
    result_path = output_root / str(results[0].get("relative_path", ""))
    try:
        route = json.loads(route_path.read_text())
        cell_result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return [f"{label} trace_unsupported evidence is unreadable: {exc}"]
    identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
    unsigned = dict(route) if isinstance(route, dict) else {}
    claimed_result_sha = unsigned.pop("result_sha256", None)
    if (
        not isinstance(route, dict)
        or route.get("schema_version") != 1
        or route.get("kind") != "ac9_trace_budget_route"
        or route.get("identity") != identity
        or route.get("cell_id") != row.get("slug")
        or route.get("outcome") != "typed_unavailable"
        or route.get("method") != "unavailable"
        or not isinstance(route.get("typed_reason"), str)
        or not route["typed_reason"]
        or claimed_result_sha != _canonical_sha256(unsigned)
    ):
        problems.append(f"{label} trace_unsupported route record is not sealed terminal evidence")
    if (
        not isinstance(cell_result, dict)
        or cell_result.get("identity") != identity
        or cell_result.get("authority_cell_sha256")
        != row.get("authority_cell_sha256")
        or cell_result.get("disposition") != "failed"
        or cell_result.get("typed_reason") != "trace_unsupported"
        or cell_result.get("trace_route_result_sha256") != claimed_result_sha
    ):
        problems.append(f"{label} cell result is not bound to the terminal route")
    retained = route.get("retained") if isinstance(route, dict) else None
    retained_artifacts = retained.get("artifacts") if isinstance(retained, dict) else None
    trace_root = route_path.parent / "trace"
    actual_inventory: list[dict] = []
    if not isinstance(retained_artifacts, list) or not retained_artifacts:
        problems.append(f"{label} route retained inventory is empty")
    elif trace_root.is_symlink() or not trace_root.is_dir():
        problems.append(f"{label} retained trace directory is absent")
    else:
        for path in sorted(trace_root.rglob("*")):
            if path.is_symlink():
                problems.append(f"{label} retained trace contains symlink: {path}")
                continue
            if path.is_file():
                actual_inventory.append({
                    "relative_path": path.relative_to(trace_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
        claimed_inventory = [
            {
                "relative_path": item.get("relative_path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
            }
            for item in retained_artifacts if isinstance(item, dict)
        ]
        if claimed_inventory != actual_inventory:
            problems.append(f"{label} retained route inventory does not match disk")
        if (
            retained.get("file_count") != len(actual_inventory)
            or retained.get("bytes")
            != sum(item["size_bytes"] for item in actual_inventory)
            or retained.get("tree_sha256") != _canonical_sha256(actual_inventory)
        ):
            problems.append(f"{label} retained route aggregate seal is stale")
        unavailable_path = trace_root / "ac9_unavailable_evidence.json"
        try:
            unavailable = json.loads(unavailable_path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            problems.append(f"{label} unavailable evidence is unreadable: {exc}")
        else:
            if (
                not isinstance(unavailable, dict)
                or unavailable.get("kind") != "ac9_typed_unavailable_evidence"
                or unavailable.get("typed_reason") != route.get("typed_reason")
                or unavailable.get("method") != "unavailable"
            ):
                problems.append(f"{label} unavailable evidence does not match route")
    return problems


def _matrix_sol_env_pin(
    matrix: Any, *, label: str, problems: list[str],
) -> dict | None:
    """Return the current SOL fingerprint pin, or ``None`` for legacy matrices."""
    if not isinstance(matrix, dict):
        return None
    input_pins = matrix.get("input_pins")
    if not isinstance(input_pins, dict) or SOL_NATIVE_ENV_PIN_SECTION not in input_pins:
        return None
    pin = input_pins.get(SOL_NATIVE_ENV_PIN_SECTION)
    if not isinstance(pin, dict):
        problems.append(f"{label} is not an object")
        return {}
    if set(pin) != _SOL_ENV_PIN_FIELDS:
        problems.append(
            f"{label} fields={sorted(pin)} != exact fields="
            f"{sorted(_SOL_ENV_PIN_FIELDS)}"
        )
    if pin.get("path") != SOL_NATIVE_ENV_PIN_PATH:
        problems.append(
            f"{label}.path={pin.get('path')!r} != certified path "
            f"{SOL_NATIVE_ENV_PIN_PATH!r}"
        )
    _validate_hash(f"{label}.sha256", pin.get("sha256"), problems)
    if pin.get("native_env_hash") != SOL_CERTIFIED_NATIVE_ENV_HASH:
        problems.append(
            f"{label}.native_env_hash={pin.get('native_env_hash')!r} != "
            f"certified hash {SOL_CERTIFIED_NATIVE_ENV_HASH}"
        )
    return pin


def _matrix_sol_cell(
    matrix: dict, identity: dict, *, label: str, problems: list[str],
) -> dict | None:
    rows = matrix.get("cells")
    if not isinstance(rows, list):
        problems.append(f"{label} cannot bind attestation: capture matrix cells is not an array")
        return None
    matches = [
        row for row in rows
        if isinstance(row, dict) and row.get("identity") == identity
    ]
    if len(matches) != 1:
        problems.append(
            f"{label} must match exactly one capture-matrix row; found {len(matches)}"
        )
        return None
    row = matches[0]
    invocation = row.get("invocation")
    contract = (
        invocation.get("adapter_contract") if isinstance(invocation, dict) else None
    )
    environment = contract.get("environment") if isinstance(contract, dict) else None
    if environment != SOL_CERTIFIED_ENVIRONMENT:
        problems.append(
            f"{label} matrix adapter environment is not the certified "
            "CUDA 13.2/GCC 14 contract"
        )
    if not isinstance(contract, dict) or contract.get(
        "native_env_hash"
    ) != SOL_CERTIFIED_NATIVE_ENV_HASH:
        problems.append(f"{label} matrix adapter native_env_hash is not certified")
    return row


def _pinned_sol_fingerprint(
    pin: dict, *, matrix: dict, repo_root: Path, source_commit: Any,
    label: str, problems: list[str],
) -> dict | None:
    raw_path = pin.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    try:
        path = _resolve_input_path(raw_path, repo_root)
    except DoneManifestError as exc:
        problems.append(f"{label} cannot resolve certified fingerprint: {exc}")
        return None
    payload, observed_sha, origin = _reference_bytes_at_seal(
        path, expected_sha256=pin.get("sha256"), repo_root=repo_root,
        source_commit=source_commit,
    )
    if payload is None:
        problems.append(f"{label} certified fingerprint is unavailable: {path}")
        return None
    if observed_sha != pin.get("sha256"):
        problems.append(
            f"{label} certified fingerprint hash is stale: "
            f"matrix={pin.get('sha256')!r} actual={observed_sha} origin={origin}"
        )
    doc = _load_json_bytes(payload, path, "SOL native-env fingerprint", problems)
    if doc is None:
        return None
    components = doc.get("components")
    if (
        doc.get("schema_version") != 1
        or doc.get("fingerprint_id") != "sm86_sol_native_env_fingerprint_v1"
        or doc.get("arch") != matrix.get("arch")
        or doc.get("native_env_hash") != SOL_CERTIFIED_NATIVE_ENV_HASH
        or doc.get("degraded") is not False
        or doc.get("missing_components") != []
        or not isinstance(components, dict)
        or set(components) != set(SOL_NATIVE_ENV_COMPONENTS)
        or not all(isinstance(value, str) for value in components.values())
        or _canonical_sha256(components) != SOL_CERTIFIED_NATIVE_ENV_HASH
    ):
        problems.append(f"{label} pinned SOL native-env fingerprint is not certified")
        return None
    return doc


def _sol_attestation_path(
    row: dict, *, output_root: Path, label: str, problems: list[str],
) -> Path | None:
    artifacts = row.get("artifacts")
    safe_artifacts = artifacts if isinstance(artifacts, list) else []
    matches = [
        artifact for artifact in safe_artifacts
        if isinstance(artifact, dict) and _is_sol_env_attestation_artifact(artifact)
    ]
    if len(matches) != 1:
        problems.append(
            f"{label} requires exactly one sealed "
            f"{SOL_NATIVE_ENV_ATTESTATION_FILENAME}; found {len(matches)}"
        )
        return None
    artifact = matches[0]
    rel = artifact.get("relative_path")
    if not _valid_relative_artifact_path(rel):
        return None
    candidate = output_root / rel
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        problems.append(f"{label} SOL native-env attestation is unavailable/unsafe: {exc}")
        return None
    if candidate.is_symlink() or not resolved.is_file():
        problems.append(f"{label} SOL native-env attestation is not a regular file")
        return None
    try:
        size = resolved.stat().st_size
        sha = _sha256_file(resolved)
    except OSError as exc:
        problems.append(f"{label} cannot hash SOL native-env attestation: {exc}")
        return None
    if size != artifact.get("size_bytes"):
        problems.append(f"{label} SOL native-env attestation size seal is stale")
    if sha != artifact.get("sha256"):
        problems.append(f"{label} SOL native-env attestation artifact hash is stale")
    return resolved


def _verify_sol_attestation_document(
    doc: Any, *, row: dict, pin: dict, matrix: dict,
    pinned_fingerprint: dict | None, label: str,
) -> list[str]:
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"{label} SOL native-env attestation is not an object"]
    if set(doc) != _SOL_ENV_ATTESTATION_FIELDS:
        problems.append(
            f"{label} SOL native-env attestation fields={sorted(doc)} != exact "
            f"fields={sorted(_SOL_ENV_ATTESTATION_FIELDS)}"
        )
    identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
    fixed = {
        "schema_version": SOL_NATIVE_ENV_ATTESTATION_SCHEMA_VERSION,
        "kind": SOL_NATIVE_ENV_ATTESTATION_KIND,
        "identity": identity,
        "certified_native_env_hash": SOL_CERTIFIED_NATIVE_ENV_HASH,
        "observed_native_env_hash": SOL_CERTIFIED_NATIVE_ENV_HASH,
        "certified_hash_equal": True,
        "degraded": False,
        "missing_components": [],
        "matrix_pin": pin,
    }
    for field, expected in fixed.items():
        if doc.get(field) != expected:
            problems.append(
                f"{label} SOL native-env attestation {field}="
                f"{doc.get(field)!r} != {expected!r}"
            )

    components = doc.get("components")
    components_valid = (
        isinstance(components, dict)
        and set(components) == set(SOL_NATIVE_ENV_COMPONENTS)
        and all(isinstance(value, str) for value in components.values())
        and all(value != "unknown" for value in components.values())
    )
    if not components_valid:
        problems.append(f"{label} SOL native-env components are incomplete or degraded")
    else:
        components_sha = _canonical_sha256(components)
        if components_sha != SOL_CERTIFIED_NATIVE_ENV_HASH:
            problems.append(
                f"{label} SOL native-env components rehash to {components_sha}, "
                f"not {SOL_CERTIFIED_NATIVE_ENV_HASH}"
            )
        if doc.get("components_sha256") != components_sha:
            problems.append(f"{label} SOL native-env components_sha256 is stale")
        if (
            pinned_fingerprint is not None
            and components != pinned_fingerprint.get("components")
        ):
            problems.append(
                f"{label} SOL native-env components differ from the pinned fingerprint"
            )

    compiler = doc.get("compiler_identity")
    if not isinstance(compiler, dict) or set(compiler) != _SOL_COMPILER_IDENTITY_FIELDS:
        problems.append(f"{label} canonical compiler identity is absent or malformed")
    else:
        requested = compiler.get("requested")
        realpath = compiler.get("resolved_realpath")
        version_line = compiler.get("version_line")
        if requested != SOL_CERTIFIED_NVCC_CCBIN:
            problems.append(f"{label} canonical compiler request is not g++-14")
        if (
            not isinstance(realpath, str) or not realpath
            or not PurePosixPath(realpath).is_absolute()
            or ".." in PurePosixPath(realpath).parts
            or "\n" in realpath or "\r" in realpath
        ):
            problems.append(f"{label} canonical compiler realpath is invalid")
        _validate_hash(
            f"{label}.compiler_identity.executable_sha256",
            compiler.get("executable_sha256"), problems,
        )
        if (
            not isinstance(version_line, str) or not version_line
            or version_line != version_line.strip()
            or "\n" in version_line or "\r" in version_line
        ):
            problems.append(f"{label} canonical compiler version_line is invalid")

    claimed_seal = doc.get("attestation_sha256")
    unsigned = {key: value for key, value in doc.items() if key != "attestation_sha256"}
    if claimed_seal != _canonical_sha256(unsigned):
        problems.append(f"{label} SOL native-env attestation seal is stale")
    return problems


def verify_manifest_sol_native_env_attestations(
    manifest: dict, matrix: dict, *, output_root: Path | str | None = None,
    repo_root: Path | str | None = None, source_commit: Any = None,
) -> list[str]:
    """Verify current-matrix SOL env attestations, preserving legacy matrices.

    Absence of the direct matrix pin identifies a pre-attestation matrix.  Once
    that pin is present, every terminal SOL result must carry exactly one
    content-verified attestation.  This function is also used by campaign
    accounting when a historically valid DONE is joined to the current matrix.
    """
    problems: list[str] = []
    pin = _matrix_sol_env_pin(
        matrix, label=f"input_pins.{SOL_NATIVE_ENV_PIN_SECTION}", problems=problems,
    )
    if pin is None:
        return problems
    if not pin:
        return problems
    rows = manifest.get("cell_results")
    safe_rows = rows if isinstance(rows, list) else []
    terminal_sol_rows = [
        (index, row) for index, row in enumerate(safe_rows)
        if (
            isinstance(row, dict)
            and row.get("roster_source") == "sol_execbench"
            and _is_terminal_capture_result(row)
        )
    ]
    if not terminal_sol_rows:
        return problems
    output_raw = manifest.get("output_root") if output_root is None else str(output_root)
    if not isinstance(output_raw, str) or not Path(output_raw).is_absolute():
        problems.append("SOL native-env attestation verification requires absolute output_root")
        return problems
    root = Path(output_raw).resolve()
    pinned_fingerprint = None
    if repo_root is not None:
        pinned_fingerprint = _pinned_sol_fingerprint(
            pin, matrix=matrix, repo_root=Path(repo_root),
            source_commit=(manifest.get("source_commit_pre")
                           if source_commit is None else source_commit),
            label="SOL native-env attestation", problems=problems,
        )

    for index, row in terminal_sol_rows:
        label = f"cell_results[{index}]"
        identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
        _matrix_sol_cell(matrix, identity, label=label, problems=problems)
        path = _sol_attestation_path(
            row, output_root=root, label=label, problems=problems,
        )
        if path is None:
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            problems.append(f"{label} SOL native-env attestation is unreadable: {exc}")
            continue
        problems.extend(_verify_sol_attestation_document(
            doc, row=row, pin=pin, matrix=matrix,
            pinned_fingerprint=pinned_fingerprint, label=label,
        ))
    return problems


def _sol_definition_artifact(
    row: dict, *, output_root: Path, basename: str, label: str,
    problems: list[str],
) -> tuple[Path | None, dict | None]:
    artifacts = row.get("artifacts")
    safe_artifacts = artifacts if isinstance(artifacts, list) else []
    matches = [
        artifact for artifact in safe_artifacts
        if (
            isinstance(artifact, dict)
            and artifact.get("artifact_kind") == "sol_capture_evidence"
            and isinstance(artifact.get("relative_path"), str)
            and PurePosixPath(artifact["relative_path"]).name == basename
            and PurePosixPath(artifact["relative_path"]).parent.name == "sol_stage"
        )
    ]
    if len(matches) != 1:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label} requires exactly one sealed sol_stage/{basename}; "
            f"found {len(matches)}"
        )
        return None, None
    local_problems: list[str] = []
    path = _resolve_sealed_artifact(
        matches[0], output_root=output_root,
        label=f"{label} sol_stage/{basename}", problems=local_problems,
    )
    problems.extend(
        f"SOL_ATTESTATION_STATE_MISMATCH: {problem}"
        for problem in local_problems
    )
    return path, matches[0]


def _matrix_sol_definition_pin(
    matrix_row: dict, *, label: str, problems: list[str],
) -> str | None:
    invocation = matrix_row.get("invocation")
    logical = (
        invocation.get("logical_workload")
        if isinstance(invocation, dict) else None
    )
    pin = (
        logical.get("definition_file_sha256")
        if isinstance(logical, dict) else None
    )
    if not isinstance(pin, str) or not _SHA64_RE.fullmatch(pin):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label} matrix row lacks a pinned definition_file_sha256"
        )
        return None
    return pin


def _e387_legacy_definition_code_context(
    *, prepare_manifest: Any, repo_root: Path, source_commit: Any,
    label: str, problems: list[str],
) -> dict[str, str] | None:
    """Bind the sole seven-field attestation exception to exact Git blobs."""

    if source_commit != SOL_E387_LEGACY_SOURCE_COMMIT:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label} seven-field V2 attestation is not sealed at e387"
        )
        return None
    comparator_blob = _git_blob_at_commit(
        repo_root, source_commit, SOL_E387_LEGACY_COMPARATOR_PATH
    )
    adapter_blob = _git_blob_at_commit(
        repo_root, source_commit, SOL_E387_LEGACY_CAPTURE_ADAPTER_PATH
    )
    comparator_sha = (
        hashlib.sha256(comparator_blob).hexdigest()
        if comparator_blob is not None else None
    )
    adapter_sha = (
        hashlib.sha256(adapter_blob).hexdigest()
        if adapter_blob is not None else None
    )
    if comparator_sha != SOL_E387_LEGACY_COMPARATOR_SHA256:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label} e387 comparator blob hash is not the allowlisted value"
        )
    if adapter_sha != SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label} e387 capture-adapter blob hash is not the allowlisted value"
        )
    artifacts = (
        prepare_manifest.get("artifacts")
        if isinstance(prepare_manifest, dict) else None
    )
    adapter_entry = (
        artifacts.get("adapter_source")
        if isinstance(artifacts, dict) else None
    )
    raw_adapter_path = (
        adapter_entry.get("path")
        if isinstance(adapter_entry, dict) else None
    )
    declared_adapter_sha = (
        adapter_entry.get("sha256")
        if isinstance(adapter_entry, dict) else None
    )
    try:
        adapter_path = PurePosixPath(raw_adapter_path)
        path_matches = (
            adapter_path.is_absolute()
            and tuple(adapter_path.parts[-len(
                SOL_E387_LEGACY_CAPTURE_ADAPTER_PATH.parts
            ):]) == SOL_E387_LEGACY_CAPTURE_ADAPTER_PATH.parts
        )
    except TypeError:
        path_matches = False
    if (
        not isinstance(adapter_entry, dict)
        or set(adapter_entry) != {"path", "sha256"}
        or not path_matches
        or declared_adapter_sha != SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256
    ):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label} prepare manifest does not bind the exact e387 adapter"
        )
    if (
        comparator_sha != SOL_E387_LEGACY_COMPARATOR_SHA256
        or adapter_sha != SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256
        or declared_adapter_sha != SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256
        or not path_matches
    ):
        return None
    return {
        "source_commit": source_commit,
        "comparator_sha256": comparator_sha,
        "capture_adapter_sha256": adapter_sha,
    }


def _canonical_definition_attestation_bytes(value: Any) -> bytes | None:
    try:
        import sol_definition_equivalence as sol_def  # noqa: E402
        return sol_def.serialize_attestation(value)
    except (ImportError, TypeError, ValueError, UnicodeError):
        return None


def _verify_sol_definition_attestation(
    row: dict, *, matrix_row: dict, output_root: Path, label: str,
    repo_root: Path, source_commit: Any,
) -> list[str]:
    """Recompute one SOL V2 seal from its four published stage records."""
    try:
        import sol_definition_equivalence as sol_def  # noqa: E402
    except ImportError:
        sol_def = None
    legacy_row_shape = bool(
        sol_def is not None
        and sol_def.is_e387_legacy_attestation_shape(
            row.get("definition_equivalence")
        )
    )
    problems = _definition_attestation_structure_problems(
        row.get("definition_equivalence"),
        label=f"{label}.definition_equivalence",
        allow_e387_legacy_shape=(
            legacy_row_shape
            and source_commit == SOL_E387_LEGACY_SOURCE_COMMIT
        ),
    )
    claimed_sha = row.get("definition_equivalence_attestation_sha256")
    if not isinstance(claimed_sha, str) or not _SHA64_RE.fullmatch(claimed_sha):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: "
            f"{label}.definition_equivalence_attestation_sha256 is not sha256"
        )
    pinned_source_sha = _matrix_sol_definition_pin(
        matrix_row, label=label, problems=problems,
    )

    located = {
        name: _sol_definition_artifact(
            row, output_root=output_root, basename=basename, label=label,
            problems=problems,
        )
        for name, basename in {
            "attestation": SOL_DEFINITION_EQUIVALENCE_FILENAME,
            "prepare_manifest": SOL_DEFINITION_PREPARE_MANIFEST_FILENAME,
            "source": SOL_DEFINITION_SOURCE_FILENAME,
            "staged": SOL_DEFINITION_STAGED_FILENAME,
        }.items()
    }
    if any(path is None for path, _artifact in located.values()):
        return problems
    attestation_path, attestation_artifact = located["attestation"]
    manifest_path, _manifest_artifact = located["prepare_manifest"]
    source_path, _source_artifact = located["source"]
    staged_path, _staged_artifact = located["staged"]
    assert attestation_path is not None and attestation_artifact is not None
    assert manifest_path is not None and source_path is not None and staged_path is not None
    stage_roots = {
        path.parent for path in (
            attestation_path, manifest_path, source_path, staged_path,
        )
    }
    if len(stage_roots) != 1:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: definition evidence spans "
            "multiple sol_stage roots"
        )
        return problems

    try:
        attestation_bytes = attestation_path.read_bytes()
        attestation_doc = json.loads(attestation_bytes)
        prepare_manifest = json.loads(manifest_path.read_text())
        source_bytes = source_path.read_bytes()
        staged_bytes = staged_path.read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        problems.append(
            f"SOL_ATTESTATION_STATE_MISMATCH: {label} stage evidence unreadable: {exc}"
        )
        return problems
    if not isinstance(attestation_doc, dict):
        problems.append(
            "SOL_ATTESTATION_RULE_MISMATCH: attestation file is not an object"
        )
        return problems
    problems.extend(_definition_attestation_structure_problems(
        attestation_doc, label=f"{label} attestation file",
        allow_e387_legacy_shape=(
            legacy_row_shape
            and source_commit == SOL_E387_LEGACY_SOURCE_COMMIT
        ),
    ))
    actual_attestation_sha = hashlib.sha256(attestation_bytes).hexdigest()
    if (
        actual_attestation_sha != claimed_sha
        or attestation_artifact.get("sha256") != actual_attestation_sha
    ):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: attestation SHA does not bind "
            "the published file"
        )
    if row.get("definition_equivalence") != attestation_doc:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: DONE attestation differs from "
            "the published attestation file"
        )

    source_sha = hashlib.sha256(source_bytes).hexdigest()
    staged_sha = hashlib.sha256(staged_bytes).hexdigest()
    if pinned_source_sha is not None and source_sha != pinned_source_sha:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: source definition SHA differs "
            "from the matrix per-row pin"
        )
    if (
        attestation_doc.get("source_definition_sha256") != source_sha
        or attestation_doc.get("staged_definition_sha256") != staged_sha
    ):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: attestation definition SHAs "
            "do not rederive from sealed bytes"
        )

    if not isinstance(prepare_manifest, dict):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: prepare manifest is not an object"
        )
        return problems
    manifest_artifacts = prepare_manifest.get("artifacts")
    expected_manifest_artifacts = {
        "definition": (SOL_DEFINITION_STAGED_FILENAME, staged_sha),
        "definition_source": (SOL_DEFINITION_SOURCE_FILENAME, source_sha),
        "definition_equivalence_attestation": (
            SOL_DEFINITION_EQUIVALENCE_FILENAME, actual_attestation_sha,
        ),
    }
    if not isinstance(manifest_artifacts, dict):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: prepare manifest artifacts is "
            "not an object"
        )
    else:
        for name, (expected_path, expected_sha) in expected_manifest_artifacts.items():
            if manifest_artifacts.get(name) != {
                "path": expected_path, "sha256": expected_sha,
            }:
                problems.append(
                    "SOL_ATTESTATION_STATE_MISMATCH: prepare manifest "
                    f"artifacts.{name} does not bind the sealed stage file"
                )
    if (
        prepare_manifest.get("source_definition_sha256") != source_sha
        or prepare_manifest.get("staged_definition_sha256") != staged_sha
        or prepare_manifest.get("definition_equivalence") != attestation_doc
    ):
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: prepare manifest definition "
            "records do not match the V2 attestation"
        )
    unsigned_manifest = dict(prepare_manifest)
    claimed_stage_definition_sha = unsigned_manifest.pop(
        "stage_definition_sha256", None
    )
    if (
        not isinstance(claimed_stage_definition_sha, str)
        or claimed_stage_definition_sha != _sol_stage_manifest_sha256(
            unsigned_manifest
        )
    ):
        # The adapter's stage-definition algorithm uses the same canonical
        # object hash.  Its semantics are deliberately unchanged by V2.
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: prepare manifest stage-definition "
            "seal is stale"
        )

    try:
        import sol_definition_equivalence as sol_def  # noqa: E402
    except ImportError as exc:
        problems.append(
            "SOL_ATTESTATION_STATE_MISMATCH: definition-equivalence verifier "
            f"unavailable: {exc}"
        )
        return problems
    try:
        comparison = sol_def.compare_definition_bytes(source_bytes, staged_bytes)
    except sol_def.SolDefinitionEquivalenceError as exc:
        problems.append(f"{exc.code}: {exc.detail}")
        return problems
    if not comparison.equal:
        pointers = list(comparison.difference_pointers)
        problems.append(
            "sol_stage_source_mismatch: sealed definitions differ at "
            f"{pointers[:20]!r}; total={len(pointers)}"
        )
        return problems
    expected_attestation = comparison.attestation
    if legacy_row_shape:
        context = _e387_legacy_definition_code_context(
            prepare_manifest=prepare_manifest,
            repo_root=repo_root,
            source_commit=source_commit,
            label=label,
            problems=problems,
        )
        if context is not None:
            try:
                canonical_bytes = sol_def.serialize_e387_legacy_attestation(
                    attestation_doc,
                    expected_attestation,
                    **context,
                )
            except sol_def.SolDefinitionEquivalenceError as exc:
                problems.append(f"{exc.code}: {exc.detail}")
            else:
                if canonical_bytes != attestation_bytes:
                    problems.append(
                        "SOL_ATTESTATION_STATE_MISMATCH: e387 attestation "
                        "file is not deterministically serialized"
                    )
    else:
        canonical_bytes = _canonical_definition_attestation_bytes(
            attestation_doc
        )
        if canonical_bytes is None or canonical_bytes != attestation_bytes:
            problems.append(
                "SOL_ATTESTATION_STATE_MISMATCH: attestation file does not use "
                "the deterministic V2 serialization"
            )
        try:
            ineligible_pointers = (
                sol_def.ineligible_normalized_state_pointers(
                    attestation_doc, expected_attestation
                )
            )
        except sol_def.SolDefinitionEquivalenceError as exc:
            problems.append(f"{exc.code}: {exc.detail}")
            return problems
        if ineligible_pointers:
            problems.append(
                "SOL_ATTESTATION_POINTER_INELIGIBLE: recorded normalized "
                "states are outside the sealed definitions: "
                f"{list(ineligible_pointers)!r}"
            )
        if expected_attestation != attestation_doc:
            problems.append(
                "SOL_ATTESTATION_STATE_MISMATCH: recorded V2 states differ "
                "from recomputation over sealed source/staged bytes"
            )
    return problems


def verify_manifest_sol_definition_attestations(
    manifest: dict, matrix: dict, *, output_root: Path | str | None = None,
    repo_root: Path | str = REPO_ROOT, source_commit: Any = None,
) -> list[str]:
    """Verify every current-matrix SOL cell's exact V2 stage attestation."""
    rows = manifest.get("cell_results")
    safe_rows = rows if isinstance(rows, list) else []
    sol_rows = [
        (index, row) for index, row in enumerate(safe_rows)
        if isinstance(row, dict) and row.get("roster_source") == "sol_execbench"
    ]
    if not sol_rows:
        return []

    # A pre-V2 matrix cannot certify a V2 row.  Current admitted matrices carry
    # a direct per-row definition SHA, while historical non-SOL evidence is
    # handled only by the exact portable-bootstrap allowlist.
    matrix_cells = matrix.get("cells") if isinstance(matrix, dict) else None
    has_v2_pins = any(
        isinstance(cell, dict)
        and cell.get("identity", {}).get("roster_source") == "sol_execbench"
        and isinstance(
            (((cell.get("invocation") or {}).get("logical_workload") or {}).get(
                "definition_file_sha256"
            )),
            str,
        )
        for cell in (matrix_cells if isinstance(matrix_cells, list) else [])
    )
    if not has_v2_pins:
        return [
            "SOL_ATTESTATION_RULE_MISMATCH: current matrix lacks SOL V2 "
            "definition pins"
        ]
    output_raw = manifest.get("output_root") if output_root is None else str(output_root)
    if not isinstance(output_raw, str) or not Path(output_raw).is_absolute():
        return [
            "SOL_ATTESTATION_STATE_MISMATCH: V2 verification requires an "
            "absolute output_root"
        ]
    root = Path(output_raw).resolve()
    repository = Path(repo_root)
    sealed_commit = (
        manifest.get("source_commit_pre")
        if source_commit is None else source_commit
    )
    problems: list[str] = []
    for index, row in sol_rows:
        label = f"cell_results[{index}]"
        identity = {"roster_source": row.get("roster_source"), "slug": row.get("slug")}
        matrix_row = _matrix_sol_cell(
            matrix, identity, label=label, problems=problems,
        )
        if matrix_row is None:
            continue
        problems.extend(_verify_sol_definition_attestation(
            row,
            matrix_row=matrix_row,
            output_root=root,
            label=label,
            repo_root=repository,
            source_commit=sealed_commit,
        ))
    return problems


_MAX_NCU_DIMENSION = (1 << 31) - 1
_MAX_NCU_DEVICE_ORDINAL = (1 << 31) - 1


def _parse_ncu_dimension(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*(?:\(\s*)?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
        r"\s*(?:\)\s*)?",
        value,
    )
    if (
        match is None
        or (value.lstrip().startswith("(") != value.rstrip().endswith(")"))
    ):
        return None
    try:
        dimensions = tuple(int(component) for component in match.groups())
    except ValueError:
        return None
    if any(
        component <= 0 or component > _MAX_NCU_DIMENSION
        for component in dimensions
    ):
        return None
    return dimensions  # type: ignore[return-value]


def _parse_ncu_device(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    raw_ordinal = stripped if stripped.isdecimal() else None
    if raw_ordinal is None:
        match = re.fullmatch(r".*\((\d+)\)\s*", stripped)
        raw_ordinal = match.group(1) if match is not None else None
    if raw_ordinal is None:
        return None
    try:
        ordinal = int(raw_ordinal)
    except ValueError:
        return None
    return (
        ordinal if ordinal <= _MAX_NCU_DEVICE_ORDINAL else None
    )


def _ncu_observation_launch_key(
    observation: dict,
) -> tuple[str, tuple[int, int, int], tuple[int, int, int], int] | None:
    identity = observation.get("identity")
    if not isinstance(identity, dict):
        return None
    name = identity.get("Kernel Name")
    block = _parse_ncu_dimension(identity.get("Block Size"))
    grid = _parse_ncu_dimension(identity.get("Grid Size"))
    device = _parse_ncu_device(identity.get("Device"))
    if (
        observation.get("kernel_name_base") != "mangled"
        or not isinstance(name, str)
        or not name.strip()
        or block is None
        or grid is None
        or device is None
    ):
        return None
    return name, block, grid, device


def _trace_launch_key(
    launch: Any,
) -> tuple[str, tuple[int, int, int], tuple[int, int, int], int] | None:
    if not isinstance(launch, dict):
        return None
    raw_name = launch.get("kernel_name")
    suffix = "___0"
    if (
        not isinstance(raw_name, str)
        or not raw_name.endswith(suffix)
        or not raw_name[:-len(suffix)]
        or raw_name[:-len(suffix)].endswith(suffix)
        or raw_name.startswith("regex:")
    ):
        return None
    block = launch.get("block_dim")
    grid = launch.get("grid_dim")
    device = launch.get("device_id")
    if (
        not isinstance(block, list)
        or len(block) != 3
        or not isinstance(grid, list)
        or len(grid) != 3
        or any(type(component) is not int or component <= 0 for component in [*block, *grid])
        or type(device) is not int
        or device < 0
    ):
        return None
    return (
        raw_name[:-len(suffix)],
        tuple(block),
        tuple(grid),
        device,
    )  # type: ignore[return-value]


def _ncu_launch_observation_semantic_problems(
    observation: dict, *, label: str,
) -> list[str]:
    identity = observation.get("identity")
    if not isinstance(identity, dict):
        return [f"{label} NCU launch observation identity is not an object"]
    problems: list[str] = []
    for field in ("Block Size", "Grid Size"):
        if _parse_ncu_dimension(identity.get(field)) is None:
            problems.append(f"{label} NCU launch observation {field} is invalid")
    if _parse_ncu_device(identity.get("Device")) is None:
        problems.append(f"{label} NCU launch observation Device is invalid")
    return problems


def _verify_ncu_collection_receipt(
    row: dict,
    *,
    manifest: dict,
    preflight: dict | None,
    matrix: dict | None = None,
    output_root: Path,
    label: str,
) -> list[str]:
    """Bind an admitted collection receipt to retained preflight authority."""
    if row.get("disposition") != "admitted":
        return []
    problems: list[str] = []
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, list):
        return [f"{label}.artifacts is not an array"]
    receipt_artifacts = [
        artifact for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("artifact_kind") == "ncu_collection_receipt"
    ]
    if len(receipt_artifacts) != 1:
        return [
            f"{label} requires exactly one ncu_collection_receipt artifact "
            f"(observed={len(receipt_artifacts)})"
        ]
    if not isinstance(preflight, dict):
        return [f"{label} cannot derive NCU authority from retained preflight"]
    checks = preflight.get("checks")
    if not isinstance(checks, dict):
        return [f"{label} preflight lacks checks.ncu authority"]
    ncu = checks.get("ncu")
    if not isinstance(ncu, dict):
        return [f"{label} preflight lacks checks.ncu authority"]
    try:
        expected = ncu_action_executor.build_expected_ncu_identity(
            resolved_path=ncu["resolved_path"],
            executable_sha256=ncu["executable_sha256"],
            version_line=ncu["version_line"],
        )
    except (KeyError, ncu_action_executor.NcuActionExecutorError) as exc:
        return [f"{label} preflight NCU authority is invalid: {exc}"]
    relative = receipt_artifacts[0].get("relative_path")
    if not _valid_relative_artifact_path(relative):
        return [f"{label} NCU receipt path is invalid"]
    receipt_path = PurePosixPath(relative)
    source = row.get("roster_source")
    slug = row.get("slug")
    if not isinstance(source, str) or not isinstance(slug, str):
        return [f"{label} NCU receipt cell identity is invalid"]
    expected_cell_prefixes = {
        (source, slug),
        ("cells", f"{source}__{slug}"),
    }
    expected_receipt_parents = {
        (*prefix, "logs") for prefix in expected_cell_prefixes
    }
    if receipt_path.parent.parts not in expected_receipt_parents:
        return [f"{label} NCU receipt path is outside its DONE cell"]
    receipt_name = receipt_path.name
    receipt_suffix = "_ncu_collection_receipt.json"
    receipt_tag = receipt_name.removesuffix(receipt_suffix)
    if (
        receipt_path.parent.name != "logs"
        or not receipt_tag
        or not receipt_name.endswith(receipt_suffix)
        or any(not (character.isalnum() or character in "_-")
               for character in receipt_tag)
    ):
        return [f"{label} NCU receipt path is not canonical"]
    receipt = _load_strict_json(
        output_root / relative, f"{label}.ncu_collection_receipt", problems,
    )
    if receipt is None:
        return problems or [f"{label} NCU receipt is unavailable"]
    try:
        ncu_collection_receipt.validate_schema(receipt)
        immutable.verify_model_seal(receipt, "receipt_seal_sha256")
    except (
        ncu_collection_receipt.CollectionReceiptError,
        immutable.ImmutableArtifactError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        problems.append(f"{label} NCU receipt is invalid: {exc}")
        return problems
    expected_identity = {
        "resolved_path": expected.resolved_path,
        "executable_sha256": expected.executable_sha256,
        "version_line": expected.version_line,
    }
    tool_versions = manifest.get("tool_versions")
    expected_ncu_version = (
        tool_versions.get("ncu") if isinstance(tool_versions, dict) else None
    )
    try:
        ncu_version_matches = capture_toolchain_contract.ncu_version_matches(
            expected.version_line, expected_ncu_version,
        )
    except capture_toolchain_contract.CaptureToolchainContractError:
        ncu_version_matches = False
    if not ncu_version_matches:
        problems.append(
            f"{label} retained preflight NCU version differs from DONE pin"
        )
    expected_binary_sha256: str | None = None
    expected_shape_point: str | None = None
    if isinstance(matrix, dict):
        matrix_row = _route_matrix_row(
            matrix,
            {"roster_source": source, "slug": slug},
            label=label,
            problems=problems,
        )
        invocation = (
            matrix_row.get("invocation")
            if isinstance(matrix_row, dict) else None
        )
        if not isinstance(invocation, dict):
            problems.append(f"{label} matrix invocation is invalid")
        else:
            candidate_binary_sha256 = invocation.get("executable_sha256")
            if (
                not isinstance(candidate_binary_sha256, str)
                or _SHA64_RE.fullmatch(candidate_binary_sha256) is None
            ):
                problems.append(
                    f"{label} matrix invocation executable_sha256 is invalid"
                )
            else:
                expected_binary_sha256 = candidate_binary_sha256
            logical_workload = invocation.get("logical_workload")
            if isinstance(logical_workload, dict):
                candidate_shape_point = logical_workload.get("value")
                if (
                    logical_workload.get("kind") == "shape_point"
                    and isinstance(candidate_shape_point, str)
                    and candidate_shape_point
                ):
                    expected_shape_point = candidate_shape_point
    artifact_by_path = {
        artifact.get("relative_path"): artifact
        for artifact in artifacts
        if (
            isinstance(artifact, dict)
            and isinstance(artifact.get("relative_path"), str)
        )
    }
    receipt_cell_dir = (output_root / relative).parent.parent
    role_kinds = {
        "ncu_report": "ncu_report",
        "label_run_manifest": "label_run_manifest",
        "ncu_launch_observation": "ncu_launch_observation",
        "label_capture_manifest": "execution_log",
    }
    receipt_bindings = receipt.get("artifacts")
    external_bindings: list[dict] = []
    observation: dict | None = None
    launch_observation: dict | None = None
    capture_manifest: dict | None = None
    report_sha256: str | None = None
    if not isinstance(receipt_bindings, list):
        problems.append(f"{label} NCU receipt artifacts are not an array")
    else:
        for binding in receipt_bindings:
            if not isinstance(binding, dict):
                continue
            role = binding.get("role")
            internal_relative = binding.get("relative_path")
            if not isinstance(role, str) or role not in role_kinds:
                continue
            if not _valid_relative_artifact_path(internal_relative):
                problems.append(
                    f"{label} NCU receipt {role} path is invalid"
                )
                continue
            if role == "label_capture_manifest":
                capture_relative = PurePosixPath(internal_relative)
                expected_capture_relative = PurePosixPath(
                    "logs", f"{receipt_tag}_capture_manifest.json"
                )
                if capture_relative != expected_capture_relative:
                    problems.append(
                        f"{label} NCU receipt capture manifest path is invalid"
                    )
                    continue
            else:
                role_relative = PurePosixPath(internal_relative)
                if (
                    role_relative.parent != PurePosixPath(".")
                    and (
                        not role_relative.parts
                        or role_relative.parts[0] != "labels"
                    )
                ):
                    problems.append(
                        f"{label} NCU receipt {role} path is invalid"
                    )
                    continue
            bound_path = receipt_cell_dir / PurePosixPath(internal_relative)
            try:
                bound_relative = bound_path.relative_to(output_root).as_posix()
            except ValueError:
                problems.append(
                    f"{label} NCU receipt {role} path escapes output root"
                )
                continue
            artifact = artifact_by_path.get(bound_relative)
            if artifact is None:
                problems.append(
                    f"{label} NCU receipt {role} is absent from DONE artifacts"
                )
                continue
            if artifact.get("artifact_kind") != role_kinds[role]:
                problems.append(
                    f"{label} NCU receipt {role} artifact kind differs from DONE"
                )
            if artifact.get("sha256") != binding.get("sha256"):
                problems.append(
                    f"{label} NCU receipt {role} hash differs from DONE"
                )
            if artifact.get("size_bytes") != binding.get("size_bytes"):
                problems.append(
                    f"{label} NCU receipt {role} size differs from DONE"
                )
            try:
                external_bindings.append(ncu_collection_receipt.artifact_binding(
                    role=role,
                    relative_path=internal_relative,
                    sha256=artifact.get("sha256"),
                    size_bytes=artifact.get("size_bytes"),
                ))
            except ncu_collection_receipt.CollectionReceiptError as exc:
                problems.append(
                    f"{label} NCU receipt {role} external binding is invalid: {exc}"
                )
            if role == "ncu_report":
                report_sha256 = artifact.get("sha256")
            if role == "ncu_launch_observation":
                observation = _load_strict_json(
                    output_root / bound_relative,
                    f"{label}.ncu_launch_observation",
                    problems,
                )
                if observation is not None:
                    if (
                        observation.get("observation_sha256")
                        != receipt.get("launch_observation_sha256")
                    ):
                        problems.append(
                            f"{label} NCU receipt launch observation seal differs"
                        )
                    if observation.get("report_sha256") != report_sha256:
                        problems.append(
                            f"{label} launch observation report differs from NCU report"
                        )
                    try:
                        unsealed_observation = {
                            key: value for key, value in observation.items()
                            if key != "observation_sha256"
                        }
                        encoded_observation = json.dumps(
                            unsealed_observation,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                        ).encode("utf-8")
                        if (
                            hashlib.sha256(encoded_observation).hexdigest()
                            != observation.get("observation_sha256")
                        ):
                            problems.append(
                                f"{label} launch observation seal is not canonical"
                            )
                    except (TypeError, UnicodeError, ValueError) as exc:
                        problems.append(
                            f"{label} launch observation is not canonical: {exc}"
                        )
                launch_observation = observation
            if role == "label_capture_manifest":
                capture = _load_strict_json(
                    output_root / bound_relative,
                    f"{label}.label_capture_manifest",
                    problems,
                )
                if capture is not None and capture.get("command") != receipt.get(
                    "command"
                ):
                    problems.append(
                        f"{label} NCU receipt command differs from capture manifest"
                    )
                capture_manifest = capture
    if receipt.get("preflight_sha256") != manifest.get("preflight_sha256"):
        problems.append(f"{label} NCU receipt preflight identity differs")
    if receipt.get("status") != "pass" or receipt.get("exit_status") != 0:
        problems.append(f"{label} NCU receipt is not a successful collection")
    if receipt.get("ncu_identity") != expected_identity:
        problems.append(
            f"{label} NCU receipt identity differs from retained preflight"
        )
    run_manifests = [
        artifact for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("artifact_kind") == "label_run_manifest"
    ]
    sidecar_artifacts = [
        artifact for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("artifact_kind") == "label_sidecar"
    ]
    run: dict | None = None
    sidecar: dict | None = None
    if len(run_manifests) != 1:
        problems.append(
            f"{label} requires exactly one label_run_manifest artifact "
            f"(observed={len(run_manifests)})"
        )
    else:
        run_relative = run_manifests[0].get("relative_path")
        if _valid_relative_artifact_path(run_relative):
            run = _load_strict_json(
                output_root / run_relative,
                f"{label}.label_run_manifest",
                problems,
            )
            if run is not None and run.get("ncu_identity") != expected_identity:
                problems.append(
                    f"{label} label run manifest NCU identity differs from "
                    "retained preflight"
                )
    if len(sidecar_artifacts) != 1:
        problems.append(
            f"{label} requires exactly one label_sidecar artifact "
            f"(observed={len(sidecar_artifacts)})"
        )
    else:
        sidecar_relative = sidecar_artifacts[0].get("relative_path")
        if _valid_relative_artifact_path(sidecar_relative):
            sidecar_path = PurePosixPath(sidecar_relative)
            is_cell_root_sidecar = sidecar_path.parent.parts in expected_cell_prefixes
            is_label_tree_sidecar = any(
                sidecar_path.parts[:len(prefix) + 1]
                == (*prefix, "labels")
                for prefix in expected_cell_prefixes
            )
            if not (is_cell_root_sidecar or is_label_tree_sidecar):
                problems.append(
                    f"{label} label sidecar path is outside its DONE cell"
                )
            else:
                sidecar = _load_strict_json(
                    output_root / sidecar_relative,
                    f"{label}.label_sidecar",
                    problems,
                )
                if sidecar is not None and "schema_version" in sidecar:
                    if sidecar.get("schema_version") != 5:
                        problems.append(
                            f"{label} label sidecar schema_version is invalid"
                        )
                    else:
                        try:
                            import jsonschema

                            jsonschema.validate(
                                sidecar,
                                json.loads(
                                    LOCAL_LABEL_SIDECAR_SCHEMA_PATH.read_text()
                                ),
                            )
                        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                            problems.append(
                                f"{label} label sidecar schema is unavailable: {exc}"
                            )
                        except jsonschema.ValidationError as exc:
                            problems.append(
                                f"{label} label sidecar schema is invalid: {exc.message}"
                            )
    if run is not None:
        run_required_types = {
            "schema_version": int,
            "source_commit": str,
            "binary_sha256": str,
            "ncu_identity": dict,
            "arch": str,
            "slug": str,
            "shape_point": str,
            "repeat_index": int,
            "launch_index": int,
        }
        for field, expected_type in run_required_types.items():
            if type(run.get(field)) is not expected_type:
                problems.append(
                    f"{label} NCU run manifest {field} is invalid"
                )
            elif field == "binary_sha256" and _SHA64_RE.fullmatch(
                run[field]
            ) is None:
                problems.append(
                    f"{label} NCU run manifest binary_sha256 is invalid"
                )
        if run.get("schema_version") != 1:
            problems.append(
                f"{label} NCU run manifest schema_version is invalid"
            )
        run_expected = {
            "source_commit": manifest.get("source_commit_pre"),
            "arch": manifest.get("arch"),
            "slug": slug,
            "repeat_index": 0,
        }
        if expected_binary_sha256 is not None:
            run_expected["binary_sha256"] = expected_binary_sha256
        if expected_shape_point is not None:
            run_expected["shape_point"] = expected_shape_point
        for field, expected_value in run_expected.items():
            if (
                type(run.get(field)) is not type(expected_value)
                or run.get(field) != expected_value
            ):
                problems.append(
                    f"{label} NCU run manifest {field} differs from DONE identity"
                )
        if sidecar is not None:
            sidecar_required_types = {
                "device": str,
                "slug": str,
                "shape_point": str,
                "repeat_index": int,
                "launch_index": int,
                "binary_sha256": str,
            }
            for field, expected_type in sidecar_required_types.items():
                if type(sidecar.get(field)) is not expected_type:
                    problems.append(
                        f"{label} label sidecar {field} is invalid"
                    )
                elif field == "binary_sha256" and _SHA64_RE.fullmatch(
                    sidecar[field]
                ) is None:
                    problems.append(
                        f"{label} label sidecar binary_sha256 is invalid"
                    )
            sidecar_expected = {
                "device": run.get("arch"),
                "slug": run.get("slug"),
                "shape_point": run.get("shape_point"),
                "repeat_index": run.get("repeat_index"),
                "launch_index": run.get("launch_index"),
                "binary_sha256": run.get("binary_sha256"),
            }
            for field, expected_value in sidecar_expected.items():
                if (
                    type(sidecar.get(field)) is not type(expected_value)
                    or sidecar.get(field) != expected_value
                ):
                    problems.append(
                        f"{label} label sidecar {field} differs from NCU run"
                    )
        if capture_manifest is not None:
            capture_required_types = {
                "schema_version": int,
                "identity": dict,
                "source_commit": str,
                "binary_sha256": str,
                "command": list,
                "collection_environment": dict,
                "working_directory": str,
                "requested_launch_index": int,
            }
            for field, expected_type in capture_required_types.items():
                if type(capture_manifest.get(field)) is not expected_type:
                    problems.append(
                        f"{label} capture manifest {field} is invalid"
                    )
                elif field == "binary_sha256" and _SHA64_RE.fullmatch(
                    capture_manifest[field]
                ) is None:
                    problems.append(
                        f"{label} capture manifest binary_sha256 is invalid"
                    )
            if capture_manifest.get("schema_version") != 1:
                problems.append(
                    f"{label} capture manifest schema_version is invalid"
                )
            capture_expected = {
                "identity": {"roster_source": source, "slug": slug},
                "source_commit": run.get("source_commit"),
                "binary_sha256": run.get("binary_sha256"),
                "requested_launch_index": run.get("launch_index"),
            }
            for field, expected_value in capture_expected.items():
                if (
                    type(capture_manifest.get(field)) is not type(expected_value)
                    or capture_manifest.get(field) != expected_value
                ):
                    problems.append(
                        f"{label} capture manifest {field} differs from DONE identity"
                    )
    if observation is not None and run is not None:
        try:
            parse_ncu_metrics.validate_launch_observation_document(
                observation,
                requested_launch_skip=run.get("launch_index"),
                requested_launch_count=1,
                report_sha256=report_sha256,
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            problems.append(
                f"{label} NCU launch observation contract differs: {exc}"
            )
        else:
            problems.extend(_ncu_launch_observation_semantic_problems(
                observation, label=label,
            ))
    if (
        len(external_bindings) == 4
        and launch_observation is not None
        and capture_manifest is not None
    ):
        raw_environment = capture_manifest.get("collection_environment")
        raw_working_directory = capture_manifest.get("working_directory")
        if (
            isinstance(raw_environment, dict)
            and isinstance(raw_working_directory, str)
        ):
            try:
                ncu_collection_receipt.validate_receipt(
                    receipt,
                    expected_preflight_sha256=manifest.get("preflight_sha256"),
                    expected_ncu_identity=expected,
                    expected_command=capture_manifest.get("command"),
                    expected_environment=(
                        ncu_action_executor.NcuCollectionEnvironment(
                            raw_environment
                        )
                    ),
                    expected_working_directory=Path(raw_working_directory),
                    expected_exit_status=0,
                    expected_artifacts=external_bindings,
                    expected_launch_observation_sha256=(
                        launch_observation.get("observation_sha256")
                    ),
                )
            except (
                ncu_collection_receipt.CollectionReceiptError,
                ncu_action_executor.NcuActionExecutorError,
                TypeError,
                UnicodeError,
                ValueError,
            ) as exc:
                problems.append(
                    f"{label} NCU receipt external bindings differ: {exc}"
                )
        else:
            problems.append(
                f"{label} capture manifest lacks collection environment authority"
            )
    return problems


def _verify_referenced_files(
    manifest: dict,
    repo_root: Path,
    verification_output_root: Path | str | None = None,
) -> list[str]:
    problems: list[str] = []
    loaded: dict[str, dict] = {}
    input_paths = manifest.get("input_paths")
    if not isinstance(input_paths, dict):
        return ["input_paths is not an object"]
    source_commit = manifest.get("source_commit_pre")
    for label, hash_field in _INPUT_HASH_FIELDS.items():
        raw = input_paths.get(label)
        if not isinstance(raw, str) or not raw:
            continue
        try:
            path = _resolve_input_path(raw, repo_root)
        except DoneManifestError as exc:
            problems.append(str(exc))
            continue
        path = _historical_bootstrap_repo_reference(
            path, manifest=manifest, repo_root=repo_root,
        )
        payload, actual_sha, origin = _reference_bytes_at_seal(
            path, expected_sha256=manifest.get(hash_field),
            repo_root=repo_root, source_commit=source_commit,
        )
        if payload is None:
            problems.append(f"cannot read {label} {path} at current or sealed source")
            continue
        if actual_sha != manifest.get(hash_field):
            problems.append(
                f"stale {label} hash: manifest={manifest.get(hash_field)!r} "
                f"actual={actual_sha} origin={origin}")
        value = _load_json_bytes(payload, path, label, problems)
        if value is not None:
            loaded[label] = value

    pre = manifest.get("source_commit_pre")
    arch = manifest.get("arch")
    pins_sha = manifest.get("pins_sha256")
    pins = loaded.get("pins")
    if not _is_historical_bootstrap(manifest) and isinstance(pins, dict):
        toolchain = pins.get("toolchain")
        pinned_ncu_version = (
            toolchain.get("ncu_version_prefix")
            if isinstance(toolchain, dict) else None
        )
        done_tool_versions = manifest.get("tool_versions")
        done_ncu_version = (
            done_tool_versions.get("ncu")
            if isinstance(done_tool_versions, dict) else None
        )
        try:
            pinned_ncu_version = (
                capture_toolchain_contract.normalized_ncu_version_prefix(
                    pinned_ncu_version
                )
            )
            done_ncu_version = (
                capture_toolchain_contract.normalized_ncu_version_prefix(
                    done_ncu_version
                )
            )
        except capture_toolchain_contract.CaptureToolchainContractError:
            problems.append("DONE or pins NCU version prefix is invalid")
        else:
            if pinned_ncu_version != done_ncu_version:
                problems.append(
                    "DONE NCU tool version differs from sealed pins"
                )
    preflight = loaded.get("preflight")
    if preflight is not None:
        if preflight.get("capture_admitted") is not True:
            problems.append("preflight did not admit capture")
        if preflight.get("working_tree_clean") is not True:
            problems.append("preflight working_tree_clean is not true")
        if preflight.get("git_head") != pre:
            problems.append(
                f"preflight git_head={preflight.get('git_head')!r} != source={pre!r}")
        if preflight.get("pins_sha256") != pins_sha:
            problems.append("preflight pins_sha256 does not match DONE pins_sha256")
        if preflight.get("blockers") not in (None, []):
            problems.append(f"preflight has blockers: {preflight.get('blockers')!r}")
        if arch is not None and preflight.get("arch") != arch:
            problems.append("preflight arch does not match DONE arch")
        if not _is_historical_bootstrap(manifest):
            admission = manifest.get("admission")
            generated = (
                admission.get("preflight_generated_at_utc")
                if isinstance(admission, dict) else None
            )
            if generated != preflight.get("generated_at_utc"):
                problems.append(
                    "DONE admission preflight_generated_at_utc does not match "
                    "the sealed preflight"
                )

    authority = loaded.get("roster_authority")
    member_rows = manifest.get("chunk_membership")
    member_map: dict[tuple[str, str], dict] = {}
    if isinstance(member_rows, list):
        for i, row in enumerate(member_rows):
            identity = _identity(row, f"chunk_membership[{i}]", problems)
            if identity is not None and isinstance(row, dict):
                member_map[identity] = row
    if authority is not None:
        if authority.get("source_commit") != pre:
            problems.append("roster authority source_commit does not match DONE source")
        authority_rows = authority.get("cells")
        if not isinstance(authority_rows, list):
            problems.append("roster authority cells is not an array")
        else:
            indexed: dict[tuple[str, str], dict] = {}
            duplicates: set[tuple[str, str]] = set()
            for i, row in enumerate(authority_rows):
                identity = _identity(row, f"roster_authority.cells[{i}]", problems)
                if identity is None:
                    continue
                if identity in indexed:
                    duplicates.add(identity)
                indexed[identity] = row
            if duplicates:
                problems.append(f"roster authority has duplicate identities: {sorted(duplicates)}")
            for identity, member in sorted(member_map.items()):
                row = indexed.get(identity)
                if row is None:
                    problems.append(f"chunk identity absent from roster authority: {identity}")
                    continue
                if row.get("resolution_status") != "runnable":
                    problems.append(
                        f"chunk authority identity is not runnable: {identity} "
                        f"({row.get('resolution_status')!r})")
                actual_row_sha = authority_cell_sha256(row)
                if member.get("authority_cell_sha256") != actual_row_sha:
                    problems.append(
                        f"authority row hash mismatch for {identity}: "
                        f"manifest={member.get('authority_cell_sha256')!r} "
                        f"actual={actual_row_sha}")

    matrix = loaded.get("capture_matrix")
    if matrix is not None:
        matrix_commit = matrix.get("source_commit", matrix.get("git_head"))
        if matrix_commit != pre:
            problems.append(
                f"capture matrix source commit={matrix_commit!r} != DONE source={pre!r}")
        if arch is not None and matrix.get("arch") != arch:
            problems.append("capture matrix arch does not match DONE arch")
        if (
            not _is_historical_bootstrap(manifest)
            and manifest.get("matrix_seal_sha256")
            != matrix.get("matrix_seal_sha256")
        ):
            problems.append("DONE matrix_seal_sha256 does not match capture matrix")
        if not _is_historical_bootstrap(manifest):
            matrix_unsealed = {
                key: value for key, value in matrix.items()
                if key != "matrix_seal_sha256"
            }
            if matrix.get("matrix_seal_sha256") != _canonical_sha256(matrix_unsealed):
                problems.append("capture matrix seal does not rederive")
        matrix_ids = _matrix_chunk_identities(
            matrix, manifest.get("chunk_id"), problems,
        )
        manifest_ids = set(member_map)
        missing = sorted(matrix_ids - manifest_ids)
        extra = sorted(manifest_ids - matrix_ids)
        if missing:
            problems.append(f"DONE is missing capture-matrix chunk cells: {missing}")
        if extra:
            problems.append(f"DONE contains cells outside capture-matrix chunk: {extra}")

        # Production matrices expose full rows as well as compact chunk lists.
        # Bind each selected identity and authority hash to that row when the
        # richer table is present; a self-consistent but invented chunk list is
        # not sufficient resume authority.
        matrix_cells = matrix.get("cells")
        if matrix_cells is not None:
            if not isinstance(matrix_cells, list):
                problems.append("capture matrix cells is not an array")
            else:
                indexed_matrix: dict[tuple[str, str], dict] = {}
                duplicate_matrix: set[tuple[str, str]] = set()
                for i, row in enumerate(matrix_cells):
                    identity_row = row.get("identity") if isinstance(row, dict) else None
                    identity = _identity(
                        identity_row, f"capture_matrix.cells[{i}].identity", problems,
                    )
                    if identity is None or not isinstance(row, dict):
                        continue
                    if identity in indexed_matrix:
                        duplicate_matrix.add(identity)
                    indexed_matrix[identity] = row
                if duplicate_matrix:
                    problems.append(
                        "capture matrix cells has duplicate identities: "
                        f"{sorted(duplicate_matrix)}")
                for identity in sorted(matrix_ids):
                    row = indexed_matrix.get(identity)
                    if row is None:
                        problems.append(
                            f"chunk identity absent from capture matrix cells: {identity}")
                        continue
                    if row.get("disposition") != "runnable":
                        problems.append(
                            f"capture matrix chunk identity is not runnable: {identity}")
                    if (row.get("authority_cell_sha256")
                            != member_map.get(identity, {}).get("authority_cell_sha256")):
                        problems.append(
                            f"capture matrix authority hash mismatch for {identity}")

        matrix_authority = matrix.get("authority")
        if (isinstance(matrix_authority, dict)
                and matrix_authority.get("sha256")
                != manifest.get("roster_authority_sha256")):
            problems.append(
                "capture matrix authority pin does not match DONE authority hash")
        matrix_pins = matrix.get("input_pins")
        if isinstance(matrix_pins, dict):
            for section, done_field in (
                ("protocol_pins", "pins_sha256"),
                ("aggregate_retention_budget", "trace_budget_sha256"),
            ):
                pin = matrix_pins.get(section)
                if (isinstance(pin, dict)
                        and pin.get("sha256") != manifest.get(done_field)):
                    problems.append(
                        f"capture matrix {section} pin does not match DONE {done_field}")
            required_pin_sections = [
                "protocol_pins", "aggregate_retention_budget",
                "per_cell_trace_budgets", "metric_registry",
            ]
            # SOL provenance was added after the first immutable bootstrap.
            # Rehash it when a matrix pins it; absence remains valid for the
            # older sealed matrix and is rejected by the current matrix schema.
            if "sol_shrink_provenance" in matrix_pins:
                required_pin_sections.append("sol_shrink_provenance")
            if "sol_runnability" in matrix_pins:
                required_pin_sections.append("sol_runnability")
            if SOL_NATIVE_ENV_PIN_SECTION in matrix_pins:
                required_pin_sections.append(SOL_NATIVE_ENV_PIN_SECTION)
            for section in required_pin_sections:
                pin = matrix_pins.get(section)
                if not isinstance(pin, dict):
                    problems.append(f"capture matrix lacks input_pins.{section}")
                    continue
                raw_path = pin.get("path")
                expected_sha = pin.get("sha256")
                if not isinstance(raw_path, str) or not raw_path:
                    problems.append(f"capture matrix {section} path is absent")
                    continue
                try:
                    pinned_path = _resolve_input_path(raw_path, repo_root)
                except DoneManifestError as exc:
                    problems.append(
                        f"cannot rehash capture matrix {section} input: {exc}")
                    continue
                _, observed_sha, origin = _reference_bytes_at_seal(
                    pinned_path, expected_sha256=expected_sha,
                    repo_root=repo_root, source_commit=pre,
                )
                if observed_sha is None:
                    problems.append(
                        f"cannot rehash capture matrix {section} input at "
                        f"current or sealed source: {pinned_path}"
                    )
                    continue
                pinned_path = _historical_bootstrap_repo_reference(
                    pinned_path, manifest=manifest, repo_root=repo_root,
                )
                if observed_sha != expected_sha:
                    problems.append(
                        f"capture matrix {section} input hash is stale: "
                        f"matrix={expected_sha!r} actual={observed_sha} "
                        f"origin={origin}")

    output_raw = (manifest.get("output_root") if verification_output_root is None
                  else str(verification_output_root))
    if isinstance(output_raw, str) and Path(output_raw).is_absolute():
        requested_output_root = Path(output_raw)
        try:
            output_root = requested_output_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            problems.append(f"cannot resolve published output_root: {exc}")
            return problems
        if requested_output_root.is_symlink():
            problems.append("published output_root is a symlink")
        if "superseded" in output_root.parts:
            problems.append(
                f"published output_root resolves beneath superseded/: {output_root}"
            )
        if PREPARATION_FAILURE_ROOT_NAME in output_root.parts:
            problems.append(
                "PREPARATION_ARCHIVE_LEAK: published output_root resolves beneath "
                f"{PREPARATION_FAILURE_ROOT_NAME}/: {output_root}"
            )
        if CAPTURE_DEFECT_ROOT_NAME in output_root.parts:
            problems.append(
                "CAPTURE_DEFECT_ARCHIVE_LEAK: published output_root resolves beneath "
                f"{CAPTURE_DEFECT_ROOT_NAME}/: {output_root}"
            )
        raw_results = manifest.get("cell_results")
        safe_results = raw_results if isinstance(raw_results, list) else []
        admission_runs: dict[str, datetime] = {}
        if not _is_historical_bootstrap(manifest):
            admission_parse_problems: list[str] = []
            _, admission_runs = _normalize_admission(
                manifest.get("admission"), problems=admission_parse_problems,
            )
        listed_paths: set[str] = set()
        for i, row in enumerate(safe_results):
            if not isinstance(row, dict):
                continue
            raw_artifacts = row.get("artifacts")
            safe_artifacts = raw_artifacts if isinstance(raw_artifacts, list) else []
            for j, artifact in enumerate(safe_artifacts):
                if not isinstance(artifact, dict):
                    continue
                rel = artifact.get("relative_path")
                label = f"cell_results[{i}].artifacts[{j}]"
                if not _valid_relative_artifact_path(rel):
                    continue
                listed_paths.add(rel)
                candidate = output_root / rel
                try:
                    resolved = candidate.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    problems.append(f"cannot resolve {label} {candidate}: {exc}")
                    continue
                try:
                    resolved.relative_to(output_root)
                except ValueError:
                    problems.append(
                        f"{label} escapes output_root via path/symlink: {rel!r}")
                    continue
                if PREPARATION_FAILURE_ROOT_NAME in resolved.parts:
                    problems.append(
                        "PREPARATION_ARCHIVE_LEAK: "
                        f"{label} resolves beneath {PREPARATION_FAILURE_ROOT_NAME}/: "
                        f"{resolved}"
                    )
                    continue
                if not resolved.is_file():
                    problems.append(f"{label} is not a regular file: {resolved}")
                    continue
                try:
                    actual_size = resolved.stat().st_size
                    actual_sha = _sha256_file(resolved)
                except OSError as exc:
                    problems.append(f"cannot hash {label} {resolved}: {exc}")
                    continue
                if actual_size != artifact.get("size_bytes"):
                    problems.append(
                        f"artifact size mismatch for {rel!r}: "
                        f"manifest={artifact.get('size_bytes')!r} actual={actual_size}")
                if actual_sha != artifact.get("sha256"):
                    problems.append(
                        f"artifact sha256 mismatch for {rel!r}: "
                        f"manifest={artifact.get('sha256')!r} actual={actual_sha}")
            if not _is_historical_bootstrap(manifest):
                problems.extend(_verify_ncu_collection_receipt(
                    row,
                    manifest=manifest,
                    preflight=preflight,
                    matrix=matrix,
                    output_root=output_root,
                    label=f"cell_results[{i}]",
                ))
                problems.extend(_verify_cell_capture_provenance(
                    row, manifest=manifest, output_root=output_root,
                    admission_runs=admission_runs,
                    label=f"cell_results[{i}]",
                ))
                problems.extend(_verify_trace_manifest_launch_contract(
                    row, matrix=matrix, output_root=output_root,
                    label=f"cell_results[{i}]",
                ))
            if any(
                isinstance(artifact, dict)
                and artifact.get("artifact_kind") == "trace_route_record"
                for artifact in safe_artifacts
            ):
                problems.extend(_verify_trace_route_evidence(
                    row, manifest=manifest, matrix=matrix,
                    preflight=preflight, output_root=output_root,
                    label=f"cell_results[{i}]",
                ))
            if (
                row.get("disposition") == "failed"
                and row.get("typed_reason") == "trace_unsupported"
            ):
                problems.extend(_verify_trace_unsupported_evidence(
                    row, output_root=output_root, label=f"cell_results[{i}]",
                ))
        problems.extend(verify_manifest_preparation_failure_closure(manifest))
        problems.extend(verify_manifest_capture_defect_closure(manifest))
        if not _is_historical_bootstrap(manifest):
            for index, superseded_root in enumerate(sorted(
                _manifest_supersession_roots(manifest, safe_results),
                key=str,
            )):
                root_problems = _verify_supersession_root(
                    superseded_root, manifest=manifest, rows=safe_results,
                    label=f"supersession_root[{index}]",
                )
                problems.extend(
                    f"SUPERSESSION_LEDGER_INCONSISTENT: {problem}"
                    for problem in root_problems
                )
        if matrix is not None:
            problems.extend(verify_manifest_sol_native_env_attestations(
                manifest, matrix, output_root=output_root,
                repo_root=repo_root, source_commit=pre,
            ))
            problems.extend(verify_manifest_sol_definition_attestations(
                manifest,
                matrix,
                output_root=output_root,
                repo_root=repo_root,
                source_commit=pre,
            ))
        actual_paths: set[str] = set()
        if not output_root.is_dir():
            problems.append(f"output_root is not a directory: {output_root}")
        else:
            for candidate in sorted(output_root.rglob("*")):
                rel = candidate.relative_to(output_root).as_posix()
                if candidate.is_symlink():
                    problems.append(f"output_root contains symlink: {rel!r}")
                    continue
                if candidate.is_file() and rel != "DONE.json":
                    actual_paths.add(rel)
        unlisted = sorted(actual_paths - listed_paths)
        missing = sorted(listed_paths - actual_paths)
        if unlisted:
            problems.append(f"output_root contains unlisted artifacts: {unlisted}")
        if missing:
            problems.append(f"sealed artifacts absent from output_root inventory: {missing}")
    return problems


def verify_seal(
    manifest: dict,
    *,
    repo_root: Path | str = REPO_ROOT,
    rehash_references: bool = True,
    require_current_head: bool = True,
    require_clean_tree: bool = True,
    verification_output_root: Path | str | None = None,
) -> list[str]:
    """Return all reasons this DONE manifest is unsafe for resume.

    The strict defaults are intentional.  ``rehash_references=False`` is useful
    only to inspect an archived manifest structurally; it is not resume proof.
    Likewise, the git gates may be disabled by offline auditors but never by the
    emitter or campaign runner.  ``verification_output_root`` is the one narrow
    staging exception: a runner may verify bytes beneath an absolute staging
    directory before atomically renaming it to the manifest's final
    ``output_root``.  Resume verification must omit the override.
    """
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    problems = _schema_problems(manifest)
    problems.extend(_semantic_problems(manifest))
    root = Path(repo_root)
    if require_current_head:
        actual_head = current_source_commit(root)
        if actual_head != manifest.get("source_commit_pre"):
            problems.append(
                f"current git HEAD={actual_head!r} != sealed source commit="
                f"{manifest.get('source_commit_pre')!r}")
    if require_clean_tree and not git_tree_clean(root):
        problems.append("current git working tree is dirty or unavailable")
    if rehash_references:
        if (verification_output_root is not None
                and not Path(verification_output_root).is_absolute()):
            problems.append("verification_output_root override must be absolute")
        else:
            problems.extend(_verify_referenced_files(
                manifest, root, verification_output_root,
            ))
    return problems


def _assert_schema_valid(manifest: dict) -> None:
    import jsonschema  # lazy dependency used by the campaign tooling
    try:
        jsonschema.validate(manifest, json.loads(SCHEMA_PATH.read_text()))
    except jsonschema.ValidationError as exc:
        raise DoneManifestError(
            f"emitted manifest fails its own schema: {exc.message}") from exc


def _normalize_legacy_failures(
    typed_failures: list[dict] | None, results: list[dict],
) -> list[dict]:
    derived = [
        {
            "roster_source": row["roster_source"],
            "slug": row["slug"],
            "typed_reason": row["typed_reason"],
            "detail": row["detail"],
        }
        for row in results if row["disposition"] == "failed"
    ]
    if typed_failures is None:
        return derived
    normalized: list[dict] = []
    for i, row in enumerate(typed_failures):
        if not isinstance(row, dict):
            raise DoneManifestError(f"typed_failures[{i}] is not an object")
        source = row.get("roster_source")
        slug = row.get("slug")
        if source is None:
            matches = [r for r in derived if r["slug"] == slug]
            if len(matches) == 1:
                source = matches[0]["roster_source"]
        normalized.append({
            "roster_source": source,
            "slug": slug,
            "typed_reason": row.get("typed_reason"),
            "detail": row.get("detail"),
        })
    normalized.sort(key=lambda row: (str(row["roster_source"]), str(row["slug"])))
    if normalized != derived:
        raise DoneManifestError(
            "legacy typed_failures does not exactly equal failed cell results")
    return derived


def build_done_manifest(
    *,
    command: str,
    source_commit_pre: str,
    source_commit_post: str,
    working_tree_clean: bool,
    tool_versions: dict,
    output_root: str,
    expected_cells: int,
    chunk_membership: list[dict],
    cell_results: list[dict],
    pins_sha256: str,
    roster_authority_sha256: str,
    trace_budget_sha256: str,
    capture_matrix_sha256: str,
    matrix_seal_sha256: str,
    preflight_sha256: str,
    admission: dict,
    input_paths: dict,
    chunk_id: str,
    source_head_at_seal: str | None = None,
    admitted: int | None = None,
    excluded: int | None = None,
    typed_failures: list[dict] | None = None,
    protocol_version: str | None = None,
    arch: str | None = None,
    generated_at_utc: str | None = None,
    repo_root: Path | str = REPO_ROOT,
    verification_output_root: Path | str | None = None,
) -> dict:
    """Build a schema-valid seal and strictly verify it before returning.

    ``admitted``, ``excluded`` and ``typed_failures`` remain accepted for old
    callers, but are assertions only: v2 derives them from exact cell results.
    """
    if not isinstance(command, str) or not command:
        raise DoneManifestError("command must be a nonempty string")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise DoneManifestError("chunk_id must be a nonempty string")
    if working_tree_clean is not True:
        raise DoneManifestError("working_tree_clean must be true; dirty chunks cannot seal")
    if not isinstance(input_paths, dict):
        raise DoneManifestError("input_paths must be an object")
    _validate_problems: list[str] = []
    _validate_hash("matrix_seal_sha256", matrix_seal_sha256, _validate_problems)
    _normalize_admission(admission, problems=_validate_problems)
    if _validate_problems:
        raise DoneManifestError("; ".join(_validate_problems))
    normalized_admission = {
        "preflight_generated_at_utc": admission["preflight_generated_at_utc"],
        "capture_invocations": [
            {
                "run_id": row["run_id"],
                "admitted_at_utc": row["admitted_at_utc"],
            }
            for row in admission["capture_invocations"]
        ],
    }

    head = source_commit_post if source_head_at_seal is None else source_head_at_seal
    for name, value in (
        ("source_commit_pre", source_commit_pre),
        ("source_commit_post", source_commit_post),
        ("source_head_at_seal", head),
    ):
        if not isinstance(value, str) or not _SHA40_RE.fullmatch(value):
            raise DoneManifestError(
                f"{name} is not a 40-hex commit sha: {value!r}")
    if not (source_commit_pre == source_commit_post == head):
        raise DoneManifestError(
            f"source/head drift: pre={source_commit_pre!r} "
            f"post={source_commit_post!r} head={head!r}")

    membership = _normalize_membership(chunk_membership)
    results = _normalize_results(
        cell_results,
        allow_e387_legacy_definition_attestation=(
            source_commit_pre
            == source_commit_post
            == head
            == SOL_E387_LEGACY_SOURCE_COMMIT
        ),
    )
    member_map = _identity_map(membership)
    result_map = _identity_map(results)
    if set(member_map) != set(result_map):
        missing = sorted(set(member_map) - set(result_map))
        extra = sorted(set(result_map) - set(member_map))
        raise DoneManifestError(
            f"exact membership/result mismatch: missing={missing}, extra={extra}")
    for identity in sorted(member_map):
        if (member_map[identity]["authority_cell_sha256"]
                != result_map[identity]["authority_cell_sha256"]):
            raise DoneManifestError(
                f"authority hash differs between membership/result for {identity}")
    if expected_cells != len(membership) or expected_cells < 1:
        raise DoneManifestError(
            f"expected_cells={expected_cells!r} != positive exact membership "
            f"size={len(membership)}")

    derived_counts = Counter(row["disposition"] for row in results)
    counts = {name: derived_counts[name] for name in CELL_DISPOSITIONS}
    if admitted is not None and admitted != counts["admitted"]:
        raise DoneManifestError(
            f"admitted={admitted!r} != derived admitted={counts['admitted']}")
    if excluded is not None and excluded != counts["excluded"]:
        raise DoneManifestError(
            f"excluded={excluded!r} != derived excluded={counts['excluded']}")
    failures = _normalize_legacy_failures(typed_failures, results)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "chunk_id": chunk_id,
        "chunk_membership_sha256": chunk_membership_sha256(membership),
        "arch": arch,
        "generated_at_utc": generated_at_utc,
        "command": command,
        "source_commit_pre": source_commit_pre,
        "source_commit_post": source_commit_post,
        "source_head_at_seal": head,
        "source_commit_stable": True,
        "working_tree_clean": True,
        "protocol_version": protocol_version,
        "tool_versions": tool_versions,
        "pins_sha256": pins_sha256,
        "roster_authority_sha256": roster_authority_sha256,
        "trace_budget_sha256": trace_budget_sha256,
        "capture_matrix_sha256": capture_matrix_sha256,
        "matrix_seal_sha256": matrix_seal_sha256,
        "preflight_sha256": preflight_sha256,
        "admission": normalized_admission,
        "input_paths": {name: str(input_paths.get(name, ""))
                        for name in _INPUT_HASH_FIELDS},
        "output_root": output_root,
        "expected_cells": expected_cells,
        "counts": counts,
        "chunk_membership": membership,
        "cell_results": results,
        "typed_failures": failures,
    }
    _assert_schema_valid(manifest)
    problems = verify_seal(
        manifest,
        repo_root=repo_root,
        verification_output_root=verification_output_root,
    )
    if problems:
        raise DoneManifestError("; ".join(problems))
    return manifest


def _tool_versions_from_pins(pins: dict) -> dict:
    tc = pins.get("toolchain", {})
    return {
        "ncu": tc.get("ncu_version_prefix"),
        "driver": tc.get("driver_version"),
        "nvcc": tc.get("nvcc_release"),
        "nvbit": tc.get("nvbit_version"),
    }


def _membership_from_files(matrix: dict, authority: dict, chunk_id: str) -> list[dict]:
    problems: list[str] = []
    identities = _matrix_chunk_identities(matrix, chunk_id, problems)
    indexed: dict[tuple[str, str], dict] = {}
    for i, row in enumerate(authority.get("cells") or []):
        identity = _identity(row, f"roster_authority.cells[{i}]", problems)
        if identity is not None:
            if identity in indexed:
                problems.append(f"roster authority has duplicate identity {identity}")
            indexed[identity] = row
    membership: list[dict] = []
    for identity in sorted(identities):
        row = indexed.get(identity)
        if row is None:
            problems.append(f"matrix identity absent from authority: {identity}")
            continue
        membership.append({
            "roster_source": identity[0],
            "slug": identity[1],
            "authority_cell_sha256": authority_cell_sha256(row),
        })
    if problems:
        raise DoneManifestError("; ".join(problems))
    return membership


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--command", required=True)
    ap.add_argument("--source-commit-pre", required=True)
    ap.add_argument("--source-commit-post", default=None)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--roster-authority", required=True)
    ap.add_argument("--trace-budget", required=True)
    ap.add_argument("--capture-matrix", required=True)
    ap.add_argument("--preflight", required=True)
    ap.add_argument(
        "--admission", required=True,
        help="JSON admission object from the persistent chunk run journal",
    )
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--cells", required=True,
                    help="JSON cell_results array with disposition + artifact hashes")
    ap.add_argument("--admitted", type=int, default=None,
                    help="legacy assertion; v2 derives this from --cells")
    ap.add_argument("--excluded", type=int, default=None,
                    help="legacy assertion; v2 derives this from --cells")
    ap.add_argument("--expected-cells", type=int, default=None,
                    help="legacy assertion; defaults to exact matrix chunk size")
    ap.add_argument("--typed-failures", default=None,
                    help="legacy JSON assertion; failures derive from --cells")
    ap.add_argument("--chunk-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    def _abs(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else REPO_ROOT / path

    try:
        paths = {
            "pins": _abs(args.pins).resolve(),
            "roster_authority": _abs(args.roster_authority).resolve(),
            "trace_budget": _abs(args.trace_budget).resolve(),
            "capture_matrix": _abs(args.capture_matrix).resolve(),
            "preflight": _abs(args.preflight).resolve(),
        }
        pins = json.loads(paths["pins"].read_text())
        authority = json.loads(paths["roster_authority"].read_text())
        matrix = json.loads(paths["capture_matrix"].read_text())
        results = json.loads(_abs(args.cells).read_text())
        admission = json.loads(_abs(args.admission).read_text())
        failures = (json.loads(_abs(args.typed_failures).read_text())
                    if args.typed_failures else None)
        membership = _membership_from_files(matrix, authority, args.chunk_id)
        member_hashes = {
            (row["roster_source"], row["slug"]): row["authority_cell_sha256"]
            for row in membership
        }
        if not isinstance(results, list):
            raise DoneManifestError("--cells must contain a JSON array")
        for row in results:
            if isinstance(row, dict) and "authority_cell_sha256" not in row:
                row["authority_cell_sha256"] = member_hashes.get(
                    (row.get("roster_source"), row.get("slug")))
        source_post = args.source_commit_post or current_source_commit(REPO_ROOT)
        expected = len(membership) if args.expected_cells is None else args.expected_cells
        manifest = build_done_manifest(
            command=args.command,
            source_commit_pre=args.source_commit_pre,
            source_commit_post=source_post,
            source_head_at_seal=current_source_commit(REPO_ROOT),
            working_tree_clean=git_tree_clean(REPO_ROOT),
            tool_versions=_tool_versions_from_pins(pins),
            output_root=str(_abs(args.output_root).resolve()),
            admitted=args.admitted,
            excluded=args.excluded,
            expected_cells=expected,
            chunk_membership=membership,
            cell_results=results,
            typed_failures=failures,
            pins_sha256=_sha256_file(paths["pins"]),
            roster_authority_sha256=_sha256_file(paths["roster_authority"]),
            trace_budget_sha256=_sha256_file(paths["trace_budget"]),
            capture_matrix_sha256=_sha256_file(paths["capture_matrix"]),
            matrix_seal_sha256=matrix.get("matrix_seal_sha256"),
            preflight_sha256=_sha256_file(paths["preflight"]),
            admission=admission,
            input_paths={name: str(path) for name, path in paths.items()},
            protocol_version=pins.get("measurement_protocol", {}).get("version"),
            arch=pins.get("arch"),
            chunk_id=args.chunk_id,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            repo_root=REPO_ROOT,
        )
    except DoneManifestError as exc:
        print(f"campaign_done_manifest: seal REFUSED — {exc}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        print(f"campaign_done_manifest: input error: {exc}", file=sys.stderr)
        return 2

    out = _abs(args.out)
    if out.exists():
        print(f"campaign_done_manifest: seal REFUSED — output exists: {out}",
              file=sys.stderr)
        return 1
    temp = out.with_name(f".{out.name}.tmp")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temp.replace(out)
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"campaign_done_manifest: write failed: {exc}", file=sys.stderr)
        return 2

    counts = manifest["counts"]
    print(
        "campaign_done_manifest: sealed chunk "
        f"admitted={counts['admitted']} excluded={counts['excluded']} "
        f"failed={counts['failed']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
