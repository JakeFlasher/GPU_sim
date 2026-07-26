"""Preflight-manifest validator for cloud measurement runs.

Consumed by downstream collection artifacts (the kernel-admission gate,
NVBit-trace canonical-export artifacts, evidence-pack composer) to
verify that a run was anchored to a valid preflight manifest before
any cloud budget was spent. A companion SHA-256 digest
(`compute_preflight_sha256`) is what downstream artifacts embed as
`preflight_manifest_sha256` on their own output so the validator can
prove they're anchored to a real manifest.

Host capability scope: this cloud manifest schema does not carry hardware-counter
or clock-control fields. Those capabilities belong to a separately authorized
host preflight rather than this schema. This validator therefore does not check
for counter-API or clock-lock fields; the manifest schema does not include those
slots.

Schema versioning: schema_version 4 is the current version. Adds
`capabilities.mps_active` and `capabilities.vgpu_active` boolean
fields on top of v3's `capabilities.cupti_activity_api_runtime`,
`capabilities.cc_mode`, and `capabilities.nvml_capability_matrix`.
The legacy `capabilities.cupti_activity_api_available` field is
preserved for backward compatibility as required provenance. Both
CUPTI capability fields are record-only under D8: the validator checks
their presence and type, and checks the runtime field against its closed
vocabulary, but neither capability value triggers a typed rejection.
Manifests at v3 remain admissible (the new v4 fields are skipped); at
v4 the new fields are required and boolean-typed (schema-correctness
check), but neither value triggers a typed rejection. `mps_active` and
`vgpu_active` are record-only
deployment-mode signals: the validator emits them on the manifest so
downstream artifacts can choose CUPTI attribution semantics
(per-client kernel attribution under MPS) and activity-kind fallback
(`_KERNEL` instead of source-correlated kinds when HES source-
correlation is unavailable under vGPU). MIG and CC remain hard
collection gates here; MPS and vGPU do not.

Error taxonomy (`PreflightErrorKind`):

- `PreflightMissing`
    Downstream artifact has no `preflight_manifest_sha256` key, or the
    manifest it references cannot be loaded.
- `ToolchainNotPinned`
    One or more required toolchain fields (driver_version,
    cuda_version, nvbit_build_hash, nvcc_version) is missing, empty,
    or the placeholder string "unknown".
- `TargetArchMismatch`
    The GPU's compute capability is outside the admitted local-development
    arches per CLAUDE.md "Two-Device Local Development Model": sm_86
    (CC 8.6, RTX 3090 Ti, Ampere desktop) AND the sm_120 family (CC
    12.x, RTX 5060 laptop, Blackwell consumer laptop). Server arches
    (sm_80 datacenter / sm_89 / sm_90a / sm_100 datacenter Blackwell)
    and any other compute capability are rejected here.
- `SchemaInvalid`
    The manifest is not a JSON object, lacks required top-level
    sections, or a typed sub-field is the wrong type. Usually
    indicates a build bug in the preflight script.
- `ClusterAttributeUnavailable`
    The cluster-attribute probe could not demonstrate that
    `cudaFuncAttributeNonPortableClusterSizeAllowed` is available on
    the target host. Distinct from a shape-level schema error.
- `CuptiActivityUnavailable`
    Historical vocabulary (D8); no longer emitted as of Stage 4.
    Preserved to interpret historical manifests, logs, and ledgers.
- `CuptiActivityRuntimeUnavailable` (schema v3)
    Historical vocabulary (D8); no longer emitted as of Stage 4.
    Preserved to interpret historical manifests, logs, and ledgers.
- `MigSliceNotValidForCollectionEvidence` (schema v3)
    The GPU is partitioned into MIG slices (`gpu.mig_mode == "Enabled"`).
    Whole-GPU NCU provenance is required for the simulator's
    `gpc__cycles_elapsed.max` validation target, so collection evidence
    from a MIG slice is not admissible.
- `ConfidentialComputingNotSupportedForCollection` (schema v3)
    Confidential Computing mode is on without DevTools (`cc_mode == "on"`).
    NCU hardware-counter collection is unsupported in this regime, so
    collection evidence is not admissible.
Note (schema v4 mps_active / vgpu_active):
    Both boolean fields are record-only deployment-mode signals.
    The validator checks they are present and boolean-typed
    (schema-correctness); either `true` value validates clean.
    Downstream consumers read the fields to choose CUPTI attribution
    semantics — per-client kernel attribution is the only attribution
    that survives an active MPS server, and `_KERNEL` is the
    activity-kind fallback when HES source-correlation is
    unavailable under vGPU. MIG and CC remain hard collection gates
    here; MPS and vGPU do not.

Usage:

    from preflight_manifest import (
        load_manifest,
        validate_preflight_manifest,
        compute_preflight_sha256,
        PreflightError,
        PreflightErrorKind,
    )

    manifest = load_manifest(path)
    result = validate_preflight_manifest(manifest)
    if not result.ok:
        raise SystemExit(f"{result.error.kind.value}: {result.error.message}")
    sha = compute_preflight_sha256(manifest)
"""

from __future__ import annotations

import copy
import enum
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TOOLCHAIN_FIELDS: tuple[str, ...] = (
    "driver_version",
    "cuda_version",
    "nvcc_version",
    "nvbit_build_hash",
)

UNPINNED_SENTINELS: frozenset[str] = frozenset({"", "unknown", "NOT_INSTALLED"})

SM120_CAP_PREFIX: str = "12."

# Per CLAUDE.md "Two-Device Local Development Model" the validator
# admits both local-development arches: sm_86 (RTX 3090 Ti, CC 8.6,
# Ampere desktop) AND sm_120 family (RTX 5060 laptop, CC 12.x,
# Blackwell consumer laptop). The validator rejects server arches
# (sm_80 / sm_89 / sm_90a / server sm_120) and any other compute
# capability via TargetArchMismatch.
ADMITTED_CAP_PREFIXES: tuple[str, ...] = ("8.6", SM120_CAP_PREFIX)

# When the cap prefix is sm_120 family, the gate also rejects server /
# workstation Blackwell SKUs whose device name contains any of these
# case-insensitive substrings. The canonical workstation marker is
# "RTX PRO 6000 Blackwell Workstation"; the consumer laptop SKU
# (e.g. "GeForce RTX 5060 Laptop GPU") does not contain any of these
# substrings and continues to be admitted. This narrows the broad
# `12.*` cap-prefix admission to local-development consumer Blackwell
# only, matching the policy prose at lines 129-131 above and
# CLAUDE.md "Two-Device Local Development Model".
REJECTED_SM120_NAME_SUBSTRINGS: tuple[str, ...] = ("pro 6000",)

CLUSTER_ATTR_UNAVAILABLE_SENTINELS: frozenset[str] = frozenset({
    "",
    "unavailable",
    "compile_failed",
    "runtime_failed",
    "probe_failed",
    "mock_input_missing",
    "unknown",
})

# Historical legacy-availability vocabulary retained for interpreting old
# manifests, logs, and ledgers. Its values are non-gating under D8.
CUPTI_ACTIVITY_UNAVAILABLE_SENTINELS: frozenset[str] = frozenset({
    "",
    "unavailable",
    "header_missing",
    "library_missing",
    "probe_failed",
    "mock_input_missing",
    "unknown",
})

# The cupti_activity_api_runtime field is recorded provenance with a
# closed vocabulary. The seven values below are the only admissible
# strings; any other value rejects as SchemaInvalid. Every recognized
# value is non-gating under D8.
CUPTI_ACTIVITY_RUNTIME_VALID_VALUES: frozenset[str] = frozenset({
    "available",
    "compile_failed",
    "link_failed",
    "runtime_failed",
    "insufficient_privilege",
    "cc_unsupported",
    "mock_input_missing",
})

# Historical unavailable vocabulary retained for interpreting old
# manifests, logs, and ledgers. It is not an active admission gate.
CUPTI_ACTIVITY_RUNTIME_UNAVAILABLE_SENTINELS: frozenset[str] = (
    CUPTI_ACTIVITY_RUNTIME_VALID_VALUES - frozenset({"available"})
)

CC_MODE_VALID_VALUES: frozenset[str] = frozenset({
    "off",
    "devtools",
    "on",
    "unsupported_query",
    "unknown",
})

NVML_CAPABILITY_MATRIX_VALID_VALUES: frozenset[str] = frozenset({
    "full",
    "partial_power_only",
    "partial_temp_only",
    "denied",
})

# MIG modes that are admissible for collection evidence. Anything
# else (notably "Enabled") fires `MigSliceNotValidForCollectionEvidence`.
# `[N/A]` (with square brackets) is the bracketed form `nvidia-smi`
# returns on GPUs that do not support MIG (e.g. consumer Ampere
# RTX 3090 Ti sm_86); `N/A` is the unbracketed form returned by
# some driver / firmware combinations. `Disabled` is for MIG-capable
# datacenter GPUs (H100, GB202) that are not currently partitioned.
MIG_MODE_ADMISSIBLE_VALUES: frozenset[str] = frozenset({
    "Disabled",
    "N/A",
    "[N/A]",
})

REQUIRED_PERMISSION_BOOLS: tuple[str, ...] = (
    "nvidia_smi_executable",
    "nvidia_dev_readable",
)


class PreflightErrorKind(enum.Enum):
    PreflightMissing = "PreflightMissing"
    ToolchainNotPinned = "ToolchainNotPinned"
    TargetArchMismatch = "TargetArchMismatch"
    SchemaInvalid = "SchemaInvalid"
    ClusterAttributeUnavailable = "ClusterAttributeUnavailable"
    # Historical vocabulary (D8); no longer emitted as of Stage 4.
    CuptiActivityUnavailable = "CuptiActivityUnavailable"
    # Historical vocabulary (D8); no longer emitted as of Stage 4.
    CuptiActivityRuntimeUnavailable = "CuptiActivityRuntimeUnavailable"
    MigSliceNotValidForCollectionEvidence = "MigSliceNotValidForCollectionEvidence"
    ConfidentialComputingNotSupportedForCollection = (
        "ConfidentialComputingNotSupportedForCollection"
    )


class PreflightError(Exception):
    def __init__(self, kind: PreflightErrorKind, message: str) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class PreflightValidationResult:
    ok: bool
    error: PreflightError | None


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise PreflightError(
            PreflightErrorKind.PreflightMissing,
            f"preflight manifest not found at path {p}",
        )
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise PreflightError(
            PreflightErrorKind.SchemaInvalid,
            f"preflight manifest at {p} is not valid JSON: {exc}",
        ) from exc


def _err(kind: PreflightErrorKind, message: str) -> PreflightValidationResult:
    return PreflightValidationResult(ok=False, error=PreflightError(kind, message))


def _require_nonempty_str(
    container: dict[str, Any],
    key: str,
    location: str,
) -> PreflightValidationResult | None:
    value = container.get(key)
    if not isinstance(value, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"{location} missing or not a string",
        )
    if not value.strip():
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"{location} is empty",
        )
    return None


def _require_dict_section(
    container: dict[str, Any],
    key: str,
    location: str,
) -> PreflightValidationResult | None:
    value = container.get(key)
    if value is None:
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"{location} missing required section",
        )
    if not isinstance(value, dict):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"{location} is not an object",
        )
    return None


def validate_preflight_manifest(manifest: Any) -> PreflightValidationResult:
    if not isinstance(manifest, dict):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "preflight manifest root is not a JSON object",
        )

    sv = manifest.get("schema_version")
    if not isinstance(sv, int) or isinstance(sv, bool) or sv < 3:
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "schema_version missing, not an int, or less than 3",
        )
    if (e := _require_nonempty_str(manifest, "timestamp", "timestamp")):
        return e
    if (e := _require_nonempty_str(manifest, "hostname", "hostname")):
        return e

    for section in ("gpu", "toolchain", "capabilities", "permissions"):
        if (e := _require_dict_section(manifest, section, section)):
            return e

    gpu = manifest["gpu"]
    for field in ("name", "uuid", "compute_capability", "mig_mode", "compute_mode"):
        if (e := _require_nonempty_str(gpu, field, f"gpu.{field}")):
            return e
    if (e := _require_dict_section(gpu, "clocks_mhz", "gpu.clocks_mhz")):
        return e
    clocks = gpu["clocks_mhz"]
    for field in ("sm_max", "sm_current", "mem_max"):
        if (e := _require_nonempty_str(clocks, field, f"gpu.clocks_mhz.{field}")):
            return e

    toolchain = manifest["toolchain"]
    for field in REQUIRED_TOOLCHAIN_FIELDS:
        value = toolchain.get(field)
        if not isinstance(value, str):
            return _err(
                PreflightErrorKind.ToolchainNotPinned,
                f"toolchain.{field} missing or not a string",
            )
        if value.strip() in UNPINNED_SENTINELS:
            return _err(
                PreflightErrorKind.ToolchainNotPinned,
                f"toolchain.{field} is not pinned (value={value!r})",
            )

    caps = manifest["capabilities"]
    cluster_attr = caps.get("cluster_attribute_nonportable_cluster_size_allowed")
    if not isinstance(cluster_attr, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "capabilities.cluster_attribute_nonportable_cluster_size_allowed "
            "missing or not a string",
        )
    if cluster_attr.strip() in CLUSTER_ATTR_UNAVAILABLE_SENTINELS:
        return _err(
            PreflightErrorKind.ClusterAttributeUnavailable,
            "cudaFuncAttributeNonPortableClusterSizeAllowed is not "
            f"demonstrably available (probe value={cluster_attr!r})",
        )
    # Runtime capability is required, string-typed recorded provenance
    # with a closed vocabulary. Any string outside
    # CUPTI_ACTIVITY_RUNTIME_VALID_VALUES rejects as SchemaInvalid;
    # every recognized value is non-gating under D8.
    cupti_runtime = caps.get("cupti_activity_api_runtime")
    if not isinstance(cupti_runtime, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "capabilities.cupti_activity_api_runtime missing or not a string",
        )
    cupti_runtime_normalized = cupti_runtime.strip()
    if cupti_runtime_normalized not in CUPTI_ACTIVITY_RUNTIME_VALID_VALUES:
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"capabilities.cupti_activity_api_runtime is not a recognized "
            f"value (got={cupti_runtime!r}); expected one of "
            f"{sorted(CUPTI_ACTIVITY_RUNTIME_VALID_VALUES)}",
        )
    # The legacy availability capability remains required, string-typed
    # recorded provenance. Its historical vocabulary is non-gating under
    # D8; the runtime field above remains the closed-vocabulary authority.
    cupti_activity = caps.get("cupti_activity_api_available")
    if not isinstance(cupti_activity, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "capabilities.cupti_activity_api_available missing or not a string",
        )
    cc_mode_value = caps.get("cc_mode")
    if not isinstance(cc_mode_value, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "capabilities.cc_mode missing or not a string",
        )
    cc_mode_normalized = cc_mode_value.strip()
    if cc_mode_normalized not in CC_MODE_VALID_VALUES:
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"capabilities.cc_mode is not a recognized value "
            f"(got={cc_mode_value!r}); expected one of "
            f"{sorted(CC_MODE_VALID_VALUES)}",
        )
    if cc_mode_normalized == "on":
        return _err(
            PreflightErrorKind.ConfidentialComputingNotSupportedForCollection,
            "Confidential Computing mode is enabled (cc_mode='on') without "
            "DevTools; NCU hardware counter collection is unsupported in this "
            "regime, so silicon collection evidence cannot be admitted (D8: "
            "timing ground truth is NCU).",
        )

    nvml_matrix = caps.get("nvml_capability_matrix")
    if not isinstance(nvml_matrix, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "capabilities.nvml_capability_matrix missing or not a string",
        )
    if nvml_matrix.strip() not in NVML_CAPABILITY_MATRIX_VALID_VALUES:
        return _err(
            PreflightErrorKind.SchemaInvalid,
            f"capabilities.nvml_capability_matrix is not a recognized value "
            f"(got={nvml_matrix!r}); expected one of "
            f"{sorted(NVML_CAPABILITY_MATRIX_VALID_VALUES)}",
        )

    mig_mode_value = gpu.get("mig_mode", "")
    if not isinstance(mig_mode_value, str):
        return _err(
            PreflightErrorKind.SchemaInvalid,
            "gpu.mig_mode missing or not a string",
        )
    if mig_mode_value.strip() not in MIG_MODE_ADMISSIBLE_VALUES:
        return _err(
            PreflightErrorKind.MigSliceNotValidForCollectionEvidence,
            f"GPU MIG mode is {mig_mode_value!r}; collection evidence is "
            "not admissible from a MIG slice: whole-GPU NCU provenance "
            "(full GPC/SM topology and clock domains) is required for "
            "gpc__cycles_elapsed.max validation (D8).",
        )

    # schema v4: mps_active and vgpu_active are record-only
    # deployment-mode signals. Manifests at schema_version >= 4 MUST
    # carry both boolean capability fields (schema-correctness
    # check), but neither value triggers a typed rejection — both
    # `true` and `false` validate clean. Downstream consumers read
    # these fields to choose CUPTI attribution semantics
    # (per-client under MPS) and activity-kind fallback (`_KERNEL`
    # under vGPU). v3 manifests remain admissible without the fields
    # (backward compat); the validator simply skips the check at
    # sv < 4.
    if sv >= 4:
        for field in ("mps_active", "vgpu_active"):
            if field not in caps:
                return _err(
                    PreflightErrorKind.SchemaInvalid,
                    f"capabilities.{field} missing (required at schema_version >= 4)",
                )
            value = caps.get(field)
            if not isinstance(value, bool):
                return _err(
                    PreflightErrorKind.SchemaInvalid,
                    f"capabilities.{field} is not a bool (got {type(value).__name__})",
                )

    perms = manifest["permissions"]
    for field in REQUIRED_PERMISSION_BOOLS:
        value = perms.get(field)
        if not isinstance(value, bool):
            return _err(
                PreflightErrorKind.SchemaInvalid,
                f"permissions.{field} missing or not a bool",
            )
        if not value:
            return _err(
                PreflightErrorKind.SchemaInvalid,
                f"permissions.{field} is false (collection host "
                "lacks the required access)",
            )

    cap = gpu["compute_capability"]
    if not any(cap.startswith(prefix) for prefix in ADMITTED_CAP_PREFIXES):
        return _err(
            PreflightErrorKind.TargetArchMismatch,
            f"compute capability {cap!r} is not in the admitted "
            "local-development arches (sm_86 = '8.6', sm_120 family = "
            "'12.x')",
        )

    if cap.startswith(SM120_CAP_PREFIX):
        name_lower = gpu["name"].lower()
        for marker in REJECTED_SM120_NAME_SUBSTRINGS:
            if marker in name_lower:
                return _err(
                    PreflightErrorKind.TargetArchMismatch,
                    f"sm_120 device name {gpu['name']!r} matches the "
                    f"server/workstation Blackwell marker {marker!r}; "
                    "the local-development contract admits only "
                    "consumer Blackwell laptop SKUs (per "
                    "CLAUDE.md 'Two-Device Local Development Model'). "
                    "Server / workstation sm_120 evidence must be "
                    "captured under a separate (deferred) server "
                    "preflight policy, not via the local-laptop gate.",
                )

    return PreflightValidationResult(ok=True, error=None)


def _normalize_for_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(manifest)
    normalized.pop("timestamp", None)
    return normalized


def compute_preflight_sha256(manifest: dict[str, Any]) -> str:
    normalized = _normalize_for_hash(manifest)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_downstream_preflight_reference(
    downstream_artifact: dict[str, Any],
    manifest: dict[str, Any],
) -> PreflightValidationResult:
    ref = downstream_artifact.get("preflight_manifest_sha256")
    if not isinstance(ref, str) or not ref.strip():
        return _err(
            PreflightErrorKind.PreflightMissing,
            "downstream artifact has no preflight_manifest_sha256 reference",
        )
    actual = compute_preflight_sha256(manifest)
    if ref != actual:
        return _err(
            PreflightErrorKind.PreflightMissing,
            f"preflight_manifest_sha256 mismatch (artifact={ref}, manifest={actual})",
        )
    return PreflightValidationResult(ok=True, error=None)


__all__ = [
    "ADMITTED_CAP_PREFIXES",
    "REJECTED_SM120_NAME_SUBSTRINGS",
    "CC_MODE_VALID_VALUES",
    "CLUSTER_ATTR_UNAVAILABLE_SENTINELS",
    "CUPTI_ACTIVITY_RUNTIME_UNAVAILABLE_SENTINELS",
    "CUPTI_ACTIVITY_RUNTIME_VALID_VALUES",
    "CUPTI_ACTIVITY_UNAVAILABLE_SENTINELS",
    "MIG_MODE_ADMISSIBLE_VALUES",
    "NVML_CAPABILITY_MATRIX_VALID_VALUES",
    "PreflightError",
    "PreflightErrorKind",
    "PreflightValidationResult",
    "REQUIRED_PERMISSION_BOOLS",
    "REQUIRED_TOOLCHAIN_FIELDS",
    "SM120_CAP_PREFIX",
    "UNPINNED_SENTINELS",
    "compute_preflight_sha256",
    "load_manifest",
    "validate_downstream_preflight_reference",
    "validate_preflight_manifest",
]
