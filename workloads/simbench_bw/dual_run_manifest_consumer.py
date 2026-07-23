"""Consumer-side admission for `dual_run_manifest.json` (non-root measurement
the dual-run-manifest contract).

The non-root measurement contract's the dual-run-manifest contract negative tests require three typed
diagnostics:

  * `dual_run_manifest_missing` — emitted when consuming an NVBit
    phase ledger without a matching dual-run manifest. The pair is
    excluded from downstream metric and gate consumers.
  * `dual_run_manifest_ambiguous` — emitted when a manifest does not
    match exactly one CUPTI-run id and exactly one NVBit-run id, OR
    when two manifests claim the same `(cupti_run_id, nvbit_run_id)`
    pairing.
  * `dual_run_drift_exceeded` — emitted when a per-launch
    `(cupti_run_nvml_state, nvbit_run_nvml_state)` divergence
    exceeds the manifest's `drift_tolerance_ns`. The pair is
    excluded from downstream metric and gate consumers.

The schema also names a fourth diagnostic that the schema's
description introduces (`dual_run_manifest_binary_sha256_disagreement`)
when the two runs claim different `binary_sha256` values; the
consumer surfaces it as a separate typed exception so a producer-side
test can pin it explicitly.

The schema validation itself happens through the existing the validation-metric framework
`_FAMILY_SCHEMA_MAP` route (`dual_run_manifest_*.json` ->
`dual_run_manifest.schema.json`); this consumer assumes the input
manifest is already schema-valid (or fails in the the validation-metric framework framework
before reaching here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# --- Typed diagnostics ------------------------------------------------------


class DualRunManifestMissing(Exception):
    TOKEN = "dual_run_manifest_missing"

    def __init__(
        self,
        ledger_ref: str | None = None,
        expected_manifest_path: str | None = None,
    ):
        super().__init__(
            f"{self.TOKEN}: cannot consume an NVBit phase ledger "
            f"(ref={ledger_ref!r}) without a matching dual_run_manifest.json "
            f"(expected at {expected_manifest_path!r}); the pair is excluded "
            "from downstream metric and gate consumers per the dual-run-manifest contract."
        )
        self.ledger_ref = ledger_ref
        self.expected_manifest_path = expected_manifest_path

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "ledger_ref": self.ledger_ref,
            "expected_manifest_path": self.expected_manifest_path,
        }


class DualRunManifestAmbiguous(Exception):
    TOKEN = "dual_run_manifest_ambiguous"

    def __init__(
        self,
        cupti_run_id: str | None = None,
        nvbit_run_id: str | None = None,
        ambiguity_kind: str = "unknown",
        candidate_paths: tuple[str, ...] = (),
    ):
        super().__init__(
            f"{self.TOKEN}: ambiguity_kind={ambiguity_kind!r} "
            f"cupti_run_id={cupti_run_id!r} nvbit_run_id={nvbit_run_id!r}; "
            f"candidates={list(candidate_paths)}. A manifest must match "
            "exactly one CUPTI run id AND exactly one NVBit run id; "
            "ambiguous pairings are rejected per the dual-run-manifest contract."
        )
        self.cupti_run_id = cupti_run_id
        self.nvbit_run_id = nvbit_run_id
        self.ambiguity_kind = ambiguity_kind
        self.candidate_paths = candidate_paths

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "cupti_run_id": self.cupti_run_id,
            "nvbit_run_id": self.nvbit_run_id,
            "ambiguity_kind": self.ambiguity_kind,
            "candidate_paths": list(self.candidate_paths),
        }


class DualRunDriftExceeded(Exception):
    TOKEN = "dual_run_drift_exceeded"

    def __init__(
        self,
        launch_match_key: dict[str, Any],
        drift_observed_ns: int,
        drift_tolerance_ns: int,
    ):
        super().__init__(
            f"{self.TOKEN}: launch_match_key={launch_match_key!r} "
            f"drift_observed_ns={drift_observed_ns} "
            f"> drift_tolerance_ns={drift_tolerance_ns}; the pair is "
            "excluded from downstream metric and gate consumers per the dual-run-manifest contract."
        )
        self.launch_match_key = launch_match_key
        self.drift_observed_ns = drift_observed_ns
        self.drift_tolerance_ns = drift_tolerance_ns

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "launch_match_key": self.launch_match_key,
            "drift_observed_ns": self.drift_observed_ns,
            "drift_tolerance_ns": self.drift_tolerance_ns,
        }


class DualRunManifestBinarySha256Disagreement(Exception):
    TOKEN = "dual_run_manifest_binary_sha256_disagreement"

    def __init__(
        self,
        cupti_binary_sha256: str,
        nvbit_binary_sha256: str,
    ):
        super().__init__(
            f"{self.TOKEN}: cupti_binary_sha256={cupti_binary_sha256!r} "
            f"!= nvbit_binary_sha256={nvbit_binary_sha256!r}; the manifest "
            "is invalid because the two runs must have used the same kernel "
            "runner binary (per the dual-run-manifest contract binary_sha256 invariant)."
        )
        self.cupti_binary_sha256 = cupti_binary_sha256
        self.nvbit_binary_sha256 = nvbit_binary_sha256

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "cupti_binary_sha256": self.cupti_binary_sha256,
            "nvbit_binary_sha256": self.nvbit_binary_sha256,
        }


# --- Drift computation ------------------------------------------------------


def _compute_per_launch_drift_ns(
    cupti_state: dict[str, Any], nvbit_state: dict[str, Any]
) -> int:
    """Compute the load-bearing drift in nanoseconds between two NVML
    state snapshots, per the canonical `nvml_state_divergence`
    definition in `workloads/simbench_bw/schemas/canonical_serialization_spec.md`.

    The canonical rule is the absolute difference between the two
    snapshots' `sample_timestamp_ns_host_monotonic` fields. Other
    state fields (sm_clock_mhz, mem_clock_mhz, temp_c, power_mw,
    throttle_reasons_bitmask) are validated by the schema's per-field
    admission but are NOT collapsed into the wall-clock divergence
    metric — folding them in would double-count the NVML
    covariate-regression layer's contribution.
    """
    cupti_ts = int(cupti_state.get("sample_timestamp_ns_host_monotonic", 0))
    nvbit_ts = int(nvbit_state.get("sample_timestamp_ns_host_monotonic", 0))
    return abs(cupti_ts - nvbit_ts)


# --- Manifest admission -----------------------------------------------------


def admit_dual_run_manifest(
    manifest: dict[str, Any],
    *,
    drift_tolerance_ns: int | None = None,
    cupti_run_id_candidates: tuple[str, ...] = (),
) -> None:
    """Validate the the dual-run-manifest contract invariants on a single dual-run manifest.
    Raises typed diagnostics on defect; passes silently on admissible
    manifests.

    The drift tolerance is taken from the manifest's own field by
    default; the `drift_tolerance_ns` keyword overrides for tests
    that want to drive a tighter check. The
    `cupti_run_id_candidates` tuple is the list of cupti_run_id
    values the consumer found across committed CUPTI-side measurement
    manifests; if the manifest's `cupti_run_id` is not in this list
    AND the tuple is non-empty, the manifest is rejected as ambiguous
    (the schema description names this exact case)."""
    if drift_tolerance_ns is None:
        drift_tolerance_ns = int(manifest.get("drift_tolerance_ns", 0))

    cupti_run_id = str(manifest.get("cupti_run_id", ""))
    nvbit_run_id = str(manifest.get("nvbit_run_id", ""))

    # Ambiguity check: when a candidate list is provided, the manifest's
    # cupti_run_id must appear in it (per the schema description for
    # `cupti_run_id`).
    if cupti_run_id_candidates and cupti_run_id not in cupti_run_id_candidates:
        raise DualRunManifestAmbiguous(
            cupti_run_id=cupti_run_id,
            nvbit_run_id=nvbit_run_id,
            ambiguity_kind="cupti_run_id_not_in_candidates",
            candidate_paths=cupti_run_id_candidates,
        )

    # Per-launch drift check.
    for launch in manifest.get("launches", []):
        cupti_state = launch.get("cupti_run_nvml_state", {})
        nvbit_state = launch.get("nvbit_run_nvml_state", {})
        drift_observed = _compute_per_launch_drift_ns(cupti_state, nvbit_state)
        if drift_observed > drift_tolerance_ns:
            raise DualRunDriftExceeded(
                launch_match_key=launch.get("launch_match_key", {}),
                drift_observed_ns=drift_observed,
                drift_tolerance_ns=drift_tolerance_ns,
            )


def assert_binary_sha256_consistency(
    cupti_binary_sha256: str, nvbit_binary_sha256: str
) -> None:
    """Raise `DualRunManifestBinarySha256Disagreement` if the CUPTI-side
    and NVBit-side binary hashes disagree.

    The schema requires the manifest to record a single `binary_sha256`
    that is identical across both runs; this helper is exposed so a
    producer-side test can drive the disagreement path explicitly
    before the manifest is rendered."""
    if cupti_binary_sha256 != nvbit_binary_sha256:
        raise DualRunManifestBinarySha256Disagreement(
            cupti_binary_sha256=cupti_binary_sha256,
            nvbit_binary_sha256=nvbit_binary_sha256,
        )


def require_manifest_for_phase_ledger(
    ledger: dict[str, Any],
    manifest_path: Path | None,
) -> None:
    """Raise `DualRunManifestMissing` if a phase-cycle ledger is
    consumed without a matching dual-run manifest.

    `manifest_path` is the resolved path the consumer expected to
    find. None means the consumer's resolver returned no candidate
    (file not present, ref unset, or filesystem search failed).
    """
    if manifest_path is None or not manifest_path.is_file():
        raise DualRunManifestMissing(
            ledger_ref=ledger.get("dual_run_manifest_ref"),
            expected_manifest_path=str(manifest_path) if manifest_path else None,
        )


def detect_manifest_ambiguity(
    manifest_paths: tuple[Path, ...],
    *,
    cupti_run_id: str | None = None,
    nvbit_run_id: str | None = None,
) -> None:
    """Raise `DualRunManifestAmbiguous` if more than one manifest
    file claims the same `(cupti_run_id, nvbit_run_id)` pairing,
    OR if zero manifests match the requested pair.

    Used by the join layer to detect "two manifests, same pair"
    ambiguity that the schema-side validation cannot catch.
    """
    if not manifest_paths:
        raise DualRunManifestAmbiguous(
            cupti_run_id=cupti_run_id,
            nvbit_run_id=nvbit_run_id,
            ambiguity_kind="no_manifest_for_pair",
            candidate_paths=(),
        )
    if len(manifest_paths) > 1:
        raise DualRunManifestAmbiguous(
            cupti_run_id=cupti_run_id,
            nvbit_run_id=nvbit_run_id,
            ambiguity_kind="multiple_manifests_for_pair",
            candidate_paths=tuple(str(p) for p in manifest_paths),
        )
