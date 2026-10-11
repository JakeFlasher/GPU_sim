"""Predictor-vs-NCU comparison runner (library-first).

Loads ``workloads/simbench_bw/comparison_contract.toml`` via an
explicit ``load_comparison_contract(path)`` function (module import
does NOT perform IO), joins admitted predictor bundles (per
``workloads/simbench_bw/schemas/cache_predictor.schema.json``)
against admitted NCU label cells (per ``data/local_labels/...``)
on the contract-pinned 7-field key
``(slug, shape_point, binary_sha256, source_commit,
kernel_signature_sha256, arch, ncu_metric_version)``, normalizes
the NCU percent metric to fraction at point of use, computes the
contract relative-error formula
``|predicted - ground_truth| / max(ground_truth, epsilon)`` with
``epsilon=0.01``, dispatches the three predictor result arms
(scalar / bounded / typed_unavailable), enforces the per-arch
smoke-test floor (``headline_cells_per_arch >= 5``), computes a
log-space geomean over headline cells with a cell-level bootstrap
CI, and emits a deterministic schema-valid output bundle that
validates against
``workloads/simbench_bw/schemas/comparison_bundle.schema.json``.

The bounded-arm canonical is the midpoint of
``[hit_rate_lower, hit_rate_upper]``; the relative error against
``hit_rate_lower`` and the relative error against
``hit_rate_upper`` are recorded as diagnostic-only fields per the
resolved bounded-arm policy.

The runner is library-first: every behavior is reachable via
function calls on importable surfaces. A thin ``__main__`` CLI
wrapper is provided for operator-facing automation; it accepts
``--contract``, ``--predictor-bundles-glob``,
``--label-admitted-path``, and ``--output``.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import random
import statistics
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Monotonic calibration head v5 active-path integration
# ---------------------------------------------------------------------
# The v5 head lives in workloads/simbench_bw/monotonic_calibration_head.py
# and is consumed via workloads/simbench_bw/canonical_features.predict_active_path
# on bundles that carry the canonical_pre_calibration_features block. The
# legacy 3-stage multiplicative correction chain stays in cache_predictor.py
# for v4 evaluation but is bypassed when the v5 head is loaded.
#
# Per-arch transfer diagnostics: when the comparison runs in v5 active
# mode AND per-arch head sidecars are available, run_comparison emits a
# `per_arch_transfer_diagnostics` block covering both directions
# (sm_86 -> sm_120 and sm_120 -> sm_86) with the epsilon-floored 2x rule
# + the transfer median-absolute-error gate.
#
# These imports are intentionally lazy (inside helper functions) so the
# legacy v4 run_comparison path stays byte-stable for callers that do not
# opt into v5 mode.

V5_HEAD_DEFAULT_PATH = "data/local_labels/calibration_head_v5.json"
V5_CANONICAL_FEATURES_FIELD = "canonical_pre_calibration_features"
V5_PER_ARCH_TRANSFER_DIAGNOSTICS_FIELD = "per_arch_transfer_diagnostics"


def load_calibration_head_v5(
    path: str | Path = V5_HEAD_DEFAULT_PATH,
) -> dict | None:
    """Load the v5 head sidecar. Returns the parsed dict OR None when the
    sidecar is absent (legacy v4 mode). Validates `calibration_head_version`
    is 5 + `fit_status` is one of the recognized values."""
    p = Path(path)
    if not p.is_file():
        return None
    head = json.loads(p.read_text(encoding="utf-8"))
    if head.get("calibration_head_version") != 5:
        return None
    if head.get("fit_status") not in (
        "ok", "empty_cohort_no_admitted_cells",
        "indeterminate_dependencies_absent",
    ):
        return None
    return head


def apply_v5_active_path_to_bundle(
    *,
    bundle: dict,
    v5_head: dict,
    ensemble: Any,
) -> float | None:
    """Apply the v5 active path to a single bundle.

    Returns the calibrated hit-rate scalar OR None when:
    - the bundle has no canonical_pre_calibration_features block; OR
    - the v5 head fit_status is not 'ok'; OR
    - the ensemble cannot route the bundle's (arch, family_bucket).

    Uses workloads/simbench_bw/canonical_features.predict_active_path so
    legacy per-slug factor kwargs raise the typed
    ActivePathDomainMismatchError rather than silently leaking the retired
    chain into the v5 path.
    """
    if v5_head.get("fit_status") != "ok":
        return None
    features = bundle.get(V5_CANONICAL_FEATURES_FIELD)
    if features is None:
        return None
    mpi = bundle.get("manifest_pairing_index", {})
    arch = mpi.get("arch")
    # Resolve family_bucket via the canonical-features producer side; the
    # bundle does NOT carry family_bucket so the caller must wire it in
    # via the cohort_scope.admitted lookup or via a separate roster join.
    # This helper expects a pre-resolved family_bucket attribute on the
    # bundle for in-process composition; callers that join via the v5 head's
    # cohort_scope can use ensemble.predict directly.
    family_bucket = bundle.get("family_bucket")
    if not (arch and family_bucket):
        return None
    try:
        from workloads.simbench_bw.canonical_features import (
            predict_active_path,
        )
    except ImportError:
        return None
    try:
        out = predict_active_path(
            bundle_with_features=bundle,
            head_predict_fn=lambda f: ensemble.predict(
                arch, family_bucket, f,
            ),
        )
    except (ValueError, RuntimeError):
        return None
    return out.get("calibrated_hit_rate")


def attach_v5_metadata(
    base: dict[str, Any],
    v5_head_path: str | None,
    v5_active_records: list[dict] | None = None,
) -> dict[str, Any]:
    """Attach v5_head_source_commit + v5_head_fit_status + replacement
    count + per_arch_transfer_diagnostics to the output bundle. Called
    before EVERY ``return base`` arm in run_comparison v5 mode so
    typed-unavailable + headline-validity + coverage + admitted arms all
    carry the v5 diagnostic surface."""
    if not v5_head_path:
        return base
    v5_head = load_calibration_head_v5(v5_head_path)
    if v5_head is None:
        return base
    base[V5_PER_ARCH_TRANSFER_DIAGNOSTICS_FIELD] = (
        emit_per_arch_transfer_diagnostics_block(v5_head=v5_head)
    )
    base["v5_head_source_commit"] = v5_head.get("source_commit")
    base["v5_head_fit_status"] = v5_head.get("fit_status")
    base["v5_active_path_replaced_predicted_hit_rate_count"] = len(
        v5_active_records or [],
    )
    return base


def emit_per_arch_transfer_diagnostics_block(
    *,
    v5_head: dict,
) -> dict[str, Any]:
    """Compose the per_arch_transfer_diagnostics block for the comparison
    bundle by delegating to the dedicated v5 active-path driver. Returns
    a dict with both transfer directions (sm_86 -> sm_120 + sm_120 -> sm_86)
    + epsilon-floored 2x rule status + median-absolute-error gate status.

    When the driver cannot run (sklearn absent, no admitted cells on one or
    both arches), the block reports `status=indeterminate_*` on each
    direction so consumers see the methodological state instead of a
    silently-missing block.
    """
    try:
        from tools.local_labels import run_v5_active_path_comparison as drv
    except ImportError:
        return {
            "status": "indeterminate_v5_driver_unavailable",
            "reason": (
                "tools/local_labels/run_v5_active_path_comparison.py "
                "could not be imported. The v5 active-path comparison "
                "driver is the canonical source for the per_arch_transfer_diagnostics "
                "block."
            ),
        }
    cells = drv._rehydrate_cells_from_v5_head(v5_head)
    sm_120_cells = [c for c in cells if c["arch"] == "sm_120"]
    sm_86_cells = [c for c in cells if c["arch"] == "sm_86"]
    try:
        sm_120_ensemble = drv._fit_arch_head(sm_120_cells, v5_head=v5_head)
        sm_86_ensemble = drv._fit_arch_head(sm_86_cells, v5_head=v5_head)
    except ImportError:
        return {
            "status": "indeterminate_dependencies_absent",
            "reason": (
                "Required calibration-head dependencies (sklearn for "
                "IR-on-projection; pygam / lightgbm / xgboost for "
                "fallback) are not importable on this host."
            ),
        }
    return drv._compute_transfer_diagnostics(
        sm_86_ensemble=sm_86_ensemble,
        sm_120_ensemble=sm_120_ensemble,
        sm_86_cells=sm_86_cells,
        sm_120_cells=sm_120_cells,
    )


# ---------------------------------------------------------------------
# Defect-token frozenset (the runner-level vocabulary)
# ---------------------------------------------------------------------
# Every typed defect the runner can produce (either as a typed
# exception or as an entry in the output bundle's ``excluded_cells``)
# MUST be enumerated here. The bidirectional structural lockstep
# test in ``tests/test_comparison_runner.py`` asserts
# ``set(asserted_runner_defect_tokens) == _RUNNER_DEFECT_TOKENS``
# (set equality, not subset-of). Adding a token here without a
# corresponding negative-control test in the test file fails the
# lockstep; asserting a token in a test that is not here also
# fails the lockstep. The discipline mirrors
# ``BL-20260509-strict-admission-contract-first``'s contract-first
# enumeration over (axes x surfaces).
_RUNNER_DEFECT_TOKENS: frozenset[str] = frozenset({
    "comparison_contract_drift",
    "comparison_runner_join_key_unmatched",
    "comparison_runner_unknown_result_kind",
    "comparison_runner_predicted_hit_rate_out_of_range",
    "comparison_runner_typed_reason_out_of_enum",
    "comparison_runner_bounded_interval_invalid",
    "comparison_runner_duplicate_predictor_bundle",
})


# Contract-side fields the runner cross-checks at load time. Drift
# in any pinned field produces ``comparison_contract_drift``. The
# ``ncu_metric_version`` is ALSO a per-cell cross-check (against
# the label cell's ``profiler_version``); the field name appears
# in both the contract drift list and the per-cell drift surface.
# Adding a load-time pinned field requires a coordinated update
# of this tuple, the corresponding ``_PINNED_*`` constant, the
# loader's drift check, and a negative-control test.
_CONTRACT_PINNED_FIELDS: tuple[str, ...] = (
    "epsilon",
    "bootstrap_resamples",
    "ci_confidence",
    "join_key",
    "coverage_gate",
    "ncu_metric_version",
    "bounded_arm_handling",
)


# Pinned constants the contract MUST carry exactly. Drift is
# rejected at ``load_comparison_contract`` boundary.
_PINNED_EPSILON: float = 0.01
_PINNED_BOOTSTRAP_RESAMPLES: int = 2000
_PINNED_CI_CONFIDENCE: float = 0.95
_PINNED_JOIN_KEY: tuple[str, ...] = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "arch",
    "ncu_metric_version",
    "cache_layer",
)
_PINNED_SMOKE_TEST_FLOOR_PER_ARCH: int = 5
_PINNED_PUBLICATION_GRADE_FLOOR_PER_ARCH: int = 10
_PINNED_ASPIRATIONAL_TARGET_TOTAL: int = 20
# Contract-side NCU protocol pin. The runner cross-checks the
# loaded contract's ``ncu_metric_version`` against this constant
# at load time AND cross-checks every label cell's
# ``profiler_version`` against the loaded contract's value at run
# time. Any drift on either surface is the same typed defect
# (``comparison_contract_drift``) but on different boundaries.
_PINNED_NCU_METRIC_VERSION: str = "ncu_2026.1.x"
# Contract-side bounded-arm policy pin. The runner consumes
# bounded predictor bundles using midpoint canonical for the
# headline relative error and records the relative errors against
# ``hit_rate_lower`` and ``hit_rate_upper`` as diagnostics.
# ``"report_three"`` names that policy in the contract; drift is
# a coordinated change that must walk through the schema, the
# runner, and the test surface together.
_PINNED_BOUNDED_ARM_HANDLING: str = "report_three"


# DEC-T5 LOOCV-p90 residual threshold for the sm_86-only fit.
# When the chosen calibration head's leave-one-out cross-
# validation p90 residual exceeds this value, the producer fires
# ``calibration_loocv_p90_residual_exceeded``: the typed reason
# is added to the bundle's headline-validity-failed reason list
# and the bundle's calibrated-scalar arm becomes
# ``typed_unavailable`` per the calibration plan's spec. Legacy
# v1 + active v4 calibration sidecars both expose the chosen
# head's p90 in ``loocv_table[chosen_head].p90`` — the
# (historical) calibration_evaluation_v1.json carried the same
# shape under the sm_86 path.
_DEC_T5_LOOCV_P90_THRESHOLD: float = 0.30
CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED: str = (
    "calibration_loocv_p90_residual_exceeded"
)

# Variance-precondition typed reason. Fires when an upstream
# cache_predictor bundle's typed_reason is
# "calibration_variance_precondition_failed" (i.e., the
# variance-precondition gate fired because the cross-cell
# prediction stddev fell below the documented 0.10 floor). The
# comparison runner promotes this upstream typed reason to the
# comparison-level headline-validity-failed-reasons list so
# downstream consumers see the calibration variance failure on
# the comparison bundle, not just on the per-cell predictor
# bundle.
CALIBRATION_VARIANCE_PRECONDITION_FAILED: str = (
    "calibration_variance_precondition_failed"
)

# L1-head typed reasons exposed as module-level constants so the
# 3-surface lockstep test
# (workloads/simbench_bw/tests/test_predictor_l1_sass_provenance_missing_lockstep.py)
# can detect drift on this comparison_runner surface independently
# of the cache_predictor._TYPED_REASONS frozenset and the
# schemas/cache_predictor.schema.json typed_reason enum surfaces.
#
# PREDICTOR_L1_SASS_PROVENANCE_MISSING fires when the L1 head
# cannot obtain a per-PC cuobjdump-derived
# instruction_cache_op mapping for the admitted cell's binary
# (no cuobjdump output, kernel section missing, or binary not
# introspectable). Per BL-20260507-arch-token-multi-surface-lockstep,
# the literal is held in this comparison_runner surface, the
# cache_predictor._TYPED_REASONS frozenset, AND the
# cache_predictor.schema.json typed_reason enum — removing the
# constant from any one surface fires the lockstep paired-control
# regression test.
PREDICTOR_L1_SASS_PROVENANCE_MISSING: str = (
    "predictor_l1_sass_provenance_missing"
)

# PREDICTOR_L1_NO_MODELED_L1_ELIGIBLE_ACCESSES fires when SASS
# provenance IS available but the kernel's per-PC mapping
# contains zero l1_eligible=true entries (a shared-memory-only
# kernel; the L1 has nothing to model). Distinct from
# predictor_l1_sass_provenance_missing because shared-only
# kernels HAVE SASS — they just have nothing for the L1 model
# to walk.
PREDICTOR_L1_NO_MODELED_L1_ELIGIBLE_ACCESSES: str = (
    "predictor_l1_no_modeled_l1_eligible_accesses"
)

# PREDICTOR_L1_CTA_TO_SM_SCHEDULE_UNKNOWN fires when the per-SM
# L1 walker cannot resolve a CTA-to-SM scheduling policy
# (cta_to_sm_schedule_policy="unknown" or trace lacks grid_dim
# / block_dim metadata). CTAs on different SMs do not evict
# each other's L1 lines on Ampere / Blackwell baseline, so
# guessing the scheduling policy biases the prediction; the
# fail-close arm preserves typed_unavailable instead of
# emitting a load-bearing scalar derived from a guess.
PREDICTOR_L1_CTA_TO_SM_SCHEDULE_UNKNOWN: str = (
    "predictor_l1_cta_to_sm_schedule_unknown"
)

# PREDICTOR_L1_CAPACITY_INVALID fires when the L1 head's
# DEC-2 set-associative refinement detects PRESENT-BUT-INVALID
# geometry (a static-card entry's `l1_ways` field that violates
# the walker's geometry contract: l1_ways=0, or an overlarge
# l1_ways whose derived set_count < 1). Silently falling through
# to the fully-associative scalar bundle would mask a malformed
# `l1_ways` edit during way-count tuning, so the L1 head fails
# closed with this typed reason. Geometry-absent (no
# l1_ways field at all in the static-card entry) is allowed to
# fall through to the fully-associative baseline; only geometry-
# present-but-invalid triggers this typed reason.
PREDICTOR_L1_CAPACITY_INVALID: str = (
    "predictor_l1_capacity_invalid"
)

# Typed reasons that the comparison runner INJECTS itself (the predictor
# never emits them), so they are NOT members of
# ``cache_predictor._TYPED_REASONS`` (the predictor-side producer enum) and
# do NOT belong there: adding them would advance the post-commit recapture
# guards' "predictor fix commit" (computed via `git log -- cache_predictor.py`)
# for a non-predictor change. The on-disk bundle schema
# (``cache_predictor.schema.json`` typed_reason enum) is the UNION of the
# predictor-emitted reasons and these runner-injected reasons, so
# ``_validate_predictor_bundle_shape`` admits a bundle carrying either kind by
# checking ``cache_predictor._TYPED_REASONS | _CR_INJECTED_TYPED_REASONS``.
#
# ``v5_active_path_unscorable`` is written by the v5 active-path block in
# ``run_comparison`` (a scalar bundle the v5 head could not score — setup
# failure, missing canonical features, bucket outside the head's cohort scope,
# or an ensemble.predict failure — is rewritten typed_unavailable so the legacy
# predicted_hit_rate does NOT survive into the v5 headline population).
_CR_INJECTED_TYPED_REASONS: frozenset[str] = frozenset({
    "v5_active_path_unscorable",
})

# L1_MODEL_CLASS_* constants pin the closed-enum vocabulary that the
# cache_predictor.py schema accepts for the bundle's `l1_model_class`
# field. The 3-surface lockstep applies per BL-20260507: this
# surface (comparison_runner module-level constants) mirrors the
# cache_predictor.py emitter + the schema enum so deleting any one
# of the literals across the three surfaces fires the lockstep
# paired-control regression test.
L1_MODEL_CLASS_FULLY_ASSOCIATIVE_BASELINE: str = (
    "fully_associative_baseline"
)
L1_MODEL_CLASS_SET_ASSOCIATIVE_REFINEMENT: str = (
    "set_associative_refinement"
)
L1_MODEL_CLASS_SET_ASSOCIATIVE_REFINEMENT_PENDING: str = (
    "set_associative_refinement_pending"
)
# L1_MODEL_CLASS_REDUCTION_KERNEL_AWARE_CORRECTION is the
# alternative model-class lever. The L1 head applies a
# multiplicative correction to the FA/SA scalar when the
# predictor-side slug classifier identifies a reduction /
# activation kernel (suffix `_fwd_bf16_sm80`). The corrected
# scalar is emitted as predicted_hit_rate; the FA/SA result is
# preserved under evidence.predicted_l1_hit_rate_uncorrected. The
# model-class value is recorded in the bundle's `l1_model_class`
# field for downstream consumers.
L1_MODEL_CLASS_REDUCTION_KERNEL_AWARE_CORRECTION: str = (
    "reduction_kernel_aware_correction"
)


# LAUNCH_PARAMETERS_SIDECAR_IDENTITY_MISMATCH fires when the L2 head's
# gridDim density correction loads the per-cell launch-parameter
# sidecar and the sidecar's embedded identity tuple (arch / slug /
# shape_point / binary_sha256 / kernel_signature_sha256) does NOT
# match the caller's admitted-row / manifest tuple, OR when the
# sidecar's `grid_dim` payload is malformed (missing / non-dict /
# non-positive-integer x/y/z). The producer-side semantic-binding
# contract requires every predictor input that influences the bundle
# output to be content-bound to the caller identity; a stale-or-
# swapped sidecar would silently apply the wrong gridDim correction
# otherwise. Filename-only sidecar selection without identity
# validation is the bug this typed reason guards against.
LAUNCH_PARAMETERS_SIDECAR_IDENTITY_MISMATCH: str = (
    "launch_parameters_sidecar_identity_mismatch"
)

# LOCAL_LABEL_ADMITTED_L1_MEAN_MIGRATION_REQUIRED fires when the
# comparison runner ingests a label cell with no l1tex_label_status
# field (i.e., a pre-migration v3 admitted-cell record). The v4
# admission contract requires every cell to carry the new field
# so the L1 join path can route correctly. Forcing re-aggregation
# is the only safe path because a v3 record dropped the L1 mean
# at aggregation time and the comparison runner cannot reconstruct
# it post-hoc.
LOCAL_LABEL_ADMITTED_L1_MEAN_MIGRATION_REQUIRED: str = (
    "local_label_admitted_l1_mean_migration_required"
)

# LOCAL_LABEL_ADMITTED_V5_MIGRATION_REQUIRED fires when the comparison runner
# ingests an admitted-label artifact (file-level or per-cell) whose
# schema_version is present but is not 5. The schema-v5 substrate is a hard
# parse-time boundary: a stale v4 (or older versioned) admitted file/record
# fed to the v5 consumer must be rejected with a typed error rather than
# silently scored. A cell that carries l1tex_label_status is past the v3 L1
# migration gate above, so the v5 gate is the next discriminator. (A truly
# unversioned in-memory record — no schema_version key at all — is left to the
# downstream checks; real admitted cells always carry the schema-required
# schema_version, so an unversioned record can only be a synthetic fixture.)
LOCAL_LABEL_ADMITTED_V5_MIGRATION_REQUIRED: str = (
    "local_label_admitted_v5_migration_required"
)


class AdmittedLabelV5MigrationError(ValueError):
    """Raised by _load_admitted_label_cells when the on-disk admitted-label
    file declares a schema_version other than 5. The CLI surfaces the typed
    reason and exits non-zero; the operator must re-aggregate to v5."""

    typed_reason = LOCAL_LABEL_ADMITTED_V5_MIGRATION_REQUIRED

# COMPARISON_RUNNER_L1TEX_LABEL_MEAN_MISSING fires when a v4
# admitted cell carries l1tex_label_status of mean_absent_label_missing
# or mean_absent_provenance_missing AND the comparison runner is
# attempting an L1 layer join for that cell. The L1 mean is null,
# so no L1 ground truth is available; the cell is excluded from the
# L1 join pre-emptively. The L2 join for the same cell continues
# normally because the L2 ground truth is independent.
COMPARISON_RUNNER_L1TEX_LABEL_MEAN_MISSING: str = (
    "comparison_runner_l1tex_label_mean_missing"
)

# Closed-enum cache-layer set the runner accepts for the bundle's
# cache_layer field. A bundle whose cache_layer is outside this
# enum is rejected at admission time. The contract's cache_layers
# scope further restricts which of these the runner will join in a
# given comparison; a bundle whose cache_layer is in this enum but
# NOT in the contract's cache_layers is excluded with
# comparison_contract_drift + unmatched_field="cache_layer".
_BUNDLE_CACHE_LAYER_ENUM: frozenset[str] = frozenset({"L1TEX", "L2"})
# Mapping from the bundle's cache_layer enum value to the
# contract's cache_layers entry. The contract names the layers
# without the "TEX" suffix ("L1", "L2"); the bundle uses the more
# specific L1TEX label to distinguish from a future L1 data cache
# (currently the L1TEX cache is the only L1-layer surface).
_BUNDLE_TO_CONTRACT_CACHE_LAYER = {"L1TEX": "L1", "L2": "L2"}


def check_calibration_loocv_p90_residual(
    evaluation_sidecar: dict,
    *,
    threshold: float = _DEC_T5_LOOCV_P90_THRESHOLD,
) -> str | None:
    """Return ``CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED`` when the
    chosen head's LOOCV p90 residual exceeds ``threshold``
    (default DEC-T5 pinned 0.30), or ``None`` otherwise.

    ``evaluation_sidecar`` is the parsed calibration-evaluation
    sidecar payload (active path: v4; legacy/historical: v1).
    Both the active v4 sidecar produced by
    ``tools/local_labels/fit_calibration_head_v4_sm120.py`` and
    the legacy
    ``data/local_labels/calibration_evaluation_v1.json`` payload
    (produced by ``tools/local_labels/fit_calibration_head.py``)
    expose ``loocv_table[chosen_head].p90`` with the same shape.
    A caller that cannot resolve ``chosen_head`` /
    ``loocv_table`` / the chosen head's ``p90`` returns ``None``
    rather than raising — the integration with run_comparison's
    headline-validity-failed-reasons list is task25's lockstep
    widening; this scaffolding only provides the decision
    function the integration will call.
    """
    chosen = evaluation_sidecar.get("chosen_head")
    loocv = evaluation_sidecar.get("loocv_table")
    if not isinstance(chosen, str) or not isinstance(loocv, dict):
        return None
    entry = loocv.get(chosen)
    if not isinstance(entry, dict):
        return None
    p90 = entry.get("p90")
    if not isinstance(p90, (int, float)) or isinstance(p90, bool):
        return None
    if float(p90) > threshold:
        return CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED
    return None


# AC-T7 sm_86-active headline-validity typed reasons. Three NEW
# reasons that join CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED in
# the ``headline_validity_failed_reasons`` array on every emitted
# bundle. These four constants are the sm_86-active vocabulary
# subset; the canonical schema-enum-matching vocabulary is the
# union ``HEADLINE_VALIDITY_FAILED_REASONS_ALL`` (defined below)
# which also includes the two cross-arch reasons. Per the
# calibration plan AC-T7 spec the reasons MUST fail closed at
# the headline-validity guard (no suppression flag may bypass
# them).
CALIBRATION_REUSE_DISTANCE_OUT_OF_ENVELOPE: str = (
    "calibration_reuse_distance_out_of_envelope"
)
CALIBRATION_COVARIATE_OUT_OF_ENVELOPE: str = (
    "calibration_covariate_out_of_envelope"
)
TRACE_OPCODE_COVERAGE_INCOMPLETE: str = (
    "trace_opcode_coverage_incomplete"
)

# AC-T5.2 cross-arch headline-validity reasons. Both are
# ADMITTED in the schema enum (additive-extension rule; no
# schema_version bump). Their production fire-path semantics
# differ:
#
#   - UNSUPPORTED_BLACKWELL_MEMORY_PATH: FIRES from
#     run_comparison's headline-validity gate when the
#     producer emits an unsupported_memory_path sidecar
#     entry with path_resolved=true (i.e., the classifier
#     recognized the opcode and assigned a memory class but
#     the downstream allocator / cache model can't fully
#     model it — typically Blackwell TMEM / CLUSTER_DSMEM /
#     ASYNC_COPY paths). See
#     check_unsupported_blackwell_memory_path(...) below.
#
#   - TARGET_ARCH_UNLABELED_EXTRAPOLATION: still admitted
#     but NOT yet fired by any production code path. Fires
#     only on cross-arch zero-shot prediction (AC-T6), which
#     stays DEFERRED with live sm_100 evidence per the
#     user's 2026-05-16 directive.
UNSUPPORTED_BLACKWELL_MEMORY_PATH: str = (
    "unsupported_blackwell_memory_path"
)
TARGET_ARCH_UNLABELED_EXTRAPOLATION: str = (
    "target_arch_unlabeled_extrapolation"
)

# Memory-touching op classes that, when paired with
# path_resolved=true on an unsupported_memory_path sidecar
# entry, justify firing UNSUPPORTED_BLACKWELL_MEMORY_PATH.
# Sourced from the producer-side is_excluded_op_class(...)
# membership in tools/nvbit_tracer/tracer_tool/tracer_tool.cu
# (the 7 routed-to-exclusion classes). A non-memory-touching
# class (e.g., UNIFORM_OP value on a stray entry) does NOT
# promote — the entry is malformed-but-known and stays on
# TRACE_OPCODE_COVERAGE_INCOMPLETE.
_MEMORY_TOUCHING_BLACKWELL_OP_CLASSES: frozenset[str] = frozenset({
    "TMEM",
    "CLUSTER_DSMEM",
    "ASYNC_COPY",
    "TMA",
    "TEXTURE",
    "SURFACE",
    "ATOMIC",
})


# The four sm_86-active headline-validity reasons in the order
# they appear in the schema's enum. Consumers iterating the
# enum vocabulary can use this tuple directly. The order is
# alphabetical to match the schema enum.
HEADLINE_VALIDITY_FAILED_REASONS_SM_86_ACTIVE: tuple[str, ...] = (
    CALIBRATION_COVARIATE_OUT_OF_ENVELOPE,
    CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED,
    CALIBRATION_REUSE_DISTANCE_OUT_OF_ENVELOPE,
    CALIBRATION_VARIANCE_PRECONDITION_FAILED,
    TRACE_OPCODE_COVERAGE_INCOMPLETE,
)

# The two cross-arch reasons, kept as a separate tuple so the
# canonical union vocabulary `HEADLINE_VALIDITY_FAILED_REASONS_ALL`
# can iterate them as a pinned tail after the sm_86-active
# reasons (preserving canonical multi-failure order). Member
# fire-path status differs:
#   - UNSUPPORTED_BLACKWELL_MEMORY_PATH is already consumed
#     by the current `unsupported_memory_path_sidecar`
#     promotion in `run_comparison(...)`.
#   - TARGET_ARCH_UNLABELED_EXTRAPOLATION is reserved for the
#     AC-T6 cross-arch zero-shot extrapolation path, which
#     emits to a separate
#     `cross_arch_zero_shot_prediction.json` schema and
#     remains deferred per the user's 2026-05-16 directive.
# Order matches the schema enum tail.
HEADLINE_VALIDITY_FAILED_REASONS_CROSS_ARCH: tuple[str, ...] = (
    UNSUPPORTED_BLACKWELL_MEMORY_PATH,
    TARGET_ARCH_UNLABELED_EXTRAPOLATION,
)

# Full set = sm_86-active ∪ cross-arch. This is the vocabulary
# the schema enum admits (6 entries) and the
# tools/local_labels/test_headline_validity_guard_scanner.py
# AC-T7 scanner compares against. The production fire-path
# (run_comparison's headline-validity gate) iterates this
# ordered tuple so the multi-failure result preserves the
# sm_86-active reasons first, cross-arch reasons appended.
# Production fire-path status by reason:
#   - All four sm_86-active reasons fire from the
#     existing per-input detection helpers (envelope_mask,
#     covariate booleans, unknown_opcodes_count) plus
#     the calibration evaluation sidecar's LOOCV p90.
#   - UNSUPPORTED_BLACKWELL_MEMORY_PATH fires when
#     run_comparison(...) is invoked with the
#     ``unsupported_memory_path_sidecar`` kwarg carrying
#     producer entries with ``path_resolved=true`` and a
#     memory-touching ``excluded_op_class`` (see
#     ``check_unsupported_blackwell_memory_path``).
#   - TARGET_ARCH_UNLABELED_EXTRAPOLATION is the only
#     constant in this tuple that no production code path
#     yet emits; it fires only on the cross-arch zero-shot
#     prediction artifact (AC-T6 follow-up loop).
HEADLINE_VALIDITY_FAILED_REASONS_ALL: tuple[str, ...] = (
    HEADLINE_VALIDITY_FAILED_REASONS_SM_86_ACTIVE
    + HEADLINE_VALIDITY_FAILED_REASONS_CROSS_ARCH
)


def check_calibration_reuse_distance_envelope(
    envelope_mask: Any,
) -> str | None:
    """Return
    ``CALIBRATION_REUSE_DISTANCE_OUT_OF_ENVELOPE`` when any
    element of ``envelope_mask`` is truthy, else None.

    ``envelope_mask`` is the
    ``out_of_envelope_mask`` tuple element returned by
    ``calibration_head.<Class>.transform(...)``: a boolean
    array True at indices whose input value falls outside the
    training envelope. A True element means the calibrated
    prediction extrapolates beyond the training-pair predicted
    range; the consumer must NOT trust the calibrated value
    for those cells.
    """
    if envelope_mask is None:
        return None
    try:
        any_out = any(bool(x) for x in envelope_mask)
    except TypeError:
        return None
    if any_out:
        return CALIBRATION_REUSE_DISTANCE_OUT_OF_ENVELOPE
    return None


def check_calibration_covariate_envelope(
    clock_in_range: bool,
    thermal_in_range: bool,
    throttle_in_range: bool,
) -> str | None:
    """Return
    ``CALIBRATION_COVARIATE_OUT_OF_ENVELOPE`` when ANY of the
    three covariate-in-range booleans is False, else None.

    The covariates are the read-only NVML observables the
    calibration head sees as evaluation metadata (clock,
    thermal, throttle state). When the inference covariate set
    falls outside the union of training-set covariate ranges,
    the calibrated prediction is extrapolating into an
    unobserved DVFS / throttle regime; the consumer must fail
    closed.
    """
    if not (
        isinstance(clock_in_range, bool)
        and isinstance(thermal_in_range, bool)
        and isinstance(throttle_in_range, bool)
    ):
        return None
    if not (clock_in_range and thermal_in_range and throttle_in_range):
        return CALIBRATION_COVARIATE_OUT_OF_ENVELOPE
    return None


def check_trace_opcode_coverage(
    unknown_opcodes_count: int,
) -> str | None:
    """Return ``TRACE_OPCODE_COVERAGE_INCOMPLETE`` when the
    NVBit tracer encountered ``unknown_opcodes_count > 0``
    opcodes that the producer-side classifier does not
    recognize, else None.

    The Blackwell-new TMEM family (LDT, STT, UTCCP, UTCBAR,
    UTMA, etc.) is the canonical example: a trace containing
    even one unclassified opcode means the per-launch
    reuse-distance walker emitted ``kUnspecified`` /
    ``unsupported_memory_path`` for at least one access, and
    the calibrated headline cannot be trusted on that cell.
    """
    if not isinstance(unknown_opcodes_count, int) or isinstance(
        unknown_opcodes_count, bool,
    ):
        return None
    if unknown_opcodes_count > 0:
        return TRACE_OPCODE_COVERAGE_INCOMPLETE
    return None


def check_unsupported_blackwell_memory_path(
    unsupported_memory_path_sidecar: list | None,
) -> str | None:
    """Return ``UNSUPPORTED_BLACKWELL_MEMORY_PATH`` when ANY
    entry in the producer-side
    ``unsupported_memory_path.json`` sidecar satisfies BOTH:
      - ``typed_reason == "unsupported_memory_path"``
      - ``path_resolved is True`` (the classifier recognized
        the opcode AND assigned a memory class)
      - ``excluded_op_class`` is in the
        memory-touching Blackwell set
        (``_MEMORY_TOUCHING_BLACKWELL_OP_CLASSES``)
    Otherwise return ``None``.

    The promotion criteria match the AC-T5.2 plan spec
    (lines 452-457): the headline-validity guard promotes
    the unsupported-memory-path sidecar into
    ``headline_validity_failed_reasons`` when the sidecar's
    typed reason is ``unsupported_memory_path`` AND the
    missing opcode is memory-touching per the AC-T5.1
    classifier tables.

    Entries with ``path_resolved is False`` (or absent)
    indicate the classifier did NOT recognize the opcode;
    those belong on ``TRACE_OPCODE_COVERAGE_INCOMPLETE``
    instead (the trace contained an opcode the producer
    tracer cannot model at all). Entries with
    ``path_resolved is True`` but a non-memory-touching
    ``excluded_op_class`` (e.g., a stray ``UNIFORM_OP``
    value) are malformed-but-known and also stay off this
    promotion.
    """
    if not isinstance(unsupported_memory_path_sidecar, list):
        return None
    for entry in unsupported_memory_path_sidecar:
        if not isinstance(entry, dict):
            continue
        if entry.get("typed_reason") != "unsupported_memory_path":
            continue
        if entry.get("path_resolved") is not True:
            continue
        op_class = entry.get("excluded_op_class")
        if op_class in _MEMORY_TOUCHING_BLACKWELL_OP_CLASSES:
            return UNSUPPORTED_BLACKWELL_MEMORY_PATH
    return None


def _compute_headline_validity_failed_reasons(
    headline_validity_inputs: dict | None,
    calibration_evaluation_sidecar: dict | None,
    unsupported_memory_path_sidecar: list | None = None,
    upstream_predictor_typed_reasons: list[str] | None = None,
) -> list[str]:
    """Walk every applicable detection helper and collect the
    fired reasons into a deterministic-order list.

    Per the calibration plan AC-T7 multi-failure invariant:
    "the guard records ALL conditions that fire (not just the
    first)". The output order is the
    ``HEADLINE_VALIDITY_FAILED_REASONS_ALL`` order (sm_86-
    active reasons first, cross-arch reasons appended) so
    re-runs against the same inputs are byte-deterministic
    AND a future cross-arch fire-path lands its reason in a
    pinned trailing slot.

    The function tolerates missing inputs: a key absent from
    ``headline_validity_inputs`` means the corresponding
    detection helper is skipped (no reason fired). The
    calibration LOOCV reason is sourced from the existing
    evaluation sidecar; the
    ``UNSUPPORTED_BLACKWELL_MEMORY_PATH`` reason is sourced
    from ``unsupported_memory_path_sidecar`` (the
    parsed-JSON sidecar entries emitted by the producer
    tracer per AC-T5.2 -- see
    ``check_unsupported_blackwell_memory_path``).
    """
    fired: set[str] = set()
    if calibration_evaluation_sidecar is not None:
        loocv_reason = check_calibration_loocv_p90_residual(
            calibration_evaluation_sidecar,
        )
        if loocv_reason is not None:
            fired.add(loocv_reason)
    if headline_validity_inputs is not None:
        if "envelope_mask" in headline_validity_inputs:
            r = check_calibration_reuse_distance_envelope(
                headline_validity_inputs["envelope_mask"],
            )
            if r is not None:
                fired.add(r)
        if all(
            k in headline_validity_inputs for k in (
                "clock_in_range", "thermal_in_range", "throttle_in_range",
            )
        ):
            r = check_calibration_covariate_envelope(
                headline_validity_inputs["clock_in_range"],
                headline_validity_inputs["thermal_in_range"],
                headline_validity_inputs["throttle_in_range"],
            )
            if r is not None:
                fired.add(r)
        if "unknown_opcodes_count" in headline_validity_inputs:
            r = check_trace_opcode_coverage(
                headline_validity_inputs["unknown_opcodes_count"],
            )
            if r is not None:
                fired.add(r)
    if unsupported_memory_path_sidecar is not None:
        r = check_unsupported_blackwell_memory_path(
            unsupported_memory_path_sidecar,
        )
        if r is not None:
            fired.add(r)
    # Promote upstream cache_predictor typed reasons that map to
    # comparison-level headline-validity reasons. The variance-
    # precondition gate's typed reason rides through the per-cell
    # predictor bundle's typed_reason field; the comparison runner
    # surfaces it at the comparison-bundle level so downstream
    # consumers see the calibration variance failure on the
    # comparison bundle, not just on the per-cell predictor
    # bundles.
    if upstream_predictor_typed_reasons:
        for upstream_reason in upstream_predictor_typed_reasons:
            if (
                upstream_reason
                == CALIBRATION_VARIANCE_PRECONDITION_FAILED
            ):
                fired.add(CALIBRATION_VARIANCE_PRECONDITION_FAILED)
    return [
        r for r in HEADLINE_VALIDITY_FAILED_REASONS_ALL
        if r in fired
    ]


# Externally-emitted diagnostic tokens for excluded-cell records.
# Two of these are runner-level defect tokens (members of
# ``_RUNNER_DEFECT_TOKENS``); the third is an upstream
# (predictor-side) exclusion identifier kept off the runner
# vocabulary. Every excluded-cell record carries one of these
# values in its ``diagnostic`` field so the on-disk artifact
# names the failure shape externally (the prior shape carried
# only a short ``reason`` category and the runner-level token
# never reached the bundle).
_PREDICTOR_TYPED_UNAVAILABLE_DIAGNOSTIC: str = "predictor_typed_unavailable"
_EXCLUDED_DIAGNOSTIC_VALUES: frozenset[str] = frozenset({
    "comparison_runner_join_key_unmatched",
    "comparison_contract_drift",
    _PREDICTOR_TYPED_UNAVAILABLE_DIAGNOSTIC,
})


SCHEMA_VERSION: int = 1
ALGORITHM_VERSION: str = "comparison_runner_v1"


# ---------------------------------------------------------------------
# Predictor design coverage — contract-pinned per-cell bucket map
# ---------------------------------------------------------------------
# The 13 headline-cohort cells partition into two buckets that
# describe whether the Smith-Eklund 2014 LRU stack-distance baseline
# directly models the dominant cache behavior of each cell:
#
# - ``predictor_friendly`` (8 cells) — the analytical baseline is
#   in-domain. Per-cell records carry this enum and the per-bucket
#   relative-error stats appear in
#   ``per_arch_diagnostics.per_arch.<arch>.relative_error_stats_per_predictor_design_coverage.predictor_friendly``.
# - ``predictor_hostile`` (5 cells) — the LRU baseline does NOT
#   model the dominant cache behavior (carveout / persisting-window
#   / prefetch-hint / cross-launch state / spill-to-local). Per-cell
#   records carry this enum and the per-bucket stats appear under
#   ``...predictor_hostile``.
#
# The mapping key is the cohort cell's ``(slug, shape_point)``
# tuple, matching the headline cohort's per-cell identity. Cells
# outside the headline cohort do NOT appear in the map; the runner
# emits ``predictor_design_coverage = None`` on those records (the
# field is always present so the on-disk shape stays
# deterministic).
#
# Shape-point alignment: the 8 microbench cells use
# ``shape_point="trace_l2_accuracy_xl"`` matching
# ``_HEADLINE_COHORT_13`` in
# ``tools/local_labels/run_headline_cohort_predict_compare.py``.
# The 5 activation cells use ``timing_shape`` or
# ``correctness_shape`` matching the same headline cohort. The
# programmatic lockstep test
# ``test_predictor_design_coverage_partition.py::test_contract_pin_keys_equal_headline_cohort_cells``
# enforces this equality. Codex Round-20 review P2 fix.
#
# Per-cell rationale (one line each, mirrors the rationale block
# in ``plans/cache_predictor_first_geomean_microbench_label_overlap_plan.md``):
_PREDICTOR_DESIGN_COVERAGE_BY_CELL: dict[tuple[str, str], str] = {
    # predictor_friendly (3 microbench + 5 activation = 8 cells)
    # working_set_capacity_sweep — sweeps working-set bytes across
    # L1/L2 boundaries; Smith-Eklund modeled-hit rule in-domain.
    ("working_set_capacity_sweep", "trace_l2_accuracy_xl"): "predictor_friendly",
    # l2_thrash_pointer_chase — pointer-chase exceeds L2 capacity;
    # LRU correctly predicts the near-zero hit-rate floor.
    ("l2_thrash_pointer_chase", "trace_l2_accuracy_xl"): "predictor_friendly",
    # dram_stream_stride_mix — predictable stride; per-launch reuse
    # distance histogram captures the access pattern faithfully.
    ("dram_stream_stride_mix", "trace_l2_accuracy_xl"): "predictor_friendly",
    # layernorm_fwd_bf16_sm80 — element-wise mean+var reduce + norm;
    # mostly streaming with O(N) reuse on the input vector.
    ("layernorm_fwd_bf16_sm80", "timing_shape"): "predictor_friendly",
    # rmsnorm_fwd_bf16_sm80 — element-wise RMS reduce + norm; same
    # access shape as layernorm.
    ("rmsnorm_fwd_bf16_sm80", "timing_shape"): "predictor_friendly",
    # silu_fwd_bf16_sm80/correctness_shape — small-N element-wise
    # SILU; LRU is still the right model class.
    ("silu_fwd_bf16_sm80", "correctness_shape"): "predictor_friendly",
    # silu_fwd_bf16_sm80/timing_shape — large-N element-wise SILU;
    # LRU in-domain.
    ("silu_fwd_bf16_sm80", "timing_shape"): "predictor_friendly",
    # softmax_safe_bf16_sm80 — multi-pass over the same vector;
    # high reuse; LRU in-domain.
    ("softmax_safe_bf16_sm80", "timing_shape"): "predictor_friendly",
    # predictor_hostile (5 microbench)
    # carveout_sensitivity_probe — sweeps L1 carveout ratio; the
    # predictor uses a single fixed cache_model_capacity_bytes per
    # arch with no carveout-aware capacity adjustment.
    ("carveout_sensitivity_probe", "trace_l2_accuracy_xl"): "predictor_hostile",
    # l2_persisting_window_sweep — exercises L2 persisting-allocation
    # policy; the LRU baseline has no persisting-window awareness.
    ("l2_persisting_window_sweep", "trace_l2_accuracy_xl"): "predictor_hostile",
    # l2_prefetch_hint_sweep — exercises prefetch-hint cache
    # modifiers; the LRU baseline does not model prefetch-induced
    # fills.
    ("l2_prefetch_hint_sweep", "trace_l2_accuracy_xl"): "predictor_hostile",
    # warm_cold_pair_relaunch — relaunch pair preserves cache state
    # across kernel boundaries; the LRU baseline resets per launch.
    ("warm_cold_pair_relaunch", "trace_l2_accuracy_xl"): "predictor_hostile",
    # register_spill_local_mem — spills to local memory (LDL/STL);
    # the predictor's sector_stream consumer does not separately
    # model the local-memory cache hierarchy.
    ("register_spill_local_mem", "trace_l2_accuracy_xl"): "predictor_hostile",
}
_PREDICTOR_DESIGN_COVERAGE_BUCKETS: tuple[str, ...] = (
    "predictor_friendly",
    "predictor_hostile",
)


# Default contract location. The loader resolves this when no
# explicit path is supplied. Resolution is lazy (computed inside
# the loader); module import does NOT touch the filesystem.
DEFAULT_CONTRACT_PATH: Path = (
    Path(__file__).resolve().parent / "comparison_contract.toml"
)


# Device-string-to-arch-token mapping. Both local equal-citizen
# devices appear here. Future devices land via additive entries
# without breaking the runner's contract: an unmapped device is
# treated as ``"unknown"`` and its bundles fail the join cleanly.
_DEVICE_TO_ARCH: dict[str, str] = {
    "NVIDIA GeForce RTX 5090": "sm_120",
    "NVIDIA GeForce RTX 5060 Laptop GPU": "sm_120",
    "NVIDIA GeForce RTX 3090 Ti": "sm_86",
}


# Numerical floor for log-space geomean stability when a per-cell
# relative_error rounds to exactly zero (predicted == ground_truth).
# log(0) is -inf which would collapse the geomean to zero across
# the entire admitted set; the floor preserves the geomean shape
# while still letting a near-perfect cell contribute close-to-zero
# weight. Pinned at 1e-12 (well below the contract's epsilon of
# 0.01); a smaller value would risk numerical underflow.
_GEOMEAN_LOG_FLOOR: float = 1e-12


# ---------------------------------------------------------------------
# Typed exceptions (each carries TOKEN + to_json mirroring the
# phase_cycle_ledger_consumer pattern)
# ---------------------------------------------------------------------


class ComparisonContractDrift(Exception):
    """Raised when the loaded contract carries a value that does
    not match a pinned constant, OR when a label cell's
    ``profiler_version`` differs from the contract's pinned
    ``ncu_metric_version``."""

    TOKEN = "comparison_contract_drift"

    def __init__(
        self,
        *,
        unmatched_field: str,
        contract_value: Any = None,
        observed_value: Any = None,
        detail: str = "",
    ) -> None:
        super().__init__(
            f"{self.TOKEN}: contract drift on field "
            f"{unmatched_field!r} "
            f"(contract={contract_value!r}, observed={observed_value!r})"
            + (f" — {detail}" if detail else "")
        )
        self.unmatched_field = unmatched_field
        self.contract_value = contract_value
        self.observed_value = observed_value
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "unmatched_field": self.unmatched_field,
            "contract_value": self.contract_value,
            "observed_value": self.observed_value,
            "detail": self.detail,
        }


class ComparisonRunnerUnknownResultKind(Exception):
    """Raised when a predictor bundle carries a ``result_kind`` value
    outside ``{scalar, bounded, typed_unavailable}``."""

    TOKEN = "comparison_runner_unknown_result_kind"

    def __init__(self, *, observed_value: Any, detail: str = "") -> None:
        super().__init__(
            f"{self.TOKEN}: unknown result_kind {observed_value!r}"
            + (f" — {detail}" if detail else "")
        )
        self.observed_value = observed_value
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "observed_value": self.observed_value,
            "detail": self.detail,
        }


class ComparisonRunnerPredictedHitRateOutOfRange(Exception):
    """Raised when a scalar predictor bundle's ``predicted_hit_rate``
    is outside ``[0, 1]``. Defensive second admission boundary; the
    predictor schema already enforces this at its own boundary."""

    TOKEN = "comparison_runner_predicted_hit_rate_out_of_range"

    def __init__(self, *, observed_value: float, detail: str = "") -> None:
        super().__init__(
            f"{self.TOKEN}: predicted_hit_rate={observed_value!r} "
            f"is outside [0, 1]"
            + (f" — {detail}" if detail else "")
        )
        self.observed_value = observed_value
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "observed_value": self.observed_value,
            "detail": self.detail,
        }


class ComparisonRunnerBoundedIntervalInvalid(Exception):
    """Raised when a ``result_kind=bounded`` predictor bundle's
    interval is malformed: ``hit_rate_lower > hit_rate_upper``,
    either bound outside ``[0, 1]``, or either bound non-finite
    (NaN, ±inf). Defensive second admission boundary; the
    bounded-arm canonical (midpoint) is well-defined
    arithmetically for any pair of floats but admitting an
    interval that declares an empty set OR exits the [0, 1]
    band would publish a schema-valid but operationally
    nonsensical comparison result. Mirrors the scalar arm's
    ``ComparisonRunnerPredictedHitRateOutOfRange`` discipline."""

    TOKEN = "comparison_runner_bounded_interval_invalid"

    def __init__(
        self,
        *,
        hit_rate_lower: Any,
        hit_rate_upper: Any,
        detail: str = "",
    ) -> None:
        super().__init__(
            f"{self.TOKEN}: bounded interval "
            f"[lower={hit_rate_lower!r}, "
            f"upper={hit_rate_upper!r}] is invalid"
            + (f" — {detail}" if detail else "")
        )
        self.hit_rate_lower = hit_rate_lower
        self.hit_rate_upper = hit_rate_upper
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "hit_rate_lower": self.hit_rate_lower,
            "hit_rate_upper": self.hit_rate_upper,
            "detail": self.detail,
        }


class ComparisonRunnerDuplicatePredictorBundle(Exception):
    """Raised when two or more predictor bundles in the input list
    share the same 7-field join key (slug, shape_point,
    binary_sha256, source_commit, kernel_signature_sha256, arch,
    ncu_metric_version). Defensive admission boundary: the
    contract's per-arch admission floor is per cell, so admitting
    duplicates would let N-1 unique cells plus one duplicate
    falsely satisfy the floor and overweight the duplicated
    cell's relative_error in the headline geomean. Most often
    surfaces when the operator's CLI glob
    (``--predictor-bundles-glob``) double-matches the same
    bundle file (e.g., a stale + fresh copy under different
    paths, or a wildcard that resolves through a symlink and a
    real path). Mirrors the
    ``ComparisonRunnerBoundedIntervalInvalid`` discipline (Round
    18): a structurally-valid input that would publish a
    schema-valid but semantically-incorrect comparison result."""

    TOKEN = "comparison_runner_duplicate_predictor_bundle"

    def __init__(
        self,
        *,
        join_key: tuple[str, ...],
        slug: str = "",
        shape_point: str = "",
        detail: str = "",
    ) -> None:
        super().__init__(
            f"{self.TOKEN}: predictor bundle 7-field join_key "
            f"{join_key!r} (slug={slug!r}, shape_point="
            f"{shape_point!r}) appears more than once in the "
            f"input list"
            + (f" — {detail}" if detail else "")
        )
        self.join_key = join_key
        self.slug = slug
        self.shape_point = shape_point
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "join_key": list(self.join_key),
            "slug": self.slug,
            "shape_point": self.shape_point,
            "detail": self.detail,
        }


class ComparisonRunnerTypedReasonOutOfEnum(Exception):
    """Raised when a typed_unavailable predictor bundle carries a
    ``typed_reason`` value not in
    ``cache_predictor._TYPED_REASONS``. The reason is structurally
    valid (a string) but operationally unknown to this runner
    version."""

    TOKEN = "comparison_runner_typed_reason_out_of_enum"

    def __init__(
        self, *, observed_value: Any, detail: str = ""
    ) -> None:
        super().__init__(
            f"{self.TOKEN}: typed_reason {observed_value!r} "
            f"not in cache_predictor._TYPED_REASONS"
            + (f" — {detail}" if detail else "")
        )
        self.observed_value = observed_value
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "observed_value": self.observed_value,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------
# Runtime contract object
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageGate:
    smoke_test_floor_per_arch: int
    publication_grade_floor_per_arch: int
    aspirational_target_total: int
    require_both_arches_for_cross_arch: bool


@dataclass(frozen=True)
class PerArchDiagnosticsConfig:
    """Per-arch diagnostics config pinned in
    ``[per_arch_diagnostics_config]``. The loader enforces
    set-equality between ``diagnostic_vocabulary`` and the runner's
    ``_EXCLUDED_DIAGNOSTIC_VALUES`` constant; drift raises
    ``ComparisonContractDrift`` before any cell is processed."""

    diagnostic_vocabulary: tuple[str, ...]
    quartile_min_n: int


_PINNED_PER_ARCH_DIAGNOSTICS_QUARTILE_MIN_N: int = 4
_PER_ARCH_DIAGNOSTICS_CONFIG_KNOWN_KEYS: frozenset[str] = frozenset({
    "diagnostic_vocabulary",
    "quartile_min_n",
})


def _build_per_arch_diagnostics_config_or_drift(
    section: dict | None,
) -> PerArchDiagnosticsConfig:
    """Validate the loaded ``[per_arch_diagnostics_config]`` section
    and return the corresponding ``PerArchDiagnosticsConfig``. Drift
    raises ``ComparisonContractDrift`` so the runner fails-closed at
    the loader boundary."""
    if section is None:
        raise ComparisonContractDrift(
            unmatched_field="per_arch_diagnostics_config",
            contract_value=None,
            observed_value="<missing section>",
            detail=(
                "contract must carry [per_arch_diagnostics_config]; "
                "the runner relies on the pinned diagnostic_vocabulary "
                "+ quartile_min_n at load time"
            ),
        )
    extra_keys = set(section.keys()) - _PER_ARCH_DIAGNOSTICS_CONFIG_KNOWN_KEYS
    if extra_keys:
        raise ComparisonContractDrift(
            unmatched_field="per_arch_diagnostics_config",
            contract_value=sorted(_PER_ARCH_DIAGNOSTICS_CONFIG_KNOWN_KEYS),
            observed_value=sorted(extra_keys),
            detail=(
                "[per_arch_diagnostics_config] contains unknown key(s); "
                "the runner pins the section to a closed schema"
            ),
        )
    raw_vocab = section.get("diagnostic_vocabulary")
    if not isinstance(raw_vocab, list) or not all(
        isinstance(v, str) for v in raw_vocab
    ):
        raise ComparisonContractDrift(
            unmatched_field="diagnostic_vocabulary",
            contract_value="list[str]",
            observed_value=raw_vocab,
            detail=(
                "[per_arch_diagnostics_config].diagnostic_vocabulary "
                "must be a list of strings"
            ),
        )
    vocab_tuple = tuple(raw_vocab)
    if set(vocab_tuple) != _EXCLUDED_DIAGNOSTIC_VALUES:
        raise ComparisonContractDrift(
            unmatched_field="diagnostic_vocabulary",
            contract_value=sorted(_EXCLUDED_DIAGNOSTIC_VALUES),
            observed_value=sorted(vocab_tuple),
            detail=(
                "[per_arch_diagnostics_config].diagnostic_vocabulary "
                "must match _EXCLUDED_DIAGNOSTIC_VALUES exactly "
                "(set equality)"
            ),
        )
    quartile_min_n = section.get("quartile_min_n")
    if quartile_min_n != _PINNED_PER_ARCH_DIAGNOSTICS_QUARTILE_MIN_N:
        raise ComparisonContractDrift(
            unmatched_field="quartile_min_n",
            contract_value=_PINNED_PER_ARCH_DIAGNOSTICS_QUARTILE_MIN_N,
            observed_value=quartile_min_n,
            detail=(
                "[per_arch_diagnostics_config].quartile_min_n is pinned "
                "at the small-n threshold above which q25/q50/q75 emit"
            ),
        )
    return PerArchDiagnosticsConfig(
        diagnostic_vocabulary=vocab_tuple,
        quartile_min_n=int(quartile_min_n),
    )


@dataclass(frozen=True)
class ComparisonContract:
    """In-memory representation of ``comparison_contract.toml`` after
    pinned-value validation. The loader returns this; the runner
    consumes this. Tests can construct one directly to exercise
    alternative paths without round-tripping through TOML."""

    ncu_metric_name: str
    ncu_metric_version: str
    percent_to_fraction: bool
    epsilon: float
    bootstrap_resamples: int
    ci_confidence: float
    bootstrap_unit: str
    bounded_arm_handling: str
    typed_unavailable_handling: str
    join_key: tuple[str, ...]
    coverage_gate: CoverageGate
    log_space_geomean: bool
    per_arch_diagnostics_config: PerArchDiagnosticsConfig | None = None
    arch_set: tuple[str, ...] = field(default_factory=tuple)
    cache_layers: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------
# Contract loader
# ---------------------------------------------------------------------


def load_comparison_contract(
    path: Path | None = None,
) -> ComparisonContract:
    """Load the comparison contract from disk and validate every
    pinned constant. ``path=None`` resolves to
    ``DEFAULT_CONTRACT_PATH``. Drift in any pinned field raises
    ``ComparisonContractDrift`` with the offending field name in
    ``unmatched_field``."""
    if path is None:
        path = DEFAULT_CONTRACT_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    error_formula = raw.get("error_formula") or {}
    geomean_section = raw.get("geomean") or {}
    coverage_gate_section = raw.get("coverage_gate") or {}
    ncu_metric = raw.get("ncu_metric") or {}
    join_key_section = raw.get("join_key") or {}
    scope = raw.get("scope") or {}
    per_arch_diagnostics_section = raw.get("per_arch_diagnostics_config")
    per_arch_diagnostics_cfg = _build_per_arch_diagnostics_config_or_drift(
        per_arch_diagnostics_section,
    )

    epsilon = float(error_formula.get("epsilon", -1))
    if epsilon != _PINNED_EPSILON:
        raise ComparisonContractDrift(
            unmatched_field="epsilon",
            contract_value=epsilon,
            observed_value=_PINNED_EPSILON,
            detail="contract epsilon must equal the pinned floor",
        )

    bootstrap_resamples = int(geomean_section.get("bootstrap_resamples", -1))
    if bootstrap_resamples != _PINNED_BOOTSTRAP_RESAMPLES:
        raise ComparisonContractDrift(
            unmatched_field="bootstrap_resamples",
            contract_value=bootstrap_resamples,
            observed_value=_PINNED_BOOTSTRAP_RESAMPLES,
            detail="contract bootstrap_resamples must equal the pinned floor",
        )

    ci_confidence = float(geomean_section.get("ci_confidence", -1))
    if ci_confidence != _PINNED_CI_CONFIDENCE:
        raise ComparisonContractDrift(
            unmatched_field="ci_confidence",
            contract_value=ci_confidence,
            observed_value=_PINNED_CI_CONFIDENCE,
            detail="contract ci_confidence must equal the pinned floor",
        )

    join_fields = tuple(join_key_section.get("fields") or ())
    if join_fields != _PINNED_JOIN_KEY:
        raise ComparisonContractDrift(
            unmatched_field="join_key",
            contract_value=list(join_fields),
            observed_value=list(_PINNED_JOIN_KEY),
            detail=(
                "contract join_key.fields must exactly match the "
                "pinned 8-field tuple in declaration order"
            ),
        )

    ncu_metric_version = str(ncu_metric.get("ncu_metric_version", ""))
    if ncu_metric_version != _PINNED_NCU_METRIC_VERSION:
        raise ComparisonContractDrift(
            unmatched_field="ncu_metric_version",
            contract_value=ncu_metric_version,
            observed_value=_PINNED_NCU_METRIC_VERSION,
            detail=(
                "contract ncu_metric_version is the protocol pin; "
                "drift must be a deliberate coordinated update of "
                "the contract, the runner pin, and the label-side "
                "profiler_version cross-check"
            ),
        )

    bounded_arm_handling = str(
        geomean_section.get("bounded_arm_handling", "")
    )
    if bounded_arm_handling != _PINNED_BOUNDED_ARM_HANDLING:
        raise ComparisonContractDrift(
            unmatched_field="bounded_arm_handling",
            contract_value=bounded_arm_handling,
            observed_value=_PINNED_BOUNDED_ARM_HANDLING,
            detail=(
                "contract bounded_arm_handling pins the bounded-arm "
                "policy; drift requires a coordinated walk through "
                "the schema, the runner's per-cell record fields, "
                "and the test surface"
            ),
        )

    smoke_floor = int(
        coverage_gate_section.get("smoke_test_floor_per_arch", -1)
    )
    pub_floor = int(
        coverage_gate_section.get("publication_grade_floor_per_arch", -1)
    )
    target_total = int(
        coverage_gate_section.get("aspirational_target_total", -1)
    )
    if (
        smoke_floor != _PINNED_SMOKE_TEST_FLOOR_PER_ARCH
        or pub_floor != _PINNED_PUBLICATION_GRADE_FLOOR_PER_ARCH
        or target_total != _PINNED_ASPIRATIONAL_TARGET_TOTAL
    ):
        raise ComparisonContractDrift(
            unmatched_field="coverage_gate",
            contract_value={
                "smoke_test_floor_per_arch": smoke_floor,
                "publication_grade_floor_per_arch": pub_floor,
                "aspirational_target_total": target_total,
            },
            observed_value={
                "smoke_test_floor_per_arch": _PINNED_SMOKE_TEST_FLOOR_PER_ARCH,
                "publication_grade_floor_per_arch": _PINNED_PUBLICATION_GRADE_FLOOR_PER_ARCH,
                "aspirational_target_total": _PINNED_ASPIRATIONAL_TARGET_TOTAL,
            },
            detail=(
                "contract coverage_gate thresholds must equal the "
                "pinned floors"
            ),
        )

    # require_both_arches_for_cross_arch must be present and a true
    # boolean (TOML's bool, not int / str). Missing or non-bool values
    # raise ComparisonContractDrift; bool is an int subclass in Python
    # so the type check uses ``type(...) is bool`` to reject 0/1 ints
    # that masquerade as booleans.
    if "require_both_arches_for_cross_arch" not in coverage_gate_section:
        raise ComparisonContractDrift(
            unmatched_field="require_both_arches_for_cross_arch",
            contract_value=None,
            observed_value="<missing>",
            detail=(
                "contract [coverage_gate] must declare "
                "require_both_arches_for_cross_arch as a boolean"
            ),
        )
    require_both_arches_value = coverage_gate_section.get(
        "require_both_arches_for_cross_arch"
    )
    if type(require_both_arches_value) is not bool:
        raise ComparisonContractDrift(
            unmatched_field="require_both_arches_for_cross_arch",
            contract_value=type(require_both_arches_value).__name__,
            observed_value="bool",
            detail=(
                "contract [coverage_gate].require_both_arches_for_cross_arch "
                "must be a TOML boolean (true/false), not int or string"
            ),
        )

    return ComparisonContract(
        ncu_metric_name=str(ncu_metric.get("name", "")),
        ncu_metric_version=ncu_metric_version,
        percent_to_fraction=bool(
            ncu_metric.get("percent_to_fraction", True)
        ),
        epsilon=epsilon,
        bootstrap_resamples=bootstrap_resamples,
        ci_confidence=ci_confidence,
        bootstrap_unit=str(geomean_section.get("bootstrap_unit", "cell")),
        bounded_arm_handling=bounded_arm_handling,
        typed_unavailable_handling=str(
            geomean_section.get("typed_unavailable_handling", "exclude"),
        ),
        join_key=join_fields,
        coverage_gate=CoverageGate(
            smoke_test_floor_per_arch=smoke_floor,
            publication_grade_floor_per_arch=pub_floor,
            aspirational_target_total=target_total,
            require_both_arches_for_cross_arch=require_both_arches_value,
        ),
        log_space_geomean=bool(
            error_formula.get("log_space_geomean", True)
        ),
        per_arch_diagnostics_config=per_arch_diagnostics_cfg,
        arch_set=tuple(scope.get("arch_set") or ()),
        cache_layers=tuple(scope.get("cache_layers") or ()),
    )


# ---------------------------------------------------------------------
# Pre-validation helpers
# ---------------------------------------------------------------------


def _arch_for_device(device: str) -> str:
    return _DEVICE_TO_ARCH.get(device, "unknown")


def _validate_predictor_bundle_shape(bundle: dict) -> None:
    """Pre-flight admission gate. Hard-rejects bundles whose
    ``result_kind`` is unknown, whose scalar ``predicted_hit_rate``
    is out of range, whose typed_unavailable ``typed_reason`` is
    not in the predictor's enum, or whose ``cache_layer`` (when
    present) is outside the closed-enum set. Each rejection raises
    a typed exception with TOKEN matching ``_RUNNER_DEFECT_TOKENS``.

    Imports ``cache_predictor`` lazily so module-import remains
    IO-free and the unit-test surface can decouple test fixtures
    from observation_builder dependencies."""
    cache_layer = bundle.get("cache_layer")
    if (
        cache_layer is not None
        and cache_layer not in _BUNDLE_CACHE_LAYER_ENUM
    ):
        raise ComparisonContractDrift(
            unmatched_field="cache_layer",
            contract_value=sorted(_BUNDLE_CACHE_LAYER_ENUM),
            observed_value=cache_layer,
            detail=(
                "bundle cache_layer must be one of the closed-enum "
                "values (L1TEX, L2) or omitted for legacy L2 bundles"
            ),
        )
    result_kind = bundle.get("result_kind")
    if result_kind not in {"scalar", "bounded", "typed_unavailable"}:
        raise ComparisonRunnerUnknownResultKind(
            observed_value=result_kind,
            detail="result_kind must be one of {scalar, bounded, typed_unavailable}",
        )
    if result_kind == "scalar":
        rate = bundle.get("predicted_hit_rate")
        try:
            rate_f = float(rate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rate_f = math.nan
        if not (0.0 <= rate_f <= 1.0):
            raise ComparisonRunnerPredictedHitRateOutOfRange(
                observed_value=rate_f if not math.isnan(rate_f) else rate,
                detail="scalar predicted_hit_rate must be in [0, 1]",
            )
    if result_kind == "bounded":
        lower_raw = bundle.get("hit_rate_lower")
        upper_raw = bundle.get("hit_rate_upper")
        try:
            lower_f = float(lower_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            lower_f = math.nan
        try:
            upper_f = float(upper_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            upper_f = math.nan
        if math.isnan(lower_f) or math.isinf(lower_f):
            raise ComparisonRunnerBoundedIntervalInvalid(
                hit_rate_lower=(
                    lower_raw if math.isnan(lower_f) else lower_f
                ),
                hit_rate_upper=upper_raw,
                detail=(
                    "hit_rate_lower must be a finite float in [0, 1]"
                ),
            )
        if math.isnan(upper_f) or math.isinf(upper_f):
            raise ComparisonRunnerBoundedIntervalInvalid(
                hit_rate_lower=lower_raw,
                hit_rate_upper=(
                    upper_raw if math.isnan(upper_f) else upper_f
                ),
                detail=(
                    "hit_rate_upper must be a finite float in [0, 1]"
                ),
            )
        if not (0.0 <= lower_f <= 1.0):
            raise ComparisonRunnerBoundedIntervalInvalid(
                hit_rate_lower=lower_f,
                hit_rate_upper=upper_f,
                detail=(
                    "hit_rate_lower out of range; bounded interval "
                    "endpoints must be in [0, 1]"
                ),
            )
        if not (0.0 <= upper_f <= 1.0):
            raise ComparisonRunnerBoundedIntervalInvalid(
                hit_rate_lower=lower_f,
                hit_rate_upper=upper_f,
                detail=(
                    "hit_rate_upper out of range; bounded interval "
                    "endpoints must be in [0, 1]"
                ),
            )
        if lower_f > upper_f:
            raise ComparisonRunnerBoundedIntervalInvalid(
                hit_rate_lower=lower_f,
                hit_rate_upper=upper_f,
                detail=(
                    "hit_rate_lower > hit_rate_upper; the interval "
                    "declares an empty set"
                ),
            )
    if result_kind == "typed_unavailable":
        typed_reason = bundle.get("typed_reason")
        # Lazy import keeps module import IO-free.
        suite_root = Path(__file__).resolve().parent
        if str(suite_root) not in sys.path:
            sys.path.insert(0, str(suite_root))
        import cache_predictor as cp  # type: ignore[import-not-found]
        # Admit BOTH the predictor-emitted reasons (cp._TYPED_REASONS) and the
        # reasons the comparison runner injects on bundles it mutates
        # (_CR_INJECTED_TYPED_REASONS). The runner-injected reasons are not in
        # the predictor's producer enum on purpose (see _CR_INJECTED_TYPED_REASONS).
        if typed_reason not in (cp._TYPED_REASONS | _CR_INJECTED_TYPED_REASONS):
            raise ComparisonRunnerTypedReasonOutOfEnum(
                observed_value=typed_reason,
                detail=(
                    "typed_reason must be a member of "
                    "cache_predictor._TYPED_REASONS or "
                    "comparison_runner._CR_INJECTED_TYPED_REASONS"
                ),
            )


# The five bundle-side join-key fields the runner reads from
# ``manifest_pairing_index``. The full 7-tuple join key is these
# five plus ``arch`` (derived from ``device``) plus
# ``ncu_metric_version`` (the contract's pinned value). A bundle
# whose ``manifest_pairing_index`` is absent, not a dict, or missing
# any of these five non-empty values cannot have its 7-tuple join
# key resolved; the runner emits ``comparison_contract_drift`` with
# ``unmatched_field = "manifest_pairing_index"`` before the join is
# attempted.
_BUNDLE_JOIN_KEY_FIELDS: tuple[str, ...] = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)


def _bundle_has_valid_manifest_pairing_index(bundle: dict) -> bool:
    """Return True iff the bundle carries a ``manifest_pairing_index``
    dict whose five required fields are present and non-empty
    strings. False when the field is missing, not a dict, or any of
    the five required fields is absent / non-string / empty."""
    mpi = bundle.get("manifest_pairing_index")
    if not isinstance(mpi, dict):
        return False
    for field_name in _BUNDLE_JOIN_KEY_FIELDS:
        v = mpi.get(field_name)
        if not isinstance(v, str) or v == "":
            return False
    return True


def _bundle_join_key(
    bundle: dict, contract: ComparisonContract,
) -> tuple[str, ...]:
    """Build a bundle's 8-tuple join key. The first 5 fields come
    from ``manifest_pairing_index``; ``arch`` is derived from
    ``device``; ``ncu_metric_version`` is the contract's pinned
    value (the predictor doesn't know it); ``cache_layer`` reads
    the bundle's optional ``cache_layer`` enum, defaulting to ``L2``
    for legacy bundles that pre-date the layer-aware migration."""
    mpi = bundle.get("manifest_pairing_index") or {}
    return (
        str(mpi.get("slug", "")),
        str(mpi.get("shape_point", "")),
        str(mpi.get("binary_sha256", "")),
        str(mpi.get("source_commit", "")),
        str(mpi.get("kernel_signature_sha256", "")),
        _arch_for_device(str(bundle.get("device", ""))),
        contract.ncu_metric_version,
        str(bundle.get("cache_layer", "L2")),
    )


def _cell_join_key(cell: dict, cache_layer: str = "L2") -> tuple[str, ...]:
    """Build a cell's 8-tuple join key. The first 7 fields come
    from the admitted-cell record; ``cache_layer`` is broadcast per
    contract-pinned layer (each cell yields one synthetic join key
    per layer in the contract's ``cache_layers``)."""
    return (
        str(cell.get("slug", "")),
        str(cell.get("shape_point", "")),
        str(cell.get("binary_sha256", "")),
        str(cell.get("source_commit", "")),
        str(cell.get("kernel_signature_sha256", "")),
        str(cell.get("arch", "")),
        str(cell.get("profiler_version", "")),
        str(cache_layer),
    )


def _join_key_dict(key: tuple[str, ...]) -> dict[str, str]:
    return {
        "slug": key[0],
        "shape_point": key[1],
        "binary_sha256": key[2],
        "source_commit": key[3],
        "kernel_signature_sha256": key[4],
        "arch": key[5],
        "ncu_metric_version": key[6],
        "cache_layer": key[7] if len(key) > 7 else "L2",
    }


def _find_unmatched_field(
    bundle_key: tuple[str, ...],
    cell_keys: list[tuple[str, ...]],
) -> str:
    """Search for the cell with the highest field-match overlap
    against the bundle's 7-tuple key. Return the FIRST field name
    (in ``_PINNED_JOIN_KEY`` declaration order) where the best-match
    cell differs. If no cells exist, return ``"unknown"``."""
    if not cell_keys:
        return "unknown"
    best_diffs: list[int] | None = None
    best_cell: tuple[str, ...] | None = None
    for cell_key in cell_keys:
        diffs = [
            i for i in range(len(bundle_key))
            if bundle_key[i] != cell_key[i]
        ]
        if best_diffs is None or len(diffs) < len(best_diffs):
            best_diffs = diffs
            best_cell = cell_key
    if best_diffs is None or not best_diffs:
        return "unknown"
    return _PINNED_JOIN_KEY[best_diffs[0]]


# ---------------------------------------------------------------------
# Per-cell relative-error helpers
# ---------------------------------------------------------------------


def _rel_err(predicted: float, ground_truth: float, epsilon: float) -> float:
    return abs(predicted - ground_truth) / max(ground_truth, epsilon)


def _percent_to_fraction(pct: float) -> float:
    return pct / 100.0


# ---------------------------------------------------------------------
# Bootstrap CI on the headline geomean (cell-level resampling)
# ---------------------------------------------------------------------


def _safe_log(x: float) -> float:
    return math.log(max(x, _GEOMEAN_LOG_FLOOR))


def _log_space_geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sum(_safe_log(v) for v in values)
    return math.exp(s / len(values))


def _bootstrap_ci_over_cells(
    per_cell_relative_errors: list[float],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """Cell-level bootstrap CI on the log-space geomean. Resampling
    unit is the cell; seed is deterministic so byte-identical
    re-runs produce byte-identical CIs."""
    n = len(per_cell_relative_errors)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        v = _log_space_geomean(per_cell_relative_errors)
        return (v, v, 0.0)
    rng = random.Random(seed)
    resamples: list[float] = []
    for _ in range(n_resamples):
        sample = [
            per_cell_relative_errors[rng.randrange(n)] for _ in range(n)
        ]
        resamples.append(_log_space_geomean(sample))
    resamples.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int(math.floor((alpha / 2.0) * n_resamples)))
    upper_idx = min(
        n_resamples - 1,
        int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1,
    )
    ci_low = resamples[lower_idx]
    ci_high = resamples[upper_idx]
    return (ci_low, ci_high, (ci_high - ci_low) / 2.0)


# ---------------------------------------------------------------------
# Per-arch and unattributed excluded-reason tally helpers
# ---------------------------------------------------------------------


def _tally_excluded_diagnostic_counts(
    records: list[dict],
) -> dict[str, int]:
    """Walk a list of excluded-cell records and tally by diagnostic
    using the three-token ``_EXCLUDED_DIAGNOSTIC_VALUES`` vocabulary.
    Returns a dict with all three tokens as keys (zero defaults). A
    record whose ``diagnostic`` is not a member of the vocabulary
    raises ``ComparisonContractDrift`` before any tally is updated;
    the regression guard catches vocabulary drift the schema-side
    enum enforcement would also surface, but on the producer-side
    boundary."""
    counts: dict[str, int] = {
        token: 0 for token in _EXCLUDED_DIAGNOSTIC_VALUES
    }
    for record in records:
        diagnostic = record.get("diagnostic")
        if diagnostic not in _EXCLUDED_DIAGNOSTIC_VALUES:
            raise ComparisonContractDrift(
                unmatched_field="diagnostic",
                contract_value=sorted(_EXCLUDED_DIAGNOSTIC_VALUES),
                observed_value=diagnostic,
                detail=(
                    "excluded-cell diagnostic must be a member of "
                    "_EXCLUDED_DIAGNOSTIC_VALUES"
                ),
            )
        counts[diagnostic] += 1
    return counts


def _per_arch_excluded_counts(
    excluded_cells_for_arch: list[dict],
) -> dict[str, int]:
    """Tally a per-arch slice of excluded-cell records (records with
    ``join_key.arch == arch``) by diagnostic. Caller is responsible
    for filtering the records to the per-arch slice before the call."""
    return _tally_excluded_diagnostic_counts(excluded_cells_for_arch)


def _unattributed_excluded_counts(
    excluded_cells_lacking_arch: list[dict],
) -> dict[str, int]:
    """Tally excluded-cell records lacking ``join_key.arch`` by
    diagnostic. Returns the same shape as
    ``_per_arch_excluded_counts``: the three-token vocabulary with
    non-negative integer counts (zero defaults)."""
    return _tally_excluded_diagnostic_counts(excluded_cells_lacking_arch)


# ---------------------------------------------------------------------
# Per-arch statistics + diagnostics wrapper
# ---------------------------------------------------------------------


def _value_list_stats(values: list[float]) -> dict | None:
    """Generic small-n-aware stats helper. Consumed by per-arch
    relative-error stats, signed-residual stats, and bounded-arm
    lower/upper stats. Same semantics in every consumer:

    - ``None`` when n == 0 (the no_cells branch).
    - ``{mean, stdev, min, max}`` for 1 <= n <= 3 (insufficient
      branch). n == 1 stdev is pinned to 0.0 because Python's
      ``statistics.stdev()`` raises StatisticsError for n == 1.
    - ``{mean, stdev, min, max, q25, q50, q75}`` for n >= 4
      (quartile branch). Quartile method is
      ``statistics.quantiles(data, n=4, method="exclusive")``
      (Python default; Tukey hinges).
    """
    n = len(values)
    if n == 0:
        return None
    if n >= 2:
        stdev_v = statistics.stdev(values)
    else:
        stdev_v = 0.0
    block: dict = {
        "mean": statistics.fmean(values),
        "stdev": stdev_v,
        "min": min(values),
        "max": max(values),
    }
    if n >= 4:
        quartiles = statistics.quantiles(
            values, n=4, method="exclusive",
        )
        block["q25"] = quartiles[0]
        block["q50"] = quartiles[1]
        block["q75"] = quartiles[2]
    return block


def _status_for_n(n: int) -> str:
    """Map a cell-count to the canonical statistics_status enum
    value. Used by every per-arch stats sub-block (relative_error,
    signed_residual, bounded-arm lower/upper) so the small-n
    behavior stays consistent."""
    if n == 0:
        return "no_cells"
    if n >= 4:
        return "quartile_emitted"
    return "insufficient_cells_for_quartile_statistics"


def _per_arch_stats(
    included_cells_for_arch: list[dict],
    contract: ComparisonContract,
) -> dict | None:
    """Compute per-arch statistics on the relative_error values
    extracted from an arch-filtered slice of included cells.
    Delegates to ``_value_list_stats``."""
    values = [
        float(c["relative_error"]) for c in included_cells_for_arch
    ]
    return _value_list_stats(values)


def _per_arch_signed_residual_stats(
    included_cells_for_arch: list[dict],
) -> dict | None:
    """Compute per-arch signed-residual statistics. Residual is
    defined as ``predicted_hit_rate - ground_truth_fraction`` so
    values can be negative. Same small-n shape as
    ``_per_arch_stats``: None for no_cells, 4-field block for
    1 <= n <= 3, 7-field block for n >= 4."""
    values = [
        float(c["predicted_hit_rate"]) - float(c["ground_truth_fraction"])
        for c in included_cells_for_arch
    ]
    return _value_list_stats(values)


def _per_bucket_relative_error_stats(
    included_cells_for_arch: list[dict],
) -> dict[str, dict]:
    """Compute per-arch, per-predictor-design-coverage-bucket
    relative-error statistics. The returned dict has one key per
    bucket in ``_PREDICTOR_DESIGN_COVERAGE_BUCKETS``; each value is
    ``{statistics_status, stats}`` mirroring the per-arch outer
    statistics_status semantics so empty buckets emit
    ``{statistics_status: "no_cells", stats: None}`` and populated
    buckets emit the small-n-aware shape from
    ``_value_list_stats``.

    Cells whose ``predictor_design_coverage`` is None (out of the
    headline cohort) are filtered out of every bucket — they
    contribute to the per-arch outer stats but not to either
    bucket's per-bucket stats. This keeps the bucket aggregation
    bound to the contract-pinned cohort partition."""
    block: dict[str, dict] = {}
    for bucket in _PREDICTOR_DESIGN_COVERAGE_BUCKETS:
        bucket_cells = [
            c for c in included_cells_for_arch
            if c.get("predictor_design_coverage") == bucket
        ]
        n = len(bucket_cells)
        values = [float(c["relative_error"]) for c in bucket_cells]
        block[bucket] = {
            "statistics_status": _status_for_n(n),
            "stats": _value_list_stats(values),
        }
    return block


def _bounded_arm_relative_error_stats(
    included_cells_for_arch: list[dict],
) -> dict:
    """Compute per-arch bounded-arm lower/upper relative-error
    statistics. Aggregates only over included cells with
    ``predictor_result_kind == "bounded"``; scalar cells are
    excluded from the bounded-arm sample by definition.

    The returned block is always present with:

    - ``n_bounded_cells``: count of bounded-arm cells in this slice.
    - ``lower``: ``{statistics_status, stats}`` for the per-cell
      ``relative_error_lower_diagnostic`` values.
    - ``upper``: ``{statistics_status, stats}`` for the per-cell
      ``relative_error_upper_diagnostic`` values.

    Each arm's ``statistics_status`` is driven by the bounded-cell
    count and is independent of the per-arch outer
    ``statistics_status`` (which is driven by n_headline_cells over
    BOTH scalar and bounded cells)."""
    bounded_cells = [
        c for c in included_cells_for_arch
        if c.get("predictor_result_kind") == "bounded"
    ]
    n_bounded = len(bounded_cells)
    lower_values = [
        float(c["relative_error_lower_diagnostic"]) for c in bounded_cells
    ]
    upper_values = [
        float(c["relative_error_upper_diagnostic"]) for c in bounded_cells
    ]
    return {
        "n_bounded_cells": n_bounded,
        "lower": {
            "statistics_status": _status_for_n(n_bounded),
            "stats": _value_list_stats(lower_values),
        },
        "upper": {
            "statistics_status": _status_for_n(n_bounded),
            "stats": _value_list_stats(upper_values),
        },
    }


def _per_arch_diagnostics(
    included_cells: list[dict],
    excluded_cells: list[dict],
    arch_set: tuple[str, ...] | list[str],
    contract: ComparisonContract,
    bootstrap_seed: int = 0,
) -> dict:
    """Aggregate per-arch summary statistics, exclusion-token tallies,
    and a coverage-gate verdict into the wrapper shape enforced by
    ``comparison_bundle.schema.json``'s ``per_arch_diagnostics``
    sub-schema. Every arch in ``arch_set`` appears as a key in
    ``per_arch`` with deterministic shape (zero-count arches produce
    no_cells records). Records lacking ``join_key.arch`` tally only
    into ``unattributed_excluded_reason_counts``.

    The coverage_gate_verdict reports the live-only admission floor:
    cells with ``evidence_kind == "live"`` count toward the floor;
    synthetic cells do not.

    The ``relative_error_geomean_ci`` per-arch block is computed via
    cell-level bootstrap resampling on the log-space geomean using
    the supplied ``bootstrap_seed`` (so byte-identical re-runs with
    the same seed produce byte-identical CIs). For zero-headline-cell
    arches the block is the deterministic empty-sample object
    ``{geomean: 0.0, ci_low: 0.0, ci_high: 0.0, ci_half_width: 0.0,
    bootstrap_resamples: contract.bootstrap_resamples,
    bootstrap_seed: <seed>}`` — never ``None``. The empty-sample
    helpers ``_log_space_geomean`` and ``_bootstrap_ci_over_cells``
    return zeros for an empty input list, so the four numeric CI
    fields (``geomean``, ``ci_low``, ``ci_high``, ``ci_half_width``)
    stay zero regardless of seed when no cells are present; the
    recorded ``bootstrap_seed`` field still follows the caller's
    input.

    Determinism contract: ``per_arch`` keys are inserted in
    ``sorted(arch_set)`` order so the on-disk JSON (written via
    ``write_bundle()`` using ``json.dump(..., sort_keys=True)``) is
    byte-deterministic regardless of the caller's ``arch_set`` input
    order. The same property holds for nested dicts via the global
    sort-keys serialization. The regression-guard tests in
    ``test_per_arch_diagnostics_determinism.py`` lock the contract.
    """
    floor = contract.coverage_gate.smoke_test_floor_per_arch

    excluded_by_arch: dict[str, list[dict]] = {a: [] for a in arch_set}
    excluded_unattributed: list[dict] = []
    # Layer-aware pre-flight reasons are not "comparable cell"
    # rejections (the runner never reached the join attempt). Skip
    # them in the per-arch comparable count so the top-level
    # comparable_cells_per_arch surface and the per-arch
    # n_comparable_cells surface agree. The records still appear in
    # the on-disk excluded_cells list for operator audit.
    _pre_flight_exclusion_reasons = {
        LOCAL_LABEL_ADMITTED_L1_MEAN_MIGRATION_REQUIRED,
        COMPARISON_RUNNER_L1TEX_LABEL_MEAN_MISSING,
    }
    for record in excluded_cells:
        if str(record.get("reason", "")) in _pre_flight_exclusion_reasons:
            continue
        join_key = record.get("join_key") or {}
        arch = (
            join_key.get("arch") if isinstance(join_key, dict) else None
        )
        if not arch:
            excluded_unattributed.append(record)
            continue
        if arch in excluded_by_arch:
            excluded_by_arch[arch].append(record)
        else:
            excluded_unattributed.append(record)

    included_by_arch: dict[str, list[dict]] = {a: [] for a in arch_set}
    for cell in included_cells:
        join_key = cell.get("join_key") or {}
        arch = (
            join_key.get("arch") if isinstance(join_key, dict) else None
        )
        if arch in included_by_arch:
            included_by_arch[arch].append(cell)

    per_arch: dict[str, dict] = {}
    for arch in sorted(arch_set):
        included_for_arch = included_by_arch.get(arch, [])
        excluded_for_arch = excluded_by_arch.get(arch, [])
        n_headline = len(included_for_arch)
        n_comparable = n_headline + len(excluded_for_arch)
        stats = _per_arch_stats(included_for_arch, contract)
        signed_residual_stats = _per_arch_signed_residual_stats(
            included_for_arch
        )
        bounded_arm_stats = _bounded_arm_relative_error_stats(
            included_for_arch
        )
        per_bucket_stats = _per_bucket_relative_error_stats(
            included_for_arch
        )
        if n_headline == 0:
            statistics_status = "no_cells"
        elif n_headline >= 4:
            statistics_status = "quartile_emitted"
        else:
            statistics_status = (
                "insufficient_cells_for_quartile_statistics"
            )
        rel_errs = [
            float(c["relative_error"]) for c in included_for_arch
        ]
        geomean = _log_space_geomean(rel_errs)
        ci_low, ci_high, ci_half = _bootstrap_ci_over_cells(
            rel_errs,
            n_resamples=contract.bootstrap_resamples,
            confidence=contract.ci_confidence,
            seed=int(bootstrap_seed),
        )
        geomean_ci_block = {
            "geomean": geomean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_half_width": ci_half,
            "bootstrap_resamples": contract.bootstrap_resamples,
            "bootstrap_seed": int(bootstrap_seed),
        }
        excluded_counts = _per_arch_excluded_counts(excluded_for_arch)
        # Coverage-gate live count: count UNIQUE admitted cells
        # (by the 7-field identity tuple excluding cache_layer),
        # not (cell, cache_layer) records. Codex code-review P1:
        # when the contract enables both L1 and L2 layers, a
        # single admitted cell appears in `included_for_arch`
        # twice (once per layer); the naive record-count would
        # double-count the cell, falsely passing the per-cell
        # coverage floor. The unique-cell count is the floor's
        # documented semantic.
        # Build BOTH per-evidence-kind unique-cell sets on the
        # same 7-field identity tuple so the coverage diagnostic
        # reports both halves on the same basis. Codex R52
        # code-review P2: computing `n_synthetic = n_headline -
        # n_live` mixed record-count semantics with unique-cell
        # semantics when both L1+L2 layers are enabled and every
        # cell was live on both layers, the diagnostic falsely
        # reported `n_synthetic == n_live` because n_headline
        # double-counted the cells via cache_layer.
        live_unique_cells: set[tuple[str, ...]] = set()
        synthetic_unique_cells: set[tuple[str, ...]] = set()
        for c in included_for_arch:
            join_key = c.get("join_key", {})
            identity_excl_layer = (
                str(join_key.get("slug", "")),
                str(join_key.get("shape_point", "")),
                str(join_key.get("binary_sha256", "")),
                str(join_key.get("source_commit", "")),
                str(join_key.get("kernel_signature_sha256", "")),
                str(join_key.get("arch", "")),
                str(join_key.get("ncu_metric_version", "")),
            )
            if c.get("evidence_kind") == "live":
                live_unique_cells.add(identity_excl_layer)
            else:
                synthetic_unique_cells.add(identity_excl_layer)
        n_live = len(live_unique_cells)
        n_synthetic = len(synthetic_unique_cells)
        # Per-layer relative-error stat blocks. Each block reports
        # n_cells / n_scalar / n_bounded / n_typed_unavailable +
        # geomean for the per-layer slice. The included slice
        # contributes the scalar + bounded counts (and the geomean
        # over relative_error); the excluded slice carrying
        # diagnostic="predictor_typed_unavailable" contributes the
        # typed_unavailable count for the matching layer (joined
        # bundles whose result_kind=typed_unavailable are routed to
        # excluded_cells with join_key.cache_layer populated from
        # the bundle's 8-tuple key). n_cells == n_scalar + n_bounded
        # + n_typed_unavailable so the on-disk count is the total
        # of joined bundles per (arch, layer). The geomean stays
        # computed from scalar/bounded included cells only;
        # typed-unavailable bundles do NOT contribute fake relative
        # errors.
        typed_unavail_per_layer: dict[str, int] = {
            "L1TEX": 0, "L2": 0,
        }
        for record in excluded_for_arch:
            if (
                str(record.get("diagnostic", ""))
                != _PREDICTOR_TYPED_UNAVAILABLE_DIAGNOSTIC
            ):
                continue
            join_key = record.get("join_key") or {}
            if not isinstance(join_key, dict):
                continue
            record_layer = str(join_key.get("cache_layer", "L2"))
            if record_layer in typed_unavail_per_layer:
                typed_unavail_per_layer[record_layer] += 1
        per_layer_stats: dict[str, dict] = {}
        for layer in ("L1TEX", "L2"):
            slice_for_layer = [
                c for c in included_for_arch
                if str(c.get("cache_layer", "L2")) == layer
            ]
            slice_rel_errs = [
                float(c["relative_error"]) for c in slice_for_layer
            ]
            slice_n_scalar = sum(
                1 for c in slice_for_layer
                if str(c.get("predictor_result_kind", "")) == "scalar"
            )
            slice_n_bounded = sum(
                1 for c in slice_for_layer
                if str(c.get("predictor_result_kind", "")) == "bounded"
            )
            slice_n_typed_unavailable = (
                typed_unavail_per_layer[layer]
            )
            slice_n = (
                slice_n_scalar
                + slice_n_bounded
                + slice_n_typed_unavailable
            )
            slice_geomean = _log_space_geomean(slice_rel_errs)
            per_layer_stats[layer] = {
                "n_cells": slice_n,
                "n_scalar": slice_n_scalar,
                "n_bounded": slice_n_bounded,
                "n_typed_unavailable": slice_n_typed_unavailable,
                "geomean": slice_geomean,
            }
        per_arch[arch] = {
            "arch": arch,
            "n_comparable_cells": n_comparable,
            "n_headline_cells": n_headline,
            "statistics_status": statistics_status,
            "relative_error_stats": stats,
            "relative_error_stats_l1tex": per_layer_stats["L1TEX"],
            "relative_error_stats_l2": per_layer_stats["L2"],
            "signed_residual_stats": signed_residual_stats,
            "bounded_arm_relative_error_stats": bounded_arm_stats,
            "relative_error_stats_per_predictor_design_coverage": (
                per_bucket_stats
            ),
            "relative_error_geomean_ci": geomean_ci_block,
            "excluded_reason_counts": excluded_counts,
            "coverage_gate_verdict": {
                "arch": arch,
                "headline_count_live": n_live,
                "headline_count_synthetic": n_synthetic,
                "floor": floor,
                "meets_floor": n_live >= floor,
            },
        }

    unattributed = _unattributed_excluded_counts(excluded_unattributed)
    return {
        "per_arch": per_arch,
        "unattributed_excluded_reason_counts": unattributed,
    }


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def run_comparison(
    *,
    contract: ComparisonContract,
    predictor_bundles: list[dict],
    label_cells: list[dict],
    bootstrap_seed: int = 0,
    split_label: str = "all",
    target_arch: str | None = None,
    calibration_evaluation_sidecar: dict | None = None,
    headline_validity_inputs: dict | None = None,
    unsupported_memory_path_sidecar: list | None = None,
    v5_head_path: str | None = None,
) -> dict:
    """Execute the comparison and return a schema-valid output
    bundle. The output bundle validates against
    ``workloads/simbench_bw/schemas/comparison_bundle.schema.json``.

    ``target_arch`` selects the single-arch baseline admission mode.
    When ``None`` (the default), top-level admission honors the
    contract's ``coverage_gate.require_both_arches_for_cross_arch``
    setting (the canonical cross-arch behavior). When set to a
    member of ``contract.arch_set``, top-level admission gates on
    that single arch's floor alone — the per-arch diagnostics for
    the other arches still appear in ``per_arch_diagnostics`` but
    they no longer block the top-level ``result_kind="admitted"``
    decision. The intended use is the documented
    ``Verified-live-<arch>`` baseline shape where one host has
    cleared the floor and the cross-arch leg is queued. The mode
    does NOT relax any per-arch floor or geomean check; it only
    relaxes the cross-arch gating.

    Pipeline:
      1. Pre-validate every predictor bundle's shape (unknown
         result_kind, out-of-range scalar, typed_reason out of
         enum hard-reject as typed exceptions).
      2. Cross-check every label cell's profiler_version against
         the contract's ncu_metric_version (drift records the cell
         in excluded_cells with reason=comparison_contract_drift).
      3. Join admitted bundles to admitted cells on the 7-field
         key. Exact match → include. Bundle-side near-miss →
         exclude with reason=join_key_unmatched and unmatched_field
         naming the offending field.
      4. For each include, compute relative_error per the contract
         formula. Scalar uses predicted_hit_rate; bounded uses the
         midpoint of [hit_rate_lower, hit_rate_upper] as the
         canonical headline plus lower/upper diagnostic fields;
         typed_unavailable joins go to excluded_cells (reason=
         predictor_typed_unavailable) and count toward comparable
         but NOT headline.
      5. Cell-side post-join pass: every admissible label cell
         whose 7-tuple key was not joined by any bundle is emitted
         as a comparison_runner_join_key_unmatched excluded record
         (with unmatched_field derived against the nearest valid
         bundle key). This is the supersession-bookkeeping path
         for historical labels that remain on disk but lack a
         matching HEAD-coherent predictor bundle.
      6. Tally comparable_cells_per_arch and headline_cells_per_arch.
         Apply coverage gate: if the active admission set does not
         clear the smoke_test floor, emit typed_unavailable with
         reason=headline_geomean_admitted_cell_count_below_threshold.
         The active admission set is either ``{target_arch}`` (when
         ``target_arch`` is set) OR the contract's arch_set under
         the cross-arch rule.
      7. Otherwise compute the log-space geomean over headline
         cells with cell-level bootstrap CI.
    """
    if target_arch is not None and target_arch not in contract.arch_set:
        raise ValueError(
            f"target_arch={target_arch!r} is not a member of "
            f"contract.arch_set={contract.arch_set!r}; choose one "
            f"of the arches the contract pins or omit the kwarg "
            f"to use the cross-arch admission path."
        )
    # Round-14 (Codex Round 13 Blocking Side Issue #2): when
    # `target_arch` is set, narrow `label_cells` to that arch
    # BEFORE any join-key / profiler-version / layer-broadcast
    # bookkeeping runs. Pre-Round-14 the runner walked ALL
    # label_cells (sm_86 + sm_120) and produced
    # comparable_cells_per_arch + headline_cells_per_arch keyed
    # on both arches even when target_arch was set; the
    # cell-side unmatched pass then emitted
    # `comparison_runner_join_key_unmatched` for every
    # non-target-arch label that had no matching bundle. The
    # filter below converts the runner into a true single-arch
    # mode: only target-arch labels enter the comparison
    # universe; per-arch surfaces contain only {target_arch}.
    if target_arch is not None:
        label_cells = [c for c in label_cells if c.get("arch") == target_arch]
    # V5 active path: when v5_head_path is supplied, REPLACE each bundle's
    # predicted_hit_rate with the v5 ensemble's prediction BEFORE the
    # legacy headline computation runs. This is the bundle-preprocessing
    # strategy that makes the published headline metrics actual v5 numbers
    # (not legacy v4 with a v5 side block). Bundles without the canonical
    # block keep their legacy predicted_hit_rate. Bundles where the v5
    # ensemble cannot route (unknown bucket, sklearn absent) also keep
    # their legacy value; the headline is then a hybrid where the v5 path
    # admits what it can and the legacy path covers the rest.
    v5_active_records: list[dict] = []

    def _mark_all_scalar_v5_unscorable(reason: str) -> None:
        """Fail-closed sweep: mark every scalar bundle typed_unavailable
        with typed_reason=v5_active_path_unscorable + a structured
        _v5_unscorable_reason so legacy predicted_hit_rate does NOT survive
        into the v5 headline population on ANY setup-failure path."""
        for bundle in predictor_bundles:
            if bundle.get("result_kind") != "scalar":
                continue
            bundle["result_kind"] = "typed_unavailable"
            bundle["typed_reason"] = "v5_active_path_unscorable"
            bundle["_v5_unscorable_reason"] = reason
            bundle.setdefault("underconfigured_bin_count", 0)
            bundle.setdefault("admitted_bin_count", 0)
            for k in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                bundle.pop(k, None)

    if v5_head_path:
        v5_head = load_calibration_head_v5(v5_head_path)
        if v5_head is None:
            # v5_head_path supplied but the head sidecar is absent / invalid:
            # fail-closed sweep.
            _mark_all_scalar_v5_unscorable(
                "setup_failure:v5_head_load_returned_none",
            )
        elif v5_head.get("fit_status") == "ok":
            try:
                from tools.local_labels import (
                    run_v5_active_path_comparison as _v5drv,
                )
                cells = _v5drv._rehydrate_cells_from_v5_head(v5_head)
                # Pass v5_head so the selected-class factory drives the
                # ensemble (was: default IR fallback because v5_head was
                # dropped on this call site).
                ensemble = (
                    _v5drv._fit_arch_head(cells, v5_head=v5_head)
                    if cells else None
                )
                if ensemble is None:
                    # Fail-closed: cells could not rehydrate OR factory
                    # dispatch failed. Moved OUTSIDE the
                    # if-ensemble-is-not-None branch (was unreachable in
                    # Round 8). Now top-level inside the fit_status=="ok"
                    # arm so it ACTUALLY fires.
                    _mark_all_scalar_v5_unscorable(
                        "setup_failure:fit_arch_head_returned_none",
                    )
                else:
                    bucket_lookup: dict[tuple[str, str, str], str] = {}
                    for c in cells:
                        bucket_lookup[
                            (c["arch"], c["slug"], c["shape_point"])
                        ] = c["family_bucket"]
                    # V5-mode fail-closed: in v5 mode, every scalar bundle
                    # MUST be either (a) successfully scored by the v5
                    # ensemble OR (b) marked typed_unavailable so legacy
                    # predicted_hit_rate does NOT survive into the v5
                    # headline population.
                    for bundle in predictor_bundles:
                        if ensemble is None:
                            break
                        if bundle.get("result_kind") != "scalar":
                            continue
                        mpi = bundle.get("manifest_pairing_index", {})
                        # The closed MPI carries no arch; derive it from the
                        # bundle's canonical top-level `device` the SAME way
                        # `_bundle_join_key` does, falling back only when the
                        # optional MPI arch is absent. Without this, real
                        # bundles (MPI arch="") key as ("", slug, shape_point),
                        # miss every (arch, slug, shape_point) bucket entry, and
                        # are ALL wrongly marked unscorable instead of scored by
                        # the v5 head. `key[0]` is also the arch passed to
                        # `ensemble.predict` below, so this resolves both sites.
                        arch = str(mpi.get("arch", "")) or _arch_for_device(
                            str(bundle.get("device", "")),
                        )
                        key = (
                            arch,
                            str(mpi.get("slug", "")),
                            str(mpi.get("shape_point", "")),
                        )
                        v5_unscorable_reason: str | None = None
                        if bundle.get(V5_CANONICAL_FEATURES_FIELD) is None:
                            v5_unscorable_reason = (
                                "missing_canonical_pre_calibration_features"
                            )
                        else:
                            bucket = bucket_lookup.get(key)
                            if bucket is None:
                                v5_unscorable_reason = (
                                    "bucket_not_in_v5_head_cohort_scope"
                                )
                            else:
                                try:
                                    v5_pred = float(ensemble.predict(
                                        key[0], bucket,
                                        bundle[V5_CANONICAL_FEATURES_FIELD],
                                    ))
                                except (RuntimeError, ImportError, KeyError) as exc:
                                    v5_unscorable_reason = (
                                        f"ensemble_predict_failed:{type(exc).__name__}"
                                    )
                                else:
                                    legacy_pred = bundle.get("predicted_hit_rate")
                                    bundle["predicted_hit_rate"] = v5_pred
                                    v5_active_records.append({
                                        "manifest_pairing_index": dict(mpi),
                                        "v5_prediction": v5_pred,
                                        "legacy_predicted_hit_rate": legacy_pred,
                                    })
                        if v5_unscorable_reason is not None:
                            # Mark as typed_unavailable so legacy headline
                            # scoring excludes this bundle (fail-closed).
                            bundle["result_kind"] = "typed_unavailable"
                            bundle["typed_reason"] = "v5_active_path_unscorable"
                            bundle["_v5_unscorable_reason"] = v5_unscorable_reason
                            # The scalar arm's required fields (predicted_hit_rate,
                            # ci_low/high/half_width, ci_confidence) stay on the
                            # bundle; the typed_unavailable arm's required fields
                            # (underconfigured_bin_count, admitted_bin_count) get
                            # synthesized to 0 to satisfy the schema.
                            bundle.setdefault("underconfigured_bin_count", 0)
                            bundle.setdefault("admitted_bin_count", 0)
                            # Drop scalar-arm-only fields per the schema's
                            # typed_unavailable arm's "not.anyOf" rules.
                            for k in (
                                "predicted_hit_rate", "ci_low", "ci_high",
                                "ci_half_width", "ci_confidence",
                            ):
                                bundle.pop(k, None)
            except ImportError as exc:
                _mark_all_scalar_v5_unscorable(
                    f"setup_failure:driver_import:{type(exc).__name__}"
                )
        else:
            # v5 head loaded but fit_status != ok (e.g. empty cohort OR
            # dependencies absent): same fail-closed sweep.
            _mark_all_scalar_v5_unscorable(
                f"setup_failure:fit_status_{v5_head.get('fit_status')}"
            )
    for bundle in predictor_bundles:
        _validate_predictor_bundle_shape(bundle)

    excluded: list[dict] = []
    admissible_cells: list[dict] = []
    # Track first-6-key tuples (slug, shape_point, binary_sha256,
    # source_commit, kernel_signature_sha256, arch) of cells
    # excluded by the protocol-pin drift. The matching predictor
    # bundle's first-6 fields will equal one of these tuples; the
    # bundle-side join-fallback below MUST suppress its
    # comparison_runner_join_key_unmatched record for those bundles
    # because the protocol-pin drift is already the sole correct
    # explanation. The exclusion shape is exclusive: a near-miss on
    # ``ncu_metric_version`` produces only ``comparison_contract_drift``,
    # never a duplicate ``comparison_runner_join_key_unmatched`` for
    # the same first-six key.
    drifted_cell_first_six_keys: set[tuple[str, ...]] = set()
    for cell in label_cells:
        # Pre-flight: reject v3 admitted-cell records that lack the
        # l1tex_label_status field. The v4 admission contract
        # requires every cell to carry the new field so the L1 join
        # path can route correctly; a v3 record cannot be safely
        # consumed because it dropped the L1 mean at aggregation
        # time. Forcing re-aggregation is the only safe path.
        if "l1tex_label_status" not in cell:
            cell_key = _cell_join_key(cell, cache_layer="L2")
            excluded.append({
                "diagnostic": "comparison_contract_drift",
                "reason": LOCAL_LABEL_ADMITTED_L1_MEAN_MIGRATION_REQUIRED,
                "join_key": _join_key_dict(cell_key),
                "unmatched_field": "l1tex_label_status",
            })
            continue
        # Pre-flight: reject a stale versioned admitted-cell record whose
        # schema_version is present but not 5 (a v4/v3/v2 wire artifact fed to
        # the v5 consumer). This is the schema-v5 mixed-version boundary at the
        # consumer ingress; the cell cannot be safely scored because its wire
        # contract predates the unit-tagged substrate.
        cell_schema_version = cell.get("schema_version")
        if cell_schema_version is not None and cell_schema_version != 5:
            cell_key = _cell_join_key(cell, cache_layer="L2")
            excluded.append({
                "diagnostic": "comparison_contract_drift",
                "reason": LOCAL_LABEL_ADMITTED_V5_MIGRATION_REQUIRED,
                "join_key": _join_key_dict(cell_key),
                "unmatched_field": "schema_version",
                "cell_schema_version": cell_schema_version,
            })
            continue
        profiler_version = str(cell.get("profiler_version", ""))
        if profiler_version != contract.ncu_metric_version:
            cell_key = _cell_join_key(cell, cache_layer="L2")
            excluded.append({
                "diagnostic": "comparison_contract_drift",
                "reason": "comparison_contract_drift",
                "join_key": _join_key_dict(cell_key),
                "unmatched_field": "ncu_metric_version",
                "label_profiler_version": profiler_version,
                "contract_ncu_metric_version": contract.ncu_metric_version,
            })
            drifted_cell_first_six_keys.add(cell_key[:6])
            continue
        admissible_cells.append(cell)

    # Map the contract's cache_layers (TOML-side enum, e.g. ["L1", "L2"])
    # to the bundle-side enum ("L1TEX" / "L2") so the 8-tuple join key
    # uses the bundle-side label uniformly. When the contract did not
    # pin cache_layers (legacy contract on disk), the runner falls
    # back to a single L2 layer to preserve legacy behavior; new
    # contracts pin ["L1", "L2"] explicitly.
    contract_layers = contract.cache_layers if contract.cache_layers else ("L2",)
    bundle_side_layers: list[str] = []
    for contract_layer in contract_layers:
        bundle_layer = next(
            (
                b for b, c in _BUNDLE_TO_CONTRACT_CACHE_LAYER.items()
                if c == contract_layer
            ),
            contract_layer,
        )
        bundle_side_layers.append(bundle_layer)

    # Broadcast each admissible cell across the bundle-side layers.
    # The L1TEX-broadcast variant is excluded pre-emptively when the
    # cell's l1tex_label_status declares the L1 mean is absent; the
    # L2-broadcast variant continues normally for the same cell so
    # missing L1 ground truth does not regress L2 coverage.
    cells_by_key: dict[tuple[str, ...], dict] = {}
    cells_keys: list[tuple[str, ...]] = []
    # Track L1 cell keys whose label was declared missing upstream
    # via the `l1tex_label_status != "mean_present"` branch below.
    # The later bundle-side `cell is None` branch must suppress
    # `comparison_runner_join_key_unmatched` exclusions for these
    # keys; the upstream `comparison_runner_l1tex_label_mean_missing`
    # record is the sole correct explanation, and emitting a second
    # exclusion for the same L1 bundle double-counts the cell in
    # `comparable_cells_per_arch` and shows the wrong reason.
    l1_missing_label_keys: set[tuple[str, ...]] = set()
    # Per the schema's `l1tex_label_status` enum at
    # `workloads/simbench_bw/schemas/local_label_admitted.schema.json`,
    # the runner excludes the L1 join ONLY for the two
    # `mean_absent_*` statuses (the L1 mean genuinely cannot be
    # derived). The `mean_present` status is consumed normally.
    # The `all_zero` status is ALSO consumed normally (the cell
    # captured per-repeat L1 values but each was exactly 0.0,
    # e.g., vendor_sgemm_cublas/timing_shape whose cublas-backed
    # kernel bypasses the L1tex cache pipeline);
    # cache_hit_rate_l1tex_sector_pct is the non-null number 0.0
    # which the L1 head's relative-error computation handles via
    # the existing epsilon denominator. The schema description
    # explicitly says "the runner consumes the cell normally for
    # L2 prediction and emits a per-cell diagnostic noting the
    # L1tex zero distribution".
    _L1_MEAN_ABSENT_STATUSES = frozenset({
        "mean_absent_label_missing",
        "mean_absent_provenance_missing",
    })
    for cell in admissible_cells:
        for layer in bundle_side_layers:
            if layer == "L1TEX":
                status = str(cell.get("l1tex_label_status", ""))
                if status in _L1_MEAN_ABSENT_STATUSES:
                    cell_key = _cell_join_key(cell, cache_layer=layer)
                    l1_missing_label_keys.add(cell_key)
                    excluded.append({
                        "diagnostic": "comparison_runner_join_key_unmatched",
                        "reason": COMPARISON_RUNNER_L1TEX_LABEL_MEAN_MISSING,
                        "join_key": _join_key_dict(cell_key),
                        "unmatched_field": "cache_hit_rate_l1tex_sector_pct",
                        "l1tex_label_status": status,
                    })
                    continue
            key = _cell_join_key(cell, cache_layer=layer)
            cells_by_key.setdefault(key, cell)
            cells_keys.append(key)

    arch_set = (
        contract.arch_set
        if contract.arch_set
        else tuple(sorted({k[5] for k in cells_keys}))
    )
    # Round-14 (Codex Round 13 Blocking Side Issue #2): when
    # `target_arch` is set, narrow `arch_set` to the single
    # target arch so the per-arch surfaces
    # (comparable_cells_per_arch, headline_cells_per_arch,
    # per_arch_diagnostics.per_arch) emit ONLY the target arch
    # key. Pre-Round-14 these dicts pre-populated all
    # contract.arch_set keys with 0, leaving a stale `sm_86: 0`
    # in single-arch-mode output. AC-5.2 requires
    # `arches == [target_arch]` and per-arch surfaces with
    # only the target-arch key.
    if target_arch is not None:
        arch_set = (target_arch,)
    comparable_per_arch: dict[str, int] = {a: 0 for a in arch_set}
    headline_per_arch: dict[str, int] = {a: 0 for a in arch_set}

    included: list[dict] = []
    matched_cell_keys: set[tuple[str, ...]] = set()
    # Track every valid bundle 7-tuple key seen during the bundle
    # loop so the cell-side post-join pass can derive a meaningful
    # ``unmatched_field`` for each admissible cell that no bundle
    # joined (e.g., the historical activation labels at
    # ``source_commit=0c16e335...`` whose nearest HEAD bundle key
    # differs only on ``source_commit``).
    valid_bundle_keys: list[tuple[str, ...]] = []
    # Per-cell uniqueness gate: the contract's smoke-test floor is
    # per cell (5 unique HEAD-coherent live cells per arch), so a
    # duplicate bundle MUST hard-reject before any per-arch
    # counting or geomean-weighting fires. Without this gate, a
    # CLI glob that resolves to two copies of the same bundle
    # (e.g., a stale + fresh path, or a symlink that
    # double-matches) would let N-1 unique cells plus one
    # duplicate falsely satisfy the 5-cell floor and overweight
    # the duplicated cell's relative_error.
    seen_bundle_keys: set[tuple[str, ...]] = set()
    for bundle in predictor_bundles:
        # Pre-join contract check: a bundle without a usable
        # manifest_pairing_index cannot have its 7-tuple join key
        # resolved. Emit comparison_contract_drift before any join
        # attempt; the exclusion record carries no synthetic
        # join_key (the runner has no key to record), so it routes
        # to the unattributed bucket downstream.
        if not _bundle_has_valid_manifest_pairing_index(bundle):
            excluded.append({
                "diagnostic": "comparison_contract_drift",
                "reason": "comparison_contract_drift",
                "unmatched_field": "manifest_pairing_index",
            })
            continue
        # Per-contract cache-layer gating: a bundle whose cache_layer
        # is the bundle-side label for a layer NOT in the contract's
        # cache_layers is excluded BEFORE the 8-tuple join. The
        # contract's cache_layers pins which layers the comparison
        # is over; a bundle outside that set is comparison_contract_drift.
        bundle_layer = str(bundle.get("cache_layer", "L2"))
        contract_layer = _BUNDLE_TO_CONTRACT_CACHE_LAYER.get(
            bundle_layer, bundle_layer,
        )
        if (
            contract.cache_layers
            and contract_layer not in contract.cache_layers
        ):
            excluded.append({
                "diagnostic": "comparison_contract_drift",
                "reason": "comparison_contract_drift",
                "unmatched_field": "cache_layer",
                "bundle_cache_layer": bundle_layer,
                "contract_cache_layers": list(contract.cache_layers),
            })
            continue
        bundle_key = _bundle_join_key(bundle, contract)
        if bundle_key in seen_bundle_keys:
            mpi = bundle.get("manifest_pairing_index", {}) or {}
            raise ComparisonRunnerDuplicatePredictorBundle(
                join_key=bundle_key,
                slug=str(mpi.get("slug", "")),
                shape_point=str(mpi.get("shape_point", "")),
                detail=(
                    "duplicate bundle file in --predictor-bundles-glob"
                    " — the contract's per-cell admission floor "
                    "rejects double-counting the same join key"
                ),
            )
        seen_bundle_keys.add(bundle_key)
        valid_bundle_keys.append(bundle_key)
        cell = cells_by_key.get(bundle_key)
        result_kind = bundle.get("result_kind")
        if cell is None:
            # Suppress the duplicate exclusion when this bundle's
            # first-6-key matches a label cell that was already
            # drift-excluded by the protocol-pin check above. The
            # ``comparison_contract_drift`` record is the sole
            # correct explanation for that key; emitting an
            # additional ``comparison_runner_join_key_unmatched``
            # record would double-count the same drift surface.
            if bundle_key[:6] in drifted_cell_first_six_keys:
                continue
            # Same suppression for L1 cells whose label was already
            # excluded with `comparison_runner_l1tex_label_mean_missing`
            # upstream. That record is the sole correct explanation
            # for the missing join; emitting a second
            # `join_key_unmatched` for the same L1 bundle would
            # inflate `comparable_cells_per_arch` AND surface the
            # wrong reason in per-arch diagnostics.
            if bundle_key in l1_missing_label_keys:
                continue
            unmatched_field = _find_unmatched_field(bundle_key, cells_keys)
            excluded.append({
                "diagnostic": "comparison_runner_join_key_unmatched",
                "reason": "join_key_unmatched",
                "join_key": _join_key_dict(bundle_key),
                "unmatched_field": unmatched_field,
            })
            continue
        matched_cell_keys.add(bundle_key)
        arch = bundle_key[5]
        if arch not in comparable_per_arch:
            comparable_per_arch[arch] = 0
            headline_per_arch[arch] = 0
        comparable_per_arch[arch] += 1
        if result_kind == "typed_unavailable":
            excluded.append({
                "diagnostic": _PREDICTOR_TYPED_UNAVAILABLE_DIAGNOSTIC,
                "reason": "predictor_typed_unavailable",
                "join_key": _join_key_dict(bundle_key),
                "upstream_typed_reason": str(
                    bundle.get("typed_reason", "")
                ),
                "label_profiler_version": str(
                    cell.get("profiler_version", "")
                ),
            })
            continue
        # Layer-aware ground-truth resolution. The bundle's
        # cache_layer (defaulting to L2 for legacy bundles) selects
        # which per-cell field carries the ground truth: the L1 head
        # reads cache_hit_rate_l1tex_sector_pct; the L2 head reads
        # cache_hit_rate_l2_sector_pct. A cell broadcast to the
        # L1TEX layer is guaranteed by the cell-side pre-flight to
        # have l1tex_label_status="mean_present" with a non-null L1
        # mean (mean_absent_* cells were excluded above before
        # cells_by_key was populated).
        bundle_layer_local = str(bundle.get("cache_layer", "L2"))
        if bundle_layer_local == "L1TEX":
            ground_truth_pct = float(
                cell.get("cache_hit_rate_l1tex_sector_pct") or 0.0
            )
        else:
            ground_truth_pct = float(
                cell.get("cache_hit_rate_l2_sector_pct", 0.0)
            )
        ground_truth = (
            _percent_to_fraction(ground_truth_pct)
            if contract.percent_to_fraction
            else ground_truth_pct
        )
        evidence_kind = str(cell.get("evidence_kind", "synthetic"))
        # Resolve the cohort bucket label for this (slug, shape_point)
        # so per-cell records carry the contract-pinned classifier.
        # Cells outside the headline cohort are emitted with None;
        # the field is always present so the on-disk shape stays
        # deterministic for downstream consumers.
        bucket_label = _PREDICTOR_DESIGN_COVERAGE_BY_CELL.get(
            (bundle_key[0], bundle_key[1])
        )
        # Optional cross-arch extrapolation provenance for the gridDim
        # density penalty alpha. Propagated verbatim from the predictor
        # bundle's top-level grid_dim_alpha_source when present.
        # METADATA ONLY: never consumed by an admission gate or
        # coverage-floor branch; surfaced for downstream auditors only.
        alpha_source_extra = (
            {"grid_dim_alpha_source": bundle.get("grid_dim_alpha_source")}
            if bundle.get("grid_dim_alpha_source") is not None
            else {}
        )
        if result_kind == "scalar":
            predicted = float(bundle.get("predicted_hit_rate", 0.0))
            rel_err = _rel_err(predicted, ground_truth, contract.epsilon)
            included.append({
                "join_key": _join_key_dict(bundle_key),
                "predictor_result_kind": "scalar",
                "predicted_hit_rate": predicted,
                "ground_truth_fraction": ground_truth,
                "relative_error": rel_err,
                "label_profiler_version": str(
                    cell.get("profiler_version", "")
                ),
                "evidence_kind": evidence_kind,
                "predictor_design_coverage": bucket_label,
                "cache_layer": bundle_layer_local,
                **alpha_source_extra,
            })
            headline_per_arch[arch] += 1
        elif result_kind == "bounded":
            lower = float(bundle.get("hit_rate_lower", 0.0))
            upper = float(bundle.get("hit_rate_upper", 0.0))
            midpoint = (lower + upper) / 2.0
            rel_err = _rel_err(midpoint, ground_truth, contract.epsilon)
            rel_err_low = _rel_err(lower, ground_truth, contract.epsilon)
            rel_err_high = _rel_err(upper, ground_truth, contract.epsilon)
            included.append({
                "join_key": _join_key_dict(bundle_key),
                "predictor_result_kind": "bounded",
                "predicted_hit_rate": midpoint,
                "ground_truth_fraction": ground_truth,
                "relative_error": rel_err,
                "relative_error_lower_diagnostic": rel_err_low,
                "relative_error_upper_diagnostic": rel_err_high,
                "label_profiler_version": str(
                    cell.get("profiler_version", "")
                ),
                "evidence_kind": evidence_kind,
                "predictor_design_coverage": bucket_label,
                "cache_layer": bundle_layer_local,
                **alpha_source_extra,
            })
            headline_per_arch[arch] += 1

    # Cell-side post-join pass: emit one
    # ``comparison_runner_join_key_unmatched`` excluded record per
    # admissible label cell whose 7-tuple key was NOT joined by any
    # predictor bundle. Without this pass the historical activation
    # labels at ``source_commit=0c16e335...`` would silently
    # disappear from ``per_arch_diagnostics`` once the HEAD
    # re-labels exist (the HEAD predictor bundles join the HEAD
    # labels, so the bundle-side loop emits nothing for the
    # historical labels). The supersession-bookkeeping discipline
    # requires the historical labels to surface naturally as
    # unmatched-on-source_commit records.
    #
    # Suppression: a cell whose first-6-key was already excluded
    # for protocol-pin drift via the cell-side
    # ``comparison_contract_drift`` path above is NOT also emitted
    # here. The drift record is the sole correct explanation for
    # that key (mirrors the bundle-side suppression hook).
    for cell in admissible_cells:
        for layer in bundle_side_layers:
            # An L1TEX broadcast of a cell whose L1 mean is absent
            # was already excluded above with
            # comparison_runner_l1tex_label_mean_missing; skip it
            # here so the supersession-bookkeeping pass does not
            # emit a duplicate join_key_unmatched record for the
            # same (cell, layer) shape. The skip uses the same
            # `_L1_MEAN_ABSENT_STATUSES` allowlist as the earlier
            # exclusion branch (per the schema's 4-value enum:
            # mean_present / all_zero / mean_absent_label_missing /
            # mean_absent_provenance_missing). The `all_zero`
            # status is a VALID non-null L1 label (cache_hit_rate_l1tex_sector_pct
            # = 0.0 + stddev_l1tex_sector = 0.0); cells with that
            # status reach this bookkeeping pass normally so an
            # unmatched join key emits the cell-side
            # join_key_unmatched diagnostic.
            if (
                layer == "L1TEX"
                and str(cell.get("l1tex_label_status", ""))
                in _L1_MEAN_ABSENT_STATUSES
            ):
                continue
            cell_key = _cell_join_key(cell, cache_layer=layer)
            if cell_key in matched_cell_keys:
                continue
            if cell_key[:6] in drifted_cell_first_six_keys:
                continue
            unmatched_field = _find_unmatched_field(
                cell_key, valid_bundle_keys,
            )
            excluded.append({
                "diagnostic": "comparison_runner_join_key_unmatched",
                "reason": "join_key_unmatched",
                "join_key": _join_key_dict(cell_key),
                "unmatched_field": unmatched_field,
            })

    # Tally excluded records into ``comparable_per_arch`` so the
    # top-level counter stays consistent with the per-arch view's
    # ``n_comparable_cells = n_headline + len(excluded_for_arch)``
    # arithmetic. The matched-cell branch already incremented
    # ``comparable_per_arch[arch]`` for every joined bundle
    # (including those that ended up with
    # ``predictor_typed_unavailable``), so this pass MUST skip the
    # ``predictor_typed_unavailable`` reason to avoid double-
    # counting. Records without an arch (the bundle-side
    # ``comparison_contract_drift`` from a missing
    # ``manifest_pairing_index``) flow through to
    # ``unattributed_excluded_reason_counts`` in the per-arch
    # diagnostics and contribute to neither surface — that
    # symmetric behavior is what keeps the two surfaces agreeing.
    for record in excluded:
        reason = str(record.get("reason", ""))
        if reason == "predictor_typed_unavailable":
            continue
        # Skip layer-aware pre-flight exclusions: a v3-record rejection
        # or an L1-mean-absent rejection is structurally outside the
        # comparable-cells population (the runner never reached the
        # join attempt for that cell-layer pair) and counting it would
        # double-inflate per-arch counters when each cell is broadcast
        # across multiple layers.
        if reason == LOCAL_LABEL_ADMITTED_L1_MEAN_MIGRATION_REQUIRED:
            continue
        if reason == COMPARISON_RUNNER_L1TEX_LABEL_MEAN_MISSING:
            continue
        join_key = record.get("join_key") or {}
        if not isinstance(join_key, dict):
            continue
        record_arch = join_key.get("arch")
        if record_arch and record_arch in comparable_per_arch:
            comparable_per_arch[record_arch] += 1

    # Sort included / excluded for deterministic output. Records
    # carry the per-layer cache_layer field so the sort key includes
    # it as the final tiebreaker to keep L1TEX and L2 entries for
    # the same (slug, shape_point) deterministically adjacent.
    included.sort(
        key=lambda r: (
            r["join_key"]["arch"],
            r["join_key"]["slug"],
            r["join_key"]["shape_point"],
            r["join_key"].get("cache_layer", "L2"),
        )
    )
    excluded.sort(
        key=lambda r: (
            r["reason"],
            r.get("join_key", {}).get("arch", ""),
            r.get("join_key", {}).get("slug", ""),
            r.get("join_key", {}).get("shape_point", ""),
            r.get("join_key", {}).get("cache_layer", ""),
            r.get("unmatched_field", ""),
        )
    )

    pairing_index = {
        "ncu_metric_name": contract.ncu_metric_name,
        "ncu_metric_version": contract.ncu_metric_version,
        "epsilon": contract.epsilon,
        "bootstrap_resamples": contract.bootstrap_resamples,
        "bootstrap_seed": int(bootstrap_seed),
        "ci_confidence": contract.ci_confidence,
        "split_label": str(split_label),
    }

    per_arch_diagnostics_block = _per_arch_diagnostics(
        included_cells=included,
        excluded_cells=excluded,
        arch_set=arch_set if arch_set else tuple(sorted(comparable_per_arch.keys())),
        contract=contract,
        bootstrap_seed=int(bootstrap_seed),
    )

    # Persist the admission scope alongside the per-arch counters
    # so downstream readers can distinguish a Verified-live-<arch>
    # baseline (target_arch mode) from a Cross-arch-verified
    # bundle. Without this field a bundle with
    # result_kind="admitted" + sm_120.meets_floor=false is
    # ambiguous: did admission fire because target_arch="sm_86"
    # relaxed the cross-arch gate, or because the contract's
    # cross_arch_any_arch path admitted off the lone-arch floor?
    # The three documented values mirror the three branches of
    # the admission gate below
    # (target_arch / require_both_arches / any).
    if target_arch is not None:
        admission_scope = f"target_arch={target_arch}"
    elif contract.coverage_gate.require_both_arches_for_cross_arch:
        admission_scope = "cross_arch_require_both_arches"
    else:
        admission_scope = "cross_arch_any_arch"

    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "comparison_run_pairing_index": pairing_index,
        "top_level_admission_scope": admission_scope,
        # Round-14 (AC-5.2): explicit `arches` list mirroring the
        # per-arch tally keys. Required by the plan's AC-5.2 +
        # AC-1 positive tests to assert
        # `arches == ["sm_120"]` on the active sm_120-only
        # comparison bundle.
        "arches": sorted(comparable_per_arch.keys()),
        "comparable_cells_per_arch": comparable_per_arch,
        "headline_cells_per_arch": headline_per_arch,
        # Round-14 (AC-4.3): scalar_or_bounded_headline_cells_per_arch
        # mirrors headline_cells_per_arch — both surface the cells
        # that survived the typed-availability filter and have a
        # usable comparison scalar. The naming distinguishes
        # scalar/bounded admissions from the typed_unavailable
        # ones (which surface only in the excluded list). Pinning
        # the plan-named field name allows AC-4.3's
        # `>= 24` threshold to be checked against a canonical
        # field rather than re-derived per call site.
        "scalar_or_bounded_headline_cells_per_arch": dict(headline_per_arch),
        "headline_floor_per_arch": (
            contract.coverage_gate.smoke_test_floor_per_arch
        ),
        "included_cells": included,
        "excluded_cells": excluded,
        "per_arch_diagnostics": per_arch_diagnostics_block,
    }

    # AC-T7 headline-validity guard. When the caller supplies
    # EITHER ``headline_validity_inputs`` OR
    # ``calibration_evaluation_sidecar``, walk every applicable
    # detection helper plus the calibration LOOCV evaluation
    # sidecar and collect every fired typed reason into the
    # ``headline_validity_failed_reasons`` array. The companion
    # ``headline_validity_status`` enum is ``valid`` (empty array)
    # or ``typed_unavailable`` (non-empty array). Both fields are
    # added to the bundle when EITHER input is provided so the
    # production paths that thread only the calibration sidecar
    # still emit the new array carrying the LOOCV reason; legacy
    # call sites that pass NEITHER input get byte-identical
    # output. The array captures ALL fired reasons (not just the
    # first) per the plan's multi-failure invariant; the existing
    # ``typed_reason`` short-circuit below still fires the first
    # gate it hits.
    # Collect upstream typed reasons from input predictor bundles
    # so the comparison runner can promote select reasons (e.g.,
    # `calibration_variance_precondition_failed`) to the
    # comparison-level headline-validity-failed-reasons list.
    #
    # Scope the promotion to bundles that are part of the ACTIVE
    # comparison set: (a) their 7-tuple bundle_key successfully
    # joined a label cell (so the bundle is in `matched_cell_keys`
    # built by the loop above) AND (b) when target_arch is set,
    # the bundle's arch matches target_arch (defensive
    # belt-and-suspenders check in case the contract's arch_set
    # includes more arches than target_arch). Without these
    # filters, a typed_unavailable bundle for an INACTIVE arch
    # (e.g., sm_120 bundle when target_arch="sm_86") OR a stale
    # join-key-unmatched bundle in a broad glob would populate
    # `headline_validity_failed_reasons`, forcing the comparison
    # bundle to typed_unavailable even though target_arch is
    # meant to insulate the active arch from other arches'
    # failures. Codex Round-23 review P2 fix.
    upstream_predictor_typed_reasons: list[str] = []
    for _pb in predictor_bundles:
        if not isinstance(_pb, dict):
            continue
        if _pb.get("result_kind") != "typed_unavailable":
            continue
        if not _bundle_has_valid_manifest_pairing_index(_pb):
            continue
        _pb_key = _bundle_join_key(_pb, contract)
        if _pb_key not in matched_cell_keys:
            continue
        if target_arch is not None and _pb_key[5] != target_arch:
            continue
        _tr = _pb.get("typed_reason")
        if isinstance(_tr, str):
            upstream_predictor_typed_reasons.append(_tr)

    if (
        headline_validity_inputs is not None
        or calibration_evaluation_sidecar is not None
        or unsupported_memory_path_sidecar is not None
        or any(
            _tr == CALIBRATION_VARIANCE_PRECONDITION_FAILED
            for _tr in upstream_predictor_typed_reasons
        )
    ):
        failed_reasons = _compute_headline_validity_failed_reasons(
            headline_validity_inputs,
            calibration_evaluation_sidecar,
            unsupported_memory_path_sidecar,
            upstream_predictor_typed_reasons=(
                upstream_predictor_typed_reasons
            ),
        )
        base["headline_validity_failed_reasons"] = failed_reasons
        base["headline_validity_status"] = (
            "typed_unavailable" if failed_reasons else "valid"
        )

    # Comparison-level fail-closed cardinality guard. If ANY
    # headline-validity reason fires, the comparison bundle MUST
    # short-circuit to the typed_unavailable arm BEFORE the live
    # coverage gate runs. The schema's admitted arm rejects any
    # bundle with non-empty headline_validity_failed_reasons (the
    # admitted-arm cardinality invariant); admitting such a bundle
    # would produce a schema-invalid artifact. The top-level
    # ``typed_reason`` mirrors the first failed reason in
    # canonical ``HEADLINE_VALIDITY_FAILED_REASONS_ALL`` order,
    # which is the same order ``_compute_headline_validity_failed_reasons``
    # populates the array (alphabetical sm_86-active reasons
    # first, then cross-arch reasons). The runner-level
    # ``typed_reason.enum`` in
    # ``workloads/simbench_bw/schemas/comparison_bundle.schema.json``
    # is widened in lockstep with this gate so every member of
    # ``HEADLINE_VALIDITY_FAILED_REASONS_ALL`` is schema-admitted
    # as a top-level typed_reason value. This short-circuit fires
    # BEFORE the LOOCV-specific short-circuit further down so
    # any headline-validity reason (including the LOOCV reason)
    # short-circuits here first, with the top-level typed_reason
    # following the canonical alphabetical sm_86-active order.
    # The canonical precedence is therefore:
    #   1. calibration_covariate_out_of_envelope (sm_86-active, alpha[0])
    #   2. calibration_loocv_p90_residual_exceeded (sm_86-active, alpha[1])
    #   3. calibration_reuse_distance_out_of_envelope (sm_86-active, alpha[2])
    #   4. calibration_variance_precondition_failed (sm_86-active, alpha[3])
    #   5. trace_opcode_coverage_incomplete (sm_86-active, alpha[4])
    #   6. unsupported_blackwell_memory_path (cross-arch tail)
    #   7. target_arch_unlabeled_extrapolation (cross-arch tail)
    # ABOVE all of those: this short-circuit runs BEFORE the live
    # coverage gate (which fires
    # headline_geomean_admitted_cell_count_below_threshold) AND
    # BEFORE the LOOCV-evaluation-sidecar short-circuit (which
    # also fires calibration_loocv_p90_residual_exceeded), so any
    # headline-validity reason in the array takes the top-level
    # typed_reason. The pinned precedence is asserted by
    # ``test_typed_reason_precedence_follows_canonical_order`` in
    # ``workloads/simbench_bw/tests/test_comparison_runner_headline_validity.py``.
    _committed_failed_reasons = base.get(
        "headline_validity_failed_reasons",
    )
    if _committed_failed_reasons:
        base["result_kind"] = "typed_unavailable"
        base["typed_reason"] = _committed_failed_reasons[0]
        return attach_v5_metadata(
            base, v5_head_path, v5_active_records,
        )

    # Top-level admitted gate counts only live cells. Synthetic
    # fixtures declare ``evidence_kind == "synthetic"`` on every
    # cell; the live-only gate prevents synthetic-only fixtures from
    # accidentally appearing admitted. The per-arch
    # ``coverage_gate_verdict.meets_floor`` block already encodes the
    # live-only check (n_live >= floor); reuse it here so the
    # top-level decision and the per-arch verdict cannot diverge.
    #
    # When ``target_arch`` is set, top-level admission gates on that
    # single arch's floor alone (the documented Verified-live-<arch>
    # baseline shape). The per-arch diagnostics for the other arches
    # in arch_set still appear in per_arch_diagnostics, so the bundle
    # remains comprehensive for downstream Codex review or sm_120
    # cross-host augmentation. Otherwise, when
    # ``coverage_gate.require_both_arches_for_cross_arch`` is true,
    # EVERY arch in the contract's ``arch_set`` must meet its live
    # floor; this is the strict cross-arch admission semantic. When
    # false, the existing any-arch behavior is preserved.
    per_arch_records = per_arch_diagnostics_block["per_arch"]
    iter_arches = (
        arch_set if arch_set else tuple(per_arch_records.keys())
    )

    def _arch_meets_floor(arch: str) -> bool:
        return per_arch_records.get(arch, {}).get(
            "coverage_gate_verdict", {},
        ).get("meets_floor", False)

    if target_arch is not None:
        admit = _arch_meets_floor(target_arch)
    elif contract.coverage_gate.require_both_arches_for_cross_arch:
        admit = bool(iter_arches) and all(
            _arch_meets_floor(a) for a in iter_arches
        )
    else:
        admit = any(_arch_meets_floor(a) for a in iter_arches)
    if not admit:
        base["result_kind"] = "typed_unavailable"
        base["typed_reason"] = (
            "headline_geomean_admitted_cell_count_below_threshold"
        )
        return attach_v5_metadata(
            base, v5_head_path, v5_active_records,
        )

    # AC-T4 calibration-evaluation gate. When the caller supplies
    # a calibration-evaluation sidecar payload (active path: v4;
    # legacy/historical: v1), read the chosen head's LOOCV p90
    # from it and short-circuit to
    # typed_unavailable when that p90 exceeds the DEC-T5 pinned
    # threshold of 0.30. The bundle still carries all
    # already-computed diagnostics (included/excluded cell lists,
    # per_arch_diagnostics) so downstream consumers see calibration
    # was attempted but the chosen head was not trustworthy. The
    # geomean / CI fields are deliberately NOT set on this arm to
    # keep the schema's oneOf arm exclusivity intact.
    if calibration_evaluation_sidecar is not None:
        calibration_reason = check_calibration_loocv_p90_residual(
            calibration_evaluation_sidecar,
        )
        if calibration_reason is not None:
            base["result_kind"] = "typed_unavailable"
            base["typed_reason"] = calibration_reason
            return attach_v5_metadata(
                base, v5_head_path, v5_active_records,
            )

    # Filter to LIVE evidence only. The coverage-gate verdict
    # admits the population using `unique live cells` (see
    # `coverage_gate_verdict.meets_floor` above); synthetic
    # fixtures do NOT gate admission. If we include synthetic
    # cells in the published headline geomean/CI, the headline
    # is no longer derived from the same population the
    # admission gate measured, and a synthetic row's
    # relative_error can move the published value. Filter
    # `rel_errs` to `evidence_kind == "live"` here so the
    # headline statistic stays consistent with the admission
    # gate.
    if target_arch is not None:
        rel_errs = [
            r["relative_error"] for r in included
            if r["join_key"]["arch"] == target_arch
            and r.get("evidence_kind") == "live"
        ]
    else:
        rel_errs = [
            r["relative_error"] for r in included
            if r.get("evidence_kind") == "live"
        ]
    geomean = _log_space_geomean(rel_errs)
    ci_low, ci_high, ci_half = _bootstrap_ci_over_cells(
        rel_errs,
        n_resamples=contract.bootstrap_resamples,
        confidence=contract.ci_confidence,
        seed=int(bootstrap_seed),
    )
    base["result_kind"] = "admitted"
    base["geomean"] = geomean
    base["geomean_ci_low"] = ci_low
    base["geomean_ci_high"] = ci_high
    base["geomean_ci_half_width"] = ci_half

    # Round-15 (AC-5.3): dual-geomean reporting — full-cohort
    # geomean + holdout-only geomean (the 7 roster cells with
    # `split == "holdout"`). The full-cohort value mirrors the
    # legacy ``geomean`` field by construction but is exposed
    # under the AC-5.3-named key so downstream consumers can
    # assert on the canonical field. The holdout value filters
    # ``rel_errs`` further to the cells whose roster row has
    # ``split == "holdout"``. Both use the contract's bootstrap
    # pins. Holdout admits typed-unavailable bubbling: cells
    # excluded from the headline pass are NOT substituted from
    # calibration/validation.
    holdout_rel_errs = _compute_holdout_rel_errs(
        included=included, target_arch=target_arch,
    )
    base["full_cohort_geomean"] = geomean
    base["full_cohort_geomean_ci_low"] = ci_low
    base["full_cohort_geomean_ci_high"] = ci_high
    base["full_cohort_geomean_ci_half_width"] = ci_half
    holdout_geomean = _log_space_geomean(holdout_rel_errs)
    if holdout_rel_errs:
        h_low, h_high, h_half = _bootstrap_ci_over_cells(
            holdout_rel_errs,
            n_resamples=contract.bootstrap_resamples,
            confidence=contract.ci_confidence,
            seed=int(bootstrap_seed),
        )
    else:
        # No holdout cells in the included set — emit zeros for
        # CI fields rather than omit so schema-required key set
        # stays uniform across runs.
        h_low, h_high, h_half = 0.0, 0.0, 0.0
    base["holdout_geomean"] = holdout_geomean
    base["holdout_geomean_ci_low"] = h_low
    base["holdout_geomean_ci_high"] = h_high
    base["holdout_geomean_ci_half_width"] = h_half
    base["holdout_cell_count"] = len(holdout_rel_errs)
    # V5 active-path integration: when a v5 head path is supplied via
    # run_comparison(v5_head_path=...), load the head + emit the
    # per_arch_transfer_diagnostics block on the output bundle. The legacy
    # scalar predicted_hit_rate stays for downstream consumers that still
    # depend on the v4 path; the v5 block lives alongside.
    v5_head_path = locals().get("v5_head_path")
    if v5_head_path:
        base = attach_v5_metadata(
            base,
            v5_head_path,
            locals().get("v5_active_records", []) or [],
        )
    return base


# Round-15 (AC-5.3) helpers: load the frozen roster split tags
# and filter relative-error series to the holdout subset.
_ROSTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "local_labels" / "sm120_cohort_expansion_roster.toml"
)

# The roster file is the FROZEN sm_120 cohort roster: every row's split
# tag (calibration / validation / holdout) is an sm_120-cohort split
# assignment. A row's ``arch_gate`` is the BUILD-arch set, NOT the split
# arch. The split tags therefore MUST NOT be applied to a record from
# any other arch: an sm_86 cell that merely reuses a (slug, shape_point)
# present in this roster carries its OWN split (the sm_86 head's fold
# protocol), not the sm_120 cohort's. The split index is keyed on this
# arch so a non-sm_120 record can never match a roster entry and leak
# non-holdout data into the holdout subset.
_ROSTER_ARCH = "sm_120"


class RosterLoadError(RuntimeError):
    """Raised when the frozen sm_120 cohort roster cannot be loaded.

    AC-5.3 requires the dual-geomean reporter to compute the
    holdout subset against the frozen roster's split tags. A silent
    empty index here would produce ``holdout_cell_count = 0`` for
    every run -- which the AC-5.3 schema conditional then rejects on
    the sm_120 admitted arm, but only AFTER the producer wrote a
    wrong number. Failing loud at load time surfaces the diagnostic
    at the producer side with the expected path."""


def _load_roster_split_index() -> dict[tuple[str, str, str], str]:
    """Build an ``(arch, slug, shape_point) -> split`` index from the
    frozen sm_120 cohort roster. The arch component is pinned to
    ``_ROSTER_ARCH`` because every roster split tag is an sm_120-cohort
    assignment (see ``_ROSTER_ARCH``); keying by arch means a record
    from another arch that merely reuses a ``(slug, shape_point)``
    cannot match a roster entry. Off-roster cells return no entry; the
    holdout filter treats absent cells as not-in-holdout (they are not
    over-counted as holdout).

    Raises ``RosterLoadError`` if the roster file cannot be opened
    or parsed. Callers that operate in a degraded mode (e.g., tests
    that intentionally probe off-roster cell paths) MUST handle the
    typed exception explicitly rather than rely on silent
    empty-index fallback."""
    try:
        with _ROSTER_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise RosterLoadError(
            f"failed to load frozen sm_120 cohort roster at "
            f"{_ROSTER_PATH}: {exc}"
        ) from exc
    return {
        (
            _ROSTER_ARCH,
            str(r.get("slug", "")),
            str(r.get("shape_point", "")),
        ): str(r.get("split", ""))
        for r in data.get("rows", [])
    }


def _compute_holdout_rel_errs(
    *, included: list[dict], target_arch: str | None,
) -> list[float]:
    """Round-15 (AC-5.3): return the relative-error scalars for
    the subset of ``included`` headline-cell records whose roster
    row has ``split == "holdout"``.

    Applies the same upstream filter the headline-stat path uses
    (``evidence_kind == "live"`` + optional ``arch == target_arch``)
    so the holdout subset is always a slice of the same cells the
    full-cohort geomean operates over. The roster lookup is keyed on
    the record's OWN arch (see ``_ROSTER_ARCH``): the roster split tags
    are sm_120-cohort assignments, so a record from another arch that
    reuses a roster ``(slug, shape_point)`` is correctly excluded from
    the holdout subset rather than inheriting the sm_120 split."""
    split_index = _load_roster_split_index()
    holdout: list[float] = []
    for record in included:
        if record.get("evidence_kind") != "live":
            continue
        if (
            target_arch is not None
            and record.get("join_key", {}).get("arch") != target_arch
        ):
            continue
        key = (
            str(record.get("join_key", {}).get("arch", "")),
            str(record.get("join_key", {}).get("slug", "")),
            str(record.get("join_key", {}).get("shape_point", "")),
        )
        if split_index.get(key) == "holdout":
            holdout.append(float(record["relative_error"]))
    return holdout


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_calibration_evaluation_sidecar_path(
    *,
    explicit_override: Path | None,
    target_arch: str | None,
    repo_root: Path,
) -> Path | None:
    """Resolve which calibration-evaluation sidecar the
    comparison-runner CLI consumes per the active calibration
    sidecar selection rule (mirrors
    `tools/local_labels/run_headline_cohort_predict_compare.py`'s
    helper of the same name):

      1. If ``explicit_override`` is non-None, return it verbatim
         (operator authority over auto-selection).
      2. Else if ``target_arch == "sm_120"`` (the active sm_120
         framework path), prefer
         ``data/local_labels/calibration_evaluation_v4.json``
         (the AC-6 active sidecar trained on the 15 effective
         calibration cells). Raise ``RuntimeError`` naming the
         expected file when v4 is absent — the sm_120 active
         path does NOT silently fall back to v3 (retired from
         the active path under AC-6.4), v2, or v1 (sm_86-trained).
      3. Else if ``target_arch == "sm_86"`` (explicit legacy
         scope), fall back to the legacy / historical-only
         ``data/local_labels/calibration_evaluation_v1.json``
         when it exists, otherwise return None.
      4. Else (``target_arch`` is None or any other value): return
         None. This preserves the legacy behavior for callers
         that did not pass --target-arch — the comparison runner
         continues to run without a calibration gate by default,
         matching every existing synthetic-fixture test's
         expectation. The sm_120 cohort scope MUST opt in via
         --target-arch sm_120 to receive the v4 default.
    """
    if explicit_override is not None:
        return explicit_override
    data_dir = repo_root / "data" / "local_labels"
    if target_arch == "sm_120":
        active = data_dir / "calibration_evaluation_v4.json"
        if active.is_file():
            return active
        raise RuntimeError(
            "comparison_runner: --target-arch sm_120 requires the "
            "sm_120-trained calibration sidecar at "
            "data/local_labels/calibration_evaluation_v4.json "
            "(paired with calibration_head_v4.json) but neither "
            "exists on disk. The sm_120 active path does NOT "
            "silently fall back to v3 (retired from the active "
            "path under AC-6.4), v2, or v1 (sm_86-trained). Run "
            "`python3 tools/local_labels/fit_calibration_head_v4_sm120.py` "
            "to emit v4 before re-running."
        )
    if target_arch == "sm_86":
        legacy = data_dir / "calibration_evaluation_v1.json"
        return legacy if legacy.is_file() else None
    return None


def write_bundle(bundle: dict, output_path: Path) -> None:
    """Serialize a comparison output bundle as canonical JSON.
    ``sort_keys=True`` plus deterministic per-cell ordering inside
    the runner makes the on-disk bytes reproducible across re-runs
    of the same fixture under the same seed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Operator-facing CLI wrapper (argparse over the library API)
# ---------------------------------------------------------------------


def _load_predictor_bundles_from_glob(pattern: str) -> list[dict]:
    """Walk ``pattern`` (a glob expression) and load each matching
    file as a JSON predictor bundle. The order is sorted so the
    CLI is byte-deterministic across runs."""
    bundles: list[dict] = []
    for path_str in sorted(_glob.glob(pattern)):
        with open(path_str) as f:
            bundles.append(json.load(f))
    return bundles


def _load_admitted_label_cells(label_admitted_path: Path) -> list[dict]:
    """Read ``data/local_labels/label_admitted.json`` and return its
    ``cells`` array. The admitted-cells artifact carries
    ``{schema_version, cells: [...]}`` (or just a list when the
    schema version is absent for legacy shapes); both are
    accepted."""
    with open(label_admitted_path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        # File-level schema-v5 boundary: a dict admitted artifact that declares
        # a schema_version other than 5 is a stale wire contract (pre-unit-
        # tagged substrate) and is rejected with a typed error rather than
        # silently loaded. A dict without a schema_version key falls through to
        # the per-cell schema-v5 gate in run_comparison.
        file_schema_version = payload.get("schema_version")
        if file_schema_version is not None and file_schema_version != 5:
            raise AdmittedLabelV5MigrationError(
                f"{label_admitted_path} declares schema_version "
                f"{file_schema_version!r}, but the comparison runner requires "
                f"the schema-v5 admitted contract; re-aggregate to v5 "
                f"({LOCAL_LABEL_ADMITTED_V5_MIGRATION_REQUIRED})."
            )
        cells = payload.get("cells")
        if isinstance(cells, list):
            return list(cells)
    return []


def main(argv: list[str] | None = None) -> int:
    """Operator-facing CLI entry point. Loads the contract, walks
    the predictor-bundle glob, reads the admitted-cell artifact,
    runs the comparison, and writes the deterministic schema-valid
    output bundle to ``--output``.

    Exit codes:
    - 0: output bundle written successfully (admitted arm OR
      typed_unavailable arm; both are valid runner outcomes per the
      coverage gate semantics).
    - 1: a typed runner exception fired (contract drift, unknown
      result_kind, scalar predicted_hit_rate out of range, or
      typed_reason not in the predictor enum). The exception's
      typed token is printed to stderr.
    - 2: argparse-level usage error (handled inside argparse).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Predictor-vs-NCU comparison runner. Loads the "
            "comparison contract, joins admitted predictor bundles "
            "against admitted NCU label cells on the contract-pinned "
            "7-field key, and writes a deterministic schema-valid "
            "output bundle."
        ),
    )
    parser.add_argument(
        "--contract",
        required=True,
        help="Path to comparison_contract.toml.",
    )
    parser.add_argument(
        "--predictor-bundles-glob",
        required=True,
        help=(
            "Glob expression matching the predictor bundle JSON "
            "files emitted by cache_predictor.write_bundle."
        ),
    )
    parser.add_argument(
        "--label-admitted-path",
        required=True,
        help=(
            "Path to data/local_labels/label_admitted.json. "
            "An empty admitted-cells array is permitted; the "
            "runner emits typed_unavailable when no headline "
            "cells clear the smoke-test floor."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to write the deterministic schema-valid "
            "comparison output bundle."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help=(
            "Pinned seed for the cell-level bootstrap CI. "
            "Re-runs under the same seed produce a byte-identical "
            "output bundle."
        ),
    )
    parser.add_argument(
        "--split-label",
        default="all",
        help=(
            "Label recorded in the output bundle's "
            "comparison_run_pairing_index.split_label field."
        ),
    )
    parser.add_argument(
        "--target-arch",
        default=None,
        help=(
            "Optional single-arch admission gate. When set to one "
            "of the contract's arch_set members (e.g. 'sm_86'), "
            "top-level admission gates on that arch's floor alone "
            "(the documented Verified-live-<arch> baseline shape). "
            "When omitted, top-level admission honors the "
            "contract's coverage_gate.require_both_arches_for_cross_arch "
            "setting (the canonical cross-arch behavior)."
        ),
    )
    parser.add_argument(
        "--calibration-evaluation-sidecar",
        type=Path,
        default=None,
        help=(
            # AC-6.4: v4 is the active sm_120 default; v3 is
            # retired from the active path under the AC-6.4
            # historical-only contract. v1 remains as the legacy
            # sm_86 default.
            "Optional explicit path to a calibration_evaluation "
            "sidecar (v1 or v4 shape; v4 produced by "
            "tools/local_labels/fit_calibration_head_v4_sm120.py; "
            "v1 produced by tools/local_labels/fit_calibration_head.py). "
            "When set, the override is honored verbatim regardless "
            "of --target-arch. When omitted, the active selection "
            "rule fires: --target-arch sm_120 defaults to "
            "data/local_labels/calibration_evaluation_v4.json (the "
            "sm_120-trained head) and FAILS CLOSED if v4 is absent; "
            "any other --target-arch falls back to the legacy / "
            "historical-only "
            "data/local_labels/calibration_evaluation_v1.json when "
            "that file exists; otherwise None (no calibration gate). "
            "Retired v3/v2 sidecars "
            "(data/local_labels/calibration_evaluation_v3.json, "
            "calibration_evaluation_v2.json) are kept on disk for "
            "audit but are NEVER consumed by the active sm_120 path."
        ),
    )
    parser.add_argument(
        "--v5-head-path",
        default=None,
        help=(
            "Optional path to the v5 monotonic calibration head sidecar "
            "(data/local_labels/calibration_head_v5.json by default). When "
            "set, the output bundle includes a per_arch_transfer_diagnostics "
            "block + v5_head_source_commit + v5_head_fit_status fields. "
            "Bundles carrying the canonical_pre_calibration_features block "
            "are routed through workloads/simbench_bw/canonical_features."
            "predict_active_path for the v5 active path (legacy v4 path "
            "stays available; v5 lives alongside)."
        ),
    )
    args = parser.parse_args(argv)

    # AC-6.4 active calibration sidecar selection rule (mirrors
    # the headline-runner wrapper): explicit override always
    # honored; --target-arch sm_120 defaults to v4 and fails
    # closed when v4
    # is absent; any other target_arch falls back to v1 when
    # present; otherwise None.
    calibration_sidecar_path = _resolve_calibration_evaluation_sidecar_path(
        explicit_override=args.calibration_evaluation_sidecar,
        target_arch=getattr(args, "target_arch", None),
        repo_root=_REPO_ROOT,
    )
    calibration_evaluation_sidecar: dict | None = None
    if calibration_sidecar_path is not None:
        with calibration_sidecar_path.open(
            "r", encoding="utf-8",
        ) as fh:
            calibration_evaluation_sidecar = json.load(fh)

    try:
        contract = load_comparison_contract(Path(args.contract))
        predictor_bundles = _load_predictor_bundles_from_glob(
            args.predictor_bundles_glob
        )
        label_cells = _load_admitted_label_cells(
            Path(args.label_admitted_path)
        )
        bundle = run_comparison(
            contract=contract,
            predictor_bundles=predictor_bundles,
            label_cells=label_cells,
            bootstrap_seed=args.bootstrap_seed,
            split_label=args.split_label,
            target_arch=args.target_arch,
            calibration_evaluation_sidecar=calibration_evaluation_sidecar,
            v5_head_path=args.v5_head_path,
        )
    except (
        ComparisonContractDrift,
        ComparisonRunnerUnknownResultKind,
        ComparisonRunnerPredictedHitRateOutOfRange,
        ComparisonRunnerBoundedIntervalInvalid,
        ComparisonRunnerTypedReasonOutOfEnum,
        ComparisonRunnerDuplicatePredictorBundle,
        AdmittedLabelV5MigrationError,
    ) as exc:
        print(
            f"comparison_runner: typed defect: {exc!s}",
            file=sys.stderr,
        )
        return 1

    write_bundle(bundle, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
