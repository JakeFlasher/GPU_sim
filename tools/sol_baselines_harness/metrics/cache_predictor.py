"""Per-launch cache hit-rate predictor.

Consumes a parsed Tier-2 sidecar ``LaunchFrame``, routes each
``CompactAccessRecord`` through ``cache_access_policy.route_access``
to decide L1/L2 participation, replays the eligible 32 B sector
ids through ``SetAssociativeSimulator`` (one instance for L1, one
for L2), and returns a result dict carrying raw L1 + L2 hit-rates
+ per-layer hit/total counts + provenance.

A separate composer (``compose_predictor_bundle``) turns the
result dict into a ``PredictorBundle`` (scalar or typed_unavailable
arm) suitable for serialization through
``predictor_bundle_schema.validate_predictor_bundle``. The
composer accepts a fitted ``IsotonicCalibrator`` and a
calibration-time confidence half-width; the predicted_hit_rate
field is the calibrated L2 read-only sector hit-rate when both
are supplied, otherwise the raw L2 hit-rate. The L1 hit-rate +
both layers' raw counts ride along as extra ``arm_payload``
fields so a downstream consumer can reproduce the predicted
hit-rate from the raw counts.

Fail-closed branches:

* Truncated frame (``frame.truncation is not None``) →
  ``trace_budget_exceeded`` typed-unavailable.
* Sampled / compressed frame
  (``frame.sampled_or_compressed_trace``) →
  ``trace_budget_exceeded`` typed-unavailable, deterministic
  seed derived from ``frame.launch_uid + frame.binary_sha256``.
* Unknown / uncalibrated routing
  (``cache_access_policy.AccessRouting.policy_uncalibrated``) →
  ``policy_uncalibrated`` typed-unavailable, provenance names
  the offending record's sequence_id + op_kind + cache_modifier
  + address_space.
* L1 or L2 layer with zero admissible accesses
  (no record routed to that layer) →
  ``no_admissible_accesses`` typed-unavailable, provenance
  names the empty layer.
* Missing / typed-unavailable geometry spec →
  ``cache_geometry_unresolved`` typed-unavailable.

The module imports only stdlib + sibling
``baselines_harness.metrics`` modules; no NCU dependency, no
clock-policy mutation, no privileged CUPTI surface.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .cache_access_policy import (
    AccessRouting,
    CACHE_MODIFIER_DEFAULT,
    route_access,
)
from .cache_geometry import resolve_cache_geometry
from .isotonic_calibrator import IsotonicCalibrator
from .predictor_bundle_schema import (
    PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
    PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE,
    ManifestPairingIndex,
    PredictorBundle,
    PredictorBundleSchemaError,
    validate_predictor_bundle,
)
from .predictor_design_coverage import (
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    lookup_predictor_design_coverage,
)
from .set_associative_simulator import SetAssociativeSimulator
from .tier2_sidecar_parser import CompactAccessRecord, LaunchFrame
from .typed_unavailable_bundle import (
    PREDICTOR_ABSTENTION_REASONS,
    TYPED_UNAVAILABLE_RESULT_KIND,
    make_typed_unavailable_bundle,
    validate_typed_unavailable_bundle,
)
from .warmup_protocol import (
    CACHE_INITIAL_STATE_COLD,
    CACHE_INITIAL_STATE_WARM,
    WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
    WARMUP_PROTOCOL_UNKNOWN,
)


# Python equivalent of the C++
# ``tier2_compute_deterministic_sampling_seed`` helper in
# ``nvbit_tracer/metric_extractor/tier2_frame_serializer.h``. Used
# to derive provenance seeds for sampled/compressed frames that
# do not carry a truncation marker (and therefore have no
# producer-written seed). The formula MUST match the C++ producer
# byte-for-byte so a downstream consumer can verify the seed
# against the truncation marker's value when both paths apply.
_TIER2_SAMPLING_SEED_PREFIX = b"solbench-tier2-sampling-v1\x00"


# Per-arch isotonic-calibrator parameter sidecar discovery.
# Each entry maps an `arch_tag` to a relative path under the repo
# root where the sidecar TOML lives. The predictor body looks up
# the calibrator params for the live host's arch by indexing this
# dict — there are NO flow-control branches on the arch string;
# new arch support lands by adding a new entry here AND committing
# the corresponding sidecar TOML next to the canonical
# `label_ledger_v1_<arch>_partition.toml` ledger.
#
# Empty string means "the canonical sm_86 fit-at-runtime path
# in `IsotonicCalibrator.fit(records)` is the authoritative
# source; no on-disk params file"; this preserves the existing
# sm_86 prediction-surface behavior byte-for-byte until DEC-C
# escalates to an explicit reserialization. The sm_120 sidecar
# is the first entry that ships a checked-in params file.
_CALIBRATOR_PARAMS_BY_ARCH: dict[str, str] = {
    "sm_86": "",
    "sm_120": (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "isotonic_params_sm120.toml"
    ),
}


# an earlier iteration / the design contract loader hardening: the per-arch live ledger + split
# paths the loader recomputes hashes from when the caller does NOT
# pass an explicit `expected_*_sha256` override. Empty string means
# "no on-disk ledger / split for this arch" — applies to the sm_86
# fit-at-runtime path; the loader skips the live-hash check on those.
_LEDGER_PATH_BY_ARCH: dict[str, str] = {
    "sm_86": "",
    "sm_120": (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "label_ledger_v1_sm120_partition.toml"
    ),
}


_VALIDATION_SPLIT_PATH_BY_ARCH: dict[str, str] = {
    "sm_86": "",
    "sm_120": (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "calibration_validation_split_v1_sm120.toml"
    ),
}


# an earlier iteration / the design contract loader hardening: closed structural vocabulary the
# loader accepts for `meta.calibrator_typed_reason`. Any value outside
# this set raises `calibrator_sidecar_typed_reason_invalid`. The
# vocabulary is intentionally a frozen union of the structural sm_120
# blockers (DEC-B, DEC-A, metric-family, repeatability) and the
# cohort-side reasons. Adding a new reason requires explicit
# operator-vetted code edit; the deployment surface refuses anything
# else fail-closed.
_VALID_SIDECAR_TYPED_UNAVAILABLE_REASONS: frozenset = frozenset({
    "sm120_l2_geometry_unverified",
    "sm120_fp8_block_scale_metric_unverified",
    "sm120_nvfp4_unsupported_consumer_arch",
    "sm120_tma_metric_family_unverified",
    "sm120_repeatability_threshold_exceeded",
    "hostile_subclass_uncalibrated",
    "local_memory_private_cache_policy",
    "shared_memory_no_read_only_sector_traffic",
    "warm_cold_protocol_unpinned",
})


# Closed typed-reason vocabulary for the sidecar loader. Every
# failure surface raises ``PredictorCalibratorLoaderError`` with
# one of these reasons so the deployed report layer can render the
# diagnostic without parsing message strings.
PREDICTOR_CALIBRATOR_LOADER_TYPED_REASONS: frozenset = frozenset({
    "calibrator_unknown_arch_tag",
    "calibrator_sidecar_missing",
    "calibrator_sidecar_malformed_toml",
    "calibrator_sidecar_missing_required_field",
    "calibrator_sidecar_arch_mismatch",
    "calibrator_sidecar_ledger_hash_mismatch",
    "calibrator_sidecar_anchor_shape_invalid",
    "calibrator_sidecar_metric_family_invalid",
    "calibrator_sidecar_typed_reason_invalid",
    "calibrator_live_artifact_missing",
})


class PredictorCalibratorLoaderError(Exception):
    """Raised by ``load_calibrator_for_arch`` on any failure to
    reconstruct a fitted ``IsotonicCalibrator`` from an on-disk
    sidecar. Carries the typed reason string so callers compose
    typed-unavailable bundles or report-table errors without
    parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in PREDICTOR_CALIBRATOR_LOADER_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of "
                f"{sorted(PREDICTOR_CALIBRATOR_LOADER_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


_SIDECAR_REQUIRED_META_FIELDS: tuple = (
    "schema_version",
    "arch",
    "captured_at_utc",
    "ledger_sha256",
    "validation_split_sha256",
    "fit_record_count",
    "admitted_cell_count",
    "ncu_version",
    "driver_version",
    "calibrator_class_name",
    "accepted_metric_selectors",
)


# the design contract reproducibility-block fields: every seed input the fitter
# consumed must be stamped in the persisted sidecar so a future
# re-fit on the same ledger SHA-256 + same dependency versions
# produces byte-identical params. The load-time gate refuses a
# sidecar that omits any of these.
_SIDECAR_REQUIRED_REPRODUCIBILITY_FIELDS: tuple = (
    "python_version",
    "numpy_version",
    "scipy_version",
    "sklearn_version",
    "fitter_sort_key",
    "float_format",
)


# an earlier iteration typed-unavailable sidecar marker: returned by
# `load_calibrator_for_arch` when a sidecar reports
# `meta.calibrator_status = "typed_unavailable"`. The composer
# inspects this instance to thread the calibrator's typed
# reason into the bundle's reasons array (e.g.
# `sm120_l2_geometry_unverified`).
from dataclasses import dataclass as _round4_dataclass


@_round4_dataclass(frozen=True)
class TypedUnavailableSidecar:
    arch: str
    calibrator_typed_reason: str
    ledger_sha256: str
    validation_split_sha256: str


_FORBIDDEN_METRIC_FAMILIES: tuple = (
    # Store-conflated metric families — must NEVER appear in a
    # sidecar's accepted_metric_selectors list. See the project
    # metric-family discipline and the read-only-sector carve-out.
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "syslts__t_sector_hit_rate.pct",
)


def _resolve_sidecar_path(relative: str):
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / relative


def _sha256_of_file(path) -> str:
    """SHA-256 of the file at ``path`` as a hex string. Used by the
    loader to compute live ledger / split hashes for the design contract
    default-on validation gate.
    """
    import hashlib
    from pathlib import Path
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _validate_sidecar_meta(
    meta: "Mapping[str, Any]",
    reproducibility: "Mapping[str, Any]",
    *,
    expected_arch: str,
    expected_ledger_sha256: "Optional[str]" = None,
    expected_split_sha256: "Optional[str]" = None,
) -> None:
    for field in _SIDECAR_REQUIRED_META_FIELDS:
        if field not in meta:
            raise PredictorCalibratorLoaderError(
                "calibrator_sidecar_missing_required_field",
                f"sidecar [meta] is missing required field "
                f"{field!r}",
            )
    # the design contract reproducibility block: re-running on the same ledger
    # SHA-256 with the same dependency versions must produce
    # byte-identical params. Every seed input is a load-time gate.
    for field in _SIDECAR_REQUIRED_REPRODUCIBILITY_FIELDS:
        if field not in reproducibility:
            raise PredictorCalibratorLoaderError(
                "calibrator_sidecar_missing_required_field",
                f"sidecar [reproducibility] is missing required "
                f"field {field!r}",
            )
    if meta["arch"] != expected_arch:
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_arch_mismatch",
            f"sidecar [meta].arch = {meta['arch']!r}; expected "
            f"{expected_arch!r}",
        )
    selectors = list(meta.get("accepted_metric_selectors", ()))
    for selector in selectors:
        for forbidden in _FORBIDDEN_METRIC_FAMILIES:
            if selector == forbidden:
                raise PredictorCalibratorLoaderError(
                    "calibrator_sidecar_metric_family_invalid",
                    f"sidecar [meta].accepted_metric_selectors "
                    f"contains forbidden store-conflated metric "
                    f"family {forbidden!r}",
                )
    if expected_ledger_sha256 is not None:
        if meta["ledger_sha256"] != expected_ledger_sha256:
            raise PredictorCalibratorLoaderError(
                "calibrator_sidecar_ledger_hash_mismatch",
                f"sidecar [meta].ledger_sha256 = "
                f"{meta['ledger_sha256']!r}; expected "
                f"{expected_ledger_sha256!r} from the live ledger",
            )
    if expected_split_sha256 is not None:
        if meta["validation_split_sha256"] != expected_split_sha256:
            raise PredictorCalibratorLoaderError(
                "calibrator_sidecar_ledger_hash_mismatch",
                f"sidecar [meta].validation_split_sha256 = "
                f"{meta['validation_split_sha256']!r}; expected "
                f"{expected_split_sha256!r} from the live "
                f"calibration_validation_split file",
            )
    # an earlier iteration / when the sidecar reports
    # `calibrator_status = "typed_unavailable"`, the
    # `calibrator_typed_reason` MUST come from the closed structural
    # vocabulary. Missing-field or out-of-vocabulary values raise
    # `calibrator_sidecar_typed_reason_invalid` — the deployment
    # surface refuses to load free-form reasons.
    if meta.get("calibrator_status") == "typed_unavailable":
        if "calibrator_typed_reason" not in meta:
            raise PredictorCalibratorLoaderError(
                "calibrator_sidecar_typed_reason_invalid",
                "typed-unavailable sidecar is missing required "
                "field 'calibrator_typed_reason'",
            )
        reason = meta["calibrator_typed_reason"]
        if reason not in _VALID_SIDECAR_TYPED_UNAVAILABLE_REASONS:
            raise PredictorCalibratorLoaderError(
                "calibrator_sidecar_typed_reason_invalid",
                f"sidecar [meta].calibrator_typed_reason = "
                f"{reason!r}; expected one of "
                f"{sorted(_VALID_SIDECAR_TYPED_UNAVAILABLE_REASONS)}",
            )


def _validate_anchor_block(
    block: "Mapping[str, Any]", *, head_label: str,
) -> "tuple[list, list]":
    x = block.get("x_thresholds")
    y = block.get("y_thresholds")
    if x is None or y is None:
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_anchor_shape_invalid",
            f"sidecar [{head_label}] is missing x_thresholds "
            f"or y_thresholds",
        )
    if not isinstance(x, list) or not isinstance(y, list):
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_anchor_shape_invalid",
            f"sidecar [{head_label}] x/y must be lists; got "
            f"{type(x).__name__} / {type(y).__name__}",
        )
    if len(x) != len(y):
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_anchor_shape_invalid",
            f"sidecar [{head_label}] x_thresholds count {len(x)} "
            f"!= y_thresholds count {len(y)}",
        )
    if len(x) < 1:
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_anchor_shape_invalid",
            f"sidecar [{head_label}] anchor lists are empty",
        )
    return x, y


def load_calibrator_for_arch(
    arch_tag: str,
    *,
    expected_ledger_sha256: "Optional[str]" = None,
    expected_split_sha256: "Optional[str]" = None,
    sidecar_path_override: "Optional[str]" = None,
    ledger_path_override: "Optional[str]" = None,
    split_path_override: "Optional[str]" = None,
):
    """Return a fitted ``IsotonicCalibrator`` reconstructed from
    the on-disk sidecar for ``arch_tag``, or ``None`` when the
    arch's dispatch entry is the empty string (the legacy
    fit-at-runtime path).

    Schema gates:
      * ``arch_tag`` must be in ``_CALIBRATOR_PARAMS_BY_ARCH``
        (raises ``calibrator_unknown_arch_tag``).
      * The sidecar file must exist (raises
        ``calibrator_sidecar_missing``).
      * The sidecar TOML must parse and have every required
        meta field (raises
        ``calibrator_sidecar_malformed_toml`` or
        ``calibrator_sidecar_missing_required_field``).
      * ``meta.arch`` must equal ``arch_tag`` (raises
        ``calibrator_sidecar_arch_mismatch``).
      * No selector in ``meta.accepted_metric_selectors`` may
        match a forbidden store-conflated family (raises
        ``calibrator_sidecar_metric_family_invalid``).
      * When ``expected_ledger_sha256`` is supplied, the
        sidecar's ``ledger_sha256`` must match it (raises
        ``calibrator_sidecar_ledger_hash_mismatch``).
      * ``l1_head`` / ``l2_head`` anchor lists must be
        same-length non-empty (raises
        ``calibrator_sidecar_anchor_shape_invalid``).

    The returned calibrator carries the persisted L1 + L2 head
    state (sklearn ``IsotonicRegression`` reconstructed from
    the anchor points + ``out_of_bounds="clip"``). It is
    callable via ``predict_l1`` / ``predict_l2`` immediately;
    no separate ``fit`` call is required.

    Path-override parameters (``sidecar_path_override`` etc.)
    let a variant-aware caller (e.g. the coverage-expansion
    variant wrapper) point the loader at a non-default sidecar /
    ledger / split for the same ``arch_tag`` while preserving
    the same schema and hash gates. Each override defaults to
    ``None`` so the historical lookup signature is byte-stable
    for every existing caller. When supplied, the override
    replaces the per-arch table lookup for that artifact only.
    """
    import tomllib
    if arch_tag not in _CALIBRATOR_PARAMS_BY_ARCH:
        raise PredictorCalibratorLoaderError(
            "calibrator_unknown_arch_tag",
            f"arch_tag {arch_tag!r} is not in "
            f"_CALIBRATOR_PARAMS_BY_ARCH; known arches: "
            f"{sorted(_CALIBRATOR_PARAMS_BY_ARCH)}",
        )
    if sidecar_path_override is not None:
        relative = sidecar_path_override
    else:
        relative = _CALIBRATOR_PARAMS_BY_ARCH[arch_tag]
    if not relative:
        return None
    path = _resolve_sidecar_path(relative)
    if not path.exists():
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_missing",
            f"sidecar {path} does not exist for arch_tag "
            f"{arch_tag!r}",
        )
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise PredictorCalibratorLoaderError(
            "calibrator_sidecar_malformed_toml",
            f"sidecar {path} does not parse as TOML: {exc!r}",
        ) from exc
    meta = data.get("meta") or {}
    reproducibility = data.get("reproducibility") or {}
    # an earlier iteration / the design contract loader hardening: when the caller does NOT pass
    # explicit `expected_*_sha256` values, default to recomputing the
    # live on-disk ledger + split hashes from the arch-keyed dispatch
    # tables. This makes the deployment surface fail-closed against
    # stale sidecars by default — a sidecar stamped from yesterday's
    # ledger raises `calibrator_sidecar_ledger_hash_mismatch` without
    # any caller-side hash discipline.
    #
    # an earlier iteration / Codex review mainline gap #2: if the arch has a
    # configured non-empty live ledger / split path but that file
    # does not exist on disk AND the caller did not supply the
    # corresponding `expected_*_sha256`, the loader cannot perform
    # the default-on validation. Fail closed with
    # `calibrator_live_artifact_missing` instead of silently leaving
    # `expected_*_sha256` as None.
    if ledger_path_override is not None:
        ledger_relative = ledger_path_override
    elif arch_tag in _LEDGER_PATH_BY_ARCH:
        ledger_relative = _LEDGER_PATH_BY_ARCH[arch_tag]
    else:
        ledger_relative = ""
    if ledger_relative:
        ledger_path = _resolve_sidecar_path(ledger_relative)
        if ledger_path.exists():
            if expected_ledger_sha256 is None:
                expected_ledger_sha256 = _sha256_of_file(ledger_path)
        elif expected_ledger_sha256 is None:
            raise PredictorCalibratorLoaderError(
                "calibrator_live_artifact_missing",
                f"arch_tag {arch_tag!r} has configured live ledger "
                f"path {ledger_path} which does not exist on disk, "
                f"and no explicit expected_ledger_sha256 was supplied; "
                f"the default-on hash check cannot be performed",
            )
    if split_path_override is not None:
        split_relative = split_path_override
    elif arch_tag in _VALIDATION_SPLIT_PATH_BY_ARCH:
        split_relative = _VALIDATION_SPLIT_PATH_BY_ARCH[arch_tag]
    else:
        split_relative = ""
    if split_relative:
        split_path = _resolve_sidecar_path(split_relative)
        if split_path.exists():
            if expected_split_sha256 is None:
                expected_split_sha256 = _sha256_of_file(split_path)
        elif expected_split_sha256 is None:
            raise PredictorCalibratorLoaderError(
                "calibrator_live_artifact_missing",
                f"arch_tag {arch_tag!r} has configured live validation "
                f"split path {split_path} which does not exist on disk, "
                f"and no explicit expected_split_sha256 was supplied; "
                f"the default-on hash check cannot be performed",
            )
    # the design contract hardening: validate the sidecar schema FIRST (an earlier iteration
    # fix to Codex blocking #1). Even when the sidecar is
    # typed-unavailable, the schema MUST be valid — bypassing
    # validation would let a malformed or arch-mismatched typed-
    # unavailable sidecar pass silently. After the meta + anchor
    # validation, branch on `calibrator_status`.
    _validate_sidecar_meta(
        meta,
        reproducibility,
        expected_arch=arch_tag,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_split_sha256=expected_split_sha256,
    )
    # the design contract ledger-side hook: when the sidecar's
    # `meta.calibrator_status` is `"typed_unavailable"`, the
    # calibrator is intentionally absent (sm_120 in the DEC-B-
    # pending state). Return a `TypedUnavailableSidecar` so the
    # composer can propagate the typed reason into the bundle's
    # reasons array. The sm_86 legacy path (empty sidecar entry)
    # still returns `None` via the early return at line 343.
    if meta.get("calibrator_status") == "typed_unavailable":
        return TypedUnavailableSidecar(
            arch=meta["arch"],
            calibrator_typed_reason=meta.get(
                "calibrator_typed_reason",
                "calibrator_typed_unavailable_unspecified",
            ),
            ledger_sha256=meta["ledger_sha256"],
            validation_split_sha256=meta["validation_split_sha256"],
        )
    l1_block = data.get("l1_head") or {}
    l2_block = data.get("l2_head") or {}
    l1_x, l1_y = _validate_anchor_block(l1_block, head_label="l1_head")
    l2_x, l2_y = _validate_anchor_block(l2_block, head_label="l2_head")
    from sklearn.isotonic import IsotonicRegression
    head_l1 = IsotonicRegression(out_of_bounds="clip")
    head_l1.fit(l1_x, l1_y)
    head_l2 = IsotonicRegression(out_of_bounds="clip")
    head_l2.fit(l2_x, l2_y)
    calibrator = IsotonicCalibrator()
    calibrator._l1_head = head_l1
    calibrator._l2_head = head_l2
    calibrator._admitted_cell_count = int(meta["admitted_cell_count"])
    return calibrator


_REQUIRED_GEOMETRY_FIELDS: tuple = (
    "l2_capacity_bytes",
    "num_l2_slices",
    "associativity_l2",
    "line_size_bytes",
    "sector_size_bytes",
)


def _validate_geometry_spec(
    spec: Mapping[str, Any],
) -> tuple:
    """Validate the five required integer fields of a
    geometry spec AND the simulator's two consistency
    invariants.

    Returns ``(bad_field_name, bad_kind)`` where
    ``bad_kind`` is one of ``"missing"``, ``"non_int"``,
    ``"non_positive"``, ``"non_integral_set_count"``,
    ``"line_sector_mismatch"``, or
    ``"policy_outside_closed_vocabulary"``. Returns
    ``(None, None)`` when every required field is a
    positive integer AND the simulator's consistency
    checks pass.

    Per-field checks:
      * Field present.
      * Field is an integer (with explicit bool guard
        because ``isinstance(True, int)`` is True in
        Python).
      * Field value is positive.

    Consistency invariants (mirroring the
    ``SetAssociativeSimulator.__init__`` checks the
    validator runs upstream of):
      * ``(l2_capacity_bytes // num_l2_slices) %
        (line_size_bytes * associativity_l2) == 0`` —
        bytes-per-slice must divide evenly by
        bytes-per-set so ``num_sets_per_slice`` is an
        integer. Diagnostic kind:
        ``non_integral_set_count`` on field
        ``num_sets_per_slice``.
      * ``line_size_bytes % sector_size_bytes == 0`` —
        line size must be a multiple of the sector size
        so ``sectors_per_line`` is an integer.
        Diagnostic kind: ``line_sector_mismatch`` on
        field ``line_size_bytes``.

    Consistency checks run AFTER the per-field checks
    pass, so a missing / non-int / non-positive field
    is reported with the per-field kind first (not
    buried under an inconsistency).
    """
    for field in _REQUIRED_GEOMETRY_FIELDS:
        value = spec.get(field)
        if value is None:
            return field, "missing"
        if isinstance(value, bool) or not isinstance(value, int):
            return field, "non_int"
        if value <= 0:
            return field, "non_positive"
    # Consistency invariant 1: bytes-per-slice must
    # divide evenly by bytes-per-set so num_sets_per_slice
    # is an integer. Mirrors the simulator's check at
    # set_associative_simulator.py:87-94.
    bytes_per_slice = spec["l2_capacity_bytes"] // spec["num_l2_slices"]
    bytes_per_set = spec["line_size_bytes"] * spec["associativity_l2"]
    if bytes_per_slice % bytes_per_set != 0:
        return "num_sets_per_slice", "non_integral_set_count"
    # Consistency invariant 2: line_size_bytes must be a
    # multiple of sector_size_bytes so sectors_per_line is
    # an integer. Mirrors the simulator's check at
    # set_associative_simulator.py:97-101.
    if spec["line_size_bytes"] % spec["sector_size_bytes"] != 0:
        return "line_size_bytes", "line_sector_mismatch"
    # Consistency invariant 3: when the geometry spec
    # carries an explicit ``set_index_policy``, the value
    # must be in the simulator's closed vocabulary. The
    # field defaults to ``page_aligned_shift7`` when absent
    # so legacy specs stay valid; only an explicit invalid
    # value fail-closes.
    policy = spec.get("set_index_policy")
    if policy is not None and policy not in (
        "page_aligned_shift7",
        "line_indexed",
    ):
        return "set_index_policy", "policy_outside_closed_vocabulary"
    return None, None


def _compute_deterministic_sampling_seed(
    launch_uid: int, binary_sha256: bytes,
) -> int:
    if len(binary_sha256) != 32:
        raise ValueError(
            f"binary_sha256 must be 32 bytes; got {len(binary_sha256)}",
        )
    ctx = hashlib.sha256()
    ctx.update(_TIER2_SAMPLING_SEED_PREFIX)
    ctx.update(launch_uid.to_bytes(8, "little"))
    ctx.update(binary_sha256)
    digest = ctx.digest()
    return int.from_bytes(digest[:8], "little")


# Typed-unavailable reasons emitted by the predictor body. Every
# reason here MUST be in
# ``typed_unavailable_bundle.PREDICTOR_ABSTENTION_REASONS`` so
# ``make_typed_unavailable_bundle`` accepts it.
TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED = "trace_budget_exceeded"
TYPED_UNAVAILABLE_REASON_POLICY_UNCALIBRATED = "policy_uncalibrated"
TYPED_UNAVAILABLE_REASON_NO_ADMISSIBLE_ACCESSES = "no_admissible_accesses"
TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED = "cache_geometry_unresolved"
TYPED_UNAVAILABLE_REASON_COVERAGE_BELOW_SMOKE_FLOOR = "coverage_below_smoke_floor"
TYPED_UNAVAILABLE_REASON_WARMUP_COUNT_OUT_OF_RANGE = "warmup_count_out_of_range"
TYPED_UNAVAILABLE_REASON_FRAMES_SEQUENCE_EMPTY = "frames_sequence_empty"
TYPED_UNAVAILABLE_REASON_MEASURED_CACHE_INITIAL_STATE_INVALID = (
    "measured_cache_initial_state_invalid"
)


_REPLAY_STATUS_OK = "ok"
_REPLAY_STATUS_POLICY_UNCALIBRATED = "policy_uncalibrated"

# sm_86 L1 data-cache associativity. The CUDA Programming Guide
# compute-capability-8.6 tables document the L1 data cache as
# 4-way set-associative on Ampere (per-SM unified L1/scratchpad
# with operator-configurable carveout). Future arches override
# this via the `associativity_l1` field on the geometry spec
# when present; the bare default is used when the resolver
# omits it.
_DEFAULT_L1_ASSOCIATIVITY_BY_ARCH = {
    "sm_86": 4,
    "sm_120": 4,
    "sm_100a": 4,
}
_FALLBACK_L1_ASSOCIATIVITY = 4


def _build_l1_geometry_spec(
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive an L1-sized geometry dict from the L2 geometry
    spec the resolver supplies.

    The deployed simulator constructor accepts the canonical
    five-field geometry (``l2_capacity_bytes``,
    ``num_l2_slices``, ``associativity_l2``, ``line_size_bytes``,
    ``sector_size_bytes``). The L1 simulator instance reuses the
    same constructor by re-purposing those fields so each L1
    instance models the per-SM L1 cache capacity, not the
    shared 6 MiB L2:

      * ``l2_capacity_bytes`` -> ``l1_kb_per_sm * 1024``
        (the per-SM L1 unified-cache size in bytes).
      * ``num_l2_slices`` -> 1 (L1 is per-SM, no slicing
        across the chip from a single-SM-trace view).
      * ``associativity_l2`` -> ``associativity_l1`` from the
        spec if present, otherwise the arch-specific default
        (sm_86 = 4-way per the CUDA Programming Guide
        compute-capability tables).
      * ``line_size_bytes`` and ``sector_size_bytes`` -> mirror
        the L2 geometry (the on-chip cache line and sector
        sizes are common across L1 and L2 on Ampere /
        Blackwell).
      * ``geometry_source`` is preserved so a downstream
        consumer can identify whether the geometry came from
        the static device card or an operator override.

    Fails closed when ``l1_kb_per_sm`` is missing or
    non-positive: returns ``None`` and the caller surfaces a
    typed-unavailable bundle naming the upstream field.
    """
    l1_kb = spec.get("l1_kb_per_sm")
    if l1_kb is None:
        return None
    if isinstance(l1_kb, bool) or not isinstance(l1_kb, int):
        return None
    if l1_kb <= 0:
        return None
    arch = spec.get("arch")
    associativity_l1 = spec.get(
        "associativity_l1",
        _DEFAULT_L1_ASSOCIATIVITY_BY_ARCH.get(
            arch, _FALLBACK_L1_ASSOCIATIVITY,
        ),
    )
    return {
        "arch": arch,
        "l2_capacity_bytes": l1_kb * 1024,
        "num_l2_slices": 1,
        "associativity_l2": associativity_l1,
        "line_size_bytes": spec["line_size_bytes"],
        "sector_size_bytes": spec["sector_size_bytes"],
        "geometry_source": spec.get("geometry_source"),
        # Mirror the L1 capacity-source field so a downstream
        # consumer can audit which side of the spec produced
        # this geometry without re-deriving.
        "l1_kb_per_sm": l1_kb,
        # The L1 set-index hash MUST be line-indexed rather
        # than the L2 page-aligned shift-7 hash: the L1
        # capacity is per-SM (~192 KB on sm_86), and the
        # legacy shift-7 hash collapses a 4 KiB sector range
        # into a single global slot, which causes pathological
        # eviction patterns on whole-kernel access traces
        # that don't match real L1 cache behavior. The
        # line-indexed mapping is the canonical
        # set-associative-cache routing (each consecutive line
        # lands in a distinct set), and it preserves the L1
        # eviction signal that the coupled L1->L2 scoring
        # helper depends on.
        "set_index_policy": "line_indexed",
    }


def _replay_records_through_caches(
    records: Sequence[Any],
    *,
    l1_simulator: SetAssociativeSimulator,
    l2_simulator: SetAssociativeSimulator,
    count_hits: bool,
    constant_memory_counts_for_l2_read_score: bool = False,
) -> Dict[str, Any]:
    """Replay every sector in ``records`` through the coupled
    L1 -> L2 cache hierarchy. For each access record, route it
    via ``cache_access_policy.route_access`` and apply the
    coupled-hierarchy semantics:

    * If ``routing.l1_read`` is true, the sector is probed in
      ``l1_simulator``. On L1 miss AND
      ``routing.counts_for_l2_read_score`` is true, the sector
      then probes ``l2_simulator`` (the sector reached L2 due
      to L1 inclusion). L1 hits stay in L1 and do not reach the
      L2 read denominator -- this matches NCU's
      ``lts__t_sectors_srcunit_tex_op_read.sum`` classification,
      which only counts sectors that traverse the LSU pipe to
      the tex source unit AFTER missing L1.
    * If ``routing.l1_read`` is false AND
      ``routing.counts_for_l2_read_score`` is true, the sector
      probes ``l2_simulator`` directly (L1-bypass / atomic /
      TMA-style reads that skip L1 by policy).
    * Otherwise -- store-only classes, shared-memory traffic,
      local-memory traffic -- the sector touches neither layer
      and contributes to neither denominator.

    When ``count_hits`` is true, the helper returns per-layer
    hit + total counts for the measured scoring path. When
    false, the helper still mutates the simulator state (the
    warmup replay path: the cache state evolves but the hit
    counts are not surfaced to the caller).

    The helper does not abort on
    ``routing.policy_uncalibrated`` -- it returns a result dict
    with ``status = "policy_uncalibrated"`` carrying the
    offending record so the caller can build an appropriately-
    framed typed-unavailable bundle.
    """
    l1_hit_count = 0
    l1_total = 0
    l2_hit_count = 0
    l2_total = 0
    for record in records:
        routing = route_access(
            op_kind=record.op_kind,
            address_space=record.address_space,
            cache_modifier=record.cache_modifier,
            l1_access_policy=getattr(
                record, "l1_access_policy", 0,
            ),
            constant_memory_counts_for_l2_read_score=(
                constant_memory_counts_for_l2_read_score
            ),
        )
        if routing.policy_uncalibrated:
            return {
                "status": _REPLAY_STATUS_POLICY_UNCALIBRATED,
                "offending_record": record,
            }
        for sector_id in record.sector_ids:
            l1_was_miss = False
            if routing.l1_read:
                l1_result = l1_simulator.access(sector_id)
                if count_hits:
                    l1_total += 1
                    if l1_result == SetAssociativeSimulator.HIT:
                        l1_hit_count += 1
                l1_was_miss = (
                    l1_result != SetAssociativeSimulator.HIT
                )
            if routing.counts_for_l2_read_score:
                # L2 sees the sector when:
                #   (a) the L1 probe missed (coupled inclusion
                #       semantics), OR
                #   (b) the routing bypasses L1 entirely (direct-L2
                #       reads such as ``.cg`` cache modifier,
                #       atomics, TMA reads).
                reach_l2 = l1_was_miss if routing.l1_read else True
                if reach_l2:
                    l2_result = l2_simulator.access(sector_id)
                    if count_hits:
                        l2_total += 1
                        if l2_result == SetAssociativeSimulator.HIT:
                            l2_hit_count += 1
    return {
        "status": _REPLAY_STATUS_OK,
        "l1_hit_count": l1_hit_count,
        "l1_total_count": l1_total,
        "l2_hit_count": l2_hit_count,
        "l2_total_count": l2_total,
    }


def predict_from_launch_frame(
    frame: LaunchFrame,
    *,
    cell_name: str,
    geometry_spec: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a per-launch hit-rate prediction or a typed-
    unavailable abstention for ``frame``.

    Existing fail-closed branches are preserved verbatim:

    * Truncated frames (``frame.truncation is not None``) and
      sampled/compressed frames return a typed-unavailable bundle
      with reason ``trace_budget_exceeded``; the bundle reasons
      list also carries the partition's
      ``typed_unavailable_reason`` first when ``cell_name`` is a
      predeclared typed-unavailable cell. Provenance carries the
      producer-written seed (truncation path) or the
      deterministic Python-mirror seed (sampled path).

    * Clean frames flow into the new predictor body: every
      ``CompactAccessRecord`` is routed via
      ``cache_access_policy.route_access(op_kind, address_space,
      cache_modifier)``. Records whose routing is
      ``policy_uncalibrated`` fail the WHOLE launch to a typed-
      unavailable bundle with reason ``policy_uncalibrated`` and
      provenance naming the offending record's sequence_id +
      decision triplet. Otherwise:

      - sectors from records with ``l1_read=True`` are replayed
        through an L1 simulator instance;
      - sectors from records with ``counts_for_l2_read_score=True``
        are replayed through an L2 simulator instance (the L2
        read-only sector hit-rate the calibration corpus
        targets);
      - the L1 and L2 raw hit-rates + per-layer hit/total counts
        are returned in the result dict's ``arm_payload``
        equivalent fields.

    * ``geometry_spec`` MUST be a fully-resolved dict from
      ``cache_geometry.resolve_cache_geometry`` (or
      compatibly-shaped fixture). ``None`` or a spec whose
      ``geometry_source == 'typed_unavailable'`` routes the
      launch to typed-unavailable with reason
      ``cache_geometry_unresolved``.

    * L1 OR L2 with zero admissible accesses (no record routed
      to that layer) routes the launch to typed-unavailable
      with reason ``no_admissible_accesses`` and provenance
      naming the empty layer.

    The result dict shape:

    * Scalar path: ``{result_kind: 'scalar',
      raw_l1_hit_rate, raw_l2_hit_rate, l1_hit_count,
      l1_total_count, l2_hit_count, l2_total_count, provenance}``.

    * Typed-unavailable path: ``{result_kind: 'typed_unavailable',
      reasons, provenance, predicted_l1_hit_rate=None,
      predicted_l2_hit_rate=None}`` (legacy dict shape preserved
      for backward compatibility with the abstention-branch
      callers).

    The result is intentionally a dict, NOT a ``PredictorBundle``:
    bundle composition needs caller-supplied identity (pairing
    index, algorithm version, input sidecar digests) and
    calibration state (fitted calibrator + LOOCV half-width)
    that a single launch cannot supply on its own. The companion
    ``compose_predictor_bundle`` helper takes the result dict +
    the caller-supplied context and produces a validated
    ``PredictorBundle``.
    """
    if frame.truncation is not None:
        trunc = frame.truncation
        provenance = {
            "launch_uid": frame.launch_uid,
            "deterministic_sampling_seed": trunc.deterministic_sampling_seed,
            "next_sequence_id_not_emitted": trunc.next_sequence_id_not_emitted,
            "dropped_record_count_estimate": trunc.dropped_record_count_estimate,
            "budget_bytes_limit": trunc.budget_bytes_limit,
            "budget_bytes_used_before_marker": trunc.budget_bytes_used_before_marker,
            "abstention_cause": "truncation_marker",
        }
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED,
                ),
                provenance=provenance,
            )
        )
    if frame.sampled_or_compressed_trace:
        seed = _compute_deterministic_sampling_seed(
            frame.launch_uid, frame.binary_sha256,
        )
        provenance = {
            "launch_uid": frame.launch_uid,
            "deterministic_sampling_seed": seed,
            "abstention_cause": "sampled_or_compressed_trace",
            "binary_sha256": frame.binary_sha256.hex(),
        }
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED,
                ),
                provenance=provenance,
            )
        )
    # Clean-frame predictor path.
    #
    # Route through the predictor-design-coverage partition
    # BEFORE simulator work. A predeclared typed-unavailable
    # cell (e.g., shared-memory,
    # warm-cold-protocol-unpinned, persisting-window) must
    # never produce a scalar prediction regardless of what the
    # simulator would say on its raw output — the partition
    # contract pins these cells as untrustworthy by design.
    entry = lookup_predictor_design_coverage(cell_name)
    if entry.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                # The partition's own typed_unavailable_reason
                # auto-injects as the first entry; no extra
                # reasons needed.
                extra_reasons=(),
                provenance={
                    "launch_uid": frame.launch_uid,
                    "cell_name": cell_name,
                    "abstention_cause": "predictor_design_coverage",
                    "predictor_bucket": entry.predictor_bucket,
                    "partition_reason": entry.typed_unavailable_reason,
                },
            )
        )
    # Geometry validation. The simulator requires five
    # integer fields; missing / non-positive / non-int
    # values must fail closed BEFORE simulator construction
    # rather than letting `SetAssociativeSimulator.__init__`
    # propagate a `ValueError` past the typed-unavailable
    # boundary.
    if geometry_spec is None or geometry_spec.get(
        "geometry_source"
    ) == "typed_unavailable":
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED,
                ),
                provenance={
                    "launch_uid": frame.launch_uid,
                    "abstention_cause": "geometry_unresolved",
                    "geometry_source": (
                        geometry_spec.get("geometry_source")
                        if geometry_spec
                        else None
                    ),
                },
            )
        )
    bad_field, bad_kind = _validate_geometry_spec(geometry_spec)
    if bad_field is not None:
        provenance: Dict[str, Any] = {
            "launch_uid": frame.launch_uid,
            "abstention_cause": "geometry_field_invalid",
            "offending_field": bad_field,
            "offending_field_kind": bad_kind,
        }
        # For per-field diagnostics ("missing" / "non_int" /
        # "non_positive") carry the offending value verbatim.
        # For consistency diagnostics
        # ("non_integral_set_count" / "line_sector_mismatch")
        # carry the relevant subset of geometry values so a
        # downstream consumer can reproduce the divisibility
        # arithmetic.
        if bad_kind == "non_integral_set_count":
            provenance["l2_capacity_bytes"] = (
                geometry_spec["l2_capacity_bytes"]
            )
            provenance["num_l2_slices"] = geometry_spec["num_l2_slices"]
            provenance["associativity_l2"] = geometry_spec["associativity_l2"]
            provenance["line_size_bytes"] = geometry_spec["line_size_bytes"]
            provenance["sector_size_bytes"] = (
                geometry_spec["sector_size_bytes"]
            )
        elif bad_kind == "line_sector_mismatch":
            provenance["line_size_bytes"] = geometry_spec["line_size_bytes"]
            provenance["sector_size_bytes"] = (
                geometry_spec["sector_size_bytes"]
            )
        else:
            provenance["offending_field_value"] = (
                geometry_spec.get(bad_field)
            )
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED,
                ),
                provenance=provenance,
            )
        )
    # L1-specific geometry (capacity derived from
    # ``l1_kb_per_sm`` so the L1 simulator models the per-SM
    # data-cache size, not the shared 6 MiB L2). Falls closed to
    # ``cache_geometry_unresolved`` when ``l1_kb_per_sm`` is
    # missing / non-positive -- the L2 geometry is already
    # validated above; the L1 side is independently required.
    l1_geometry_spec = _build_l1_geometry_spec(geometry_spec)
    if l1_geometry_spec is None:
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED,
                ),
                provenance={
                    "launch_uid": frame.launch_uid,
                    "abstention_cause": "geometry_field_invalid",
                    "offending_field": "l1_kb_per_sm",
                    "offending_field_kind": "missing_or_non_positive",
                    "offending_field_value": (
                        geometry_spec.get("l1_kb_per_sm")
                    ),
                },
            )
        )
    # Coupled L1 -> L2 replay. ``_replay_records_through_caches``
    # routes each record once, probes L1 first when
    # ``routing.l1_read`` is true, and feeds L2 only on L1 miss
    # (or on direct-L2 routes where L1 is bypassed). This mirrors
    # NCU's L2 read-only sector classification: L1 hits do not
    # reach the LSU-pipe -> tex source-unit denominator. L1 and
    # L2 use distinct geometries so the L1 eviction signal that
    # drives the L1-miss feed actually reflects the per-SM
    # working-set boundary.
    l1_simulator = SetAssociativeSimulator(l1_geometry_spec)
    l2_simulator = SetAssociativeSimulator(geometry_spec)
    replay = _replay_records_through_caches(
        frame.records,
        l1_simulator=l1_simulator,
        l2_simulator=l2_simulator,
        count_hits=True,
        constant_memory_counts_for_l2_read_score=bool(
            geometry_spec.get(
                "l2_constant_memory_counts_for_read_score", False,
            )
        ),
    )
    if replay["status"] == _REPLAY_STATUS_POLICY_UNCALIBRATED:
        offending = replay["offending_record"]
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_POLICY_UNCALIBRATED,
                ),
                provenance={
                    "launch_uid": frame.launch_uid,
                    "abstention_cause": "policy_uncalibrated",
                    "offending_sequence_id": offending.sequence_id,
                    "offending_op_kind": offending.op_kind,
                    "offending_cache_modifier": (
                        offending.cache_modifier
                    ),
                    "offending_address_space": (
                        offending.address_space
                    ),
                },
            )
        )
    l1_hit_count = replay["l1_hit_count"]
    l1_total = replay["l1_total_count"]
    l2_hit_count = replay["l2_hit_count"]
    l2_total = replay["l2_total_count"]
    # Zero-admission gate per layer. Under coupled hierarchy a
    # zero L2 total means every L1-admitted sector hit in L1 AND
    # no direct-L2 route admitted a sector -- a legitimate
    # signal that the kernel's read working set fits entirely in
    # L1 and the L2 read hit-rate is undefined.
    if l1_total == 0 or l2_total == 0:
        empty_layer = (
            "l1" if l1_total == 0 and l2_total != 0
            else "l2" if l2_total == 0 and l1_total != 0
            else "both"
        )
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_NO_ADMISSIBLE_ACCESSES,
                ),
                provenance={
                    "launch_uid": frame.launch_uid,
                    "abstention_cause": "no_admissible_accesses",
                    "empty_layer": empty_layer,
                    "l1_admissible_access_count": l1_total,
                    "l2_admissible_access_count": l2_total,
                },
            )
        )
    raw_l1_hit_rate = l1_hit_count / l1_total
    raw_l2_hit_rate = l2_hit_count / l2_total
    return {
        "result_kind": PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
        "raw_l1_hit_rate": raw_l1_hit_rate,
        "raw_l2_hit_rate": raw_l2_hit_rate,
        "l1_hit_count": l1_hit_count,
        "l1_total_count": l1_total,
        "l2_hit_count": l2_hit_count,
        "l2_total_count": l2_total,
        "provenance": {
            "launch_uid": frame.launch_uid,
            "cell_name": cell_name,
            "geometry_source": geometry_spec.get("geometry_source"),
            "l2_capacity_bytes": geometry_spec.get("l2_capacity_bytes"),
            "num_l2_slices": geometry_spec.get("num_l2_slices"),
        },
    }


_OFFENDING_FRAME_ROLE_WARMUP = "warmup"
_OFFENDING_FRAME_ROLE_MEASURED = "measured"


def _resolve_warmed_sequence_protocol(
    warmup_count: int,
    measured_cache_initial_state: str,
) -> tuple:
    """Return the canonical ``(warmup_protocol,
    cache_initial_state)`` pair for ``warmup_count`` +
    ``measured_cache_initial_state``.

    ``warmup_count >= 1`` corresponds to NCU's
    ``--launch-skip N --launch-count 1`` invocation. Under
    NCU's default ``--replay-mode kernel`` (the project-
    pinned calibration invocation), the measured launch
    begins with a COLD cache because NCU re-runs the
    profiled kernel multiple times and flushes L1 between
    metric replays. The caller
    therefore drives the helper with
    ``measured_cache_initial_state="cold"`` to match. For
    cells calibrated under a protocol that preserves the
    warmed cache (e.g., ``--replay-mode application`` or
    a hand-crafted fixture), the caller passes
    ``measured_cache_initial_state="warm"`` and the helper
    returns the WARM state. The protocol family stays
    ``ncu_launch_skip_n_count_1`` in both cases.

    ``warmup_count == 0`` is the cold-equivalent degenerate
    path the helper accepts for parity with
    ``predict_from_launch_frame`` on a single-frame
    sequence. The canonical ``derive_warmup_protocol`` helper
    rejects ``launch_skip_count == 0`` because it falls
    outside the recognized NCU + warm-cold-pair families;
    the warmed-sequence helper signals the same condition by
    emitting ``warmup_protocol=unknown`` plus a cold initial
    cache state regardless of the
    ``measured_cache_initial_state`` argument.
    """
    if warmup_count >= 1:
        return (
            WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
            measured_cache_initial_state,
        )
    return WARMUP_PROTOCOL_UNKNOWN, CACHE_INITIAL_STATE_COLD


def predict_warmed_launch_sequence(
    frames: Sequence[LaunchFrame],
    *,
    warmup_count: int,
    cell_name: str,
    geometry_spec: Optional[Mapping[str, Any]] = None,
    measured_cache_initial_state: str = CACHE_INITIAL_STATE_WARM,
) -> Dict[str, Any]:
    """Compute a per-launch hit-rate prediction for the
    ``warmup_count``-th frame, with the measured-frame cache
    initial state controlled by
    ``measured_cache_initial_state``.

    Two prediction modes:

    * ``measured_cache_initial_state="warm"`` (legacy
      default): one ``SetAssociativeSimulator`` pair (L1 + L2)
      is constructed once, then `frames[0..warmup_count)` are
      replayed silently into those simulators before the
      final `frames[warmup_count]` is measured with hit counts.
      Matches a protocol that preserves the warmed cache
      across the warmup → measurement boundary (e.g.,
      ``--replay-mode application``).

    * ``measured_cache_initial_state="cold"`` (NCU
      ``--replay-mode kernel`` truth): fresh L1 + L2
      simulators are constructed for the measured frame
      `frames[warmup_count]` and the warmup-frame replay
      is SKIPPED. Reflects NCU's kernel-replay behavior:
      the profiler re-runs the kernel multiple times for
      different metric subsets and flushes L1 between
      metric replays, so the "measured" launch the harness
      observes is a cold-start replay even though it
      follows N nominal warmup launches in trace order.
      Truncation / sampled-trace / policy-uncalibrated
      checks on the warmup frames are still performed (an
      offending warmup frame indicates a fundamentally
      unhonest trace and propagates the abstention).

    Mirrors NCU's `--launch-skip N --launch-count 1` semantics:
    `warmup_count` warmup launches followed by one measured
    launch on the same warmed cache. Cell calibration ledgers
    label the `warmup_count + 1`-th launch (1-indexed) /
    frame index `warmup_count` (0-indexed); the helper makes
    that same frame the measured frame.

    Returns the same result dict shape as
    ``predict_from_launch_frame``:

      * Scalar path: ``{result_kind: 'scalar', raw_l1_hit_rate,
        raw_l2_hit_rate, l1_hit_count, l1_total_count,
        l2_hit_count, l2_total_count, provenance}``.
      * Typed-unavailable path:
        ``{result_kind: 'typed_unavailable', reasons,
        provenance, predicted_l1_hit_rate=None,
        predicted_l2_hit_rate=None}``.

    The provenance dict on the scalar path adds six
    warmed-sequence fields beyond the cold-path provenance:

      * ``warmup_count`` (int, the caller-supplied parameter)
      * ``launch_skip_count`` (alias of ``warmup_count``
        matching the canonical ledger field name in
        ``warmup_protocol.WarmupProtocolRecord``)
      * ``warmup_protocol`` — for ``warmup_count >= 1``
        this is the canonical
        ``"ncu_launch_skip_n_count_1"`` (matching the
        ledger entries captured with NCU
        ``--launch-skip N --launch-count 1``); for
        ``warmup_count == 0`` the cold-equivalent
        degenerate path emits
        ``"unknown"`` (no canonical NCU/warm-cold-pair
        family covers a zero-warmup measurement)
      * ``cache_initial_state`` — for ``warmup_count >= 1``
        this is ``"warm"`` (the N preceding warmups
        established a warm cache at the start of the
        measured launch); for ``warmup_count == 0`` it is
        ``"cold"`` (the simulator pair starts fresh and no
        warmup ran)
      * ``measured_frame_uid`` (the launch_uid of the frame
        whose hit/miss counts populated the result)
      * ``total_frame_count`` (``len(frames)``)

    Abstention precedence:

      1. **Partition typed-unavailable** — a cell whose
         canonical partition bucket is
         ``predictor_typed_unavailable`` short-circuits
         FIRST, before frames / warmup_count / geometry
         validation. The partition's disposition is the
         most fundamental contract; an unmapped or empty
         frames sequence does not override it.
      2. **Call-level validation** — ``frames`` empty
         (``frames_sequence_empty``) or ``warmup_count``
         non-int / boolean / negative /
         ``>= len(frames)``
         (``warmup_count_out_of_range``).
      3. **Geometry validation** — ``None`` spec,
         ``geometry_source == 'typed_unavailable'``, or an
         invalid field (``cache_geometry_unresolved``).
      4. **Per-frame abstentions** during warmup OR
         measurement propagate verbatim from
         ``predict_from_launch_frame``'s per-frame logic:

         * Truncation marker → ``trace_budget_exceeded``
           with provenance naming ``offending_frame_index``
           + ``offending_frame_role`` (``"warmup"`` or
           ``"measured"``).
         * Sampled / compressed trace →
           ``trace_budget_exceeded`` with the same
           offending-frame provenance.
         * ``policy_uncalibrated`` access →
           ``policy_uncalibrated`` with the same
           offending-frame provenance + offending record's
           sequence_id + decision triplet.
         * Measured frame's L1 OR L2 has zero admissible
           accesses → ``no_admissible_accesses`` with
           provenance naming the empty layer.

    The helper is import-clean for the deployed framework
    path: no NCU dependency, no clock-policy mutation, no
    privileged CUPTI surface, no calibration-subtree import.
    """
    # Validate ``measured_cache_initial_state`` ahead of any
    # other check: invalid values fail closed via the
    # typed-unavailable path with a typed reason that is
    # specific to the cold/warm semantic. The valid surface
    # is exactly {warm, cold}; the ``unknown`` sentinel is
    # NOT accepted here because the prediction caller MUST
    # commit to one or the other (downstream protocol-match
    # would route an unknown to a different typed reason).
    if measured_cache_initial_state not in (
        CACHE_INITIAL_STATE_WARM, CACHE_INITIAL_STATE_COLD,
    ):
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_MEASURED_CACHE_INITIAL_STATE_INVALID,
                ),
                provenance={
                    "cell_name": cell_name,
                    "abstention_cause": (
                        "measured_cache_initial_state_invalid"
                    ),
                    "measured_cache_initial_state": (
                        measured_cache_initial_state
                    ),
                    "warmup_count": warmup_count,
                    "launch_skip_count": warmup_count,
                    "total_frame_count": len(frames),
                },
            )
        )

    # Canonical partition disposition: a predeclared
    # typed-unavailable cell short-circuits FIRST so the
    # partition's disposition wins over frames /
    # warmup_count / geometry contract violations.
    entry = lookup_predictor_design_coverage(cell_name)
    if entry.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(),
                provenance={
                    "cell_name": cell_name,
                    "abstention_cause": "predictor_design_coverage",
                    "predictor_bucket": entry.predictor_bucket,
                    "partition_reason": entry.typed_unavailable_reason,
                    "warmup_count": warmup_count,
                    "launch_skip_count": warmup_count,
                    "total_frame_count": len(frames),
                },
            )
        )

    # Call-level validation: empty frames + warmup bounds.
    # ``warmup_count`` rejects booleans explicitly because
    # ``isinstance(True, int)`` is ``True`` in Python.
    if len(frames) == 0:
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_FRAMES_SEQUENCE_EMPTY,
                ),
                provenance={
                    "cell_name": cell_name,
                    "abstention_cause": "frames_sequence_empty",
                    "warmup_count": warmup_count,
                    "launch_skip_count": warmup_count,
                    "total_frame_count": 0,
                },
            )
        )
    if (
        isinstance(warmup_count, bool)
        or not isinstance(warmup_count, int)
        or warmup_count < 0
        or warmup_count >= len(frames)
    ):
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_WARMUP_COUNT_OUT_OF_RANGE,
                ),
                provenance={
                    "cell_name": cell_name,
                    "abstention_cause": "warmup_count_out_of_range",
                    "warmup_count": warmup_count,
                    "launch_skip_count": warmup_count,
                    "total_frame_count": len(frames),
                },
            )
        )

    # Geometry validation.
    if geometry_spec is None or geometry_spec.get(
        "geometry_source"
    ) == "typed_unavailable":
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED,
                ),
                provenance={
                    "cell_name": cell_name,
                    "abstention_cause": "geometry_unresolved",
                    "geometry_source": (
                        geometry_spec.get("geometry_source")
                        if geometry_spec
                        else None
                    ),
                    "warmup_count": warmup_count,
                    "launch_skip_count": warmup_count,
                    "total_frame_count": len(frames),
                },
            )
        )
    bad_field, bad_kind = _validate_geometry_spec(geometry_spec)
    if bad_field is not None:
        provenance: Dict[str, Any] = {
            "cell_name": cell_name,
            "abstention_cause": "geometry_field_invalid",
            "offending_field": bad_field,
            "offending_field_kind": bad_kind,
            "warmup_count": warmup_count,
            "launch_skip_count": warmup_count,
            "total_frame_count": len(frames),
        }
        if bad_kind == "non_integral_set_count":
            provenance["l2_capacity_bytes"] = (
                geometry_spec["l2_capacity_bytes"]
            )
            provenance["num_l2_slices"] = geometry_spec["num_l2_slices"]
            provenance["associativity_l2"] = (
                geometry_spec["associativity_l2"]
            )
            provenance["line_size_bytes"] = (
                geometry_spec["line_size_bytes"]
            )
            provenance["sector_size_bytes"] = (
                geometry_spec["sector_size_bytes"]
            )
        elif bad_kind == "line_sector_mismatch":
            provenance["line_size_bytes"] = (
                geometry_spec["line_size_bytes"]
            )
            provenance["sector_size_bytes"] = (
                geometry_spec["sector_size_bytes"]
            )
        else:
            provenance["offending_field_value"] = (
                geometry_spec.get(bad_field)
            )
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED,
                ),
                provenance=provenance,
            )
        )

    warmup_protocol_value, cache_initial_state_value = (
        _resolve_warmed_sequence_protocol(
            warmup_count, measured_cache_initial_state,
        )
    )
    measured_frame = frames[warmup_count]
    # L1-specific geometry (capacity derived from
    # ``l1_kb_per_sm`` so the L1 simulator models the per-SM
    # data-cache size, not the shared 6 MiB L2). Fails closed
    # the same way the cold-path predictor does when
    # ``l1_kb_per_sm`` is missing / non-positive.
    l1_geometry_spec = _build_l1_geometry_spec(geometry_spec)
    if l1_geometry_spec is None:
        return _bundle_to_dict(
            make_typed_unavailable_bundle(
                cell_name,
                extra_reasons=(
                    TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED,
                ),
                provenance={
                    "cell_name": cell_name,
                    "abstention_cause": "geometry_field_invalid",
                    "offending_field": "l1_kb_per_sm",
                    "offending_field_kind": "missing_or_non_positive",
                    "offending_field_value": (
                        geometry_spec.get("l1_kb_per_sm")
                    ),
                    "launch_uid": measured_frame.launch_uid,
                    "warmup_count": warmup_count,
                    "launch_skip_count": warmup_count,
                    "warmup_protocol": warmup_protocol_value,
                    "cache_initial_state": cache_initial_state_value,
                    "total_frame_count": len(frames),
                },
            )
        )
    l1_simulator = SetAssociativeSimulator(l1_geometry_spec)
    l2_simulator = SetAssociativeSimulator(geometry_spec)
    l1_hit_count = 0
    l2_hit_count = 0
    l1_total = 0
    l2_total = 0

    for frame_index in range(warmup_count + 1):
        frame = frames[frame_index]
        is_measured = frame_index == warmup_count
        offending_frame_role = (
            _OFFENDING_FRAME_ROLE_MEASURED
            if is_measured
            else _OFFENDING_FRAME_ROLE_WARMUP
        )

        if frame.truncation is not None:
            trunc = frame.truncation
            return _bundle_to_dict(
                make_typed_unavailable_bundle(
                    cell_name,
                    extra_reasons=(
                        TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED,
                    ),
                    provenance={
                        "cell_name": cell_name,
                        "abstention_cause": "truncation_marker",
                        "launch_uid": frame.launch_uid,
                        "offending_frame_index": frame_index,
                        "offending_frame_role": offending_frame_role,
                        "deterministic_sampling_seed": (
                            trunc.deterministic_sampling_seed
                        ),
                        "next_sequence_id_not_emitted": (
                            trunc.next_sequence_id_not_emitted
                        ),
                        "dropped_record_count_estimate": (
                            trunc.dropped_record_count_estimate
                        ),
                        "budget_bytes_limit": trunc.budget_bytes_limit,
                        "budget_bytes_used_before_marker": (
                            trunc.budget_bytes_used_before_marker
                        ),
                        "warmup_count": warmup_count,
                        "launch_skip_count": warmup_count,
                        "warmup_protocol": warmup_protocol_value,
                        "cache_initial_state": cache_initial_state_value,
                        "total_frame_count": len(frames),
                    },
                )
            )
        if frame.sampled_or_compressed_trace:
            seed = _compute_deterministic_sampling_seed(
                frame.launch_uid, frame.binary_sha256,
            )
            return _bundle_to_dict(
                make_typed_unavailable_bundle(
                    cell_name,
                    extra_reasons=(
                        TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED,
                    ),
                    provenance={
                        "cell_name": cell_name,
                        "abstention_cause": "sampled_or_compressed_trace",
                        "launch_uid": frame.launch_uid,
                        "binary_sha256": frame.binary_sha256.hex(),
                        "deterministic_sampling_seed": seed,
                        "offending_frame_index": frame_index,
                        "offending_frame_role": offending_frame_role,
                        "warmup_count": warmup_count,
                        "launch_skip_count": warmup_count,
                        "warmup_protocol": warmup_protocol_value,
                        "cache_initial_state": cache_initial_state_value,
                        "total_frame_count": len(frames),
                    },
                )
            )

        # Cold-mode measured-frame access: instantiate FRESH
        # L1 + L2 simulators just before scoring the measured
        # frame so the warmed state accumulated by the warmup
        # replay is discarded. Mirrors NCU
        # ``--replay-mode kernel`` cache-flush behavior at the
        # measured launch boundary. Warm-mode keeps the
        # simulators warmed by prior iterations.
        if (
            is_measured
            and measured_cache_initial_state
            == CACHE_INITIAL_STATE_COLD
        ):
            l1_simulator = SetAssociativeSimulator(
                l1_geometry_spec,
            )
            l2_simulator = SetAssociativeSimulator(
                geometry_spec,
            )

        replay = _replay_records_through_caches(
            frame.records,
            l1_simulator=l1_simulator,
            l2_simulator=l2_simulator,
            count_hits=is_measured,
            constant_memory_counts_for_l2_read_score=bool(
                geometry_spec.get(
                    "l2_constant_memory_counts_for_read_score", False,
                )
            ),
        )
        if replay["status"] == _REPLAY_STATUS_POLICY_UNCALIBRATED:
            offending = replay["offending_record"]
            return _bundle_to_dict(
                make_typed_unavailable_bundle(
                    cell_name,
                    extra_reasons=(
                        TYPED_UNAVAILABLE_REASON_POLICY_UNCALIBRATED,
                    ),
                    provenance={
                        "cell_name": cell_name,
                        "abstention_cause": "policy_uncalibrated",
                        "launch_uid": frame.launch_uid,
                        "offending_frame_index": frame_index,
                        "offending_frame_role": offending_frame_role,
                        "offending_sequence_id": offending.sequence_id,
                        "offending_op_kind": offending.op_kind,
                        "offending_cache_modifier": (
                            offending.cache_modifier
                        ),
                        "offending_address_space": (
                            offending.address_space
                        ),
                        "warmup_count": warmup_count,
                        "launch_skip_count": warmup_count,
                        "warmup_protocol": warmup_protocol_value,
                        "cache_initial_state": cache_initial_state_value,
                        "total_frame_count": len(frames),
                    },
                )
            )

        if is_measured:
            l1_hit_count = replay["l1_hit_count"]
            l1_total = replay["l1_total_count"]
            l2_hit_count = replay["l2_hit_count"]
            l2_total = replay["l2_total_count"]
            if l1_total == 0 or l2_total == 0:
                empty_layer = (
                    "l1" if l1_total == 0 and l2_total != 0
                    else "l2" if l2_total == 0 and l1_total != 0
                    else "both"
                )
                return _bundle_to_dict(
                    make_typed_unavailable_bundle(
                        cell_name,
                        extra_reasons=(
                            TYPED_UNAVAILABLE_REASON_NO_ADMISSIBLE_ACCESSES,
                        ),
                        provenance={
                            "cell_name": cell_name,
                            "abstention_cause": "no_admissible_accesses",
                            "launch_uid": frame.launch_uid,
                            "offending_frame_index": frame_index,
                            "offending_frame_role": (
                                _OFFENDING_FRAME_ROLE_MEASURED
                            ),
                            "empty_layer": empty_layer,
                            "l1_admissible_access_count": l1_total,
                            "l2_admissible_access_count": l2_total,
                            "warmup_count": warmup_count,
                            "launch_skip_count": warmup_count,
                            "warmup_protocol": warmup_protocol_value,
                            "cache_initial_state": cache_initial_state_value,
                            "total_frame_count": len(frames),
                        },
                    )
                )

    raw_l1_hit_rate = l1_hit_count / l1_total
    raw_l2_hit_rate = l2_hit_count / l2_total
    return {
        "result_kind": PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
        "raw_l1_hit_rate": raw_l1_hit_rate,
        "raw_l2_hit_rate": raw_l2_hit_rate,
        "l1_hit_count": l1_hit_count,
        "l1_total_count": l1_total,
        "l2_hit_count": l2_hit_count,
        "l2_total_count": l2_total,
        "provenance": {
            "launch_uid": measured_frame.launch_uid,
            "cell_name": cell_name,
            "geometry_source": geometry_spec.get("geometry_source"),
            "l2_capacity_bytes": geometry_spec.get("l2_capacity_bytes"),
            "num_l2_slices": geometry_spec.get("num_l2_slices"),
            "warmup_count": warmup_count,
            "launch_skip_count": warmup_count,
            "warmup_protocol": warmup_protocol_value,
            "cache_initial_state": cache_initial_state_value,
            "total_frame_count": len(frames),
            "measured_frame_uid": measured_frame.launch_uid,
        },
    }


def _bundle_to_dict(bundle) -> Dict[str, Any]:
    """Convert a ``TypedUnavailableBundle`` dataclass to the
    legacy dict shape this module's abstention-branch callers
    expect. The factory's validator has already fired by the
    time this runs; the dict is purely a backward-compatible
    serialization at the edge.

    The dict adds two predictor-specific scalar placeholders
    (``predicted_l1_hit_rate``, ``predicted_l2_hit_rate`` both
    set to ``None``) that the factory's bundle does not carry
    because they are predictor-output fields, not bundle-input
    fields.
    """
    validate_typed_unavailable_bundle(bundle)
    return {
        "result_kind": bundle.result_kind,
        "reasons": list(bundle.reasons),
        "provenance": dict(bundle.provenance),
        "predicted_l1_hit_rate": None,
        "predicted_l2_hit_rate": None,
    }


def compose_predictor_bundle(
    launch_result: Mapping[str, Any],
    *,
    manifest_pairing_index: ManifestPairingIndex,
    algorithm_version: str,
    input_sidecar_digests: tuple = (),
    predictor_design_coverage: str,
    calibrator: Optional[IsotonicCalibrator] = None,
    ci_half_width: Optional[float] = None,
    ci_confidence: float = 0.95,
    arch_tag: Optional[str] = None,
) -> PredictorBundle:
    """Compose a ``PredictorBundle`` from a
    ``predict_from_launch_frame`` result dict.

    Routes to the scalar arm when ``launch_result['result_kind']``
    is ``'scalar'`` AND a fitted ``calibrator`` + a non-None
    ``ci_half_width`` are supplied. Otherwise routes to the
    typed_unavailable arm:

      * If ``launch_result['result_kind'] ==
        'typed_unavailable'`` the bundle is built from the
        existing reasons + provenance.
      * If ``launch_result['result_kind'] == 'scalar'`` but the
        calibration context is missing, the bundle is built
        with reason ``coverage_below_smoke_floor`` (no
        calibrated head supplied) and provenance carrying the
        raw L1/L2 counts so a downstream consumer still sees
        the per-layer accounting.

    The composed bundle is validated through
    ``validate_predictor_bundle`` before return — invalid
    inputs (bad pairing index, off-canonical
    ci_confidence, missing input sidecar digests, etc.)
    raise ``PredictorBundleSchemaError``.

    The scalar arm's predicted_hit_rate is the calibrated L2
    read-only sector hit-rate — the metric the calibration
    corpus targets, computed from the L2 simulator's per-
    layer hit/total counts (records gated by
    `counts_for_l2_read_score=True` from the routing layer
    so stores and local-memory traffic are excluded). The L1
    rate + both layers' raw counts ride along as extra
    arm_payload fields so a downstream consumer can
    reproduce the predicted hit-rate from the raw counts.
    """
    result_kind = launch_result.get("result_kind")
    cell_name = (
        launch_result.get("provenance", {}).get("cell_name", "")
        or manifest_pairing_index.slug
    )
    # the design contract / taskC1 arch-keyed runtime dispatch: when the caller did
    # NOT supply a fitted calibrator AND an arch_tag is supplied,
    # route through `load_calibrator_for_arch(arch_tag)`. sm_86 has
    # an empty-string sidecar entry (legacy fit-at-runtime); the
    # loader returns None there and composition falls through to the
    # typed-unavailable arm. Unknown arch_tags raise the loader's
    # typed error. Operator-supplied calibrators always win.
    # an earlier iteration: a `TypedUnavailableSidecar` return means the sidecar
    # is structurally fail-closed (e.g. sm_120 + DEC-B pending); the
    # composer threads the sidecar's `calibrator_typed_reason` into
    # the bundle's reasons array so the structural state surfaces in
    # the deployed output (NOT the generic
    # `coverage_below_smoke_floor`).
    sidecar_typed_unavailable_reason: Optional[str] = None
    if calibrator is None and arch_tag is not None:
        loaded = load_calibrator_for_arch(arch_tag)
        if isinstance(loaded, TypedUnavailableSidecar):
            sidecar_typed_unavailable_reason = (
                loaded.calibrator_typed_reason
            )
        elif loaded is not None:
            calibrator = loaded
    if (
        result_kind == PREDICTOR_BUNDLE_RESULT_KIND_SCALAR
        and calibrator is not None
        and calibrator.is_fitted
        and ci_half_width is not None
    ):
        return _compose_scalar_arm(
            launch_result=launch_result,
            manifest_pairing_index=manifest_pairing_index,
            algorithm_version=algorithm_version,
            input_sidecar_digests=input_sidecar_digests,
            predictor_design_coverage=predictor_design_coverage,
            calibrator=calibrator,
            ci_half_width=ci_half_width,
            ci_confidence=ci_confidence,
        )
    if result_kind == TYPED_UNAVAILABLE_RESULT_KIND:
        reasons = tuple(launch_result.get("reasons", ()))
        provenance = dict(launch_result.get("provenance", {}))
    elif sidecar_typed_unavailable_reason is not None:
        # Structural fail-closed: the sidecar told us the calibrator
        # is typed-unavailable. Surface that reason verbatim instead
        # of a generic coverage abstention.
        reasons = (sidecar_typed_unavailable_reason,)
        provenance = {
            "cell_name": cell_name,
            "abstention_cause": "sidecar_typed_unavailable",
            "sidecar_typed_reason": sidecar_typed_unavailable_reason,
        }
    else:
        # Scalar result without calibrator → coverage abstention.
        reasons = (
            TYPED_UNAVAILABLE_REASON_COVERAGE_BELOW_SMOKE_FLOOR,
        )
        provenance = {
            "cell_name": cell_name,
            "abstention_cause": "no_calibrator_supplied",
        }
    # Thread the raw simulator counts when the source
    # `launch_result` carried them (the scalar-result fallback
    # path). Typed-unavailable result dicts from the predictor
    # body do not carry these because the simulator never ran.
    raw_l1 = launch_result.get("raw_l1_hit_rate")
    raw_l2 = launch_result.get("raw_l2_hit_rate")
    return _compose_typed_unavailable_arm(
        reasons=reasons,
        provenance=provenance,
        manifest_pairing_index=manifest_pairing_index,
        algorithm_version=algorithm_version,
        input_sidecar_digests=input_sidecar_digests,
        predictor_design_coverage=predictor_design_coverage,
        raw_l1_hit_rate=raw_l1,
        raw_l2_hit_rate=raw_l2,
        l1_hit_count=launch_result.get("l1_hit_count"),
        l1_total_count=launch_result.get("l1_total_count"),
        l2_hit_count=launch_result.get("l2_hit_count"),
        l2_total_count=launch_result.get("l2_total_count"),
    )


def _compose_scalar_arm(
    *,
    launch_result: Mapping[str, Any],
    manifest_pairing_index: ManifestPairingIndex,
    algorithm_version: str,
    input_sidecar_digests: tuple,
    predictor_design_coverage: str,
    calibrator: IsotonicCalibrator,
    ci_half_width: float,
    ci_confidence: float,
) -> PredictorBundle:
    raw_l2 = float(launch_result["raw_l2_hit_rate"])
    raw_l1 = float(launch_result["raw_l1_hit_rate"])
    calibrated_l2 = float(calibrator.predict_l2(raw_l2))
    calibrated_l1 = float(calibrator.predict_l1(raw_l1))
    # `predicted_hit_rate` MUST remain the calibrated L2
    # value. The CI half-width is the calibration-corpus-
    # level confidence band; clipping the band's bounds to
    # [0, 1] does NOT mutate the model output. Re-centering
    # the prediction to a clamped midpoint would silently
    # disagree with what `calibrator.predict_l2(raw_l2)`
    # returned.
    predicted_hit_rate = calibrated_l2
    ci_low = max(0.0, calibrated_l2 - ci_half_width)
    ci_high = min(1.0, calibrated_l2 + ci_half_width)
    # Schema invariant: `ci_half_width == (ci_high - ci_low) /
    # 2` within 1e-9. When the symmetric interval clamps at a
    # [0,1] boundary the reported half-width must reflect the
    # actual asymmetric clipped width / 2, not the original
    # input parameter.
    actual_half_width = (ci_high - ci_low) / 2.0
    arm_payload = {
        "predicted_hit_rate": predicted_hit_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": actual_half_width,
        "ci_confidence": ci_confidence,
        # Extra fields the validator allows: per-layer raw
        # counts + the calibrator's per-layer point estimates
        # so a downstream consumer can audit the chain from
        # raw count → raw rate → calibrated rate → reported
        # predicted_hit_rate. The L2 read-only sector hit-
        # rate is the calibration target; the L1 rate is
        # reported alongside as supplementary measurement.
        "raw_l1_hit_rate": raw_l1,
        "raw_l2_hit_rate": raw_l2,
        "calibrated_l1_hit_rate": calibrated_l1,
        "calibrated_l2_hit_rate": calibrated_l2,
        "l1_hit_count": launch_result["l1_hit_count"],
        "l1_total_count": launch_result["l1_total_count"],
        "l2_hit_count": launch_result["l2_hit_count"],
        "l2_total_count": launch_result["l2_total_count"],
    }
    bundle = PredictorBundle(
        result_kind=PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
        predictor_design_coverage=predictor_design_coverage,
        manifest_pairing_index=manifest_pairing_index,
        algorithm_version=algorithm_version,
        arm_payload=arm_payload,
        input_sidecar_digests=tuple(input_sidecar_digests),
    )
    validate_predictor_bundle(bundle)
    return bundle


def _compose_typed_unavailable_arm(
    *,
    reasons: tuple,
    provenance: Mapping[str, Any],
    manifest_pairing_index: ManifestPairingIndex,
    algorithm_version: str,
    input_sidecar_digests: tuple,
    predictor_design_coverage: str,
    raw_l1_hit_rate: Optional[float] = None,
    raw_l2_hit_rate: Optional[float] = None,
    l1_hit_count: Optional[int] = None,
    l1_total_count: Optional[int] = None,
    l2_hit_count: Optional[int] = None,
    l2_total_count: Optional[int] = None,
) -> PredictorBundle:
    """Build a typed-unavailable arm with the upstream
    provenance + raw counts threaded through as extra
    schema-allowed fields. The required schema fields
    (`typed_reason`, `underconfigured_bin_count`,
    `admitted_bin_count`) are emitted alongside the
    optional `typed_reasons` (tuple of all reasons in
    order), `provenance` (dict), and any present raw
    L1/L2 hit-rates + counts. The schema's typed-
    unavailable forbidden list only forbids the scalar
    + bounded fields, so these extras pass validation.
    """
    typed_reason = reasons[0] if reasons else ""
    arm_payload: Dict[str, Any] = {
        "typed_reason": typed_reason,
        "underconfigured_bin_count": 0,
        "admitted_bin_count": 0,
    }
    # Thread the full reasons tuple + provenance dict so a
    # downstream consumer sees every contributing reason and
    # the cause/launch_uid/cell_name provenance the predictor
    # body assembled.
    if reasons:
        arm_payload["typed_reasons"] = tuple(reasons)
    if provenance:
        arm_payload["provenance"] = dict(provenance)
    # Thread the raw L1/L2 counts when they were captured by
    # the predictor body before abstention (e.g., a clean
    # frame that produced a scalar result then fell through
    # to typed-unavailable because no calibrator was
    # supplied). Downstream consumers can still audit the
    # raw simulator output on a fail-closed bundle.
    if raw_l1_hit_rate is not None:
        arm_payload["raw_l1_hit_rate"] = raw_l1_hit_rate
    if raw_l2_hit_rate is not None:
        arm_payload["raw_l2_hit_rate"] = raw_l2_hit_rate
    if l1_hit_count is not None:
        arm_payload["l1_hit_count"] = l1_hit_count
    if l1_total_count is not None:
        arm_payload["l1_total_count"] = l1_total_count
    if l2_hit_count is not None:
        arm_payload["l2_hit_count"] = l2_hit_count
    if l2_total_count is not None:
        arm_payload["l2_total_count"] = l2_total_count
    bundle = PredictorBundle(
        result_kind=PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE,
        predictor_design_coverage=predictor_design_coverage,
        manifest_pairing_index=manifest_pairing_index,
        algorithm_version=algorithm_version,
        arm_payload=arm_payload,
        input_sidecar_digests=tuple(input_sidecar_digests),
    )
    validate_predictor_bundle(bundle)
    return bundle


__all__ = [
    "TYPED_UNAVAILABLE_REASON_CACHE_GEOMETRY_UNRESOLVED",
    "TYPED_UNAVAILABLE_REASON_COVERAGE_BELOW_SMOKE_FLOOR",
    "TYPED_UNAVAILABLE_REASON_FRAMES_SEQUENCE_EMPTY",
    "TYPED_UNAVAILABLE_REASON_NO_ADMISSIBLE_ACCESSES",
    "TYPED_UNAVAILABLE_REASON_POLICY_UNCALIBRATED",
    "TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED",
    "TYPED_UNAVAILABLE_REASON_WARMUP_COUNT_OUT_OF_RANGE",
    "compose_predictor_bundle",
    "predict_from_launch_frame",
    "predict_warmed_launch_sequence",
]
