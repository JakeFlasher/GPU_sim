"""Tier-2 cache predictor accuracy diagnostic (calibration scope only).

Reads the static NCU label ledger that lives under this same
`calibration/cache_prediction/` prefix and compares the deployed
Tier-2 predictor's calibrated leave-one-out cross-validation
(LOOCV) held-out output against NCU ground truth across the full
canonical 235-problem SOL-ExecBench partition. Fails closed when
any problem exceeds 10% relative error on either L1 or L2 hit
rate, OR has a non-`measured` status (canonical typed_unavailable,
diagnostic-pipeline failure, or corpus-expansion gap).

The diagnostic is calibration-scoped: it may read the NCU label
ledger as offline ground truth, but it neither imports `ncu` /
`nv-nsight-cu-cli` nor invokes any NCU tooling at runtime. The
deployed predictor path and the B200 surface remain free of
runtime NCU dependencies.

Canonical source of truth for the 235-problem partition is the
markdown directory `references/NVIDIA_Manuals/solbench_problems/
problem_NNN_<slug>.md`. Each markdown filename derives one
canonical problem slug; the slug is then mapped to its benchmark
category (L1, L2, Quant, FIB) via a cross-reference against the
`sol-execbench/data/benchmark/<category>/<NNN_slug>/` asset tree.
The reference markdown directory MUST contain exactly 235 unique
slugs; the benchmark tree MUST resolve every slug to exactly one
category; the per-category counts MUST match `{L1: 94, L2: 82,
Quant: 33, FIB: 26}`.

Public surface:

  * `ACCURACY_DIAGNOSTIC_TYPED_REASONS` — closed frozenset of
    diagnostic-pipeline failure reasons (separate from the
    deployed-scope
    `metrics.typed_unavailable_bundle.PREDICTOR_ABSTENTION_REASONS`).
  * `CORPUS_EXPANSION_TYPED_REASONS` — closed frozenset of
    corpus-expansion gap reasons.
  * `HOST_ARCH_UNSUPPORTED_TYPED_REASONS` — closed frozenset of
    sm_86 hardware-exclusion reasons (NVFP4 / TCGen05 / TMEM /
    sm_90+ TMA / cluster / FP8-e4m3 / per-host compile or runtime
    failure). Disjoint from the other three vocabularies.
  * `STATUS_MEASURED`, `STATUS_TYPED_UNAVAILABLE`,
    `STATUS_DIAGNOSTIC_FAILED`, `STATUS_CORPUS_EXPANSION_PENDING`,
    `STATUS_HOST_ARCH_UNSUPPORTED` — string constants for the
    per-problem row's `status` field.
  * `HostArchAvailabilityEntry`,
    `load_host_arch_availability_manifest(path)`,
    `detect_host_arch_unsupported(problem, manifest)` — loader +
    pure detector hook that consults the optional
    `host_arch_availability_v1.toml` manifest. The detector fires
    BEFORE the cohort lookup in `_build_per_problem_row` so
    hardware-exclusion rows do not get shadowed by upstream
    corpus-expansion prerequisite checks.
  * `CATEGORY_L1`, `CATEGORY_L2`, `CATEGORY_QUANT`,
    `CATEGORY_FIB` — string constants for the per-problem row's
    `problem_category` field (the on-disk `FlashInfer-Bench`
    benchmark directory is mapped to the short label `FIB`).
  * `CANONICAL_CATEGORY_COUNTS` — fixed mapping
    `{L1: 94, L2: 82, Quant: 33, FIB: 26}`. Sum = 235.
  * `NAN_SENTINEL` — the literal string `"nan"`. Non-measured
    rows emit this in their four rel-err fields to keep TOML
    parseable while making the absence of a numeric value
    obvious.
  * `CanonicalSolbenchProblem` — frozen dataclass:
    `problem_slug`, `problem_category`, `reference_markdown`,
    `benchmark_dir`.
  * `enumerate_canonical_solbench_problems(repo_root=...)` —
    walks the reference markdown directory and returns the
    canonical 235-problem list, validated to match
    `CANONICAL_CATEGORY_COUNTS` exactly.
  * `canonical_typed_unavailable_cell_names()` — returns the
    frozenset of cell names (from `PREDICTOR_DESIGN_COVERAGE_BY_CELL`)
    whose `predictor_bucket` is `predictor_typed_unavailable`.
    The schema validator uses this to reject `typed_unavailable`
    rows whose cell_name is not one of the canonical cohort
    entries.
  * `PerProblemRecord` — TypedDict describing the gate input
    row.
  * `PerProblemRecordSchema` — frozen dataclass capturing all
    21 named row fields with type-hinted attributes for
    downstream serializer consumers.
  * `validate_per_problem_record(record)` — pure validator;
    raises `PerProblemRecordSchemaError` with typed reasons.
  * `AccuracyGateError(typed_reason, message)` with typed
    reasons: `invalid_threshold`, `invalid_denominator_floor`,
    `record_missing_field`, `record_invalid_status`,
    `record_invalid_rel_err`, `record_set_incomplete`,
    `record_duplicate_slug`, `record_unknown_slug`,
    `canonical_partition_drift`.
  * `PerProblemRecordSchemaError(typed_reason, message)` with
    typed reasons enumerated in the class docstring.
  * `AccuracyDiagnosticResult` — frozen dataclass: gate output.
  * `compute_accuracy_diagnostic_gate(records, *, threshold=0.10,
    eps=0.01, canonical_problems=None)` — pure function. First
    validates that `records` covers exactly the canonical 235
    slug set (no duplicates, no unknown slugs, no missing slugs).
    Then fails closed if any record has non-`measured` status
    OR any measured record exceeds the threshold on calibrated
    L1 or L2 relative error.

Importing this module does NOT invoke NCU, CUPTI, or any GPU
runtime. It is pure data + pure functions with stdlib-only
dependencies plus pure-value imports of
`PREDICTOR_ABSTENTION_REASONS` (for cross-vocabulary boundary
checking) and `PREDICTOR_DESIGN_COVERAGE_BY_CELL` (for
canonical typed-unavailable cohort enforcement).
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from sol_baselines_harness.metrics.predictor_design_coverage import (
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    PREDICTOR_DESIGN_COVERAGE_BY_CELL,
    partition_size,
)
from sol_baselines_harness.metrics.typed_unavailable_bundle import (
    PREDICTOR_ABSTENTION_REASONS,
)
from sol_baselines_harness.calibration.cache_prediction.calibration_validation_split import (
    CalibrationValidationSplit,
    CalibrationValidationSplitError,
    load_calibration_validation_split_manifest,
)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

STATUS_MEASURED: str = "measured"
STATUS_TYPED_UNAVAILABLE: str = "typed_unavailable"
STATUS_DIAGNOSTIC_FAILED: str = "diagnostic_failed"
STATUS_CORPUS_EXPANSION_PENDING: str = "corpus_expansion_pending"
STATUS_HOST_ARCH_UNSUPPORTED: str = "host_arch_unsupported"

_VALID_STATUSES: frozenset[str] = frozenset({
    STATUS_MEASURED,
    STATUS_TYPED_UNAVAILABLE,
    STATUS_DIAGNOSTIC_FAILED,
    STATUS_CORPUS_EXPANSION_PENDING,
    STATUS_HOST_ARCH_UNSUPPORTED,
})


# ---------------------------------------------------------------------------
# Category enum + canonical counts
# ---------------------------------------------------------------------------

CATEGORY_L1: str = "L1"
CATEGORY_L2: str = "L2"
CATEGORY_QUANT: str = "Quant"
CATEGORY_FIB: str = "FIB"

# The on-disk `sol-execbench/data/benchmark/` directory uses
# `FlashInfer-Bench`; the diagnostic's external vocabulary uses
# the short label `FIB`.
_BENCHMARK_DIR_TO_CATEGORY: dict[str, str] = {
    "L1": CATEGORY_L1,
    "L2": CATEGORY_L2,
    "Quant": CATEGORY_QUANT,
    "FlashInfer-Bench": CATEGORY_FIB,
}

CANONICAL_CATEGORY_COUNTS: dict[str, int] = {
    CATEGORY_L1: 94,
    CATEGORY_L2: 82,
    CATEGORY_QUANT: 33,
    CATEGORY_FIB: 26,
}


# Sentinel emitted in TOML for the four rel-err fields when the
# row's `status` is not `measured`. Stays a parseable string so
# `tomllib.loads()` accepts it; signals "no numeric rel-err here"
# to downstream report consumers.
NAN_SENTINEL: str = "nan"


# The user-acceptance threshold (DEC-1, uniform across friendly +
# hostile measured cells). The plan's user-acceptance overlay
# requires every measured cell to land below this on BOTH L1 and
# L2 calibrated rel-err; failure flags the row's
# `pass_user_target_threshold` False and prevents the compound
# gate from passing for that row.
USER_TARGET_THRESHOLD_PCT: float = 10.0
_USER_TARGET_THRESHOLD_FRACTION: float = USER_TARGET_THRESHOLD_PCT / 100.0


# Contractual tiered gate thresholds (CLAUDE.md operator-set
# 2026-05-16). The friendly cohort lives under tighter rel-err
# constraints (well-behaved kernels expected to be predicted
# accurately), the hostile cohort lives under a looser bound
# (carveout-sensitive / persisting-window / etc.). The user-
# acceptance gate is independent and uniform per DEC-1.
_TIERED_THRESHOLD_PREDICTOR_FRIENDLY_PCT: float = 5.0
_TIERED_THRESHOLD_PREDICTOR_HOSTILE_PCT: float = 15.0


# ---------------------------------------------------------------------------
# Diagnostic-failure typed-reason vocabulary
# ---------------------------------------------------------------------------

# Pipeline-level diagnostic failures. Separate from the closed
# deployed-scope `PREDICTOR_ABSTENTION_REASONS` frozenset.
#
# `ncu_label_no_read_only_traffic` routes admitted sm_86 kernels
# whose ledger carries a strict-loader-valid `[[unavailable]]`
# row with that same reason to `status = diagnostic_failed`, not
# `corpus_expansion_pending / ncu_labels_missing`. NCU's read-
# only sector hit-rate metrics are undefined for kernels with no
# read traffic; this is a terminal calibration outcome (not a
# corpus-pending prerequisite gap) and the diagnostic must
# distinguish the two.
ACCURACY_DIAGNOSTIC_TYPED_REASONS: frozenset[str] = frozenset({
    "ledger_entry_missing",
    "join_key_drift",
    "sidecar_parse_failed",
    "launch_index_mismatch",
    "predictor_typed_unavailable_non_canonical",
    "ncu_label_no_read_only_traffic",
})


# Corpus-expansion gaps: problems in the canonical 235-problem
# list whose prerequisite assets are not yet in place.
CORPUS_EXPANSION_TYPED_REASONS: frozenset[str] = frozenset({
    "cu_hand_port_missing",
    "ncu_labels_missing",
    "cohort_classification_missing",
})


# Hardware exclusions: problems whose kernels require sm_100a /
# sm_90+ / sm_89+ features the local sm_86 host does not provide.
# Such problems are carved out of the <10% rel-err denominator
# (they are exempt from the numerator AND denominator) so the
# user-acceptance gate can converge on what the sm_86 host can
# actually measure. The closed vocabulary mirrors the static
# classification dimensions visible in the canonical problem
# specs under `references/NVIDIA_Manuals/solbench_problems/`:
# dtype tier (NVFP4 / FP8 e4m3), tensor-core variant (TCGen05 /
# TMEM), SM arch-conditional features (TMA / cluster), plus two
# runtime-only failure modes for the trial-compile sweep step.
HOST_ARCH_UNSUPPORTED_TYPED_REASONS: frozenset[str] = frozenset({
    "requires_nvfp4_sm100",
    "requires_fp8_e4m3_sm89_plus",
    "requires_tcgen05_sm100",
    "requires_tcgen05_sm100a",
    "requires_tmem_sm100",
    "requires_tma_sm90_plus",
    "requires_cluster_sm90_plus",
    "compile_failure_sm86",
    "runtime_failure_sm86",
})


# Typed reasons that may appear in the per-row sm_120 cohort
# assignment when ``sm120_status == sm120_compat_unverified_cohort``.
# Disjoint from ``HOST_ARCH_UNSUPPORTED_TYPED_REASONS`` so each
# vocabulary stays single-purpose.
SM_120_COHORT_UNVERIFIED_TYPED_REASONS: frozenset[str] = frozenset({
    "nvfp4_sm120_compat_unverified",
})


# Fail-fast safety net: the new diagnostic vocabularies must not
# overlap with the closed `PREDICTOR_ABSTENTION_REASONS` set.
assert ACCURACY_DIAGNOSTIC_TYPED_REASONS.isdisjoint(
    PREDICTOR_ABSTENTION_REASONS
), "ACCURACY_DIAGNOSTIC_TYPED_REASONS must not share members with PREDICTOR_ABSTENTION_REASONS"
assert CORPUS_EXPANSION_TYPED_REASONS.isdisjoint(
    PREDICTOR_ABSTENTION_REASONS
), "CORPUS_EXPANSION_TYPED_REASONS must not share members with PREDICTOR_ABSTENTION_REASONS"
assert ACCURACY_DIAGNOSTIC_TYPED_REASONS.isdisjoint(
    CORPUS_EXPANSION_TYPED_REASONS
), "ACCURACY_DIAGNOSTIC_TYPED_REASONS must not share members with CORPUS_EXPANSION_TYPED_REASONS"
assert HOST_ARCH_UNSUPPORTED_TYPED_REASONS.isdisjoint(
    PREDICTOR_ABSTENTION_REASONS
), "HOST_ARCH_UNSUPPORTED_TYPED_REASONS must not share members with PREDICTOR_ABSTENTION_REASONS"
assert HOST_ARCH_UNSUPPORTED_TYPED_REASONS.isdisjoint(
    ACCURACY_DIAGNOSTIC_TYPED_REASONS
), "HOST_ARCH_UNSUPPORTED_TYPED_REASONS must not share members with ACCURACY_DIAGNOSTIC_TYPED_REASONS"
assert SM_120_COHORT_UNVERIFIED_TYPED_REASONS.isdisjoint(
    HOST_ARCH_UNSUPPORTED_TYPED_REASONS
), "SM_120_COHORT_UNVERIFIED_TYPED_REASONS must not overlap HOST_ARCH_UNSUPPORTED_TYPED_REASONS"
assert SM_120_COHORT_UNVERIFIED_TYPED_REASONS.isdisjoint(
    PREDICTOR_ABSTENTION_REASONS
), "SM_120_COHORT_UNVERIFIED_TYPED_REASONS must not overlap PREDICTOR_ABSTENTION_REASONS"
assert HOST_ARCH_UNSUPPORTED_TYPED_REASONS.isdisjoint(
    CORPUS_EXPANSION_TYPED_REASONS
), "HOST_ARCH_UNSUPPORTED_TYPED_REASONS must not share members with CORPUS_EXPANSION_TYPED_REASONS"


# ---------------------------------------------------------------------------
# Canonical typed-unavailable cohort lookup
# ---------------------------------------------------------------------------

def canonical_typed_unavailable_cell_names() -> frozenset[str]:
    """Return the cell names whose canonical bucket is
    `predictor_typed_unavailable`.

    The 5-cell cohort is part of `PREDICTOR_DESIGN_COVERAGE_BY_CELL`.
    The schema validator uses this set to reject any
    `status=typed_unavailable` row whose `cell_name` is not one of
    the 5 canonical entries (silent reclassification of admitted
    cells to typed_unavailable is forbidden).
    """

    return frozenset(
        entry.cell_name
        for entry in PREDICTOR_DESIGN_COVERAGE_BY_CELL.values()
        if entry.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE
    )


# ---------------------------------------------------------------------------
# Canonical problem enumeration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalSolbenchProblem:
    """One entry in the canonical 235-problem partition.

    `problem_slug` is the diagnostic's external slug, taking the
    form `problem_<NNN>_<slug>` where the prefix and suffix are
    derived from the markdown filename
    `references/NVIDIA_Manuals/solbench_problems/
    problem_<NNN>_<slug>.md`. `problem_category` is resolved by
    matching the slug's `<NNN>_<slug>` body against subdirectory
    names under `sol-execbench/data/benchmark/<category>/`.
    `reference_markdown` is the absolute path to the source-of-
    truth markdown file. `benchmark_dir` is the absolute path to
    the matching benchmark asset directory.
    """

    problem_slug: str
    problem_category: str
    reference_markdown: Path
    benchmark_dir: Path


_REFERENCE_FILENAME_RE = re.compile(
    r"^problem_(?P<body>\d{3}_[A-Za-z0-9_]+)\.md$"
)


def _resolve_benchmark_dir(
    *,
    slug_body: str,
    problem_slug: str,
    benchmark_lookup: Mapping[str, tuple[str, Path]],
    already_consumed: set[str],
) -> tuple[str, Path]:
    """Map a reference-markdown slug body to its benchmark
    category + directory.

    Strategy:
      1. Prefer an exact match (`slug_body` is the benchmark
         directory name).
      2. Otherwise look for benchmark directories whose name
         starts with `<slug_body>` followed by a boundary
         character (`-`, `.`, or `_`). This handles the small
         number of reference markdown filenames whose slug is a
         truncated prefix of the longer benchmark directory name.
      3. Skip benchmark directories that have already been
         consumed by a previous reference slug; this prevents two
         truncated reference filenames from racing for the same
         benchmark dir.
      4. If exactly one un-consumed candidate remains, use it.
         Otherwise fail closed with a `canonical_partition_drift`
         error.
    """

    if slug_body in benchmark_lookup and slug_body not in already_consumed:
        return benchmark_lookup[slug_body]

    candidates: list[tuple[str, Path]] = []
    for candidate_body, (category, path) in benchmark_lookup.items():
        if candidate_body in already_consumed:
            continue
        # The boundary-character requirement avoids partial-word
        # collisions: `001_attention` would otherwise match
        # `001_attention_softmax_*` as a prefix.
        if candidate_body == slug_body:
            candidates.append((category, path))
            continue
        if not candidate_body.startswith(slug_body):
            continue
        following = candidate_body[len(slug_body):]
        if following and following[0] in ("-", ".", "_"):
            candidates.append((category, path))

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise AccuracyGateError(
            "canonical_partition_drift",
            f"reference slug {problem_slug!r} has no matching "
            f"benchmark directory candidate",
        )
    raise AccuracyGateError(
        "canonical_partition_drift",
        f"reference slug {problem_slug!r} matches multiple "
        f"benchmark directory candidates: "
        f"{sorted(str(p) for _, p in candidates)!r}",
    )


def enumerate_canonical_solbench_problems(
    repo_root: Path | None = None,
) -> list[CanonicalSolbenchProblem]:
    """Return the canonical 235-problem partition.

    Reads the markdown reference directory
    `<repo_root>/references/NVIDIA_Manuals/solbench_problems/`
    as the source of truth, parses each `problem_<NNN>_<slug>.md`
    filename, then maps each slug to its benchmark category by
    cross-referencing
    `<repo_root>/sol-execbench/data/benchmark/<category>/<NNN_slug>/`.

    Validates that:
      - the reference directory contains 235 unique markdown
        filenames matching the `problem_<NNN>_<slug>.md` pattern;
      - every parsed slug resolves to exactly one benchmark
        category directory entry;
      - the per-category counts match `CANONICAL_CATEGORY_COUNTS`
        exactly (`{L1: 94, L2: 82, Quant: 33, FIB: 26}`).

    Raises `AccuracyGateError("canonical_partition_drift", ...)`
    on any drift (missing reference directory, duplicate slug,
    orphan slug, count drift, malformed filename).

    Returns the list sorted by `(problem_category, problem_slug)`
    for determinism.
    """

    if repo_root is None:
        # Walk up from this module to the repo root:
        # <repo>/tools/sol_baselines_harness/calibration/
        # cache_prediction/accuracy_diagnostic.py.
        repo_root = Path(__file__).resolve().parents[4]

    reference_root = (
        repo_root / "references" / "NVIDIA_Manuals" / "solbench_problems"
    )
    if not reference_root.is_dir():
        raise AccuracyGateError(
            "canonical_partition_drift",
            f"reference solbench-problems directory not found at "
            f"{reference_root!s}",
        )

    benchmark_root = repo_root / "sol-execbench" / "data" / "benchmark"
    if not benchmark_root.is_dir():
        raise AccuracyGateError(
            "canonical_partition_drift",
            f"sol-execbench benchmark root not found at {benchmark_root!s}",
        )

    # Build a lookup `<NNN_slug>` -> (category, benchmark_dir) by
    # walking the four canonical category directories. Each
    # benchmark dir entry is the directory name; slug bodies must
    # be unique across all four categories (one benchmark entry
    # per problem).
    benchmark_lookup: dict[str, tuple[str, Path]] = {}
    for benchmark_dir_name, category_label in _BENCHMARK_DIR_TO_CATEGORY.items():
        category_dir = benchmark_root / benchmark_dir_name
        if not category_dir.is_dir():
            raise AccuracyGateError(
                "canonical_partition_drift",
                f"benchmark category dir not found: {category_dir!s}",
            )
        for entry in sorted(category_dir.iterdir()):
            if not entry.is_dir():
                continue
            slug_body = entry.name
            if slug_body in benchmark_lookup:
                prior_category, prior_dir = benchmark_lookup[slug_body]
                raise AccuracyGateError(
                    "canonical_partition_drift",
                    f"benchmark slug body {slug_body!r} appears in "
                    f"both {prior_category!r} ({prior_dir!s}) and "
                    f"{category_label!r} ({entry!s})",
                )
            benchmark_lookup[slug_body] = (category_label, entry)

    # Now read the markdown reference directory as the canonical
    # input list.
    problems: list[CanonicalSolbenchProblem] = []
    seen_slugs: set[str] = set()
    consumed_benchmark_slug_bodies: set[str] = set()
    observed_counts: dict[str, int] = {
        CATEGORY_L1: 0,
        CATEGORY_L2: 0,
        CATEGORY_QUANT: 0,
        CATEGORY_FIB: 0,
    }

    for ref_path in sorted(reference_root.iterdir()):
        if not ref_path.is_file() or ref_path.suffix != ".md":
            continue
        match = _REFERENCE_FILENAME_RE.match(ref_path.name)
        if match is None:
            raise AccuracyGateError(
                "canonical_partition_drift",
                f"reference markdown filename {ref_path.name!r} does "
                f"not match the expected pattern "
                f"problem_<NNN>_<slug>.md",
            )
        slug_body = match.group("body")
        problem_slug = f"problem_{slug_body}"

        if problem_slug in seen_slugs:
            raise AccuracyGateError(
                "canonical_partition_drift",
                f"duplicate reference slug {problem_slug!r}",
            )
        seen_slugs.add(problem_slug)

        category_label, benchmark_dir = _resolve_benchmark_dir(
            slug_body=slug_body,
            problem_slug=problem_slug,
            benchmark_lookup=benchmark_lookup,
            already_consumed=consumed_benchmark_slug_bodies,
        )
        consumed_benchmark_slug_bodies.add(benchmark_dir.name)

        problems.append(
            CanonicalSolbenchProblem(
                problem_slug=problem_slug,
                problem_category=category_label,
                reference_markdown=ref_path,
                benchmark_dir=benchmark_dir,
            )
        )
        observed_counts[category_label] += 1

    # Fail-closed on per-category drift before returning.
    for category, expected in CANONICAL_CATEGORY_COUNTS.items():
        observed = observed_counts[category]
        if observed != expected:
            raise AccuracyGateError(
                "canonical_partition_drift",
                f"category {category}: observed {observed}, expected {expected}",
            )

    total = sum(observed_counts.values())
    if total != 235:
        raise AccuracyGateError(
            "canonical_partition_drift",
            f"total observed reference problem count {total} != 235",
        )

    problems.sort(key=lambda p: (p.problem_category, p.problem_slug))
    return problems


# ---------------------------------------------------------------------------
# Per-problem record TypedDict + schema dataclass
# ---------------------------------------------------------------------------

class PerProblemRecord(TypedDict, total=False):
    """The gate-input row shape. All 24 named fields are listed
    here for type-checker discovery; downstream serializers
    consume `PerProblemRecordSchema` (below) for full
    validation.
    """

    problem_slug: str
    problem_category: str
    cell_name: str
    predictor_bucket: str
    ncu_l1_label: float
    ncu_l2_label: float
    predicted_l1_raw: float
    predicted_l2_raw: float
    predicted_l1_calibrated: float
    predicted_l2_calibrated: float
    l1_abs_pp_error: float
    l2_abs_pp_error: float
    l1_calibrated_abs_pp_error: float
    l2_calibrated_abs_pp_error: float
    l1_raw_rel_err: float
    l2_raw_rel_err: float
    l1_calibrated_rel_err: float
    l2_calibrated_rel_err: float
    denominator_floor: float
    status: str
    typed_reason: str
    pass_user_target_threshold: bool
    pass_tiered_gate: bool
    effective_threshold_pct: float


# The required-key list for the per-row schema validator. The
# four rel-err fields and the three new verdict fields appear as
# `float | str` / `bool` in serialized TOML (numeric for measured
# rows; `NAN_SENTINEL` string or False for non-measured rows).
# Numeric error fields (`*_abs_pp_error`) are only meaningful for
# measured rows but the schema requires the keys to be present in
# every serialized row.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "problem_slug",
    "problem_category",
    "cell_name",
    "predictor_bucket",
    "ncu_l1_label",
    "ncu_l2_label",
    "predicted_l1_raw",
    "predicted_l2_raw",
    "predicted_l1_calibrated",
    "predicted_l2_calibrated",
    "l1_abs_pp_error",
    "l2_abs_pp_error",
    "l1_calibrated_abs_pp_error",
    "l2_calibrated_abs_pp_error",
    "l1_raw_rel_err",
    "l2_raw_rel_err",
    "l1_calibrated_rel_err",
    "l2_calibrated_rel_err",
    "denominator_floor",
    "status",
    "typed_reason",
    "pass_user_target_threshold",
    "pass_tiered_gate",
    "effective_threshold_pct",
)
assert len(_REQUIRED_FIELDS) == 24


_REL_ERR_FIELDS: tuple[str, ...] = (
    "l1_raw_rel_err",
    "l2_raw_rel_err",
    "l1_calibrated_rel_err",
    "l2_calibrated_rel_err",
)


@dataclass(frozen=True)
class PerProblemRecordSchema:
    """All 24 base fields of a serialized per-problem row, plus
    an optional `audit_warning` tuple carrying gap-field names
    when the denominator audit found L1/L2 sector-total drift
    above the 5% threshold.

    Rel-err fields are `float | str`: numeric for `measured`
    rows, the literal string `"nan"` (NAN_SENTINEL) for
    non-measured rows. Numeric error fields (`*_abs_pp_error`,
    `*_label`, `predicted_*`) are similarly `float | str` to
    accept the same sentinel; report consumers know to treat
    `NAN_SENTINEL` as "no numeric value here".

    The three verdict fields `pass_user_target_threshold`,
    `pass_tiered_gate`, and `effective_threshold_pct` carry the
    per-row outcome of both gates (user-acceptance <10% on
    calibrated L1+L2 rel-err, AND the contractual tiered gate
    against the cohort's threshold). Booleans only — non-measured
    rows are False on both verdicts and `NAN_SENTINEL` on the
    threshold. The denominator_box aggregates these per-row
    booleans into the compound gate count.

    `audit_warning` is an empty tuple when no audit warning was
    raised. The serializer emits the field when non-empty so
    operators can find audit-failing cells in the per-cell rows
    without joining against the audit table.

    `validate_per_problem_record()` is the canonical entry point
    for constructing one of these from a dict-shaped row.
    """

    problem_slug: str
    problem_category: str
    cell_name: str
    predictor_bucket: str
    ncu_l1_label: float | str
    ncu_l2_label: float | str
    predicted_l1_raw: float | str
    predicted_l2_raw: float | str
    predicted_l1_calibrated: float | str
    predicted_l2_calibrated: float | str
    l1_abs_pp_error: float | str
    l2_abs_pp_error: float | str
    l1_calibrated_abs_pp_error: float | str
    l2_calibrated_abs_pp_error: float | str
    l1_raw_rel_err: float | str
    l2_raw_rel_err: float | str
    l1_calibrated_rel_err: float | str
    l2_calibrated_rel_err: float | str
    denominator_floor: float
    status: str
    typed_reason: str
    pass_user_target_threshold: bool
    pass_tiered_gate: bool
    effective_threshold_pct: float | str
    audit_warning: tuple[str, ...] = ()


class PerProblemRecordSchemaError(Exception):
    """Typed exception raised by `validate_per_problem_record(...)`.

    Typed reasons:
      * `missing_field` — a required key is absent.
      * `invalid_status` — `status` is not in the declared
        four-value enum.
      * `invalid_category` — `problem_category` is not in
        `CANONICAL_CATEGORY_COUNTS`.
      * `invalid_typed_reason_for_measured` — `status=measured`
        row has non-empty `typed_reason`.
      * `invalid_typed_reason_vocabulary` — `status=...` row
        carries a `typed_reason` value not in the vocabulary
        expected for its status.
      * `typed_unavailable_non_canonical_cell` — `status=
        typed_unavailable` row whose `cell_name` is not one of
        the 5 canonical typed-unavailable cohort cells.
      * `invalid_rel_err_value` — a rel-err field is neither a
        finite float nor the literal `NAN_SENTINEL`.
      * `invalid_denominator_floor` — `denominator_floor` is
        non-positive or non-finite.
      * `non_measured_rel_err_must_be_sentinel` — non-measured
        row has a numeric rel-err where `NAN_SENTINEL` is
        required.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _is_finite_float(value: Any) -> bool:
    """Return True only for finite real numbers (NOT bool, NaN,
    or inf)."""

    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _rel_err_field_is_valid(value: Any) -> bool:
    """A rel-err field value is valid if it is either a finite
    real number OR the literal `NAN_SENTINEL` string.
    """

    if value == NAN_SENTINEL:
        return True
    return _is_finite_float(value)


# Tolerance for the arithmetic-consistency check on measured
# rows. Absolute tolerance; the stored rel-err and abs-pp-error
# fields are read from a TOML round-trip so float64 truncation
# is the dominant error source, well under 1e-6 in absolute
# terms for the units we use (rel-err in [0, 1], abs-pp-error
# in [0, 100]).
_ARITHMETIC_TOLERANCE: float = 1e-6


_MEASURED_NUMERIC_FIELDS: tuple[str, ...] = (
    "ncu_l1_label",
    "ncu_l2_label",
    "predicted_l1_raw",
    "predicted_l2_raw",
    "predicted_l1_calibrated",
    "predicted_l2_calibrated",
    "l1_abs_pp_error",
    "l2_abs_pp_error",
    "l1_calibrated_abs_pp_error",
    "l2_calibrated_abs_pp_error",
)

# Subset of `_MEASURED_NUMERIC_FIELDS` that must be hit rates
# in the closed interval [0, 1]. The abs-pp-error fields are
# excluded because they are percentage points whose legal range
# is [0, 100] and they are bound by the arithmetic-consistency
# check downstream.
_MEASURED_HIT_RATE_FIELDS: tuple[str, ...] = (
    "ncu_l1_label",
    "ncu_l2_label",
    "predicted_l1_raw",
    "predicted_l2_raw",
    "predicted_l1_calibrated",
    "predicted_l2_calibrated",
)


# (stored_abs_pp_error_field, stored_rel_err_field, prediction_field, label_field)
# Each tuple drives the recomputation of one (abs-pp-error, rel-err)
# pair from its underlying prediction + label.
_ARITHMETIC_TRIPLES: tuple[tuple[str, str, str, str], ...] = (
    ("l1_abs_pp_error", "l1_raw_rel_err", "predicted_l1_raw", "ncu_l1_label"),
    ("l2_abs_pp_error", "l2_raw_rel_err", "predicted_l2_raw", "ncu_l2_label"),
    (
        "l1_calibrated_abs_pp_error",
        "l1_calibrated_rel_err",
        "predicted_l1_calibrated",
        "ncu_l1_label",
    ),
    (
        "l2_calibrated_abs_pp_error",
        "l2_calibrated_rel_err",
        "predicted_l2_calibrated",
        "ncu_l2_label",
    ),
)


def _validate_measured_row_arithmetic(
    record: Mapping[str, Any], denominator_floor: float
) -> None:
    """Enforce that every measured row's stored rel-err and
    abs-pp-error fields match what the formulas produce from
    the labels + predictions.

    Recomputes each (abs-pp-error, rel-err) pair as:
      - abs_pp_error = abs(predicted - label) * 100.0
      - rel_err     = abs(predicted - label) / max(label, eps)

    where `eps = denominator_floor` is the row's own stored
    floor. Raises `PerProblemRecordSchemaError` with typed
    reasons `abs_pp_error_arithmetic_inconsistent` or
    `rel_err_arithmetic_inconsistent` on the first mismatch.

    All ten numeric fields (labels + predictions + abs-pp-errors)
    must be finite real numbers; non-finite values raise
    `invalid_measured_numeric_field`.
    """

    for field in _MEASURED_NUMERIC_FIELDS:
        value = record[field]
        if not _is_finite_float(value):
            raise PerProblemRecordSchemaError(
                "invalid_measured_numeric_field",
                f"measured row field {field!r} must be a finite real "
                f"number, got {value!r}",
            )

    # Hit-rate fields are fractions in [0, 1]. The abs-pp-error
    # fields are percentage points (legally in [0, 100]) and
    # are already covered by the arithmetic-consistency check
    # below; only the 6 hit-rate-shaped fields need the range
    # guard.
    for hit_rate_field in _MEASURED_HIT_RATE_FIELDS:
        value = float(record[hit_rate_field])
        if not (0.0 <= value <= 1.0):
            raise PerProblemRecordSchemaError(
                "invalid_measured_hit_rate",
                f"measured row field {hit_rate_field!r} must be a hit "
                f"rate in [0.0, 1.0]; got {value!r}",
            )

    for abs_pp_field, rel_err_field, pred_field, label_field in _ARITHMETIC_TRIPLES:
        prediction = float(record[pred_field])
        label = float(record[label_field])
        delta = abs(prediction - label)
        expected_abs_pp = delta * 100.0
        expected_rel_err = delta / max(label, denominator_floor)

        stored_abs_pp = float(record[abs_pp_field])
        if abs(stored_abs_pp - expected_abs_pp) > _ARITHMETIC_TOLERANCE:
            raise PerProblemRecordSchemaError(
                "abs_pp_error_arithmetic_inconsistent",
                f"stored {abs_pp_field}={stored_abs_pp!r} disagrees "
                f"with formula abs({pred_field}={prediction!r} - "
                f"{label_field}={label!r}) * 100 = "
                f"{expected_abs_pp!r} by more than "
                f"{_ARITHMETIC_TOLERANCE!r}",
            )

        stored_rel_err = float(record[rel_err_field])
        if abs(stored_rel_err - expected_rel_err) > _ARITHMETIC_TOLERANCE:
            raise PerProblemRecordSchemaError(
                "rel_err_arithmetic_inconsistent",
                f"stored {rel_err_field}={stored_rel_err!r} disagrees "
                f"with formula abs({pred_field}={prediction!r} - "
                f"{label_field}={label!r}) / max({label_field}, "
                f"denominator_floor={denominator_floor!r}) = "
                f"{expected_rel_err!r} by more than "
                f"{_ARITHMETIC_TOLERANCE!r}",
            )


def _validate_non_measured_row_numeric_sentinels(
    record: Mapping[str, Any],
) -> None:
    """Enforce that every numeric field on a non-measured row
    carries the literal `NAN_SENTINEL` string. The four rel-err
    fields are already checked separately; this function covers
    the labels, predictions, and abs-pp-error fields.

    Raises `PerProblemRecordSchemaError` with typed reason
    `non_measured_numeric_field_must_be_sentinel`.
    """

    for field in _MEASURED_NUMERIC_FIELDS:
        value = record[field]
        if value != NAN_SENTINEL:
            raise PerProblemRecordSchemaError(
                "non_measured_numeric_field_must_be_sentinel",
                f"non-measured row field {field!r} must equal "
                f"NAN_SENTINEL ({NAN_SENTINEL!r}), got {value!r}",
            )


def compute_user_target_verdict_fields(
    *,
    status: str,
    predictor_bucket: str,
    l1_calibrated_rel_err: float | str,
    l2_calibrated_rel_err: float | str,
) -> dict[str, Any]:
    """Return the three verdict fields for a per-problem row:
    `pass_user_target_threshold`, `pass_tiered_gate`,
    `effective_threshold_pct`.

    For `status == STATUS_MEASURED`:
      * `pass_user_target_threshold = (l1 < 0.10 AND l2 < 0.10)`
        — the uniform user-acceptance gate from DEC-1.
      * `pass_tiered_gate = (l1 < t/100 AND l2 < t/100)` where
        `t` is 5.0 for predictor_friendly and 15.0 for
        predictor_hostile.
      * `effective_threshold_pct` is the cohort's contractual
        threshold (`t` above).
      * `predictor_bucket` MUST be one of friendly/hostile for
        measured rows; anything else raises
        `measured_row_invalid_predictor_bucket_for_tiered_gate`.

    For non-measured rows (every status other than measured):
      * Both verdicts return False (typed-unavailable and
        diagnostic-failed are fail-closed; host_arch_unsupported
        is exempt and aggregated separately in the
        `[denominator_box]` source-of-truth; corpus-pending has
        not produced a measurement yet).
      * `effective_threshold_pct` returns `NAN_SENTINEL`.
    """

    if status == STATUS_MEASURED:
        if not _is_finite_float(l1_calibrated_rel_err):
            raise PerProblemRecordSchemaError(
                "invalid_rel_err_value",
                f"measured row l1_calibrated_rel_err must be a "
                f"finite real to compute verdict fields; got "
                f"{l1_calibrated_rel_err!r}",
            )
        if not _is_finite_float(l2_calibrated_rel_err):
            raise PerProblemRecordSchemaError(
                "invalid_rel_err_value",
                f"measured row l2_calibrated_rel_err must be a "
                f"finite real to compute verdict fields; got "
                f"{l2_calibrated_rel_err!r}",
            )
        l1 = float(l1_calibrated_rel_err)
        l2 = float(l2_calibrated_rel_err)
        pass_user = bool(
            l1 < _USER_TARGET_THRESHOLD_FRACTION
            and l2 < _USER_TARGET_THRESHOLD_FRACTION
        )
        if predictor_bucket == _PREDICTOR_BUCKET_FRIENDLY:
            threshold_pct = _TIERED_THRESHOLD_PREDICTOR_FRIENDLY_PCT
        elif predictor_bucket == _PREDICTOR_BUCKET_HOSTILE:
            threshold_pct = _TIERED_THRESHOLD_PREDICTOR_HOSTILE_PCT
        else:
            raise PerProblemRecordSchemaError(
                "measured_row_invalid_predictor_bucket_for_tiered_gate",
                f"measured row predictor_bucket "
                f"{predictor_bucket!r} is not one of "
                f"{{predictor_friendly, predictor_hostile}}; "
                f"cannot derive the contractual tiered threshold.",
            )
        threshold_fraction = threshold_pct / 100.0
        pass_tiered = bool(
            l1 < threshold_fraction and l2 < threshold_fraction
        )
        return {
            "pass_user_target_threshold": pass_user,
            "pass_tiered_gate": pass_tiered,
            "effective_threshold_pct": threshold_pct,
        }
    return {
        "pass_user_target_threshold": False,
        "pass_tiered_gate": False,
        "effective_threshold_pct": NAN_SENTINEL,
    }


_VERDICT_FIELDS: tuple[str, ...] = (
    "pass_user_target_threshold",
    "pass_tiered_gate",
    "effective_threshold_pct",
)


def _validate_per_row_verdict_fields(
    record: Mapping[str, Any],
) -> None:
    """Enforce status-conditional semantics on the three verdict
    fields. Computed expected values must match stored values to
    a bit-exact level for the booleans and exactly for the
    threshold-pct sentinel; mismatches raise
    `verdict_field_mismatch` (per-field typed reason
    enumeration in the message).

    For non-measured rows, the booleans MUST be False and the
    threshold MUST be `NAN_SENTINEL`. For measured rows, the
    booleans MUST match the recomputed value from the stored
    rel-err fields, and the threshold MUST equal 5.0 (friendly)
    or 15.0 (hostile) per the cohort.
    """

    for field in _VERDICT_FIELDS:
        if field not in record:
            raise PerProblemRecordSchemaError(
                "missing_field",
                f"per-problem row missing required field {field!r}",
            )

    pass_user = record["pass_user_target_threshold"]
    pass_tiered = record["pass_tiered_gate"]
    threshold_pct = record["effective_threshold_pct"]

    # Type-check booleans before recomputation; TOML round-trip
    # of `True` / `False` produces real bools; anything else is a
    # schema violation.
    if not isinstance(pass_user, bool):
        raise PerProblemRecordSchemaError(
            "verdict_field_type",
            f"pass_user_target_threshold must be bool, got "
            f"{pass_user!r} ({type(pass_user).__name__})",
        )
    if not isinstance(pass_tiered, bool):
        raise PerProblemRecordSchemaError(
            "verdict_field_type",
            f"pass_tiered_gate must be bool, got "
            f"{pass_tiered!r} ({type(pass_tiered).__name__})",
        )

    expected = compute_user_target_verdict_fields(
        status=record["status"],
        predictor_bucket=record.get("predictor_bucket", ""),
        l1_calibrated_rel_err=record["l1_calibrated_rel_err"],
        l2_calibrated_rel_err=record["l2_calibrated_rel_err"],
    )

    if pass_user != expected["pass_user_target_threshold"]:
        raise PerProblemRecordSchemaError(
            "verdict_field_mismatch",
            f"pass_user_target_threshold stored as {pass_user!r} "
            f"but recomputed as "
            f"{expected['pass_user_target_threshold']!r} "
            f"(status={record['status']!r}, "
            f"l1_calibrated_rel_err="
            f"{record['l1_calibrated_rel_err']!r}, "
            f"l2_calibrated_rel_err="
            f"{record['l2_calibrated_rel_err']!r})",
        )
    if pass_tiered != expected["pass_tiered_gate"]:
        raise PerProblemRecordSchemaError(
            "verdict_field_mismatch",
            f"pass_tiered_gate stored as {pass_tiered!r} but "
            f"recomputed as {expected['pass_tiered_gate']!r} "
            f"(status={record['status']!r}, predictor_bucket="
            f"{record.get('predictor_bucket', '')!r}, "
            f"l1_calibrated_rel_err="
            f"{record['l1_calibrated_rel_err']!r}, "
            f"l2_calibrated_rel_err="
            f"{record['l2_calibrated_rel_err']!r})",
        )
    expected_threshold = expected["effective_threshold_pct"]
    if expected_threshold == NAN_SENTINEL:
        if threshold_pct != NAN_SENTINEL:
            raise PerProblemRecordSchemaError(
                "verdict_field_mismatch",
                f"non-measured row effective_threshold_pct must "
                f"equal NAN_SENTINEL ({NAN_SENTINEL!r}), got "
                f"{threshold_pct!r}",
            )
    else:
        if (
            isinstance(threshold_pct, bool)
            or not isinstance(threshold_pct, (int, float))
            or float(threshold_pct) != float(expected_threshold)
        ):
            raise PerProblemRecordSchemaError(
                "verdict_field_mismatch",
                f"measured row effective_threshold_pct stored as "
                f"{threshold_pct!r} but recomputed as "
                f"{expected_threshold!r} for predictor_bucket="
                f"{record.get('predictor_bucket', '')!r}",
            )


def validate_per_problem_record(
    record: Mapping[str, Any],
) -> PerProblemRecordSchema:
    """Validate one per-problem row against the schema.

    Raises `PerProblemRecordSchemaError` with a typed reason on
    any rule violation. On success, returns the frozen
    `PerProblemRecordSchema` dataclass populated from the input
    dict.
    """

    # 1) Required-field presence.
    for required in _REQUIRED_FIELDS:
        if required not in record:
            raise PerProblemRecordSchemaError(
                "missing_field",
                f"per-problem row missing required field {required!r}",
            )

    status = record["status"]
    # Type-check before set membership: TOML can supply a list
    # or dict here, and `value in set` raises a raw `TypeError`
    # on unhashable types. The CLI's exception handler only
    # catches PerProblemRecordSchemaError, so a raw TypeError
    # would propagate as a traceback instead of the documented
    # exit code 2.
    if not isinstance(status, str):
        raise PerProblemRecordSchemaError(
            "invalid_status",
            f"status must be a string, got {status!r} "
            f"({type(status).__name__})",
        )
    if status not in _VALID_STATUSES:
        raise PerProblemRecordSchemaError(
            "invalid_status",
            f"status {status!r} not in declared set "
            f"{sorted(_VALID_STATUSES)!r}",
        )

    category = record["problem_category"]
    # Same type-check guard as `status` above.
    if not isinstance(category, str):
        raise PerProblemRecordSchemaError(
            "invalid_category",
            f"problem_category must be a string, got {category!r} "
            f"({type(category).__name__})",
        )
    if category not in CANONICAL_CATEGORY_COUNTS:
        raise PerProblemRecordSchemaError(
            "invalid_category",
            f"problem_category {category!r} not in "
            f"{sorted(CANONICAL_CATEGORY_COUNTS.keys())!r}",
        )

    denominator_floor = record["denominator_floor"]
    if not _is_finite_float(denominator_floor) or float(denominator_floor) <= 0.0:
        raise PerProblemRecordSchemaError(
            "invalid_denominator_floor",
            f"denominator_floor must be a positive finite real, got "
            f"{denominator_floor!r}",
        )

    # 2) Status-dependent typed_reason vocabulary rules.
    typed_reason = record["typed_reason"]
    # Type-check before the frozenset membership tests below.
    # TOML can supply a list / dict here, and `value in <frozenset>`
    # raises raw `TypeError` on unhashable types. The CLI catches
    # `PerProblemRecordSchemaError` but not `TypeError`, so a raw
    # TypeError would propagate as a traceback instead of the
    # documented exit code 2. Empty string and None are still
    # valid for measured rows (handled by the explicit tuple
    # membership below).
    if typed_reason is not None and not isinstance(typed_reason, str):
        raise PerProblemRecordSchemaError(
            "invalid_typed_reason_vocabulary",
            f"typed_reason must be a string (or None / empty string "
            f"for measured rows), got {typed_reason!r} "
            f"({type(typed_reason).__name__})",
        )
    if status == STATUS_MEASURED:
        if typed_reason not in ("", None):
            raise PerProblemRecordSchemaError(
                "invalid_typed_reason_for_measured",
                f"status=measured row must carry empty typed_reason, "
                f"got {typed_reason!r}",
            )
    elif status == STATUS_TYPED_UNAVAILABLE:
        if typed_reason not in PREDICTOR_ABSTENTION_REASONS:
            raise PerProblemRecordSchemaError(
                "invalid_typed_reason_vocabulary",
                f"status=typed_unavailable row requires typed_reason in "
                f"PREDICTOR_ABSTENTION_REASONS, got {typed_reason!r}",
            )
        canonical_cells = canonical_typed_unavailable_cell_names()
        cell_name = record["cell_name"]
        if cell_name not in canonical_cells:
            raise PerProblemRecordSchemaError(
                "typed_unavailable_non_canonical_cell",
                f"status=typed_unavailable row has cell_name "
                f"{cell_name!r} which is not in the canonical "
                f"typed-unavailable cohort "
                f"{sorted(canonical_cells)!r}",
            )
    elif status == STATUS_DIAGNOSTIC_FAILED:
        if typed_reason not in ACCURACY_DIAGNOSTIC_TYPED_REASONS:
            raise PerProblemRecordSchemaError(
                "invalid_typed_reason_vocabulary",
                f"status=diagnostic_failed row requires typed_reason in "
                f"ACCURACY_DIAGNOSTIC_TYPED_REASONS, got "
                f"{typed_reason!r}",
            )
    elif status == STATUS_CORPUS_EXPANSION_PENDING:
        if typed_reason not in CORPUS_EXPANSION_TYPED_REASONS:
            raise PerProblemRecordSchemaError(
                "invalid_typed_reason_vocabulary",
                f"status=corpus_expansion_pending row requires typed_reason "
                f"in CORPUS_EXPANSION_TYPED_REASONS, got "
                f"{typed_reason!r}",
            )
    elif status == STATUS_HOST_ARCH_UNSUPPORTED:
        if typed_reason not in HOST_ARCH_UNSUPPORTED_TYPED_REASONS:
            raise PerProblemRecordSchemaError(
                "host_arch_typed_reason_vocabulary",
                f"status=host_arch_unsupported row requires typed_reason "
                f"in HOST_ARCH_UNSUPPORTED_TYPED_REASONS "
                f"{sorted(HOST_ARCH_UNSUPPORTED_TYPED_REASONS)!r}, got "
                f"{typed_reason!r}",
            )

    # 3) Rel-err field validity. For measured rows, all four
    # rel-err fields must be finite numeric. For non-measured
    # rows, all four must be the NAN_SENTINEL.
    for field in _REL_ERR_FIELDS:
        value = record[field]
        if status == STATUS_MEASURED:
            if not _is_finite_float(value):
                raise PerProblemRecordSchemaError(
                    "invalid_rel_err_value",
                    f"measured row field {field!r} must be a finite "
                    f"real number, got {value!r}",
                )
        else:
            if value != NAN_SENTINEL:
                raise PerProblemRecordSchemaError(
                    "non_measured_rel_err_must_be_sentinel",
                    f"non-measured row field {field!r} must equal "
                    f"NAN_SENTINEL ({NAN_SENTINEL!r}), got {value!r}",
                )

    # 4) Measured-row arithmetic consistency. Reconstruct all
    # four rel-err and abs-pp-error fields from the labels and
    # predictions; reject inconsistencies. Non-measured rows
    # must carry NAN_SENTINEL on every numeric field (already
    # checked above for rel-err; the abs-pp-error / label /
    # prediction fields are also sentinel-only for those).
    if status == STATUS_MEASURED:
        _validate_measured_row_arithmetic(record, float(denominator_floor))
    else:
        _validate_non_measured_row_numeric_sentinels(record)

    # 5) User-acceptance + tiered-gate verdict consistency.
    # The three verdict fields must agree with the values
    # recomputed from the row's status, predictor_bucket, and
    # calibrated rel-err fields.
    _validate_per_row_verdict_fields(record)

    typed_reason_value = "" if typed_reason is None else typed_reason
    audit_warning_value: tuple[str, ...] = tuple(
        record.get("audit_warning", ())
    )
    return PerProblemRecordSchema(
        problem_slug=record["problem_slug"],
        problem_category=record["problem_category"],
        cell_name=record["cell_name"],
        predictor_bucket=record["predictor_bucket"],
        ncu_l1_label=record["ncu_l1_label"],
        ncu_l2_label=record["ncu_l2_label"],
        predicted_l1_raw=record["predicted_l1_raw"],
        predicted_l2_raw=record["predicted_l2_raw"],
        predicted_l1_calibrated=record["predicted_l1_calibrated"],
        predicted_l2_calibrated=record["predicted_l2_calibrated"],
        l1_abs_pp_error=record["l1_abs_pp_error"],
        l2_abs_pp_error=record["l2_abs_pp_error"],
        l1_calibrated_abs_pp_error=record["l1_calibrated_abs_pp_error"],
        l2_calibrated_abs_pp_error=record["l2_calibrated_abs_pp_error"],
        l1_raw_rel_err=record["l1_raw_rel_err"],
        l2_raw_rel_err=record["l2_raw_rel_err"],
        l1_calibrated_rel_err=record["l1_calibrated_rel_err"],
        l2_calibrated_rel_err=record["l2_calibrated_rel_err"],
        denominator_floor=float(denominator_floor),
        status=status,
        typed_reason=typed_reason_value,
        pass_user_target_threshold=record["pass_user_target_threshold"],
        pass_tiered_gate=record["pass_tiered_gate"],
        effective_threshold_pct=record["effective_threshold_pct"],
        audit_warning=audit_warning_value,
    )


# ---------------------------------------------------------------------------
# Cohort + overall accuracy summary schemas
# ---------------------------------------------------------------------------

COHORT_FRIENDLY: str = "friendly"
COHORT_HOSTILE: str = "hostile"
COHORT_TYPED_UNAVAILABLE: str = "typed_unavailable"

_VALID_COHORTS: frozenset[str] = frozenset({
    COHORT_FRIENDLY,
    COHORT_HOSTILE,
    COHORT_TYPED_UNAVAILABLE,
})


CACHE_LEVEL_L1: str = "l1"
CACHE_LEVEL_L2: str = "l2"

_VALID_CACHE_LEVELS: frozenset[str] = frozenset({CACHE_LEVEL_L1, CACHE_LEVEL_L2})


# Sentinel emitted in cohort summary numeric-aggregate fields
# when the cohort contains fewer than 5 measured cells (the
# typed_unavailable cohort always falls into this case because
# no typed_unavailable cell is measured).
NAN_TOO_FEW_MEASURED_CELLS: str = "nan_too_few_measured_cells"


_COHORT_NUMERIC_AGGREGATE_FIELDS: tuple[str, ...] = (
    "mean_abs_pp_error",
    "max_abs_pp_error",
    "p95_abs_pp_error",
    "geomean_rel_err",
    "max_rel_err",
)


_COHORT_SUMMARY_REQUIRED_FIELDS: tuple[str, ...] = (
    "cohort",
    "cache_level",
    "admitted_cell_count",
    "mean_abs_pp_error",
    "max_abs_pp_error",
    "p95_abs_pp_error",
    "geomean_rel_err",
    "max_rel_err",
    "max_rel_err_cell_name",
    "cells_above_10pct_rel_err_count",
    "cells_above_10pct_rel_err_names",
    "accuracy_gate_passed",
)
assert len(_COHORT_SUMMARY_REQUIRED_FIELDS) == 12


@dataclass(frozen=True)
class CohortAccuracySummaryRow:
    """One row of the `[[cohort_accuracy_summary]]` table.

    The 6 rows correspond to the cohort cross-product
    `{friendly, hostile, typed_unavailable} x {l1, l2}`. The
    typed_unavailable cohort always emits
    `accuracy_gate_passed=false` because its 5 canonical cells
    have no scalar prediction; their `admitted_cell_count` is 5,
    `cells_above_10pct_rel_err_names` lists all 5 slugs, and
    numeric aggregates carry the `NAN_TOO_FEW_MEASURED_CELLS`
    sentinel (no measured cells contribute).

    Numeric aggregate fields are `float | str` so the sentinel
    can coexist with finite numerics. A cohort with
    `measured_cell_count < 5` (5 = COVERAGE_FLOOR_SMOKE) emits
    the sentinel rather than a numeric value.
    """

    cohort: str
    cache_level: str
    admitted_cell_count: int
    mean_abs_pp_error: float | str
    max_abs_pp_error: float | str
    p95_abs_pp_error: float | str
    geomean_rel_err: float | str
    max_rel_err: float | str
    max_rel_err_cell_name: str
    cells_above_10pct_rel_err_count: int
    cells_above_10pct_rel_err_names: tuple[str, ...]
    accuracy_gate_passed: bool


_OVERALL_SUMMARY_REQUIRED_FIELDS: tuple[str, ...] = (
    "cell_count",
    "cells_above_10pct_rel_err_count",
    "cells_above_10pct_rel_err_names",
    "accuracy_gate_passed",
    "calibration_only_measured_count",
    "validation_only_measured_count",
    "calibration_only_l1_mean_rel_err",
    "calibration_only_l2_mean_rel_err",
    "validation_only_l1_mean_rel_err",
    "validation_only_l2_mean_rel_err",
)


# Sentinel value carried by the per-split rel-err fields when
# no calibration / validation split manifest is available at
# diagnostic build time (e.g., legacy unit-test fixtures that
# don't yet thread the split manifest through). Production
# diagnostic runs always supply the split manifest; the
# sentinel preserves backward compatibility for paths that
# pre-date the calibration / validation split.
SPLIT_AWARE_REL_ERR_NA_SENTINEL: float = -1.0
SPLIT_AWARE_COUNT_NA_SENTINEL: int = -1


@dataclass(frozen=True)
class OverallAccuracySummary:
    """The `[overall_accuracy_summary]` table.

    `cell_count` is the cohort-partition total (today: 226 deployed
    partition cells, distinct from the 235-canonical-problem total);
    `accuracy_gate_passed` is the logical AND across all cohort rows.

    Calibration / validation split-aware fields: the six
    `calibration_only_*` /
    `validation_only_*` fields independently report measured
    cell counts and mean L1/L2 relative-error fractions across
    the two splits, so a future calibrator-refit can be scored
    on `validation_only` data while the training fit uses only
    `calibration_only`. The fields carry the sentinel
    `SPLIT_AWARE_REL_ERR_NA_SENTINEL` (-1.0) / `SPLIT_AWARE_COUNT_NA_SENTINEL`
    (-1) when no split manifest is available at build time.
    """

    cell_count: int
    cells_above_10pct_rel_err_count: int
    cells_above_10pct_rel_err_names: tuple[str, ...]
    accuracy_gate_passed: bool
    calibration_only_measured_count: int = SPLIT_AWARE_COUNT_NA_SENTINEL
    validation_only_measured_count: int = SPLIT_AWARE_COUNT_NA_SENTINEL
    calibration_only_l1_mean_rel_err: float = SPLIT_AWARE_REL_ERR_NA_SENTINEL
    calibration_only_l2_mean_rel_err: float = SPLIT_AWARE_REL_ERR_NA_SENTINEL
    validation_only_l1_mean_rel_err: float = SPLIT_AWARE_REL_ERR_NA_SENTINEL
    validation_only_l2_mean_rel_err: float = SPLIT_AWARE_REL_ERR_NA_SENTINEL


def _is_finite_or_sentinel(
    value: Any, sentinel: str
) -> bool:
    """Return True when `value` is either the literal sentinel
    string or a finite real number.
    """

    if value == sentinel:
        return True
    return _is_finite_float(value)


def validate_cohort_accuracy_summary_row(
    row: Mapping[str, Any],
    *,
    accuracy_threshold: float = 0.10,
) -> CohortAccuracySummaryRow:
    """Validate one `[[cohort_accuracy_summary]]` row.

    ``accuracy_threshold`` (default 0.10) is the rel-err cutoff
    the aggregator used to populate
    ``cells_above_10pct_rel_err_names`` and to clear
    ``max_rel_err_cell_name``. The validator must use the SAME
    threshold so reports built with a non-default
    ``accuracy_threshold`` (e.g., 0.20) do not falsely reject
    rows whose ``max_rel_err`` is between 0.10 and the
    configured threshold.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `cohort_summary_missing_field` — any of the 12 required
        fields is absent.
      * `cohort_summary_invalid_cohort` — `cohort` is not in
        `{friendly, hostile, typed_unavailable}`.
      * `cohort_summary_invalid_cache_level` — `cache_level` is
        not in `{l1, l2}`.
      * `cohort_summary_invalid_admitted_cell_count` —
        `admitted_cell_count` is not a non-negative integer.
      * `cohort_summary_invalid_numeric_aggregate` — any of the 5
        numeric aggregate fields is neither finite real nor the
        `NAN_TOO_FEW_MEASURED_CELLS` sentinel.
      * `cohort_summary_unsorted_failures` —
        `cells_above_10pct_rel_err_names` is not alphabetically
        sorted.
      * `cohort_summary_failure_count_mismatch` — the count and
        names fields disagree.
      * `cohort_summary_gate_passed_with_failures` —
        `accuracy_gate_passed=true` while
        `cells_above_10pct_rel_err_count > 0`.
      * `cohort_summary_typed_unavailable_gate_must_fail` — the
        typed_unavailable cohort emits `accuracy_gate_passed=true`
        (it must always be false because those cells have no
        scalar prediction).
      * `cohort_summary_max_rel_err_cell_name_not_in_failures` —
        the `max_rel_err_cell_name` is not in
        `cells_above_10pct_rel_err_names` when `max_rel_err >
        0.10`.
    """

    for field in _COHORT_SUMMARY_REQUIRED_FIELDS:
        if field not in row:
            raise PerProblemRecordSchemaError(
                "cohort_summary_missing_field",
                f"cohort summary row missing required field {field!r}",
            )

    cohort = row["cohort"]
    if cohort not in _VALID_COHORTS:
        raise PerProblemRecordSchemaError(
            "cohort_summary_invalid_cohort",
            f"cohort {cohort!r} not in {sorted(_VALID_COHORTS)!r}",
        )

    cache_level = row["cache_level"]
    if cache_level not in _VALID_CACHE_LEVELS:
        raise PerProblemRecordSchemaError(
            "cohort_summary_invalid_cache_level",
            f"cache_level {cache_level!r} not in "
            f"{sorted(_VALID_CACHE_LEVELS)!r}",
        )

    admitted_cell_count = row["admitted_cell_count"]
    if (
        not isinstance(admitted_cell_count, int)
        or isinstance(admitted_cell_count, bool)
        or admitted_cell_count <= 0
    ):
        raise PerProblemRecordSchemaError(
            "cohort_summary_invalid_admitted_cell_count",
            f"admitted_cell_count must be a positive int (no empty "
            f"cohort permitted in the canonical partition), got "
            f"{admitted_cell_count!r}",
        )

    for numeric_field in _COHORT_NUMERIC_AGGREGATE_FIELDS:
        value = row[numeric_field]
        if not _is_finite_or_sentinel(value, NAN_TOO_FEW_MEASURED_CELLS):
            raise PerProblemRecordSchemaError(
                "cohort_summary_invalid_numeric_aggregate",
                f"cohort summary field {numeric_field!r} must be a "
                f"finite real number or "
                f"NAN_TOO_FEW_MEASURED_CELLS sentinel, got {value!r}",
            )

    failure_names = tuple(row["cells_above_10pct_rel_err_names"])
    if list(failure_names) != sorted(failure_names):
        raise PerProblemRecordSchemaError(
            "cohort_summary_unsorted_failures",
            f"cells_above_10pct_rel_err_names must be alphabetically "
            f"sorted, got {list(failure_names)!r}",
        )

    failure_count = row["cells_above_10pct_rel_err_count"]
    if (
        not isinstance(failure_count, int)
        or isinstance(failure_count, bool)
        or failure_count < 0
    ):
        raise PerProblemRecordSchemaError(
            "cohort_summary_failure_count_mismatch",
            f"cells_above_10pct_rel_err_count must be a non-negative "
            f"int, got {failure_count!r}",
        )
    if failure_count != len(failure_names):
        raise PerProblemRecordSchemaError(
            "cohort_summary_failure_count_mismatch",
            f"cells_above_10pct_rel_err_count={failure_count} does not "
            f"match len(cells_above_10pct_rel_err_names)="
            f"{len(failure_names)}",
        )

    accuracy_gate_passed = row["accuracy_gate_passed"]
    if not isinstance(accuracy_gate_passed, bool):
        raise PerProblemRecordSchemaError(
            "cohort_summary_failure_count_mismatch",
            f"accuracy_gate_passed must be bool, got "
            f"{accuracy_gate_passed!r}",
        )

    if accuracy_gate_passed and failure_count > 0:
        raise PerProblemRecordSchemaError(
            "cohort_summary_gate_passed_with_failures",
            f"accuracy_gate_passed=true while "
            f"cells_above_10pct_rel_err_count={failure_count} > 0",
        )

    if cohort == COHORT_TYPED_UNAVAILABLE and accuracy_gate_passed:
        raise PerProblemRecordSchemaError(
            "cohort_summary_typed_unavailable_gate_must_fail",
            f"typed_unavailable cohort must always emit "
            f"accuracy_gate_passed=false (those cells have no "
            f"scalar prediction)",
        )

    max_rel_err_value = row["max_rel_err"]
    max_rel_err_cell_name = row["max_rel_err_cell_name"]
    if (
        _is_finite_float(max_rel_err_value)
        and float(max_rel_err_value) > accuracy_threshold
        and max_rel_err_cell_name not in failure_names
    ):
        raise PerProblemRecordSchemaError(
            "cohort_summary_max_rel_err_cell_name_not_in_failures",
            f"max_rel_err={max_rel_err_value!r} > "
            f"{accuracy_threshold!r} but "
            f"max_rel_err_cell_name={max_rel_err_cell_name!r} is "
            f"not in cells_above_10pct_rel_err_names "
            f"({list(failure_names)!r})",
        )

    return CohortAccuracySummaryRow(
        cohort=cohort,
        cache_level=cache_level,
        admitted_cell_count=admitted_cell_count,
        mean_abs_pp_error=row["mean_abs_pp_error"],
        max_abs_pp_error=row["max_abs_pp_error"],
        p95_abs_pp_error=row["p95_abs_pp_error"],
        geomean_rel_err=row["geomean_rel_err"],
        max_rel_err=max_rel_err_value,
        max_rel_err_cell_name=max_rel_err_cell_name,
        cells_above_10pct_rel_err_count=failure_count,
        cells_above_10pct_rel_err_names=failure_names,
        accuracy_gate_passed=accuracy_gate_passed,
    )


def validate_overall_accuracy_summary(
    table: Mapping[str, Any],
) -> OverallAccuracySummary:
    """Validate the single `[overall_accuracy_summary]` table.

    Enforces `cell_count == predictor_design_coverage.partition_size()`
    (the deployed cohort-partition total, distinct from the
    235-problem total which lives in
    `[coverage_summary].total_problems`); sorted failure-name list,
    count/names consistency, and gate coherence.
    """

    for field in _OVERALL_SUMMARY_REQUIRED_FIELDS:
        if field not in table:
            raise PerProblemRecordSchemaError(
                "overall_summary_missing_field",
                f"overall summary table missing required field {field!r}",
            )

    cell_count = table["cell_count"]
    # The overall summary's cell_count is the deployed cohort
    # partition total (today: 21 cohort A + 15 cohort B = 36).
    # That count is NOT the 235-problem total; the 235-problem
    # total lives in `[coverage_summary].total_problems`, because
    # the overall accuracy summary is the logical AND of the six
    # cohort rows (which cover only the cohort partition's
    # calibration cells). The literal 36 was hardcoded historically;
    # consulting `partition_size()` at runtime lets cohort growth
    # (new sm_86-compatible canonical problems being predeclared
    # in `predictor_design_coverage.py`) land without a lockstep
    # validator edit.
    expected_cell_count = partition_size()
    if (
        not isinstance(cell_count, int)
        or isinstance(cell_count, bool)
        or cell_count != expected_cell_count
    ):
        raise PerProblemRecordSchemaError(
            "overall_summary_invalid_cell_count",
            f"overall summary cell_count must equal "
            f"{expected_cell_count} (the cohort partition total "
            f"from predictor_design_coverage.partition_size()), "
            f"got {cell_count!r}",
        )

    failure_names = tuple(table["cells_above_10pct_rel_err_names"])
    if list(failure_names) != sorted(failure_names):
        raise PerProblemRecordSchemaError(
            "overall_summary_unsorted_failures",
            f"cells_above_10pct_rel_err_names must be alphabetically "
            f"sorted, got {list(failure_names)!r}",
        )

    failure_count = table["cells_above_10pct_rel_err_count"]
    if (
        not isinstance(failure_count, int)
        or isinstance(failure_count, bool)
        or failure_count < 0
        or failure_count != len(failure_names)
    ):
        raise PerProblemRecordSchemaError(
            "overall_summary_failure_count_mismatch",
            f"cells_above_10pct_rel_err_count={failure_count!r} "
            f"inconsistent with len(names)={len(failure_names)}",
        )

    accuracy_gate_passed = table["accuracy_gate_passed"]
    if not isinstance(accuracy_gate_passed, bool):
        raise PerProblemRecordSchemaError(
            "overall_summary_failure_count_mismatch",
            f"accuracy_gate_passed must be bool, got "
            f"{accuracy_gate_passed!r}",
        )
    if accuracy_gate_passed and failure_count > 0:
        raise PerProblemRecordSchemaError(
            "overall_summary_gate_passed_with_failures",
            f"overall accuracy_gate_passed=true while "
            f"cells_above_10pct_rel_err_count={failure_count} > 0",
        )

    # Calibration / validation split-aware fields. Each must
    # be present (the
    # presence-required gate above runs first via
    # _OVERALL_SUMMARY_REQUIRED_FIELDS). Validate type +
    # sentinel-or-finite-range for each.
    calibration_only_count = table["calibration_only_measured_count"]
    validation_only_count = table["validation_only_measured_count"]
    for field_name, field_value in (
        ("calibration_only_measured_count", calibration_only_count),
        ("validation_only_measured_count", validation_only_count),
    ):
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
        ):
            raise PerProblemRecordSchemaError(
                "overall_summary_invalid_split_count",
                f"{field_name}={field_value!r} must be an int "
                f"(non-negative or the sentinel "
                f"{SPLIT_AWARE_COUNT_NA_SENTINEL})",
            )
        if (
            field_value != SPLIT_AWARE_COUNT_NA_SENTINEL
            and field_value < 0
        ):
            raise PerProblemRecordSchemaError(
                "overall_summary_invalid_split_count",
                f"{field_name}={field_value!r} must be "
                f">= 0 or the sentinel "
                f"{SPLIT_AWARE_COUNT_NA_SENTINEL}",
            )

    rel_err_field_names = (
        "calibration_only_l1_mean_rel_err",
        "calibration_only_l2_mean_rel_err",
        "validation_only_l1_mean_rel_err",
        "validation_only_l2_mean_rel_err",
    )
    rel_err_values: dict[str, float] = {}
    for field_name in rel_err_field_names:
        field_value = table[field_name]
        if (
            not isinstance(field_value, (int, float))
            or isinstance(field_value, bool)
        ):
            raise PerProblemRecordSchemaError(
                "overall_summary_invalid_split_rel_err",
                f"{field_name}={field_value!r} must be a "
                f"finite float (>= 0.0) or the sentinel "
                f"{SPLIT_AWARE_REL_ERR_NA_SENTINEL}",
            )
        coerced = float(field_value)
        if (
            coerced != SPLIT_AWARE_REL_ERR_NA_SENTINEL
            and (math.isnan(coerced) or coerced < 0.0)
        ):
            raise PerProblemRecordSchemaError(
                "overall_summary_invalid_split_rel_err",
                f"{field_name}={field_value!r} must be a "
                f"non-negative finite float or the sentinel "
                f"{SPLIT_AWARE_REL_ERR_NA_SENTINEL}",
            )
        rel_err_values[field_name] = coerced

    # Per-split coherence: each (count, l1_mean, l2_mean) triple
    # is validated independently. The legitimate states for one
    # split are:
    #   * Legacy / no-split-manifest: count == sentinel AND
    #     both means == sentinel. The whole row is "no split
    #     available" for this side.
    #   * Zero measured rows: count == 0 AND both means ==
    #     sentinel (no measured rows to average over yet).
    #   * Populated: count > 0 AND both means finite non-
    #     negative.
    # Any other combination — count == sentinel with finite
    # mean, count > 0 with mean == sentinel, count == 0 with
    # finite mean — is a partial population that the schema
    # rejects. The earlier per-field type/range checks
    # already rejected count < 0 (with the sentinel
    # carve-out) and negative or NaN rel_err.
    def _check_split_triple(
        side: str,
        count: int,
        l1_mean: float,
        l2_mean: float,
    ) -> None:
        count_is_sentinel = count == SPLIT_AWARE_COUNT_NA_SENTINEL
        l1_is_sentinel = l1_mean == SPLIT_AWARE_REL_ERR_NA_SENTINEL
        l2_is_sentinel = l2_mean == SPLIT_AWARE_REL_ERR_NA_SENTINEL
        if count_is_sentinel:
            if not (l1_is_sentinel and l2_is_sentinel):
                raise PerProblemRecordSchemaError(
                    "overall_summary_split_partially_populated",
                    f"{side}: count is the sentinel "
                    f"{SPLIT_AWARE_COUNT_NA_SENTINEL} but at "
                    f"least one rel-err mean is finite "
                    f"({l1_mean!r} / {l2_mean!r}). Either set "
                    f"both means to the sentinel "
                    f"{SPLIT_AWARE_REL_ERR_NA_SENTINEL} (no "
                    f"split manifest at build time) or set "
                    f"count to a non-negative integer.",
                )
            return
        # count >= 0 here.
        if count == 0:
            if not (l1_is_sentinel and l2_is_sentinel):
                raise PerProblemRecordSchemaError(
                    "overall_summary_split_partially_populated",
                    f"{side}: count is 0 but at least one "
                    f"rel-err mean is finite "
                    f"({l1_mean!r} / {l2_mean!r}). With zero "
                    f"measured rows there is nothing to "
                    f"average; both means must carry the "
                    f"sentinel {SPLIT_AWARE_REL_ERR_NA_SENTINEL}.",
                )
            return
        # count > 0 here.
        if l1_is_sentinel or l2_is_sentinel:
            raise PerProblemRecordSchemaError(
                "overall_summary_split_partially_populated",
                f"{side}: count is {count} > 0 but at least "
                f"one rel-err mean is the sentinel "
                f"{SPLIT_AWARE_REL_ERR_NA_SENTINEL} "
                f"(l1={l1_mean!r}, l2={l2_mean!r}). Populate "
                f"both means with finite non-negative values "
                f"computed from the {count} measured row(s).",
            )

    # Cross-split check: either BOTH sides are sentinel
    # (legacy / no split manifest) OR BOTH sides have a
    # non-negative count (populated split context). Mixing
    # one sentinel count with one populated count lets a
    # report claim a single aggregated train/validation
    # rel-err on the populated side while hiding the other,
    # which the calibration / validation split contract
    # forbids: aggregating train and validation rel-err into
    # a single number must not be representable.
    calibration_count_is_sentinel = (
        calibration_only_count == SPLIT_AWARE_COUNT_NA_SENTINEL
    )
    validation_count_is_sentinel = (
        validation_only_count == SPLIT_AWARE_COUNT_NA_SENTINEL
    )
    if calibration_count_is_sentinel != validation_count_is_sentinel:
        raise PerProblemRecordSchemaError(
            "overall_summary_split_partially_populated",
            f"split-aware counts must be EITHER both the "
            f"sentinel {SPLIT_AWARE_COUNT_NA_SENTINEL} (legacy "
            f"row representing no split manifest) OR both >= 0 "
            f"(populated split context). Got "
            f"calibration_only_measured_count="
            f"{calibration_only_count!r}, "
            f"validation_only_measured_count="
            f"{validation_only_count!r}.",
        )

    _check_split_triple(
        "calibration_only",
        calibration_only_count,
        rel_err_values["calibration_only_l1_mean_rel_err"],
        rel_err_values["calibration_only_l2_mean_rel_err"],
    )
    _check_split_triple(
        "validation_only",
        validation_only_count,
        rel_err_values["validation_only_l1_mean_rel_err"],
        rel_err_values["validation_only_l2_mean_rel_err"],
    )

    return OverallAccuracySummary(
        cell_count=cell_count,
        cells_above_10pct_rel_err_count=failure_count,
        cells_above_10pct_rel_err_names=failure_names,
        accuracy_gate_passed=accuracy_gate_passed,
        calibration_only_measured_count=calibration_only_count,
        validation_only_measured_count=validation_only_count,
        calibration_only_l1_mean_rel_err=rel_err_values[
            "calibration_only_l1_mean_rel_err"
        ],
        calibration_only_l2_mean_rel_err=rel_err_values[
            "calibration_only_l2_mean_rel_err"
        ],
        validation_only_l1_mean_rel_err=rel_err_values[
            "validation_only_l1_mean_rel_err"
        ],
        validation_only_l2_mean_rel_err=rel_err_values[
            "validation_only_l2_mean_rel_err"
        ],
    )


# ---------------------------------------------------------------------------
# Coverage summary table schema
# ---------------------------------------------------------------------------

_COVERAGE_SUMMARY_REQUIRED_FIELDS: tuple[str, ...] = (
    "total_problems",
    "category_counts",
    "friendly_cells",
    "hostile_cells",
    "typed_unavailable_cells",
    "diagnostic_failed_cells",
    "corpus_expansion_pending_cells",
    "host_arch_unsupported_cells",
    "measured_cell_count",
    "typed_unavailable_reasons",
    "diagnostic_failed_reasons",
    "corpus_expansion_pending_by_reason",
    "host_arch_unsupported_reasons",
    "admitted_measured_fraction",
    "coverage_floor_met",
)


@dataclass(frozen=True)
class CoverageSummaryTable:
    """The `[coverage_summary]` table covering the canonical
    235-problem partition.

    `total_problems` is always 235. `category_counts` carries the
    fixed `{l1: 94, l2: 82, quant: 33, fib: 26}` split.

    The mutually-exclusive partition: every canonical problem
    falls in exactly one of friendly / hostile / typed_unavailable
    (the cohort-classified subset) OR corpus_expansion_pending
    (problems lacking hand-port / labels / cohort classification)
    OR host_arch_unsupported (problems whose kernels cannot
    execute on the local sm_86 host and are exempt from the
    user-acceptance gate via the `host_arch_availability_v1.toml`
    manifest). `friendly_cells + hostile_cells +
    typed_unavailable_cells + corpus_expansion_pending_cells +
    host_arch_unsupported_cells + measured_cell_count +
    diagnostic_failed_cells == total_problems` is the 7-bucket
    sum invariant.

    `diagnostic_failed_cells` is an overlay subset of
    `friendly_cells + hostile_cells`: admitted-cohort cells that
    failed during pipeline execution. typed_unavailable cells
    cannot fail diagnostically because the predictor is never
    invoked on them. corpus_expansion_pending cells have not yet
    reached the pipeline. host_arch_unsupported cells are carved
    out before any cohort / hand-port / ledger lookup.

    `measured_cell_count` is the count of admitted cells that
    successfully reached a measured-row emission. The invariant
    is `measured_cell_count <= friendly_cells + hostile_cells -
    diagnostic_failed_cells`.

    `admitted_measured_fraction = measured_cell_count /
    total_problems`. `coverage_floor_met` is True iff
    `measured_cell_count >= COVERAGE_FLOOR_SMOKE` (5).
    """

    total_problems: int
    category_counts: Mapping[str, int]
    friendly_cells: int
    hostile_cells: int
    typed_unavailable_cells: int
    diagnostic_failed_cells: int
    corpus_expansion_pending_cells: int
    host_arch_unsupported_cells: int
    measured_cell_count: int
    typed_unavailable_reasons: Mapping[str, int]
    diagnostic_failed_reasons: Mapping[str, int]
    corpus_expansion_pending_by_reason: Mapping[str, int]
    host_arch_unsupported_reasons: Mapping[str, int]
    admitted_measured_fraction: float
    coverage_floor_met: bool


_COVERAGE_FLOOR_SMOKE: int = 5


def validate_coverage_summary_table(
    table: Mapping[str, Any],
) -> CoverageSummaryTable:
    """Validate the `[coverage_summary]` table.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `coverage_summary_missing_field`
      * `coverage_summary_invalid_total_problems` —
        `total_problems != 235`.
      * `coverage_summary_invalid_category_counts` —
        `category_counts` does not equal
        `{l1: 94, l2: 82, quant: 33, fib: 26}`.
      * `coverage_summary_bucket_sum_mismatch` —
        `friendly_cells + hostile_cells +
        typed_unavailable_cells != total_problems`.
      * `coverage_summary_runtime_failures_exceed_admitted` —
        `diagnostic_failed_cells +
        corpus_expansion_pending_cells > friendly_cells +
        hostile_cells`.
      * `coverage_summary_measured_count_exceeds_admitted` —
        `measured_cell_count > friendly_cells + hostile_cells -
        diagnostic_failed_cells - corpus_expansion_pending_cells`.
      * `coverage_summary_invalid_reason_vocabulary` — any
        frequency-dict key is outside its vocabulary.
      * `coverage_summary_invalid_admitted_fraction` —
        `admitted_measured_fraction` is not in `[0, 1]` or
        disagrees with `measured_cell_count / total_problems`.
      * `coverage_summary_invalid_coverage_floor_met` —
        `coverage_floor_met` disagrees with the `>=
        COVERAGE_FLOOR_SMOKE` rule.
    """

    for field in _COVERAGE_SUMMARY_REQUIRED_FIELDS:
        if field not in table:
            raise PerProblemRecordSchemaError(
                "coverage_summary_missing_field",
                f"coverage summary missing required field {field!r}",
            )

    total_problems = table["total_problems"]
    if total_problems != 235:
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_total_problems",
            f"total_problems must equal 235, got {total_problems!r}",
        )

    category_counts = dict(table["category_counts"])
    expected_category_counts = {
        "l1": 94, "l2": 82, "quant": 33, "fib": 26,
    }
    if category_counts != expected_category_counts:
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_category_counts",
            f"category_counts must equal {expected_category_counts!r}, "
            f"got {category_counts!r}",
        )

    bucket_count_fields = (
        "friendly_cells",
        "hostile_cells",
        "typed_unavailable_cells",
        "corpus_expansion_pending_cells",
        "host_arch_unsupported_cells",
        "diagnostic_failed_cells",
        "measured_cell_count",
    )
    for field_name in bucket_count_fields:
        value = table[field_name]
        # bool is a subclass of int; reject it explicitly so
        # `True`/`False` cannot smuggle into integer counts.
        if isinstance(value, bool) or not isinstance(value, int):
            raise PerProblemRecordSchemaError(
                "coverage_summary_invalid_count",
                f"coverage_summary.{field_name} must be a non-bool "
                f"integer; got {value!r} ({type(value).__name__})",
            )
        # Reject negative bucket counts. A negative value would
        # otherwise pass the type check and could still satisfy
        # the seven-bucket sum=235 invariant (e.g.,
        # friendly_cells=-1 paired with measured_cell_count=236),
        # producing an internally inconsistent CoverageSummaryTable.
        if value < 0:
            raise PerProblemRecordSchemaError(
                "coverage_summary_invalid_count",
                f"coverage_summary.{field_name} must be >= 0; "
                f"got {value!r}",
            )
    friendly = table["friendly_cells"]
    hostile = table["hostile_cells"]
    typed_unavail = table["typed_unavailable_cells"]
    corpus_pending = table["corpus_expansion_pending_cells"]
    host_arch_unsup = table["host_arch_unsupported_cells"]
    diag_failed = table["diagnostic_failed_cells"]
    measured = table["measured_cell_count"]
    # Exact-bucket invariant: the seven buckets sum to exactly
    # the canonical 235-problem total. host_arch_unsupported is
    # the seventh bucket (added after the host-arch carve-out
    # detector landed): problems whose kernels cannot execute on
    # the local sm_86 host are exempt from the user-acceptance
    # denominator and are counted here BEFORE the cohort /
    # hand-port / ledger detectors fire.
    bucket_sum = (
        friendly + hostile + typed_unavail
        + diag_failed + corpus_pending
        + host_arch_unsup + measured
    )
    if bucket_sum != total_problems:
        raise PerProblemRecordSchemaError(
            "coverage_summary_bucket_sum_mismatch",
            f"friendly_cells({friendly}) + hostile_cells({hostile}) + "
            f"typed_unavailable_cells({typed_unavail}) + "
            f"diagnostic_failed_cells({diag_failed}) + "
            f"corpus_expansion_pending_cells({corpus_pending}) + "
            f"host_arch_unsupported_cells({host_arch_unsup}) + "
            f"measured_cell_count({measured}) = {bucket_sum} "
            f"!= total_problems({total_problems})",
        )

    typed_unavail_reasons = dict(table["typed_unavailable_reasons"])
    for reason in typed_unavail_reasons:
        if reason not in PREDICTOR_ABSTENTION_REASONS:
            raise PerProblemRecordSchemaError(
                "coverage_summary_invalid_reason_vocabulary",
                f"typed_unavailable_reasons key {reason!r} not in "
                f"PREDICTOR_ABSTENTION_REASONS",
            )

    diag_reasons = dict(table["diagnostic_failed_reasons"])
    for reason in diag_reasons:
        if reason not in ACCURACY_DIAGNOSTIC_TYPED_REASONS:
            raise PerProblemRecordSchemaError(
                "coverage_summary_invalid_reason_vocabulary",
                f"diagnostic_failed_reasons key {reason!r} not in "
                f"ACCURACY_DIAGNOSTIC_TYPED_REASONS",
            )

    corpus_reasons = dict(table["corpus_expansion_pending_by_reason"])
    for reason in corpus_reasons:
        if reason not in CORPUS_EXPANSION_TYPED_REASONS:
            raise PerProblemRecordSchemaError(
                "coverage_summary_invalid_reason_vocabulary",
                f"corpus_expansion_pending_by_reason key {reason!r} not "
                f"in CORPUS_EXPANSION_TYPED_REASONS",
            )

    host_arch_unsupported_reasons_dict = dict(
        table["host_arch_unsupported_reasons"],
    )
    for reason in host_arch_unsupported_reasons_dict:
        if reason not in HOST_ARCH_UNSUPPORTED_TYPED_REASONS:
            raise PerProblemRecordSchemaError(
                "coverage_summary_invalid_reason_vocabulary",
                f"host_arch_unsupported_reasons key {reason!r} not "
                f"in HOST_ARCH_UNSUPPORTED_TYPED_REASONS",
            )

    admitted_fraction_raw = table["admitted_measured_fraction"]
    # Reject bool / non-numeric / non-finite / out-of-range values
    # BEFORE the tolerance check. `float(nan)` returns nan and
    # `abs(nan - x) > tol` is False, so without this guard a
    # report carrying `admitted_measured_fraction = nan` (or the
    # string `"nan"`) would silently validate.
    if (
        isinstance(admitted_fraction_raw, bool)
        or not isinstance(admitted_fraction_raw, (int, float))
    ):
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_admitted_fraction",
            f"admitted_measured_fraction must be a non-bool real number; "
            f"got {admitted_fraction_raw!r} "
            f"({type(admitted_fraction_raw).__name__})",
        )
    admitted_fraction = float(admitted_fraction_raw)
    if not math.isfinite(admitted_fraction):
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_admitted_fraction",
            f"admitted_measured_fraction must be finite; "
            f"got {admitted_fraction_raw!r}",
        )
    if not (0.0 <= admitted_fraction <= 1.0):
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_admitted_fraction",
            f"admitted_measured_fraction must be in [0.0, 1.0]; "
            f"got {admitted_fraction_raw!r}",
        )
    expected_fraction = measured / total_problems
    if abs(admitted_fraction - expected_fraction) > _ARITHMETIC_TOLERANCE:
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_admitted_fraction",
            f"admitted_measured_fraction({admitted_fraction}) does not "
            f"match measured_cell_count/total_problems"
            f"={expected_fraction}",
        )

    coverage_floor_met = table["coverage_floor_met"]
    if not isinstance(coverage_floor_met, bool):
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_coverage_floor_met",
            f"coverage_floor_met must be bool, got "
            f"{coverage_floor_met!r}",
        )
    expected_floor = measured >= _COVERAGE_FLOOR_SMOKE
    if coverage_floor_met != expected_floor:
        raise PerProblemRecordSchemaError(
            "coverage_summary_invalid_coverage_floor_met",
            f"coverage_floor_met={coverage_floor_met} disagrees with "
            f"(measured_cell_count >= 5)={expected_floor}",
        )

    return CoverageSummaryTable(
        total_problems=total_problems,
        category_counts=category_counts,
        friendly_cells=friendly,
        hostile_cells=hostile,
        typed_unavailable_cells=typed_unavail,
        diagnostic_failed_cells=diag_failed,
        corpus_expansion_pending_cells=corpus_pending,
        host_arch_unsupported_cells=host_arch_unsup,
        measured_cell_count=measured,
        typed_unavailable_reasons=typed_unavail_reasons,
        diagnostic_failed_reasons=diag_reasons,
        corpus_expansion_pending_by_reason=corpus_reasons,
        host_arch_unsupported_reasons=host_arch_unsupported_reasons_dict,
        admitted_measured_fraction=admitted_fraction,
        coverage_floor_met=coverage_floor_met,
    )


# ---------------------------------------------------------------------------
# Denominator box source-of-truth table + overall user-target summary
# ---------------------------------------------------------------------------

_DENOMINATOR_BOX_REQUIRED_FIELDS: tuple[str, ...] = (
    "total_problems",
    "measured",
    "typed_unavailable",
    "diagnostic_failed",
    "corpus_pending",
    "host_arch_unsupported",
    "passing_user_target",
    "passing_tiered_gate",
    "passing_compound_gate",
)


@dataclass(frozen=True)
class DenominatorBox:
    """The `[denominator_box]` source-of-truth count table.

    Carries the partition counts under the 7-bucket
    mutually-exclusive partition (the same partition as
    `[coverage_summary]` but presented separately so the
    final-acceptance dashboard can reconcile against ONE table)
    plus three aggregate gate counts:

      * `passing_user_target`: count of rows where
        `pass_user_target_threshold == True` (uniform 10% gate
        per DEC-1; non-measured rows count False).
      * `passing_tiered_gate`: count of rows where
        `pass_tiered_gate == True` (5% friendly / 15% hostile;
        typed-unavailable + non-measured count False).
      * `passing_compound_gate`: count of rows where BOTH
        verdicts are True (which on the current schema is
        equivalent to measured rows that pass both gates) PLUS
        the host_arch_unsupported bucket carved out by the host-
        arch availability manifest and included in the final-
        acceptance condition
        `passing_compound_gate == measured + host_arch_unsupported`.

    Invariants the validator enforces:
      * `measured + typed_unavailable + diagnostic_failed
         + corpus_pending + host_arch_unsupported == total_problems`
         (where `total_problems == 235`).
      * `0 <= passing_* <= total_problems`.
      * `passing_compound_gate <= passing_user_target + host_arch_unsupported`.
      * `passing_compound_gate <= passing_tiered_gate + host_arch_unsupported`.
    """

    total_problems: int
    measured: int
    typed_unavailable: int
    diagnostic_failed: int
    corpus_pending: int
    host_arch_unsupported: int
    passing_user_target: int
    passing_tiered_gate: int
    passing_compound_gate: int


def validate_denominator_box(
    table: Mapping[str, Any],
) -> DenominatorBox:
    """Validate the `[denominator_box]` table.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `denominator_box_missing_field`.
      * `denominator_box_invalid_count` — a count field is a
        non-int / bool / negative / out-of-range value.
      * `denominator_box_partition_mismatch` — the five status
        buckets do not sum to `total_problems`.
      * `denominator_box_gate_count_exceeds_partition` — a
        `passing_*` field exceeds `total_problems` or violates
        the cross-gate inequality.
    """

    for field in _DENOMINATOR_BOX_REQUIRED_FIELDS:
        if field not in table:
            raise PerProblemRecordSchemaError(
                "denominator_box_missing_field",
                f"denominator_box table missing required field "
                f"{field!r}",
            )

    for field in _DENOMINATOR_BOX_REQUIRED_FIELDS:
        value = table[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise PerProblemRecordSchemaError(
                "denominator_box_invalid_count",
                f"denominator_box.{field} must be a non-bool "
                f"non-negative integer; got {value!r} "
                f"({type(value).__name__})",
            )

    total_problems = table["total_problems"]
    if total_problems != 235:
        raise PerProblemRecordSchemaError(
            "denominator_box_invalid_count",
            f"denominator_box.total_problems must equal 235, "
            f"got {total_problems!r}",
        )

    measured = table["measured"]
    typed_unavail = table["typed_unavailable"]
    diag_failed = table["diagnostic_failed"]
    corpus_pending = table["corpus_pending"]
    host_arch_unsup = table["host_arch_unsupported"]
    partition_sum = (
        measured + typed_unavail + diag_failed
        + corpus_pending + host_arch_unsup
    )
    if partition_sum != total_problems:
        raise PerProblemRecordSchemaError(
            "denominator_box_partition_mismatch",
            f"denominator_box partition "
            f"(measured={measured} + typed_unavailable={typed_unavail} + "
            f"diagnostic_failed={diag_failed} + "
            f"corpus_pending={corpus_pending} + "
            f"host_arch_unsupported={host_arch_unsup}) = "
            f"{partition_sum} != total_problems({total_problems})",
        )

    passing_user = table["passing_user_target"]
    passing_tiered = table["passing_tiered_gate"]
    passing_compound = table["passing_compound_gate"]
    for field, value in (
        ("passing_user_target", passing_user),
        ("passing_tiered_gate", passing_tiered),
        ("passing_compound_gate", passing_compound),
    ):
        if value > total_problems:
            raise PerProblemRecordSchemaError(
                "denominator_box_gate_count_exceeds_partition",
                f"denominator_box.{field}={value} exceeds "
                f"total_problems({total_problems})",
            )

    # Compound gate counts measured-passing rows + host_arch
    # carveout. It cannot exceed either constituent count.
    if passing_compound > passing_user + host_arch_unsup:
        raise PerProblemRecordSchemaError(
            "denominator_box_gate_count_exceeds_partition",
            f"denominator_box.passing_compound_gate("
            f"{passing_compound}) > passing_user_target("
            f"{passing_user}) + host_arch_unsupported("
            f"{host_arch_unsup}); compound cannot pass more "
            f"rows than user-acceptance plus the carveout.",
        )
    if passing_compound > passing_tiered + host_arch_unsup:
        raise PerProblemRecordSchemaError(
            "denominator_box_gate_count_exceeds_partition",
            f"denominator_box.passing_compound_gate("
            f"{passing_compound}) > passing_tiered_gate("
            f"{passing_tiered}) + host_arch_unsupported("
            f"{host_arch_unsup}); compound cannot pass more "
            f"rows than tiered plus the carveout.",
        )

    return DenominatorBox(
        total_problems=total_problems,
        measured=measured,
        typed_unavailable=typed_unavail,
        diagnostic_failed=diag_failed,
        corpus_pending=corpus_pending,
        host_arch_unsupported=host_arch_unsup,
        passing_user_target=passing_user,
        passing_tiered_gate=passing_tiered,
        passing_compound_gate=passing_compound,
    )


_OVERALL_USER_TARGET_SUMMARY_REQUIRED_FIELDS: tuple[str, ...] = (
    "measured_pass_count",
    "measured_fail_count",
    "user_target_threshold_pct",
    "gate_compound_passed",
)


@dataclass(frozen=True)
class OverallUserTargetSummary:
    """The `[overall_user_target_summary]` table.

    Carries the user-acceptance gate aggregation: of the
    measured rows, how many pass the uniform <10% threshold
    (DEC-1), how many fail, and whether the compound gate
    (contract AND user-acceptance) passes across the whole
    canonical partition.

    The `user_target_threshold_pct` field is the uniform
    user-acceptance threshold (10.0 today). The
    `gate_compound_passed` bool mirrors `meta.gate_compound`
    for downstream consumers that read this section.
    """

    measured_pass_count: int
    measured_fail_count: int
    user_target_threshold_pct: float
    gate_compound_passed: bool


def validate_overall_user_target_summary(
    table: Mapping[str, Any],
) -> OverallUserTargetSummary:
    """Validate the `[overall_user_target_summary]` table.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `overall_user_target_summary_missing_field`.
      * `overall_user_target_summary_invalid_count` — non-int /
        bool / negative count value.
      * `overall_user_target_summary_invalid_threshold` —
        `user_target_threshold_pct` is not finite OR is outside
        (0, 100].
      * `overall_user_target_summary_invalid_gate_bool` —
        `gate_compound_passed` is not a Python bool.
    """

    for field in _OVERALL_USER_TARGET_SUMMARY_REQUIRED_FIELDS:
        if field not in table:
            raise PerProblemRecordSchemaError(
                "overall_user_target_summary_missing_field",
                f"overall_user_target_summary table missing "
                f"required field {field!r}",
            )

    for count_field in ("measured_pass_count", "measured_fail_count"):
        value = table[count_field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise PerProblemRecordSchemaError(
                "overall_user_target_summary_invalid_count",
                f"overall_user_target_summary.{count_field} must "
                f"be a non-bool non-negative integer; got "
                f"{value!r} ({type(value).__name__})",
            )

    threshold_pct = table["user_target_threshold_pct"]
    if (
        isinstance(threshold_pct, bool)
        or not isinstance(threshold_pct, (int, float))
        or not math.isfinite(float(threshold_pct))
        or not (0.0 < float(threshold_pct) <= 100.0)
    ):
        raise PerProblemRecordSchemaError(
            "overall_user_target_summary_invalid_threshold",
            f"overall_user_target_summary.user_target_threshold_pct "
            f"must be a finite real in (0, 100]; got "
            f"{threshold_pct!r}",
        )

    gate_passed = table["gate_compound_passed"]
    if not isinstance(gate_passed, bool):
        raise PerProblemRecordSchemaError(
            "overall_user_target_summary_invalid_gate_bool",
            f"overall_user_target_summary.gate_compound_passed "
            f"must be a Python bool; got {gate_passed!r} "
            f"({type(gate_passed).__name__})",
        )

    return OverallUserTargetSummary(
        measured_pass_count=table["measured_pass_count"],
        measured_fail_count=table["measured_fail_count"],
        user_target_threshold_pct=float(threshold_pct),
        gate_compound_passed=gate_passed,
    )


# ---------------------------------------------------------------------------
# Denominator audit row schema
# ---------------------------------------------------------------------------

_DENOMINATOR_AUDIT_REQUIRED_FIELDS: tuple[str, ...] = (
    "cell_name",
    "tier2_l1_total",
    "ncu_l1_total",
    "tier2_l2_total",
    "ncu_l2_total",
    "l1_total_rel_gap",
    "l2_total_rel_gap",
    "audit_passed",
)


_DENOMINATOR_AUDIT_GAP_THRESHOLD: float = 0.05


@dataclass(frozen=True)
class DenominatorAuditRow:
    """One row of the new `[[denominator_audit]]` table (L1+L2
    sector-total audit). The existing `[[l2_denominator_audit]]`
    table emitted by `end_to_end_smoke.py` remains in place; this
    table is additive.

    `audit_passed` is True only when BOTH L1 and L2 rel-gaps are
    within `_DENOMINATOR_AUDIT_GAP_THRESHOLD` (5%). The accuracy
    gate (`compute_accuracy_diagnostic_gate`) is hit-rate-only;
    audit failures are reported but do NOT auto-block the gate.
    """

    cell_name: str
    tier2_l1_total: float
    ncu_l1_total: float
    tier2_l2_total: float
    ncu_l2_total: float
    l1_total_rel_gap: float
    l2_total_rel_gap: float
    audit_passed: bool


def validate_denominator_audit_row(
    row: Mapping[str, Any],
) -> DenominatorAuditRow:
    """Validate one `[[denominator_audit]]` row.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `denominator_audit_missing_field`.
      * `denominator_audit_invalid_total` — any sector-total
        field is non-finite or negative.
      * `denominator_audit_invalid_gap` — any rel-gap field is
        non-finite or negative.
      * `denominator_audit_passed_with_excess_gap` —
        `audit_passed=true` but either rel-gap exceeds 5%.
      * `denominator_audit_failed_with_clean_gaps` —
        `audit_passed=false` but both rel-gaps are within 5%
        (sanity check; prevents silent inversion).
    """

    for field in _DENOMINATOR_AUDIT_REQUIRED_FIELDS:
        if field not in row:
            raise PerProblemRecordSchemaError(
                "denominator_audit_missing_field",
                f"denominator audit row missing required field {field!r}",
            )

    for total_field in (
        "tier2_l1_total", "ncu_l1_total", "tier2_l2_total", "ncu_l2_total",
    ):
        value = row[total_field]
        if not _is_finite_float(value) or float(value) < 0.0:
            raise PerProblemRecordSchemaError(
                "denominator_audit_invalid_total",
                f"{total_field} must be a non-negative finite real, got "
                f"{value!r}",
            )

    for gap_field in ("l1_total_rel_gap", "l2_total_rel_gap"):
        value = row[gap_field]
        if not _is_finite_float(value) or float(value) < 0.0:
            raise PerProblemRecordSchemaError(
                "denominator_audit_invalid_gap",
                f"{gap_field} must be a non-negative finite real, got "
                f"{value!r}",
            )

    audit_passed = row["audit_passed"]
    if not isinstance(audit_passed, bool):
        raise PerProblemRecordSchemaError(
            "denominator_audit_invalid_gap",
            f"audit_passed must be bool, got {audit_passed!r}",
        )

    l1_gap = float(row["l1_total_rel_gap"])
    l2_gap = float(row["l2_total_rel_gap"])

    # Storage-consistency: recompute the rel-gap from the
    # row's `tier2_*_total` and `ncu_*_total` and reject when
    # the stored gap disagrees by more than the tolerance.
    # Without this check, a tampered row (e.g.,
    # `tier2_l1_total=0`, `ncu_l1_total=100`, `l1_total_rel_gap=0`,
    # `audit_passed=true`) validates cleanly even though the
    # totals show a 100% gap.
    tier2_l1 = float(row["tier2_l1_total"])
    ncu_l1 = float(row["ncu_l1_total"])
    tier2_l2 = float(row["tier2_l2_total"])
    ncu_l2 = float(row["ncu_l2_total"])
    expected_l1_gap = abs(tier2_l1 - ncu_l1) / max(ncu_l1, 1.0)
    expected_l2_gap = abs(tier2_l2 - ncu_l2) / max(ncu_l2, 1.0)
    if abs(l1_gap - expected_l1_gap) > _ARITHMETIC_TOLERANCE:
        raise PerProblemRecordSchemaError(
            "denominator_audit_gap_inconsistent",
            f"stored l1_total_rel_gap={l1_gap!r} disagrees with formula "
            f"abs(tier2_l1_total={tier2_l1!r} - ncu_l1_total={ncu_l1!r}) "
            f"/ max(ncu_l1_total, 1.0) = {expected_l1_gap!r}",
        )
    if abs(l2_gap - expected_l2_gap) > _ARITHMETIC_TOLERANCE:
        raise PerProblemRecordSchemaError(
            "denominator_audit_gap_inconsistent",
            f"stored l2_total_rel_gap={l2_gap!r} disagrees with formula "
            f"abs(tier2_l2_total={tier2_l2!r} - ncu_l2_total={ncu_l2!r}) "
            f"/ max(ncu_l2_total, 1.0) = {expected_l2_gap!r}",
        )

    gaps_within_threshold = (
        l1_gap <= _DENOMINATOR_AUDIT_GAP_THRESHOLD
        and l2_gap <= _DENOMINATOR_AUDIT_GAP_THRESHOLD
    )

    if audit_passed and not gaps_within_threshold:
        raise PerProblemRecordSchemaError(
            "denominator_audit_passed_with_excess_gap",
            f"audit_passed=true but rel-gaps exceed "
            f"{_DENOMINATOR_AUDIT_GAP_THRESHOLD}: "
            f"l1_total_rel_gap={l1_gap}, l2_total_rel_gap={l2_gap}",
        )
    if not audit_passed and gaps_within_threshold:
        raise PerProblemRecordSchemaError(
            "denominator_audit_failed_with_clean_gaps",
            f"audit_passed=false but both rel-gaps are within "
            f"{_DENOMINATOR_AUDIT_GAP_THRESHOLD}: "
            f"l1_total_rel_gap={l1_gap}, l2_total_rel_gap={l2_gap}",
        )

    return DenominatorAuditRow(
        cell_name=row["cell_name"],
        tier2_l1_total=float(row["tier2_l1_total"]),
        ncu_l1_total=float(row["ncu_l1_total"]),
        tier2_l2_total=float(row["tier2_l2_total"]),
        ncu_l2_total=float(row["ncu_l2_total"]),
        l1_total_rel_gap=l1_gap,
        l2_total_rel_gap=l2_gap,
        audit_passed=audit_passed,
    )


# ---------------------------------------------------------------------------
# Deterministic [meta] table schema
# ---------------------------------------------------------------------------

_DIAGNOSTIC_META_REQUIRED_FIELDS: tuple[str, ...] = (
    "implementation_commit_sha",
    "ledger_path",
    "ledger_sha256",
    "nvbit_so_sha256",
    "gpu_arch",
    "gpu_device_name",
    "ncu_metric_version",
    "driver_version",
    "trace_budget_bytes",
    "timeout_seconds",
    "report_generated_at_utc",
    "denominator_floor",
    "accuracy_threshold",
    "gate_compound",
)


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_ALL_ZERO_SHA_PATTERN = re.compile(r"^0+$")


def _resolve_implementation_commit_sha_from_git(
    repo_root: Path | None = None,
) -> str:
    """Return the current HEAD commit SHA via `git rev-parse HEAD`.

    Returns the literal sentinel `"git_unavailable"` when git is
    not installed or `repo_root` is not a git working tree. This
    keeps the diagnostic usable on hosts without git (e.g. a B200
    provider that ships only the runtime stack) while still
    producing a deterministic fingerprint when git is available.
    """

    import subprocess

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "git_unavailable"
    if result.returncode != 0:
        return "git_unavailable"
    sha = result.stdout.strip()
    if not _GIT_SHA_RE.match(sha):
        return "git_unavailable"
    return sha


def _verify_commit_sha_resolves(
    sha: str, repo_root: Path | None = None,
) -> bool:
    """Return True iff `git cat-file -t <sha>` returns `commit`.

    Returns True (skipping the check) when git is not available
    OR when the SHA is the `git_unavailable` sentinel; the caller
    then relies on the regex pattern check alone.
    """

    if sha == "git_unavailable":
        return True

    import subprocess
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", sha],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return True
    if result.returncode != 0:
        return False
    return result.stdout.strip() == "commit"


_DIAGNOSTIC_MODULE_PATH_IN_TREE: str = (
    "tools/sol_baselines_harness/calibration/cache_prediction/accuracy_diagnostic.py"
)


def _verify_diagnostic_module_exists_at_sha(
    sha: str, repo_root: Path | None = None,
) -> bool:
    """Return True iff the diagnostic module file exists in the
    tree at the given commit SHA.

    Runs `git cat-file -e <sha>:<module_path>`. Returns True
    (skipping the check) when git is unavailable OR when the
    SHA is the `git_unavailable` sentinel.
    """

    if sha == "git_unavailable":
        return True

    import subprocess
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}:{_DIAGNOSTIC_MODULE_PATH_IN_TREE}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return True
    return result.returncode == 0


@dataclass(frozen=True)
class DiagnosticMetaTable:
    """The `[meta]` table of the diagnostic accuracy report.

    `implementation_commit_sha` is the commit containing the
    diagnostic code that generated the report (7-40 lowercase hex
    characters). The `report_artifact_commit_sha` field is
    explicitly NOT recorded; downstream verifiers reject reports
    that carry it.

    Two consecutive runs against the same inputs MUST produce
    byte-identical TOML reports modulo `report_generated_at_utc`.
    """

    implementation_commit_sha: str
    ledger_path: str
    ledger_sha256: str
    nvbit_so_sha256: str
    gpu_arch: str
    gpu_device_name: str
    ncu_metric_version: str
    driver_version: str
    trace_budget_bytes: int
    timeout_seconds: int
    report_generated_at_utc: str
    denominator_floor: float
    accuracy_threshold: float
    gate_compound: bool


def validate_diagnostic_meta_table(
    table: Mapping[str, Any],
) -> DiagnosticMetaTable:
    """Validate the diagnostic `[meta]` table.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `diagnostic_meta_missing_field`.
      * `diagnostic_meta_invalid_implementation_commit_sha` —
        not a 7-40 lowercase-hex string.
      * `diagnostic_meta_forbidden_report_artifact_commit_sha`
        — the table carries the dropped
        `report_artifact_commit_sha` field.
      * `diagnostic_meta_invalid_numeric_field` — a numeric field
        is non-finite or out of its expected range.
    """

    if "report_artifact_commit_sha" in table:
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_forbidden_report_artifact_commit_sha",
            f"diagnostic meta table carries the dropped "
            f"`report_artifact_commit_sha` field; only "
            f"`implementation_commit_sha` is recorded.",
        )

    for field in _DIAGNOSTIC_META_REQUIRED_FIELDS:
        if field not in table:
            raise PerProblemRecordSchemaError(
                "diagnostic_meta_missing_field",
                f"diagnostic meta table missing required field "
                f"{field!r}",
            )

    impl_sha = table["implementation_commit_sha"]
    if not isinstance(impl_sha, str):
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_invalid_implementation_commit_sha",
            f"implementation_commit_sha must be a string, got "
            f"{type(impl_sha).__name__!r}",
        )
    if impl_sha == "git_unavailable":
        # Tolerated when git is not available at report-generation
        # time (e.g. a B200 provider that ships only the runtime
        # stack). The sentinel preserves the field's presence for
        # downstream consumers; the report's provenance falls back
        # to the meta table's other fields (ledger_sha256,
        # nvbit_so_sha256, gpu_arch, gpu_device_name).
        pass
    elif _ALL_ZERO_SHA_PATTERN.match(impl_sha):
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_zero_commit_sha",
            f"implementation_commit_sha must not be the all-zero "
            f"placeholder; got {impl_sha!r}. Pass a real SHA via "
            f"build_accuracy_diagnostic_report(implementation_commit_sha=...) "
            f"or rely on the default `git rev-parse HEAD` resolution.",
        )
    elif not _GIT_SHA_RE.match(impl_sha):
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_invalid_implementation_commit_sha",
            f"implementation_commit_sha must be 7-40 lowercase hex "
            f"characters OR the `git_unavailable` sentinel, got "
            f"{impl_sha!r}",
        )
    else:
        # Verify the SHA actually resolves to a commit object
        # via `git cat-file -t`. The helper returns True when
        # git is unavailable, so this check is a no-op on hosts
        # without git. When git IS available, a fake hex SHA
        # like `"fffffff"` will fail this check.
        if not _verify_commit_sha_resolves(impl_sha):
            raise PerProblemRecordSchemaError(
                "diagnostic_meta_commit_sha_unresolved",
                f"implementation_commit_sha {impl_sha!r} does not "
                f"resolve to a commit object via `git cat-file -t`; "
                f"the SHA's hex form is valid but no such commit "
                f"exists in this repository.",
            )
        # Verify the diagnostic module exists in the tree at
        # that commit. A pre-diagnostic commit (e.g. a commit
        # from before `accuracy_diagnostic.py` landed) would
        # resolve but would not have the module file; reject
        # such SHAs so the report cannot claim it was emitted
        # from a commit that physically could not have emitted
        # it.
        if not _verify_diagnostic_module_exists_at_sha(impl_sha):
            raise PerProblemRecordSchemaError(
                "diagnostic_meta_implementation_module_missing_at_sha",
                f"implementation_commit_sha {impl_sha!r} resolves to "
                f"a commit, but the diagnostic module "
                f"{_DIAGNOSTIC_MODULE_PATH_IN_TREE!r} does not exist "
                f"at that commit; the report cannot have been "
                f"emitted by that codebase.",
            )

    denominator_floor = table["denominator_floor"]
    if not _is_finite_float(denominator_floor) or float(denominator_floor) <= 0.0:
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_invalid_numeric_field",
            f"denominator_floor must be a positive finite real, got "
            f"{denominator_floor!r}",
        )

    accuracy_threshold = table["accuracy_threshold"]
    if (
        not _is_finite_float(accuracy_threshold)
        or not (0.0 <= float(accuracy_threshold) <= 1.0)
    ):
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_invalid_numeric_field",
            f"accuracy_threshold must be a finite real in [0, 1], got "
            f"{accuracy_threshold!r}",
        )

    for int_field in ("trace_budget_bytes", "timeout_seconds"):
        value = table[int_field]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise PerProblemRecordSchemaError(
                "diagnostic_meta_invalid_numeric_field",
                f"{int_field} must be a non-negative int, got {value!r}",
            )

    gate_compound = table["gate_compound"]
    if not isinstance(gate_compound, bool):
        raise PerProblemRecordSchemaError(
            "diagnostic_meta_invalid_numeric_field",
            f"gate_compound must be bool, got {gate_compound!r} "
            f"({type(gate_compound).__name__})",
        )

    return DiagnosticMetaTable(
        implementation_commit_sha=impl_sha,
        ledger_path=table["ledger_path"],
        ledger_sha256=table["ledger_sha256"],
        nvbit_so_sha256=table["nvbit_so_sha256"],
        gpu_arch=table["gpu_arch"],
        gpu_device_name=table["gpu_device_name"],
        ncu_metric_version=table["ncu_metric_version"],
        driver_version=table["driver_version"],
        trace_budget_bytes=int(table["trace_budget_bytes"]),
        timeout_seconds=int(table["timeout_seconds"]),
        report_generated_at_utc=table["report_generated_at_utc"],
        denominator_floor=float(denominator_floor),
        accuracy_threshold=float(accuracy_threshold),
        gate_compound=gate_compound,
    )


# ---------------------------------------------------------------------------
# Gate result + typed exception
# ---------------------------------------------------------------------------

class AccuracyGateError(Exception):
    """Typed exception raised by the accuracy diagnostic.

    `typed_reason` is one of:
      * `invalid_threshold` — gate called with threshold outside
        `[0.0, 1.0]`.
      * `invalid_denominator_floor` — gate called with `eps`
        non-positive.
      * `record_missing_field` — gate input row is missing a
        required field.
      * `record_invalid_status` — gate input row carries a
        `status` value outside the four declared statuses.
      * `record_invalid_rel_err` — measured row carries a
        non-finite (`NaN` / `inf` / non-numeric / sentinel
        string) rel-err value.
      * `record_missing_problem_slug` — input row lacks a
        non-empty `problem_slug` string field. The all-235
        completeness check is keyed by `problem_slug`; fallback
        identifiers are NOT used for completeness validation.
      * `record_set_incomplete` — input record set does not
        cover the canonical 235 slugs (missing canonical entries).
      * `record_duplicate_slug` — input contains the same
        canonical slug more than once.
      * `record_unknown_slug` — input contains a slug not in the
        canonical 235 list.
      * `record_arithmetic_inconsistent` — measured row carries
        labels + predictions whose `abs(predicted - label) /
        max(label, eps)` does not match the stored
        `l*_calibrated_rel_err`. The gate uses prediction +
        label as the source of truth and rejects stored values
        that disagree.
      * `canonical_partition_drift` — raised by
        `enumerate_canonical_solbench_problems()` when the source
        directories drift from the canonical partition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class AccuracyDiagnosticResult:
    """Output of `compute_accuracy_diagnostic_gate(...)`.

    `accuracy_gate_passed` is the top-level boolean: True only
    when the input covers the canonical 235 slugs exactly AND
    every record is `status=measured` AND each measured record's
    `l1_calibrated_rel_err` and `l2_calibrated_rel_err` are both
    <= `threshold`.

    `cells_above_10pct_rel_err_names` is the alphabetically
    sorted list of `problem_slug` values that caused the gate
    to fail.

    `failure_categories` groups failing slugs by failure
    category. `record_count` is the total number of records the
    gate considered (post-completeness validation, this equals
    235). `threshold` and `denominator_floor` are echoed for
    determinism + downstream report headers.
    """

    accuracy_gate_passed: bool
    cells_above_10pct_rel_err_names: tuple[str, ...]
    failure_categories: Mapping[str, tuple[str, ...]]
    record_count: int
    threshold: float
    denominator_floor: float


_FAILURE_CATEGORIES: tuple[str, ...] = (
    "measured_over_l1_threshold",
    "measured_over_l2_threshold",
    "typed_unavailable",
    "diagnostic_failed",
    "corpus_expansion_pending",
)


def _validate_record_set_completeness(
    records: Sequence[Mapping[str, Any]],
    canonical_problems: Sequence[CanonicalSolbenchProblem],
) -> None:
    """Confirm `records` covers the canonical 235 slugs exactly.

    Every record MUST carry an explicit non-empty `problem_slug`
    string field; missing or empty values raise
    `record_missing_problem_slug` (fallback identifiers are for
    error messages only, not for completeness validation).
    Raises `AccuracyGateError` with typed reasons
    `record_missing_problem_slug`, `record_duplicate_slug`,
    `record_unknown_slug`, or `record_set_incomplete` on any
    drift. An empty input fires `record_set_incomplete` (235
    missing canonical slugs).
    """

    canonical_slug_set = frozenset(p.problem_slug for p in canonical_problems)
    input_slugs: list[str] = []
    for index, record in enumerate(records):
        slug = record.get("problem_slug")
        if not isinstance(slug, str) or not slug:
            raise AccuracyGateError(
                "record_missing_problem_slug",
                f"record[{index}] missing required non-empty "
                f"'problem_slug' field; got {slug!r}",
            )
        input_slugs.append(slug)

    input_slug_counts: dict[str, int] = {}
    for slug in input_slugs:
        input_slug_counts[slug] = input_slug_counts.get(slug, 0) + 1

    duplicates = sorted(
        slug for slug, count in input_slug_counts.items() if count > 1
    )
    if duplicates:
        raise AccuracyGateError(
            "record_duplicate_slug",
            f"input contains {len(duplicates)} duplicated slug(s): "
            f"{duplicates!r}",
        )

    input_slug_set = frozenset(input_slug_counts.keys())
    unknown = sorted(input_slug_set - canonical_slug_set)
    if unknown:
        raise AccuracyGateError(
            "record_unknown_slug",
            f"input contains {len(unknown)} slug(s) not in the "
            f"canonical 235-problem partition: {unknown!r}",
        )

    missing = sorted(canonical_slug_set - input_slug_set)
    if missing:
        raise AccuracyGateError(
            "record_set_incomplete",
            f"input is missing {len(missing)} canonical slug(s) of "
            f"235; first missing slugs: {missing[:5]!r}",
        )


def compute_accuracy_diagnostic_gate(
    records: Sequence[PerProblemRecord | Mapping[str, Any]],
    *,
    threshold: float = 0.10,
    eps: float = 0.01,
    canonical_problems: Sequence[CanonicalSolbenchProblem] | None = None,
) -> AccuracyDiagnosticResult:
    """Apply the fail-closed accuracy gate to a record set.

    Decision logic:
      * Validate threshold ∈ [0, 1] and eps > 0 (raise
        `AccuracyGateError("invalid_threshold", ...)` or
        `("invalid_denominator_floor", ...)`).
      * Validate that `records` covers the canonical 235 slugs
        exactly: no duplicates, no unknown slugs, no missing
        canonical slugs. Empty input fires
        `record_set_incomplete`.
      * For each record:
          - `status` must be one of the four declared statuses;
            otherwise raise
            `AccuracyGateError("record_invalid_status", ...)`.
          - if `status == measured`, the record MUST carry
            finite `l1_calibrated_rel_err` and
            `l2_calibrated_rel_err` (non-finite values raise
            `AccuracyGateError("record_invalid_rel_err", ...)`).
            The gate marks the record as failing if either
            calibrated rel-err exceeds `threshold`.
          - any non-`measured` status counts as a gate failure.
      * `accuracy_gate_passed` is True only when no record
        contributed any slug to the failure list.

    `canonical_problems` overrides the on-disk enumeration; pass
    a synthetic list to exercise the gate without filesystem I/O.
    Defaults to the result of
    `enumerate_canonical_solbench_problems()`.

    Returns a frozen `AccuracyDiagnosticResult`.
    """

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise AccuracyGateError(
            "invalid_threshold",
            f"threshold must be a real number, got {type(threshold).__name__}",
        )
    if not (0.0 <= float(threshold) <= 1.0):
        raise AccuracyGateError(
            "invalid_threshold",
            f"threshold must be in [0.0, 1.0], got {threshold!r}",
        )

    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise AccuracyGateError(
            "invalid_denominator_floor",
            f"eps must be a real number, got {type(eps).__name__}",
        )
    eps_float = float(eps)
    # `eps <= 0.0` alone accepts `nan` (every nan comparison is
    # False) and `inf` (inf > 0). The denominator-floor contract
    # is positive AND finite; reject non-finite values explicitly
    # so the gate cannot return a `nan`/`inf` floor through to the
    # validator's arithmetic-consistency checks.
    if not math.isfinite(eps_float):
        raise AccuracyGateError(
            "invalid_denominator_floor",
            f"eps must be finite, got {eps!r}",
        )
    if eps_float <= 0.0:
        raise AccuracyGateError(
            "invalid_denominator_floor",
            f"eps must be > 0.0, got {eps!r}",
        )

    if canonical_problems is None:
        canonical_problems = enumerate_canonical_solbench_problems()

    _validate_record_set_completeness(records, canonical_problems)

    failures_by_category: dict[str, list[str]] = {
        category: [] for category in _FAILURE_CATEGORIES
    }
    failing_slug_set: set[str] = set()

    threshold_f = float(threshold)

    for index, record in enumerate(records):
        if "status" not in record:
            raise AccuracyGateError(
                "record_missing_field",
                f"record[{index}] missing required 'status' field",
            )

        status = record["status"]
        if status not in _VALID_STATUSES:
            raise AccuracyGateError(
                "record_invalid_status",
                f"record[{index}] status {status!r} not in declared set "
                f"{sorted(_VALID_STATUSES)!r}",
            )

        slug = _resolve_problem_slug(record, index)

        if status == STATUS_MEASURED:
            for required in (
                "l1_calibrated_rel_err",
                "l2_calibrated_rel_err",
            ):
                if required not in record:
                    raise AccuracyGateError(
                        "record_missing_field",
                        f"record[{index}] (status=measured) missing "
                        f"required field {required!r}",
                    )

            l1_raw_value = record["l1_calibrated_rel_err"]
            l2_raw_value = record["l2_calibrated_rel_err"]
            if not _is_finite_float(l1_raw_value):
                raise AccuracyGateError(
                    "record_invalid_rel_err",
                    f"record[{index}] (status=measured) "
                    f"l1_calibrated_rel_err must be a finite real "
                    f"number, got {l1_raw_value!r}",
                )
            if not _is_finite_float(l2_raw_value):
                raise AccuracyGateError(
                    "record_invalid_rel_err",
                    f"record[{index}] (status=measured) "
                    f"l2_calibrated_rel_err must be a finite real "
                    f"number, got {l2_raw_value!r}",
                )

            l1_rel = float(l1_raw_value)
            l2_rel = float(l2_raw_value)

            # Relative error is mathematically non-negative
            # (abs(predicted - label) / max(label, eps)). A
            # negative finite value indicates an upstream signed-
            # error bug; fail closed instead of treating it as a
            # passing cell (the existing `> threshold` check
            # would otherwise quietly accept it).
            if l1_rel < 0.0:
                raise AccuracyGateError(
                    "record_invalid_rel_err",
                    f"record[{index}] (status=measured) "
                    f"l1_calibrated_rel_err must be >= 0; "
                    f"got {l1_raw_value!r}",
                )
            if l2_rel < 0.0:
                raise AccuracyGateError(
                    "record_invalid_rel_err",
                    f"record[{index}] (status=measured) "
                    f"l2_calibrated_rel_err must be >= 0; "
                    f"got {l2_raw_value!r}",
                )

            # Arithmetic-consistency check: when the row carries
            # the label + prediction + denominator-floor fields,
            # the stored rel-err MUST match the formula
            # `abs(predicted_calibrated - label) /
            # max(label, denominator_floor)`. The synthetic gate
            # fixtures that only carry status + rel-err skip this
            # check, but any 21-field row passes through it.
            effective_l1_rel, effective_l2_rel = (
                _check_gate_arithmetic_consistency_for_measured_row(
                    record=record, index=index, stored_l1_rel=l1_rel,
                    stored_l2_rel=l2_rel, eps=float(eps),
                )
            )

            if effective_l1_rel > threshold_f:
                failures_by_category["measured_over_l1_threshold"].append(slug)
                failing_slug_set.add(slug)
            if effective_l2_rel > threshold_f:
                failures_by_category["measured_over_l2_threshold"].append(slug)
                failing_slug_set.add(slug)
        elif status == STATUS_TYPED_UNAVAILABLE:
            failures_by_category["typed_unavailable"].append(slug)
            failing_slug_set.add(slug)
        elif status == STATUS_DIAGNOSTIC_FAILED:
            failures_by_category["diagnostic_failed"].append(slug)
            failing_slug_set.add(slug)
        elif status == STATUS_CORPUS_EXPANSION_PENDING:
            failures_by_category["corpus_expansion_pending"].append(slug)
            failing_slug_set.add(slug)

    sorted_failures = tuple(sorted(failing_slug_set))
    accuracy_gate_passed = len(sorted_failures) == 0

    failure_categories: dict[str, tuple[str, ...]] = {
        category: tuple(sorted(slugs))
        for category, slugs in failures_by_category.items()
    }

    return AccuracyDiagnosticResult(
        accuracy_gate_passed=accuracy_gate_passed,
        cells_above_10pct_rel_err_names=sorted_failures,
        failure_categories=failure_categories,
        record_count=len(records),
        threshold=threshold_f,
        denominator_floor=float(eps),
    )


def _resolve_problem_slug(
    record: Mapping[str, Any], index: int
) -> str:
    """Return the best available identifier for a per-problem record.

    Prefers `problem_slug`, falls back to `cell_name`, falls back
    to a synthesized `"<unknown_record_{index}>"` sentinel.
    """

    slug = record.get("problem_slug")
    if isinstance(slug, str) and slug:
        return slug
    cell_name = record.get("cell_name")
    if isinstance(cell_name, str) and cell_name:
        return cell_name
    return f"<unknown_record_{index}>"


def _check_gate_arithmetic_consistency_for_measured_row(
    *,
    record: Mapping[str, Any],
    index: int,
    stored_l1_rel: float,
    stored_l2_rel: float,
    eps: float,
) -> tuple[float, float]:
    """Recompute calibrated L1 / L2 relative error from
    `ncu_*_label`, `predicted_*_calibrated`, and a denominator
    floor when those fields are present on the input row.

    Returns ``(effective_l1_rel, effective_l2_rel)``: the rel-err
    values the gate's threshold check should use. When labels +
    predictions are present, this is the formula recomputed at
    the caller's ``eps`` (so a CLI invocation with
    ``--eps X`` actually re-gates at X even when the row was
    generated at a different stored floor). When labels +
    predictions are absent (minimal-shape input), the returned
    values are the caller-supplied stored values unchanged.

    Also raises ``AccuracyGateError("record_arithmetic_
    inconsistent", ...)`` if the stored rel-err disagrees with
    the formula at the ROW's own ``denominator_floor`` field by
    more than ``_ARITHMETIC_TOLERANCE`` (this catches storage
    corruption / hand-edited mismatches). The eps-adjusted
    rel-err is computed independently from the storage-
    consistency check, so a stored value that matches the row's
    own floor passes the consistency check even when ``eps``
    differs from that floor.

    Rows that lack the label/prediction fields skip both
    checks; the gate then thresholds the stored rel-err
    directly (minimal-shape compatibility).
    """

    eps_float = float(eps)
    row_floor_field = record.get("denominator_floor")
    if _is_finite_float(row_floor_field) and float(row_floor_field) > 0.0:
        row_floor = float(row_floor_field)
    else:
        row_floor = eps_float

    effective: dict[str, float] = {
        "l1": stored_l1_rel,
        "l2": stored_l2_rel,
    }

    for cache_level, label_field, pred_field, stored_rel in (
        (
            "l1",
            "ncu_l1_label",
            "predicted_l1_calibrated",
            stored_l1_rel,
        ),
        (
            "l2",
            "ncu_l2_label",
            "predicted_l2_calibrated",
            stored_l2_rel,
        ),
    ):
        if label_field not in record or pred_field not in record:
            # Minimal-shape path: caller supplied only
            # `status` + rel-err (no label / prediction). The
            # arithmetic recompute is skipped intentionally;
            # `effective[cache_level]` remains the stored value.
            continue
        label_value = record[label_field]
        pred_value = record[pred_field]
        # Distinguish "absent" (the case above) from "present
        # but invalid". A row that includes the label/prediction
        # fields but supplies non-finite values (e.g., NaN, a
        # sentinel string) cannot bypass the arithmetic recompute
        # because the caller declared the values exist; reject
        # the row outright so an upstream signed-error / sentinel
        # leak surfaces instead of producing a passing verdict.
        if not _is_finite_float(label_value):
            raise AccuracyGateError(
                "record_invalid_arithmetic_input",
                f"record[{index}] (status=measured) {label_field}="
                f"{label_value!r} is present but not a finite real "
                f"number; cannot recompute "
                f"{cache_level}_calibrated_rel_err",
            )
        if not _is_finite_float(pred_value):
            raise AccuracyGateError(
                "record_invalid_arithmetic_input",
                f"record[{index}] (status=measured) {pred_field}="
                f"{pred_value!r} is present but not a finite real "
                f"number; cannot recompute "
                f"{cache_level}_calibrated_rel_err",
            )
        label = float(label_value)
        prediction = float(pred_value)

        # Storage-consistency check: the stored rel-err must
        # match the formula at the row's OWN stored floor (this
        # catches storage corruption and tamper-edited rows).
        expected_at_row_floor = abs(prediction - label) / max(label, row_floor)
        if abs(stored_rel - expected_at_row_floor) > _ARITHMETIC_TOLERANCE:
            raise AccuracyGateError(
                "record_arithmetic_inconsistent",
                f"record[{index}] (status=measured) stored "
                f"{cache_level}_calibrated_rel_err={stored_rel!r} "
                f"disagrees with formula abs({pred_field}={prediction!r} "
                f"- {label_field}={label!r}) / max({label_field}, "
                f"denominator_floor={row_floor!r}) = "
                f"{expected_at_row_floor!r} by more than "
                f"{_ARITHMETIC_TOLERANCE!r}",
            )

        # Eps-adjusted rel-err: this is the value the gate's
        # threshold check uses. When eps matches the row's
        # floor, this equals the stored value; when eps differs,
        # the threshold check sees the recomputed value so
        # `--eps X` actually re-gates at X.
        effective[cache_level] = abs(prediction - label) / max(label, eps_float)

    return effective["l1"], effective["l2"]


# ---------------------------------------------------------------------------
# Active fail-closed detectors
# ---------------------------------------------------------------------------

# Map from internal category label to the on-disk baselines/
# subdirectory name. Note that `FIB` -> `FIB` (not the
# `FlashInfer-Bench` name the benchmark tree uses); the
# baselines/ tree is the diagnostic's own per-problem hand-port
# directory and uses the short category labels by convention.
_BASELINES_DIR_FOR_CATEGORY: dict[str, str] = {
    CATEGORY_L1: "L1",
    CATEGORY_L2: "L2",
    CATEGORY_QUANT: "Quant",
    CATEGORY_FIB: "FIB",
}


def detect_cu_hand_port_present(
    problem: CanonicalSolbenchProblem,
    baselines_root: Path,
) -> bool:
    """Return True iff a *.cu hand-port exists for the problem.

    The hand-port directory lives at `<baselines_root>/<CATEGORY>/
    <problem_slug>/` where `<problem_slug>` includes the
    `problem_` prefix (matches the on-disk convention documented
    in `baselines/README.md`). A populated directory contains at
    least one `*.cu` file under the `cutlass/` or `cuda_cpp/`
    variant subdirectory.
    """

    category_dir_name = _BASELINES_DIR_FOR_CATEGORY[problem.problem_category]
    candidate = baselines_root / category_dir_name / problem.problem_slug
    if not candidate.is_dir():
        return False
    for entry in candidate.rglob("*.cu"):
        return True
    return False


@dataclass(frozen=True)
class LedgerLookupResult:
    """Result of `detect_ledger_entry_for_problem(...)`.

    `present` is True when the ledger has at least one entry
    whose `cell_name` matches the given slug. When True,
    `ncu_l1_hit_rate` and `ncu_l2_hit_rate` carry the labels
    from the first matching entry (the ledger may carry multiple
    entries per cell with different `launch_skip_count`; the
    diagnostic uses the first match for day-1 reporting).
    """

    present: bool
    ncu_l1_hit_rate: float | None
    ncu_l2_hit_rate: float | None


def detect_ledger_entry_for_problem(
    problem: CanonicalSolbenchProblem,
    ledger_data: Mapping[str, Any],
) -> LedgerLookupResult:
    """Return a `LedgerLookupResult` for the problem.

    The match key is exact: `entry["cell_name"] == problem.problem_slug`.
    Calibration cells whose `cell_name` is a synthetic `calib_*`
    string never match a 235-problem slug; only entries with
    real per-cell labels are considered "present".

    This helper deliberately ignores the ledger's `[[unavailable]]`
    table. Canonical predictor-typed-unavailable cohort cells
    already route through the cohort-first detector chain in
    `_build_per_problem_row` before this lookup fires, so their
    classification does not depend on a ledger-lookup hit. Other
    problems that show up in `[[unavailable]]` (for example a
    hand-ported kernel that requires sm_100 features and could
    not be profiled on the local host) have NO labels usable for
    a rel-err comparison; they fall through to `present=False`
    so the upstream detector classifies them as
    `corpus_expansion_pending / ncu_labels_missing`.
    """

    entries = ledger_data.get("entries", ())
    for entry in entries:
        if entry.get("cell_name") == problem.problem_slug:
            label = entry.get("label", {})
            return LedgerLookupResult(
                present=True,
                ncu_l1_hit_rate=label.get("l1_global_load_hit_rate"),
                ncu_l2_hit_rate=label.get("l2_read_hit_rate"),
            )
    return LedgerLookupResult(
        present=False, ncu_l1_hit_rate=None, ncu_l2_hit_rate=None,
    )


@dataclass(frozen=True)
class ZeroReadEvidenceResult:
    """Result of `detect_zero_read_unavailable_row(...)`.

    `present` is True when the ledger's `[[unavailable]]` table
    contains at least one entry whose `cell_name` matches the
    given slug AND whose `unavailable_reason` is
    `"ncu_label_no_read_only_traffic"`. When True,
    `source_commit` carries the entry's `source_commit` field so
    the diagnostic can preserve the row's provenance trail in
    downstream reports.

    The dataclass is intentionally narrow: this detector is the
    only path that turns a strict-loader-valid no-read
    `[[unavailable]]` row into the `status = diagnostic_failed`
    terminal classification. Other `[[unavailable]]` reasons
    (`probe_binary_missing_for_slug`,
    `partition_typed_unavailable_warm_cold_protocol_unpinned`, …)
    fall through to the existing detector chain.
    """

    present: bool
    source_commit: str | None


def detect_zero_read_unavailable_row(
    problem: CanonicalSolbenchProblem,
    ledger_data: Mapping[str, Any],
) -> ZeroReadEvidenceResult:
    """Return a `ZeroReadEvidenceResult` for the problem.

    Scans `ledger_data["unavailable"]` for entries with
    `cell_name == problem.problem_slug` AND
    `unavailable_reason == "ncu_label_no_read_only_traffic"`.

    The match is exact on both fields. Entries that match the
    cell_name but carry a different `unavailable_reason` (for
    example `probe_binary_missing_for_slug`,
    `partition_typed_unavailable_warm_cold_protocol_unpinned`,
    or any other reason) are deliberately NOT matched here so
    the upstream detector chain can route those by their own
    semantics. NCU's read-only sector hit-rate metrics are
    undefined for write-only kernels (no L1 / L2 reads of read-
    only buffers); the harvest pipeline writes this row when the
    NCU CSV exposes zero in both `l1tex__t_sectors_op_read.sum`
    and `lts__t_sectors_op_read.sum` for the measured launch.

    Returns `present=False` when `ledger_data` lacks the
    `unavailable` key or no matching entry exists.
    """

    unavailable = ledger_data.get("unavailable", ())
    for entry in unavailable:
        if entry.get("cell_name") != problem.problem_slug:
            continue
        if entry.get("unavailable_reason") != "ncu_label_no_read_only_traffic":
            continue
        return ZeroReadEvidenceResult(
            present=True,
            source_commit=entry.get("source_commit"),
        )
    return ZeroReadEvidenceResult(present=False, source_commit=None)


def detect_cohort_classification(
    problem: CanonicalSolbenchProblem,
) -> str | None:
    """Return the canonical `predictor_bucket` for the problem
    if `PREDICTOR_DESIGN_COVERAGE_BY_CELL` contains an entry
    whose `cell_name` matches the problem's slug.

    Returns one of `predictor_friendly`, `predictor_hostile`,
    `predictor_typed_unavailable`, or `None` (not classified).
    """

    entry = PREDICTOR_DESIGN_COVERAGE_BY_CELL.get(problem.problem_slug)
    if entry is None:
        return None
    return entry.predictor_bucket


# ---------------------------------------------------------------------------
# Host-architecture availability manifest (sm_86 carve-out)
# ---------------------------------------------------------------------------

HOST_ARCH_MANIFEST_RELATIVE_PATH: str = (
    "tools/sol_baselines_harness/calibration/cache_prediction/"
    "host_arch_availability_v1.toml"
)


# Canonical in-tree path to the calibration / validation
# split manifest. The diagnostic builder loads it by default
# so the `[overall_accuracy_summary]` block reports
# calibration_only train rel-err and validation_only
# validation rel-err as independent fields.
CALIBRATION_VALIDATION_SPLIT_RELATIVE_PATH: str = (
    "tools/sol_baselines_harness/calibration/cache_prediction/"
    "calibration_validation_split_v1.toml"
)

_HOST_ARCH_STATUS_SM86_COMPATIBLE: str = "sm86_compatible_cohort"
_HOST_ARCH_STATUS_UNSUPPORTED: str = STATUS_HOST_ARCH_UNSUPPORTED

_HOST_ARCH_MANIFEST_STATUSES: frozenset[str] = frozenset({
    _HOST_ARCH_STATUS_SM86_COMPATIBLE,
    _HOST_ARCH_STATUS_UNSUPPORTED,
})


# sm_120 cohort vocabulary. Every row carries a ``sm120_status``
# field whose value is drawn from this set. The
# ``sm120_compat_unverified_cohort`` status admits an opaque "we
# have not yet verified whether this row runs on the CUTLASS
# sm_120 block-scaled MMA path or requires B200-only TCGen05/TMEM"
# state for NVFP4 problems; a follow-up plan flips each unverified
# row into either ``sm120_compatible_cohort`` or
# ``host_arch_unsupported`` with a mechanism-specific typed_reason.
SM_120_STATUS_COMPATIBLE: str = "sm120_compatible_cohort"
SM_120_STATUS_COMPAT_UNVERIFIED: str = "sm120_compat_unverified_cohort"

SM_120_MANIFEST_STATUSES: frozenset[str] = frozenset({
    SM_120_STATUS_COMPATIBLE,
    SM_120_STATUS_COMPAT_UNVERIFIED,
    _HOST_ARCH_STATUS_UNSUPPORTED,
})


@dataclass(frozen=True)
class HostArchAvailabilityEntry:
    """One row of `host_arch_availability_v1.toml`.

    sm_86 side (legacy fields, unchanged):
      Either ``status == sm86_compatible_cohort`` (compiles and
      runs on sm_86 and is expected to land a measured rel-err
      verdict) or ``status == host_arch_unsupported`` with
      ``typed_reason`` from the closed
      ``HOST_ARCH_UNSUPPORTED_TYPED_REASONS`` vocabulary.

    sm_120 side (per-row cohort assignment for the live local
    Blackwell-consumer host):
      Every row carries ``sm120_status`` from
      ``SM_120_MANIFEST_STATUSES``. ``host_arch_unsupported`` /
      ``sm120_compat_unverified_cohort`` rows additionally carry a
      ``sm120_typed_reason`` from
      ``HOST_ARCH_UNSUPPORTED_TYPED_REASONS`` and
      ``SM_120_COHORT_UNVERIFIED_TYPED_REASONS`` respectively.
      ``sm120_compatible_cohort`` rows carry an optional
      ``sm120_cohort_bucket`` (informational) and no
      ``sm120_typed_reason``.

    ``cohort_bucket`` / ``sm120_cohort_bucket`` are informational
    for compatible rows; they suggest the bucket but do not
    override the canonical ``predictor_design_coverage.py``
    partition.
    """

    slug: str
    status: str
    typed_reason: str = ""
    cohort_bucket: str = ""
    notes: str = ""
    sm120_status: str = ""
    sm120_typed_reason: str = ""
    sm120_cohort_bucket: str = ""


def load_host_arch_availability_manifest(
    manifest_path: Path | None,
    *,
    canonical_slugs: frozenset[str] | None = None,
) -> Mapping[str, HostArchAvailabilityEntry] | None:
    """Load the host-arch availability manifest if present.

    Returns `None` when `manifest_path` is `None` OR the file
    does not exist; in that case the detector hook
    `detect_host_arch_unsupported` always returns `None` and the
    detector chain proceeds with its pre-existing branches.

    Validates: TOML parse, schema_version field, every
    `[[problems]]` row carries `slug` + `status`, `status` is in
    `_HOST_ARCH_MANIFEST_STATUSES`, `host_arch_unsupported` rows
    carry a `typed_reason` from `HOST_ARCH_UNSUPPORTED_TYPED_REASONS`,
    `sm86_compatible_cohort` rows carry no `typed_reason`. Duplicate
    slugs raise `host_arch_manifest_duplicate_slug`.

    When `canonical_slugs` is provided, the loader also enforces
    canonical-completeness:

      * Any slug in the manifest that is not in `canonical_slugs`
        is rejected with `host_arch_manifest_non_canonical_slug`.
      * Any slug in `canonical_slugs` that is missing from the
        manifest is rejected with
        `coverage_summary_unclassified_problem`.

    `canonical_slugs=None` skips the completeness check (used by
    unit tests that load a partial manifest). The production path
    via `build_accuracy_diagnostic_report()` always passes the
    canonical set so a real manifest must enumerate every
    canonical problem exactly once.
    """

    if manifest_path is None or not manifest_path.is_file():
        return None
    import tomllib

    raw = manifest_path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    schema_version = data.get("manifest_meta", {}).get("schema_version")
    if schema_version != 1:
        raise PerProblemRecordSchemaError(
            "host_arch_manifest_invalid_schema_version",
            f"host-arch availability manifest at {manifest_path!s} "
            f"must declare [manifest_meta].schema_version = 1, got "
            f"{schema_version!r}",
        )
    problems = data.get("problems", ())
    by_slug: dict[str, HostArchAvailabilityEntry] = {}
    for row in problems:
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug:
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_missing_slug",
                f"host-arch manifest row missing slug field: {row!r}",
            )
        row_status = row.get("status")
        if row_status not in _HOST_ARCH_MANIFEST_STATUSES:
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_invalid_status",
                f"host-arch manifest row for slug {slug!r} has "
                f"status {row_status!r}; expected one of "
                f"{sorted(_HOST_ARCH_MANIFEST_STATUSES)!r}",
            )
        typed_reason = row.get("typed_reason", "")
        cohort_bucket = row.get("cohort_bucket", "")
        if row_status == _HOST_ARCH_STATUS_UNSUPPORTED:
            if typed_reason not in HOST_ARCH_UNSUPPORTED_TYPED_REASONS:
                raise PerProblemRecordSchemaError(
                    "host_arch_typed_reason_vocabulary",
                    f"host-arch manifest row for slug {slug!r} has "
                    f"status=host_arch_unsupported but typed_reason "
                    f"{typed_reason!r} is not in "
                    f"{sorted(HOST_ARCH_UNSUPPORTED_TYPED_REASONS)!r}",
                )
        else:
            if typed_reason:
                raise PerProblemRecordSchemaError(
                    "host_arch_manifest_typed_reason_on_compatible",
                    f"host-arch manifest row for slug {slug!r} has "
                    f"status=sm86_compatible_cohort but carries "
                    f"typed_reason {typed_reason!r}; only "
                    f"host_arch_unsupported rows may carry a reason",
                )

        # sm_120 cohort fields (required for every row).
        sm120_status = row.get("sm120_status")
        if sm120_status is None:
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_missing_sm120_status",
                f"host-arch manifest row for slug {slug!r} is missing "
                f"the required sm120_status field",
            )
        if sm120_status not in SM_120_MANIFEST_STATUSES:
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_invalid_sm120_status",
                f"host-arch manifest row for slug {slug!r} has "
                f"sm120_status {sm120_status!r}; expected one of "
                f"{sorted(SM_120_MANIFEST_STATUSES)!r}",
            )
        sm120_typed_reason = row.get("sm120_typed_reason", "")
        sm120_cohort_bucket = row.get("sm120_cohort_bucket", "")
        if sm120_status == _HOST_ARCH_STATUS_UNSUPPORTED:
            if sm120_typed_reason not in HOST_ARCH_UNSUPPORTED_TYPED_REASONS:
                raise PerProblemRecordSchemaError(
                    "host_arch_manifest_invalid_sm120_typed_reason",
                    f"host-arch manifest row for slug {slug!r} has "
                    f"sm120_status=host_arch_unsupported but "
                    f"sm120_typed_reason {sm120_typed_reason!r} is "
                    f"not in {sorted(HOST_ARCH_UNSUPPORTED_TYPED_REASONS)!r}",
                )
        elif sm120_status == SM_120_STATUS_COMPAT_UNVERIFIED:
            if sm120_typed_reason not in SM_120_COHORT_UNVERIFIED_TYPED_REASONS:
                raise PerProblemRecordSchemaError(
                    "host_arch_manifest_invalid_sm120_typed_reason",
                    f"host-arch manifest row for slug {slug!r} has "
                    f"sm120_status=sm120_compat_unverified_cohort "
                    f"but sm120_typed_reason {sm120_typed_reason!r} "
                    f"is not in {sorted(SM_120_COHORT_UNVERIFIED_TYPED_REASONS)!r}",
                )
        else:
            # sm120_compatible_cohort: typed_reason must NOT be set,
            # but cohort_bucket is permitted (informational).
            if sm120_typed_reason:
                raise PerProblemRecordSchemaError(
                    "host_arch_manifest_sm120_typed_reason_on_compatible",
                    f"host-arch manifest row for slug {slug!r} has "
                    f"sm120_status=sm120_compatible_cohort but "
                    f"carries sm120_typed_reason "
                    f"{sm120_typed_reason!r}; only "
                    f"unsupported/unverified rows may carry one",
                )
        # The blanket ``requires_nvfp4_sm100`` typed reason on the
        # sm_120 side would silently hide whether a row actually
        # uses a B200-only mechanism vs runs on the sm_120 CUTLASS
        # block-scaled MMA path. Reject it explicitly so the
        # narrower vocabulary stays load-bearing.
        if sm120_typed_reason == "requires_nvfp4_sm100":
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_sm120_blanket_nvfp4_reason",
                f"host-arch manifest row for slug {slug!r} uses the "
                f"blanket sm120_typed_reason 'requires_nvfp4_sm100'; "
                f"use a mechanism-specific reason (e.g. "
                f"'requires_tcgen05_sm100a') when the kernel "
                f"requires B200-only mechanisms, or "
                f"'nvfp4_sm120_compat_unverified' when the per-row "
                f"mechanism is not yet verified",
            )
        if slug in by_slug:
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_duplicate_slug",
                f"host-arch manifest contains duplicate slug "
                f"{slug!r}",
            )
        by_slug[slug] = HostArchAvailabilityEntry(
            slug=slug,
            status=row_status,
            typed_reason=typed_reason,
            cohort_bucket=cohort_bucket,
            notes=row.get("notes", ""),
            sm120_status=sm120_status,
            sm120_typed_reason=sm120_typed_reason,
            sm120_cohort_bucket=sm120_cohort_bucket,
        )

    if canonical_slugs is not None:
        # Reject non-canonical slugs first so the error surface
        # points at "you put a wrong slug in" before reporting
        # "you're missing some slugs". Sorting the diff list makes
        # the error message stable across runs.
        non_canonical = sorted(set(by_slug) - canonical_slugs)
        if non_canonical:
            raise PerProblemRecordSchemaError(
                "host_arch_manifest_non_canonical_slug",
                f"host-arch manifest at {manifest_path!s} carries "
                f"{len(non_canonical)} slug(s) not in the canonical "
                f"235-problem partition: {non_canonical!r}",
            )
        missing = sorted(canonical_slugs - set(by_slug))
        if missing:
            raise PerProblemRecordSchemaError(
                "coverage_summary_unclassified_problem",
                f"host-arch manifest at {manifest_path!s} is missing "
                f"{len(missing)} canonical problem(s): "
                f"{missing!r}. Every canonical problem must be "
                f"classified as either sm86_compatible_cohort or "
                f"host_arch_unsupported.",
            )
    return by_slug


def detect_host_arch_unsupported(
    problem: CanonicalSolbenchProblem,
    host_arch_manifest: Mapping[str, HostArchAvailabilityEntry] | None,
) -> str | None:
    """Return the `host_arch_unsupported` typed_reason for the
    problem, OR `None` when the problem is either absent from the
    manifest or recorded as sm_86-compatible.

    Pure function: no I/O. The manifest itself is read once by
    `load_host_arch_availability_manifest` and threaded through
    the detector chain. A `None` manifest means the carve-out is
    not yet authored on this host; the detector then always
    returns `None`, and the existing detector chain proceeds
    unchanged.
    """

    if host_arch_manifest is None:
        return None
    entry = host_arch_manifest.get(problem.problem_slug)
    if entry is None:
        return None
    if entry.status != _HOST_ARCH_STATUS_UNSUPPORTED:
        return None
    return entry.typed_reason


_PREDICTOR_BUCKET_HOST_ARCH_UNSUPPORTED: str = (
    "predictor_host_arch_unsupported"
)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

_PREDICTOR_BUCKET_FRIENDLY: str = "predictor_friendly"
_PREDICTOR_BUCKET_HOSTILE: str = "predictor_hostile"


@dataclass(frozen=True)
class MeasuredCohortPayloadRow:
    """A serialized per-cell calibration cohort measurement.

    The accuracy diagnostic builds `cohort_cell_payloads` from
    the 226-cell predictor partition at runtime; the measured
    subset of those payloads is serialized as
    `[[measured_cohort_payloads]]` rows so the parsed report
    carries enough data to recompute the calibration_only
    split-aware aggregates without re-running the predictor.
    Only the fields needed for split-aware recomputation are
    persisted: `cell_name` (matches the split manifest's
    by_slug key) and the two calibrated rel-err values.
    """

    cell_name: str
    l1_calibrated_rel_err: float
    l2_calibrated_rel_err: float


@dataclass(frozen=True)
class DiagnosticReport:
    """The full diagnostic report.

    Aggregates the emitted tables. `per_cell_rows` is a
    tuple of 235 frozen `PerProblemRecordSchema` instances (one
    per canonical problem). `cohort_summary_rows` is exactly six
    (the cohort cross-product). `overall_summary`,
    `coverage_summary`, `meta`, and `denominator_audit_rows` are
    the originally landed tables; `denominator_box` and
    `overall_user_target_summary` are added by the user-target
    schema migration to carry the source-of-truth count surface
    and the user-acceptance gate aggregation respectively.
    `measured_cohort_payloads` carries the serialized per-cell
    calibration-cohort measurements (the subset of the runtime
    `cohort_cell_payloads` whose status is `measured`); the
    diagnostic validator uses this list to recompute the
    `calibration_only_*` split-aware fields and reject
    tampered values.
    """

    per_cell_rows: tuple[PerProblemRecordSchema, ...]
    cohort_summary_rows: tuple[CohortAccuracySummaryRow, ...]
    overall_summary: OverallAccuracySummary
    coverage_summary: CoverageSummaryTable
    denominator_audit_rows: tuple[DenominatorAuditRow, ...]
    meta: DiagnosticMetaTable
    denominator_box: DenominatorBox
    overall_user_target_summary: OverallUserTargetSummary
    measured_cohort_payloads: tuple[MeasuredCohortPayloadRow, ...] = ()


# Default predictor callback used when the caller does not
# provide one. Emits `diagnostic_failed` with
# `sidecar_parse_failed` for every admitted cell — the honest
# day-1 state when the Tier-2 sidecar / predictor wiring has not
# yet been hooked into the report builder.
def _default_predictor_callback(
    problem: CanonicalSolbenchProblem,
    cohort_bucket: str,
    ledger_lookup: LedgerLookupResult,
) -> Mapping[str, Any]:
    """Return per-cell predictor outputs for an admitted cell.

    The default returns a `status=diagnostic_failed` /
    `typed_reason=sidecar_parse_failed` record because the
    builder has not yet been wired to the deployed predictor.
    Callers supply a callback that returns either:
      * a `measured` payload with the 10 numeric fields plus a
        `denominator_floor` value, OR
      * a `diagnostic_failed` payload with `typed_reason`, OR
      * a `typed_unavailable` payload with `typed_reason`.
    """

    return {
        "status": STATUS_DIAGNOSTIC_FAILED,
        "typed_reason": "sidecar_parse_failed",
    }


def predictor_callback_from_smoke_records(
    smoke_records: Sequence[Any],
    *,
    ledger_data: Mapping[str, Any] | None = None,
):
    """Return a `predictor_callback` for `build_accuracy_diagnostic_report`
    that consumes the smoke harness's `CapturedSmokeRecord`
    outputs.

    The callback maps `problem.problem_slug` to a smoke record
    by `record.cell_name == problem.problem_slug`. When a record
    is found AND it carries non-None calibrated predictions, the
    callback returns a `measured` payload populated from the
    record's fields. Otherwise the callback returns a
    `diagnostic_failed` payload with a typed reason derived from
    the record's status and the ledger join-key state:
      * No matching record -> `ledger_entry_missing`.
      * `record.status == "MEASURED"` AND all 6 prediction +
        label fields populated -> `measured` payload (carries
        Tier-2 sector totals when the smoke record exposes them).
      * `record.status == "MEASURED"` but the record's
        `binary_sha256` provenance disagrees with the ledger
        entry's -> `join_key_drift`.
      * `record.status == "TYPED_UNAVAILABLE"` ->
        `typed_unavailable`.
      * Any other case (status=FAILED, missing prediction
        fields) -> `sidecar_parse_failed`.

    `ledger_data` is the canonical label-ledger TOML payload
    (the same dict that gets passed to `build_accuracy_diagnostic_report`).
    When provided, the callback runs the join-key drift detector;
    when omitted, join_key_drift detection is silent (the
    callback can't compare against an unprovided source of
    truth).
    """

    refuse_legacy_ncu_calibration('accuracy_diagnostic.predictor_callback_from_smoke_records')
    records_by_cell: dict[str, Any] = {}
    for record in smoke_records:
        records_by_cell[record.cell_name] = record

    # Ledger join-key lookup: per cell, capture BOTH
    # `binary_sha256` and `kernel_signature_sha256` (when the
    # ledger entry exposes them). The callback compares a smoke
    # record's same-named fields against the ledger; any
    # mismatch on either field fires `join_key_drift`.
    ledger_join_keys_by_cell: dict[str, dict[str, str]] = {}
    if ledger_data is not None:
        for entry in ledger_data.get("entries", ()):
            cell_name = entry.get("cell_name")
            if not cell_name:
                continue
            keys: dict[str, str] = {}
            binary_sha = entry.get("binary_sha256")
            if binary_sha:
                keys["binary_sha256"] = binary_sha
            kernel_sha = entry.get("kernel_signature_sha256")
            if kernel_sha:
                keys["kernel_signature_sha256"] = kernel_sha
            if keys:
                ledger_join_keys_by_cell[cell_name] = keys

    def _callback(
        problem: CanonicalSolbenchProblem,
        cohort_bucket: str,
        ledger_lookup: LedgerLookupResult,
    ) -> Mapping[str, Any]:
        record = records_by_cell.get(problem.problem_slug)
        if record is None:
            return {
                "status": STATUS_DIAGNOSTIC_FAILED,
                "typed_reason": "ledger_entry_missing",
            }

        # The smoke record's `status` enum uses the deployed-scope
        # vocabulary. Real `CapturedSmokeRecord` instances carry
        # the canonical lowercase values (`measured`,
        # `typed_unavailable`, `failed`) emitted by the smoke
        # producer; older test scaffolds passed the uppercase
        # variants via `types.SimpleNamespace`. Compare in upper
        # case so both shapes route correctly to the same
        # diagnostic branch.
        record_status = (getattr(record, "status", "") or "").upper()
        if record_status == "MEASURED" and all(
            getattr(record, f, None) is not None
            for f in (
                "predicted_l1_hit_rate", "predicted_l2_hit_rate",
                "predicted_l1_calibrated", "predicted_l2_calibrated",
                "ncu_l1_hit_rate", "ncu_l2_hit_rate",
            )
        ):
            # Active join_key_drift detector: when the ledger
            # exposes a join-key (binary_sha256 OR
            # kernel_signature_sha256) for this cell AND the
            # smoke record exposes the same-named field, the
            # two must match. Mismatch on EITHER field fires
            # `join_key_drift`.
            ledger_keys = ledger_join_keys_by_cell.get(
                problem.problem_slug, {},
            )
            for join_key_field in (
                "binary_sha256",
                "kernel_signature_sha256",
            ):
                ledger_value = ledger_keys.get(join_key_field)
                record_value = getattr(record, join_key_field, None)
                if (
                    ledger_value is not None
                    and record_value is not None
                    and ledger_value != record_value
                ):
                    return {
                        "status": STATUS_DIAGNOSTIC_FAILED,
                        "typed_reason": "join_key_drift",
                    }
            # Use the ledger lookup's labels (the same ledger the
            # diagnostic meta records in `[meta].ledger_path` /
            # `ledger_sha256`) as the canonical NCU label for the
            # rel-err computation. The smoke record's local copy
            # may be stale if the ledger was updated between the
            # smoke run and the diagnostic run. Fall back to the
            # record's value only if the ledger lookup has no
            # label (shouldn't happen for entries in `[[entries]]`
            # but defensive).
            ncu_l1 = ledger_lookup.ncu_l1_hit_rate
            ncu_l2 = ledger_lookup.ncu_l2_hit_rate
            if ncu_l1 is None:
                ncu_l1 = record.ncu_l1_hit_rate
            if ncu_l2 is None:
                ncu_l2 = record.ncu_l2_hit_rate
            payload = {
                "status": STATUS_MEASURED,
                "ncu_l1_label": ncu_l1,
                "ncu_l2_label": ncu_l2,
                "predicted_l1_raw": record.predicted_l1_hit_rate,
                "predicted_l2_raw": record.predicted_l2_hit_rate,
                "predicted_l1_calibrated": record.predicted_l1_calibrated,
                "predicted_l2_calibrated": record.predicted_l2_calibrated,
            }
            # Carry the smoke record's Tier-2 sector totals into
            # the payload so the report builder's denominator
            # audit can compute the Tier-2-vs-NCU rel-gap from
            # real captured counts instead of zero placeholders.
            l1_total = getattr(record, "l1_sectors_total", None)
            l2_total = getattr(record, "l2_sectors_total", None)
            if l1_total is not None:
                payload["tier2_l1_total"] = float(l1_total)
            if l2_total is not None:
                payload["tier2_l2_total"] = float(l2_total)
            return payload
        if record_status == "TYPED_UNAVAILABLE":
            # The cell's bucket is typed_unavailable but we
            # reach this callback only for friendly/hostile
            # cells; route through typed_unavailable so the row
            # status is honest. The typed_reason is best-effort
            # from the canonical entry; fall back to
            # cache_geometry_unresolved.
            canonical_entry = PREDICTOR_DESIGN_COVERAGE_BY_CELL.get(
                problem.problem_slug,
            )
            typed_reason = (
                canonical_entry.typed_unavailable_reason
                if canonical_entry and canonical_entry.typed_unavailable_reason
                else "cache_geometry_unresolved"
            )
            return {
                "status": STATUS_TYPED_UNAVAILABLE,
                "typed_reason": typed_reason,
            }
        # Smoke record status is FAILED OR the MEASURED branch
        # bailed because some prediction field was None. Both
        # cases map to a real `sidecar_parse_failed` diagnostic
        # row (the smoke-side measurement path failed to produce
        # all required fields). This is the genuine
        # `sidecar_parse_failed` detector path, not a default
        # stub.
        return {
            "status": STATUS_DIAGNOSTIC_FAILED,
            "typed_reason": "sidecar_parse_failed",
        }

    return _callback


def serialize_diagnostic_sections_for_smoke_report(
    report: DiagnosticReport,
) -> str:
    """Render the diagnostic-specific TOML sections so they can
    be appended to an existing smoke-report TOML.

    The output starts with a `# --- Accuracy diagnostic sections ---`
    delimiter so a downstream operator can find the boundary
    visually. The same sections are also re-loadable by
    `parse_diagnostic_report` when extracted from the combined
    file.
    """

    parts: list[str] = []
    parts.append("\n# --- Accuracy diagnostic sections ---\n")
    parts.append(_format_table(
        "accuracy_diagnostic_meta", _dataclass_to_dict(report.meta),
    ))
    parts.append(_format_table(
        "accuracy_diagnostic_coverage_summary",
        _dataclass_to_dict(report.coverage_summary),
    ))
    parts.append(_format_table(
        "accuracy_diagnostic_overall_summary",
        _dataclass_to_dict(report.overall_summary),
    ))
    parts.append(_format_table(
        "accuracy_diagnostic_overall_user_target_summary",
        _dataclass_to_dict(report.overall_user_target_summary),
    ))
    parts.append(_format_table(
        "accuracy_diagnostic_denominator_box",
        _dataclass_to_dict(report.denominator_box),
    ))
    parts.append(_format_array_of_tables(
        "accuracy_diagnostic_per_cell",
        [_dataclass_to_dict(r) for r in report.per_cell_rows],
    ))
    parts.append(_format_array_of_tables(
        "accuracy_diagnostic_cohort_summary",
        [_dataclass_to_dict(r) for r in report.cohort_summary_rows],
    ))
    parts.append(_format_array_of_tables(
        "accuracy_diagnostic_denominator_audit",
        [_dataclass_to_dict(r) for r in report.denominator_audit_rows],
    ))
    parts.append(_format_array_of_tables(
        "accuracy_diagnostic_measured_cohort_payloads",
        [_dataclass_to_dict(r) for r in report.measured_cohort_payloads],
    ))
    return "\n".join(parts)


def _build_per_problem_row(
    problem: CanonicalSolbenchProblem,
    *,
    baselines_root: Path,
    ledger_data: Mapping[str, Any],
    denominator_floor: float,
    predictor_callback,
    host_arch_manifest: (
        Mapping[str, HostArchAvailabilityEntry] | None
    ) = None,
) -> dict:
    """Decide one canonical problem's row.

    Detector chain (priority order; host-arch availability is
    checked first because problems whose kernels cannot execute
    on the local sm_86 host are carved out of the user-acceptance
    denominator BEFORE any cohort / hand-port / ledger lookup
    fires):
      1. Host-arch manifest classifies the problem as
         host_arch_unsupported -> host_arch_unsupported row with
         a typed_reason from
         ``HOST_ARCH_UNSUPPORTED_TYPED_REASONS``.
      2. Cohort = predictor_typed_unavailable -> typed_unavailable
         row with the canonical typed_reason from
         ``PREDICTOR_DESIGN_COVERAGE_BY_CELL`` (fallback
         ``cache_geometry_unresolved``). This branch fires for the
         canonical predictor-typed-unavailable problem slugs even
         when their hand-port is absent, because the predictor
         abstains on these cells by design and the report must
         record that abstention rather than a corpus-expansion
         prerequisite failure.
      3. *.cu hand-port missing -> corpus_expansion_pending /
         cu_hand_port_missing.
      4. Hand-port exists but no ledger entry ->
         corpus_expansion_pending / ncu_labels_missing.
      5. Hand-port + ledger exist but no cohort classification ->
         corpus_expansion_pending / cohort_classification_missing.
      6. Cohort = friendly / hostile -> call predictor_callback.
         The callback's return decides the final row (measured /
         diagnostic_failed / typed_unavailable-non-canonical).
    """

    if predictor_callback is not None:
        refuse_legacy_ncu_calibration('accuracy_diagnostic._build_per_problem_row.predictor_callback')
    row: dict[str, Any] = {
        "problem_slug": problem.problem_slug,
        "problem_category": problem.problem_category,
        "cell_name": "",
        "predictor_bucket": "",
        "ncu_l1_label": NAN_SENTINEL,
        "ncu_l2_label": NAN_SENTINEL,
        "predicted_l1_raw": NAN_SENTINEL,
        "predicted_l2_raw": NAN_SENTINEL,
        "predicted_l1_calibrated": NAN_SENTINEL,
        "predicted_l2_calibrated": NAN_SENTINEL,
        "l1_abs_pp_error": NAN_SENTINEL,
        "l2_abs_pp_error": NAN_SENTINEL,
        "l1_calibrated_abs_pp_error": NAN_SENTINEL,
        "l2_calibrated_abs_pp_error": NAN_SENTINEL,
        "l1_raw_rel_err": NAN_SENTINEL,
        "l2_raw_rel_err": NAN_SENTINEL,
        "l1_calibrated_rel_err": NAN_SENTINEL,
        "l2_calibrated_rel_err": NAN_SENTINEL,
        "denominator_floor": denominator_floor,
        "status": "",
        "typed_reason": "",
        # Non-measured default; the measured branch overwrites
        # both booleans and the threshold via
        # `compute_user_target_verdict_fields` after the rel-err
        # fields are populated. Every non-measured row remains
        # False on both verdicts and `NAN_SENTINEL` on the
        # threshold by construction.
        "pass_user_target_threshold": False,
        "pass_tiered_gate": False,
        "effective_threshold_pct": NAN_SENTINEL,
    }

    # Host-architecture exclusions fire FIRST. A problem the local
    # sm_86 host cannot compile/run is exempt from the user-
    # acceptance numerator AND denominator; recording it before
    # the cohort / hand-port / ledger checks keeps the exemption
    # from being shadowed by upstream prerequisite gaps. When the
    # manifest is absent (no carve-out authored yet) the detector
    # returns None and the chain proceeds unchanged.
    host_arch_reason = detect_host_arch_unsupported(
        problem, host_arch_manifest,
    )
    if host_arch_reason is not None:
        row["cell_name"] = problem.problem_slug
        row["predictor_bucket"] = _PREDICTOR_BUCKET_HOST_ARCH_UNSUPPORTED
        row["status"] = STATUS_HOST_ARCH_UNSUPPORTED
        row["typed_reason"] = host_arch_reason
        return row

    # Canonical predictor-typed-unavailable cells fail closed
    # regardless of hand-port / ledger state. Problem rows in the
    # canonical typed-unavailable cohort emit
    # `status=typed_unavailable`, NOT `corpus_expansion_pending`,
    # even when their *.cu hand-port is absent. Resolve the
    # cohort classification BEFORE the corpus-expansion
    # prerequisite checks so an absent hand-port does not shadow
    # the typed-unavailable reason.
    cohort_bucket = detect_cohort_classification(problem)
    if cohort_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        canonical_entry = PREDICTOR_DESIGN_COVERAGE_BY_CELL[problem.problem_slug]
        row["cell_name"] = problem.problem_slug
        row["predictor_bucket"] = PREDICTOR_BUCKET_TYPED_UNAVAILABLE
        row["status"] = STATUS_TYPED_UNAVAILABLE
        row["typed_reason"] = (
            canonical_entry.typed_unavailable_reason
            or "cache_geometry_unresolved"
        )
        return row

    if not detect_cu_hand_port_present(problem, baselines_root):
        row["status"] = STATUS_CORPUS_EXPANSION_PENDING
        row["typed_reason"] = "cu_hand_port_missing"
        row["predictor_bucket"] = "predictor_corpus_pending"
        return row

    # Zero-read NCU evidence: an admitted sm_86 kernel whose hand-
    # port exists AND whose ledger has a strict-loader-valid
    # `[[unavailable]]` row with reason
    # `ncu_label_no_read_only_traffic` is a TERMINAL calibration
    # outcome, not a corpus-pending prerequisite gap. NCU's read-
    # only sector hit-rate metrics are undefined for write-only
    # kernels. Emit `status = diagnostic_failed` with the matching
    # typed reason and the entry's `cell_name`. The cohort bucket
    # comes from the canonical partition when present (so the
    # diagnostic preserves the predictor_friendly / hostile
    # assignment), with a safe `"predictor_diagnostic_failed"`
    # fallback when the cohort partition has not yet recorded the
    # slug.
    zero_read = detect_zero_read_unavailable_row(problem, ledger_data)
    if zero_read.present:
        row["cell_name"] = problem.problem_slug
        cohort_for_row = detect_cohort_classification(problem)
        row["predictor_bucket"] = (
            cohort_for_row
            if cohort_for_row is not None
            else "predictor_diagnostic_failed"
        )
        row["status"] = STATUS_DIAGNOSTIC_FAILED
        row["typed_reason"] = "ncu_label_no_read_only_traffic"
        return row

    ledger_lookup = detect_ledger_entry_for_problem(problem, ledger_data)
    if not ledger_lookup.present:
        row["status"] = STATUS_CORPUS_EXPANSION_PENDING
        row["typed_reason"] = "ncu_labels_missing"
        row["predictor_bucket"] = "predictor_corpus_pending"
        return row

    if cohort_bucket is None:
        row["status"] = STATUS_CORPUS_EXPANSION_PENDING
        row["typed_reason"] = "cohort_classification_missing"
        row["predictor_bucket"] = "predictor_corpus_pending"
        return row

    row["cell_name"] = problem.problem_slug
    row["predictor_bucket"] = cohort_bucket

    # cohort_bucket is friendly or hostile. Defer to the
    # predictor callback for the actual measurement.
    payload = predictor_callback(problem, cohort_bucket, ledger_lookup)
    if not isinstance(payload, Mapping):
        raise AccuracyGateError(
            "record_invalid_status",
            f"predictor_callback for {problem.problem_slug!r} returned "
            f"a non-mapping {type(payload).__name__!r}",
        )
    status = payload.get("status")
    if status == STATUS_MEASURED:
        # The callback must supply the 10 numeric fields; the
        # builder computes the abs-pp-error and rel-err pairs.
        ncu_l1 = float(payload["ncu_l1_label"])
        ncu_l2 = float(payload["ncu_l2_label"])
        pred_l1_raw = float(payload["predicted_l1_raw"])
        pred_l2_raw = float(payload["predicted_l2_raw"])
        pred_l1_cal = float(payload["predicted_l1_calibrated"])
        pred_l2_cal = float(payload["predicted_l2_calibrated"])
        l1_cal_rel_err = abs(pred_l1_cal - ncu_l1) / max(ncu_l1, denominator_floor)
        l2_cal_rel_err = abs(pred_l2_cal - ncu_l2) / max(ncu_l2, denominator_floor)
        row.update({
            "ncu_l1_label": ncu_l1,
            "ncu_l2_label": ncu_l2,
            "predicted_l1_raw": pred_l1_raw,
            "predicted_l2_raw": pred_l2_raw,
            "predicted_l1_calibrated": pred_l1_cal,
            "predicted_l2_calibrated": pred_l2_cal,
            "l1_abs_pp_error": abs(pred_l1_raw - ncu_l1) * 100.0,
            "l2_abs_pp_error": abs(pred_l2_raw - ncu_l2) * 100.0,
            "l1_calibrated_abs_pp_error": abs(pred_l1_cal - ncu_l1) * 100.0,
            "l2_calibrated_abs_pp_error": abs(pred_l2_cal - ncu_l2) * 100.0,
            "l1_raw_rel_err": abs(pred_l1_raw - ncu_l1) / max(ncu_l1, denominator_floor),
            "l2_raw_rel_err": abs(pred_l2_raw - ncu_l2) / max(ncu_l2, denominator_floor),
            "l1_calibrated_rel_err": l1_cal_rel_err,
            "l2_calibrated_rel_err": l2_cal_rel_err,
            "status": STATUS_MEASURED,
            "typed_reason": "",
        })
        # Compute the per-row user-acceptance + tiered-gate
        # verdicts from the freshly populated rel-err fields.
        # The validator independently recomputes and matches.
        row.update(compute_user_target_verdict_fields(
            status=STATUS_MEASURED,
            predictor_bucket=row["predictor_bucket"],
            l1_calibrated_rel_err=l1_cal_rel_err,
            l2_calibrated_rel_err=l2_cal_rel_err,
        ))
        # Carry Tier-2 sector totals (if the callback supplied
        # them) into an out-of-schema field that the denominator
        # audit builder consumes downstream. Stripped before
        # validation/serialization.
        if "tier2_l1_total" in payload:
            row["_tier2_l1_total"] = float(payload["tier2_l1_total"])
        if "tier2_l2_total" in payload:
            row["_tier2_l2_total"] = float(payload["tier2_l2_total"])
        return row
    if status == STATUS_DIAGNOSTIC_FAILED:
        row["status"] = STATUS_DIAGNOSTIC_FAILED
        row["typed_reason"] = payload.get("typed_reason", "sidecar_parse_failed")
        return row
    if status == STATUS_TYPED_UNAVAILABLE:
        # The predictor abstained on an admitted friendly/hostile
        # problem. The canonical typed-unavailable cohort is
        # already routed before the predictor callback fires, so
        # any typed-unavailable payload reaching here is a
        # non-canonical abstention. `validate_per_problem_record`
        # restricts `status=typed_unavailable` to the canonical
        # cohort; emit the row as `diagnostic_failed` with the
        # `predictor_typed_unavailable_non_canonical` reason from
        # the existing `ACCURACY_DIAGNOSTIC_TYPED_REASONS`
        # vocabulary so the row validates and the diagnostic
        # sections still emit.
        row["status"] = STATUS_DIAGNOSTIC_FAILED
        row["typed_reason"] = "predictor_typed_unavailable_non_canonical"
        return row
    raise AccuracyGateError(
        "record_invalid_status",
        f"predictor_callback for {problem.problem_slug!r} returned "
        f"unrecognized status {status!r}",
    )


def _build_denominator_audit_rows(
    *,
    per_cell_rows: Sequence[Mapping[str, Any]],
    ledger_data: Mapping[str, Any],
) -> list[dict]:
    """Build `[[denominator_audit]]` rows for measured cells.

    Compares the Tier-2 sidecar L1/L2 sector totals against the
    ledger's sector totals. Emits one row per measured cell with
    `cell_name`, totals, rel-gaps, and `audit_passed`. When the
    Tier-2 sidecar totals are not available (no sidecar attached
    to the predictor callback), the audit emits the row with
    audit_passed=False AND rel-gaps populated from the ledger's
    own totals — the gap cannot be meaningfully computed without
    Tier-2 input, so the row reports `nan` placeholders.

    For the day-1 baseline (no sidecar wiring yet), this function
    emits audit rows for the small subset of measured cells with
    `tier2_l1_total = 0` (sentinel) and `audit_passed = False`
    via the alternate "no Tier-2 data" code path. The downstream
    audit_warning linkage on per-cell rows is also driven from
    these audit rows.
    """

    # Build a quick lookup `cell_name` -> ledger entry for sector
    # totals.
    ledger_by_cell: dict[str, Mapping[str, Any]] = {}
    for entry in ledger_data.get("entries", ()):
        ledger_by_cell[entry["cell_name"]] = entry

    audit_rows: list[dict] = []
    for row in per_cell_rows:
        if row.get("status") != STATUS_MEASURED:
            continue
        cell_name = row.get("cell_name", row.get("problem_slug", ""))
        ledger_entry = ledger_by_cell.get(cell_name)
        if ledger_entry is None:
            continue
        label = ledger_entry.get("label", {})
        ncu_l1_total = float(label.get("l1_global_load_sectors_total", 0.0))
        ncu_l2_total = float(label.get("l2_read_sectors_total", 0.0))
        # Tier-2 totals come from the predictor callback's
        # payload (when the smoke record carried `l1_sectors_total`
        # / `l2_sectors_total`). The row's `_tier2_l1_total` /
        # `_tier2_l2_total` are stripped before schema
        # validation but consumed here.
        tier2_l1_total = float(row.get("_tier2_l1_total", 0.0))
        tier2_l2_total = float(row.get("_tier2_l2_total", 0.0))
        # Rel-gap = abs(tier2 - ncu) / max(ncu, 1) (avoid div by 0).
        l1_gap = (
            abs(tier2_l1_total - ncu_l1_total) / max(ncu_l1_total, 1.0)
        )
        l2_gap = (
            abs(tier2_l2_total - ncu_l2_total) / max(ncu_l2_total, 1.0)
        )
        audit_passed = (
            l1_gap <= _DENOMINATOR_AUDIT_GAP_THRESHOLD
            and l2_gap <= _DENOMINATOR_AUDIT_GAP_THRESHOLD
        )
        audit_rows.append({
            "cell_name": cell_name,
            "tier2_l1_total": tier2_l1_total,
            "ncu_l1_total": ncu_l1_total,
            "tier2_l2_total": tier2_l2_total,
            "ncu_l2_total": ncu_l2_total,
            "l1_total_rel_gap": l1_gap,
            "l2_total_rel_gap": l2_gap,
            "audit_passed": audit_passed,
        })

    return audit_rows


def _strip_audit_input_fields(per_cell_rows: list[dict]) -> None:
    """Strip the internal `_tier2_l*_total` fields from per-cell
    rows after the denominator audit consumes them. The row
    schema validator does not recognize these fields.
    """

    for row in per_cell_rows:
        row.pop("_tier2_l1_total", None)
        row.pop("_tier2_l2_total", None)


def _attach_audit_warning_to_per_cell_rows(
    *,
    per_cell_rows: list[dict],
    denominator_audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    """For per-cell rows whose corresponding denominator-audit
    row failed, set `audit_warning` to the list of fields with
    excess gap. Modifies `per_cell_rows` in place.

    Per DEC-4, the audit is non-blocking on the accuracy gate;
    `audit_warning` is purely a flag for operator inspection
    and does NOT affect `accuracy_gate_passed`.
    """

    audit_by_cell: dict[str, Mapping[str, Any]] = {}
    for audit_row in denominator_audit_rows:
        audit_by_cell[audit_row["cell_name"]] = audit_row

    for row in per_cell_rows:
        if row.get("status") != STATUS_MEASURED:
            continue
        cell_name = row.get("cell_name", "")
        audit_row = audit_by_cell.get(cell_name)
        if audit_row is None:
            continue
        warnings: list[str] = []
        if audit_row["l1_total_rel_gap"] > _DENOMINATOR_AUDIT_GAP_THRESHOLD:
            warnings.append("l1_total_rel_gap")
        if audit_row["l2_total_rel_gap"] > _DENOMINATOR_AUDIT_GAP_THRESHOLD:
            warnings.append("l2_total_rel_gap")
        if warnings:
            row["audit_warning"] = warnings


def _build_cohort_cell_payloads(
    *,
    ledger_data: Mapping[str, Any],
    denominator_floor: float,
    predictor_callback,
) -> list[dict]:
    """Walk the 36-cell calibration partition and produce one
    payload per cell for the cohort summary aggregator.

    For each cell in `PREDICTOR_DESIGN_COVERAGE_BY_CELL`:
      * typed_unavailable cells emit `status=typed_unavailable`.
      * friendly/hostile cells call `predictor_callback` with a
        synthetic `CanonicalSolbenchProblem` constructed from
        the cell_name (the callback receives the cell-name slug
        so it can route to the right ledger entry).

    Returns a list of payload dicts ready for
    `_aggregate_cohort_rows`.
    """

    if predictor_callback is not None:
        refuse_legacy_ncu_calibration('accuracy_diagnostic._build_cohort_cell_payloads.predictor_callback')
    payloads: list[dict] = []
    for entry in PREDICTOR_DESIGN_COVERAGE_BY_CELL.values():
        cell_name = entry.cell_name
        bucket = entry.predictor_bucket

        payload: dict[str, Any] = {
            "cell_name": cell_name,
            "predictor_bucket": bucket,
            "status": STATUS_DIAGNOSTIC_FAILED,
            "typed_reason": "sidecar_parse_failed",
        }

        if bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            payload["status"] = STATUS_TYPED_UNAVAILABLE
            payload["typed_reason"] = (
                entry.typed_unavailable_reason
                or "cache_geometry_unresolved"
            )
            payloads.append(payload)
            continue

        # Look up the cell in the ledger to get NCU labels.
        label_l1: float | None = None
        label_l2: float | None = None
        for ledger_entry in ledger_data.get("entries", ()):
            if ledger_entry.get("cell_name") == cell_name:
                label = ledger_entry.get("label", {})
                label_l1 = label.get("l1_global_load_hit_rate")
                label_l2 = label.get("l2_read_hit_rate")
                break

        if label_l1 is None or label_l2 is None:
            # No NCU label available; report diagnostic_failed.
            payload["status"] = STATUS_DIAGNOSTIC_FAILED
            payload["typed_reason"] = "ledger_entry_missing"
            payloads.append(payload)
            continue

        # Pass a synthetic problem record to the callback. The
        # callback can choose to return a `measured` payload or
        # a `diagnostic_failed` payload.
        synthetic_problem = CanonicalSolbenchProblem(
            problem_slug=cell_name,
            problem_category=CATEGORY_L1,  # placeholder; cohort summary doesn't care
            reference_markdown=Path(),
            benchmark_dir=Path(),
        )
        synthetic_lookup = LedgerLookupResult(
            present=True,
            ncu_l1_hit_rate=label_l1,
            ncu_l2_hit_rate=label_l2,
        )
        callback_result = predictor_callback(
            synthetic_problem, bucket, synthetic_lookup,
        )
        if not isinstance(callback_result, Mapping):
            payload["status"] = STATUS_DIAGNOSTIC_FAILED
            payload["typed_reason"] = "sidecar_parse_failed"
            payloads.append(payload)
            continue

        status = callback_result.get("status")
        if status == STATUS_MEASURED:
            ncu_l1 = float(callback_result["ncu_l1_label"])
            ncu_l2 = float(callback_result["ncu_l2_label"])
            pred_l1_raw = float(callback_result["predicted_l1_raw"])
            pred_l2_raw = float(callback_result["predicted_l2_raw"])
            pred_l1_cal = float(callback_result["predicted_l1_calibrated"])
            pred_l2_cal = float(callback_result["predicted_l2_calibrated"])
            payload.update({
                "status": STATUS_MEASURED,
                "typed_reason": "",
                "ncu_l1_label": ncu_l1,
                "ncu_l2_label": ncu_l2,
                "predicted_l1_raw": pred_l1_raw,
                "predicted_l2_raw": pred_l2_raw,
                "predicted_l1_calibrated": pred_l1_cal,
                "predicted_l2_calibrated": pred_l2_cal,
                "l1_abs_pp_error": abs(pred_l1_raw - ncu_l1) * 100.0,
                "l2_abs_pp_error": abs(pred_l2_raw - ncu_l2) * 100.0,
                "l1_calibrated_abs_pp_error": abs(pred_l1_cal - ncu_l1) * 100.0,
                "l2_calibrated_abs_pp_error": abs(pred_l2_cal - ncu_l2) * 100.0,
                "l1_raw_rel_err": abs(pred_l1_raw - ncu_l1) / max(ncu_l1, denominator_floor),
                "l2_raw_rel_err": abs(pred_l2_raw - ncu_l2) / max(ncu_l2, denominator_floor),
                "l1_calibrated_rel_err": abs(pred_l1_cal - ncu_l1) / max(ncu_l1, denominator_floor),
                "l2_calibrated_rel_err": abs(pred_l2_cal - ncu_l2) / max(ncu_l2, denominator_floor),
            })
        elif status == STATUS_DIAGNOSTIC_FAILED:
            payload["status"] = STATUS_DIAGNOSTIC_FAILED
            payload["typed_reason"] = callback_result.get(
                "typed_reason", "sidecar_parse_failed",
            )
        elif status == STATUS_TYPED_UNAVAILABLE:
            payload["status"] = STATUS_TYPED_UNAVAILABLE
            payload["typed_reason"] = callback_result.get("typed_reason", "")

        payloads.append(payload)

    return payloads


def _aggregate_cohort_rows(
    cohort_cell_payloads: Sequence[Mapping[str, Any]],
    *,
    accuracy_threshold: float = 0.10,
) -> list[dict]:
    """Build the six `[[cohort_accuracy_summary]]` rows.

    Cohort summary scope is the 36-cell calibration partition
    (NOT the 235-problem partition). The caller passes one
    payload per calibration cell, each with `cell_name`,
    `predictor_bucket` (one of `predictor_friendly`,
    `predictor_hostile`, `predictor_typed_unavailable`),
    `status`, and (for measured cells) the 4 rel-err and 4
    abs-pp-error fields plus `typed_reason`.

    ``accuracy_threshold`` (default 0.10) is the rel-err cutoff
    used to populate the per-cohort failure-cell list. Must
    match the caller's `[meta].accuracy_threshold` so the
    cohort failure-counting agrees with the report's own
    metadata.
    """

    cohort_map = {
        _PREDICTOR_BUCKET_FRIENDLY: COHORT_FRIENDLY,
        _PREDICTOR_BUCKET_HOSTILE: COHORT_HOSTILE,
        PREDICTOR_BUCKET_TYPED_UNAVAILABLE: COHORT_TYPED_UNAVAILABLE,
    }

    # Partition the calibration cells into the three cohort
    # buckets (no cache-level split yet).
    per_cohort_cells: dict[str, list[Mapping[str, Any]]] = {
        COHORT_FRIENDLY: [],
        COHORT_HOSTILE: [],
        COHORT_TYPED_UNAVAILABLE: [],
    }
    for payload in cohort_cell_payloads:
        cohort = cohort_map.get(payload.get("predictor_bucket"))
        if cohort is None:
            continue
        per_cohort_cells[cohort].append(payload)

    cohort_summary_rows: list[dict] = []
    for cohort in (COHORT_FRIENDLY, COHORT_HOSTILE, COHORT_TYPED_UNAVAILABLE):
        cells = per_cohort_cells[cohort]
        for cache_level in (CACHE_LEVEL_L1, CACHE_LEVEL_L2):
            measured = [
                c for c in cells if c.get("status") == STATUS_MEASURED
            ]
            rel_err_field = (
                "l1_calibrated_rel_err" if cache_level == CACHE_LEVEL_L1
                else "l2_calibrated_rel_err"
            )
            abs_pp_field = (
                "l1_calibrated_abs_pp_error" if cache_level == CACHE_LEVEL_L1
                else "l2_calibrated_abs_pp_error"
            )

            non_measured_names = sorted(
                c["cell_name"] for c in cells
                if c.get("status") != STATUS_MEASURED
            )
            over_threshold_names = sorted(
                c["cell_name"] for c in measured
                if _is_finite_float(c.get(rel_err_field))
                and float(c[rel_err_field]) > accuracy_threshold
            )
            # Combined sorted failure list (non-measured +
            # measured-over-threshold).
            failure_names_set = set(non_measured_names) | set(over_threshold_names)
            failure_names = tuple(sorted(failure_names_set))

            if len(measured) < 5:
                mean_pp: float | str = NAN_TOO_FEW_MEASURED_CELLS
                max_pp: float | str = NAN_TOO_FEW_MEASURED_CELLS
                p95_pp: float | str = NAN_TOO_FEW_MEASURED_CELLS
                geomean: float | str = NAN_TOO_FEW_MEASURED_CELLS
                max_rel: float | str = NAN_TOO_FEW_MEASURED_CELLS
                max_rel_cell_name = ""
            else:
                pp_values = [float(c[abs_pp_field]) for c in measured]
                rel_values = [float(c[rel_err_field]) for c in measured]
                mean_pp = sum(pp_values) / len(pp_values)
                max_pp = max(pp_values)
                sorted_pp = sorted(pp_values)
                p95_index = int(0.95 * (len(sorted_pp) - 1))
                p95_pp = sorted_pp[p95_index]
                geomean = math.exp(
                    sum(math.log(max(v, 1e-12)) for v in rel_values) / len(rel_values)
                )
                max_rel = max(rel_values)
                max_rel_cell_name = sorted(
                    c["cell_name"] for c in measured
                    if abs(float(c[rel_err_field]) - max_rel) < 1e-12
                )[0]
                if max_rel <= accuracy_threshold:
                    # max_rel_cell_name is in failure list only when
                    # max_rel > accuracy_threshold; clear it
                    # otherwise to keep the
                    # validator's membership invariant happy.
                    max_rel_cell_name = ""

            gate_passed = (
                cohort != COHORT_TYPED_UNAVAILABLE
                and len(failure_names) == 0
            )

            cohort_summary_rows.append({
                "cohort": cohort,
                "cache_level": cache_level,
                "admitted_cell_count": len(cells),
                "mean_abs_pp_error": mean_pp,
                "max_abs_pp_error": max_pp,
                "p95_abs_pp_error": p95_pp,
                "geomean_rel_err": geomean,
                "max_rel_err": max_rel,
                "max_rel_err_cell_name": max_rel_cell_name,
                "cells_above_10pct_rel_err_count": len(failure_names),
                "cells_above_10pct_rel_err_names": failure_names,
                "accuracy_gate_passed": gate_passed,
            })

    return cohort_summary_rows


def _build_overall_summary(
    per_cell_rows: Sequence[Mapping[str, Any]],
    cohort_summary_rows: Sequence[Mapping[str, Any]],
    *,
    split_map: Mapping[str, str] | None = None,
    cohort_cell_payloads: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Build the `[overall_accuracy_summary]` table.

    `cell_count` is the deployed cohort partition total (via
    `predictor_design_coverage.partition_size()`), NOT the
    235-problem total. `accuracy_gate_passed` is the AND of all
    six cohort rows' gate flags. The failure list aggregates
    per-cohort failures.

    `split_map` (optional): the by-slug split assignment from
    `CalibrationValidationSplit.by_slug`. When provided, the
    six split-aware fields are populated by aggregating
    measured per-cell rows joined to the split assignment:
    `calibration_only_*` and `validation_only_*` independent
    measured counts and mean L1/L2 relative-error fractions.
    When `split_map is None`, the six fields carry their
    sentinel values (preserving backward compatibility with
    legacy build paths that pre-date the calibration /
    validation split).
    """

    all_failures: set[str] = set()
    for cohort_row in cohort_summary_rows:
        for slug in cohort_row["cells_above_10pct_rel_err_names"]:
            all_failures.add(slug)

    failure_names = tuple(sorted(all_failures))
    gate_passed = all(
        row["accuracy_gate_passed"] for row in cohort_summary_rows
    )

    if split_map is None:
        return {
            "cell_count": partition_size(),
            "cells_above_10pct_rel_err_count": len(failure_names),
            "cells_above_10pct_rel_err_names": failure_names,
            "accuracy_gate_passed": gate_passed,
            "calibration_only_measured_count": SPLIT_AWARE_COUNT_NA_SENTINEL,
            "validation_only_measured_count": SPLIT_AWARE_COUNT_NA_SENTINEL,
            "calibration_only_l1_mean_rel_err": SPLIT_AWARE_REL_ERR_NA_SENTINEL,
            "calibration_only_l2_mean_rel_err": SPLIT_AWARE_REL_ERR_NA_SENTINEL,
            "validation_only_l1_mean_rel_err": SPLIT_AWARE_REL_ERR_NA_SENTINEL,
            "validation_only_l2_mean_rel_err": SPLIT_AWARE_REL_ERR_NA_SENTINEL,
        }

    split_aware_fields = _build_split_aware_summary_from_rows(
        per_cell_rows=per_cell_rows,
        cohort_cell_payloads=(
            cohort_cell_payloads if cohort_cell_payloads is not None else ()
        ),
        split_map=split_map,
    )
    return {
        "cell_count": partition_size(),
        "cells_above_10pct_rel_err_count": len(failure_names),
        "cells_above_10pct_rel_err_names": failure_names,
        "accuracy_gate_passed": gate_passed,
        **split_aware_fields,
    }


def _mean_or_sentinel(values: list[float]) -> float:
    """Return the arithmetic mean of `values`, or the
    documented sentinel `SPLIT_AWARE_REL_ERR_NA_SENTINEL` when
    `values` is empty.
    """
    if not values:
        return SPLIT_AWARE_REL_ERR_NA_SENTINEL
    return sum(values) / len(values)


def _row_slug(row: Mapping[str, Any]) -> str | None:
    """Read the slug from a per-cell row or cohort payload row,
    in priority order: `problem_slug` (per-cell), then
    `cell_name` (cohort payload), then `slug` (generic).
    """
    return (
        row.get("problem_slug")
        or row.get("cell_name")
        or row.get("slug")
    )


def _row_rel_err_pair(
    row: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Return `(l1_calibrated_rel_err, l2_calibrated_rel_err)`
    as floats, or `None` when either field is missing or
    non-numeric.
    """
    l1 = row.get("l1_calibrated_rel_err")
    l2 = row.get("l2_calibrated_rel_err")
    if not isinstance(l1, (int, float)) or not isinstance(l2, (int, float)):
        return None
    return float(l1), float(l2)


def _build_split_aware_summary_from_rows(
    *,
    per_cell_rows: Sequence[Mapping[str, Any]],
    cohort_cell_payloads: Sequence[Mapping[str, Any]],
    split_map: Mapping[str, str],
) -> dict:
    """Compute the six split-aware summary fields from the row
    sources + split map. Returns a dict carrying the six
    keys (`calibration_only_measured_count`,
    `validation_only_measured_count`,
    `calibration_only_l1_mean_rel_err`,
    `calibration_only_l2_mean_rel_err`,
    `validation_only_l1_mean_rel_err`,
    `validation_only_l2_mean_rel_err`).

    Source discipline (enforced because the same helper is
    reused by `validate_diagnostic_report()` for tamper
    detection):

      * `calibration_only` slugs read their measured rel-err
        from `cohort_cell_payloads`, keyed by `cell_name`.
        The training partition lives in the cohort payload
        stream and covers both the 21 `calib_*` corpus cells
        and the 15 cohort-B `problem_*` cells.
      * `validation_only` slugs read their measured rel-err
        from `per_cell_rows`, keyed by `problem_slug`. The
        canonical 235-problem stream is the validation set's
        authoritative source. When a `validation_only` slug
        is absent from `per_cell_rows` entirely, the helper
        falls back to `cohort_cell_payloads` so a non-canonical
        validation-only slug (e.g., a future synthesized
        manifest entry without a per-problem row yet) still
        contributes.
      * `quarantine` slugs contribute to NEITHER aggregate.
      * Each slug contributes at most once per side. Status
        filtering (`status == measured`) happens BEFORE the
        per-side selection so a non-measured row in the wrong
        source can NOT shadow a measured row from the right
        source.
    """
    cohort_measured_by_slug: dict[str, tuple[float, float]] = {}
    for payload in cohort_cell_payloads:
        if payload.get("status") != STATUS_MEASURED:
            continue
        slug = _row_slug(payload)
        if slug is None or slug in cohort_measured_by_slug:
            continue
        pair = _row_rel_err_pair(payload)
        if pair is None:
            continue
        cohort_measured_by_slug[slug] = pair

    per_cell_measured_by_slug: dict[str, tuple[float, float]] = {}
    per_cell_slugs: set[str] = set()
    for row in per_cell_rows:
        slug = _row_slug(row)
        if slug is None:
            continue
        per_cell_slugs.add(slug)
        if row.get("status") != STATUS_MEASURED:
            continue
        if slug in per_cell_measured_by_slug:
            continue
        pair = _row_rel_err_pair(row)
        if pair is None:
            continue
        per_cell_measured_by_slug[slug] = pair

    calibration_only_l1: list[float] = []
    calibration_only_l2: list[float] = []
    validation_only_l1: list[float] = []
    validation_only_l2: list[float] = []
    for slug, assignment in split_map.items():
        if assignment == "calibration_only":
            pair = cohort_measured_by_slug.get(slug)
            if pair is None:
                continue
            calibration_only_l1.append(pair[0])
            calibration_only_l2.append(pair[1])
        elif assignment == "validation_only":
            # Prefer the canonical per-cell row. Fall back to the
            # cohort payload ONLY when the slug is absent from
            # per_cell_rows entirely; when a canonical per-cell
            # row exists but is non-measured (e.g.
            # diagnostic_failed / typed_unavailable), the
            # canonical source's "not measured" verdict is
            # authoritative and a same-slug measured cohort
            # payload MUST NOT override it.
            pair = per_cell_measured_by_slug.get(slug)
            if pair is None:
                if slug in per_cell_slugs:
                    continue
                pair = cohort_measured_by_slug.get(slug)
            if pair is None:
                continue
            validation_only_l1.append(pair[0])
            validation_only_l2.append(pair[1])
        # quarantine slugs are deliberately ignored for both
        # train and validation aggregates.

    return {
        "calibration_only_measured_count": len(calibration_only_l1),
        "validation_only_measured_count": len(validation_only_l1),
        "calibration_only_l1_mean_rel_err": _mean_or_sentinel(
            calibration_only_l1,
        ),
        "calibration_only_l2_mean_rel_err": _mean_or_sentinel(
            calibration_only_l2,
        ),
        "validation_only_l1_mean_rel_err": _mean_or_sentinel(
            validation_only_l1,
        ),
        "validation_only_l2_mean_rel_err": _mean_or_sentinel(
            validation_only_l2,
        ),
    }


def _build_coverage_summary(
    per_cell_rows: Sequence[Mapping[str, Any]],
) -> dict:
    """Build the `[coverage_summary]` table.

    Counts buckets across all 235 per-cell rows.
    """

    friendly = 0
    hostile = 0
    typed_unavail = 0
    diag_failed = 0
    corpus_pending = 0
    measured_count = 0
    typed_unavail_reasons: dict[str, int] = {}
    diag_reasons: dict[str, int] = {}
    corpus_reasons: dict[str, int] = {}
    category_counts: dict[str, int] = {
        "l1": 0, "l2": 0, "quant": 0, "fib": 0,
    }
    category_map = {
        CATEGORY_L1: "l1", CATEGORY_L2: "l2",
        CATEGORY_QUANT: "quant", CATEGORY_FIB: "fib",
    }

    # Seven mutually exclusive buckets summing to 235:
    # `total_problems == friendly_cells + hostile_cells +
    # typed_unavailable_cells + diagnostic_failed_cells +
    # corpus_expansion_pending_cells + host_arch_unsupported_cells +
    # measured_cell_count`.
    #
    # Assignment priority (each problem in EXACTLY one bucket):
    #   corpus_expansion_pending: status=corpus_expansion_pending
    #   diagnostic_failed: status=diagnostic_failed
    #   typed_unavailable: status=typed_unavailable
    #   measured: status=measured (and cohort=friendly/hostile)
    #   friendly: classified-friendly with no other status
    #     (transient state; on day-1 with the default callback
    #     this is 0 because admitted friendly cells go to
    #     diagnostic_failed)
    #   hostile: classified-hostile with no other status
    host_arch_unsupported_count = 0
    host_arch_unsupported_reasons: dict[str, int] = {}
    for row in per_cell_rows:
        category_counts[category_map[row["problem_category"]]] += 1
        bucket = row.get("predictor_bucket")
        status = row.get("status")
        if status == STATUS_CORPUS_EXPANSION_PENDING:
            corpus_pending += 1
            reason = row.get("typed_reason", "")
            corpus_reasons[reason] = corpus_reasons.get(reason, 0) + 1
        elif status == STATUS_DIAGNOSTIC_FAILED:
            diag_failed += 1
            reason = row.get("typed_reason", "")
            diag_reasons[reason] = diag_reasons.get(reason, 0) + 1
        elif status == STATUS_TYPED_UNAVAILABLE:
            typed_unavail += 1
            reason = row.get("typed_reason", "")
            typed_unavail_reasons[reason] = typed_unavail_reasons.get(reason, 0) + 1
        elif status == STATUS_HOST_ARCH_UNSUPPORTED:
            host_arch_unsupported_count += 1
            reason = row.get("typed_reason", "")
            host_arch_unsupported_reasons[reason] = (
                host_arch_unsupported_reasons.get(reason, 0) + 1
            )
        elif status == STATUS_MEASURED:
            measured_count += 1
        elif bucket == _PREDICTOR_BUCKET_FRIENDLY:
            # Classified friendly but no terminal status assigned.
            friendly += 1
        elif bucket == _PREDICTOR_BUCKET_HOSTILE:
            hostile += 1

    return {
        "total_problems": 235,
        "category_counts": category_counts,
        "friendly_cells": friendly,
        "hostile_cells": hostile,
        "typed_unavailable_cells": typed_unavail,
        "diagnostic_failed_cells": diag_failed,
        "corpus_expansion_pending_cells": corpus_pending,
        "host_arch_unsupported_cells": host_arch_unsupported_count,
        "measured_cell_count": measured_count,
        "typed_unavailable_reasons": typed_unavail_reasons,
        "diagnostic_failed_reasons": diag_reasons,
        "corpus_expansion_pending_by_reason": corpus_reasons,
        "host_arch_unsupported_reasons": host_arch_unsupported_reasons,
        "admitted_measured_fraction": measured_count / 235,
        "coverage_floor_met": measured_count >= _COVERAGE_FLOOR_SMOKE,
    }


def _count_status(
    per_cell_rows: Sequence[Mapping[str, Any]], status: str,
) -> int:
    return sum(1 for row in per_cell_rows if row.get("status") == status)


def _build_denominator_box(
    per_cell_rows: Sequence[Mapping[str, Any]],
) -> dict:
    """Compute the source-of-truth `[denominator_box]` counts
    from the per-cell rows. Walks the rows once and tallies the
    mutually-exclusive status partition plus the three gate
    aggregates. Caller validates via
    `validate_denominator_box`.
    """

    measured = _count_status(per_cell_rows, STATUS_MEASURED)
    typed_unavail = _count_status(per_cell_rows, STATUS_TYPED_UNAVAILABLE)
    diag_failed = _count_status(per_cell_rows, STATUS_DIAGNOSTIC_FAILED)
    corpus_pending = _count_status(per_cell_rows, STATUS_CORPUS_EXPANSION_PENDING)
    host_arch_unsup = _count_status(per_cell_rows, STATUS_HOST_ARCH_UNSUPPORTED)
    passing_user = sum(
        1 for row in per_cell_rows
        if bool(row.get("pass_user_target_threshold"))
    )
    passing_tiered = sum(
        1 for row in per_cell_rows
        if bool(row.get("pass_tiered_gate"))
    )
    # Compound gate = measured rows that pass BOTH verdicts PLUS
    # the host_arch_unsupported carveout. Final acceptance
    # condition: `passing_compound_gate == measured +
    # host_arch_unsupported` when every measured row passes both
    # gates.
    passing_compound_measured = sum(
        1 for row in per_cell_rows
        if row.get("status") == STATUS_MEASURED
        and bool(row.get("pass_user_target_threshold"))
        and bool(row.get("pass_tiered_gate"))
    )
    passing_compound = passing_compound_measured + host_arch_unsup
    return {
        "total_problems": 235,
        "measured": measured,
        "typed_unavailable": typed_unavail,
        "diagnostic_failed": diag_failed,
        "corpus_pending": corpus_pending,
        "host_arch_unsupported": host_arch_unsup,
        "passing_user_target": passing_user,
        "passing_tiered_gate": passing_tiered,
        "passing_compound_gate": passing_compound,
    }


def _build_overall_user_target_summary(
    per_cell_rows: Sequence[Mapping[str, Any]],
    denominator_box: Mapping[str, Any],
) -> dict:
    """Compute the `[overall_user_target_summary]` aggregation
    from the per-cell rows + the denominator_box. The `gate_
    compound_passed` bool is True iff every non-
    host_arch_unsupported problem reaches `measured` status AND
    passes both gates.
    """

    measured_pass = sum(
        1 for row in per_cell_rows
        if row.get("status") == STATUS_MEASURED
        and bool(row.get("pass_user_target_threshold"))
    )
    measured_fail = sum(
        1 for row in per_cell_rows
        if row.get("status") == STATUS_MEASURED
        and not bool(row.get("pass_user_target_threshold"))
    )
    # Final acceptance condition: passing_compound_gate ==
    # measured + host_arch_unsupported.
    gate_compound_passed = bool(
        denominator_box["passing_compound_gate"]
        == denominator_box["measured"]
        + denominator_box["host_arch_unsupported"]
        and denominator_box["typed_unavailable"] == 0
        and denominator_box["diagnostic_failed"] == 0
        and denominator_box["corpus_pending"] == 0
    )
    return {
        "measured_pass_count": measured_pass,
        "measured_fail_count": measured_fail,
        "user_target_threshold_pct": USER_TARGET_THRESHOLD_PCT,
        "gate_compound_passed": gate_compound_passed,
    }


def _build_meta_table(
    *,
    implementation_commit_sha: str,
    ledger_path: str,
    ledger_sha256: str,
    gpu_arch: str = "sm_86",
    gpu_device_name: str = "NVIDIA GeForce RTX 3090 Ti",
    nvbit_so_sha256: str = "0" * 64,
    ncu_metric_version: str = "ncu_2026_1",
    driver_version: str = "unknown",
    trace_budget_bytes: int = 0,
    timeout_seconds: int = 0,
    denominator_floor: float = 0.01,
    accuracy_threshold: float = 0.10,
    report_generated_at_utc: str = "",
    gate_compound: bool = False,
) -> dict:
    """Build the `[meta]` table.

    `gate_compound` is the final compound-gate verdict (True iff
    BOTH the contractual tiered gate AND the user-acceptance
    <10% gate pass across every sm_86-compatible problem). The
    builder receives the bool from upstream aggregation; today's
    pre-survey state defaults to False.
    """

    import datetime as _datetime
    if not report_generated_at_utc:
        report_generated_at_utc = _datetime.datetime.now(
            _datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "implementation_commit_sha": implementation_commit_sha,
        "ledger_path": ledger_path,
        "ledger_sha256": ledger_sha256,
        "nvbit_so_sha256": nvbit_so_sha256,
        "gpu_arch": gpu_arch,
        "gpu_device_name": gpu_device_name,
        "ncu_metric_version": ncu_metric_version,
        "driver_version": driver_version,
        "trace_budget_bytes": trace_budget_bytes,
        "timeout_seconds": timeout_seconds,
        "report_generated_at_utc": report_generated_at_utc,
        "denominator_floor": denominator_floor,
        "accuracy_threshold": accuracy_threshold,
        "gate_compound": gate_compound,
    }


def _resolve_ledger_ncu_metric_version(ledger_data: Any) -> str:
    """Resolve the ledger's NCU metric version for the diagnostic meta.

    Scans every level the ledger may expose the field at:
      * ``ledger_data["meta"]["ncu_metric_version"]``
      * ``ledger_data["entries"][*]["ncu_metric_version"]``
      * ``ledger_data["entries"][*]["label"]["ncu_metric_version"]``

    All non-empty values seen must agree. Returns the unique value
    when at least one level exposes it. Returns the literal
    ``"unknown"`` only when no level exposes a value.

    Raises ``AccuracyGateError("diagnostic_meta_ncu_metric_version_conflict", ...)``
    when any two levels disagree -- including a meta value that
    contradicts an entry / label value.
    """
    if not isinstance(ledger_data, Mapping):
        return "unknown"

    seen: set[str] = set()

    meta = ledger_data.get("meta", {})
    if isinstance(meta, Mapping):
        meta_value = meta.get("ncu_metric_version")
        if isinstance(meta_value, str) and meta_value:
            seen.add(meta_value)

    entries = ledger_data.get("entries", ())
    if isinstance(entries, Sequence):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            entry_value = entry.get("ncu_metric_version")
            if isinstance(entry_value, str) and entry_value:
                seen.add(entry_value)
            label = entry.get("label", {})
            if isinstance(label, Mapping):
                label_value = label.get("ncu_metric_version")
                if isinstance(label_value, str) and label_value:
                    seen.add(label_value)

    if len(seen) == 1:
        return next(iter(seen))
    if len(seen) > 1:
        raise AccuracyGateError(
            "diagnostic_meta_ncu_metric_version_conflict",
            "ledger exposes conflicting ncu_metric_version values "
            "across meta / entries / labels: "
            f"{sorted(seen)!r}",
        )

    return "unknown"


def _merge_sm86_expanded_completed_labels(
    ledger_data: dict, expanded_ncu_ledger_path: Path,
) -> int:
    """Append the expanded sm_86 ledger's COMPLETED cells to `ledger_data`
    in the full-partition `cell_name` / `label` entry shape so
    `detect_ledger_entry_for_problem` finds them (they leave `ncu_labels_missing`).
    A completed expanded row carries `slug`, `completed = true`,
    `ncu_l1_hit_rate_mean`, `ncu_l2_hit_rate_mean`. Returns the merged count."""
    import tomllib
    data = tomllib.loads(expanded_ncu_ledger_path.read_text("utf-8"))
    entries = ledger_data.setdefault("entries", [])
    if isinstance(entries, tuple):
        entries = list(entries)
        ledger_data["entries"] = entries
    have = {e.get("cell_name") for e in entries}
    merged = 0
    for e in data.get("entries", ()):
        # Only completed cohort_b rows carry the inline NCU label means; seed rows
        # reference provenance SHAs and are labeled via the full-partition ledger.
        if not e.get("completed"):
            continue
        if "ncu_l1_hit_rate_mean" not in e or "ncu_l2_hit_rate_mean" not in e:
            continue
        slug = e["slug"]
        if slug in have:
            continue
        entries.append({
            "cell_name": slug,
            "label": {
                "l1_global_load_hit_rate": float(e["ncu_l1_hit_rate_mean"]),
                "l2_read_hit_rate": float(e["ncu_l2_hit_rate_mean"]),
            },
            "source": "sm86_expanded_ncu_ledger",
        })
        have.add(slug)
        merged += 1
    return merged


def _merge_sm86_expanded_per_cell_presence(
    ledger_data: dict, expanded_validation_report_path: Path,
) -> int:
    """Make EVERY expanded validation-report `[[per_cell]]` cell `present` in
    `ledger_data` so it reaches `_sm86_expanded_predictor_callback` instead of
    stalling at the `ncu_labels_missing` ledger-presence gate. MEASURED cells get
    their real NCU labels; EXCLUDED cells (capability-OOD / CV-quarantine /
    raw-unavailable — incl. the seed cells absent from the full partition) get a
    `cell_name`-only marker (no label) — the callback overrides them to a typed
    abstention regardless, so the absent label is never read. The 2 zero-read cells
    are still caught earlier by the `[[unavailable]]` zero-read detector. Existing
    cell_names are never clobbered. Returns the merged count."""
    import tomllib
    report = tomllib.loads(expanded_validation_report_path.read_text("utf-8"))
    entries = ledger_data.setdefault("entries", [])
    if isinstance(entries, tuple):
        entries = list(entries)
        ledger_data["entries"] = entries
    have = {e.get("cell_name") for e in entries}
    merged = 0
    for pc in report.get("per_cell", ()):
        slug = pc["slug"]
        if slug in have:
            continue
        entry = {"cell_name": slug, "source": "sm86_expanded_validation_report"}
        if pc.get("classification") == "measured":
            entry["label"] = {
                "l1_global_load_hit_rate": float(pc["ncu_l1_label"]),
                "l2_read_hit_rate": float(pc["ncu_l2_label"]),
            }
        entries.append(entry)
        have.add(slug)
        merged += 1
    return merged


def _sm86_expanded_predictor_callback(
    expanded_validation_report_path: Path, base_callback,
):
    """A `predictor_callback` driven by the expanded sm_86 validation report's
    `[[per_cell]]` table. For the 65 expanded cells it returns the
    authoritative payload — `measured` (with the deployed per-cell predictions) for
    admitted cells, or a `diagnostic_failed` abstention with the mapped typed reason
    for capability-OOD / NCU-terminal / raw-unavailable exclusions — so the diagnostic
    leaves the stale full-partition fallback. Cells outside the expanded universe
    defer to `base_callback`."""
    refuse_legacy_ncu_calibration('accuracy_diagnostic._sm86_expanded_predictor_callback')
    import tomllib
    if not expanded_validation_report_path.is_file():
        raise AccuracyGateError(
            "canonical_partition_drift",
            f"expanded sm86 validation report not found at {expanded_validation_report_path!s}",
        )
    report = tomllib.loads(expanded_validation_report_path.read_text("utf-8"))
    by_slug = {pc["slug"]: pc for pc in report.get("per_cell", ())}

    def _excluded_reason(pc: Mapping[str, Any]) -> str:
        cls = pc["classification"]
        if cls == "capability_excluded":
            return "predictor_typed_unavailable_non_canonical"
        if cls == "ncu_terminal_excluded":
            # zero-read traffic surfaces specifically; CV-quarantine abstains.
            return ("ncu_label_no_read_only_traffic"
                    if pc.get("typed_reason") == "ncu_label_no_read_only_traffic"
                    else "predictor_typed_unavailable_non_canonical")
        return "sidecar_parse_failed"  # raw_unavailable: the Tier-2 trace was absent

    def _callback(problem, cohort_bucket, ledger_lookup):
        pc = by_slug.get(problem.problem_slug)
        if pc is None:
            return base_callback(problem, cohort_bucket, ledger_lookup)
        if pc["classification"] == "measured":
            return {
                "status": STATUS_MEASURED,
                "ncu_l1_label": float(pc["ncu_l1_label"]),
                "ncu_l2_label": float(pc["ncu_l2_label"]),
                "predicted_l1_raw": float(pc["predicted_l1_raw"]),
                "predicted_l2_raw": float(pc["predicted_l2_raw"]),
                "predicted_l1_calibrated": float(pc["predicted_l1_calibrated"]),
                "predicted_l2_calibrated": float(pc["predicted_l2_calibrated"]),
            }
        return {"status": STATUS_DIAGNOSTIC_FAILED, "typed_reason": _excluded_reason(pc)}

    return _callback


def build_accuracy_diagnostic_report(
    *,
    repo_root: Path | None = None,
    baselines_root: Path | None = None,
    ledger_path: Path | None = None,
    implementation_commit_sha: str | None = None,
    denominator_floor: float = 0.01,
    accuracy_threshold: float = 0.10,
    predictor_callback=None,
    canonical_problems: Sequence[CanonicalSolbenchProblem] | None = None,
    nvbit_so_sha256: str | None = None,
    trace_budget_bytes: int | None = None,
    timeout_seconds: int | None = None,
    host_arch_manifest_path: Path | None = None,
    calibration_validation_split_path: Path | None = None,
    expanded_ncu_ledger_path: Path | None = None,
    expanded_validation_report_path: Path | None = None,
) -> DiagnosticReport:
    """Walk the canonical 235-problem partition and assemble a
    full diagnostic report.

    Defaults:
      * `repo_root` derived from the module path.
      * `baselines_root` = `<repo_root>/baselines/`.
      * `ledger_path` = `<repo_root>/tools/sol_baselines_harness/
        calibration/cache_prediction/label_ledger_v1_full_partition.toml`.
      * `implementation_commit_sha` = result of `git rev-parse HEAD`
        in `repo_root`, or the literal sentinel `git_unavailable`
        when git is not installed. The all-zero placeholder is
        rejected.
      * `predictor_callback` = `_default_predictor_callback` (emits
        diagnostic_failed for admitted cells).
    """

    if predictor_callback is not None:
        refuse_legacy_ncu_calibration('accuracy_diagnostic.predictor_callback')
    if expanded_ncu_ledger_path is not None:
        refuse_legacy_ncu_calibration(
            "accuracy_diagnostic.expanded_ncu_ledger_path"
        )
    if expanded_validation_report_path is not None:
        refuse_legacy_ncu_calibration(
            "accuracy_diagnostic.expanded_validation_report_path"
        )
    import hashlib
    import tomllib

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    if baselines_root is None:
        baselines_root = repo_root / "baselines"
    if ledger_path is None:
        ledger_path = (
            repo_root / "tools" / "sol_baselines_harness" / "calibration"
            / "cache_prediction" / "label_ledger_v1_full_partition.toml"
        )
    if implementation_commit_sha is None:
        implementation_commit_sha = _resolve_implementation_commit_sha_from_git(
            repo_root=repo_root,
        )
    if predictor_callback is None:
        predictor_callback = _default_predictor_callback
        # the expanded sm_86 validation report's per-cell table is the
        # authoritative predictor outcome for the 65 expanded cells. When supplied,
        # drive the predictor callback from it so admitted cells become `measured`
        # and the typed exclusions surface — instead of the stale full-partition
        # `sidecar_parse_failed` / `ncu_labels_missing` fallback.
        if expanded_validation_report_path is not None:
            predictor_callback = _sm86_expanded_predictor_callback(
                expanded_validation_report_path, _default_predictor_callback)
    if canonical_problems is None:
        canonical_problems = enumerate_canonical_solbench_problems(repo_root=repo_root)

    if not ledger_path.is_file():
        # Missing ledger FILE raises (vs a missing entry which
        # becomes a typed-row).
        raise AccuracyGateError(
            "canonical_partition_drift",
            f"label ledger file not found at {ledger_path!s}",
        )
    raw = ledger_path.read_bytes()
    ledger_sha = hashlib.sha256(raw).hexdigest()
    ledger_data = tomllib.loads(raw.decode("utf-8"))

    # sm_86 expanded-ledger consumption: `expanded_ncu_ledger_path` merges the
    # completed cohort_b NCU labels. NOTE: on the REAL full-partition ledger those
    # 53 cohort_b cells are already present+labeled, so this merge is a near-no-op
    # there — the AUTHORITATIVE consumption is `expanded_validation_report_path`
    # below (the `[[per_cell]]` table), which the production path should pass.
    # This param is retained for back-compat / standalone-ledger callers.
    if expanded_ncu_ledger_path is not None:
        if not expanded_ncu_ledger_path.is_file():
            raise AccuracyGateError(
                "canonical_partition_drift",
                f"expanded sm86 ncu ledger not found at {expanded_ncu_ledger_path!s}",
            )
        _merge_sm86_expanded_completed_labels(ledger_data, expanded_ncu_ledger_path)
    if expanded_validation_report_path is not None:
        if not expanded_validation_report_path.is_file():
            raise AccuracyGateError(
                "canonical_partition_drift",
                f"expanded sm86 validation report not found at "
                f"{expanded_validation_report_path!s}",
            )
        # Make every expanded per_cell cell (admitted AND excluded — incl. the seed
        # cells absent from the full partition) present so it reaches the expanded
        # predictor callback and surfaces its report-authoritative status.
        _merge_sm86_expanded_per_cell_presence(ledger_data, expanded_validation_report_path)

    # The host-arch manifest is REQUIRED on the production
    # diagnostic path. Without it, every canonical problem
    # would silently reclassify as ordinary corpus_expansion_
    # pending work and the sm_86 denominator boundary would be
    # invisible. Fail closed when the path is absent (or points
    # at a missing file). The loader still accepts
    # `canonical_slugs=None` directly for unit tests that load
    # partial fixtures; that escape hatch is intentional and
    # not reachable from this production builder.
    if host_arch_manifest_path is None:
        host_arch_manifest_path = (
            repo_root / HOST_ARCH_MANIFEST_RELATIVE_PATH
        )
    if not host_arch_manifest_path.is_file():
        raise PerProblemRecordSchemaError(
            "host_arch_manifest_missing",
            f"host-arch availability manifest not found at "
            f"{host_arch_manifest_path!s}; the production "
            f"diagnostic path requires the source-of-truth "
            f"manifest so every canonical problem is "
            f"classified as either sm86_compatible_cohort or "
            f"host_arch_unsupported.",
        )
    canonical_slug_set = frozenset(
        problem.problem_slug for problem in canonical_problems
    )
    host_arch_manifest = load_host_arch_availability_manifest(
        host_arch_manifest_path,
        canonical_slugs=canonical_slug_set,
    )

    # Load the calibration / validation split manifest. The
    # production diagnostic path requires it so the
    # `[overall_accuracy_summary]` block reports
    # calibration_only train rel-err and validation_only
    # validation rel-err as independent fields. Fail closed
    # when the path is missing or the manifest fails strict
    # validation; do NOT silently emit no-split sentinels on
    # the production builder.
    if calibration_validation_split_path is None:
        calibration_validation_split_path = (
            repo_root / CALIBRATION_VALIDATION_SPLIT_RELATIVE_PATH
        )
    if not calibration_validation_split_path.is_file():
        raise PerProblemRecordSchemaError(
            "calibration_validation_split_unavailable",
            f"calibration / validation split manifest not "
            f"found at {calibration_validation_split_path!s}; "
            f"the production diagnostic path requires it so "
            f"the overall summary reports calibration_only "
            f"train rel-err and validation_only validation "
            f"rel-err independently.",
        )
    expected_partition_slugs = frozenset(
        PREDICTOR_DESIGN_COVERAGE_BY_CELL.keys(),
    )
    try:
        split_manifest = load_calibration_validation_split_manifest(
            calibration_validation_split_path,
            expected_slugs=expected_partition_slugs,
        )
    except CalibrationValidationSplitError as exc:
        raise PerProblemRecordSchemaError(
            "calibration_validation_split_unavailable",
            f"calibration / validation split manifest at "
            f"{calibration_validation_split_path!s} failed "
            f"strict validation ({exc.typed_reason}: "
            f"{exc.message})",
        ) from exc
    split_map: Mapping[str, str] = split_manifest.by_slug

    per_cell_rows: list[dict] = []
    for problem in canonical_problems:
        row = _build_per_problem_row(
            problem,
            baselines_root=baselines_root,
            ledger_data=ledger_data,
            denominator_floor=denominator_floor,
            predictor_callback=predictor_callback,
            host_arch_manifest=host_arch_manifest,
        )
        per_cell_rows.append(row)

    # The cohort_accuracy_summary table aggregates the 36-cell
    # calibration partition (NOT the 235-problem partition).
    # Build cell payloads from PREDICTOR_DESIGN_COVERAGE_BY_CELL
    # directly: for each calibration cell, use the canonical
    # entry's cell_name and predictor_bucket, then look up the
    # ledger label and (for friendly/hostile cells) call the
    # predictor callback to get measurement payload.
    cohort_cell_payloads = _build_cohort_cell_payloads(
        ledger_data=ledger_data,
        denominator_floor=denominator_floor,
        predictor_callback=predictor_callback,
    )

    cohort_summary_rows = _aggregate_cohort_rows(
        cohort_cell_payloads,
        accuracy_threshold=accuracy_threshold,
    )
    overall_summary = _build_overall_summary(
        per_cell_rows,
        cohort_summary_rows,
        split_map=split_map,
        cohort_cell_payloads=cohort_cell_payloads,
    )
    coverage_summary = _build_coverage_summary(per_cell_rows)

    # Emit denominator audit rows for measured cells and attach
    # audit_warning fields to the per-cell rows whose audit gaps
    # exceed the 5% threshold. The audit reads `_tier2_l*_total`
    # fields the predictor callback attached; those fields are
    # then stripped before schema validation.
    denominator_audit_rows = _build_denominator_audit_rows(
        per_cell_rows=per_cell_rows, ledger_data=ledger_data,
    )
    _attach_audit_warning_to_per_cell_rows(
        per_cell_rows=per_cell_rows,
        denominator_audit_rows=denominator_audit_rows,
    )
    _strip_audit_input_fields(per_cell_rows)

    # Build the source-of-truth denominator_box from the per-cell
    # rows BEFORE constructing the meta table — `meta.gate_compound`
    # mirrors the user-target summary's `gate_compound_passed`
    # bool, which in turn depends on the denominator_box's
    # `passing_compound_gate` count.
    denominator_box_table = _build_denominator_box(per_cell_rows)
    overall_user_target_table = _build_overall_user_target_summary(
        per_cell_rows, denominator_box_table,
    )

    # Read the ledger's NCU metric version from meta, falling
    # back to a unique value across entries / labels. Conflicting
    # values across entries fail closed via the helper.
    ncu_metric_version = _resolve_ledger_ncu_metric_version(ledger_data)

    meta_table = _build_meta_table(
        implementation_commit_sha=implementation_commit_sha,
        ledger_path=str(ledger_path),
        ledger_sha256=ledger_sha,
        nvbit_so_sha256=nvbit_so_sha256 if nvbit_so_sha256 else "0" * 64,
        ncu_metric_version=ncu_metric_version,
        trace_budget_bytes=trace_budget_bytes if trace_budget_bytes is not None else 0,
        timeout_seconds=timeout_seconds if timeout_seconds is not None else 0,
        denominator_floor=denominator_floor,
        accuracy_threshold=accuracy_threshold,
        gate_compound=overall_user_target_table["gate_compound_passed"],
    )

    # Validate every emitted structure.
    validated_per_cell = tuple(
        validate_per_problem_record(row) for row in per_cell_rows
    )
    validated_cohort = tuple(
        validate_cohort_accuracy_summary_row(
            row, accuracy_threshold=accuracy_threshold,
        )
        for row in cohort_summary_rows
    )
    validated_overall = validate_overall_accuracy_summary(overall_summary)
    validated_coverage = validate_coverage_summary_table(coverage_summary)
    validated_audit = tuple(
        validate_denominator_audit_row(row) for row in denominator_audit_rows
    )
    validated_meta = validate_diagnostic_meta_table(meta_table)
    validated_denominator_box = validate_denominator_box(denominator_box_table)
    validated_overall_user_target = validate_overall_user_target_summary(
        overall_user_target_table,
    )

    measured_cohort_payloads_tuple = tuple(
        MeasuredCohortPayloadRow(
            cell_name=payload["cell_name"],
            l1_calibrated_rel_err=float(
                payload["l1_calibrated_rel_err"],
            ),
            l2_calibrated_rel_err=float(
                payload["l2_calibrated_rel_err"],
            ),
        )
        for payload in cohort_cell_payloads
        if payload.get("status") == STATUS_MEASURED
        and "l1_calibrated_rel_err" in payload
        and "l2_calibrated_rel_err" in payload
    )

    report = DiagnosticReport(
        per_cell_rows=validated_per_cell,
        cohort_summary_rows=validated_cohort,
        overall_summary=validated_overall,
        coverage_summary=validated_coverage,
        denominator_audit_rows=validated_audit,
        meta=validated_meta,
        denominator_box=validated_denominator_box,
        overall_user_target_summary=validated_overall_user_target,
        measured_cohort_payloads=measured_cohort_payloads_tuple,
    )
    validate_diagnostic_report(report)
    return report


# ---------------------------------------------------------------------------
# TOML serialization + parsing
# ---------------------------------------------------------------------------

def _toml_escape_string(value: str) -> str:
    """Escape a string for TOML basic-string output."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _toml_format_value(value: Any) -> str:
    """Render a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return f'"{NAN_SENTINEL}"'
        return repr(float(value))
    if isinstance(value, str):
        return f'"{_toml_escape_string(value)}"'
    if isinstance(value, (tuple, list)):
        items = [_toml_format_value(v) for v in value]
        return "[" + ", ".join(items) + "]"
    if isinstance(value, Mapping):
        items = [
            f'{_toml_escape_string(str(k))} = {_toml_format_value(v)}'
            for k, v in value.items()
        ]
        return "{ " + ", ".join(items) + " }"
    raise TypeError(f"unhandled TOML value type: {type(value).__name__}")


def _format_table(table_name: str, fields: Mapping[str, Any]) -> str:
    """Format a single TOML table."""
    lines = [f"[{table_name}]"]
    for key, value in fields.items():
        lines.append(f"{key} = {_toml_format_value(value)}")
    return "\n".join(lines) + "\n"


def _format_array_of_tables(
    table_name: str, rows: Sequence[Mapping[str, Any]],
) -> str:
    """Format an array of TOML tables (`[[name]]` syntax)."""
    parts: list[str] = []
    for row in rows:
        parts.append(f"[[{table_name}]]")
        for key, value in row.items():
            parts.append(f"{key} = {_toml_format_value(value)}")
        parts.append("")
    return "\n".join(parts)


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert a frozen dataclass to a plain dict for TOML
    emission. Nested mappings become inline TOML mappings.
    """
    from dataclasses import asdict
    result = asdict(obj)
    # Coerce tuple fields to list so the TOML formatter handles
    # them uniformly.
    for k, v in list(result.items()):
        if isinstance(v, tuple):
            result[k] = list(v)
    return result


def serialize_diagnostic_report(report: DiagnosticReport) -> str:
    """Serialize the report to a TOML document.

    Layout (in this order so the round-trip parser can read
    sections sequentially):
      * `[meta]`
      * `[coverage_summary]`
      * `[overall_accuracy_summary]`
      * `[overall_user_target_summary]`
      * `[denominator_box]`
      * `[[per_cell_accuracy]]`  (235 rows)
      * `[[cohort_accuracy_summary]]`  (6 rows)
      * `[[denominator_audit]]`  (variable; may be empty)
    """

    parts: list[str] = []
    parts.append(_format_table("meta", _dataclass_to_dict(report.meta)))
    parts.append(_format_table("coverage_summary", _dataclass_to_dict(report.coverage_summary)))
    parts.append(_format_table("overall_accuracy_summary", _dataclass_to_dict(report.overall_summary)))
    parts.append(_format_table(
        "overall_user_target_summary",
        _dataclass_to_dict(report.overall_user_target_summary),
    ))
    parts.append(_format_table(
        "denominator_box",
        _dataclass_to_dict(report.denominator_box),
    ))
    parts.append(_format_array_of_tables(
        "per_cell_accuracy",
        [_dataclass_to_dict(r) for r in report.per_cell_rows],
    ))
    parts.append(_format_array_of_tables(
        "cohort_accuracy_summary",
        [_dataclass_to_dict(r) for r in report.cohort_summary_rows],
    ))
    parts.append(_format_array_of_tables(
        "denominator_audit",
        [_dataclass_to_dict(r) for r in report.denominator_audit_rows],
    ))
    parts.append(_format_array_of_tables(
        "measured_cohort_payloads",
        [_dataclass_to_dict(r) for r in report.measured_cohort_payloads],
    ))
    return "\n".join(parts)


_REQUIRED_REPORT_SECTIONS: tuple[str, ...] = (
    "meta",
    "coverage_summary",
    "overall_accuracy_summary",
    "overall_user_target_summary",
    "denominator_box",
    "per_cell_accuracy",
    "cohort_accuracy_summary",
)


_EXPECTED_COHORT_CACHE_LEVEL_PAIRS: frozenset[tuple[str, str]] = frozenset({
    (COHORT_FRIENDLY, CACHE_LEVEL_L1),
    (COHORT_FRIENDLY, CACHE_LEVEL_L2),
    (COHORT_HOSTILE, CACHE_LEVEL_L1),
    (COHORT_HOSTILE, CACHE_LEVEL_L2),
    (COHORT_TYPED_UNAVAILABLE, CACHE_LEVEL_L1),
    (COHORT_TYPED_UNAVAILABLE, CACHE_LEVEL_L2),
})


def validate_diagnostic_report(report: DiagnosticReport) -> None:
    """Report-level invariant check.

    Enforces:
      * exactly 235 per-cell rows;
      * exactly 6 cohort summary rows covering the canonical
        `(cohort, cache_level)` cross-product;
      * `coverage_summary.total_problems == 235`;
      * `overall_summary.cell_count == predictor_design_coverage.partition_size()`;
      * no duplicate `problem_slug` values in per_cell;
      * every per_cell row's `problem_slug` is the canonical
        slug set from `enumerate_canonical_solbench_problems()`.

    Raises `PerProblemRecordSchemaError` with typed reasons:
      * `diagnostic_report_invalid_per_cell_count`
      * `diagnostic_report_invalid_cohort_count`
      * `diagnostic_report_missing_cohort_pair`
      * `diagnostic_report_duplicate_problem_slug`
      * `diagnostic_report_non_canonical_slug`
      * `diagnostic_report_denominator_box_mismatch`
      * `diagnostic_report_overall_user_target_mismatch`
      * `diagnostic_report_meta_gate_compound_mismatch`
    """

    if len(report.per_cell_rows) != 235:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_invalid_per_cell_count",
            f"diagnostic report must contain exactly 235 per_cell "
            f"rows, got {len(report.per_cell_rows)}",
        )

    if len(report.cohort_summary_rows) != 6:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_invalid_cohort_count",
            f"diagnostic report must contain exactly 6 cohort "
            f"summary rows, got {len(report.cohort_summary_rows)}",
        )

    cohort_pairs = {
        (row.cohort, row.cache_level)
        for row in report.cohort_summary_rows
    }
    missing_pairs = _EXPECTED_COHORT_CACHE_LEVEL_PAIRS - cohort_pairs
    if missing_pairs:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_missing_cohort_pair",
            f"diagnostic report cohort summary rows do not cover the "
            f"canonical (cohort, cache_level) cross-product; missing "
            f"{sorted(missing_pairs)!r}",
        )

    seen_slugs: set[str] = set()
    for row in report.per_cell_rows:
        if row.problem_slug in seen_slugs:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_duplicate_problem_slug",
                f"diagnostic report contains duplicate problem_slug "
                f"{row.problem_slug!r}",
            )
        seen_slugs.add(row.problem_slug)

    # Cross-check the per-cell statuses against coverage_summary.
    # The coverage builder uses seven mutually exclusive buckets;
    # a parsed TOML can stale-encode coverage counts that no
    # longer match the per_cell rows. Recompute buckets + reason
    # histograms from the per_cell rows and require equality.
    recomputed = _recompute_coverage_from_per_cell_rows(report.per_cell_rows)
    coverage = report.coverage_summary
    expected_counts = {
        "friendly_cells": coverage.friendly_cells,
        "hostile_cells": coverage.hostile_cells,
        "typed_unavailable_cells": coverage.typed_unavailable_cells,
        "diagnostic_failed_cells": coverage.diagnostic_failed_cells,
        "corpus_expansion_pending_cells": coverage.corpus_expansion_pending_cells,
        "host_arch_unsupported_cells": coverage.host_arch_unsupported_cells,
        "measured_cell_count": coverage.measured_cell_count,
    }
    for field_name, expected in expected_counts.items():
        actual = recomputed["counts"][field_name]
        if actual != expected:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_coverage_count_mismatch",
                f"coverage_summary.{field_name}={expected} but per_cell "
                f"rows imply {actual}",
            )

    expected_histograms = {
        "typed_unavailable_reasons": dict(coverage.typed_unavailable_reasons),
        "diagnostic_failed_reasons": dict(coverage.diagnostic_failed_reasons),
        "corpus_expansion_pending_by_reason": dict(
            coverage.corpus_expansion_pending_by_reason
        ),
        "host_arch_unsupported_reasons": dict(
            coverage.host_arch_unsupported_reasons
        ),
    }
    for field_name, expected_hist in expected_histograms.items():
        actual_hist = recomputed["reasons"][field_name]
        if actual_hist != expected_hist:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_coverage_reason_mismatch",
                f"coverage_summary.{field_name}={expected_hist!r} but "
                f"per_cell rows imply {actual_hist!r}",
            )

    # Cross-check the overall summary against the cohort rows. A
    # parsed TOML can stale-encode an overall verdict that
    # disagrees with the cohort rows (e.g., overall says gate
    # passed while cohort rows record failures). Recompute the
    # overall failure list / count / pass-flag from the cohort
    # rows and require equality.
    recomputed_overall_failures = set()
    recomputed_overall_pass = True
    for cohort_row in report.cohort_summary_rows:
        for slug in cohort_row.cells_above_10pct_rel_err_names:
            recomputed_overall_failures.add(slug)
        if not cohort_row.accuracy_gate_passed:
            recomputed_overall_pass = False
    recomputed_overall_names = tuple(sorted(recomputed_overall_failures))
    recomputed_overall_count = len(recomputed_overall_names)
    overall = report.overall_summary
    if tuple(overall.cells_above_10pct_rel_err_names) != recomputed_overall_names:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_overall_summary_mismatch",
            f"overall_accuracy_summary.cells_above_10pct_rel_err_names="
            f"{tuple(overall.cells_above_10pct_rel_err_names)!r} but cohort "
            f"rows imply {recomputed_overall_names!r}",
        )
    if overall.cells_above_10pct_rel_err_count != recomputed_overall_count:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_overall_summary_mismatch",
            f"overall_accuracy_summary.cells_above_10pct_rel_err_count="
            f"{overall.cells_above_10pct_rel_err_count!r} but cohort rows "
            f"imply {recomputed_overall_count!r}",
        )
    if overall.accuracy_gate_passed != recomputed_overall_pass:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_overall_summary_mismatch",
            f"overall_accuracy_summary.accuracy_gate_passed="
            f"{overall.accuracy_gate_passed!r} but cohort rows imply "
            f"{recomputed_overall_pass!r}",
        )

    # Cross-check the denominator_box / overall_user_target_summary
    # / meta.gate_compound against the per_cell rows. The parser
    # accepts these tables independently; a tampered TOML can
    # claim a passing compound gate while the rows still fail.
    # Recompute from the rows and require exact equality.
    per_cell_dicts = [_dataclass_to_dict(row) for row in report.per_cell_rows]
    expected_denominator_box_dict = _build_denominator_box(per_cell_dicts)
    expected_denominator_box = validate_denominator_box(
        expected_denominator_box_dict,
    )
    if report.denominator_box != expected_denominator_box:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_denominator_box_mismatch",
            f"denominator_box does not match per_cell rows:\n"
            f"  parsed:     {report.denominator_box!r}\n"
            f"  recomputed: {expected_denominator_box!r}",
        )

    expected_user_target_dict = _build_overall_user_target_summary(
        per_cell_dicts, expected_denominator_box_dict,
    )
    expected_user_target = validate_overall_user_target_summary(
        expected_user_target_dict,
    )
    if report.overall_user_target_summary != expected_user_target:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_overall_user_target_mismatch",
            f"overall_user_target_summary does not match per_cell "
            f"rows + denominator_box:\n"
            f"  parsed:     {report.overall_user_target_summary!r}\n"
            f"  recomputed: {expected_user_target!r}",
        )

    if report.meta.gate_compound != report.overall_user_target_summary.gate_compound_passed:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_meta_gate_compound_mismatch",
            f"meta.gate_compound="
            f"{report.meta.gate_compound!r} disagrees with "
            f"overall_user_target_summary.gate_compound_passed="
            f"{report.overall_user_target_summary.gate_compound_passed!r}; "
            f"both fields must agree on the compound-gate verdict.",
        )

    # Recompute the six split-aware fields from per_cell_rows
    # + measured_cohort_payloads + the canonical split manifest,
    # and reject any mismatch. A parsed report can otherwise
    # carry hand-tampered split-aware counts/means that pass
    # the per-field schema check; the recomputation catches
    # tampering by deriving the expected values from the
    # report's own row data + the in-tree manifest.
    #
    # All-sentinel state is acceptable ONLY when the report
    # truly carries no measured split data: empty
    # `measured_cohort_payloads` AND no measured per_cell row
    # whose `problem_slug` maps to `calibration_only` or
    # `validation_only` in the canonical split manifest. A
    # missing manifest is treated as "no data to check" iff the
    # report is also all-sentinel with no measured cohort
    # payload (the legacy / pre-split-manifest fallback);
    # otherwise the missing manifest is itself a tamper-detection
    # failure.
    overall = report.overall_summary
    split_fields_all_sentinel = (
        overall.calibration_only_measured_count
        == SPLIT_AWARE_COUNT_NA_SENTINEL
        and overall.validation_only_measured_count
        == SPLIT_AWARE_COUNT_NA_SENTINEL
        and overall.calibration_only_l1_mean_rel_err
        == SPLIT_AWARE_REL_ERR_NA_SENTINEL
        and overall.calibration_only_l2_mean_rel_err
        == SPLIT_AWARE_REL_ERR_NA_SENTINEL
        and overall.validation_only_l1_mean_rel_err
        == SPLIT_AWARE_REL_ERR_NA_SENTINEL
        and overall.validation_only_l2_mean_rel_err
        == SPLIT_AWARE_REL_ERR_NA_SENTINEL
    )
    repo_root_for_split = (
        Path(__file__).resolve().parents[4]
    )
    split_manifest_path = (
        repo_root_for_split
        / CALIBRATION_VALIDATION_SPLIT_RELATIVE_PATH
    )
    manifest_present = split_manifest_path.is_file()
    split_manifest_for_check = None
    if manifest_present:
        try:
            split_manifest_for_check = (
                load_calibration_validation_split_manifest(
                    split_manifest_path,
                    expected_slugs=frozenset(
                        PREDICTOR_DESIGN_COVERAGE_BY_CELL.keys(),
                    ),
                )
            )
        except CalibrationValidationSplitError as exc:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_split_aware_summary_mismatch",
                f"canonical split manifest failed validation "
                f"during diagnostic report cross-check "
                f"({exc.typed_reason}: {exc.message})",
            ) from exc
    has_measured_cohort = bool(report.measured_cohort_payloads)
    has_measured_per_cell = any(
        row.status == STATUS_MEASURED
        for row in report.per_cell_rows
    )
    has_split_assigned_measured_per_cell = False
    if split_manifest_for_check is not None:
        manifest_by_slug = split_manifest_for_check.by_slug
        for row in report.per_cell_rows:
            if row.status != STATUS_MEASURED:
                continue
            assignment = manifest_by_slug.get(row.problem_slug)
            if assignment in (
                "calibration_only", "validation_only",
            ):
                has_split_assigned_measured_per_cell = True
                break
    if split_manifest_for_check is not None:
        # Manifest available: classify per_cell rows against
        # splits. Only manifest-classified measured per_cell rows
        # count as measured split data.
        has_any_measured_split_data = (
            has_measured_cohort
            or has_split_assigned_measured_per_cell
        )
    else:
        # Manifest missing: cannot prove that no measured per_cell
        # row maps to calibration_only / validation_only. Treat
        # ANY measured per_cell row (or measured cohort payload)
        # as measured split data and force the validator into the
        # manifest-missing fail-closed branch below.
        has_any_measured_split_data = (
            has_measured_cohort or has_measured_per_cell
        )
    legacy_no_split_fallback = (
        split_fields_all_sentinel
        and not has_any_measured_split_data
    )
    if not legacy_no_split_fallback:
        if not manifest_present:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_split_aware_summary_mismatch",
                f"split-aware summary fields are populated or "
                f"the report carries measured split data, but "
                f"the canonical split manifest is missing at "
                f"{split_manifest_path!s}; cannot recompute to "
                f"verify the parsed values.",
            )
        cohort_payload_view = [
            {
                "cell_name": payload.cell_name,
                "status": STATUS_MEASURED,
                "l1_calibrated_rel_err": payload.l1_calibrated_rel_err,
                "l2_calibrated_rel_err": payload.l2_calibrated_rel_err,
            }
            for payload in report.measured_cohort_payloads
        ]
        recomputed_split_fields = _build_split_aware_summary_from_rows(
            per_cell_rows=per_cell_dicts,
            cohort_cell_payloads=cohort_payload_view,
            split_map=split_manifest_for_check.by_slug,
        )
        for field_name, parsed_value in (
            (
                "calibration_only_measured_count",
                overall.calibration_only_measured_count,
            ),
            (
                "validation_only_measured_count",
                overall.validation_only_measured_count,
            ),
            (
                "calibration_only_l1_mean_rel_err",
                overall.calibration_only_l1_mean_rel_err,
            ),
            (
                "calibration_only_l2_mean_rel_err",
                overall.calibration_only_l2_mean_rel_err,
            ),
            (
                "validation_only_l1_mean_rel_err",
                overall.validation_only_l1_mean_rel_err,
            ),
            (
                "validation_only_l2_mean_rel_err",
                overall.validation_only_l2_mean_rel_err,
            ),
        ):
            recomputed_value = recomputed_split_fields[field_name]
            if isinstance(parsed_value, float) and isinstance(
                recomputed_value, float
            ):
                if abs(parsed_value - recomputed_value) > 1e-9:
                    raise PerProblemRecordSchemaError(
                        "diagnostic_report_split_aware_summary_mismatch",
                        f"overall_accuracy_summary.{field_name}="
                        f"{parsed_value!r} disagrees with the "
                        f"value recomputed from per_cell_rows + "
                        f"measured_cohort_payloads + split "
                        f"manifest: {recomputed_value!r}",
                    )
            elif parsed_value != recomputed_value:
                raise PerProblemRecordSchemaError(
                    "diagnostic_report_split_aware_summary_mismatch",
                    f"overall_accuracy_summary.{field_name}="
                    f"{parsed_value!r} disagrees with the value "
                    f"recomputed from per_cell_rows + "
                    f"measured_cohort_payloads + split manifest: "
                    f"{recomputed_value!r}",
                )

    # Measured rows MUST each be accompanied by exactly one
    # denominator audit row keyed by cell_name. The audit table is
    # a per-cell audit, not a global one, so duplicate rows and
    # orphan rows are both rejected. Measured rows with an empty
    # cell_name cannot be linked to any audit row and are rejected
    # outright.
    measured_cell_names: dict[str, int] = {}
    for row in report.per_cell_rows:
        if row.status != STATUS_MEASURED:
            continue
        cell_name = row.cell_name
        if not cell_name:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_measured_row_missing_cell_name",
                f"measured per_cell row for problem_slug "
                f"{row.problem_slug!r} has an empty cell_name; "
                f"audit linkage requires a non-empty cell_name",
            )
        measured_cell_names[cell_name] = measured_cell_names.get(cell_name, 0) + 1

    if measured_cell_names and len(report.denominator_audit_rows) == 0:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_missing_audit_for_measured",
            f"diagnostic report contains {sum(measured_cell_names.values())} "
            f"measured cell(s) but emits zero denominator_audit rows; the "
            f"audit table must accompany every measured row.",
        )

    audit_counts: dict[str, int] = {}
    for audit_row in report.denominator_audit_rows:
        audit_counts[audit_row.cell_name] = audit_counts.get(audit_row.cell_name, 0) + 1

    for cell_name, expected_count in measured_cell_names.items():
        actual = audit_counts.get(cell_name, 0)
        if actual == 0:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_audit_row_missing_for_measured_cell",
                f"measured cell {cell_name!r} has no matching "
                f"denominator_audit row",
            )
        if actual != expected_count:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_audit_row_duplicate",
                f"measured cell {cell_name!r} has {actual} matching "
                f"denominator_audit rows; expected {expected_count}",
            )

    orphan_audit = set(audit_counts) - set(measured_cell_names)
    if orphan_audit:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_audit_row_orphan",
            f"denominator_audit rows reference cell_name(s) not present "
            f"as measured per_cell rows: {sorted(orphan_audit)[:5]!r}",
        )

    # Cross-check the per-cell slugs against the canonical 235
    # SOL-ExecBench problem partition.
    #
    # The slug-set check is skipped ONLY when the canonical
    # reference directory itself is absent (test fixture with a
    # synthetic repo_root that omits the canonical tree). When
    # the directory is present, any drift (count mismatch,
    # duplicate slug, orphan slug, missing benchmark dir, etc.)
    # propagated by `enumerate_canonical_solbench_problems` MUST
    # fail the report because the diagnostic is supposed to fail
    # closed on exactly that scenario.
    repo_root_for_canonical = Path(__file__).resolve().parents[4]
    canonical_reference_dir = (
        repo_root_for_canonical
        / "references" / "NVIDIA_Manuals" / "solbench_problems"
    )
    if not canonical_reference_dir.is_dir():
        # Fixture environment: canonical tree is absent. Skip the
        # slug-set cross-check; the count + cohort + coverage
        # invariants still apply.
        return
    try:
        canonical_problems = enumerate_canonical_solbench_problems(
            repo_root=repo_root_for_canonical,
        )
    except AccuracyGateError as exc:
        # Canonical tree exists but has drifted. Fail closed.
        raise PerProblemRecordSchemaError(
            "diagnostic_report_canonical_partition_drift",
            f"canonical partition enumeration failed: "
            f"{exc.typed_reason}: {exc.message}",
        ) from exc
    canonical_slugs = frozenset(p.problem_slug for p in canonical_problems)
    non_canonical = seen_slugs - canonical_slugs
    if non_canonical:
        raise PerProblemRecordSchemaError(
            "diagnostic_report_non_canonical_slug",
            f"diagnostic report contains "
            f"{len(non_canonical)} per_cell row(s) whose "
            f"problem_slug is not in the canonical 235-problem "
            f"partition; first few: "
            f"{sorted(non_canonical)[:5]!r}",
        )

    # Cross-check each row's problem_category against the
    # canonical slug -> category map. A canonical slug with a
    # tampered category would otherwise pass the slug-set check
    # above but mis-classify the row in downstream category-
    # specific summaries.
    canonical_category_by_slug = {
        p.problem_slug: p.problem_category for p in canonical_problems
    }
    for row in report.per_cell_rows:
        canonical_category = canonical_category_by_slug.get(row.problem_slug)
        if canonical_category is None:
            # Should be unreachable after the non_canonical check
            # above, but guard defensively.
            continue
        if row.problem_category != canonical_category:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_problem_category_mismatch",
                f"per_cell row {row.problem_slug!r} has "
                f"problem_category={row.problem_category!r}, but the "
                f"canonical enumeration maps that slug to "
                f"{canonical_category!r}",
            )


def _recompute_coverage_from_per_cell_rows(
    per_cell_rows: Sequence[Any],
) -> dict:
    """Recompute the coverage buckets + reason histograms from
    per_cell rows using the same seven-mutually-exclusive-bucket
    priority that ``_build_coverage_summary`` uses (seventh
    bucket: host_arch_unsupported).
    """
    counts = {
        "friendly_cells": 0,
        "hostile_cells": 0,
        "typed_unavailable_cells": 0,
        "diagnostic_failed_cells": 0,
        "corpus_expansion_pending_cells": 0,
        "host_arch_unsupported_cells": 0,
        "measured_cell_count": 0,
    }
    reasons = {
        "typed_unavailable_reasons": {},
        "diagnostic_failed_reasons": {},
        "corpus_expansion_pending_by_reason": {},
        "host_arch_unsupported_reasons": {},
    }
    for row in per_cell_rows:
        bucket = row.predictor_bucket
        status = row.status
        reason = row.typed_reason or ""
        if status == STATUS_CORPUS_EXPANSION_PENDING:
            counts["corpus_expansion_pending_cells"] += 1
            reasons["corpus_expansion_pending_by_reason"][reason] = (
                reasons["corpus_expansion_pending_by_reason"].get(reason, 0) + 1
            )
        elif status == STATUS_DIAGNOSTIC_FAILED:
            counts["diagnostic_failed_cells"] += 1
            reasons["diagnostic_failed_reasons"][reason] = (
                reasons["diagnostic_failed_reasons"].get(reason, 0) + 1
            )
        elif status == STATUS_TYPED_UNAVAILABLE:
            counts["typed_unavailable_cells"] += 1
            reasons["typed_unavailable_reasons"][reason] = (
                reasons["typed_unavailable_reasons"].get(reason, 0) + 1
            )
        elif status == STATUS_HOST_ARCH_UNSUPPORTED:
            counts["host_arch_unsupported_cells"] += 1
            reasons["host_arch_unsupported_reasons"][reason] = (
                reasons["host_arch_unsupported_reasons"].get(reason, 0) + 1
            )
        elif status == STATUS_MEASURED:
            counts["measured_cell_count"] += 1
        elif bucket == _PREDICTOR_BUCKET_FRIENDLY:
            counts["friendly_cells"] += 1
        elif bucket == _PREDICTOR_BUCKET_HOSTILE:
            counts["hostile_cells"] += 1
    return {"counts": counts, "reasons": reasons}


_SMOKE_PREFIXED_SECTION_NAMES: tuple[tuple[str, str], ...] = (
    ("meta", "accuracy_diagnostic_meta"),
    ("coverage_summary", "accuracy_diagnostic_coverage_summary"),
    ("overall_accuracy_summary", "accuracy_diagnostic_overall_summary"),
    ("per_cell_accuracy", "accuracy_diagnostic_per_cell"),
    ("cohort_accuracy_summary", "accuracy_diagnostic_cohort_summary"),
    ("denominator_audit", "accuracy_diagnostic_denominator_audit"),
    ("denominator_box", "accuracy_diagnostic_denominator_box"),
    (
        "overall_user_target_summary",
        "accuracy_diagnostic_overall_user_target_summary",
    ),
    (
        "measured_cohort_payloads",
        "accuracy_diagnostic_measured_cohort_payloads",
    ),
)


def _detect_prefixed_diagnostic_namespace(data: Mapping[str, Any]) -> bool:
    """Return True iff the TOML carries any of the
    `accuracy_diagnostic_*` prefixed section names (i.e. it's
    a smoke-emitted combined TOML rather than a standalone
    diagnostic TOML).
    """

    for _unprefixed, prefixed in _SMOKE_PREFIXED_SECTION_NAMES:
        if prefixed in data:
            return True
    return False


def _resolve_report_section(
    data: Mapping[str, Any],
    unprefixed: str,
    prefixed: str,
    *,
    prefer_prefixed: bool,
) -> Any:
    """Return the report section.

    When `prefer_prefixed` is True (combined smoke TOML), the
    prefixed key is checked first so the smoke harness's own
    unprefixed `[meta]` does not shadow `[accuracy_diagnostic_meta]`.
    When `prefer_prefixed` is False (standalone diagnostic TOML),
    the unprefixed key is checked first. Returns `None` when
    neither key is present.
    """

    first, second = (prefixed, unprefixed) if prefer_prefixed else (unprefixed, prefixed)
    if first in data:
        return data[first]
    if second in data:
        return data[second]
    return None


def parse_diagnostic_report(text: str) -> DiagnosticReport:
    """Parse a TOML report back into a `DiagnosticReport`.

    Accepts either the standalone diagnostic TOML layout (with
    unprefixed `[meta]`, `[coverage_summary]`, etc.) OR the
    smoke-emitted layout (with `[accuracy_diagnostic_*]` prefix
    on every section). Requires all four core sections to be
    present (meta, coverage_summary, overall_accuracy_summary,
    per_cell_accuracy, cohort_accuracy_summary); missing
    sections raise `PerProblemRecordSchemaError` with typed
    reason `diagnostic_report_missing_section`.

    The denominator audit section is optional in parsing only
    when the report's measured cell count is zero; non-zero
    measured cell counts require the audit to be present.

    Runs `validate_diagnostic_report` so the returned report
    satisfies the 235 per_cell / 6 cohort cardinality invariants
    AND the canonical slug-set invariant.
    """

    import tomllib
    data = tomllib.loads(text)

    # When the TOML carries any `accuracy_diagnostic_*` section,
    # prefer the prefixed namespace for EVERY section. This
    # handles the combined smoke TOML case where the smoke
    # harness has emitted its own unprefixed `[meta]` block that
    # would otherwise shadow `[accuracy_diagnostic_meta]`.
    prefer_prefixed = _detect_prefixed_diagnostic_namespace(data)

    sections: dict[str, Any] = {}
    optional_sections: frozenset[str] = frozenset({
        "denominator_audit",
        "measured_cohort_payloads",
    })
    for unprefixed, prefixed in _SMOKE_PREFIXED_SECTION_NAMES:
        value = _resolve_report_section(
            data, unprefixed, prefixed,
            prefer_prefixed=prefer_prefixed,
        )
        if value is None and unprefixed not in optional_sections:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_missing_section",
                f"diagnostic report is missing required section "
                f"{unprefixed!r} (also accepted as "
                f"{prefixed!r}); available top-level keys: "
                f"{sorted(data.keys())!r}",
            )
        sections[unprefixed] = value

    meta = validate_diagnostic_meta_table(sections["meta"])
    coverage = validate_coverage_summary_table(sections["coverage_summary"])
    overall = validate_overall_accuracy_summary(
        sections["overall_accuracy_summary"],
    )
    per_cell = tuple(
        validate_per_problem_record(r)
        for r in sections["per_cell_accuracy"]
    )
    # Use the report's own meta-declared threshold so a parsed
    # report that was built with a non-default `accuracy_threshold`
    # validates cleanly against that same threshold.
    cohort = tuple(
        validate_cohort_accuracy_summary_row(
            r, accuracy_threshold=meta.accuracy_threshold,
        )
        for r in sections["cohort_accuracy_summary"]
    )
    audit_rows_raw = sections["denominator_audit"] or ()
    audit = tuple(
        validate_denominator_audit_row(r) for r in audit_rows_raw
    )
    denominator_box = validate_denominator_box(sections["denominator_box"])
    overall_user_target = validate_overall_user_target_summary(
        sections["overall_user_target_summary"],
    )
    measured_cohort_payloads_raw = (
        sections.get("measured_cohort_payloads") or ()
    )
    measured_cohort_payloads: list[MeasuredCohortPayloadRow] = []
    for index, row in enumerate(measured_cohort_payloads_raw):
        if not isinstance(row, Mapping):
            raise PerProblemRecordSchemaError(
                "diagnostic_report_split_aware_summary_mismatch",
                f"[[measured_cohort_payloads]] index {index} "
                f"is not a table",
            )
        try:
            measured_cohort_payloads.append(MeasuredCohortPayloadRow(
                cell_name=row["cell_name"],
                l1_calibrated_rel_err=float(row["l1_calibrated_rel_err"]),
                l2_calibrated_rel_err=float(row["l2_calibrated_rel_err"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise PerProblemRecordSchemaError(
                "diagnostic_report_split_aware_summary_mismatch",
                f"[[measured_cohort_payloads]] index {index}: "
                f"malformed entry ({exc})",
            ) from exc
    report = DiagnosticReport(
        per_cell_rows=per_cell,
        cohort_summary_rows=cohort,
        overall_summary=overall,
        coverage_summary=coverage,
        denominator_audit_rows=audit,
        meta=meta,
        denominator_box=denominator_box,
        overall_user_target_summary=overall_user_target,
        measured_cohort_payloads=tuple(measured_cohort_payloads),
    )
    validate_diagnostic_report(report)
    return report


__all__ = [
    "ACCURACY_DIAGNOSTIC_TYPED_REASONS",
    "CORPUS_EXPANSION_TYPED_REASONS",
    "HOST_ARCH_UNSUPPORTED_TYPED_REASONS",
    "HOST_ARCH_MANIFEST_RELATIVE_PATH",
    "HostArchAvailabilityEntry",
    "load_host_arch_availability_manifest",
    "detect_host_arch_unsupported",
    "STATUS_MEASURED",
    "STATUS_TYPED_UNAVAILABLE",
    "STATUS_DIAGNOSTIC_FAILED",
    "STATUS_CORPUS_EXPANSION_PENDING",
    "STATUS_HOST_ARCH_UNSUPPORTED",
    "CATEGORY_L1",
    "CATEGORY_L2",
    "CATEGORY_QUANT",
    "CATEGORY_FIB",
    "CANONICAL_CATEGORY_COUNTS",
    "NAN_SENTINEL",
    "NAN_TOO_FEW_MEASURED_CELLS",
    "COHORT_FRIENDLY",
    "COHORT_HOSTILE",
    "COHORT_TYPED_UNAVAILABLE",
    "CACHE_LEVEL_L1",
    "CACHE_LEVEL_L2",
    "CanonicalSolbenchProblem",
    "enumerate_canonical_solbench_problems",
    "canonical_typed_unavailable_cell_names",
    "PerProblemRecord",
    "PerProblemRecordSchema",
    "validate_per_problem_record",
    "PerProblemRecordSchemaError",
    "CohortAccuracySummaryRow",
    "OverallAccuracySummary",
    "validate_cohort_accuracy_summary_row",
    "validate_overall_accuracy_summary",
    "CoverageSummaryTable",
    "validate_coverage_summary_table",
    "DenominatorAuditRow",
    "validate_denominator_audit_row",
    "DenominatorBox",
    "validate_denominator_box",
    "OverallUserTargetSummary",
    "validate_overall_user_target_summary",
    "USER_TARGET_THRESHOLD_PCT",
    "compute_user_target_verdict_fields",
    "DiagnosticMetaTable",
    "validate_diagnostic_meta_table",
    "AccuracyGateError",
    "AccuracyDiagnosticResult",
    "compute_accuracy_diagnostic_gate",
    "DiagnosticReport",
    "LedgerLookupResult",
    "detect_cu_hand_port_present",
    "detect_ledger_entry_for_problem",
    "detect_cohort_classification",
    "build_accuracy_diagnostic_report",
    "serialize_diagnostic_report",
    "parse_diagnostic_report",
    "validate_diagnostic_report",
    "_resolve_implementation_commit_sha_from_git",
    "_verify_commit_sha_resolves",
    "predictor_callback_from_smoke_records",
    "serialize_diagnostic_sections_for_smoke_report",
]
