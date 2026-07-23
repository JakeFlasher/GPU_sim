"""Disk-discovery probe over the 235-problem corpus + the
landed ``baselines/`` tree + the previously-fit calibration
artifacts.

Builds the per-slug diagnostics dict and the env set the
``emit_ladder_state`` reducer consumes. Read-only: no nvcc, no
sol-execbench, no NCU invocation. Future harvest work will
refresh the diagnostics with measured evidence; this initial
pass reduces over the on-disk evidence already available.

Disk evidence summary:

- Problem-spec corpus: 235 markdown files under
  ``references/NVIDIA_Manuals/solbench_problems/``.
- ``*.cu`` presence: walk
  ``baselines/{L1,L2,Quant,FIB}/<slug>/{cutlass,cuda_cpp}/``;
  the legacy flat layout under ``baselines/L1_direct_ncu_binaries/``
  is NOT a problem-slug directory and is excluded.
- Previously-fit NCU label evidence: cells appearing in
  ``label_ledger_v1_sm120_partition.toml`` with a ``problem_``
  cell-name prefix. The 19 ``calib_*`` calibration smoke binaries
  in the same ledger are NOT corpus problems and are filtered
  out.
- Validation-only split: cells in
  ``historical_label_split_v1_sm120.toml`` with
  ``split == "validation_only"``.
- Simulator-unavailable exclusions: cells in
  ``sm120_simulator_unavailable_cells_v1.toml`` keyed by
  ``typed_reason`` — the two reasons map to the two
  Extension-pending env sets.
- Compile / sol-execbench correctness verification: cells
  appearing in ``sm120_compile_verification_probe_v1.toml`` with
  ``compile_verified == true`` (and optionally
  ``sol_execbench_correct == true``). Source-presence WITHOUT
  this probe evidence does NOT promote a cell to
  ``cu_compiles``; the conservative position is the reducer's
  fall-through to ``cu_compile_failed``.
- Arch-incompatible: slugs whose names match the NVFP4 family
  (the 5060 GB206 / sm_120 host lacks NVFP4 / TCGen05 / wgmma
  per CLAUDE.md). Hard-coded by slug-name prefix initially;
  pilot harvest work will replace this with the canonical-bucket
  predictor coverage map.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PROBLEMS_DIR = (
    _REPO_ROOT / "references" / "NVIDIA_Manuals" / "solbench_problems"
)
_BASELINES_ROOT = _REPO_ROOT / "baselines"
_CAL_DIR = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
)
_BASELINE_LEDGER = _CAL_DIR / "label_ledger_v1_sm120_partition.toml"
_HISTORICAL_SPLIT = _CAL_DIR / "historical_label_split_v1_sm120.toml"
_SIMULATOR_UNAVAILABLE = (
    _CAL_DIR / "sm120_simulator_unavailable_cells_v1.toml"
)
_COMPILE_VERIFICATION_PROBE = (
    _CAL_DIR / "sm120_compile_verification_probe_v1.toml"
)
_PILOT_NCU_STATUS = (
    _CAL_DIR / "sm120_pilot_ncu_repeatability_v1.toml"
)
_PILOT_TIER2_STATUS = (
    _CAL_DIR / "sm120_pilot_tier2_harvest_status_v1.toml"
)
_PILOT_PREDICTOR_PREDICTIONS = (
    _CAL_DIR / "sm120_pilot_predictor_predictions_v1.toml"
)
# Frozen sm_120 source-presence snapshot (sm86-coverage-loop R13). The two
# sm120 coverage TOMLs (readiness ladder + expansion diagnostics) are the design contract
# byte-immutable, but their fresh-emission determinism tests re-walk
# ``baselines/*/*/*.cu`` live. the design contract adds NEW sm_86 ports (new ``.cu`` files),
# which would change the live ``cu_present`` and break the frozen ladder's
# determinism without any sm_120 artifact legitimately changing. The canonical
# discovery therefore reads ``cu_present`` from this frozen snapshot (the
# sm_120-session source-presence) so new sm_86 ports do not perturb the frozen
# sm_120 ladder/diagnostics. Pass ``frozen_cu_presence_snapshot=None`` to
# ``probe_disk_state`` for the live walk (e.g. an sm_120 artifact-evolution
# session that legitimately re-emits the ladder).
_FROZEN_CU_PRESENCE_SNAPSHOT = (
    _CAL_DIR / "sm120_frozen_cu_presence_snapshot_v1.toml"
)

# Canonical category roots under ``baselines/``. The corpus is
# expected to live in exactly one of these per slug; cross-category
# duplication is not a current concern.
_BASELINE_CATEGORIES: tuple[str, ...] = ("L1", "L2", "Quant", "FIB")
_BASELINE_LANGUAGE_SUBDIRS: tuple[str, ...] = ("cutlass", "cuda_cpp")

# Slug-name prefix indicators for arch-incompatible cells the
# 5060 GB206 / sm_120 host cannot execute. NVFP4 is the only
# family currently named into the corpus; TCGen05 / wgmma show
# up in implementation details (not slug names) and are caught by
# the pilot-stage canonical-bucket resolution.
_ARCH_INCOMPAT_SUBSTRINGS: tuple[str, ...] = ("nvfp4",)

# Mapping from simulator-unavailable typed reasons (baseline
# manifest vocabulary) → ladder Extension env set names. Pinned
# alongside ``_VALID_SIMULATOR_OOD_TYPED_REASONS``.
_OOD_TYPED_REASON_TO_EXT_ENV: dict[str, str] = {
    "simulator_l2_metric_family_undercount": "ext_a2_pending",
    "hostile_subclass_uncalibrated": "ext_b_pending",
}


@dataclass(frozen=True, slots=True)
class PilotEvidence:
    """Per-slug pilot harvest evidence sourced from the pilot NCU
    repeatability, Tier-2 harvest status, and predictor-predictions
    artifacts.

    Fields default to ``None`` for slugs the pilot pipeline never
    touched; ``build_diagnostics_and_env`` translates a missing /
    ``None`` field to the sidecar's documented sentinel."""

    ncu_attempted: bool | None = None
    ncu_completed: bool | None = None
    ncu_cv_observed: float | None = None
    ncu_failure_reason: str | None = None
    tier2_attempted: bool | None = None
    tier2_completed: bool | None = None
    tier2_record_count: int | None = None
    tier2_truncation_relaxation_applied: bool | None = None
    tier2_failure_reason: str | None = None
    predictor_success: bool | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """All disk evidence the ladder reducer needs.

    ``corpus_slugs`` is a sorted tuple (ASCII order) of all 235
    problem-spec stems; the ladder emission consumes it as the
    canonical iteration order.

    ``correctness_failed`` carries slugs whose verification probe
    recorded a successful compile but at least one workload
    failed; those slugs cap at ``cu_compiles`` in the ladder
    (correctness did NOT pass, but compile did).

    ``pilot_evidence`` maps slug → ``PilotEvidence`` for the 12
    pilot cells (or however many appear in the pilot artifacts);
    slugs outside the pilot cohort are absent from the dict.
    """

    corpus_slugs: tuple[str, ...]
    cu_present: frozenset[str]
    ledger_present: frozenset[str]
    validation_only_present: frozenset[str]
    ext_a2_pending: frozenset[str]
    ext_b_pending: frozenset[str]
    arch_incompat_nvfp4_tcgen05: frozenset[str]
    compile_verified: frozenset[str]
    sol_execbench_correct_verified: frozenset[str]
    compile_failed: frozenset[str]
    correctness_failed: frozenset[str]
    pilot_evidence: dict[str, PilotEvidence] = field(default_factory=dict)


def probe_disk_state(
    *,
    problems_dir: Path = _PROBLEMS_DIR,
    baselines_root: Path = _BASELINES_ROOT,
    baseline_ledger: Path = _BASELINE_LEDGER,
    historical_split: Path = _HISTORICAL_SPLIT,
    simulator_unavailable: Path = _SIMULATOR_UNAVAILABLE,
    compile_verification_probe: Path = _COMPILE_VERIFICATION_PROBE,
    pilot_ncu_status: Path = _PILOT_NCU_STATUS,
    pilot_tier2_status: Path = _PILOT_TIER2_STATUS,
    pilot_predictor_predictions: Path = _PILOT_PREDICTOR_PREDICTIONS,
    frozen_cu_presence_snapshot: Path | None = _FROZEN_CU_PRESENCE_SNAPSHOT,
) -> DiscoveryResult:
    """Walk disk + read baseline artifacts → ``DiscoveryResult``.

    Read-only end-to-end; no subprocess invocation. Path
    arguments default to the in-repo canonical locations but are
    overridable for unit tests that plant tampered fixtures.

    ``cu_present`` (the per-slug ``*.cu`` source-presence) defaults to the
    FROZEN sm_120 snapshot ``frozen_cu_presence_snapshot`` rather than a live
    walk, so the design contract-byte-immutable sm_120 coverage ladder / expansion
    diagnostics stay deterministic when the sm_86 loop adds NEW ports' ``.cu``
    files (see ``_FROZEN_CU_PRESENCE_SNAPSHOT``). Pass
    ``frozen_cu_presence_snapshot=None`` to use the live ``baselines/`` walk
    (e.g. a future sm_120 artifact-evolution session, or a unit test planting a
    tampered ``baselines_root``). Synthetic ``DiscoveryResult`` reducer tests
    bypass this function and are unaffected either way.
    """
    corpus_slugs = _enumerate_problem_corpus(problems_dir)
    if frozen_cu_presence_snapshot is not None:
        # FAIL CLOSED: a requested frozen snapshot that is missing must NOT
        # silently degrade to a live walk (that would let the design contract-frozen
        # sm_120 ladder/diagnostics drift to a fresh live-disk reduction). The
        # ONLY way to get the live walk is the explicit opt-out
        # ``frozen_cu_presence_snapshot=None``.
        if not frozen_cu_presence_snapshot.is_file():
            raise FileNotFoundError(
                "frozen cu-presence snapshot not found: "
                f"{frozen_cu_presence_snapshot}. Pass "
                "frozen_cu_presence_snapshot=None to opt into the live "
                "baselines/ walk."
            )
        cu_present = _read_frozen_cu_presence(
            frozen_cu_presence_snapshot, set(corpus_slugs)
        )
    else:
        cu_present = _walk_cu_present(baselines_root, set(corpus_slugs))
    ledger_present = _read_ledger_problem_cells(baseline_ledger)
    validation_only_present = _read_validation_only_problem_cells(
        historical_split
    )
    ext_a2, ext_b = _read_simulator_unavailable_partitions(
        simulator_unavailable
    )
    arch_incompat = _classify_arch_incompat(corpus_slugs)
    (
        verified_compile,
        verified_correct,
        failed_compile,
        failed_correctness,
    ) = _read_compile_verification_probe(compile_verification_probe)
    pilot_evidence = _read_pilot_evidence(
        ncu_path=pilot_ncu_status,
        tier2_path=pilot_tier2_status,
        predictor_path=pilot_predictor_predictions,
    )
    return DiscoveryResult(
        corpus_slugs=corpus_slugs,
        cu_present=frozenset(cu_present),
        ledger_present=frozenset(ledger_present),
        validation_only_present=frozenset(validation_only_present),
        ext_a2_pending=frozenset(ext_a2),
        ext_b_pending=frozenset(ext_b),
        arch_incompat_nvfp4_tcgen05=frozenset(arch_incompat),
        compile_verified=frozenset(verified_compile),
        sol_execbench_correct_verified=frozenset(verified_correct),
        compile_failed=frozenset(failed_compile),
        correctness_failed=frozenset(failed_correctness),
        pilot_evidence=pilot_evidence,
    )


def build_diagnostics_and_env(
    result: DiscoveryResult,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Project disk evidence into the reducer's input shape.

    Per-slug diagnostic booleans are set as follows from
    on-disk evidence (NOT from source presence alone):

    - ``cu_compiles`` → True when slug ∈ ``compile_verified``
      (probe artifact records a successful build) OR slug ∈
      ``ledger_present`` (baseline NCU run requires a working
      build to have completed). Source presence WITHOUT either
      of those does NOT set this field; the reducer's fall-
      through assigns ``cu_compile_failed`` in that case
      (conservative position: no evidence of successful build).
    - ``sol_execbench_correct`` → True when slug ∈
      ``sol_execbench_correct_verified`` (probe artifact records
      every workload status==PASSED) OR slug ∈ ``ledger_present``
      (baseline NCU run requires a passing solution).
    - ``ncu_label_available``, ``tier2_trace_available``,
      ``predictor_prediction_available`` → True when slug ∈
      ``ledger_present`` (baseline ledger presence proves the
      cell completed the NCU + Tier-2 + simulator + calibrator
      pipeline).
    - ``mape_scored`` → True when slug ∈ ``validation_only_present``
      (the baseline validation report scored a per-cell MAPE for
      that cell).
    - ``compile_failed`` → True when slug ∈ ``compile_failed``
      (probe records the build attempted and exited non-zero).

    The env set carries the four-key membership map the reducer
    expects: ``cu_present``, ``arch_incompat_nvfp4_tcgen05``,
    ``ext_a2_pending``, ``ext_b_pending``.
    """
    diagnostics: dict[str, dict[str, Any]] = {}
    for slug in result.corpus_slugs:
        d: dict[str, Any] = {}
        correctness_evidence = (
            slug in result.sol_execbench_correct_verified
            or slug in result.ledger_present
        )
        # Correctness success implies compile success (a kernel
        # cannot produce PASSED results without first compiling),
        # so any cell with correctness evidence also has compile
        # evidence. Cells with only direct compile evidence (probe
        # surface success without correctness rows) also count.
        compile_evidence = (
            slug in result.compile_verified
            or slug in result.ledger_present
            or correctness_evidence
        )
        if compile_evidence:
            d["cu_compiles"] = True
        if correctness_evidence:
            d["sol_execbench_correct"] = True
        if slug in result.ledger_present:
            d["ncu_label_available"] = True
            d["tier2_trace_available"] = True
            d["predictor_prediction_available"] = True
        if slug in result.validation_only_present:
            d["mape_scored"] = True
        if slug in result.compile_failed:
            d["compile_failed"] = True
        # Pilot-evidence overlay: when the slug has a row in the
        # pilot NCU / Tier-2 / predictor-predictions artifacts,
        # fold its fields into the diagnostics dict on top of the
        # ledger-derived defaults. Pilot evidence is ADDITIVE — it
        # can only refine, not retract: missing fields keep the
        # ledger-derived values. The ladder reducer consumes the
        # merged dict.
        pilot = result.pilot_evidence.get(slug)
        if pilot is not None:
            if pilot.ncu_attempted is not None:
                d["ncu_attempted"] = pilot.ncu_attempted
            if pilot.ncu_cv_observed is not None:
                d["ncu_cv_observed"] = pilot.ncu_cv_observed
            if pilot.ncu_failure_reason is not None:
                d["ncu_quarantine_reason"] = pilot.ncu_failure_reason
            if pilot.ncu_completed is True:
                d["ncu_label_available"] = True
            if pilot.tier2_attempted is not None:
                d["tier2_attempted"] = pilot.tier2_attempted
            if pilot.tier2_completed is True:
                d["tier2_trace_available"] = True
            if pilot.tier2_record_count is not None:
                # Only populate truncation_record_count when the
                # producer's per-launch staging cap was hit and the
                # offline-calibration truncation-relaxation flag
                # was applied. Untruncated rows leave the diagnostics
                # field as NULL_INT to signal "no truncation event
                # to report" rather than "0 records captured".
                if pilot.tier2_truncation_relaxation_applied:
                    d["tier2_truncation_record_count"] = (
                        pilot.tier2_record_count
                    )
            if pilot.tier2_failure_reason is not None:
                d["tier2_quarantine_reason"] = pilot.tier2_failure_reason
            if pilot.predictor_success is True:
                d["predictor_prediction_available"] = True
        diagnostics[slug] = d
    env: dict[str, set[str]] = {
        "cu_present": set(result.cu_present),
        "arch_incompat_nvfp4_tcgen05": set(result.arch_incompat_nvfp4_tcgen05),
        "ext_a2_pending": set(result.ext_a2_pending),
        "ext_b_pending": set(result.ext_b_pending),
    }
    return diagnostics, env


def _enumerate_problem_corpus(problems_dir: Path) -> tuple[str, ...]:
    slugs = []
    for entry in problems_dir.iterdir():
        if entry.is_file() and entry.suffix == ".md":
            slugs.append(entry.stem)
    return tuple(sorted(slugs))


def _walk_cu_present(
    baselines_root: Path, corpus_slugs: set[str]
) -> set[str]:
    present: set[str] = set()
    for category in _BASELINE_CATEGORIES:
        category_dir = baselines_root / category
        if not category_dir.is_dir():
            continue
        for slug_dir in category_dir.iterdir():
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            if slug not in corpus_slugs:
                continue
            for lang_subdir in _BASELINE_LANGUAGE_SUBDIRS:
                lang_dir = slug_dir / lang_subdir
                if not lang_dir.is_dir():
                    continue
                if any(p.suffix == ".cu" for p in lang_dir.iterdir()):
                    present.add(slug)
                    break
    return present


def _read_frozen_cu_presence(
    snapshot_path: Path, corpus_slugs: set[str]
) -> set[str]:
    """Load the frozen sm_120 ``cu_present`` snapshot (the sm_120-session
    source-presence). Intersected with the live corpus so a snapshot slug that
    no longer maps to a corpus problem is silently dropped (defensive; the
    snapshot is expected to be a subset of the corpus)."""
    data = tomllib.loads(snapshot_path.read_text("utf-8"))
    slugs = data.get("cu_present_slugs", [])
    return {s for s in slugs if isinstance(s, str) and s in corpus_slugs}


def _read_ledger_problem_cells(ledger_path: Path) -> set[str]:
    data = tomllib.loads(ledger_path.read_text("utf-8"))
    cells: set[str] = set()
    for entry in data.get("entries", []):
        cell = entry.get("cell_name", "")
        if isinstance(cell, str) and cell.startswith("problem_"):
            cells.add(cell)
    return cells


def _read_validation_only_problem_cells(split_path: Path) -> set[str]:
    data = tomllib.loads(split_path.read_text("utf-8"))
    cells: set[str] = set()
    for entry in data.get("splits", []):
        slug = entry.get("slug", "")
        if (
            isinstance(slug, str)
            and slug.startswith("problem_")
            and entry.get("split") == "validation_only"
        ):
            cells.add(slug)
    return cells


def _read_simulator_unavailable_partitions(
    manifest_path: Path,
) -> tuple[set[str], set[str]]:
    data = tomllib.loads(manifest_path.read_text("utf-8"))
    ext_a2: set[str] = set()
    ext_b: set[str] = set()
    for entry in data.get("excluded", []):
        slug = entry.get("cell_name", "")
        reason = entry.get("typed_reason", "")
        if not isinstance(slug, str) or not slug.startswith("problem_"):
            continue
        target_env = _OOD_TYPED_REASON_TO_EXT_ENV.get(reason)
        if target_env == "ext_a2_pending":
            ext_a2.add(slug)
        elif target_env == "ext_b_pending":
            ext_b.add(slug)
    return ext_a2, ext_b


def _classify_arch_incompat(corpus_slugs: tuple[str, ...]) -> set[str]:
    flagged: set[str] = set()
    for slug in corpus_slugs:
        slug_lower = slug.lower()
        if any(token in slug_lower for token in _ARCH_INCOMPAT_SUBSTRINGS):
            flagged.add(slug)
    return flagged


def _read_compile_verification_probe(
    probe_path: Path,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Read the verification-probe artifact (if present) and
    return (compile_verified, sol_execbench_correct_verified,
    compile_failed, correctness_failed) slug sets.

    ``correctness_failed`` carries slugs where compile_verified
    is true but sol_execbench_correct_verified is false — the
    binary built cleanly but at least one workload failed the
    correctness check. The diagnostics sidecar uses this signal
    to set the per-cell ``correctness_failed`` field.

    Missing probe artifact is non-fatal: the discovery degrades
    gracefully to "no probe evidence", which means non-ledger
    cells with source files fall through to ``cu_compile_failed``
    via the reducer's conservative semantics. The probe must be
    refreshed by ``compile_verification_probe.refresh_probe(...)``
    when new source files land or when the harness traces are
    re-captured.
    """
    if not probe_path.is_file():
        return set(), set(), set(), set()
    data = tomllib.loads(probe_path.read_text("utf-8"))
    verified_compile: set[str] = set()
    verified_correct: set[str] = set()
    failed_compile: set[str] = set()
    failed_correctness: set[str] = set()
    for entry in data.get("entries", []):
        slug = entry.get("slug", "")
        if not isinstance(slug, str):
            continue
        compile_ok = bool(entry.get("compile_verified", False))
        correct_ok = bool(entry.get("sol_execbench_correct_verified", False))
        if compile_ok:
            verified_compile.add(slug)
        if correct_ok:
            verified_correct.add(slug)
        if bool(entry.get("compile_failed", False)):
            failed_compile.add(slug)
        if compile_ok and not correct_ok:
            failed_correctness.add(slug)
    return verified_compile, verified_correct, failed_compile, failed_correctness


def _read_pilot_evidence(
    *,
    ncu_path: Path,
    tier2_path: Path,
    predictor_path: Path,
) -> dict[str, PilotEvidence]:
    """Read the pilot NCU repeatability, Tier-2 harvest status,
    and predictor-predictions artifacts, returning a per-slug
    ``PilotEvidence`` overlay.

    Each missing artifact contributes a no-op (slug → field stays
    ``None``); slugs absent from all three artifacts have no entry
    in the returned dict. The diagnostics emitter consumes this
    overlay to refresh the per-cell ``ncu_attempted`` /
    ``ncu_cv_observed`` / ``ncu_quarantine_reason`` /
    ``tier2_attempted`` / ``tier2_truncation_record_count`` /
    ``tier2_quarantine_reason`` / ``predictor_prediction_available``
    fields for the pilot cohort.

    Closed-vocabulary semantics:

    * ``ncu_failure_reason`` is the verbatim NCU artifact token
      (e.g. ``ncu_kernel_name_not_pinned``, ``ncu_subprocess_failed``,
      ``ncu_repeatability_failed``). The ladder reducer treats the
      empty string ``""`` as the success sentinel; ATen-only pilots
      that pre-spawn-skipped get ``ncu_failure_reason=""`` because
      a pre-spawn skip is NOT a measurement-quality quarantine —
      the cell is structurally non-admissible at the NCU gate, but
      the diagnostics layer reflects this via
      ``ncu_attempted=False`` rather than via the quarantine token.
    * ``tier2_failure_reason`` follows the same pattern: empty for
      success rows, verbatim token for typed failures
      (``tier2_budget_exceeded``, ``tier2_no_launches_captured``,
      etc.).
    """
    evidence: dict[str, PilotEvidence] = {}

    def _ensure(slug: str) -> PilotEvidence:
        if slug not in evidence:
            evidence[slug] = PilotEvidence()
        return evidence[slug]

    def _update(slug: str, **fields: Any) -> None:
        current = _ensure(slug)
        merged = {
            "ncu_attempted": current.ncu_attempted,
            "ncu_completed": current.ncu_completed,
            "ncu_cv_observed": current.ncu_cv_observed,
            "ncu_failure_reason": current.ncu_failure_reason,
            "tier2_attempted": current.tier2_attempted,
            "tier2_completed": current.tier2_completed,
            "tier2_record_count": current.tier2_record_count,
            "tier2_truncation_relaxation_applied": (
                current.tier2_truncation_relaxation_applied
            ),
            "tier2_failure_reason": current.tier2_failure_reason,
            "predictor_success": current.predictor_success,
        }
        merged.update(fields)
        evidence[slug] = PilotEvidence(**merged)

    if ncu_path.is_file():
        doc = tomllib.loads(ncu_path.read_text("utf-8"))
        for row in doc.get("entries", []):
            slug = row.get("slug", "")
            if not isinstance(slug, str) or not slug:
                continue
            attempted = bool(row.get("attempted", False))
            completed = bool(row.get("completed", False))
            # The ladder reducer's NCU quarantine check uses
            # the maximum of L1/L2 CV vs the 0.05 cap, so we
            # surface that single scalar to the diagnostics
            # sidecar. Both null → leave as None.
            l1_cv = row.get("ncu_l1_cv")
            l2_cv = row.get("ncu_l2_cv")
            if l1_cv is None and l2_cv is None:
                cv = None
            else:
                cv = max(
                    float(l1_cv if l1_cv is not None else 0.0),
                    float(l2_cv if l2_cv is not None else 0.0),
                )
            # Only treat the artifact's failure_reason as a
            # measurement-quality quarantine when the NCU
            # subprocess was actually spawned (attempted=True)
            # AND the artifact records a non-empty reason. A
            # pre-spawn skip (attempted=False) gets ""; the
            # ladder reducer relies on attempted+reason to fire
            # the quarantine cap.
            raw_reason = row.get("failure_reason", "")
            if attempted and isinstance(raw_reason, str) and raw_reason:
                ncu_reason = raw_reason
            else:
                ncu_reason = ""
            _update(
                slug,
                ncu_attempted=attempted,
                ncu_completed=completed,
                ncu_cv_observed=cv,
                ncu_failure_reason=ncu_reason,
            )

    if tier2_path.is_file():
        doc = tomllib.loads(tier2_path.read_text("utf-8"))
        for row in doc.get("entries", []):
            slug = row.get("slug", "")
            if not isinstance(slug, str) or not slug:
                continue
            attempted = bool(row.get("attempted", False))
            completed = bool(row.get("completed", False))
            record_count = row.get("tier2_record_count")
            truncation = bool(
                row.get("truncation_relaxation_applied", False)
            )
            raw_reason = row.get("failure_reason", "")
            tier2_reason = (
                raw_reason
                if isinstance(raw_reason, str) and raw_reason
                else ""
            )
            _update(
                slug,
                tier2_attempted=attempted,
                tier2_completed=completed,
                tier2_record_count=(
                    int(record_count)
                    if isinstance(record_count, int) and record_count >= 0
                    else None
                ),
                tier2_truncation_relaxation_applied=truncation,
                tier2_failure_reason=tier2_reason,
            )

    if predictor_path.is_file():
        doc = tomllib.loads(predictor_path.read_text("utf-8"))
        for row in doc.get("entries", []):
            slug = row.get("slug", "")
            if not isinstance(slug, str) or not slug:
                continue
            success = bool(row.get("predictor_success", False))
            _update(slug, predictor_success=success)

    return evidence
