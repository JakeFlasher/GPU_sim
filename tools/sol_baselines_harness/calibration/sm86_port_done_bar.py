"""sm_86 per-problem DONE-bar validator + coverage-ledger schema.

A ported (slug, language) cell is DONE on the live sm_86 host when ALL
of the following hold:

* its ``solution.json`` declares ``target_hardware == ["LOCAL"]``, carries
  an explicit ``-gencode arch=compute_86,code=sm_86`` cuda_cflag, and uses
  a language in ``{cuda_cpp, cutlass}``;
* every workload in the problem's ``workload.jsonl`` passed correctness
  (``workloads_passed == workloads_total`` with a positive total);
* a full counter-free evidence bundle was emitted whose ``manifest.json``
  carries ``evidence_label == "Profiled-local-sm_86"`` and
  ``host.kind == "local_sm_8_6"``, with NO surface in ``failed`` state and
  every non-``ok`` surface carrying a ``waiver_reason``.

The validator re-derives the surface health from the manifest itself, so a
manifest whose ``evidence_label`` overstates the real surface-status map
(e.g. claims ``Profiled-local-sm_86`` while a surface is ``failed``) is
still flagged.

This module also defines the per-problem coverage-ledger row schema
(slug, language, workloads_passed/total, evidence_label, bundle_path,
surface_status map, plus the computed done verdict). The bulk-port
milestone expands the ledger to the full ported set; this module seeds it
from the existing pilot bundles and provides the consistency gate.

Import-only inside calibration / harness paths; carries no NCU or
clock-policy surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback path
    import tomli as tomllib  # type: ignore[no-redef]


# --- The DONE-bar contract (closed vocabularies) ----------------------

REQUIRED_TARGET_HARDWARE: tuple[str, ...] = ("LOCAL",)
#: Load-bearing fragment of the explicit gencode flag. Matches both the
#: single-string ``"-gencode arch=compute_86,code=sm_86"`` form and the
#: split ``["-gencode", "arch=compute_86,code=sm_86"]`` form.
REQUIRED_GENCODE_FRAGMENT = "arch=compute_86,code=sm_86"
VALID_LANGUAGES: tuple[str, ...] = ("cuda_cpp", "cutlass")
REQUIRED_EVIDENCE_LABEL = "Profiled-local-sm_86"
REQUIRED_HOST_KIND = "local_sm_8_6"

#: A surface in this state disqualifies the cell unconditionally.
DISQUALIFYING_SURFACE_STATUS = "failed"
#: A surface in any of these states is acceptable ONLY with a
#: ``waiver_reason`` (the manifest lint already enforces this; the
#: validator double-checks so an overstating manifest is caught).
WAIVER_REQUIRED_SURFACE_STATUSES: frozenset[str] = frozenset(
    {"unavailable", "blocked", "skipped", "degraded"}
)
ACCEPTABLE_OK_STATUS = "ok"

#: The full five eval-driving counter-free surfaces a DONE bundle MUST carry
#: (the plan's "full five-surface" requirement). A skipped-with-waiver entry
#: still counts as present. ``cache_prediction`` is an optional passive 6th
#: surface and is NOT required.
REQUIRED_EVAL_DRIVING_SURFACES: tuple[str, ...] = (
    "static",
    "sanitizer",
    "nvbit_metrics",
    "cupti_activity",
    "nsys",
)

#: The six known manifest surface names (the five required eval-driving
#: surfaces + the optional passive cache_prediction). Keeps the parameterised
#: blocking-reason vocabulary closed over real surface names.
KNOWN_SURFACE_NAMES: frozenset[str] = frozenset(REQUIRED_EVAL_DRIVING_SURFACES) | {
    "cache_prediction"
}

#: Locked surface-status vocabulary (mirrors the manifest lint layer). A
#: required or optional-present surface whose status falls outside this set
#: (empty / missing / unknown) cannot certify DONE.
VALID_SURFACE_STATUSES: frozenset[str] = WAIVER_REQUIRED_SURFACE_STATUSES | {
    DISQUALIFYING_SURFACE_STATUS,
    ACCEPTABLE_OK_STATUS,
}

#: Closed base vocabulary of done-bar blocking reasons (surface reasons
#: are parameterised below).
DONE_BAR_BASE_REASONS: tuple[str, ...] = (
    "target_hardware_not_local",
    "missing_sm86_gencode",
    "invalid_language",
    "workloads_incomplete",
    "evidence_label_not_profiled_local_sm86",
    "host_kind_not_local_sm_8_6",
    "ledger_evidence_label_mismatch",
    "surface_status_map_mismatch",
)
SURFACE_FAILED_PREFIX = "surface_failed:"
SURFACE_UNWAIVED_PREFIX = "surface_unwaived:"
MISSING_SURFACE_PREFIX = "missing_surface:"
SURFACE_STATUS_INVALID_PREFIX = "surface_status_invalid:"
#: A surface that actually ran (status ok) but whose recorded
#: ``cache_identity.solution_source_hash`` no longer matches the current
#: solution.json bytes — the bundle was produced from a stale solution (a
#: build-flag edit busts cache identity per the harness cache contract).
SURFACE_PROVENANCE_STALE_PREFIX = "surface_provenance_stale:"

DONE_BAR_SCHEMA_VERSION = 1
COVERAGE_LEDGER_SCHEMA_VERSION = 1

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
DEFAULT_COVERAGE_LEDGER_PATH = _THIS_DIR / "sm86_coverage_ledger_v1.toml"


class Sm86PortDoneBarError(ValueError):
    """Raised by :func:`assert_done` when a cell fails the DONE bar."""


@dataclass(frozen=True)
class DoneBarVerdict:
    is_done: bool
    blocking_reasons: tuple[str, ...]


def reason_is_in_vocabulary(reason: str) -> bool:
    """True iff ``reason`` belongs to the closed done-bar vocabulary.

    Parameterised surface reasons are closed over real surface names:
    ``missing_surface:`` only over the five required eval-driving surfaces;
    ``surface_failed:`` / ``surface_unwaived:`` / ``surface_status_invalid:``
    over the six known manifest surface names.
    """
    if reason in DONE_BAR_BASE_REASONS:
        return True
    for prefix, allowed in (
        (MISSING_SURFACE_PREFIX, frozenset(REQUIRED_EVAL_DRIVING_SURFACES)),
        (SURFACE_FAILED_PREFIX, KNOWN_SURFACE_NAMES),
        (SURFACE_UNWAIVED_PREFIX, KNOWN_SURFACE_NAMES),
        (SURFACE_STATUS_INVALID_PREFIX, KNOWN_SURFACE_NAMES),
        (SURFACE_PROVENANCE_STALE_PREFIX, KNOWN_SURFACE_NAMES),
    ):
        if reason.startswith(prefix):
            return reason[len(prefix):] in allowed
    return False


def _cuda_cflags(solution: Mapping) -> list[str]:
    spec = solution.get("spec", {}) if isinstance(solution, Mapping) else {}
    compile_options = spec.get("compile_options", {}) or {}
    flags = compile_options.get("cuda_cflags")
    if flags is None:
        # Tolerate a flattened spec.cuda_cflags shape.
        flags = spec.get("cuda_cflags", [])
    return [str(f) for f in (flags or [])]


def sm86_solution_spec_blocking_reasons(solution: Mapping) -> tuple[str, ...]:
    """Solution-side DONE-bar checks (the design contract(a) contract): exactly
    ``target_hardware == ["LOCAL"]``, an explicit
    ``-gencode arch=compute_86,code=sm_86`` cuda_cflag, and exactly one
    language in ``{cuda_cpp, cutlass}``. Returned in deterministic order.

    Factored out so the scaffold can verify it emits the design contract-conforming sm_86
    specs against the same single source of truth the DONE bar enforces.
    """
    spec = solution.get("spec", {}) if isinstance(solution, Mapping) else {}
    reasons: list[str] = []
    if list(spec.get("target_hardware", []) or []) != list(REQUIRED_TARGET_HARDWARE):
        reasons.append("target_hardware_not_local")
    if REQUIRED_GENCODE_FRAGMENT not in " ".join(_cuda_cflags(solution)):
        reasons.append("missing_sm86_gencode")
    # The DONE bar certifies exactly one (slug, language) cell, so a
    # multi-language spec (or an empty one) is rejected.
    languages = list(spec.get("languages", []) or [])
    if tuple(languages) not in (("cuda_cpp",), ("cutlass",)):
        reasons.append("invalid_language")
    return tuple(reasons)


def evaluate_done_bar(
    *,
    solution: Mapping,
    manifest: Mapping,
    workloads_passed: int,
    workloads_total: int,
    ledger_evidence_label: str | None = None,
    ledger_surface_status: Mapping[str, str] | None = None,
    solution_source_hash: str | None = None,
) -> DoneBarVerdict:
    """Return the DONE-bar verdict for one cell.

    ``solution`` / ``manifest`` are the parsed ``solution.json`` /
    ``manifest.json`` dicts. ``ledger_*`` are optional cross-checks of a
    coverage-ledger row against the live manifest. ``solution_source_hash``
    is the SHA-256 of the *current* solution.json bytes (the scheme the
    surfaces orchestrator stamps into each surface's
    ``cache_identity.solution_source_hash``); when supplied, every surface
    that actually RAN (status ``ok``) must carry a matching hash, else the
    bundle is provenance-stale (a build-flag edit busts cache identity).
    Blocking reasons are returned in a deterministic order.
    """
    reasons: list[str] = []

    # (a) solution.json contract -------------------------------------
    reasons.extend(sm86_solution_spec_blocking_reasons(solution))

    # (c) correctness ------------------------------------------------
    if workloads_total <= 0 or workloads_passed != workloads_total:
        reasons.append("workloads_incomplete")

    # (d) evidence bundle / manifest ---------------------------------
    if manifest.get("evidence_label") != REQUIRED_EVIDENCE_LABEL:
        reasons.append("evidence_label_not_profiled_local_sm86")
    host = manifest.get("host", {}) if isinstance(manifest.get("host"), Mapping) else {}
    if host.get("kind") != REQUIRED_HOST_KIND:
        reasons.append("host_kind_not_local_sm_8_6")

    surfaces = manifest.get("surfaces", {})
    surfaces_map = surfaces if isinstance(surfaces, Mapping) else {}
    # The full five eval-driving surfaces must each be present as a manifest
    # entry (a skipped-with-waiver entry counts as present); missing-surface
    # reasons precede the status checks. An empty/partial/non-mapping surfaces
    # block therefore cannot be certified DONE.
    for name in REQUIRED_EVAL_DRIVING_SURFACES:
        if not isinstance(surfaces_map.get(name), Mapping):
            reasons.append(f"{MISSING_SURFACE_PREFIX}{name}")
    # Validate status + health for every KNOWN surface present (required + the
    # optional cache_prediction). A present surface whose status is outside the
    # locked vocabulary (empty / missing / unknown) is flagged and skips the
    # failed/unwaived checks. Unknown surface names are left to the manifest
    # lint (rejected upstream), keeping this reason vocabulary closed.
    for name in sorted(KNOWN_SURFACE_NAMES):
        entry = surfaces_map.get(name)
        if not isinstance(entry, Mapping):
            continue
        status = entry.get("status")
        if status not in VALID_SURFACE_STATUSES:
            reasons.append(f"{SURFACE_STATUS_INVALID_PREFIX}{name}")
            continue
        if status == DISQUALIFYING_SURFACE_STATUS:
            reasons.append(f"{SURFACE_FAILED_PREFIX}{name}")
        elif status in WAIVER_REQUIRED_SURFACE_STATUSES and not entry.get(
            "waiver_reason"
        ):
            reasons.append(f"{SURFACE_UNWAIVED_PREFIX}{name}")
        # Provenance: a surface that actually RAN (ok) must have been
        # produced from the CURRENT solution.json. The orchestrator stamps
        # SHA-256(solution.json bytes) into cache_identity.solution_source_hash;
        # a build-flag edit (e.g. adding the sm_86 gencode) changes that hash,
        # so a bundle whose ok-surface hash no longer matches is stale evidence
        # and cannot certify DONE. Skipped/waived surfaces did not run and are
        # exempt.
        if status == ACCEPTABLE_OK_STATUS and solution_source_hash is not None:
            identity = entry.get("cache_identity")
            stored = (
                identity.get("solution_source_hash")
                if isinstance(identity, Mapping)
                else None
            )
            if stored != solution_source_hash:
                reasons.append(f"{SURFACE_PROVENANCE_STALE_PREFIX}{name}")

    # Optional ledger cross-checks -----------------------------------
    if (
        ledger_evidence_label is not None
        and ledger_evidence_label != manifest.get("evidence_label")
    ):
        reasons.append("ledger_evidence_label_mismatch")
    if ledger_surface_status is not None:
        manifest_map = {
            name: entry.get("status")
            for name, entry in surfaces_map.items()
            if isinstance(entry, Mapping)
        }
        if dict(ledger_surface_status) != manifest_map:
            reasons.append("surface_status_map_mismatch")

    return DoneBarVerdict(is_done=not reasons, blocking_reasons=tuple(reasons))


def assert_done(
    *,
    solution: Mapping,
    manifest: Mapping,
    workloads_passed: int,
    workloads_total: int,
    ledger_evidence_label: str | None = None,
    ledger_surface_status: Mapping[str, str] | None = None,
) -> None:
    """Raise :class:`Sm86PortDoneBarError` if the cell is not DONE."""
    verdict = evaluate_done_bar(
        solution=solution,
        manifest=manifest,
        workloads_passed=workloads_passed,
        workloads_total=workloads_total,
        ledger_evidence_label=ledger_evidence_label,
        ledger_surface_status=ledger_surface_status,
    )
    if not verdict.is_done:
        raise Sm86PortDoneBarError(
            "sm86_port_not_done: " + ", ".join(verdict.blocking_reasons)
        )


# --- Coverage-ledger row schema ---------------------------------------

@dataclass(frozen=True)
class CoverageRow:
    slug: str
    language: str
    category: str
    baseline_dir: str
    bundle_path: str
    workloads_passed: int
    workloads_total: int
    evidence_label: str
    surface_status: dict[str, str] = field(default_factory=dict)
    done: bool = False
    blocking_reasons: tuple[str, ...] = ()
    # A cell whose correctness genuinely fails on sm_86 carries
    # deferred_correctness=True + a concrete deferred_reason (the design contract / the
    # deferred.json ledger). Mutually exclusive with done=True.
    deferred_correctness: bool = False
    deferred_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "language": self.language,
            "category": self.category,
            "baseline_dir": self.baseline_dir,
            "bundle_path": self.bundle_path,
            "workloads_passed": self.workloads_passed,
            "workloads_total": self.workloads_total,
            "evidence_label": self.evidence_label,
            "surface_status": dict(self.surface_status),
            "done": self.done,
            "blocking_reasons": list(self.blocking_reasons),
            "deferred_correctness": self.deferred_correctness,
            "deferred_reason": self.deferred_reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> "CoverageRow":
        return cls(
            slug=str(raw["slug"]),
            language=str(raw["language"]),
            category=str(raw["category"]),
            baseline_dir=str(raw["baseline_dir"]),
            bundle_path=str(raw["bundle_path"]),
            workloads_passed=int(raw["workloads_passed"]),
            workloads_total=int(raw["workloads_total"]),
            evidence_label=str(raw["evidence_label"]),
            surface_status={str(k): str(v) for k, v in dict(raw.get("surface_status", {})).items()},
            done=bool(raw.get("done", False)),
            blocking_reasons=tuple(str(r) for r in raw.get("blocking_reasons", [])),
            deferred_correctness=bool(raw.get("deferred_correctness", False)),
            deferred_reason=str(raw.get("deferred_reason", "")),
        )


def load_coverage_ledger(
    path: Path = DEFAULT_COVERAGE_LEDGER_PATH,
) -> tuple[dict, list[CoverageRow]]:
    """Parse the coverage ledger TOML into (meta, rows)."""
    path = Path(path)
    if not path.is_file():
        raise Sm86PortDoneBarError(f"coverage ledger missing: {path}")
    with path.open("rb") as handle:
        doc = tomllib.load(handle)
    meta = dict(doc.get("meta", {}))
    rows = [CoverageRow.from_dict(r) for r in doc.get("rows", [])]
    return meta, rows


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _solution_file_sha256(path: Path) -> str:
    """SHA-256 of solution.json's file bytes — the exact scheme the surfaces
    orchestrator (`dispatch_plan._sha256_of_file`) stamps into each surface's
    ``cache_identity.solution_source_hash`` and that `report.py` re-derives to
    detect stale bundles. Kept in sync with those by definition (file bytes)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_row_from_disk(
    row: CoverageRow, *, repo_root: Path = _REPO_ROOT
) -> DoneBarVerdict:
    """Re-derive a row's DONE-bar verdict from its on-disk solution.json +
    manifest.json (recompute, do not trust the stored verdict). Includes the
    bundle-provenance check: each ok surface's recorded solution_source_hash
    must match SHA-256(current solution.json)."""
    repo_root = Path(repo_root)
    solution_path = repo_root / row.baseline_dir / "solution.json"
    manifest_path = repo_root / row.bundle_path / "manifest.json"
    if not solution_path.is_file():
        raise Sm86PortDoneBarError(f"solution.json missing: {solution_path}")
    if not manifest_path.is_file():
        raise Sm86PortDoneBarError(f"manifest.json missing: {manifest_path}")
    solution = _read_json(solution_path)
    manifest = _read_json(manifest_path)
    return evaluate_done_bar(
        solution=solution,
        manifest=manifest,
        workloads_passed=row.workloads_passed,
        workloads_total=row.workloads_total,
        ledger_evidence_label=row.evidence_label,
        ledger_surface_status=row.surface_status,
        solution_source_hash=_solution_file_sha256(solution_path),
    )


def assert_coverage_row_consistent(
    row: CoverageRow, *, repo_root: Path = _REPO_ROOT
) -> None:
    """Raise :class:`Sm86PortDoneBarError` when a coverage row disagrees
    with its on-disk bundle: the bundle manifest must exist, carry
    ``host.kind == local_sm_8_6``, match the row's ``evidence_label`` and
    ``surface_status`` map, and the DONE-bar verdict recomputed from disk
    must equal the row's stored ``done`` / ``blocking_reasons``. Both the
    consistency test and its planted negatives route through this helper so
    the QA assertion path is exercised.

    A ``deferred_correctness`` row (correctness genuinely failed on sm_86) is
    validated separately: it must carry a non-empty ``deferred_reason`` and
    must not also be ``done``; it has no DONE bundle to cross-check."""
    repo_root = Path(repo_root)
    if row.deferred_correctness:
        if row.done:
            raise Sm86PortDoneBarError(
                f"coverage_row_deferred_and_done: {row.slug} cannot be both "
                "deferred_correctness and done"
            )
        if not row.deferred_reason.strip():
            raise Sm86PortDoneBarError(
                f"coverage_row_deferred_reason_empty: {row.slug} is "
                "deferred_correctness but carries no concrete reason"
            )
        return
    if row.deferred_reason:
        raise Sm86PortDoneBarError(
            f"coverage_row_deferred_reason_without_flag: {row.slug} carries a "
            "deferred_reason but deferred_correctness is False"
        )
    manifest_path = repo_root / row.bundle_path / "manifest.json"
    if not manifest_path.is_file():
        raise Sm86PortDoneBarError(
            f"coverage_row_bundle_missing: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    host = manifest.get("host", {}) if isinstance(manifest.get("host"), Mapping) else {}
    if host.get("kind") != REQUIRED_HOST_KIND:
        raise Sm86PortDoneBarError(
            f"coverage_row_host_kind_mismatch: {row.slug}: manifest host.kind "
            f"{host.get('kind')!r} != {REQUIRED_HOST_KIND!r}"
        )
    if manifest.get("evidence_label") != row.evidence_label:
        raise Sm86PortDoneBarError(
            f"coverage_row_evidence_label_mismatch: {row.slug}: row "
            f"{row.evidence_label!r} != manifest {manifest.get('evidence_label')!r}"
        )
    surfaces = manifest.get("surfaces", {})
    manifest_map = {
        name: str(entry.get("status"))
        for name, entry in surfaces.items()
        if isinstance(entry, Mapping)
    }
    if dict(row.surface_status) != manifest_map:
        raise Sm86PortDoneBarError(
            f"coverage_row_surface_status_mismatch: {row.slug}"
        )
    verdict = evaluate_row_from_disk(row, repo_root=repo_root)
    if verdict.is_done != row.done or list(verdict.blocking_reasons) != list(
        row.blocking_reasons
    ):
        raise Sm86PortDoneBarError(
            f"coverage_row_verdict_drift: {row.slug}: stored "
            f"done={row.done} reasons={list(row.blocking_reasons)} != recomputed "
            f"done={verdict.is_done} reasons={list(verdict.blocking_reasons)}"
        )


# --- Coverage-ledger generation --------------------------------------
#
# The ledger is generated from two correctness authorities, both of which
# record per-cell workload pass counts; each cell's ``done`` /
# ``blocking_reasons`` verdict is recomputed from disk (manifest +
# solution.json), never trusted from the authority:
#
# * the FROZEN 5-cell pilot ledger (``sm86_pilot_correctness_v1.toml``,
#   the design contract seed) — read by :func:`build_seed_coverage_rows`;
# * the GROWING bulk-port ledger (``sm86_bulk_correctness_v1.toml``,
#   the design contract/M2) — read by :func:`build_bulk_coverage_rows`.
#
# :func:`build_all_coverage_rows` folds the two into the single committed
# ``sm86_coverage_ledger_v1.toml`` (sorted, with a duplicate-(slug,lang)
# guard so a cell can never be double-counted across the two authorities).

PILOT_CORRECTNESS_PATH = _THIS_DIR / "cache_prediction" / "sm86_pilot_correctness_v1.toml"
BULK_CORRECTNESS_PATH = _THIS_DIR / "cache_prediction" / "sm86_bulk_correctness_v1.toml"


def _done_coverage_row_from_disk(
    *,
    slug: str,
    language: str,
    category: str,
    baseline_dir: str,
    bundle_path: str,
    workloads_passed: int,
    workloads_total: int,
    repo_root: Path,
) -> CoverageRow:
    """Build a (non-deferred) coverage row by reading the bundle manifest and
    recomputing the DONE-bar verdict from disk. Shared by the seed and bulk
    generators so both axes use identical surface-status + provenance logic."""
    manifest = _read_json(repo_root / bundle_path / "manifest.json")
    surfaces = manifest.get("surfaces", {})
    surface_status = {
        name: str(entry.get("status"))
        for name, entry in sorted(surfaces.items())
        if isinstance(entry, Mapping)
    }
    evidence_label = str(manifest.get("evidence_label"))
    provisional = CoverageRow(
        slug=slug,
        language=language,
        category=category,
        baseline_dir=baseline_dir,
        bundle_path=bundle_path,
        workloads_passed=workloads_passed,
        workloads_total=workloads_total,
        evidence_label=evidence_label,
        surface_status=surface_status,
    )
    verdict = evaluate_row_from_disk(provisional, repo_root=repo_root)
    return CoverageRow(
        slug=slug,
        language=language,
        category=category,
        baseline_dir=baseline_dir,
        bundle_path=bundle_path,
        workloads_passed=workloads_passed,
        workloads_total=workloads_total,
        evidence_label=evidence_label,
        surface_status=surface_status,
        done=verdict.is_done,
        blocking_reasons=verdict.blocking_reasons,
    )


def build_seed_coverage_rows(repo_root: Path = _REPO_ROOT) -> list[CoverageRow]:
    """Build coverage rows for the existing pilot bundles by reading the
    (frozen) pilot correctness ledger + each bundle's manifest, and
    computing the DONE-bar verdict from disk."""
    repo_root = Path(repo_root)
    with PILOT_CORRECTNESS_PATH.open("rb") as handle:
        pilot = tomllib.load(handle)
    rows: list[CoverageRow] = []
    for cell in pilot.get("cells", []):
        pilot_bundle = str(cell["bundle_dir"])
        # Prefer a fresh the design contract coverage bundle (``…__sm86_cov``) emitted from
        # the gencode-conformed solution: its surfaces' solution_source_hash
        # matches the current solution.json (provenance-valid), while the
        # frozen pilot bundle at ``…__sm86`` is left byte-identical for the
        # the design contract pilot-SHA tests. Fall back to the pilot bundle when no coverage
        # bundle exists (then the provenance check leaves the row not-DONE).
        cov_bundle = pilot_bundle + "_cov"
        if (repo_root / cov_bundle / "manifest.json").is_file():
            bundle_path = cov_bundle
        else:
            bundle_path = pilot_bundle
        rows.append(
            _done_coverage_row_from_disk(
                slug=str(cell["slug"]),
                language=str(cell["language"]),
                category=str(cell["category"]),
                baseline_dir=str(cell["baseline_dir"]),
                bundle_path=bundle_path,
                workloads_passed=int(cell["passed_workloads"]),
                workloads_total=int(cell["expected_workloads"]),
                repo_root=repo_root,
            )
        )
    rows.sort(key=lambda r: (r.category, r.slug))
    return rows


def build_bulk_coverage_rows(repo_root: Path = _REPO_ROOT) -> list[CoverageRow]:
    """Build coverage rows for the design contract bulk-port cells from the growing
    ``sm86_bulk_correctness_v1.toml`` authority. A cell entry is either:

    * a DONE candidate — carries ``bundle_dir`` + ``passed_workloads`` /
      ``expected_workloads``; the row's verdict is recomputed from the
      bundle on disk (so a stale/under-strict bundle leaves it not-DONE);
    * a ``deferred_correctness`` cell — carries a non-empty
      ``deferred_reason`` and no DONE bundle.

    Returns ``[]`` when the bulk authority file does not yet exist."""
    repo_root = Path(repo_root)
    if not BULK_CORRECTNESS_PATH.is_file():
        return []
    with BULK_CORRECTNESS_PATH.open("rb") as handle:
        doc = tomllib.load(handle)
    rows: list[CoverageRow] = []
    for cell in doc.get("cells", []):
        slug = str(cell["slug"])
        language = str(cell["language"])
        category = str(cell["category"])
        baseline_dir = str(cell["baseline_dir"])
        if cell.get("deferred_correctness"):
            rows.append(
                CoverageRow(
                    slug=slug,
                    language=language,
                    category=category,
                    baseline_dir=baseline_dir,
                    bundle_path="",
                    workloads_passed=int(cell.get("passed_workloads", 0)),
                    workloads_total=int(cell.get("expected_workloads", 0)),
                    evidence_label="",
                    surface_status={},
                    done=False,
                    deferred_correctness=True,
                    deferred_reason=str(cell.get("deferred_reason", "")),
                )
            )
            continue
        rows.append(
            _done_coverage_row_from_disk(
                slug=slug,
                language=language,
                category=category,
                baseline_dir=baseline_dir,
                bundle_path=str(cell["bundle_dir"]),
                workloads_passed=int(cell["passed_workloads"]),
                workloads_total=int(cell["expected_workloads"]),
                repo_root=repo_root,
            )
        )
    return rows


def build_all_coverage_rows(repo_root: Path = _REPO_ROOT) -> list[CoverageRow]:
    """Fold the frozen 5-cell seed + the growing bulk-port rows into the
    single committed coverage ledger. Raises on a duplicate
    (category, slug, language) so a cell can never be double-counted across
    the two correctness authorities. Sorted by (category, slug)."""
    seed = build_seed_coverage_rows(repo_root)
    bulk = build_bulk_coverage_rows(repo_root)
    seen: set[tuple[str, str, str]] = set()
    combined: list[CoverageRow] = []
    for r in list(seed) + list(bulk):
        key = (r.category, r.slug, r.language)
        if key in seen:
            raise Sm86PortDoneBarError(
                f"coverage_row_duplicate: {key} appears in both the seed and "
                "bulk correctness authorities"
            )
        seen.add(key)
        combined.append(r)
    combined.sort(key=lambda r: (r.category, r.slug))
    return combined


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_str_array(values: Sequence[str]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _toml_inline_table(mapping: Mapping[str, str]) -> str:
    parts = [f"{k} = {_toml_str(str(v))}" for k, v in sorted(mapping.items())]
    return "{" + ", ".join(parts) + "}"


def render_coverage_ledger_toml(rows: Sequence[CoverageRow]) -> str:
    done_count = sum(1 for r in rows if r.done)
    lines: list[str] = []
    lines.append("# sm_86 port coverage ledger (seed). GENERATED by")
    lines.append("# scripts/baselines_harness/calibration/sm86_port_done_bar.py")
    lines.append("# from the frozen pilot correctness ledger + on-disk manifests.")
    lines.append("# Each row's `done` / `blocking_reasons` is recomputed from disk")
    lines.append("# by the DONE-bar validator; the bulk-port milestone expands it.")
    lines.append("")
    lines.append(f"schema_version = {COVERAGE_LEDGER_SCHEMA_VERSION}")
    lines.append("")
    lines.append("[meta]")
    lines.append('host_arch = "sm_86"')
    lines.append('host_kind = "local_sm_8_6"')
    lines.append(
        'seed_source = "calibration/cache_prediction/sm86_pilot_correctness_v1.toml"'
    )
    lines.append(f"row_count = {len(rows)}")
    lines.append(f"done_count = {done_count}")
    lines.append(
        f"deferred_correctness_count = {sum(1 for r in rows if r.deferred_correctness)}"
    )
    lines.append(f"not_done_count = {len(rows) - done_count}")
    lines.append("")
    for r in rows:
        lines.append("[[rows]]")
        lines.append(f"slug = {_toml_str(r.slug)}")
        lines.append(f"language = {_toml_str(r.language)}")
        lines.append(f"category = {_toml_str(r.category)}")
        lines.append(f"baseline_dir = {_toml_str(r.baseline_dir)}")
        lines.append(f"bundle_path = {_toml_str(r.bundle_path)}")
        lines.append(f"workloads_passed = {r.workloads_passed}")
        lines.append(f"workloads_total = {r.workloads_total}")
        lines.append(f"evidence_label = {_toml_str(r.evidence_label)}")
        lines.append(f"done = {'true' if r.done else 'false'}")
        lines.append(
            f"deferred_correctness = {'true' if r.deferred_correctness else 'false'}"
        )
        lines.append(f"deferred_reason = {_toml_str(r.deferred_reason)}")
        lines.append(f"blocking_reasons = {_toml_str_array(r.blocking_reasons)}")
        lines.append(f"surface_status = {_toml_inline_table(r.surface_status)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _main() -> int:  # pragma: no cover - manual regeneration entry point
    import sys

    rows = build_all_coverage_rows()
    DEFAULT_COVERAGE_LEDGER_PATH.write_text(
        render_coverage_ledger_toml(rows), encoding="utf-8"
    )
    done = sum(1 for r in rows if r.done)
    sys.stdout.write(
        f"wrote {DEFAULT_COVERAGE_LEDGER_PATH} ({len(rows)} rows, {done} done)\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

