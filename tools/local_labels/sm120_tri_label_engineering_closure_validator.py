"""Validator for the sm_120 tri-label duration-leg engineering-closure artifact.

Fail-closed and EVIDENCE-RECOMPUTING: it does not trust the artifact's
self-reported summaries. It loads the referenced evidence artifacts and
recomputes the closure facts, failing on any drift:

1. Closed-schema validation (recursively closed) against
   ``workloads/simbench_bw/schemas/sm120_tri_label_engineering_closure.schema.json``.
2. Every referenced evidence-artifact path exists under the repo root.
3. ``reconciling_slugs`` is recomputed from ``sm120_duration_cohort.json``.
4. ``reconciling_slugs_swept``, per-slug Goldilocks verdicts, measured
   within-budget / measured over-budget / timed-out points, and the
   trace-admissible ceiling are recomputed from
   ``duration_trace_goldilocks_sweep.json`` (including the byte sizes); a
   timeout is never accepted as a measured byte fact, and a measured fact is
   never accepted as a timeout.
5. ``any_goldilocks`` must be false and every reconciling slug must be swept
   with no Goldilocks point.
6. The production reconcile-floor cells are recomputed against
   ``duration_admitted.json``.
7. The live tri-label four-source coverage is recomputed from the roster.
8. The stamped ``source_commit`` must contain the producer + schema paths (a
   reproducible-provenance check).

This is cohort-integrity gating for an honest engineering-closure record, not a
model of GPU behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import sm120_tri_label_roster_validator as rv  # noqa: E402

_SCHEMA_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "sm120_tri_label_engineering_closure.schema.json"
)

# The production reconcile floor lists exactly this many smallest admitted cells.
FLOOR_COUNT = 3

# The closure may ONLY cite these canonical committed repo-relative evidence
# artifacts (no substitution, no absolute/`..`/symlink escape). Provenance-pins
# the evidence surface so the closure cannot validate against arbitrary local
# files.
EXPECTED_EVIDENCE_PATHS = {
    "goldilocks_sweep": "data/local_labels/duration_trace_goldilocks_sweep.json",
    "duration_cohort": "data/local_labels/sm120_duration_cohort.json",
    "duration_admitted": "data/local_labels/duration_admitted.json",
    "tri_label_roster": "data/local_labels/sm120_tri_label_roster.json",
}


def _is_unsafe_rel_path(rel: str, repo_root: Path) -> bool:
    """Reject absolute paths, parent-traversal, and any path that — after
    resolving symlinks under ``repo_root`` — escapes the repo root. (Mirrors
    tools/local_labels/sm120_tri_label_buildability_preflight.py._is_unsafe_rel_path.)"""
    if not rel:
        return True
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return True
    try:
        resolved = (repo_root / candidate).resolve(strict=False)
        resolved.relative_to(repo_root.resolve())
    except (ValueError, OSError):
        return True
    return False

# Producer + schema paths the stamped source_commit must contain.
_PROVENANCE_PATHS = (
    "tools/local_labels/build_sm120_tri_label_engineering_closure.py",
    "tools/local_labels/sm120_tri_label_engineering_closure_validator.py",
    "workloads/simbench_bw/schemas/sm120_tri_label_engineering_closure.schema.json",
)


class EngineeringClosureValidationError(ValueError):
    """Raised when the engineering-closure artifact fails a gate."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_keys(points) -> set:
    return {(p.get("slug"), p.get("size")) for p in points or []}


def _full_record(slug, size, tr, dur) -> dict:
    return {"slug": slug, "size": size, "trace_bytes": tr.get("trace_bytes"),
            "duration_ms": dur.get("duration_ms"), "duration_reconciled": dur.get("reconciled")}


def _doc_record(p: dict) -> dict:
    return {"slug": p.get("slug"), "size": p.get("size"), "trace_bytes": p.get("trace_bytes"),
            "duration_ms": p.get("duration_ms"), "duration_reconciled": p.get("duration_reconciled")}


def _recompute_sweep(sweep: dict) -> dict:
    swept: set[str] = set()
    within: dict[tuple, dict] = {}
    over: dict[tuple, dict] = {}
    timeouts: set[tuple] = set()
    skipped: dict[tuple, str] = {}
    per_slug_goldilocks: dict[str, bool] = {}
    for p in sweep.get("points", []):
        if "size" not in p:
            continue
        slug = p["slug"]
        swept.add(slug)
        per_slug_goldilocks[slug] = per_slug_goldilocks.get(slug, False) or bool(p.get("goldilocks"))
        tr, dur = p.get("trace", {}), p.get("duration", {})
        st, bbs = tr.get("trace_status"), tr.get("byte_budget_status")
        key = (slug, p["size"])
        rec = _full_record(slug, p["size"], tr, dur)
        if st == "measured" and bbs == "within_budget":
            within[key] = rec
        elif st == "measured" and bbs == "over_budget":
            over[key] = rec
        elif st == "timeout":
            timeouts.add(key)
        elif st in ("skipped_after_timeout", "skipped_after_measured_over_budget"):
            skipped[key] = st
    return {"swept": swept, "within": within, "over": over, "timeouts": timeouts,
            "skipped": skipped, "per_slug_goldilocks": per_slug_goldilocks,
            "any_goldilocks": any(per_slug_goldilocks.values()),
            "budget_mib": sweep.get("trace_budget_mib")}


def _sweep_classification_problems(sweep: dict, budget_bytes: float) -> list[str]:
    """Prove each sweep point falls in EXACTLY one internally-consistent trace
    class from its OWN bytes vs the budget — do not trust the byte_budget_status
    label. Returns a list of inconsistency problems."""
    probs: list[str] = []
    for p in sweep.get("points", []):
        if "size" not in p:
            continue
        tr = p.get("trace", {})
        st, bbs = tr.get("trace_status"), tr.get("byte_budget_status")
        tb, adm = tr.get("trace_bytes"), tr.get("trace_admissible")
        key = (p.get("slug"), p.get("size"))
        if st == "measured" and bbs == "within_budget":
            if not (isinstance(tb, int) and 1 <= tb <= budget_bytes and adm is True):
                probs.append(f"{key} within_budget inconsistent: bytes={tb} adm={adm} budget={int(budget_bytes)}")
        elif st == "measured" and bbs == "over_budget":
            if not (isinstance(tb, int) and tb > budget_bytes and adm is False):
                probs.append(f"{key} over_budget inconsistent: bytes={tb} adm={adm} budget={int(budget_bytes)}")
        elif st == "timeout":
            if not (bbs == "unmeasured_timeout" and tb is None and adm is None):
                probs.append(f"{key} timeout inconsistent: bbs={bbs} bytes={tb} adm={adm}")
        elif st in ("skipped_after_timeout", "skipped_after_measured_over_budget"):
            if not (bbs == "not_run" and tb is None and adm is None):
                probs.append(f"{key} skipped inconsistent: bbs={bbs} bytes={tb} adm={adm}")
        else:
            probs.append(f"{key} unknown/mixed trace class: trace_status={st!r} byte_budget_status={bbs!r}")
    return probs


def _commit_contains(repo_root: Path, commit: str, rel: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}:{rel}"],
            capture_output=True, text=True,
        )
        return proc.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _is_git_repo(repo_root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"
    except (FileNotFoundError, OSError):
        return False


def validate_engineering_closure(
    doc: dict, repo_root: Path = _REPO_ROOT, *, schema_path: Path | None = None,
    check_provenance: bool = True,
) -> list[str]:
    """Return a list of violations; empty means valid."""
    problems: list[str] = []

    schema = _load(schema_path or _SCHEMA_PATH)
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]

    paths = doc.get("evidence_artifact_paths") or {}
    # (0) Evidence paths must be the EXACT canonical committed repo-relative
    # artifacts (no substitution) AND path-safe (no absolute / `..` / symlink
    # escape) -- enforced independently of the schema const, before any load, so
    # the closure cannot validate against arbitrary/non-canonical local files.
    for key, expected in EXPECTED_EVIDENCE_PATHS.items():
        rel = paths.get(key)
        if rel != expected:
            problems.append(
                f"evidence_artifact_paths.{key} {rel!r} is not the canonical {expected!r}")
        elif _is_unsafe_rel_path(str(rel), repo_root):
            problems.append(f"evidence_artifact_paths.{key} {rel!r} is unsafe (absolute/`..`/symlink escape)")
    if set(paths) != set(EXPECTED_EVIDENCE_PATHS):
        problems.append(f"evidence_artifact_paths keys {sorted(paths)} != {sorted(EXPECTED_EVIDENCE_PATHS)}")
    if problems:
        return problems  # do not resolve/load non-canonical paths

    # (1) Evidence paths must exist (canonical, resolved under repo_root).
    resolved: dict[str, Path] = {}
    for key, rel in paths.items():
        p = repo_root / rel
        resolved[key] = p
        if not p.is_file():
            problems.append(f"evidence_artifact_paths.{key} -> {rel} does not exist")
    if problems:
        return problems  # cannot recompute without the evidence files

    cohort = _load(resolved["duration_cohort"])
    sweep = _load(resolved["goldilocks_sweep"])
    admitted = _load(resolved["duration_admitted"])
    roster = _load(resolved["tri_label_roster"])

    # (2) reconciling_slugs recomputed from the cohort.
    cohort_slugs = sorted({str(c["slug"]) for c in cohort.get("cells", [])})
    if sorted(doc.get("reconciling_slugs") or []) != cohort_slugs:
        problems.append(
            f"reconciling_slugs {sorted(doc.get('reconciling_slugs') or [])} != cohort slugs {cohort_slugs}")

    # (3) sweep recompute.
    sw = _recompute_sweep(sweep)
    if sorted(doc.get("reconciling_slugs_swept") or []) != sorted(sw["swept"]):
        problems.append(
            f"reconciling_slugs_swept {sorted(doc.get('reconciling_slugs_swept') or [])} != "
            f"sweep-recomputed swept slugs {sorted(sw['swept'])}")
    for slug in doc.get("reconciling_slugs") or []:
        if slug not in sw["swept"]:
            problems.append(f"reconciling slug {slug!r} is unaccounted for (not swept)")
        if sw["per_slug_goldilocks"].get(slug):
            problems.append(f"reconciling slug {slug!r} has a Goldilocks sweep point; closure is invalid")
    if sw["any_goldilocks"]:
        problems.append("sweep recomputes any_goldilocks=true; the engineering-closure does not hold")

    # (3b) trace budget must match the sweep's budget (the closure depends on it).
    if doc.get("trace_budget_mib") != sw["budget_mib"]:
        problems.append(
            f"trace_budget_mib {doc.get('trace_budget_mib')!r} != sweep budget {sw['budget_mib']!r}")
    budget_bytes = float(sw["budget_mib"] or 0) * 1024 * 1024
    # Prove each sweep point's trace class from its OWN bytes vs the budget --
    # do not trust the byte_budget_status label.
    problems.extend(_sweep_classification_problems(sweep, budget_bytes))

    # (3c) FULL measured-point records (slug, size, trace_bytes, duration_ms,
    # reconciled) must match the sweep -- not just the byte map.
    measured = doc.get("measured") or {}
    doc_within = {(p["slug"], p["size"]): _doc_record(p) for p in measured.get("trace_within_budget_points") or []}
    doc_over = {(p["slug"], p["size"]): _doc_record(p) for p in measured.get("trace_over_budget_points") or []}
    doc_timeouts = _point_keys(doc.get("timed_out_traces"))
    doc_skipped = {(p["slug"], p["size"]): p.get("trace_status") for p in doc.get("skipped_traces") or []}
    if set(doc_within) != set(sw["within"]) or any(doc_within[k] != sw["within"][k] for k in doc_within):
        problems.append("measured.trace_within_budget_points drifts from the sweep (slug/size/bytes/duration_ms/reconciled)")
    if set(doc_over) != set(sw["over"]) or any(doc_over[k] != sw["over"][k] for k in doc_over):
        problems.append("measured.trace_over_budget_points drifts from the sweep (slug/size/bytes/duration_ms/reconciled)")
    if doc_timeouts != sw["timeouts"]:
        problems.append("timed_out_traces drifts from the sweep's timed-out points")
    if doc_skipped != sw["skipped"]:
        problems.append("skipped_traces drifts from the sweep's skipped points (status/coverage)")
    # A point must appear in exactly one status class.
    classes = [set(doc_within), set(doc_over), doc_timeouts, set(doc_skipped)]
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            overlap = classes[i] & classes[j]
            if overlap:
                problems.append(f"points {sorted(overlap)} appear in more than one trace-status class")

    # (4) ceiling = largest within-budget measured trace.
    ceiling = measured.get("trace_admissible_ceiling") or {}
    if sw["within"]:
        max_key = max(sw["within"], key=lambda k: sw["within"][k]["trace_bytes"])
        if _doc_record(ceiling) != sw["within"][max_key]:
            problems.append("measured.trace_admissible_ceiling is not the largest within-budget measured trace")
    cb = ceiling.get("trace_bytes")
    if not isinstance(cb, int) or cb <= 0 or cb > budget_bytes:
        problems.append(f"trace_admissible_ceiling.trace_bytes {cb!r} is not a positive within-budget measurement")

    # (5) production reconcile-floor = the EXACT recomputed-smallest admitted
    # cells (order + medians), not merely cells that happen to be admitted.
    floor = doc.get("production_duration_reconcile_floor") or {}
    if not any("duration_admitted" in str(e) for e in floor.get("evidence") or []):
        problems.append("production_duration_reconcile_floor.evidence must reference duration_admitted.json")
    admitted_sorted = sorted(
        admitted.get("cells", []),
        key=lambda c: ((c.get("median_duration_ns") if c.get("median_duration_ns") is not None else float("inf")),
                       str(c.get("slug")), str(c.get("shape_point"))),
    )
    if not admitted_sorted:
        problems.append("duration_admitted.json has no admitted cells; production reconcile floor unsupported")
    doc_floor = floor.get("smallest_reconciling_cells") or []
    if len(doc_floor) != FLOOR_COUNT:
        problems.append(
            f"production_duration_reconcile_floor.smallest_reconciling_cells has {len(doc_floor)} cells; "
            f"must be exactly {FLOOR_COUNT}")
    # Compare to the recomputed smallest FLOOR_COUNT (fixed cardinality so a
    # truncated prefix cannot pass).
    expected_floor = [{"slug": str(c["slug"]), "shape_point": str(c["shape_point"]),
                       "median_duration_ns": c.get("median_duration_ns")} for c in admitted_sorted[:FLOOR_COUNT]]
    norm_doc_floor = [{"slug": str(c.get("slug")), "shape_point": str(c.get("shape_point")),
                       "median_duration_ns": c.get("median_duration_ns")} for c in doc_floor]
    if norm_doc_floor != expected_floor:
        problems.append("production_duration_reconcile_floor.smallest_reconciling_cells != the recomputed smallest-3 admitted cells (order/medians)")

    # (6) must NOT claim the four-source roster is satisfied; FULL coverage recompute.
    if doc.get("claims_four_source_roster_satisfied") is not False:
        problems.append("claims_four_source_roster_satisfied must be false")
    coverage = doc.get("current_tri_label_coverage") or {}
    presence = rv.presence_from_artifacts(roster, repo_root)
    live_counts = {src: 0 for src in rv.TRI_LABEL_SOURCES}
    for sources in presence.values():
        for s in sources:
            if s in live_counts:
                live_counts[s] += 1
    live_gaps = len(rv.tri_label_coverage(roster, presence))
    expected_coverage = {**live_counts, "four_source_gaps": live_gaps,
                         "total_cells": len(roster.get("cells", []))}
    if coverage != expected_coverage:
        problems.append(
            f"current_tri_label_coverage {coverage} disagrees with the live recompute {expected_coverage}")
    if live_gaps <= 0:
        problems.append("the roster has no four-source gaps; an engineering-closure is unwarranted")

    # (7) reproducible-provenance: source_commit contains the producer + schema.
    if check_provenance:
        commit = doc.get("source_commit", "")
        if _is_git_repo(repo_root):
            if not _commit_contains(repo_root, commit, _PROVENANCE_PATHS[0]):
                problems.append(
                    f"source_commit {commit} does not contain the producer "
                    f"{_PROVENANCE_PATHS[0]} (provenance is not reproducible)")
            else:
                for rel in _PROVENANCE_PATHS[1:]:
                    if not _commit_contains(repo_root, commit, rel):
                        problems.append(f"source_commit {commit} does not contain {rel}")
    return problems


def validate_engineering_closure_or_raise(doc: dict, repo_root: Path = _REPO_ROOT, **kw) -> None:
    problems = validate_engineering_closure(doc, repo_root, **kw)
    if problems:
        raise EngineeringClosureValidationError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_tri_label_engineering_closure.json",
    )
    parser.add_argument("--no-provenance", action="store_true",
                        help="skip the source_commit-contains-producer check (pre-commit dev runs)")
    args = parser.parse_args(argv)
    doc = _load(args.artifact)
    problems = validate_engineering_closure(doc, check_provenance=not args.no_provenance)
    if problems:
        print(f"INVALID ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"VALID: {args.artifact} (verdict={doc['verdict']}, closed_leg={doc['closed_leg']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
