"""Fail-closed, RECOMPUTING validator for the sm_120 L1/L2 frozen-grain closure.

It re-derives the closure from the committed evidence via the producer's
``build()`` and fails on any drift (ignoring only the HEAD-dependent
``source_commit``), plus explicit closure checks:

1. closed-schema validation;
2. the evidence-artifact paths are exactly the canonical committed paths,
   path-safe (no absolute / ``..`` / symlink escape), and exist;
3. the 2-way cell disposition (admitted_l1l2 vs repeatability_excluded) is
   recomputed from the committed roster + canonical labels + both pilots: every
   roster cell falls in exactly one class, no overlap, and the union covers all
   roster cells (every cell was collected);
4. the disposition counts are recomputed and the admitted count is below the
   required minimum (so the engineering-closure verdict holds);
5. verdict / closed_leg / claims_four_source_roster_satisfied are the closed
   consts; every repeatability_excluded cell carries the typed reason + a
   recomputed stddev that is a number strictly ABOVE the committed threshold
   (it actually failed the gate), cross-checked against its repeatability
   admission row in BOTH pilots (admitted=false, typed_reason,
   n_repeats >= min_repeats, matching stddev);
6. the cited duration-leg closure exists and is itself a valid duration-leg
   ``engineering_closure_refuted`` (the four-source requirement stays unsatisfied);
7. a full rebuild-and-compare (every field recomputed from committed inputs);
8. reproducible-provenance: ``source_commit`` contains the producer + schema +
   validator.
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

import build_sm120_l1l2_frozen_grain_closure as producer  # noqa: E402

_SCHEMA_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "sm120_l1l2_frozen_grain_engineering_closure.schema.json"
)
_PROVENANCE_PATHS = (
    "tools/local_labels/build_sm120_l1l2_frozen_grain_closure.py",
    "tools/local_labels/sm120_l1l2_frozen_grain_closure_validator.py",
    "workloads/simbench_bw/schemas/sm120_l1l2_frozen_grain_engineering_closure.schema.json",
)
_REBUILD_EXCLUDE = ("source_commit",)


class FrozenGrainClosureValidationError(ValueError):
    pass


def _duration_leg_validator():
    """The duration-leg closure's own fail-closed validator (lazy import: it
    pulls in its producer chain)."""
    import sm120_tri_label_engineering_closure_validator as dur_v  # noqa: PLC0415

    return dur_v


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _deep_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _is_unsafe_rel_path(rel: str, repo_root: Path) -> bool:
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        return True
    try:
        resolved = (repo_root / rel).resolve(strict=False)
        resolved.relative_to(repo_root.resolve())
    except (ValueError, OSError):
        return True
    return False


def _commit_contains(repo_root: Path, commit: str, rel: str) -> bool:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}:{rel}"],
            capture_output=True, text=True,
        ).returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _is_git_repo(repo_root: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (FileNotFoundError, OSError):
        return False


def validate_frozen_grain_closure(
    doc: dict,
    repo_root: Path = _REPO_ROOT,
    *,
    schema_path: Path | None = None,
    check_provenance: bool = True,
    check_rebuild: bool = True,
) -> list[str]:
    problems: list[str] = []
    schema = json.loads((schema_path or _SCHEMA_PATH).read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]

    # (2) canonical, path-safe, existing evidence paths.
    paths = doc.get("evidence_artifact_paths") or {}
    if paths != producer.EVIDENCE_PATHS:
        problems.append(
            f"evidence_artifact_paths {paths} != canonical {producer.EVIDENCE_PATHS}"
        )
    for key, rel in paths.items():
        if isinstance(rel, str) and _is_unsafe_rel_path(rel, repo_root):
            problems.append(f"evidence_artifact_paths.{key} {rel!r} is unsafe")
        elif isinstance(rel, str) and not (repo_root / rel).is_file():
            problems.append(f"evidence_artifact_paths.{key} -> {rel} does not exist")

    # (3)(4) recompute the partition from committed evidence.
    try:
        part = producer._partition(repo_root)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"partition recompute failed: {exc}")
        part = None

    disp = doc.get("cell_disposition", {})
    if part is not None:
        doc_adm = {(c["slug"], c["shape_point"]) for c in disp.get("admitted_l1l2", [])}
        doc_exc = {(c["slug"], c["shape_point"]) for c in disp.get("repeatability_excluded", [])}
        if doc_adm != part["admitted"]:
            problems.append("admitted_l1l2 set drifts from the recomputed partition")
        if doc_exc != part["excluded"]:
            problems.append("repeatability_excluded set drifts from the recomputed partition")
        # every roster cell in exactly one class (no overlap; the two classes
        # cover the full roster -- every cell was collected).
        if doc_adm & doc_exc:
            problems.append("cell_disposition classes overlap")
        if len(doc_adm) + len(doc_exc) != part["roster_total"]:
            problems.append("cell_disposition does not cover every roster cell exactly once")

    counts = doc.get("disposition_counts", {})
    if counts.get("admitted_l1l2") != len(disp.get("admitted_l1l2", [])):
        problems.append("disposition_counts.admitted_l1l2 != list length")
    if counts.get("repeatability_excluded") != len(disp.get("repeatability_excluded", [])):
        problems.append("disposition_counts.repeatability_excluded != list length")
    if doc.get("frozen_roster_l1l2_admitted_count") != counts.get("admitted_l1l2"):
        problems.append("frozen_roster_l1l2_admitted_count != disposition_counts.admitted_l1l2")
    if isinstance(counts.get("admitted_l1l2"), int) and counts["admitted_l1l2"] >= doc.get("required_min_cells", 50):
        problems.append(
            "admitted_l1l2 count meets the required minimum; the engineering-closure "
            "verdict does not hold (cohort is satisfiable)"
        )

    # (5) typed reasons + recomputed stddev for the excluded cells, PLUS a
    # fail-closed proof that every excluded cell actually FAILED the repeatability
    # gate. A matching-stddev check alone is not enough: a below-threshold cell
    # (closure + evidence agreeing) would still PASS the gate yet be labelled
    # "excluded", silently propping up the engineering-closure verdict. So each
    # excluded cell's L2 stddev must be a number strictly ABOVE the committed
    # threshold, and its repeatability admission row (cited in BOTH pilots) must
    # record the failure (admitted=false, typed_reason, n_repeats >= min, matching
    # stddev).
    rep_cfg = doc.get("repeatability_config", {})
    threshold = rep_cfg.get("stddev_threshold")
    min_repeats = rep_cfg.get("min_repeats")
    if not _is_number(threshold) or threshold <= 0:
        problems.append(
            f"repeatability_config.stddev_threshold {threshold!r} is not a positive number"
        )
        threshold = None
    if not isinstance(min_repeats, int) or isinstance(min_repeats, bool) or min_repeats < 1:
        problems.append(
            f"repeatability_config.min_repeats {min_repeats!r} is not a positive integer"
        )
        min_repeats = None

    stddev = producer._excluded_stddev(repo_root)
    rep_rows = producer._repeatability_rows(repo_root)
    for c in disp.get("repeatability_excluded", []):
        key = (c.get("slug"), c.get("shape_point"))
        label = f"{c.get('slug')}/{c.get('shape_point')}"
        if c.get("typed_reason") != producer.REASON_REPEATABILITY:
            problems.append(f"{label}: wrong repeatability typed_reason")
        sd = c.get("stddev_l2_sector")
        if not _is_number(sd):
            problems.append(f"{label}: stddev_l2_sector {sd!r} is not a number")
        elif threshold is not None and not (sd > threshold):
            problems.append(
                f"{label}: stddev_l2_sector {sd} does not exceed the repeatability "
                f"threshold {threshold} -- it would PASS the gate, so it cannot be "
                "repeatability_excluded"
            )
        if sd != stddev.get(key):
            problems.append(f"{label}: stddev_l2_sector drifts from the pilot label_excluded")

        # cross-check the repeatability admission row (the verifiable BASIS for
        # the exclusion), present in one of the two cited label_repeatability.json.
        row = rep_rows.get(key)
        if row is None:
            problems.append(f"{label}: no repeatability admission row in either pilot")
            continue
        adm = row.get("admission", {})
        if adm.get("admitted") is not False:
            problems.append(f"{label}: repeatability row admission.admitted is not false")
        if adm.get("typed_reason") != producer.REASON_REPEATABILITY:
            problems.append(
                f"{label}: repeatability row typed_reason != {producer.REASON_REPEATABILITY}"
            )
        nrep = row.get("n_repeats")
        if min_repeats is not None and (
            not isinstance(nrep, int) or isinstance(nrep, bool) or nrep < min_repeats
        ):
            problems.append(f"{label}: repeatability row n_repeats {nrep!r} < min_repeats {min_repeats}")
        row_sd = row.get("stddev_l2_sector")
        if row_sd != sd:
            problems.append(
                f"{label}: repeatability row stddev_l2_sector {row_sd!r} != closure {sd!r}"
            )
        elif threshold is not None and _is_number(row_sd) and not (row_sd > threshold):
            problems.append(
                f"{label}: repeatability row stddev_l2_sector {row_sd} does not exceed "
                f"threshold {threshold}"
            )

    # (6) the cited duration-leg closure is a valid duration-leg refutation --
    # RECURSIVELY validated by its own fail-closed validator (checking two
    # top-level fields is not enough: a corrupted duration closure that preserves
    # them would otherwise pass this multi-leg closure).
    dur = doc.get("evidence_artifact_paths", {}).get("duration_leg_closure")
    if dur and not _is_unsafe_rel_path(dur, repo_root) and (repo_root / dur).is_file():
        dur_doc = json.loads((repo_root / dur).read_text(encoding="utf-8"))
        if dur_doc.get("verdict") != "engineering_closure_refuted":
            problems.append("duration_leg_closure verdict is not engineering_closure_refuted")
        if dur_doc.get("claims_four_source_roster_satisfied") is not False:
            problems.append("duration_leg_closure does not record the four-source roster as unsatisfied")
        for p in _duration_leg_validator().validate_engineering_closure(
            dur_doc, repo_root, check_provenance=check_provenance
        ):
            problems.append(f"duration_leg_closure: {p}")

    # (7) full rebuild-and-compare.
    if check_rebuild:
        try:
            rebuilt = producer.build(repo_root)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rebuild failed: {exc}")
        else:
            a = {k: v for k, v in doc.items() if k not in _REBUILD_EXCLUDE}
            b = {k: v for k, v in rebuilt.items() if k not in _REBUILD_EXCLUDE}
            if not _deep_equal(a, b):
                diffs = sorted(k for k in set(a) | set(b) if not _deep_equal(a.get(k), b.get(k)))
                problems.append(f"closure drifts from a fresh recompute; differing keys: {diffs}")

    # (8) provenance.
    if check_provenance and _is_git_repo(repo_root):
        commit = doc.get("source_commit", "")
        for rel in _PROVENANCE_PATHS:
            if not _commit_contains(repo_root, commit, rel):
                problems.append(f"source_commit {commit} does not contain {rel}")
    return problems


def validate_or_raise(doc: dict, repo_root: Path = _REPO_ROOT, **kw) -> None:
    problems = validate_frozen_grain_closure(doc, repo_root, **kw)
    if problems:
        raise FrozenGrainClosureValidationError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_l1l2_frozen_grain_engineering_closure.json",
    )
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.artifact.read_text(encoding="utf-8"))
    problems = validate_frozen_grain_closure(doc, check_provenance=not args.no_provenance)
    if problems:
        print(f"INVALID ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    c = doc["disposition_counts"]
    print(
        f"VALID: {args.artifact} (verdict={doc['verdict']}, closed_leg={doc['closed_leg']}, "
        f"admitted={c['admitted_l1l2']}/{c['total']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
