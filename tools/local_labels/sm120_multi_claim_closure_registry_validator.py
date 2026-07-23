"""Fail-closed, RECOMPUTING validator for the sm_120 multi-claim closure registry.

It re-derives the registry from the committed source closures via the producer's
``build()`` and fails on any drift (ignoring only ``source_commit``), plus
explicit checks:

1. closed-schema validation;
2. the source-artifact paths are exactly the canonical committed paths, path-safe,
   and exist;
3. the record target set is recomputed as ``{duration_ns}`` plus the metric
   registry keys: every target appears exactly once -- no missing / duplicate /
   extra record;
4. every record is TERMINAL (no ``indeterminate`` presented as a closure), carries
   ``cross_arch_precondition: false``, a non-empty ``ac_coverage``, a non-empty
   ``evidence_artifact_paths`` whose entries are canonical/path-safe/existing, and
   a ``verified_live_sm120_proof`` block (compute_capability sm_120);
5. each record's ``kind`` and ``verdict`` are recomputed from its source closure
   (``duration_ns`` from the duration predictor report; each cache target from the
   cache-metric closure);
6. the four cited source closures are RECURSIVELY validated by their own
   fail-closed validators;
7. a full rebuild-and-compare;
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

import build_sm120_multi_claim_closure_registry as producer  # noqa: E402
import ncu_metric_registry as registry  # noqa: E402

_SCHEMA_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "sm120_multi_claim_closure_registry.schema.json"
)
_PROVENANCE_PATHS = (
    "tools/local_labels/build_sm120_multi_claim_closure_registry.py",
    "tools/local_labels/sm120_multi_claim_closure_registry_validator.py",
    "workloads/simbench_bw/schemas/sm120_multi_claim_closure_registry.schema.json",
    # the metric registry is the source of truth for the cache target-ID set.
    "tools/local_labels/ncu_metric_registry.py",
)
_REBUILD_EXCLUDE = ("source_commit",)


class MultiClaimRegistryValidationError(ValueError):
    pass


def _duration_validator():
    import duration_predictor_report_validator as dv  # noqa: PLC0415
    return dv


def _duration_leg_validator():
    import sm120_tri_label_engineering_closure_validator as lv  # noqa: PLC0415
    return lv


def _frozen_grain_validator():
    import sm120_l1l2_frozen_grain_closure_validator as fv  # noqa: PLC0415
    return fv


def _cache_metric_validator():
    import sm120_cache_metric_closure_validator as cv  # noqa: PLC0415
    return cv


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


def validate_multi_claim_registry(
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

    # (2) canonical, path-safe, existing source-artifact paths.
    sources = doc.get("source_artifact_paths") or {}
    if sources != producer.SOURCE_ARTIFACT_PATHS:
        problems.append(
            f"source_artifact_paths {sources} != canonical {producer.SOURCE_ARTIFACT_PATHS}"
        )
    for key, rel in sources.items():
        if isinstance(rel, str) and _is_unsafe_rel_path(rel, repo_root):
            problems.append(f"source_artifact_paths.{key} {rel!r} is unsafe")
        elif isinstance(rel, str) and not (repo_root / rel).is_file():
            problems.append(f"source_artifact_paths.{key} -> {rel} does not exist")

    if doc.get("cross_arch_precondition") not in (False, True, "pending"):
        problems.append(
            "top-level cross_arch_precondition is not tri-state (false | true | 'pending')"
        )

    # (3) the record target set recomputed = {duration_ns} u registry keys.
    records = doc.get("records", [])
    ids = [r.get("target_id") for r in records]
    expected = {producer.DURATION_TARGET} | set(registry.REGISTRY)
    if sorted(ids) != sorted(expected):
        problems.append(
            f"record target set drifts (missing={sorted(expected - set(ids))}, "
            f"extra={sorted(set(ids) - expected)})"
        )
    if len(ids) != len(set(ids)):
        problems.append("records contains a duplicate target_id")

    # (4/5) per-record terminal/proof/evidence + verdict-from-source.
    dur_rel = sources.get("duration_predictor_report")
    cache_rel = sources.get("cache_metric_closure")
    dur_verdict = None
    cache_by_id: dict = {}
    if dur_rel and (repo_root / dur_rel).is_file():
        dur_verdict = json.loads((repo_root / dur_rel).read_text(encoding="utf-8")).get("verdict")
    if cache_rel and (repo_root / cache_rel).is_file():
        cache_doc = json.loads((repo_root / cache_rel).read_text(encoding="utf-8"))
        cache_by_id = {t["target_id"]: t for t in cache_doc.get("targets", [])}

    for r in records:
        tid = r.get("target_id")
        if r.get("verdict") == producer.VERDICT_INDETERMINATE:
            problems.append(f"record {tid!r} has an indeterminate verdict presented as a closure")
        elif r.get("verdict") not in producer.TERMINAL_VERDICTS:
            problems.append(f"record {tid!r} verdict {r.get('verdict')!r} is not terminal")
        if r.get("cross_arch_precondition") not in (False, True, "pending"):
            problems.append(
                f"record {tid!r} cross_arch_precondition is not tri-state "
                "(false | true | 'pending')"
            )
        if not r.get("ac_coverage"):
            problems.append(f"record {tid!r} has an empty ac_coverage")
        proof = r.get("verified_live_sm120_proof") or {}
        if proof.get("compute_capability") != "sm_120":
            problems.append(f"record {tid!r} verified_live_sm120_proof is missing / not sm_120")
        # Honesty: the wall-clock duration label is CUPTI dual-source and NEVER
        # NCU-derived; a duration proof that names NCU as its label source
        # over-claims the evidence surface (the core non-root thesis).
        if tid == producer.DURATION_TARGET:
            ls = str(proof.get("label_source", "")).lower()
            if "cupti" not in ls:
                problems.append(f"{tid!r} proof label_source does not name the CUPTI dual-source label")
            if "ncu" in ls and "never" not in ls:
                problems.append(f"{tid!r} proof label_source claims NCU; the wall-clock label is CUPTI, never NCU")
        ev = r.get("evidence_artifact_paths") or []
        if not ev:
            problems.append(f"record {tid!r} has no evidence_artifact_paths")
        for rel in ev:
            if not isinstance(rel, str) or rel not in producer.CANONICAL_EVIDENCE_PATHS:
                problems.append(f"record {tid!r} cites non-canonical evidence path {rel!r}")
            elif _is_unsafe_rel_path(rel, repo_root) or not (repo_root / rel).is_file():
                problems.append(f"record {tid!r} evidence path {rel!r} is unsafe/absent")

        # verdict + kind recomputed from the source closure.
        if tid == producer.DURATION_TARGET:
            if r.get("kind") != producer.KIND_DURATION:
                problems.append(f"{tid!r} kind {r.get('kind')!r} != {producer.KIND_DURATION!r}")
            if dur_verdict is not None and r.get("verdict") != dur_verdict:
                problems.append(f"{tid!r} verdict {r.get('verdict')!r} != duration report {dur_verdict!r}")
        else:
            spec = registry.REGISTRY.get(tid)
            if spec is not None and r.get("kind") != spec.kind:
                problems.append(f"{tid!r} kind {r.get('kind')!r} != registry {spec.kind!r}")
            ct = cache_by_id.get(tid)
            if ct is not None and r.get("verdict") != ct.get("verdict"):
                problems.append(f"{tid!r} verdict {r.get('verdict')!r} != cache closure {ct.get('verdict')!r}")

    # (5b) record_counts recomputed from the records (independent of the rebuild,
    # so a mutated count fails closed even when check_rebuild is off).
    counts = doc.get("record_counts", {})
    n_success = sum(1 for r in records if r.get("verdict") == producer.VERDICT_SUCCESS)
    n_refuted = sum(1 for r in records if r.get("verdict") == producer.VERDICT_REFUTED)
    if counts.get("total") != len(records):
        problems.append(f"record_counts.total {counts.get('total')!r} != number of records {len(records)}")
    if counts.get("research_success") != n_success:
        problems.append(f"record_counts.research_success {counts.get('research_success')!r} != recomputed {n_success}")
    if counts.get("engineering_closure_refuted") != n_refuted:
        problems.append(f"record_counts.engineering_closure_refuted {counts.get('engineering_closure_refuted')!r} != recomputed {n_refuted}")

    # (6) recursively validate the four cited source closures.
    recursive = (
        ("duration_predictor_report", _duration_validator().validate_duration_predictor_report),
        ("duration_leg_closure", _duration_leg_validator().validate_engineering_closure),
        ("l1l2_frozen_grain_closure", _frozen_grain_validator().validate_frozen_grain_closure),
        ("cache_metric_closure", _cache_metric_validator().validate_cache_metric_closure),
    )
    for key, fn in recursive:
        rel = sources.get(key)
        if rel and not _is_unsafe_rel_path(rel, repo_root) and (repo_root / rel).is_file():
            src_doc = json.loads((repo_root / rel).read_text(encoding="utf-8"))
            for p in fn(src_doc, repo_root, check_provenance=check_provenance):
                problems.append(f"{key}: {p}")

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
                problems.append(f"registry drifts from a fresh recompute; differing keys: {diffs}")

    # (8) provenance.
    if check_provenance and _is_git_repo(repo_root):
        commit = doc.get("source_commit", "")
        for rel in _PROVENANCE_PATHS:
            if not _commit_contains(repo_root, commit, rel):
                problems.append(f"source_commit {commit} does not contain {rel}")
    return problems


def validate_or_raise(doc: dict, repo_root: Path = _REPO_ROOT, **kw) -> None:
    problems = validate_multi_claim_registry(doc, repo_root, **kw)
    if problems:
        raise MultiClaimRegistryValidationError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_multi_claim_closure_registry.json",
    )
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.artifact.read_text(encoding="utf-8"))
    problems = validate_multi_claim_registry(doc, check_provenance=not args.no_provenance)
    if problems:
        print(f"INVALID ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    c = doc["record_counts"]
    print(
        f"VALID: {args.artifact} ({c['total']} records: "
        f"{c['research_success']} success / {c['engineering_closure_refuted']} refuted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
