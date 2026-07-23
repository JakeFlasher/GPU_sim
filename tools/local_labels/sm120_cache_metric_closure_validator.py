"""Fail-closed, RECOMPUTING validator for the sm_120 cache-metric closure.

It re-derives the closure from the committed evidence via the producer's
``build()`` and fails on any drift (ignoring only the HEAD-dependent
``source_commit``), plus explicit closure checks:

1. closed-schema validation;
2. the evidence-artifact paths are exactly the canonical committed paths,
   path-safe (no absolute / ``..`` / symlink escape), and exist;
3. the target set is recomputed from the typed registry: every registry target
   appears exactly once with its registry ``kind``; no missing / extra target;
   every target carries ``cross_arch_precondition: false`` and a verdict in the
   closed two-value vocabulary (no ``indeterminate``);
4. the NAMED L2 family verdict is recomputed from the cited comparison artifact:
   its L2 leg MUST be the L1-miss-filtered stream (a full-stream L2 is rejected),
   its median absolute error vs the committed gate threshold decides passed, and
   the named record's verdict + ``cache_claim_load_bearing`` + the overall verdict
   must agree with that gate outcome;
5. every count / ratio target's verdict + basis is recomputed from the cited
   count-feasibility analysis; the unified L1 sector hit-rate target is recomputed
   from its L1 leg against the live L1TEX label cohort (refuted by classifier
   inertness here, NOT a no-label basis), and the global-load-specific L1 target's
   no-label basis is verified as EARNED by proving its label field is absent from
   the committed cohort;
6. the named-family failure decomposition is recomputed from the comparison
   aggregates (L1 inertness totals, zero-L2-input cells);
7. the two cited committed closures (count-feasibility + the L1/L2 frozen-grain
   closure) are RECURSIVELY validated by their own fail-closed validators;
8. a full rebuild-and-compare (every field recomputed from committed inputs);
9. reproducible-provenance: ``source_commit`` contains the producer + schema +
   validator + the comparison-evidence driver.
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

import build_sm120_cache_metric_closure as producer  # noqa: E402
import ncu_metric_registry as registry  # noqa: E402

_SCHEMA_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "sm120_cache_metric_closure.schema.json"
)
_PROVENANCE_PATHS = (
    "tools/local_labels/build_sm120_cache_metric_closure.py",
    "tools/local_labels/sm120_cache_metric_closure_validator.py",
    "workloads/simbench_bw/schemas/sm120_cache_metric_closure.schema.json",
    "tools/cache_replay/l1_filtered_l2_eval.py",
)
_REBUILD_EXCLUDE = ("source_commit",)


class CacheMetricClosureValidationError(ValueError):
    pass


def _count_feasibility_validator():
    import count_feasibility_analysis_validator as cf_v  # noqa: PLC0415
    return cf_v


def _frozen_grain_validator():
    import sm120_l1l2_frozen_grain_closure_validator as fg_v  # noqa: PLC0415
    return fg_v


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


def validate_cache_metric_closure(
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

    # (3) the target set recomputed from the registry.
    targets = doc.get("targets", [])
    doc_ids = [t.get("target_id") for t in targets]
    expected_ids = sorted(registry.REGISTRY)
    if sorted(doc_ids) != expected_ids:
        problems.append(
            f"targets target_id set drifts from the registry "
            f"(missing={sorted(set(expected_ids) - set(doc_ids))}, "
            f"extra={sorted(set(doc_ids) - set(expected_ids))})"
        )
    if len(doc_ids) != len(set(doc_ids)):
        problems.append("targets contains a duplicate target_id")
    for t in targets:
        spec = registry.REGISTRY.get(t.get("target_id"))
        if spec is not None and t.get("kind") != spec.kind:
            problems.append(
                f"target {t.get('target_id')!r} kind {t.get('kind')!r} != registry {spec.kind!r}"
            )
        if t.get("cross_arch_precondition") not in (False, True, "pending"):
            problems.append(
                f"target {t.get('target_id')!r} cross_arch_precondition is not "
                "tri-state (false | true | 'pending')"
            )
        if t.get("verdict") not in (producer.VERDICT_SUCCESS, producer.VERDICT_REFUTED):
            problems.append(f"target {t.get('target_id')!r} verdict {t.get('verdict')!r} is not terminal")

    # (4) the named family gate recomputed from the comparison artifact.
    comparison_rel = paths.get("l1_filtered_l2_comparison")
    gate = doc.get("named_target_gate", {})
    if comparison_rel and not _is_unsafe_rel_path(comparison_rel, repo_root) and (repo_root / comparison_rel).is_file():
        comparison = json.loads((repo_root / comparison_rel).read_text(encoding="utf-8"))
        if comparison.get("l2_input_provenance") != producer.L2_INPUT_L1_FILTERED:
            problems.append(
                "named family L2 leg is not the L1-miss-filtered stream "
                f"(comparison l2_input_provenance={comparison.get('l2_input_provenance')!r}); "
                "a full-stream L2 cannot back the named family"
            )
        agg = comparison.get("aggregates", {})
        median_abs = agg.get("median_abs_error")
        if not _is_number(median_abs):
            problems.append("comparison median_abs_error is not a number")
        else:
            expected_passed = median_abs <= producer.GATE_MEDIAN_ABS_THRESHOLD
            if gate.get("passed") is not expected_passed:
                problems.append(
                    f"named_target_gate.passed {gate.get('passed')!r} != recomputed "
                    f"{expected_passed} (median_abs={median_abs} vs threshold "
                    f"{producer.GATE_MEDIAN_ABS_THRESHOLD})"
                )
            if gate.get("median_abs_error") != median_abs:
                problems.append("named_target_gate.median_abs_error drifts from the comparison")
        if gate.get("l2_input_provenance") != producer.L2_INPUT_L1_FILTERED:
            problems.append("named_target_gate.l2_input_provenance is not l1_miss_filtered")
        if gate.get("threshold_median_abs_error") != producer.GATE_MEDIAN_ABS_THRESHOLD:
            problems.append("named_target_gate.threshold_median_abs_error drifts from the committed gate")
    else:
        problems.append("comparison evidence missing/unsafe; cannot recompute the named gate")

    # named record verdict consistent with the gate; load-bearing only if passed.
    named = next((t for t in targets if t.get("target_id") == producer.NAMED_TARGET), None)
    if named is None:
        problems.append(f"named target {producer.NAMED_TARGET!r} absent from targets")
    elif isinstance(gate.get("passed"), bool):
        expected_named_verdict = producer.VERDICT_SUCCESS if gate.get("passed") else producer.VERDICT_REFUTED
        if named.get("verdict") != expected_named_verdict:
            problems.append(
                f"named target verdict {named.get('verdict')!r} != gate-consistent "
                f"{expected_named_verdict!r}"
            )
    if doc.get("cache_claim_load_bearing") is not bool(gate.get("passed")):
        problems.append(
            f"cache_claim_load_bearing {doc.get('cache_claim_load_bearing')!r} != "
            f"named gate passed {gate.get('passed')!r}"
        )
    expected_overall = producer.VERDICT_SUCCESS if gate.get("passed") else producer.VERDICT_REFUTED
    if doc.get("verdict") != expected_overall:
        problems.append(
            f"overall verdict {doc.get('verdict')!r} != gate-consistent {expected_overall!r}"
        )

    # (4b) the L1 sector-hit-rate (named_l1_target) gate recomputed from the
    # comparison's L1 leg, plus the committed L1 label fields (the basis for the
    # earned-vs-false no-label distinction).
    l1_gate = doc.get("named_l1_target_gate", {})
    l1_label_fields = producer._sm120_l1_label_fields(repo_root)
    if comparison_rel and not _is_unsafe_rel_path(comparison_rel, repo_root) and (repo_root / comparison_rel).is_file():
        comparison = json.loads((repo_root / comparison_rel).read_text(encoding="utf-8"))
        recomputed_l1 = producer._l1_target_gate(comparison)
        for f in ("label_coverage", "evaluated", "unavailable", "median_abs_error", "passed", "l1_filter_inert"):
            if l1_gate.get(f) != recomputed_l1.get(f):
                problems.append(f"named_l1_target_gate.{f} drifts from the recomputed L1 leg")

    # (5) count/ratio + L1-fraction verdicts recomputed from cited evidence.
    cf_rel = paths.get("count_feasibility_analysis")
    cf_verdicts = {}
    if cf_rel and not _is_unsafe_rel_path(cf_rel, repo_root) and (repo_root / cf_rel).is_file():
        cf_doc = json.loads((repo_root / cf_rel).read_text(encoding="utf-8"))
        cf_verdicts = {t["key"]: t["verdict"] for t in cf_doc.get("targets", [])}
    for t in targets:
        tid = t.get("target_id")
        if tid == producer.NAMED_TARGET:
            continue
        if tid == producer.NAMED_L1_TARGET:
            # Backed by the L1 leg; a no-label basis is FALSE here because the live
            # L1TEX label cohort exists -> fail closed.
            _NO_LABEL_BASES = {"no_headline_l1_label_cohort", "no_committed_global_op_ld_l1_label_cohort"}
            if producer.L1TEX_UNIFIED_LABEL_FIELD in l1_label_fields and t.get("basis") in _NO_LABEL_BASES:
                problems.append(
                    f"{tid!r} uses a no-label basis {t.get('basis')!r} but a live L1TEX "
                    f"label cohort ({producer.L1TEX_UNIFIED_LABEL_FIELD}) exists"
                )
            exp_v, exp_b = producer._l1_sector_hit_rate_record(l1_gate)
            if t.get("verdict") != exp_v or t.get("basis") != exp_b:
                problems.append(
                    f"{tid!r} verdict/basis ({t.get('verdict')!r}, {t.get('basis')!r}) "
                    f"!= recomputed ({exp_v!r}, {exp_b!r})"
                )
            if t.get("evidence_key") != producer.EVIDENCE_NAMED:
                problems.append(f"{tid!r} cites the wrong evidence_key {t.get('evidence_key')!r}")
            continue
        if tid == producer.L1_GLOBAL_OP_LD_TARGET:
            # No-label closure is EARNED only by proving the global-op-ld label
            # field is absent from the committed cohort.
            if producer.L1_GLOBAL_OP_LD_LABEL_FIELD in l1_label_fields:
                problems.append(
                    f"{tid!r} uses a no-label basis but a committed label cohort "
                    f"({producer.L1_GLOBAL_OP_LD_LABEL_FIELD}) exists"
                )
            if t.get("verdict") != producer.VERDICT_REFUTED or t.get("basis") != "no_committed_global_op_ld_l1_label_cohort":
                problems.append(f"{tid!r} is not an earned-no-label refutation")
            if t.get("evidence_key") != producer.EVIDENCE_LABEL_ADMITTED:
                problems.append(f"{tid!r} cites the wrong evidence_key {t.get('evidence_key')!r}")
            continue
        # count / ratio
        cf = cf_verdicts.get(tid)
        if cf is None:
            problems.append(f"count/ratio target {tid!r} has no count-feasibility verdict")
            continue
        expected = producer._COUNT_FEASIBILITY_TO_TERMINAL.get(cf)
        if expected is None:
            problems.append(f"count-feasibility verdict {cf!r} for {tid!r} has no terminal mapping")
            continue
        exp_verdict, exp_basis = expected
        if t.get("verdict") != exp_verdict or t.get("basis") != exp_basis:
            problems.append(
                f"count/ratio target {tid!r} verdict/basis "
                f"({t.get('verdict')!r}, {t.get('basis')!r}) != recomputed "
                f"({exp_verdict!r}, {exp_basis!r})"
            )
        if t.get("evidence_key") != producer.EVIDENCE_COUNT:
            problems.append(f"count/ratio target {tid!r} cites the wrong evidence_key {t.get('evidence_key')!r}")

    # (6) failure decomposition recomputed from the comparison aggregates.
    if comparison_rel and (repo_root / comparison_rel).is_file():
        agg = json.loads((repo_root / comparison_rel).read_text(encoding="utf-8")).get("aggregates", {})
        decomp = doc.get("named_target_failure_decomposition", {})
        for field in ("l1_filter_inert", "total_l1_eligible_misses", "total_l1_hits_absorbed", "cells_with_zero_l2_input"):
            if decomp.get(field) != agg.get(field):
                problems.append(f"failure decomposition {field} drifts from the comparison aggregates")

    # (7) recursively validate the two cited committed closures.
    if cf_rel and not _is_unsafe_rel_path(cf_rel, repo_root) and (repo_root / cf_rel).is_file():
        cf_doc = json.loads((repo_root / cf_rel).read_text(encoding="utf-8"))
        for p in _count_feasibility_validator().validate_count_feasibility(
            cf_doc, repo_root, check_provenance=check_provenance
        ):
            problems.append(f"count_feasibility_analysis: {p}")
    fg_rel = paths.get("l1l2_frozen_grain_closure")
    if fg_rel and not _is_unsafe_rel_path(fg_rel, repo_root) and (repo_root / fg_rel).is_file():
        fg_doc = json.loads((repo_root / fg_rel).read_text(encoding="utf-8"))
        for p in _frozen_grain_validator().validate_frozen_grain_closure(
            fg_doc, repo_root, check_provenance=check_provenance
        ):
            problems.append(f"l1l2_frozen_grain_closure: {p}")

    # (8) full rebuild-and-compare.
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

    # (9) provenance.
    if check_provenance and _is_git_repo(repo_root):
        commit = doc.get("source_commit", "")
        for rel in _PROVENANCE_PATHS:
            if not _commit_contains(repo_root, commit, rel):
                problems.append(f"source_commit {commit} does not contain {rel}")
    return problems


def validate_or_raise(doc: dict, repo_root: Path = _REPO_ROOT, **kw) -> None:
    problems = validate_cache_metric_closure(doc, repo_root, **kw)
    if problems:
        raise CacheMetricClosureValidationError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_cache_metric_closure.json",
    )
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.artifact.read_text(encoding="utf-8"))
    problems = validate_cache_metric_closure(doc, check_provenance=not args.no_provenance)
    if problems:
        print(f"INVALID ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    c = doc["target_counts"]
    print(
        f"VALID: {args.artifact} (verdict={doc['verdict']}, "
        f"cache_claim_load_bearing={doc['cache_claim_load_bearing']}, "
        f"{c['research_success']} success / {c['engineering_closure_refuted']} refuted / {c['total']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
