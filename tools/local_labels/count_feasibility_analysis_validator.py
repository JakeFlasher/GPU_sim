"""Fail-closed, RECOMPUTING validator for the count-feasibility analysis artifact.

It re-derives the artifact from the typed metric registry via the producer's
``build()`` and fails on any drift (ignoring only the HEAD-dependent
``source_commit``), plus explicit closure checks:

1. closed-schema validation;
2. the target SET is recomputed from ``ncu_metric_registry``: every registry
   ``count`` + ``ratio`` target is present, there are no extras, and NO
   ``fraction`` (hit-rate) target leaks in (it is not a count target);
3. each target's ``ncu_tokens`` are recomputed from the registry;
4. verdicts are in the closed enum; a ``engineering_closure_refuted`` target
   carries a typed domain-mismatch reason (not ``none``) and no comparison
   geometry, and is NOT a comparable target -- so a non-comparable count is
   recorded as a falsification, never a silent failed-accuracy predictor;
5. the plan-mandated per-target components are all present + non-empty;
6. the summary counts are recomputed;
7. the cited Codex consult reference is the canonical committed path and exists;
8. a full rebuild-and-compare (every field recomputed from the registry);
9. reproducible-provenance: ``source_commit`` contains the producer + schema +
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

import ncu_metric_registry as reg  # noqa: E402
import build_count_feasibility_analysis as producer  # noqa: E402

_SCHEMA_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "count_feasibility_analysis.schema.json"
)
_CANONICAL_CONSULT = "data/local_labels/count_feasibility_codex_consult.md"
_REFUTED_GEOMETRY = "none (refuted)"
_MANDATED_COMPONENTS = (
    "trace_to_ncu_mapping",
    "aggregation_domain_mismatch",
    "coalescing_replay_write_allocation_note",
    "comparison_geometry",
)
_PROVENANCE_PATHS = (
    "tools/local_labels/build_count_feasibility_analysis.py",
    "tools/local_labels/count_feasibility_analysis_validator.py",
    "workloads/simbench_bw/schemas/count_feasibility_analysis.schema.json",
)
_REBUILD_EXCLUDE = ("source_commit",)


class CountFeasibilityValidationError(ValueError):
    pass


def _deep_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


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


def validate_count_feasibility(
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

    # (2) the target set recomputed from the registry.
    expected_keys = sorted(
        s.key for s in reg.METRIC_SPECS if s.kind in (reg.KIND_COUNT, reg.KIND_RATIO)
    )
    fraction_keys = {s.key for s in reg.METRIC_SPECS if s.kind == reg.KIND_FRACTION}
    doc_targets = {t["key"]: t for t in doc.get("targets", [])}
    doc_keys = sorted(doc_targets)
    if doc_keys != expected_keys:
        missing = sorted(set(expected_keys) - set(doc_keys))
        extra = sorted(set(doc_keys) - set(expected_keys))
        problems.append(
            f"target set drifts from the registry count/ratio targets; "
            f"missing={missing} extra={extra}"
        )
    leaked = sorted(set(doc_keys) & fraction_keys)
    if leaked:
        problems.append(
            f"fraction (hit-rate) targets leaked into the count-feasibility "
            f"analysis: {leaked} (a fraction is not a count target)"
        )

    if doc.get("registry_version") != reg.REGISTRY_VERSION:
        problems.append(
            f"registry_version {doc.get('registry_version')!r} != committed "
            f"{reg.REGISTRY_VERSION!r}"
        )

    # (3)(4)(5) per-target checks.
    for key in doc_keys:
        spec = reg.REGISTRY.get(key)
        target = doc_targets[key]
        if spec is not None:
            if target.get("kind") != spec.kind:
                problems.append(f"{key}: kind {target.get('kind')!r} != registry {spec.kind!r}")
            expected_tokens = producer._tokens_for(spec)
            if target.get("ncu_tokens") != expected_tokens:
                problems.append(f"{key}: ncu_tokens {target.get('ncu_tokens')} != registry-recomputed {expected_tokens}")
        verdict = target.get("verdict")
        reason = target.get("domain_mismatch_reason")
        geometry = target.get("comparison_geometry")
        if verdict == producer.VERDICT_REFUTED:
            if reason == "none":
                problems.append(f"{key}: refuted target must carry a typed domain-mismatch reason, not 'none'")
            if geometry != _REFUTED_GEOMETRY:
                problems.append(f"{key}: refuted target must record comparison_geometry={_REFUTED_GEOMETRY!r}, got {geometry!r}")
        elif verdict == producer.VERDICT_COMPARABLE:
            if geometry == _REFUTED_GEOMETRY:
                problems.append(f"{key}: comparable target must not record the refuted comparison_geometry")
        for component in _MANDATED_COMPONENTS:
            value = target.get(component)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{key}: mandated component {component!r} is missing/empty")

    # (6) summary counts recomputed.
    n_comp = sum(1 for t in doc_targets.values() if t.get("verdict") == producer.VERDICT_COMPARABLE)
    n_ref = sum(1 for t in doc_targets.values() if t.get("verdict") == producer.VERDICT_REFUTED)
    summary = doc.get("summary", {})
    if summary.get("n_targets") != len(doc_targets):
        problems.append(f"summary.n_targets {summary.get('n_targets')} != {len(doc_targets)}")
    if summary.get("n_comparable") != n_comp:
        problems.append(f"summary.n_comparable {summary.get('n_comparable')} != recomputed {n_comp}")
    if summary.get("n_refuted") != n_ref:
        problems.append(f"summary.n_refuted {summary.get('n_refuted')} != recomputed {n_ref}")

    # (7) canonical consult reference.
    if doc.get("codex_consult_reference") != _CANONICAL_CONSULT:
        problems.append(f"codex_consult_reference {doc.get('codex_consult_reference')!r} != canonical {_CANONICAL_CONSULT!r}")
    elif not (repo_root / _CANONICAL_CONSULT).is_file():
        problems.append(f"the cited Codex consult {_CANONICAL_CONSULT} does not exist")

    # (8) full rebuild-and-compare.
    if check_rebuild:
        try:
            rebuilt = producer.build()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rebuild failed: {exc}")
        else:
            a = {k: v for k, v in doc.items() if k not in _REBUILD_EXCLUDE}
            b = {k: v for k, v in rebuilt.items() if k not in _REBUILD_EXCLUDE}
            if not _deep_equal(a, b):
                diffs = sorted(k for k in set(a) | set(b) if not _deep_equal(a.get(k), b.get(k)))
                problems.append(f"artifact drifts from a fresh recompute; differing keys: {diffs}")

    # (9) provenance.
    if check_provenance and _is_git_repo(repo_root):
        commit = doc.get("source_commit", "")
        for rel in _PROVENANCE_PATHS:
            if not _commit_contains(repo_root, commit, rel):
                problems.append(f"source_commit {commit} does not contain {rel}")
    return problems


def validate_or_raise(doc: dict, repo_root: Path = _REPO_ROOT, **kw) -> None:
    problems = validate_count_feasibility(doc, repo_root, **kw)
    if problems:
        raise CountFeasibilityValidationError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "count_feasibility_analysis.json",
    )
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.artifact.read_text(encoding="utf-8"))
    problems = validate_count_feasibility(doc, check_provenance=not args.no_provenance)
    if problems:
        print(f"INVALID ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"VALID: {args.artifact} ({doc['summary']['n_comparable']} comparable / {doc['summary']['n_refuted']} refuted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
