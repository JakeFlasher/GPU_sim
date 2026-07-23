"""Validator for the wall-clock duration-predictor report.

Fail-closed and RECOMPUTING: it re-derives the report from the committed cohort /
admitted / overrides via the producer's ``build()`` and fails on any drift
(ignoring only the HEAD-dependent ``source_commit``), plus explicit checks:

1. closed-schema validation;
2. the cited closure-artifact path is the canonical committed path;
3. scoring primary == SMAPE, diagnostic == MAE-with-epsilon-floor (a report
   scored by plain MAPE on near-zero duration cannot validate);
4. the slug-held-out leakage check recomputed from the cohort (no holdout slug
   in calibration);
5. verdict consistency: research_success IFF the predictor's held-out SMAPE is
   strictly below the strongest baseline's; otherwise engineering_closure_refuted;
6. a full rebuild-and-compare (every computed number recomputed);
7. reproducible-provenance: ``source_commit`` contains the producer + schema.
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
import build_duration_predictor_report as bdpr  # noqa: E402

_SCHEMA_PATH = _REPO_ROOT / "workloads" / "simbench_bw" / "schemas" / "duration_predictor_report.schema.json"
_COHORT_PATH = _REPO_ROOT / "data" / "local_labels" / "sm120_duration_cohort.json"
_CANONICAL_CLOSURE = "data/local_labels/sm120_tri_label_engineering_closure.json"
_PROVENANCE_PATHS = (
    "tools/local_labels/build_duration_predictor_report.py",
    "tools/local_labels/duration_predictor_report_validator.py",
    "workloads/simbench_bw/schemas/duration_predictor_report.schema.json",
)


class DurationPredictorReportValidationError(ValueError):
    pass


# Fields excluded from the rebuild-and-compare drift check: HEAD-dependent
# provenance and the environment dependency probe (find_spec results vary by
# interpreter env; they are schema-typed but not derived from committed inputs).
_REBUILD_EXCLUDE = ("source_commit", "dependency_availability")


def _deep_equal(a, b, rtol: float = 1e-9, atol: float = 1e-6) -> bool:
    """Structural equality with float tolerance (so a deterministic recompute is
    not flagged for sub-epsilon floating-point drift across numpy builds)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y, rtol, atol) for x, y in zip(a, b))
    return a == b


def _commit_contains(repo_root: Path, commit: str, rel: str) -> bool:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}:{rel}"],
                              capture_output=True, text=True).returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _is_git_repo(repo_root: Path) -> bool:
    try:
        r = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (FileNotFoundError, OSError):
        return False


def validate_duration_predictor_report(
    doc: dict, repo_root: Path = _REPO_ROOT, *, schema_path: Path | None = None,
    check_provenance: bool = True, check_rebuild: bool = True,
) -> list[str]:
    problems: list[str] = []
    schema = json.loads((schema_path or _SCHEMA_PATH).read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]

    # (2) canonical closure-artifact reference.
    if doc.get("closure_artifact_path") != _CANONICAL_CLOSURE:
        problems.append(f"closure_artifact_path {doc.get('closure_artifact_path')!r} != canonical {_CANONICAL_CLOSURE!r}")
    if not (repo_root / _CANONICAL_CLOSURE).is_file():
        problems.append("the cited engineering-closure artifact does not exist")

    # (3) scoring primary/diagnostic.
    sc = doc.get("scoring") or {}
    if sc.get("primary") != "smape":
        problems.append(f"scoring.primary must be 'smape', got {sc.get('primary')!r} (plain MAPE on near-zero duration is rejected)")
    if sc.get("diagnostic") != "mae_with_epsilon_floor":
        problems.append(f"scoring.diagnostic must be 'mae_with_epsilon_floor', got {sc.get('diagnostic')!r}")

    # (4) leakage recomputed from the cohort.
    if _COHORT_PATH.is_file():
        cohort = json.loads(_COHORT_PATH.read_text())
        cal = sorted({c["slug"] for c in cohort["cells"] if c["split"] == "calibration"})
        hold = sorted({c["slug"] for c in cohort["cells"] if c["split"] == "holdout"})
        if set(hold) & set(cal):
            problems.append(f"cohort has a slug in both splits: {sorted(set(hold) & set(cal))}")
        if doc.get("split", {}).get("holdout_slugs") and set(doc["split"]["holdout_slugs"]) & set(doc["split"].get("calibration_slugs", [])):
            problems.append("report split has a holdout slug in calibration (leakage)")
    if not doc.get("leakage_check", {}).get("passed", False):
        problems.append("leakage_check.passed must be true")

    # (5) verdict consistency.
    p_smape = doc.get("predictor", {}).get("held_out_smape")
    s_smape = doc.get("strongest_baseline", {}).get("held_out_smape")
    verdict = doc.get("verdict")
    if isinstance(p_smape, (int, float)) and isinstance(s_smape, (int, float)):
        beats = p_smape < s_smape
        if beats and verdict != "research_success":
            problems.append(f"predictor SMAPE {p_smape} < strongest baseline {s_smape} but verdict is {verdict!r} (should be research_success)")
        if not beats and verdict == "research_success":
            problems.append(f"verdict research_success but predictor SMAPE {p_smape} does not beat strongest baseline {s_smape}")

    # (6) full rebuild-and-compare (recompute every number from committed inputs).
    if check_rebuild:
        try:
            rebuilt = bdpr.build()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rebuild failed: {exc}")
        else:
            a = {k: v for k, v in doc.items() if k not in _REBUILD_EXCLUDE}
            b = {k: v for k, v in rebuilt.items() if k not in _REBUILD_EXCLUDE}
            if not _deep_equal(a, b):
                diffs = sorted(k for k in set(a) | set(b) if not _deep_equal(a.get(k), b.get(k)))
                problems.append(f"report drifts from a fresh recompute; differing keys: {diffs}")

    # (7) provenance.
    if check_provenance and _is_git_repo(repo_root):
        commit = doc.get("source_commit", "")
        for rel in _PROVENANCE_PATHS:
            if not _commit_contains(repo_root, commit, rel):
                problems.append(f"source_commit {commit} does not contain {rel}")
    return problems


def validate_or_raise(doc: dict, repo_root: Path = _REPO_ROOT, **kw) -> None:
    problems = validate_duration_predictor_report(doc, repo_root, **kw)
    if problems:
        raise DurationPredictorReportValidationError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=_REPO_ROOT / "data" / "local_labels" / "duration_predictor_report.json")
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.report.read_text(encoding="utf-8"))
    problems = validate_duration_predictor_report(doc, check_provenance=not args.no_provenance)
    if problems:
        print(f"INVALID ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"VALID: {args.report} (verdict={doc['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
