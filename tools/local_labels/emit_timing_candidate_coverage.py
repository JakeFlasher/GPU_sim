#!/usr/bin/env python3
"""Emit task5 candidate-family coverage artifacts for the long timing cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_CATALOG = (
    REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2" / "expansion_candidates.json"
)
DEFAULT_LIMITS = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_candidate_coverage_limits_sm_86.json"
)
DEFAULT_VERDICT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_candidate_coverage_verdict_sm_86.json"
)
REQUIRED_FAMILIES = (
    "gemm_family",
    "conv",
    "attention",
    "memory_stress",
)
REQUIRED_FEATURE_REFS = (
    "nvbit_sector_summary",
    "static_sass_features",
    "static_work",
    "cache_hit_predictions",
)
REQUIRED_LABELS = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _artifact_ref(path: Path) -> dict[str, str]:
    return {"path": _repo_rel(path), "sha256": _sha256(path)}


def _catalog_maps(catalog: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    slug_to_family: dict[str, str] = {
        "gemm_hmma_bf16_sm80": "gemm_family",
        "conv2d_implicit_gemm_bf16_sm80": "conv",
        "flash_attn_fwd_bf16_sm80": "attention",
        "dram_streaming_dot_product": "memory_stress",
    }
    candidates_by_family = {family: set() for family in REQUIRED_FAMILIES}
    for row in catalog.get("candidates", []):
        family = row.get("family_bucket")
        slug = row.get("slug")
        if family in candidates_by_family and isinstance(slug, str):
            candidates_by_family[family].add(slug)
            slug_to_family.setdefault(slug, family)
    for slug, family in slug_to_family.items():
        if family in candidates_by_family:
            candidates_by_family[family].add(slug)
    return slug_to_family, {
        family: sorted(values) for family, values in candidates_by_family.items()
    }


def _cell_feature_complete(cell: dict[str, Any]) -> bool:
    labels = cell.get("labels", {})
    if any(label not in labels for label in REQUIRED_LABELS):
        return False
    refs = cell.get("feature_artifacts", {})
    for name in REQUIRED_FEATURE_REFS:
        ref = refs.get(name)
        if not isinstance(ref, dict):
            return False
        path = REPO_ROOT / str(ref.get("path", ""))
        if not path.is_file() or _sha256(path) != ref.get("sha256"):
            return False
    return True


def build_coverage_artifacts(
    *,
    cohort: dict[str, Any],
    catalog: dict[str, Any],
    cohort_path: Path = DEFAULT_COHORT,
    catalog_path: Path = DEFAULT_CATALOG,
    source_commit: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slug_to_family, candidate_slugs = _catalog_maps(catalog)
    admitted_by_family = {family: [] for family in REQUIRED_FAMILIES}
    for cell in cohort.get("cells", []):
        family = slug_to_family.get(cell.get("slug"))
        if family in admitted_by_family and _cell_feature_complete(cell):
            admitted_by_family[family].append(cell)

    family_coverage = []
    uncovered = []
    for family in REQUIRED_FAMILIES:
        cells = admitted_by_family[family]
        admitted_slugs = sorted({str(cell["slug"]) for cell in cells})
        if cells:
            status = "admitted"
            typed_reason = None
        else:
            status = "typed_falsification"
            typed_reason = "no_same_binary_long_duration_label_capture_committed"
            uncovered.append(family)
        family_coverage.append(
            {
                "family_bucket": family,
                "status": status,
                "candidate_slugs": candidate_slugs[family],
                "admitted_slugs": admitted_slugs,
                "counted_success_cells": len(cells),
                "typed_reason": typed_reason,
            }
        )

    limits = {
        "schema_version": 1,
        "artifact_kind": "timing_candidate_coverage_limits",
        "arch": cohort["arch"],
        "source_commit": source_commit or _source_commit(),
        "admitted_cohort_artifact": _artifact_ref(cohort_path),
        "candidate_catalog_artifact": _artifact_ref(catalog_path),
        "success_policy": (
            "only cells in the admitted long-duration cohort count as "
            "successful timing cells"
        ),
        "required_family_buckets": list(REQUIRED_FAMILIES),
        "family_coverage": family_coverage,
    }
    if uncovered:
        verdict = {
            "schema_version": 1,
            "artifact_kind": "timing_candidate_coverage_verdict",
            "arch": cohort["arch"],
            "source_commit": source_commit or _source_commit(),
            "terminal_state": "verdict_falsification",
            "prevents_overall_research_success": True,
            "coverage_limits_artifact": {"path": "", "sha256": "0" * 64},
            "required_family_buckets": list(REQUIRED_FAMILIES),
            "uncovered_required_families": uncovered,
            "typed_reason": "non_gemm_long_candidate_path_not_populated",
            "detail": (
                "The admitted long-duration timing cohort is missing one or "
                "more required task5 candidate-family success paths."
            ),
        }
    else:
        verdict = {
            "schema_version": 1,
            "artifact_kind": "timing_candidate_coverage_verdict",
            "arch": cohort["arch"],
            "source_commit": source_commit or _source_commit(),
            "terminal_state": "research_success",
            "prevents_overall_research_success": False,
            "coverage_limits_artifact": {"path": "", "sha256": "0" * 64},
            "required_family_buckets": list(REQUIRED_FAMILIES),
            "uncovered_required_families": [],
            "typed_reason": None,
            "detail": (
                "The admitted long-duration timing cohort contains same-binary, "
                "label-complete, feature-complete cells for every required "
                "task5 candidate family."
            ),
        }
    return limits, verdict


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--limits-out", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--verdict-out", type=Path, default=DEFAULT_VERDICT)
    args = parser.parse_args(argv)
    limits, verdict = build_coverage_artifacts(
        cohort=_load(args.cohort),
        catalog=_load(args.catalog),
        cohort_path=args.cohort,
        catalog_path=args.catalog,
    )
    _write(args.limits_out, limits)
    verdict["coverage_limits_artifact"] = _artifact_ref(args.limits_out)
    _write(args.verdict_out, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
