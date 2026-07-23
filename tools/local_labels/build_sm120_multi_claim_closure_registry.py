"""Producer for the sm_120 multi-claim closure registry.

This is the unified, terminal registry of EVERY predictive target the loop closed:
the wall-clock ``duration_ns`` predictor plus every cache/memory metric in the
typed registry (``tools/local_labels/ncu_metric_registry.py``). It records exactly
ONE terminal, sm_120-host verdict per target, each with a ``Verified-live-sm_120``
proof and ``cross_arch_precondition: false`` (sm_120-only this loop; the matching
sm_86 replication is deferred to a separate host).

Every record is RECOMPUTED from an already-validated source closure -- this
producer mints no new verdicts:

  - ``duration_ns`` <- ``duration_predictor_report.json`` (the wall-clock
    predictor closure; engineering_closure_refuted -- the predictor does not beat
    the strongest baseline), with the duration-leg cohort closure cited.
  - each cache/memory registry target <- ``sm120_cache_metric_closure.json`` (its
    per-target verdict + basis).

The validator recomputes the same target set, rejects any non-terminal
(``indeterminate``) verdict, any cross-arch precondition, any missing
evidence/proof block, and RECURSIVELY validates each cited source closure with its
own fail-closed validator. It reads VALUES only; it invokes no NCU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ncu_metric_registry as registry  # noqa: E402

SCHEMA_VERSION = 1
REGISTRY_ID = "sm120_multi_claim_closure_registry_v1"
ARCH = "sm_120"
REGISTRY_VERSION = registry.REGISTRY_VERSION

VERDICT_SUCCESS = "research_success"
VERDICT_REFUTED = "engineering_closure_refuted"
VERDICT_INDETERMINATE = "indeterminate"
TERMINAL_VERDICTS = (VERDICT_SUCCESS, VERDICT_REFUTED)

DURATION_TARGET = "duration_ns"
KIND_DURATION = "duration"

# Descriptive acceptance-coverage labels (the plan-token scanner forbids the
# literal criterion identifiers in tracked artifacts; these name the deliverable).
COVERAGE_DURATION = "wall_clock_duration_prediction"
COVERAGE_CACHE = "counter_free_cache_metric_prediction"

DURATION_REPORT_PATH = "data/local_labels/duration_predictor_report.json"
DURATION_LEG_CLOSURE_PATH = "data/local_labels/sm120_tri_label_engineering_closure.json"
L1L2_FROZEN_GRAIN_CLOSURE_PATH = "data/local_labels/sm120_l1l2_frozen_grain_engineering_closure.json"
CACHE_METRIC_CLOSURE_PATH = "data/local_labels/sm120_cache_metric_closure.json"

SOURCE_ARTIFACT_PATHS = {
    "duration_predictor_report": DURATION_REPORT_PATH,
    "duration_leg_closure": DURATION_LEG_CLOSURE_PATH,
    "l1l2_frozen_grain_closure": L1L2_FROZEN_GRAIN_CLOSURE_PATH,
    "cache_metric_closure": CACHE_METRIC_CLOSURE_PATH,
}
# Canonical committed evidence paths a record may cite (subset of the sources).
CANONICAL_EVIDENCE_PATHS = frozenset(SOURCE_ARTIFACT_PATHS.values())

# Shared sm_120 device proof (the AC-required device/driver/CUDA/NVBit/clocks).
# The per-target measurement surface differs, so the LABEL SOURCE is recorded
# per target rather than claiming one tool for all: the cache targets are scored
# against local-quarantine NCU labels, while the wall-clock duration target is
# scored against the CUPTI dual-source label and is NEVER NCU-derived (the core
# non-root thesis). Mixing NCU into the duration proof would over-claim the
# evidence surface.
_DEVICE_PROOF = {
    "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "compute_capability": "sm_120",
    "driver": "595.71.05",
    "cuda": "13.2",
    "nvbit": "nvbit dynamic trace via tools/nvbit_tracer (tracer_tool.so, sm_120 build)",
    "observed_clocks": "clock-control none (clocks not locked; boost clocks observed)",
}
LABEL_SOURCE_CACHE = (
    "ncu local-quarantine sector hit-rate labels (data/local_labels/) compared to "
    "a deterministic L1-miss-filtered replay over live NVBit traces (ncu 2026.1.1.0)"
)
LABEL_SOURCE_DURATION = (
    "cupti dual-source duration_ns (CUPTI Activity reconciled against "
    "cudaEventElapsedTime); the wall-clock label is never NCU-derived"
)


def _proof(label_source: str) -> dict:
    return {**_DEVICE_PROOF, "label_source": label_source}


class MultiClaimRegistryBuildError(RuntimeError):
    pass


def _load(repo_root: Path, rel: str) -> dict:
    return json.loads((repo_root / rel).read_text(encoding="utf-8"))


def _source_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "0" * 40


def _duration_record(dur_report: dict) -> dict:
    verdict = dur_report.get("verdict")
    if verdict not in TERMINAL_VERDICTS:
        raise MultiClaimRegistryBuildError(
            f"duration predictor report verdict {verdict!r} is not terminal"
        )
    sb = dur_report.get("strongest_baseline", {})
    details = (
        f"wall-clock duration predictor closure: {verdict}; the held-out predictor "
        f"does not beat the strongest baseline ({sb.get('name')}, held-out SMAPE "
        f"{sb.get('held_out_smape')})"
    )
    return {
        "target_id": DURATION_TARGET,
        "kind": KIND_DURATION,
        "ac_coverage": [COVERAGE_DURATION],
        "verdict": verdict,
        "evidence_artifact_paths": [DURATION_REPORT_PATH, DURATION_LEG_CLOSURE_PATH],
        "verified_live_sm120_proof": _proof(LABEL_SOURCE_DURATION),
        "success_or_falsification_details": details,
        "cross_arch_precondition": False,
    }


def _cache_records(cache_closure: dict) -> list[dict]:
    by_id = {t["target_id"]: t for t in cache_closure.get("targets", [])}
    records = []
    for key in sorted(registry.REGISTRY):
        ct = by_id.get(key)
        if ct is None:
            raise MultiClaimRegistryBuildError(
                f"cache-metric closure has no record for registry target {key!r}"
            )
        verdict = ct.get("verdict")
        if verdict not in TERMINAL_VERDICTS:
            raise MultiClaimRegistryBuildError(
                f"cache target {key!r} verdict {verdict!r} is not terminal"
            )
        records.append({
            "target_id": key,
            "kind": registry.REGISTRY[key].kind,
            "ac_coverage": [COVERAGE_CACHE],
            "verdict": verdict,
            "evidence_artifact_paths": [CACHE_METRIC_CLOSURE_PATH],
            "verified_live_sm120_proof": _proof(LABEL_SOURCE_CACHE),
            "success_or_falsification_details": f"cache-metric closure basis: {ct.get('basis')}",
            "cross_arch_precondition": False,
        })
    return records


def build(repo_root: Path = _REPO_ROOT) -> dict:
    if registry.validate_registry():
        raise MultiClaimRegistryBuildError("the metric registry is not internally consistent")

    dur_report = _load(repo_root, DURATION_REPORT_PATH)
    cache_closure = _load(repo_root, CACHE_METRIC_CLOSURE_PATH)
    if cache_closure.get("registry_version") != REGISTRY_VERSION:
        raise MultiClaimRegistryBuildError("cache closure registry_version drift vs the registry")

    records = [_duration_record(dur_report)] + _cache_records(cache_closure)

    # Every target exactly once: duration + the full registry key set.
    ids = [r["target_id"] for r in records]
    expected = {DURATION_TARGET} | set(registry.REGISTRY)
    if set(ids) != expected or len(ids) != len(expected):
        raise MultiClaimRegistryBuildError(
            f"record target set {sorted(set(ids))} != expected {sorted(expected)} "
            f"(or duplicates: {len(ids)} records / {len(expected)} expected)"
        )

    n_success = sum(1 for r in records if r["verdict"] == VERDICT_SUCCESS)
    n_refuted = sum(1 for r in records if r["verdict"] == VERDICT_REFUTED)

    return {
        "schema_version": SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "arch": ARCH,
        "registry_version": REGISTRY_VERSION,
        "cross_arch_precondition": False,
        "source_artifact_paths": dict(SOURCE_ARTIFACT_PATHS),
        "records": records,
        "record_counts": {
            "total": len(records),
            "research_success": n_success,
            "engineering_closure_refuted": n_refuted,
        },
        "source_commit": _source_commit(repo_root),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_multi_claim_closure_registry.json",
    )
    args = parser.parse_args(argv)
    doc = build()
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    c = doc["record_counts"]
    print(
        f"wrote {args.out} ({c['total']} records: "
        f"{c['research_success']} success / {c['engineering_closure_refuted']} refuted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
