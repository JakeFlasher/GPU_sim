"""Re-harvest per-cell Tier-2 traces and replay the simulator at every
candidate L2 associativity, in ONE NVBit injection per cell.

R30 deleted the per-cell `.t2a` traces (kept only their SHA), so the sm_86
geometry sweep needs a fresh harvest. To avoid one NVBit pass per associativity,
this harvester injects ONCE per cell, parses the `.t2a` into a LaunchFrame, then
runs `predict_from_launch_frame` at each candidate `associativity_l2` and records
all raw L1/L2 hit rates in a single per-cell sidecar. The large `.t2a` is deleted
after parsing (its SHA + the per-assoc raw predictions are the kept evidence).

Per-cell sidecar `raw_pred_sweep_sidecars_sm86/<slug>.json`:
  { schema_version, slug, tier2_sidecar_sha256, captured_at_utc,
    by_assoc: { "4": {raw_l1, raw_l2}, "8": {...}, ... },
    scored: bool, typed_reason: str }

CLAUDE.md Rule 1 (NVBit calibration-only on sm_86); Rule 2 (no clock mutation).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from baselines_harness.metrics.tier2_sidecar_parser import parse_launch_frames  # noqa: E402
from baselines_harness.metrics.cache_predictor import predict_from_launch_frame  # noqa: E402

_CACHE_PRED_ROOT: Final[Path] = (
    _REPO_ROOT / "scripts" / "baselines_harness" / "calibration" / "cache_prediction"
)
_RAW_TABLE: Final[Path] = _CACHE_PRED_ROOT / "sm86_expanded_raw_predictions_v1.toml"
_SIDECAR_DIR: Final[Path] = _CACHE_PRED_ROOT / "raw_pred_sweep_sidecars_sm86"
_COHORT_B_ROOT: Final[Path] = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "tests" / "calibration_cohort_b"
)
_NVBIT_SO: Final[Path] = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
)

SIDECAR_SCHEMA_VERSION: Final[int] = 2  # v2 adds the deterministic dense-reuse feature
TIER2_BUDGET_BYTES: Final[int] = 268435456


def dense_reuse_feature(frame) -> dict:
    """Deterministic, geometry-independent dense-reuse signal from a launch frame.
    `sector_id` is the cache-line id the SetAssociativeSimulator consumes
    (cache_predictor.py), so distinct sector_ids = the read footprint and
    `reuse_factor = total sector accesses / distinct sectors` measures temporal
    locality: high reuse -> data re-read -> L2 hits; reuse ~1-3 -> streaming ->
    L2 ~ 0. Computed from the trace ONLY (never the NCU label)."""
    distinct: set = set()
    total = 0
    for r in frame.records:
        for s in r.sector_ids:
            distinct.add(s)
            total += 1
    n = len(distinct)
    return {
        "distinct_sectors": n,
        "total_sector_accesses": total,
        "reuse_factor": (total / n) if n else 0.0,
        "truncated": frame.truncation is not None,
    }

# Candidate L2 associativities. Powers of 2 only, so per-slice byte count
# (6 MiB / 6 slices = 1 MiB) divides into integer set counts under the simulator
# invariant `(l2_capacity_bytes // num_l2_slices) % (line_size_bytes * assoc) == 0`
# (1048576 / 128 = 8192; 8192 % assoc == 0 for every candidate). Ampere GA102 is
# 16-way; the sweep empirically tests the neighbourhood.
CANDIDATES: Final[tuple[int, ...]] = (4, 8, 16, 32, 64)


def _geometry_spec(associativity_l2: int) -> dict:
    """sm_86 static device card with the candidate associativity_l2. All other
    fields are byte-identical to the frozen pilot `_SM86_GEOMETRY`."""
    return {
        "arch": "sm_86",
        "l1_kb_per_sm": 192,
        "l1_kb_per_sm_source": "static_device_card",
        "l2_capacity_bytes": 6 * 1024 * 1024,
        "capacity_source": "static_device_card",
        "num_l2_slices": 6,
        "num_l2_slices_source": "static_device_card",
        "sector_size_bytes": 32,
        "sector_size_source": "static_device_card",
        "line_size_bytes": 128,
        "line_size_source": "static_device_card",
        "associativity_l2": associativity_l2,
        "associativity_l2_source": "empirical_sweep_candidate",
        "l2_constant_memory_counts_for_read_score": False,
        "l2_constant_memory_counts_for_read_score_source": "static_device_card_sm_86",
        "geometry_source": "empirical_sweep_candidate",
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _harvest_env() -> dict:
    env = {k: v for k, v in os.environ.items()}
    env["CUDA_INJECTION64_PATH"] = str(_NVBIT_SO)
    env["METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP"] = "1"
    env["METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES"] = str(TIER2_BUDGET_BYTES)
    env["METRIC_EXTRACTOR_USER_MODULE_REGEX"] = ".*"
    env["METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"] = "1"
    env["CUTLASS_DIR"] = str(_REPO_ROOT / "third_party" / "cutlass")
    return env


def _scored_slugs(repo_root: Path) -> list[str]:
    import tomllib
    raw = tomllib.loads((repo_root / _RAW_TABLE.relative_to(_REPO_ROOT)).read_text())
    return sorted(e["slug"] for e in raw["entries"] if e["scored"])


def harvest_cell(slug: str, probe: Path, work_root: Path) -> dict:
    """One NVBit injection; replay the simulator at every candidate assoc.
    Never raises — an unscoreable cell carries a typed reason and no raws."""
    captured = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    base = {"schema_version": SIDECAR_SCHEMA_VERSION, "slug": slug,
            "captured_at_utc": captured, "candidates": list(CANDIDATES)}
    if not probe.is_file():
        return {**base, "scored": False, "typed_reason": "probe_binary_missing",
                "tier2_sidecar_sha256": "", "by_assoc": {}}
    cell_work = work_root / slug
    cell_work.mkdir(parents=True, exist_ok=True)
    t2a = cell_work / "tier2_address_trace.t2a"
    if t2a.exists():
        t2a.unlink()
    try:
        subprocess.run([str(probe)], cwd=str(cell_work), env=_harvest_env(),
                       capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return {**base, "scored": False, "typed_reason": "tier2_dump_absent",
                "tier2_sidecar_sha256": "", "by_assoc": {}}
    if not t2a.is_file() or t2a.stat().st_size == 0:
        return {**base, "scored": False, "typed_reason": "tier2_dump_absent",
                "tier2_sidecar_sha256": "", "by_assoc": {}}
    tier2_sha = _sha256_file(t2a)
    frames = parse_launch_frames(t2a)
    t2a.unlink()
    if not frames:
        return {**base, "scored": False, "typed_reason": "tier2_parse_no_frames",
                "tier2_sidecar_sha256": tier2_sha, "by_assoc": {}}
    frame = frames[0]
    reuse = dense_reuse_feature(frame)  # before truncation strip (uses .truncation)
    if frame.truncation is not None:
        frame = dataclasses.replace(frame, truncation=None)
    by_assoc: dict = {}
    for assoc in CANDIDATES:
        try:
            result = predict_from_launch_frame(
                frame, cell_name=slug, geometry_spec=_geometry_spec(assoc))
        except Exception as exc:  # noqa: BLE001
            by_assoc[str(assoc)] = {"scored": False,
                                    "typed_reason": f"predictor_exception:{exc!r}"}
            continue
        if result.get("result_kind") != "scalar":
            by_assoc[str(assoc)] = {"scored": False,
                                    "typed_reason": "predictor_typed_unavailable",
                                    "predictor_reasons": list(result.get("reasons") or [])}
            continue
        by_assoc[str(assoc)] = {"scored": True, "typed_reason": "",
                                "raw_l1_hit_rate": float(result["raw_l1_hit_rate"]),
                                "raw_l2_hit_rate": float(result["raw_l2_hit_rate"])}
    any_scored = any(v.get("scored") for v in by_assoc.values())
    return {**base, "scored": any_scored,
            "typed_reason": "" if any_scored else "predictor_typed_unavailable",
            "tier2_sidecar_sha256": tier2_sha, "by_assoc": by_assoc,
            "dense_reuse": reuse}


def run_harvest(repo_root: Path = _REPO_ROOT, *, resume: bool = True) -> dict:
    sidecar_dir = repo_root / _SIDECAR_DIR.relative_to(_REPO_ROOT)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    work_root = (repo_root / ".humanize" / "rlcr" / "2026-05-29_02-41-41"
                 / "r32_sweep" / "t2a_work")
    slugs = _scored_slugs(repo_root)
    summary = {"scored": [], "unavailable": [], "skipped": []}
    for slug in slugs:
        out = sidecar_dir / f"{slug}.json"
        if resume and out.is_file():
            try:
                prev = json.loads(out.read_text())
            except (OSError, ValueError):
                prev = None
            if prev and prev.get("schema_version") == SIDECAR_SCHEMA_VERSION:
                summary["skipped"].append(slug)
                continue
        t0 = time.monotonic()
        sc = harvest_cell(slug, _COHORT_B_ROOT / slug, work_root)
        out.write_text(json.dumps(sc, indent=2, sort_keys=True) + "\n")
        (summary["scored"] if sc["scored"] else summary["unavailable"]).append(slug)
        a16 = sc.get("by_assoc", {}).get("16", {})
        print(f"  {slug}: scored={sc['scored']} reason={sc['typed_reason']!r} "
              f"a16_l1={a16.get('raw_l1_hit_rate', -1):.3f} a16_l2={a16.get('raw_l2_hit_rate', -1):.3f} "
              f"({time.monotonic()-t0:.1f}s)", flush=True)
    print(f"sweep-harvest: {len(summary['scored'])} scored, "
          f"{len(summary['unavailable'])} unavailable, {len(summary['skipped'])} skipped "
          f"of {len(slugs)} scored cells", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    run_harvest(_REPO_ROOT, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
