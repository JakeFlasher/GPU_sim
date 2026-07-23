"""Emit the sm_86 expanded RAW cache-predictor predictions (calibration input).

The IsotonicCalibrator maps RAW simulator predictions -> NCU labels, so the fit
needs a raw L1/L2 hit-rate prediction per labeled cell PAIRED with the NCU
read-only-sector label. Raw must come from the simulator, never `raw == label` (the
identity-fit-smokescreen discipline). This module harvests the raw half for the
completed cohort_b cells of `label_ledger_v1_sm86_expanded.toml`:

  per cell: NVBit-inject the metric_extractor Tier-2 address dump into the
  standalone cohort_b probe binary -> tier2_address_trace.t2a -> parse_launch_frames
  -> predict_from_launch_frame (SetAssociativeSimulator, sm_86 static device card)
  -> raw L1/L2 hit-rate. A cell the simulator cannot honestly score (policy
  uncalibrated / geometry unresolved / parse failure) is recorded with a typed
  reason and NO fabricated raw value.

Two phases (decoupled for deterministic emission): `run_raw_harvest` writes one
`<slug>.json` raw-prediction sidecar per cell under `raw_pred_sidecars_sm86/`;
`emit_table` reads the sidecars + the expanded NCU-label ledger and writes
`sm86_expanded_raw_predictions_v1.toml`, one-way cross-stamping the expanded
ledger SHA. The seed cells already carry raw predictions in the frozen
`sm86_baseline_predictions_v1.toml`; they are referenced one-way, not re-harvested.

CLAUDE.md Rule 1: NVBit is calibration-only on sm_86; this module lives under
`calibration/cache_prediction/` and never on the deployed / B200 path. Rule 2: no
clock-policy mutation (NVBit injection does not touch clocks).
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
from baselines_harness.metrics.predictor_design_coverage import (  # noqa: E402
    lookup_predictor_design_coverage,
)

_CACHE_PRED_ROOT: Final[Path] = (
    _REPO_ROOT / "scripts" / "baselines_harness" / "calibration" / "cache_prediction"
)
_EXPANDED_LEDGER: Final[Path] = _CACHE_PRED_ROOT / "label_ledger_v1_sm86_expanded.toml"
_SIDECAR_DIR: Final[Path] = _CACHE_PRED_ROOT / "raw_pred_sidecars_sm86"
_OUT_TABLE: Final[Path] = _CACHE_PRED_ROOT / "sm86_expanded_raw_predictions_v1.toml"
_COHORT_B_ROOT: Final[Path] = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "tests" / "calibration_cohort_b"
)
_NVBIT_SO: Final[Path] = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
)
_NCU_DIR: Final[str] = "/usr/local/cuda-13.2/nsight-compute-2026.1.0"

TABLE_SCHEMA_VERSION: Final[int] = 1
SIDECAR_SCHEMA_VERSION: Final[int] = 1
HOST_ARCH: Final[str] = "sm_86"
TIER2_BUDGET_BYTES: Final[int] = 268435456

# sm_86 static device-card geometry (6 MiB / 6 slices / 16-way, 128 B line),
# byte-identical to the frozen pilot's `_SM86_GEOMETRY` so the expanded raw
# predictions are produced under the same simulator geometry as the seed.
_SM86_GEOMETRY: Final[dict] = {
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
    "associativity_l2": 16,
    "associativity_l2_source": "static_device_card",
    "l2_constant_memory_counts_for_read_score": False,
    "l2_constant_memory_counts_for_read_score_source": "static_device_card_sm_86",
    "geometry_source": "sm86_pilot_static_device_card_v1",
}

# Closed typed-reason vocabulary for a cell the simulator cannot score.
RAW_UNAVAILABLE_REASONS: Final[tuple[str, ...]] = (
    "tier2_dump_absent",
    "tier2_parse_no_frames",
    "predictor_typed_unavailable",
    "probe_binary_missing",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toml_s(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _harvest_env() -> dict:
    env = {k: v for k, v in os.environ.items()}
    env["PATH"] = _NCU_DIR + os.pathsep + env.get("PATH", "")
    env["CUDA_INJECTION64_PATH"] = str(_NVBIT_SO)
    env["METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP"] = "1"
    env["METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES"] = str(TIER2_BUDGET_BYTES)
    env["METRIC_EXTRACTOR_USER_MODULE_REGEX"] = ".*"
    env["METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"] = "1"
    env["CUTLASS_DIR"] = str(_REPO_ROOT / "third_party" / "cutlass")
    return env


def _load_completed_cohort_b(repo_root: Path) -> list[dict]:
    """The completed cohort_b rows of the expanded NCU-label ledger (the cells
    that have a real NCU label to pair a raw prediction against)."""
    import tomllib
    ledger = repo_root / _EXPANDED_LEDGER.relative_to(_REPO_ROOT)
    data = tomllib.loads(ledger.read_text())
    return [
        e for e in data["entries"]
        if e.get("source") == "cohort_b_probe" and e.get("completed") is True
    ]


def harvest_raw_prediction(slug: str, probe: Path, work_root: Path) -> dict:
    """Tier-2-trace the probe, simulate, return a raw-prediction sidecar dict.
    Never raises — unscoreable cells surface a typed reason + no raw value."""
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    base = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "slug": slug,
        "language": "cuda_cpp",
        "geometry_source": _SM86_GEOMETRY["geometry_source"],
        "associativity_l2": _SM86_GEOMETRY["associativity_l2"],
        "num_l2_slices": _SM86_GEOMETRY["num_l2_slices"],
        "captured_at_utc": captured_at,
    }
    if not probe.is_file():
        return {**base, "scored": False, "typed_reason": "probe_binary_missing",
                "raw_l1_hit_rate": -1.0, "raw_l2_hit_rate": -1.0, "tier2_sidecar_sha256": ""}

    cell_work = work_root / slug
    cell_work.mkdir(parents=True, exist_ok=True)
    t2a = cell_work / "tier2_address_trace.t2a"
    if t2a.exists():
        t2a.unlink()
    try:
        # The legit probes finish (run + Tier-2 dump) well under this; probes
        # that hang under NVBit injection (cuBLAS/cuDNN library kernels) are
        # killed here and recorded `tier2_dump_absent` (a faithful
        # NVBit-untraceable terminal, not a fabricated raw value).
        subprocess.run([str(probe)], cwd=str(cell_work), env=_harvest_env(),
                       capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return {**base, "scored": False, "typed_reason": "tier2_dump_absent",
                "raw_l1_hit_rate": -1.0, "raw_l2_hit_rate": -1.0, "tier2_sidecar_sha256": ""}

    if not t2a.is_file() or t2a.stat().st_size == 0:
        return {**base, "scored": False, "typed_reason": "tier2_dump_absent",
                "raw_l1_hit_rate": -1.0, "raw_l2_hit_rate": -1.0, "tier2_sidecar_sha256": ""}
    tier2_sha = _sha256_file(t2a)
    frames = parse_launch_frames(t2a)
    t2a.unlink()  # large transient binary; the raw prediction + SHA are the kept evidence
    if not frames:
        return {**base, "scored": False, "typed_reason": "tier2_parse_no_frames",
                "raw_l1_hit_rate": -1.0, "raw_l2_hit_rate": -1.0,
                "tier2_sidecar_sha256": tier2_sha}
    frame = frames[0]
    if frame.truncation is not None:
        frame = dataclasses.replace(frame, truncation=None)
    result = predict_from_launch_frame(frame, cell_name=slug, geometry_spec=_SM86_GEOMETRY)
    if result.get("result_kind") != "scalar":
        return {**base, "scored": False, "typed_reason": "predictor_typed_unavailable",
                "predictor_reasons": list(result.get("reasons") or []),
                "raw_l1_hit_rate": -1.0, "raw_l2_hit_rate": -1.0,
                "tier2_sidecar_sha256": tier2_sha}
    return {**base, "scored": True, "typed_reason": "",
            "raw_l1_hit_rate": float(result["raw_l1_hit_rate"]),
            "raw_l2_hit_rate": float(result["raw_l2_hit_rate"]),
            "tier2_sidecar_sha256": tier2_sha}


def run_raw_harvest(repo_root: Path = _REPO_ROOT, *, resume: bool = True) -> dict:
    sidecar_dir = repo_root / _SIDECAR_DIR.relative_to(_REPO_ROOT)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    work_root = repo_root / ".humanize" / "rlcr" / "2026-05-29_02-41-41" / "r30_harvest" / "t2a_work"
    rows = _load_completed_cohort_b(repo_root)
    summary = {"scored": [], "unavailable": [], "skipped": []}
    for row in rows:
        slug = row["slug"]
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
        sc = harvest_raw_prediction(slug, _COHORT_B_ROOT / slug, work_root)
        out.write_text(json.dumps(sc, indent=2, sort_keys=True) + "\n")
        (summary["scored"] if sc["scored"] else summary["unavailable"]).append(slug)
        print(f"  {slug}: scored={sc['scored']} reason={sc['typed_reason']!r} "
              f"raw_l1={sc['raw_l1_hit_rate']:.4f} raw_l2={sc['raw_l2_hit_rate']:.4f} "
              f"({time.monotonic()-t0:.1f}s)", flush=True)
    print(f"raw-harvest summary: {len(summary['scored'])} scored, "
          f"{len(summary['unavailable'])} unavailable, {len(summary['skipped'])} skipped "
          f"of {len(rows)} completed cohort_b cells", flush=True)
    return summary


def emit_table(repo_root: Path = _REPO_ROOT) -> Path:
    import tomllib
    sidecar_dir = repo_root / _SIDECAR_DIR.relative_to(_REPO_ROOT)
    ledger_path = repo_root / _EXPANDED_LEDGER.relative_to(_REPO_ROOT)
    out_path = repo_root / _OUT_TABLE.relative_to(_REPO_ROOT)
    ledger_sha = _sha256_file(ledger_path)
    labels = {
        e["slug"]: e for e in _load_completed_cohort_b(repo_root)
    }

    rows: list[dict] = []
    for slug in sorted(labels):
        sc_path = sidecar_dir / f"{slug}.json"
        if not sc_path.is_file():
            continue
        sc = json.loads(sc_path.read_text())
        rows.append({
            "slug": slug,
            "predictor_bucket": labels[slug].get("predictor_bucket", ""),
            "scored": bool(sc["scored"]),
            "typed_reason": sc["typed_reason"],
            "raw_l1_hit_rate": sc["raw_l1_hit_rate"],
            "raw_l2_hit_rate": sc["raw_l2_hit_rate"],
            "ncu_l1_hit_rate": labels[slug].get("ncu_l1_hit_rate_mean", -1.0),
            "ncu_l2_hit_rate": labels[slug].get("ncu_l2_hit_rate_mean", -1.0),
            "tier2_sidecar_sha256": sc.get("tier2_sidecar_sha256", ""),
            "sidecar_path": str(sc_path.relative_to(repo_root)),
            "sidecar_sha256": _sha256_file(sc_path),
        })
    scored = [r for r in rows if r["scored"]]
    captured = max((json.loads((sidecar_dir / f"{r['slug']}.json").read_text())["captured_at_utc"]
                    for r in rows), default="1970-01-01T00:00:00+00:00")

    L: list[str] = []
    L.append("# sm_86 EXPANDED raw cache-predictor predictions (calibrator fit input).")
    L.append("# GENERATED (deterministic) by")
    L.append("# calibration/cache_prediction/sm86_expanded/emit_expanded_raw_predictions.py")
    L.append("# from the raw_pred_sidecars_sm86/<slug>.json sidecars (Tier-2 NVBit address")
    L.append("# trace -> SetAssociativeSimulator -> raw L1/L2). Paired one-way against the")
    L.append("# expanded NCU-label ledger (cross-stamped below). Raw is the SIMULATOR output,")
    L.append("# NOT the NCU label (identity-fit-smokescreen): the isotonic re-fit learns")
    L.append("# the raw->label shift. NVBit is calibration-only (CLAUDE.md Rule 1).")
    L.append("")
    L.append("[meta]")
    L.append(f"schema_version = {TABLE_SCHEMA_VERSION}")
    L.append(f"host_arch = {_toml_s(HOST_ARCH)}")
    L.append(f"captured_at_utc = {_toml_s(captured)}")
    L.append(f"expanded_ledger_path = {_toml_s(str(_EXPANDED_LEDGER.relative_to(_REPO_ROOT)))}")
    L.append(f"expanded_ledger_sha256 = {_toml_s(ledger_sha)}")
    L.append(f"geometry_source = {_toml_s(_SM86_GEOMETRY['geometry_source'])}")
    L.append(f"associativity_l2 = {_SM86_GEOMETRY['associativity_l2']}")
    L.append(f"num_l2_slices = {_SM86_GEOMETRY['num_l2_slices']}")
    L.append(f"l2_capacity_bytes = {_SM86_GEOMETRY['l2_capacity_bytes']}")
    L.append(f"scored_count = {len(scored)}")
    L.append(f"unavailable_count = {len(rows) - len(scored)}")
    L.append(f"row_count = {len(rows)}")
    L.append("")
    for r in rows:
        L.append("[[entries]]")
        L.append(f"slug = {_toml_s(r['slug'])}")
        L.append(f"predictor_bucket = {_toml_s(r['predictor_bucket'])}")
        L.append(f"scored = {str(r['scored']).lower()}")
        L.append(f"typed_reason = {_toml_s(r['typed_reason'])}")
        L.append(f"raw_l1_hit_rate = {r['raw_l1_hit_rate']!r}")
        L.append(f"raw_l2_hit_rate = {r['raw_l2_hit_rate']!r}")
        L.append(f"ncu_l1_hit_rate = {r['ncu_l1_hit_rate']!r}")
        L.append(f"ncu_l2_hit_rate = {r['ncu_l2_hit_rate']!r}")
        L.append(f"tier2_sidecar_sha256 = {_toml_s(r['tier2_sidecar_sha256'])}")
        L.append(f"sidecar_path = {_toml_s(r['sidecar_path'])}")
        L.append(f"sidecar_sha256 = {_toml_s(r['sidecar_sha256'])}")
        L.append("")
    out_path.write_text("\n".join(L).rstrip("\n") + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(scored)} scored / {len(rows)} rows)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-only", action="store_true",
                    help="skip the NVBit harvest; re-emit the table from existing sidecars")
    args = ap.parse_args()
    if not args.emit_only:
        run_raw_harvest(_REPO_ROOT)
    emit_table(_REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
