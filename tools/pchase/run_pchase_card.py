#!/usr/bin/env python3
"""Historical p-chase v1 producer (retired from production use).

The v1 counter-free/clock64 producer is retained for archaeology and analysis
helpers only. Its collection and publication entry points refuse before reading
inputs, launching subprocesses, building binaries, or writing outputs. A future
schema-v2 producer must provide independently reviewed, target-free provenance,
locked-clock protocol metadata, and an external immutable publication root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from tools.producer_retirement import (
    report_legacy_producer_retirement,
    refuse_legacy_producer,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

SOURCE = HERE / "pchase_kernels.cu"
BINARY = HERE / "build" / "pchase"
DEFAULT_CARD_OUT = "data/pchase/sm_86_static_card.json"

KB = 1024
MB = 1024 * KB

ANALYSIS_CONFIG = {
    "step_min_jump": 0.20,        # cumulative >= 20% latency increase
    "step_dominance": 2.0,        # >= 2x the largest outside jump
    "step_member_floor": 0.10,    # adjacent-interval merge floor
    "latency_points": {
        "l1_hit_latency_cycles": {"footprint": 8 * KB, "iters": 100_000},
        "l2_hit_latency_cycles": {"footprint": 1 * MB, "iters": 100_000},
        "dram_latency_cycles": {"footprint": 128 * MB, "iters": 100_000},
    },
    "shared_latency_point": {"footprint": 16 * KB, "iters": 100_000},
    "l1_capacity_sweep_kb": [16, 32, 48, 64, 80, 96, 112, 128, 144,
                             160, 192, 224, 256],
    "l2_capacity_sweep_mb": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
    "carveout_pcts": [0, 25, 50, 75, 100],
    "carveout_sweep_kb": [32, 48, 64, 80, 96, 112, 128, 144, 160, 192],
    "conflict_strides_log2": list(range(8, 18)),   # 256B .. 128KB
    "conflict_n_lines": [4, 8, 12, 16, 24, 32],
    "mshr_ks": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    "replacement_capacity_fractions": [0.75, 1.25],
    "n_repeats": 5,
    "sweep_repeats": 3,
    "sweep_iters": 50_000,
    "nvml_active_clock_floor_mhz": 400,
    "nvml_reject_reason_mask": 0xCC,  # HW slowdown/thermal/brake + SW power cap
    "sw_thermal_bit": 0x20,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_binary(nvcc: str = "nvcc", arch: str = "sm_86") -> None:
    refuse_legacy_producer(
        "tools/pchase/run_pchase_card.py:build_binary",
        successor_requirement="schema-v2 external immutable evidence",
    )


def nvml_sample() -> dict:
    refuse_legacy_producer(
        "tools/pchase/run_pchase_card.py:nvml_sample",
        successor_requirement="schema-v2 external immutable evidence",
    )


def covariate_verdict(sample: dict, cfg: dict) -> dict:
    """Accept/annotate one post-run covariate sample per the calibrated
    NVML policy (SW thermal is boost equilibrium, not a fault)."""
    if sample.get("unavailable"):
        return {"accepted": False, "reason": "nvml_unavailable"}
    reasons = sample.get("throttle_reasons", 0)
    if reasons & cfg["nvml_reject_reason_mask"]:
        return {"accepted": False,
                "reason": f"hw_fault_or_power_cap:0x{reasons:x}"}
    clock = sample.get("sm_clock_mhz", 0)
    floor = cfg["nvml_active_clock_floor_mhz"]
    if clock < floor:
        # Idle is rejected by the active-clock floor (never by the
        # throttle bitmask): idle telemetry must not enter the
        # accepted covariate medians.
        return {"accepted": False,
                "reason": f"idle_clock_below_floor:{clock}<{floor}"}
    return {
        "accepted": True,
        "sw_thermal_active": bool(reasons & cfg["sw_thermal_bit"]),
    }


def run_probe(args: list[str], cfg: dict) -> list[dict]:
    refuse_legacy_producer(
        "tools/pchase/run_pchase_card.py:run_probe",
        successor_requirement="schema-v2 external immutable evidence",
    )


def median_dispersion(values: list[float]) -> tuple[float, float]:
    med = statistics.median(values)
    disp = statistics.pstdev(values) if len(values) > 1 else 0.0
    return med, disp


def detect_step(curve: list[tuple[float, float]], cfg: dict) -> dict:
    """Latency-step detection with transition-run merging and an explicit
    confidence rule.

    A capacity transition often SMEARS over adjacent footprints (at
    footprint == capacity the permuted ring already conflict-misses),
    so the detector merges the maximal run of CONSECUTIVE intervals
    with relative jump >= ``step_member_floor`` containing the largest
    jump, and reports a BRACKET: ``last_base_x`` (largest probed
    footprint fully at base latency — the conservative value) and
    ``first_post_x`` (smallest probed footprint at the post-step
    level). Resolved only when the merged run's cumulative jump is
    >= ``step_min_jump`` AND >= ``step_dominance`` x the largest jump
    OUTSIDE the run. Nothing between the bracket ends is ever
    fabricated."""
    if len(curve) < 3:
        return {"resolved": False, "reason": "curve_too_short"}
    member_floor = cfg.get("step_member_floor", 0.10)
    rels = []
    for i in range(1, len(curve)):
        prev, cur = curve[i - 1][1], curve[i][1]
        rels.append((cur - prev) / prev if prev > 0 else 0.0)
    best_i = max(range(len(rels)), key=lambda i: rels[i])
    if rels[best_i] <= 0:
        return {"resolved": False, "reason": "no_positive_jump"}
    lo = best_i
    while lo > 0 and rels[lo - 1] >= member_floor:
        lo -= 1
    hi = best_i
    while hi + 1 < len(rels) and rels[hi + 1] >= member_floor:
        hi += 1
    base_level = curve[lo][1]
    post_level = curve[hi + 1][1]
    cumulative = (post_level - base_level) / base_level \
        if base_level > 0 else 0.0
    outside = [rels[i] for i in range(len(rels))
               if i < lo or i > hi]
    runner = max(outside) if outside else 0.0
    resolved = (cumulative >= cfg["step_min_jump"]
                and (runner <= 0
                     or cumulative >= cfg["step_dominance"] * runner))
    return {
        "resolved": resolved,
        "last_base_x": curve[lo][0],
        "first_post_x": curve[hi + 1][0],
        "cumulative_jump_relative": round(cumulative, 4),
        "largest_outside_jump_relative": round(runner, 4),
        "transition_intervals": hi - lo + 1,
        "rule": (f"cumulative >= {cfg['step_min_jump']} and >= "
                 f"{cfg['step_dominance']}x largest outside jump; "
                 f"run members >= {member_floor}"),
    }


def summarize_covariates(records: list[dict]) -> dict:
    samples = [r["covariates"] for r in records
               if r["covariates"].get("accepted")]
    if not samples:
        return {"n_accepted": 0}
    return {
        "n_accepted": len(samples),
        "n_rejected": len(records) - len(samples),
        "sm_clock_mhz_median": statistics.median(
            s["sm_clock_mhz"] for s in samples),
        "temperature_c_median": statistics.median(
            s["temperature_c"] for s in samples),
        "power_w_median": statistics.median(
            s["power_w"] for s in samples),
        "sw_thermal_active_fraction": sum(
            1 for s in samples if s.get("sw_thermal_active")
        ) / len(samples),
    }


def value_property(values: list[float], unit: str, method: str,
                   records: list[dict], extra: dict | None = None) -> dict:
    med, disp = median_dispersion(values)
    out = {
        "kind": "value",
        "value": round(med, 4),
        "unit": unit,
        "dispersion": round(disp, 4),
        "n_repeats": len(values),
        "method": method,
        "covariates": summarize_covariates(records),
    }
    if extra:
        out.update(extra)
    return out


def unresolved_property(method: str, evidence, detail: str) -> dict:
    return {
        "kind": "typed",
        "typed_reason": "geometry_unresolved",
        "method": method,
        "detail": detail,
        "evidence": evidence,
    }


def capacity_property(points: list[tuple[int, list[dict]]],
                      unit_x: str, method: str, cfg: dict) -> dict:
    """Capacity from a latency-vs-footprint sweep via step detection."""
    curve = []
    curve_out = []
    all_records: list[dict] = []
    for x, records in points:
        med, disp = median_dispersion(
            [r["cycles_per_load"] for r in records])
        curve.append((float(x), med))
        curve_out.append({"x": x, "median_cycles_per_load": round(med, 4),
                          "dispersion": round(disp, 4),
                          "n_repeats": len(records)})
        all_records.extend(records)
    step = detect_step(curve, cfg)
    if not step.get("resolved"):
        return unresolved_property(
            method, {"curve": curve_out, "step_detection": step},
            "no dominant latency step under the confidence rule")
    return {
        "kind": "value",
        "value": step["last_base_x"],
        "value_semantics": (
            "largest probed footprint fully at base latency (the "
            "conservative bracket end; the true geometry lies within "
            "[value, bracket_upper])"
        ),
        "bracket_upper": step["first_post_x"],
        "unit": unit_x,
        "dispersion": 0.0,
        "n_repeats": min(p["n_repeats"] for p in curve_out),
        "method": method,
        "step_detection": step,
        "evidence": {"curve": curve_out},
        "covariates": summarize_covariates(all_records),
    }


def main(argv: list[str] | None = None) -> int:
    return report_legacy_producer_retirement(
        "tools/pchase/run_pchase_card.py",
        successor_requirement="schema-v2 external immutable evidence",
    )

if __name__ == "__main__":
    raise SystemExit(main())
