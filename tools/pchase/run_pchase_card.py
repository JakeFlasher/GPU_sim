#!/usr/bin/env python3
"""P-chase static-card orchestrator (counter-free by construction).

Runs the clock64 dependent-chase probe suite on the live device and
emits the committed per-arch static card. NO NCU, NO profiler, NO
clock locking: cycles come from clock64() inside the kernels, and the
only telemetry is read-only NVML queries (via nvidia-smi) recorded as
covariates around every invocation.

Honesty rules:

- Every property carries either {value, unit, dispersion, n_repeats,
  method, covariates} or a typed ``geometry_unresolved`` record with
  the raw evidence curve — a property the probes cannot pin under
  DVFS noise is typed, never guessed.
- Latency-step detection uses an explicit confidence rule: the largest
  relative jump in the median curve must be >= ``step_min_jump`` AND
  >= ``step_dominance`` times the second-largest jump; anything
  weaker is ``geometry_unresolved``.
- NVML covariate acceptance follows the calibrated policy: HW
  slowdown / HW thermal / HW power brake / SW power cap reject a
  sample; SW thermal slowdown is the normal Ampere boost equilibrium
  and is RECORDED (sw_thermal_active fraction), never rejected; idle
  is rejected by an active-clock floor, not by the throttle mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

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
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [nvcc, "-O2", f"-arch={arch}", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed: {proc.stderr[-400:]}")


def nvml_sample() -> dict:
    """One read-only telemetry sample (never mutates driver state)."""
    proc = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
         "clocks_event_reasons.active",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return {"unavailable": True}
    parts = [s.strip() for s in proc.stdout.strip().split(",")]
    try:
        return {
            "sm_clock_mhz": float(parts[0]),
            "mem_clock_mhz": float(parts[1]),
            "temperature_c": float(parts[2]),
            "power_w": float(parts[3]),
            "throttle_reasons": int(parts[4], 16),
        }
    except (ValueError, IndexError):
        return {"unavailable": True}


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
    """One binary invocation -> per-rep records, each annotated with a
    post-run covariate sample (the GPU was active moments before)."""
    proc = subprocess.run(
        [str(BINARY), *args], capture_output=True, text=True,
        timeout=600, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pchase {' '.join(args)} rc={proc.returncode}: "
            f"{proc.stderr[-300:]}")
    sample = nvml_sample()
    verdict = covariate_verdict(sample, cfg)
    records = []
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        rec["covariates"] = {**sample, **verdict}
        records.append(rec)
    return records


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--out", default=DEFAULT_CARD_OUT)
    parser.add_argument("--quick", action="store_true",
                        help="reduced repeats for smoke runs")
    args = parser.parse_args(argv)

    cfg = dict(ANALYSIS_CONFIG)
    if args.quick:
        cfg["n_repeats"] = 2
        cfg["sweep_repeats"] = 1

    build_binary(nvcc=args.nvcc, arch=args.arch)
    properties: dict[str, dict] = {}

    # --- point latencies ---
    for name, spec in cfg["latency_points"].items():
        records = run_probe(
            ["chase", str(spec["footprint"]), str(spec["iters"]),
             str(cfg["n_repeats"])], cfg)
        properties[name] = value_property(
            [r["cycles_per_load"] for r in records],
            "cycles_per_dependent_load",
            f"single-thread permuted-ring chase, footprint "
            f"{spec['footprint']} B, {spec['iters']} dependent loads, "
            "in-kernel warmup pass, clock64 timing (loop overhead "
            "included; cycles are boost-invariant)",
            records, {"footprint_bytes": spec["footprint"]})
        print(f"[pchase] {name}: "
              f"{properties[name].get('value')}", flush=True)

    spec = cfg["shared_latency_point"]
    records = run_probe(
        ["shared_chase", str(spec["footprint"]), str(spec["iters"]),
         str(cfg["n_repeats"])], cfg)
    properties["shared_latency_cycles"] = value_property(
        [r["cycles_per_load"] for r in records],
        "cycles_per_dependent_load",
        "single-thread shared-memory index chase, clock64 timing",
        records, {"footprint_bytes": spec["footprint"]})
    print(f"[pchase] shared_latency_cycles: "
          f"{properties['shared_latency_cycles'].get('value')}",
          flush=True)

    # --- L1 capacity sweep (default carveout) ---
    l1_points = []
    for kb in cfg["l1_capacity_sweep_kb"]:
        records = run_probe(
            ["chase", str(kb * KB), str(cfg["sweep_iters"]),
             str(cfg["sweep_repeats"])], cfg)
        l1_points.append((kb * KB, records))
    properties["l1_capacity_bytes"] = capacity_property(
        l1_points, "bytes",
        "latency-vs-footprint step (default carveout)", cfg)
    print(f"[pchase] l1_capacity_bytes: "
          f"{properties['l1_capacity_bytes'].get('value', 'typed')}",
          flush=True)

    # --- L2 capacity sweep ---
    l2_points = []
    for mb in cfg["l2_capacity_sweep_mb"]:
        records = run_probe(
            ["chase", str(mb * MB), str(cfg["sweep_iters"]),
             str(cfg["sweep_repeats"])], cfg)
        l2_points.append((mb * MB, records))
    properties["l2_capacity_bytes"] = capacity_property(
        l2_points, "bytes", "latency-vs-footprint step", cfg)
    print(f"[pchase] l2_capacity_bytes: "
          f"{properties['l2_capacity_bytes'].get('value', 'typed')}",
          flush=True)

    # --- carveout capacity sweep: one value-or-typed property per
    # carveout percentage (the card contract admits ONLY those kinds) ---
    for pct in cfg["carveout_pcts"]:
        points = []
        for kb in cfg["carveout_sweep_kb"]:
            records = run_probe(
                ["chase", str(kb * KB), str(cfg["sweep_iters"]),
                 str(cfg["sweep_repeats"]), str(pct)], cfg)
            points.append((kb * KB, records))
        prop = capacity_property(
            points, "bytes",
            f"latency-vs-footprint step at preferred shared-memory "
            f"carveout {pct}%", cfg)
        properties[f"l1_capacity_carveout_{pct}_pct_bytes"] = prop
        print(f"[pchase] carveout {pct}%: "
              f"{prop.get('value', 'typed')}", flush=True)

    # --- set-conflict grid (sets/ways/mapping evidence) ---
    conflict_grid = []
    base_l1 = properties.get("l1_hit_latency_cycles", {}).get("value")
    for log2s in cfg["conflict_strides_log2"]:
        stride = 1 << log2s
        for n_lines in cfg["conflict_n_lines"]:
            if n_lines * stride > 64 * MB:
                continue
            records = run_probe(
                ["conflict", str(n_lines), str(stride),
                 str(30_000), str(cfg["sweep_repeats"])], cfg)
            med, disp = median_dispersion(
                [r["cycles_per_load"] for r in records])
            conflict_grid.append({
                "stride_bytes": stride, "n_lines": n_lines,
                "median_cycles_per_load": round(med, 4),
                "dispersion": round(disp, 4),
            })
    # Ways evidence: at each conflicting stride, the smallest n_lines
    # whose latency leaves the L1-hit band marks the associativity.
    ways_estimates = []
    if base_l1:
        for log2s in cfg["conflict_strides_log2"]:
            stride = 1 << log2s
            rows = sorted(
                (g for g in conflict_grid
                 if g["stride_bytes"] == stride),
                key=lambda g: g["n_lines"])
            for row in rows:
                if row["median_cycles_per_load"] > 1.5 * base_l1:
                    ways_estimates.append(
                        {"stride_bytes": stride,
                         "first_conflicting_n_lines": row["n_lines"]})
                    break
    distinct = {e["first_conflicting_n_lines"] for e in ways_estimates}
    if len(distinct) == 1 and ways_estimates:
        n = distinct.pop()
        properties["l1_associativity_ways"] = {
            "kind": "value",
            "value": n,
            "unit": "ways",
            "dispersion": 0.0,
            "n_repeats": cfg["sweep_repeats"],
            "method": "smallest conflicting line count across "
                      "power-of-two strides (1.5x hit-latency band)",
            "evidence": {"ways_estimates": ways_estimates},
            "covariates": {"see": "conflict grid runs"},
        }
    else:
        properties["l1_associativity_ways"] = unresolved_property(
            "set-conflict grid",
            {"grid": conflict_grid, "ways_estimates": ways_estimates},
            "conflicting line counts disagree across strides (or no "
            "stride conflicts inside the probed range)")
    supplemental_evidence = {
        "l1_set_mapping_grid": {
            "method": "latency over (stride, n_lines) grid",
            "grid": conflict_grid,
            "note": (
                "Raw conflict-grid evidence supporting the "
                "l1_associativity_ways property; not a card property "
                "itself (the property contract is strictly "
                "value-or-typed)."
            ),
        },
    }
    print(f"[pchase] l1_associativity_ways: "
          f"{properties['l1_associativity_ways'].get('value', 'typed')}",
          flush=True)

    # --- MSHR evidence ---
    mshr_curve = []
    for k in cfg["mshr_ks"]:
        records = run_probe(
            ["mshr", str(k), str(20_000), str(cfg["sweep_repeats"])],
            cfg)
        med, disp = median_dispersion(
            [r["cycles_per_load"] for r in records])
        mshr_curve.append({"k": k,
                           "median_cycles_per_batch": round(med, 4),
                           "per_load": round(med / k, 4),
                           "dispersion": round(disp, 4)})
    # Knee rule (sustained saturation): the first k after which EVERY
    # subsequent step improves per-load throughput by < 10%. A single
    # sub-threshold wiggle followed by renewed improvement is DVFS
    # noise, not a knee — the first-wiggle rule flipped the outcome
    # between identical runs, which is exactly the instability the
    # typed fallback exists for.
    improvements = []
    for i in range(1, len(mshr_curve)):
        prev, cur = mshr_curve[i - 1], mshr_curve[i]
        improvements.append(
            (prev["per_load"] - cur["per_load"]) / prev["per_load"]
            if prev["per_load"] > 0 else 0.0)
    knee = None
    for i in range(len(improvements)):
        if all(imp < 0.10 for imp in improvements[i:]):
            knee = mshr_curve[i]["k"]
            break
    saturated = knee is not None and knee < cfg["mshr_ks"][-1]
    if saturated:
        properties["mshr_outstanding_misses"] = {
            "kind": "value",
            "value": knee,
            "unit": "outstanding_misses",
            "dispersion": 0.0,
            "n_repeats": cfg["sweep_repeats"],
            "method": "per-load throughput knee over k independent "
                      "DRAM chase chains (single thread)",
            "evidence": {"curve": mshr_curve},
            "covariates": {"see": "mshr runs"},
        }
    else:
        properties["mshr_outstanding_misses"] = unresolved_property(
            "k independent DRAM chase chains (single thread)",
            {"curve": mshr_curve},
            "per-load throughput keeps improving through the probed "
            "k range — the single-thread probe bounds MSHRs from "
            "below but finds no saturation knee")
    print(f"[pchase] mshr: "
          f"{properties['mshr_outstanding_misses'].get('value', 'typed')}",
          flush=True)

    # --- replacement-policy evidence ---
    l1_cap = properties["l1_capacity_bytes"].get("value")
    if isinstance(l1_cap, (int, float)):
        repl_points = {}
        for frac in cfg["replacement_capacity_fractions"]:
            fp = int(l1_cap * frac) // (128) * 128
            records = run_probe(
                ["chase", str(fp), str(cfg["sweep_iters"]),
                 str(cfg["sweep_repeats"])], cfg)
            med, _ = median_dispersion(
                [r["cycles_per_load"] for r in records])
            repl_points[str(frac)] = round(med, 4)
        hit = properties["l1_hit_latency_cycles"]["value"]
        miss = properties["l2_hit_latency_cycles"]["value"]
        over = repl_points.get("1.25")
        hit_fraction_over = (
            (miss - over) / (miss - hit) if miss > hit else None)
        properties["l1_replacement_policy"] = unresolved_property(
            "cyclic re-reference at 0.75x and 1.25x capacity",
            {"median_cycles_per_load": repl_points,
             "estimated_hit_fraction_at_1.25x": (
                 round(hit_fraction_over, 4)
                 if hit_fraction_over is not None else None)},
            "policy identification from a single cyclic pattern is "
            "not decisive; the hit-fraction evidence bounds "
            "LRU-like vs random-like behavior without pinning it")
    else:
        properties["l1_replacement_policy"] = unresolved_property(
            "cyclic re-reference probes",
            None, "L1 capacity itself unresolved")

    # --- card ---
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30)
    device, driver = ("unknown", "unknown")
    if smi.returncode == 0 and smi.stdout.strip():
        device, driver = [s.strip() for s in
                          smi.stdout.strip().split(",", 1)]
    n_typed = sum(1 for p in properties.values()
                  if p.get("kind") == "typed")
    card = {
        "schema_version": 1,
        "artifact_kind": "pchase_static_card",
        "arch": args.arch,
        "device": device,
        "driver_version": driver,
        "build": {
            "nvcc_arch": args.arch,
            "source_sha256": _sha256(SOURCE),
            "binary_sha256": _sha256(BINARY),
        },
        "analysis_config": cfg,
        "counter_free_note": (
            "All values derive from clock64() inside the probe kernels "
            "plus read-only NVML covariates; no profiler, no hardware "
            "counters, no clock locking. The probe tree lives outside "
            "the local-label quarantine, so the repo-wide static guard "
            "enforces the counter-free property by construction."
        ),
        "nvml_policy": {
            "rejected_reason_mask": hex(cfg["nvml_reject_reason_mask"]),
            "sw_thermal_treatment": (
                "recorded as boost-equilibrium covariate, never "
                "rejected"),
            "active_clock_floor_mhz": cfg["nvml_active_clock_floor_mhz"],
        },
        "properties": properties,
        "supplemental_evidence": supplemental_evidence,
        "summary": {
            "n_properties": len(properties),
            "n_typed_geometry_unresolved": n_typed,
        },
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"[pchase] wrote {args.out}: {len(properties)} properties, "
          f"{n_typed} typed geometry_unresolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
