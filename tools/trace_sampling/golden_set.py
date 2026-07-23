"""Golden-set fidelity validation for the trace-sampling estimator + gates.

Validates the SAMPLER (not the simulator) full-vs-sampled: for each admissible
golden cell the exact full-cell additive count `R_full` is known, and the HT
estimate `R̂` from a sampled subset is scored across fractions × seeds against the
declared thresholds — accuracy (p95 |R̂−R_full|/R_full ≤ COUNT_TOL), CI coverage
(≥ CI_COVERAGE_MIN), plus the NEGATIVE controls (histogram / few-wave cells MUST
be gate-forbidden — the single most important check) and the wave-segment vs
independent-block ablation.

This module is a FAST CONTROLLED UNIT SMOKE, not the authoritative real-input harness:
it covers additive COUNTS on synthetic cells (deterministic, every golden class).
The authoritative golden-set (`golden_set_real.py`) loads
captured trace dirs, decodes counts + rates with the real L2 decoder, and enforces
the CI-coverage + sampling-induced error-budget gates vs the real NCU label.
Reference metric here is the synthetic full-cell count, never an NCU value.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from admissibility_gate import M_SUBSET, MIN_SEGMENTS, classify
from counting_pass import BlockStat, CellStats
from estimator import Segment, Stratum, estimate

# Controlled-smoke thresholds.
COUNT_TOL = 0.05          # p95 relative error on additive counts
CI_COVERAGE_MIN = 0.90    # nominal-95% intervals cover R_full in >= 90% of trials
FRACTIONS = (1 / 16, 1 / 8, 1 / 4)
N_SEEDS = 5

# This CONTROLLED smoke covers only additive counts on synthetic cells; the REAL
# golden-set (golden_set_real.py) is the authoritative real-input harness and validates
# rates + CI coverage + the sampling-induced error budget on captured trace dirs.
SMOKE_NOTES = {
    "rate_and_error_budget": "validated on real trace dirs in golden_set_real.py; "
                             "this smoke covers additive counts on synthetic cells",
}


@dataclass
class EstimatorCell:
    """A cell defined at the segment level: per-segment additive count + record
    count. `R_full` = sum of counts, `x_true` = sum of records."""
    cell_id: str
    seg_counts: list[float]
    seg_records: list[float]


def _select(n_total: int, n_take: int, seed: int) -> list[int]:
    """Deterministic segment subset: segment 0 + the last as certainty, the rest
    filled by a seeded shuffle (no per-record/every-k decimation)."""
    rng = np.random.default_rng(seed)
    chosen = {0, n_total - 1}
    rest = [i for i in range(n_total) if i not in chosen]
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= n_take:
            break
        chosen.add(i)
    return sorted(chosen)


def validate_estimator_cell(cell: EstimatorCell, *, wave_segment: bool = True) -> dict:
    """Score the HT estimator full-vs-sampled across FRACTIONS × N_SEEDS."""
    counts = np.asarray(cell.seg_counts, float)
    records = np.asarray(cell.seg_records, float)
    r_full = float(counts.sum())
    x_true = float(records.sum())
    n_total = len(counts)
    rels: list[float] = []
    covered: list[bool] = []
    statuses: list[str] = []
    for frac in FRACTIONS:
        n_take = max(MIN_SEGMENTS, round(frac * n_total))
        n_take = min(n_take, n_total)
        for seed in range(N_SEEDS):
            if wave_segment:
                ids = _select(n_total, n_take, seed)
            else:
                # Independent-block ablation: a plain seeded subset, no certainty
                # cluster structure (models per-block thinning).
                rng = np.random.default_rng(seed + 9973)
                ids = sorted(rng.choice(n_total, size=n_take, replace=False).tolist())
            segments = {
                i: Segment(id=i, stratum=0, measured={"count": float(counts[i])},
                           record_count=float(records[i]))
                for i in ids
            }
            strata = [Stratum(id=0, N_h=n_total, sampled_ids=list(ids))]
            r = estimate(segments, strata, certainty_ids=[], x_true=x_true,
                         count_targets=["count"], rate_targets=[], seed=seed)
            rels.append(abs(r.point["count"] - r_full) / r_full if r_full else 0.0)
            lo, hi = r.ci["count"]
            covered.append(lo <= r_full <= hi)
            statuses.append(r.status)
    p95 = float(np.percentile(rels, 95))
    coverage = float(np.mean(covered))
    return {
        "cell_id": cell.cell_id,
        "r_full": r_full,
        "p95_rel_error": p95,
        "accuracy_ok": p95 <= COUNT_TOL,
        "ci_coverage": coverage,
        "ci_coverage_ok": coverage >= CI_COVERAGE_MIN,
        "n_trials": len(rels),
        "converged_trials": sum(s == "converged" for s in statuses),
    }


# --- controlled golden cells ---

def _streaming_cell(cell_id: str, n_seg: int, cv: float, seed: int) -> EstimatorCell:
    """Low-variance streaming cell — the HT estimator should be very accurate."""
    rng = np.random.default_rng(seed)
    counts = rng.normal(1000.0, 1000.0 * cv, n_seg).clip(min=1.0)
    # Records correlate with counts (more work -> more records) but are NOT a
    # perfect proxy, so the ratio calibration is meaningful, not degenerate.
    records = (counts * 0.5 * rng.uniform(0.9, 1.1, n_seg)).clip(min=1.0)
    return EstimatorCell(cell_id, counts.tolist(), records.tolist())


def _gate_stats(*, n_waves: int, cooperative: bool = False, atomic_density: float = 0.0,
                atomic_top256: float = 0.0, block_records=None,
                per_sm_counts=None, per_wave=None) -> CellStats:
    recs = list(block_records) if block_records is not None else [100] * 200
    blocks_per_wave = max(1, (len(recs) + n_waves - 1) // n_waves)
    blocks = [BlockStat(
        block_id=i, sm_id=0, records=r, sector_accesses=max(0, r // 32),
        wave=i // blocks_per_wave) for i, r in enumerate(recs)]
    total = sum(recs)
    derived_wave_totals: dict[int, int] = {}
    for block in blocks:
        derived_wave_totals[block.wave] = (
            derived_wave_totals.get(block.wave, 0) + block.records)
    return CellStats(
        cell_id="g", kernel_name="k", grid=(len(recs), 1, 1), block=(32, 1, 1),
        n_blocks=len(recs), total_records=total, op_class_totals={},
        blocks=blocks, per_sm_resident={0: blocks_per_wave},
        per_sm_block_ids={0: list(range(len(recs)))}, n_waves=n_waves,
        atomic_records=int(atomic_density * total),
        atomic_top256_sector_share=atomic_top256,
        per_wave_record_totals=(per_wave if per_wave is not None
                                else derived_wave_totals),
        per_sm_block_counts=(per_sm_counts if per_sm_counts is not None
                             else {0: len(recs)}), cooperative=cooperative)


def run_controlled_smoke() -> dict:
    """Fast CONTROLLED unit smoke (not the authoritative real-input harness; see
    golden_set_real.py): exercises the real estimator + gate on synthetic cells,
    covering every golden class + the negative controls + ablation."""
    failures: list[str] = []

    # Admissible cells (low + moderate variance streaming): accuracy + CI coverage.
    admissible = []
    for cid, cv in [("streaming_uniform", 0.05), ("streaming_moderate", 0.12)]:
        res = validate_estimator_cell(_streaming_cell(cid, 256, cv, seed=17))
        admissible.append(res)
        if not res["accuracy_ok"]:
            failures.append(f"{cid}: accuracy p95={res['p95_rel_error']:.4f} > {COUNT_TOL}")
        if not res["ci_coverage_ok"]:
            failures.append(f"{cid}: CI coverage {res['ci_coverage']:.2f} < {CI_COVERAGE_MIN}")

    # NEGATIVE controls (the single most important check): each MUST be forbidden.
    negatives = {
        "histogram_atomic": _gate_stats(n_waves=100, atomic_density=0.02, atomic_top256=0.9),
        "few_wave": _gate_stats(n_waves=6),  # < 16 measured waves
        "cooperative": _gate_stats(n_waves=100, cooperative=True),
    }
    negative_results = []
    for cid, stats in negatives.items():
        g = classify(stats)
        forbidden = g.method != M_SUBSET
        negative_results.append({"cell_id": cid, "verdict": g.verdict,
                                 "method": g.method, "correctly_forbidden": forbidden})
        if not forbidden:
            failures.append(f"NEGATIVE CONTROL {cid} was ADMITTED (verdict={g.verdict}) "
                            f"— the golden set is too weak")

    # Ablation: wave-segment cluster vs independent-block on the same cell.
    abl_cell = _streaming_cell("ablation", 256, 0.12, seed=17)
    wave = validate_estimator_cell(abl_cell, wave_segment=True)
    indep = validate_estimator_cell(abl_cell, wave_segment=False)
    ablation = {
        "wave_segment_p95": wave["p95_rel_error"],
        "independent_block_p95": indep["p95_rel_error"],
        # For additive COUNTS both HT estimators are ~unbiased (co-residency only
        # affects RATES — validated once the ghost-differencing seam lands).
        "note": "counts are ~unbiased under both; co-residency effect is a hit-RATE "
                "phenomenon (hit_rate_ci typed gap)",
    }

    report = {
        "schema_version": 1,
        "gates": {"count_tol": COUNT_TOL, "ci_coverage_min": CI_COVERAGE_MIN,
                  "fractions": list(FRACTIONS), "n_seeds": N_SEEDS,
                  "min_segments": MIN_SEGMENTS},
        "admissible_cells": admissible,
        "negative_controls": negative_results,
        "ablation": ablation,
        "smoke_notes": SMOKE_NOTES,
        "failures": failures,
        "pass": not failures,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = run_controlled_smoke()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    verdict = "PASS" if report["pass"] else "FAIL"
    print(f"golden-set [{verdict}]: {len(report['admissible_cells'])} admissible, "
          f"{len(report['negative_controls'])} negative controls, "
          f"{len(report['failures'])} failures")
    for f in report["failures"]:
        print(f"  FAIL: {f}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
