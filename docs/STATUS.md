# Current status

**Updated:** 2026-10-11  
**Branch:** `GPUSim_main`  
**Active plan:** `/root/gpu_cache_memory_modeling_proposal/PLAN.md`  
**Evidence root:** `/dev/shm/sm120_rtx_5090_work`

## Permanent alpha evidence policy

The live alpha lane rejects fixture/synthetic simulator tests and all
hashing/checking/validation/verification/admission/schema test gates. Keep only
real-trace/real-sidecar functionality probes. The acceptance corpus is SOLBench
plus modern/SOTA traces in `/root/sm120_rtx_5090_capture`, compared with
independent NCU ground truth; full replay precedes fast replay.

## Current state

The SM120/RTX 5090 alpha lane is executable. Full trace replay is the primary
implementation/performance path; the dynamic fast frontier is a complementary
diagnostic. The state-reconciliation commit contains the coherent card, trace, replay, service,
async-ledger, producer, workload, and reporting changes. The checkout is clean
relative to that commit.

The active goal is **Q16.1**: separate TMA load/store payload observations and
mbarrier phase observations in replay output. Q16.2 (a legal cluster/TMA or
bulk-async shape) and Q16.3 (semantic reconciliation) remain queued.

## Latest evidence

| Slice | Current result | Interpretation |
|---|---:|---|
| Q7–Q9 fresh route/async cohort | full 15.65% / fast 33.75% median cycle APE | fresh development evidence; full remains primary |
| P4 route reference | full 11.23% / fast 32.82% median APE | retained comparison corpus |
| Q10 placement shape | full 352,455 vs NCU 321,227 cycles (9.72% APE) | behavioral placement response |
| Q11 four-warp memory density | full 2.11% / fast 3.95% median APE | supported density slice |
| Q12/Q13 | no transferable RED term | residual retained, not hidden |
| Q14 `.ca`/`.cg` | 13.63% / 28.58% full APE | route sensitivity without physical-topology identification |
| Q15 TMA64 | NCU 6,741.87; full 5,318; fast 4,193 cycles | corrected process-start range; native completion time unknown |
| Q15 cluster2 | NCU 5,528.05; full 80; fast 4,004 cycles | ranks 0/1 observed; internal cluster/TMA timing unknown |

Artifacts for the newest rows are `q10_scheduler/summary.json`,
`q11_multiwarp/reconcile.json`, `q12_density/reconcile.json`,
`q15_runtime/final_reconciliation.json`, and `q7_live/q9_final_summary.json`.

## Boundaries

- The model is behavioral and logical; it does not identify proprietary SM120
  issue-port wiring, warp-selector policy, cache hash/geometry/MSHR policy,
  DRAM queue topology, or native TMA/mbarrier completion timestamps.
- A payload marker after a legal synchronization point is an observation of
  that program's visible result, not a hardware completion timestamp.
- NCU values remain independent comparison targets and are never replay inputs.
- Full replay precedes fast replay. No external detailed cycle-simulator
  comparison is part of this alpha lane.
- SM86 retained evidence remains a separate compatibility/calibration lane;
  its numbers are not silently transferred to RTX 5090.

## Operational entrypoints

Read `docs/PROJECT_MEMORY.md`, then the active plan, then only the source seam
for Q16.1. Use `docs/runbook_host.md` for commands and keep raw outputs under
the evidence root. Update this file and `PROJECT_MEMORY.md` after each real
milestone; do not replace implementation with a status-only report.
