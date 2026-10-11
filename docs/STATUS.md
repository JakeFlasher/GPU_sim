# Current status

**Updated:** 2026-10-11  
**Branch:** `GPUSim_main`  
**Active plan:** `/root/gpu_cache_memory_modeling_proposal/PLAN.md`  
**Evidence root:** `/dev/shm/sm120_rtx_5090_work`
**Full-workload corpus:** `/root/sm120_rtx_5090_capture`

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

Q16.1–Q16.3 are historical semantic-ledger slices. **Q17.1–Q17.3 are
complete with explicit native gaps:** the three replay seams were closed,
canonical recursive discovery covers all 327 required rows, Rust full replay
consumes typed TMA/sync sidecars without inventing native completion timing,
and the latest full-then-fast pair completes 327/327 captures and 11,529
launches in each mode. No endpoint coefficient or scored model change was
admitted. Q18.1 is complete as an independent control evidence slice and Q18.2
is complete with a no-change decision after comparing its mechanism-local
response against the complete canonical workload population. Q18.3 is complete
as a measured cost slice with no scope-changing optimization. Q19.1 is complete
as an independent dependency/scheduler control evidence slice; Q19.2 is
complete with a no-change decision against the canonical population. Q19.3 is
complete as a measured cost slice on the scheduler controls. Q20.1 is now
active for prospective family-separated accuracy/timing. The withdrawn 3,900-cycle
P10 post-hoc offset is not reused.

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
| Q17 residual closure | 302/302 full and 302/302 fast; 10,796 launches each | no residual categories; common 289-capture full-cycle delta is zero |
| Q17 corpus medians | full 42,690 / fast 90,320.5 model cycles per capture sum | model-relative baselines, not correctness claims |
| Q17 canonical replay | 327/327 full and 327/327 fast; 11,529 launches each | recursive nested paths included; no full replay failures |

The latest typed-sidecar full-then-fast pair joined
1,932 TMA rows (675 loads, 1,257 stores, 336 zero-byte/OOB rows;
11,242,496 payload bytes) and 402,208 sync rows (272,024 before, 130,184
after, 5,458 arrivals, 390,552 phase checks, 202,056 waits, and 136,062
distinct barrier addresses). All 34 sidecar-bearing captures and 64 affected
launches had zero missing joins. These fields are observations and semantic
joins only; they do not supply native TMA/mbarrier completion timestamps.

Q17.3 replay-cost measurements are retained at
`/dev/shm/rust_research/q17_3_8_replay_cost_20261011.json`. The process-local
sidecar cache is retained for repeated launch identities, but the uncontrolled
paired comparison does not support a suite-median speedup claim. Q18.1 is the
completed independent control slice; Q18.2 is the active comparison of its
aligned/offset/stride and working-set response against full real workload
families, with no physical cache topology inference.

Q18.1 captured four real CUDA control shapes touching 4 KiB, 64 KiB, 512 KiB,
and 4 MiB with NVBit, independent NCU and uninstrumented CUDA-event scopes;
each was replayed full first then fast. Evidence and raw capture paths are in
`/dev/shm/rust_research/q18_cache_service_shapes_20261011.json`. These controls
are an independent behavioral envelope, not a fixture gate or cache-topology
identification; Q18.2 compared them with all 327 canonical workload families.
The control model/NCU ratios span 0.123–0.647 and full-suite family medians
span 0.129–3.199, so no global service term is portable. The comparison is in
`/dev/shm/rust_research/q18_compare_full_suite_20261011.{json,md}`; Q18.3 now
measures replay cost on these new captures.

Q19.1 captured three real CUDA modes (long dependency chain, independent
arithmetic chains, and mixed global-load/ALU) with NVBit and independent NCU;
each was replayed full first then fast. Evidence is in
`/dev/shm/rust_research/q19_scheduler_shapes_20261011.json`. It is a
behavioral readiness/issue envelope only, not proprietary scheduler topology.

Q19.2 comparison found dependency-only underprediction, independent-arithmetic
near agreement and mixed-memory overprediction; no portable scheduler/readiness
term was admitted. Evidence is in
`/dev/shm/rust_research/q19_compare_full_suite_20261011.json`.

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
for Q17.3.3. Use `docs/runbook_host.md` for commands and keep raw outputs under
the evidence root. Update this file and `PROJECT_MEMORY.md` after each real
milestone; do not replace implementation with a status-only report.
