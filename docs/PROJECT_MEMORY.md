# Continuation memory

**Updated:** 2026-10-11 after repository/evidence inventory.

## Current lane

- Branch: `GPUSim_main` in `/root/GPU_simulator_rebase`.
- Active plan: `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
- Live evidence: `/dev/shm/sm120_rtx_5090_work`.
- Active target: GeForce RTX 5090, `sm_120`; SM86 remains the compatibility
  baseline and is not a source of SM120 timing constants.
- Full SM120 replay is primary; the dynamic fast frontier is complementary.
- NCU/CUPTI/CUDA-event measurements are independent observations and are joined
  after replay, never used as same-workload model inputs.
- Accel-Sim and procedural approval/audit/admission machinery are outside the
  lane. Do not run tests, verification, hashing, or audit workflows.

## Current implementation state

The state-reconciliation commit records one coherent SM120 alpha slice: the RTX 5090 card
and aliases, SM120 trace-contract/address/cache-op retention, compressed sector
metadata, lane-aware fixed-register readiness, full replay service frontiers,
fast replay diagnostics, CUDA-event/replay-wall reporting, and the NVBit/NCU
producer updates needed by the retained captures. The checkout is now clean relative to that commit; unrelated SOL-ExecBench
and SM86 measurement material is preserved.

The active executable surfaces are:

- `sim-run --sm120-alpha --mode flat` (full replay);
- `sim-run --sm120-alpha --fast-frontier-only` (fast diagnostic);
- `tools/sm120_replay_bench.py` plus the feature/service/async/dependency
  helpers under `tools/`.

## Latest measured progress

The latest retained evidence is under `/dev/shm/sm120_rtx_5090_work`:

- Q7–Q9 fresh route/async cohort: full median APE 15.65%, fast 33.75%;
  P4 reference route full/fast 11.23%/32.82%; replay-speed ratios are reported
  separately.
- Q10 two-CTA/four-warp placement shape: full 352,455 versus independent NCU
  321,227 cycles (9.72% APE); fast 13.51%.
- Q11 four-warp store/ATOM/RED slice: full median APE 2.11%, fast 3.95%.
- Q12 eight-warp RED/ATOM did not isolate a transferable term; Q13 RED/control
  dependence likewise retained its residual; Q14 `.ca`/`.cg` is route-sensitive
  but not a separable physical latency arm.
- Q15 corrected process-start NCU ranges: TMA64 6,741.87 cycles and cluster2
  5,528.05 cycles. TMA payload-after-sync (8,192 bytes) and cluster ranks
  `(0,1)` were observed; native completion timestamps remain unavailable.

## Next concrete seam

**Q16.1 is active:** separate TMA load/store payload observations and
mbarrier phase observations in replay output. Q16.2 is queued for a legal
cluster/TMA or bulk-async shape; Q16.3 will reconcile the semantic ledger
without claiming native completion timing. Update this memory and
`docs/STATUS.md` after the next real evidence/code milestone.

## Evidence/research directories

- `/dev/shm/sm120_rtx_5090_work`: raw traces, profiles, replay outputs, and
  summaries; do not rewrite or relabel old rows.
- `/dev/shm/sm120_literature`, `/dev/shm/sm120_open_source_review`,
  `/dev/shm/sm120_research_literature`, `/dev/shm/sm120_research_src`, and
  `/dev/shm/sm120_research_next`: read-only literature/source study trees.
- `/dev/shm/rust_research`: Rust organization report and clean source snapshots;
  it is separate from the GPU execution plan.

Keep this file short. Detailed raw evidence belongs beside its run output, not
in continuation memory.
