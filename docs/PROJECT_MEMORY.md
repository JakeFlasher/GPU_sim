# Continuation memory

**Updated:** 2026-10-11 after repository/evidence inventory.

## Current lane

- Branch: `GPUSim_main` in `/root/GPU_simulator_rebase`.
- Active plan: `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
- Live evidence: `/dev/shm/sm120_rtx_5090_work`; full-workload corpus: `/root/sm120_rtx_5090_capture`.
- Active target: GeForce RTX 5090, `sm_120`; SM86 remains the compatibility
  baseline and is not a source of SM120 timing constants.
- Full SM120 replay is primary; the dynamic fast frontier is complementary.
- NCU/CUPTI/CUDA-event measurements are independent observations and are joined
  after replay, never used as same-workload model inputs.
- Accel-Sim and procedural approval/audit/admission machinery are outside the
  lane. Do not run tests, verification, hashing, or audit workflows.

## Permanent alpha evidence policy

- Do not add, restore, invoke, or maintain fixture-driven/synthetic simulator
  tests or small-fixture acceptance gates. Hashing, checking, validation,
  verification, admission, manifest/schema approval, and equivalent procedural
  test workflows are permanently banned in the live alpha lane.
- Keep only real-trace/real-sidecar functionality probes where needed.
  Production runtime error handling, checked arithmetic, and fail-closed input
  handling remain implementation behavior, not test gates.
- Acceptance evidence is generalization/performance on SOLBench plus modern/SOTA
  traces under `/root/sm120_rtx_5090_capture`, compared to independent NCU
  ground truth. Full replay is primary; fast replay is complementary.
- Never treat NCU as a model input or tune the same scored workload to its NCU
  target. Do not reintroduce deleted fixture-test machinery under another name.

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

## Latest residual closure and next seam

Q17.1–Q17.2 closed the retained-corpus replay residuals without fixture gates:
SM120 uses the architecture-neutral write-lookup projection until an
independent cache sweep measures a real window; explicit
`unresolved_module_global` addresses are retained without fabricated ids; and
unqualified PTX-generic `ATOM`/`RED` routes follow dynamic Global/Shared space.
The protobuf binding was regenerated from the current tracer schema.

Fresh release replay now completes **302/302** captures and **10,796/10,796**
launches in both full and fast modes, with no residual categories. The common
289-capture full-cycle delta is zero. Raw rows and medians are in
`/dev/shm/rust_research/full_workload_replay_summary_q17_20261011.json` and
the adjacent `/dev/shm/sm120_alpha_q17_*` files.

**Q17.3 is active:** select the next independent full-suite model/performance
improvement, preserving the separation between NCU comparison targets and
simulator inputs. Update this memory and `docs/STATUS.md` after the next real
evidence/code milestone.

## Evidence/research directories

- `/dev/shm/sm120_rtx_5090_work`: retained raw traces, profiles, replay outputs, and
  summaries; do not rewrite or relabel old rows.
- `/root/sm120_rtx_5090_capture`: full SOLBench/modern-kernel capture corpus and
  independent NCU profiles for alpha performance replay; treat NCU as an
  independent target, never a replay input.
- `/dev/shm/sm120_literature`, `/dev/shm/sm120_open_source_review`,
  `/dev/shm/sm120_research_literature`, `/dev/shm/sm120_research_src`, and
  `/dev/shm/sm120_research_next`: read-only literature/source study trees.
- `/dev/shm/rust_research`: Rust organization report and clean source snapshots;
  it is separate from the GPU execution plan.

Keep this file short. Detailed raw evidence belongs beside its run output, not
in continuation memory.
