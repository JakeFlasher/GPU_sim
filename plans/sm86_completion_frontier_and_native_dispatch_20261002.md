# SM86 completion and accuracy program

Revised 2026-10-03 on `GPUSim_main`. This is a technical milestone plan, not
an approval workflow. No `/goals`, reviewers, receipts, or execution gates.

## Current focused work (2026-10-03 continuation)

1. **C7-S15 complete — warm/cold multi-launch correctness and capture.** Keep all
   three launches, use disjoint first/repeat output buffers so the second copy
   cannot hide a first-pass error, and check the eviction output and both sinks.
   No host readback between launches. Profile with application replay and
   cache-control=none; compare stream-L2 replay separately from launch-reset
   replay, with every endpoint and simulator runtime recorded before NCU.
   The fast analyzer's single-launch limitation is not a detailed replay failure.
   [NCU cache control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)
2. **C7-S29 complete — full-shape DRAM streaming dot-product slice.** The
   timing-shape binary was hardened against malformed/overflowing arguments,
   allocation failure, launch errors, and unbounded FP32 reference error.
   Native validation, Compute Sanitizer, fresh NVBit tracing, detailed replay,
   and five independent application-replay NCU captures use the same
   `n=131072, passes=4, blocks=128, threads=256` launch. The active artifact is
   `docs/sm86_c7_simbench_dram_streaming_dot_product_direct_live_v1.json`;
   the former tiny `trace_00` run is only a smoke check.
   [Ampere shared memory](https://docs.nvidia.com/cuda/ampere-tuning-guide/#unified-shared-memory-l1-texture-cache)
3. **C7-S33X active.** Nine probe families now have live
   evidence: 68 issue-port cells, 36 async-barrier cells, 30 store-ack arms,
   102 memory-retirement cells, 33 barrier-pipeline cells, 100 l1tex-admission
   cells, one l2-queue warmup/target cell, one l1tex-arrival store cell, and
   one dependent-SFU cell. C7-S33S repaired the L2 probe's
   allocation/checksum/timestamp validation; C7-S33T hardened the L1TEX
   driver and removed its stale trace-twin interface; C7-S33U retained a real
   SFU lower-bound falsifier; C7-S33V unified the result-queue binaries and
   captured collision; C7-S33W retained the separated-arm falsifier. C7-S33X
   now tests one held-out width pair while preserving the eight system-fence
   crosswalk exclusions and open completion tails; no timing/card parameter is
   selected from scored NCU targets.

## Outcome

A useful native GPU simulator must predict kernel performance, conserve the
trace's work, and validate on independent NCU measurements. Finishing a run,
printing a finite number, or covering a median with an arbitrarily wide
interval does not finish this program. Simulator runtime is recorded alone;
traditional simulators are not executed and no hardware/compression speedup
metric is introduced.

Three distinct questions must not be conflated:

1. **Model completion:** are modeled responses/acks/copy groups serviced and
   consumed exactly once, and are all unknown obligations retained?
2. **Hardware prediction:** how accurately does the model predict NCU cycles
   on unobserved workloads? A statistical prediction interval may be finite
   while the semantic completion upper remains unknown. Its probability,
   fitting population, width, and applicability must be explicit. It is not a
   worst-case silicon bound and must not clear semantic tail flags.
3. **Selective replay:** is a region transition equivalent to detailed native
   replay for future-relevant model state? This needs an independent detailed
   model differential, not a token exposing proprietary hardware cache state.
   Dispatch is not a prerequisite for accuracy of detailed replay.

## Corrections to the 2026-10-02 claims

- `close_explicit_hierarchy_tails` cleared all SM tail flags when one writeback
  timestamp existed or topology counts were zero. Neither implies complete
  ownership, ordering, initial state, or native completion. Remove this
  heuristic; keep the existing queue services and their measured model events.
- C1 async used a 64-group trace and a 16,384-group NCU executable. The apparent
  lower consistency is not a valid matched comparison. Profile the **same
  bounded executable** with the same launch arguments and fresh control data.
- C1 message passing used managed memory and spin-dependent execution. Its
  large timing dispersion cannot be attributed solely to coherence. Use
  resident device allocations and release/acquire semantics; detect divergent
  dynamic work across NVBit/NCU instead of declaring it equivalent.
- The C3 1,600-cycle padding was hand-chosen, not reproducibly fitted. Store
  pattern variants in one kernel are not family-held-out embedding workloads.
  Retain the numerical files as historical attempts but withdraw their finite
  upper/held-out validation claims. Do not enlarge the padding after a miss.
- A8 used old static controls with a freshly rebuilt binary. Rebuild/capture
  alone is insufficient; controls must be freshly extracted from that binary.
- The previous tail-witness report reads simulator aggregates and hardcodes
  unknown domains. It is not an independent extraction pass or evidence of a
  native witness. C0/C1/C2/C3 completion claims are reopened where applicable.

## Research basis

- NCU timestamps bracket grid execution with front-end wait-for-idle and
  semaphores. Software-patched counter passes do not supply workload duration.
  Stable replay/cache/clock settings and deterministic launch matching matter.
  [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility)
- PTX cache modifiers are performance hints, not ordering operations. Global
  data coherence at L2 does not imply coherent private L1 caches or mandatory
  eviction of every dirty L2 line to DRAM on kernel exit. A generic directory
  experiment is not evidence of NVIDIA's actual coherence protocol.
  [PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
- Fences establish scoped ordering, not a public cycle-count upper bound.
  `cp.async.wait_group` closes the relevant copy groups, not unrelated stores.
  [CUDA fences](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-fence-functions)
- Statistical intervals require separate model-fit and uncertainty-calibration
  samples. Marginal coverage under exchangeability is not a worst-case bound
  or a guarantee under workload-family shift. Repeats of one workload are not
  independent new workload-calibration examples.
  [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)

## Milestones — one active implementation phase

| Milestone | Concrete outcome and real completion bar | State |
|---|---|---|
| C0-R | Remove unsound flag/upper inference; rerun identical real store trace and preserve service events while upper stays open; correct prior claims | complete |
| C1-R | Matched resident-memory/async/fence probes: same binary/shape/launch, fresh control sidecar, full trace counts and NCU launch/count comparison; retain mismatches | active — async source-token repair landed; fence/visibility families remain |
| C2 | Independent **trace** extraction of stores, scopes, masks, allocations, async groups and synchronization. Separate observed facts, derived partial orders and unknowns; no invented global order | complete for observed-fact layer; native witness/effect relation open |
| C3 | Implement missing native completion obligations: per-request acknowledgements, scoped waits, copy completion, correct outstanding-work drain; demonstrate live memory/fence/copy cases plus perturbation differentials | active — model-local drains pass; physical tails remain open |
| C4 | Independently fit mechanism/observer models using new calibration-only probes, then materialize predictions on untouched cases. No ad-hoc padding or post-score model selection | partial: async group + destination-layout calibration |
| C5 | Live held-out evaluation of point error, signed bias, finite statistical interval coverage/width, repeat dispersion and OOD; keep semantic tails separate | partial: two controlled async/layout holdouts |
| C6 | Real scalar native region transition with full front-end/pipeline continuation differential and pre-mutation fallback; extend only after the first real positive case | pending |
| C7 | Inventory all available workload families (portable, native probes, SOL/other rosters). Full live campaign over the declared supported population; failures remain in coverage denominator | partial: 15-row portable population plus 26 direct SimBench rows complete; five explicit tensor/vendor replay blockers cover the other SM86 rows; research probes and SOL populations remain |
| C8 | Rerun corrected workload failures, remove obsolete interfaces, synchronize supported claims, results and runbook; no unfinished mechanism called complete | pending |

C0-R/C1-R repair the previous work; they do not finish C3/C5/C6. C7 is not
limited to 15 workloads by fiat. Native/SOL populations and practical memory/
trace budgets must be explicitly inventoried before calling a campaign full.

## C1-R experiment specification

First recapture the bounded async binary on both NVBit and NCU. Use its own
static controls and select the target rather than prewarm. Then run a small
resident-memory suite with host-computed expected values, checked allocation/
launch/synchronization errors and fixed shapes:

- store-only and store+device-fence, reused and streaming addresses;
- load/store dependency, returning atomic and non-returning reduction;
- async-copy+wait+use with cache-all and bypass routes;
- deterministic single-CTA barrier and a valid inter-CTA release/acquire case.

Native, NVBit, and NCU must use the same binary/arguments. All simulator
outputs for the batch are saved before its first NCU capture. Check selected
mangled kernel, grid/block, shared memory, registers, and executed warp count;
when scheduling changes a polling loop, call it path-nondeterministic and do
not numerically score that trace against the other path. No hidden shape
shrinking, unit conversion, or profiler values in the simulator.

5 independent NCU processes per case; application replay, cache all, clock
none on the queried 1350/10501 MHz host. Explicit kernel selection, not an
unexamined global launch-skip. Kernel time is NCU time; trace/profiler process
wall time is overhead. Record actual process timeout/failure. A missing report
is not success even if the process returned zero. Native CUDA return status
alone is not correctness: check outputs and non-finite values too.

### C1-R live progress — matched negative slices

The rebuilt 64-group `async_barrier_overlap_trace` binary was used unchanged
for native, NVBit, simulator, and NCU. Fresh controls were extracted from
that binary; NCU selected launch 1 of the prewarm+target pair. Trace target
records and NCU `sm__inst_executed.sum` both equal 9,104. NCU GPC cycles are
`20937,20950,20937,20953,20946` (median 20,946), while simulator lower bound
is 84,866; lower-bound consistency is false. This is a real matched model
mismatch, not a coverage claim. See
`docs/sm86_completion_frontier_c1r_async_matched_v1.json`.

A new resident-device-memory message-passing binary removes managed-memory
migration from the experiment. Native, NVBit, simulator and five NCU runs use
the same binary and 16-iteration argument. NCU median is 4,321 cycles; the
simulator lower bound is 1,753 and remains open for writeback/coherence. See
`docs/sm86_completion_frontier_c1r_message_device_v1.json`. The model is
conservative on the lower bound for this case, but no finite upper or native
dispatch is claimed.

C1-R remains active for fence/async model repair and matched mechanism
validation; neither slice supplies a finite completion upper. A same-binary
`c00` control differential makes the async mismatch actionable: NCU median
delta `13760` cycles (`c11 - c00`) versus simulator lower-bound delta `78852`
cycles. This points to the async-group scheduling/completion model, not a
generic launch envelope. The differential is retained in
`docs/sm86_completion_frontier_c1r_async_control_differential_v1.json`.

The first C3 repair is conservative: when an SM86 trace contains async-copy
groups, `TimingReport::latent_frontiers` now widens issue/modeled/typed-tail
frontiers and refuses a prediction lower until the async completion model is
independently closed. The repaired matched async output has null predicted
lower/upper and is retained as
`docs/sm86_completion_frontier_c3_async_refusal_v1.json`. This fixes the
previous 84866-cycle false lower; it is not an accuracy result.

A matched one-warp `c00`/`c11` live differential separates this mechanism from
multi-warp destination arbitration. For the same `async_barrier_overlap_trace`
binary and `l1 0 1 1` launch, five fresh NCU medians are 7,194 cycles (c00)
and 14,668 cycles (c11), a 7,474-cycle increment; the repaired simulator is
5,995 and 17,394, an 11,399-cycle increment. Unprofiled native `clock64`
repeats give a 13,386-cycle increment. Thus the simulator is below the direct
clock differential but above the NCU GPC endpoint; neither endpoint alone
identifies the missing async mechanism. The result is retained in
`docs/sm86_completion_frontier_c3_async_endpoint_differential_warp1_v1.json`.

An independent cache-all source-only layout pair now tests the destination
projection without changing the binary, depth, source route, or launch shape.
Compact and conflict variants preserve 256 LDGSTS instructions and source
cycles, while destination wavefronts change from 2,048 to 4,096. The repaired
simulator predicts a 4,096-cycle layout delta; five fresh NCU medians differ by
4,070 cycles (26-cycle, 0.64% delta error). Absolute endpoints remain 4.91%
and 3.87% high and hardware completion tokens remain unknown. This is narrow
model-local differential evidence, not a global predictor or semantic hardware
upper. See `docs/sm86_completion_frontier_c3_async_destination_layout_pair_v1.json`.

C4/C5 now also has a declared layout holdout. Compact/conflict NCU repeats
were used as calibration for an OLS observer over trace-derived destination
wavefronts; the severe target prediction `24,476.2..24,902.2` (point
`24,689.2`) was materialized before any severe NCU capture. Five fresh severe
NCU values are `24,822,24,855,24,875,24,891,24,840` (median `24,855`), so the
interval covers the median and point APE is 0.667%. The prediction and joined
holdout are separated in
`docs/sm86_completion_frontier_c4_destination_layout_prediction_pre_ncu_v1.json`
and `docs/sm86_completion_frontier_c4_destination_layout_holdout_v1.json`.
This remains one controlled layout family; it is not a semantic completion
upper or a general hardware-accuracy result.

The first C7 population slice is complete for the declared 15 portable-LCD
workloads. The live campaign ran native timing, NVBit tracing, two simulator
replays, and five fresh NCU repetitions for every row in 389 seconds; all
15/15 completed with zero OOM/OOD/timeout/failed-stage outcomes. The raw
campaign is under `results/alpha/c7_full_live_20261003_run/` and its tracked
summary is `docs/sm86_c7_full_live_campaign_v1.json`. It is not yet the full
repository campaign: native probes, fixed-latency rosters, SOL/other suites,
and correctness-mode joins still need an explicit population inventory. All
portable replay rows retain open dirty-writeback/coherence tails and the
campaign did not request native macro dispatch.

The population inventory is now explicit in
`docs/sm86_c7_workload_population_inventory_v1.json`: 15 active portable-LCD
rows, 92 SimBench-BW rows (31 with an SM86 arch gate), 34 research-probe
directories, and 203 SOL baseline problem directories. The portable 15 and
six direct SimBench rows have completed bounded slices; the other populations
remain pending and must receive memory/shape/OOM dispositions before any
repository-wide claim.

The portable slice has now been executed live, not merely inventoried:
`docs/sm86_c7_full_live_campaign_v1.json` records 15/15 successful rows,
75 NCU captures, 15 NVBit traces, and 30 deterministic simulator replays with
zero OOM/OOD/timeout/failed-stage outcomes. It intentionally does not claim
correctness (this performance campaign uses timing shapes) or finite accuracy;
all replay tails remain explicit. C7 remains partial until the other inventoried
populations receive bounded campaign dispositions.

### C7-S1 — direct SimBench async-copy slice — complete

The second direct SM86 SimBench slice uses the rebuilt
`cp_async_shared_tile_copy` binary and the exact same arguments for host
correctness, NVBit, replay, and five fresh NCU application captures. Host
validation passed with zero mismatches after explicit launch/synchronize error
checks. The trace contains one complete block, 644 replay records, 32
`LDGSTS` copy instructions, and four committed/waited async groups. NCU
`gpc__cycles_elapsed.max` is `4247,4233,4232,4264,4229` (median `4233`);
the simulator completion lower bound is `2190`, so lower-bound consistency is
true but no point estimator or finite upper is claimed. Dirty-writeback and
coherence tails remain open, and the independent extractor observes no
hardware completion token, owner/read-from relation, or global instrumentation
order. The fast analyzer refuses all four warps for nonfull predicate
populations; native macro dispatch remains zero. The complete matched slice is
recorded in `docs/sm86_c7_simbench_cp_async_direct_live_v1.json` with its
schema. This closes only this slice; it does not close C3/C5/C6 or the C7
population campaign.

### C7-S2 — execution specification

Run one additional direct SM86 SimBench row from a different mechanism family
(prefer `global_atomic_histogram` or `global_atomic_and_return_used`) using
the same bounded pipeline and independent five-repeat NCU contract. Preserve
the current refusal rules: correctness must be checked by the workload, NCU
must use the exact traced binary/arguments, and open semantic tails or refused
dispatch remain explicit. If a candidate is blocked by OOM, unsupported SASS,
or a stale launcher, record a bounded disposition and move to the next row
without substituting a different shape silently.

### C7-S2 — direct SimBench global reduction slice — complete

`global_atomic_histogram` now has a direct SM86 timing-shape slice using one
rebuilt binary and identical `samples=1024,bins=64,seed=42` arguments for
correctness, NVBit, replay, and five fresh NCU application captures. The host
reference checks both the identity output and the complete histogram (zero
mismatches; sum `66,560`). SASS contains `RED.E.ADD.STRONG.GPU`, and the
independent extractor observes 2,080 reduction records plus one block barrier.
Replay has full one-block coverage and 42,843 records; NCU reports the same
42,843 warp instructions. NCU `gpc__cycles_elapsed.max` is
`516202,516413,516605,515743,516297` (median `516297`), while the simulator
completion lower bound is `394976`. Lower-bound consistency is true, but the
121,321-cycle gap is not a point error: reduction read-modify-write,
dirty-writeback, and coherence tails remain open, and no hardware completion
token, owner/read-from relation, or global instrumentation order is observed.
The fast analyzer refuses the one warp for a nonfull predicate population;
native macro dispatch remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_atomic_histogram_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_atomic_histogram_direct_live_v1.schema.json`.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S3 — execution specification

Run one additional direct SM86 SimBench row from a distinct scalar-memory or
tensor family (prefer `global_vectorized_128b_copy` or
`bf16_hmma_gemm_traceable`) with the same correctness, fresh NVBit, replay, and
five-repetition NCU contract. Preserve exact binary/argument identity and
record any OOM/OOD/unsupported-SASS disposition without silently shrinking
the shape.

### C7-S3 — direct SimBench vectorized-memory slice — complete

`global_vectorized_128b_copy` now has a direct SM86 timing-shape slice using
one rebuilt binary and identical `samples=1024,bins=64,seed=42` arguments for
correctness, NVBit, replay, and five fresh NCU application captures. The host
identity-output check passed with zero mismatches; the SASS contains
`LDG.E.128`/`STG.E.128`. Replay has one-block coverage and 2,977 records, and
NCU reports the same 2,977 warp instructions. NCU
`gpc__cycles_elapsed.max` is `74859,74473,74887,75352,75109` (median
`74887`); the simulator completion lower bound is `63246`, so lower-bound
consistency is true but no point estimator or finite upper is claimed. The
independent extractor observes 512 vector loads and 512 vector stores but no
hardware completion token, owner/read-from relation, or global instrumentation
order. Dirty-writeback and coherence tails remain open. The fast analyzer
refuses the one warp for a nonfull predicate population; native macro dispatch
remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_vectorized_128b_copy_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_vectorized_128b_copy_direct_live_v1.schema.json`.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S4 — execution specification

Run one additional direct SM86 SimBench row from a tensor or scalar-compute
family (prefer `bf16_hmma_gemm_traceable` or `fp16_hmma_gemm_traceable`) with
the same checked correctness, fresh NVBit, replay, and five-repetition NCU
contract. Preserve exact binary/argument identity and record any
OOM/OOD/unsupported-SASS disposition without silently shrinking the shape.

### C7-S4 — direct SimBench BF16 HMMA slice — complete

`bf16_hmma_gemm_traceable` now has a direct SM86 timing-shape slice using one
rebuilt binary and identical `m=32,n=16,k=64,seed=42` arguments for host
correctness, NVBit, replay, and five fresh NCU application captures. The host
BF16-cast FP32 reference check passed with zero mismatches and maximum absolute
error `3.58e-7`; SASS contains `HMMA.16816.F32.BF16`. Replay has full four-block
coverage and 404 records, and NCU reports the same 404 warp instructions plus
16 tensor-pipeline instructions. NCU `gpc__cycles_elapsed.max` is
`4896,5073,4923,4926,4885` (median `4923`); the simulator completion lower
bound is `3337`, so lower-bound consistency is true but no point estimator or
finite upper is claimed. The independent extractor observes 16 HMMA records
and 64 cross-SM line-topology events, but no hardware completion token,
owner/read-from relation, or global instrumentation order. Dirty-writeback and
coherence tails remain open. The fast analyzer refuses all four warps for
nonfull predicate populations; native macro dispatch remains zero. Evidence
and schema: `docs/sm86_c7_simbench_bf16_hmma_gemm_direct_live_v1.json` and
`docs/sm86_c7_simbench_bf16_hmma_gemm_direct_live_v1.schema.json`. This closes
only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S5 — execution specification

Run one additional direct SM86 SimBench row from a different memory or tensor
family (prefer `global_scalar_32b_copy` or `int8_imma_gemm_traceable`) with the
same checked correctness, fresh NVBit, replay, and five-repetition NCU
contract. Preserve exact binary/argument identity and record any OOM/OOD or
unsupported-SASS disposition without silently shrinking the shape.

### C7-S5 — direct SimBench scalar-width slice — complete

`global_scalar_32b_copy` now has a direct SM86 timing-shape slice using one
rebuilt binary and identical `samples=1024,bins=64,seed=42` arguments for host
correctness, NVBit, replay, and five fresh NCU application captures. The host
identity-output check passed with zero mismatches; SASS contains scalar
`LDG.E`/`STG.E` and no `.128` form. Replay has one-block coverage and 11,809
records, and NCU reports the same 11,809 warp instructions. NCU
`gpc__cycles_elapsed.max` is `249180,249114,248894,249782,249151` (median
`249151`); the simulator completion lower bound is `245262`, so lower-bound
consistency is true with a 3,889-cycle residual gap, but no point estimator or
finite upper is claimed. The independent extractor observes 2,048 scalar
loads and 2,048 scalar stores but no hardware completion token, owner/read-from
relation, or global instrumentation order. Dirty-writeback and coherence tails
remain open. The fast analyzer refuses the one warp for a nonfull predicate
population; native macro dispatch remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_scalar_32b_copy_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_scalar_32b_copy_direct_live_v1.schema.json`.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S6 — execution specification

Repair `global_atomic_return_used` before capturing it: its per-lane ordinary
stores currently overwrite shared histogram bins, and its host only prints
an unrelated identity copy. Store returned-value sums in disjoint per-lane
64-bit outputs; independently check the histogram and the order-independent
sum of returned old values. Keep the registered `trace_00` shape unchanged,
verify SASS still emits ATOM/ATOMG, then run native, NVBit, replay and five NCU
processes. Check non-power-of-two/tail shapes and invalid inputs. Do not tune
the simulator from these scored values. Related MAX/AND returning variants
have the same sink defect and are next after this first real repaired slice.
Atomicity is not synchronization for the old ordinary histogram stores:
[NVIDIA legacy atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#legacy-atomic-functions).

### C7-S6 — repaired returning-atomic live slice — complete

`global_atomic_return_used` was repaired before capture. The old host driver
only checked an identity copy while the kernel overwrote histogram bins with
ordinary per-lane stores; that was not a valid atomic-return workload. The
repaired kernel writes one disjoint 64-bit returned-old-value sum per lane and
the host checks the identity output, every histogram bin, and the
order-independent sum `count*(count-1)/2`. The trace shape (`samples=1024,
bins=64,seed=42`) and three additional shapes (`32x8`, `37x13`, `1x1`) pass.
Fresh SASS contains `ATOMG.E.ADD.STRONG.GPU`. Replay has one-block coverage
and 47,139 records; NCU reports the same 47,139 warp instructions. NCU
`gpc__cycles_elapsed.max` is `1059805,1059871,1060145,1060504,1061112`
(median `1060145`); the simulator completion lower bound is `917785`, so
lower-bound consistency is true but no point estimator or finite upper is
claimed. The independent extractor observes 2,080 ATOMG records, 2,049
stores, and one block barrier, but no hardware completion token, owner/read-from
relation, or global instrumentation order. Dirty-writeback and coherence tails
remain open. The fast analyzer refuses the one warp for a nonfull predicate
population; native macro dispatch remains zero. Evidence and schema:
`docs/sm86_c7_simbench_global_atomic_return_used_direct_live_v1.json` and
`docs/sm86_c7_simbench_global_atomic_return_used_direct_live_v1.schema.json`.
This closes only this repaired slice; C3/C5/C6 and the wider C7 campaign
remain open.

### C7-S7 — execution specification

Run one additional direct SM86 SimBench row from a different atomic or tensor
family (prefer `global_atomic_and_return_used` or
`int8_imma_gemm_traceable`). Apply the same rule: repair correctness and
resource safety before profiling, preserve exact binary/arguments, and retain
any OOM/OOD/unsupported-SASS disposition without silent shape substitution.

### C7-S7 — INT8 IMMA blocked disposition — complete

The exact rebuilt int8_imma_gemm_traceable binary passes the host quantized
reference check, emits a fresh NVBit trace with 432 records and eight IMMA
instructions, and completes five same-binary NCU captures:
4526,4624,4545,4542,4567 cycles (median 4545). The independent extractor
observes 40 cross-SM line-topology events but no hardware completion token or
global instrumentation order. Replay is explicitly blocked before any timing
frontier because sim-run reports that IMMA.16832.S8.S8 has no SM86
execution-pipeline model. No simulator endpoint, fast-frontier result, native
dispatch, or accuracy comparison is claimed. The bounded blocker is preserved
in docs/sm86_c7_simbench_int8_imma_gemm_blocked_live_v1.json and its schema;
this is not a silently substituted shape or a passing pipeline row.

### C7-S8 — next active milestone

Run fp16_hmma_gemm_traceable as the next tensor-family complete slice. Reuse
the checked BF16-HMMA host safety/reference pattern, preserve exact binary and
trace arguments, and record any unsupported-SASS disposition separately from
the successful BF16 row. Do not infer INT8 accuracy from the BF16/FP32 rows.

### C7-S8 — direct SimBench FP16 HMMA slice — complete

The fp16_hmma_gemm_traceable row now has a direct SM86 timing-shape slice
using one rebuilt binary and identical m=32,n=16,k=64,seed=42 arguments for
host correctness, NVBit, replay, and five fresh NCU application captures. The
host FP16-cast FP32 reference check passed with zero mismatches and maximum
absolute error 9.54e-7; SASS contains HMMA.16816.F32. Replay has full
four-block coverage and 404 records, and NCU reports the same 404 warp
instructions plus 16 HMMA tensor-pipeline instructions. NCU
gpc__cycles_elapsed.max is 4934,4910,4990,4900,5096 (median 4934); the
simulator completion lower bound is 3337, so lower-bound consistency is true
but no point estimator or finite upper is claimed. The independent extractor
observes 16 HMMA records and 64 cross-SM line-topology events, but no hardware
completion token, owner/read-from relation, or global instrumentation order.
Dirty-writeback and coherence tails remain open. The fast analyzer refuses all
four warps for nonfull predicate populations; native macro dispatch remains
zero. Evidence and schema:
docs/sm86_c7_simbench_fp16_hmma_gemm_direct_live_v1.json and its schema. This
closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S9 — next active milestone

Run one additional direct SM86 SimBench row from the remaining memory, control,
or tensor populations (prefer global_vectorized_64b_copy,
shared_bank_conflict_stride, or tf32_hmma_gemm_traceable). Preserve the same
checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU contract; retain any unsupported-SASS or OOM/OOD
disposition without silent shape substitution.

### C7-S9 — direct SimBench 64-bit vector slice — complete

The global_vectorized_64b_copy row now has a direct SM86 timing-shape slice
using one rebuilt binary and identical samples=1024,bins=64,seed=42 arguments
for host correctness, NVBit, replay, and five fresh NCU application captures.
The host identity-output check passed with zero mismatches; SASS contains
LDG.E.64/STG.E.64. Replay has one-block coverage and 5,921 records, and NCU
reports the same 5,921 warp instructions. NCU gpc cycles were
130097,129581,130029,129449,129721 (median 129721); simulator completion
lower bound is 123918, so lower-bound consistency is true but no point
estimator or finite upper is claimed. The independent extractor observes
1,024 vector loads and 1,024 vector stores, but no hardware completion token,
owner/read-from relation, or global instrumentation order. Dirty-writeback and
coherence tails remain open. The fast analyzer refuses the one warp for a
nonfull predicate population; native macro dispatch remains zero. Evidence:
docs/sm86_c7_simbench_global_vectorized_64b_copy_direct_live_v1.json and its
schema. This closes only this slice; C3/C5/C6 and the wider C7 campaign
remain open.

### C7-S10 — next active milestone

Run one additional direct SM86 SimBench row from the remaining memory, control,
or tensor populations (prefer shared_bank_conflict_stride or
tf32_hmma_gemm_traceable). Preserve the same checked correctness, exact
binary/arguments, fresh NVBit, replay, and five-repetition NCU contract;
retain any unsupported-SASS or OOM/OOD disposition without silent shape
substitution.

### C7-S10 — direct SimBench shared-bank slice — superseded

The original artifact `docs/sm86_c7_simbench_shared_bank_conflict_stride_direct_live_v1.json`
is superseded. Its default-limit refusal was not a valid silicon OOD result:
SM86 permits dynamic shared memory above 48 KiB with an explicit opt-in.
Retain it only as historical execution evidence.

### C7-S10r — corrected direct SimBench shared-bank slice — complete

The same `trace_00` shape (64x64x4, 65,536 dynamic shared bytes) now opts into
the device's 101,376-byte SM86 limit. Host identity/sink checks and memcheck
pass. Fresh NVBit/replay/five-NCU use the exact opted-in binary; replay has
full one-block coverage and 15,350 records, and NCU reports the same 15,350
warp instructions. NCU gpc cycles are 100467,100317,100939,99980,100758
(median 100467); simulator lower is 103225, so lower-bound consistency is
false by 2,758 cycles (2.75%). Shared LDS/STS traffic, open writeback/
coherence tails, and zero native dispatch remain explicit. Evidence and schema:
`docs/sm86_c7_simbench_shared_bank_conflict_stride_corrected_live_v1.json`.
This is the active corrected slice; C3/C5/C6 remain open.

### C7-S11 — execution specification

Run one additional direct SM86 SimBench row from the remaining tensor/control
populations (prefer tf32_hmma_gemm_traceable or l2_thrash_pointer_chase).
Preserve checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU; retain every unsupported-SASS or OOM/OOD disposition.

### C7-S11 — TF32 HMMA blocked disposition — complete

The exact rebuilt tf32_hmma_gemm_traceable binary passes host TF32-rounded
reference correctness, produces a fresh NVBit trace with 588 records and 32
HMMA instructions, and completes five same-binary NCU captures:
6621,6328,6373,6360,6417 cycles (median 6373). The independent extractor
observes 112 cross-SM line-topology events but no hardware completion token or
global instrumentation order. Replay is explicitly blocked before any timing
frontier because sim-run reports that HMMA.1688.F32.TF32 has no SM86
execution-pipeline model. No simulator endpoint, fast-frontier result, native
dispatch, or accuracy comparison is claimed. Evidence:
docs/sm86_c7_simbench_tf32_hmma_gemm_blocked_live_v1.json and its schema.
This is an explicit unsupported-SASS disposition, not a silently substituted
shape or a passing pipeline row.

### C7-S12 — next active milestone

Run one additional direct SM86 SimBench row from the remaining control or
memory populations (prefer l2_thrash_pointer_chase or
global_atomic_and_return_used). Preserve checked correctness, exact
binary/arguments, fresh NVBit, replay, and five-repetition NCU; retain every
unsupported-SASS or OOM/OOD disposition.

### C7-S12 — direct SimBench pointer-chase slice — complete

The repaired l2_thrash_pointer_chase row passes host identity-output and
dependent-sink checks on its declared trace_00 shape
(chain_length=256,warps=32); unsigned bit hashing prevents the prior possible
negative-index OOB. The exact binary produces a fresh NVBit trace, full 32-block
replay coverage, and five NCU captures. Replay has 118,976 records and NCU
reports the same 118,976 warp instructions. NCU gpc cycles were
49573,49580,49739,49630,49523 (median 49580), while simulator completion lower
bound is 59545: lower-bound consistency is false by 9,965 cycles (20.1% above
the hardware median). This is a real falsifying mismatch, not a point-error
score or a reason to tune the model post hoc. The extractor reports 256
cross-SM line-topology events and 31 unknown address operands, with no hardware
completion token or global instrumentation order. Dirty-writeback and
coherence tails remain open; native dispatch remains zero. Evidence:
docs/sm86_c7_simbench_l2_thrash_pointer_chase_direct_live_v1.json and its
schema. This closes only this slice; C3/C5/C6 and the wider C7 campaign remain
open.

### C7-S13 — execution specification

Run one additional direct SM86 SimBench row from the remaining memory/control
population (prefer l2_persisting_window_sweep or warm_cold_pair_relaunch).
Preserve checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU; retain every unsupported-SASS, OOM/OOD, or falsifying
disposition without parameter tuning.

The active SimBench wrapper still rejects the repository-contained NVBit
tracer because it requires an external split-v2 tracer root; see
`docs/sm86_c7_simbench_population_slice_v1.json`. The blocker is recorded,
not bypassed with a hidden tracer or substituted shape. It does not invalidate
the eight direct bounded slices, which use the repository tracer explicitly and
retain their own exact binary/trace/control identities.

The wrapper blocker did not prevent a direct bounded timing-shape check: the
same `fp32_fma_dep_chain` binary/arguments completed native execution, NVBit,
replay, and five same-binary NCU captures. The analyzer refused all candidates
for nonfull predicate populations, so native dispatch stayed zero; simulator
8152 versus NCU median 8488 is a 3.96% mismatch. See
`docs/sm86_c7_simbench_fp32_direct_live_v1.json`. This is one probe row, not a
repository-wide claim.

Historical C3 store/service artifacts were also normalized: their schemas now
use explicit `withdrawn_historical_attempt`/`withdrawn_do_not_use` statuses
instead of encoding finite completion as current state. Raw numerical files
remain available for review, but they cannot be mistaken for active accuracy
evidence.

C6 now has a real same-shape scalar negative slice. The fixed-latency
`w4_d4_split` trace uses the exact 64-generation trace cubin for correctness,
NVBit, replay, and five NCU captures. Correctness passes; the analyzer refuses
all candidates because active/predicate populations are nonfull, so native
macro dispatch remains zero. The simulator endpoint is 3,578 cycles versus
NCU median 6,812 (47.5% low). The old 8,192-generation NCU rows are excluded
from this comparison. Artifact:
`docs/sm86_completion_frontier_c6_fixed_latency_live_v1.json`.

The phase-assay probe itself was corrected before another capture: its
`correct` field now compares the returned lane-0 token against a host
recomputation from the deterministic source initialization, in addition to
checking interval-sum arithmetic. Three direct SM86 runs (cache-all compact,
cache-all conflict, and streaming severe) pass both checks. Existing phase
captures remain historical because the executable changed; no old capture is
retroactively relabeled.

The corrected phase binary then completed one fresh NVBit + replay + five-NCU
path. Its direct run independently reports the expected token and interval
sum; the simulator endpoint is 18,305 cycles versus NCU median 16,658 (9.89%
APE), with the result-store writeback/coherence tails open. Artifact:
`docs/sm86_completion_frontier_c3_async_phase_corrected_live_v1.json`.

The resident-device message trace also has a same-input service differential:
the explicit model-local writeback/coherence queues complete 3 sectors and 2
coherence transactions, but the terminal tail flags remain true and the
semantic upper remains open. See
`docs/sm86_completion_frontier_c3_message_service_differential_v1.json`.

The next source-side repair corrected a concrete granularity error in the
model-local LDGSTS return ledger. The old code charged the eight-cycle
architecture-local token once per returned 32-byte sector; the raw trace
represents each warp-level LDGSTS as one coalesced source instruction with a
sector span. The new code charges one token per coalesced instruction and
shares its service timestamp across that instruction's sectors, with a
transactional overflow test. On the same previously captured live sweep the
post-repair lower bounds are 7,442/13,090/24,386 cycles for groups
16/32/64 and 46,978 for the untouched group-128 holdout; NCU medians are
7,904/12,254/20,949/38,348. The controlled mismatch is reduced but not
closed, and all dirty-writeback/coherence tails remain open. Artifact:
`docs/sm86_completion_frontier_c3_async_source_token_repair_v1.json`.

### C2 — observed tail facts — fact layer complete

`tools/extract_sm86_tail_facts.py` is a separate raw-protobuf plus static-SASS
extraction pass. Fresh resident-device message and matched async traces now
report store/atomic/fence/async-opcode facts, masks, allocation IDs, payload
coverage and cross-SM topology while explicitly leaving line generation,
owner/read-from, fence semantics and hardware completion unknown. Reports:
`docs/sm86_c2_message_device_tail_facts_v1.json` and
`docs/sm86_c2_async_tail_facts_v1.json`. The extractor now uses the tracer's
static function-id/PC map when available, reports deterministic same-warp
program-order edges and protocol counts, and explicitly records that no
global instrumentation order was present in either capture. It now also
reconstructs a launch-scoped logical async prefix relation (copy, commit,
wait, empty-group, and terminal-unwaited counts) from raw instructions and
decoded masks, without using SmCore aggregates. This closes the observed-fact
and protocol-partial-order deliverable. It is still not an independent native
witness: owner/read-from/generation/fence-scope/hardware completion and a
native effect/continuation relation remain open for C3/C6.

### C4/C5 exploratory observer calibration — positive statistical slice

Using only the five NCU repeats from compile-time group counts 16 and 32, an
ordinary least-squares observer model was materialized **before** profiling a
new group-128 binary. Its predicted interval was `37737..38655` cycles. The
five group-128 NCU values were `38353,38339,38348,38338,38395`: raw coverage
1.0 and point error 0.396%. This is recorded in
`docs/sm86_async_observer_calibration_holdout_v1.json`.

This is the first finite empirical NCU interval in the corrected program, but
it is not a semantic completion upper: the simulator group-128 lower bound is
168898 and dirty-writeback/coherence remain open. The calibration covers one
controlled async-group dimension, not workload-family generalization. C4/C5
remain active for independent families, OOD behavior, repeat dispersion, and
mechanism counters.

## C2/C3 technical seams

`sim/sm/src/issue.rs`: response ownership, `memory_write_ack`, GPU MEMBAR,
async groups; `sim/sm/src/system.rs`: due events, ack delivery, drain and
modeled writeback/coherence queues; `sim/sm/src/replay.rs`: coverage and tail
propagation; `sim/direct_run/src/lib.rs`: separate endpoints and projections.

Use these existing seams instead of adding reporting frameworks. A missing
hardware ownership token does not prevent an approximate model, but it prevents
calling the approximation an exact physical history. Every modeled request
needs a stable model identity, one completion, and the right consumer/scope.
Never clear all tails based on totals, topology, or presence of a timestamp.
Do not force a terminal L2 flush to make completion true; evaluate whether it
belongs in the chosen endpoint and keep its cost separate otherwise.

## C4/C5 accuracy experiment (declare before scoring)

- Fit point model and mechanism parameters only on independent calibration
  probes; keep the supplied example profile unvalidated until this is done.
- Use a third, disjoint workload-level set for interval calibration; candidate
  model selection is completed before held-out targets are collected.
- Store point error for an actual point estimator, not a lower endpoint renamed
  as a point. Report semantic lower consistency separately.
- Record medians, IQR, all repeats, signed error, per-family median/P90 APE,
  interval coverage and width. Five repeats cannot establish 95% tail coverage.
- Initial research target: median APE <=15%, P90 <=30% and 90% empirical
  interval coverage, median width/observed <=50% on a declared in-domain set.
  These are targets, not achieved results or global guarantees. Family/OOD
  transfer is reported separately and must not silently widen the domain.
- A miss stays a miss; once inspected, that row is development evidence for
  subsequent changes, never a fresh holdout again.

## C6 selective replay

First use a real native scalar region without memory effects. The independent
reference is an ordinary detailed native replay; compare front-end cursor,
pipeline releases, queue identities/order, effects and suffix results. Preserve
external events rather than discarding global state. Do not require a full
silicon cache proof for a memory-free model transition. Do not call synthetic
fixtures or diagnostic counters completion of this milestone.

## Current record

2026-10-03: C0-R is complete. The code no longer clears tail flags from coarse
queue counters/topology. The identical real store-2 trace drains 36 modeled
sectors but retains dirty-writeback/coherence uncertainty and an open observer
upper. Prior finite-upper claims are withdrawn. C1-R is now active: recapture
matched async/fence cases with fresh controls and exact launch selection.

2026-10-03 continuation: C2's extractor is now independent of the replay
validator and reports compressed-address reconstruction, launch-scoped
topology, raw dynamic order edges, static opcode joins, and logical async
copy/commit/wait prefix facts. The cacheable LDGSTS source-return ledger was
repaired to charge one service token per coalesced warp instruction rather than
per sector. Fresh source-only compact/conflict/severe captures show a
4,070-cycle NCU layout delta versus a 4,096-cycle model delta; the severe
holdout interval covers its five NCU repeats. A corrected phase probe checks
its data token independently. A real scalar fixed-latency trace remains
negative (zero candidates/native dispatch; 47.5% low on matched NCU), while
the full 15-row portable C7 live pipeline completes with 75 NCU captures,
15 NVBit traces, and 30 simulator replays. C3/C4/C5 remain partial: tails,
absolute endpoint bias, native selective dispatch, and non-portable populations
are still open.
### C7-S13 — direct SimBench persisting-window slice — complete

The repaired l2_persisting_window_sweep row passes exact identity-output checks
at allocation=1MiB, effective persistence window=524288 bytes, accesses=64,
warps=32, and completes fresh NVBit, replay, and five NCU captures. Replay has
full 32-block coverage and 159,200 records; NCU reports the same 159,200 warp
instructions. NCU gpc cycles were 30197,29757,29747,29616,29965 (median
29757), while simulator completion lower bound is 31173: lower-bound
consistency is false by 1,416 cycles (4.76% above the hardware median). This
is a real lower-bound falsifier, not a point-error score or a post-score tuning
target. The independent extractor reports 256 cross-SM line-topology events
and 31 unknown address operands, with no hardware completion token or global
instrumentation order. Dirty-writeback and coherence tails remain open; native
dispatch remains zero. Evidence:
docs/sm86_c7_simbench_l2_persisting_window_sweep_direct_live_v1.json and its
schema. This closes only this slice; C3/C5/C6 and the wider C7 campaign remain
open.

### C7-S14 — next active milestone

Run one additional direct SM86 SimBench row from the remaining control/memory
population (prefer warm_cold_pair_relaunch or l2_prefetch_hint_sweep).
Preserve checked correctness, exact binary/arguments, fresh NVBit, replay, and
five-repetition NCU; retain every unsupported-SASS, OOM/OOD, or falsifying
disposition without tuning scored parameters.

### C7-S14 — direct SimBench prefetch-hint slice — complete

The repaired l2_prefetch_hint_sweep row passes host identity-output checks at
accesses=256,prefetch_distance=4,warps=32, produces a fresh NVBit trace, and
completes five same-binary NCU captures. The compiler emits no explicit
prefetch mnemonic in static SASS, so the hint remains advisory/unobserved; the
trace still captures the workload path. Replay has full 32-block coverage and
267,712 records, and NCU reports the same 267,712 warp instructions. NCU gpc
cycles were 40653,40590,40577,40598,40726 (median 40598), while simulator
completion lower bound is 46874: lower-bound consistency is false by 6,276
cycles (15.46% above the hardware median). This is a real lower-bound
falsifier, not a point-error score or a tuning target. The independent extractor
reports 39 cross-SM line-topology events and 31 unknown address operands, with
no hardware completion token or global instrumentation order. Dirty-writeback
and coherence tails remain open; native dispatch remains zero. Evidence:
docs/sm86_c7_simbench_l2_prefetch_hint_sweep_direct_live_v1.json and its schema.
This closes only this slice; C3/C5/C6 and the wider C7 campaign remain open.

### C7-S15 — warm/cold multi-launch slice — complete

The corrected warm_cold_pair_relaunch binary uses separate first/repeat output
buffers and sinks, so the repeat pass cannot hide a first-pass error. The
uninterrupted sequence is cold pass, optional scratch eviction, and repeat pass;
host validation checks both outputs, both sinks with a floating-point error
bound, and every eviction element. Compute Sanitizer memcheck reports zero
errors. Fresh NVBit tracing records three launches with full coverage: 185,440
records for each warm/cold kernel and 44,224 for eviction. Detailed simulator
replay completes all three launches. Launch-scope lower bounds are 69,383 /
116,502 / 69,383 cycles; stream-L2 replay gives 69,383 / 116,502 / 52,557,
which demonstrates why cache scope must remain explicit. The fast frontier
refuses the multi-launch trace by its single-launch contract; this is not a
detailed replay failure.

Five application-replay NCU processes use `cache-control=none`, preserving the
application-managed cache sequence rather than flushing each replay. Instruction
counts match each launch. Cycle medians are 63,183 (first), 66,548 (eviction),
and 46,876 (repeat), with large repeat/eviction dispersion retained rather
than converted into a point score. This is cache-state observation and
multi-launch plumbing evidence, not a finite completion upper, point-accuracy,
or speedup claim. Evidence and schema:
`docs/sm86_c7_simbench_warm_cold_pair_relaunch_direct_live_v1.json` and its
schema.

### C7-S16a — persisting-window sink repair — complete

The corrected sink now includes all four C7-S13 Stage-1 anchor revisits; exact
sink/output checks pass on a fresh rebuild. C7-S10 dynamic shared-memory
opt-in and original-shape recapture are also complete. Do not change simulator
parameters from these scored observations.
The corrected live recapture is recorded in
`docs/sm86_c7_simbench_l2_persisting_window_sweep_corrected_live_v1.json`.

### C7-S16b — prefetch evidence correction — complete

The exact C7-S14 binary's static SASS contains five `CCTL.E.PF2` instructions.
The corrected artifact now reports that instruction as observed while keeping
cache effect, request completion, and hardware timing unobservable. The same
trace/replay/NCU measurements remain a lower-bound falsifier; no scored target
was used to alter the model. See
`docs/sm86_c7_simbench_l2_prefetch_hint_sweep_corrected_live_v1.json`.

### C7-S17 — corrected-artifact regression — complete

All 19 current live/blocked C7 artifacts validate against their sibling JSON
schemas; the inventory has 14 complete direct rows and two explicit replay
blockers. The Cargo workspace regression passes (38 test groups; no reported
failures). The corrected shared-bank and persisting-window artifacts are the
active records; their pre-opt-in/pre-sink artifacts remain superseded history.
No scored NCU target was used to change simulator parameters.

### C7-S18 — repaired atomic-and-return slice — complete

`global_atomic_and_return_used` was repaired before capture: ordinary stores
that raced with histogram bins were removed, disjoint return sinks were added,
and the host now checks the identity output plus deterministic final bitwise-AND
histogram. The exact trace shape passes; SASS contains
`ATOMG.E.AND.STRONG.GPU`. Replay has full one-block coverage and 46,882
records; NCU reports the same instruction count. NCU gpc cycles are
1060292,1060281,1060397,1061866,1060637 (median 1060397); simulator lower
916220 remains separated by open completion tails. Native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_global_atomic_and_return_used_direct_live_v1.json`.

### C7-S19 — carveout sensitivity slice — complete

`carveout_sensitivity_probe` now validates its identity output after explicit
shared-memory/carveout attribute setup and launch-error checks. The exact
allocation=1MiB, shared=16KiB, carveout=25%, accesses=256, warps=32 binary
passes fresh NVBit, detailed replay, memcheck, and five NCU captures. Replay
has full 32-block coverage and 824,576 records; NCU reports the same
instruction count. NCU gpc cycles are 178537,177540,177927,178647,178521
(median 178521); simulator lower is 211002, a 32,481-cycle (18.2%) falsifier.
Open writeback/coherence tails and zero native dispatch remain explicit.
Evidence: `docs/sm86_c7_simbench_carveout_sensitivity_probe_direct_live_v1.json`.

### C7-S20 — vendor SGEMM blocked disposition — complete

The exact rebuilt cuBLAS FP32 binary passes host correctness and produces a
fresh NVBit trace of ampere_sgemm_128x32_nn. Five same-binary NCU captures
report 8,784 warp instructions and cycles 8904,8905,8894,8955,8898 (median
8904). Replay is explicitly blocked: the kernel is loaded from libcublasLt.so
with no stable static module/control sidecar, and the traced dynamic record
count (29,752) cannot be silently crosswalked to NCU's 8,784. No simulator
endpoint, native dispatch, or accuracy comparison is claimed. Evidence:
`docs/sm86_c7_simbench_vendor_sgemm_cublas_blocked_live_v1.json`.

### C7-S21 — vendor strided SGEMM blocked disposition — complete

The exact rebuilt strided-batched cuBLAS FP32 binary passes host batch-0
correctness and produces a fresh NVBit trace of ampere_sgemm_128x128_nn. Five
same-binary NCU captures report 102,176 warp instructions and cycles
15524,15492,15550,15511,15503 (median 15511). Replay is explicitly blocked:
the kernel is loaded from libcublasLt.so with no stable static module/control
sidecar, and the traced dynamic record count (367,168) cannot be silently
crosswalked to NCU. No simulator endpoint, native dispatch, or accuracy
comparison is claimed. Evidence:
`docs/sm86_c7_simbench_vendor_sgemm_strided_batched_cublas_blocked_live_v1.json`.

### C7-S22 — working-set capacity slice — complete

The repaired working_set_capacity_sweep binary passes anchor identity-output
checks at working_set=1MiB, accesses=256, warps=32, with full 32-block replay
coverage and Compute Sanitizer-safe execution. Replay has 789,888 records and
NCU reports the same 789,888 warp instructions. NCU gpc cycles are
129756,129914,129648,130077,129710 (median 129756), while simulator lower is
143555: lower-bound consistency is false by 13,799 cycles (10.63%). This is a
real memory-capacity falsifier; tails remain open and native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_working_set_capacity_sweep_direct_live_v1.json`.

### C7-S23 — DRAM stream stride slice — complete

The repaired dram_stream_stride_mix binary passes host identity-output checks at
warps=32,accesses=128,stride_classes=2 and completes fresh NVBit, replay, and
five NCU captures. Replay has full 32-block coverage and 1,716,864 records;
NCU reports the same instruction count. NCU gpc cycles are
418778,418925,419105,419040,419638 (median 419040), while simulator lower is
550005: lower-bound consistency is false by 130,965 cycles (31.25%). This is a
real DRAM-stream falsifier, not a point-error score or tuning target. Tails
remain open and native dispatch is zero. Evidence:
`docs/sm86_c7_simbench_dram_stream_stride_mix_direct_live_v1.json`.

### C7-S24 — local dynamic-index slice — complete

The repaired local_dynamic_index_array binary passes host identity-output checks
at samples=1024,bins=64, exposes STL.128/LDL local traffic in SASS, and
completes fresh NVBit, replay, and five NCU captures. Replay has full one-block
coverage and 72,655 records; NCU reports the same 72,655 warp instructions.
NCU gpc cycles are 1243158,1244664,1244381,1243032,1243714 (median 1243714),
while simulator lower is 1443183: lower-bound consistency is false by 199,469
cycles (16.04%). This is a real local-memory falsifier; tails remain open and
native dispatch is zero. Evidence:
`docs/sm86_c7_simbench_local_dynamic_index_array_direct_live_v1.json`.

### C7-S25 — local volatile-array slice — complete

The repaired local_volatile_array binary passes host identity-output checks and
exposes STL local traffic in SASS. Replay has full one-block coverage and
29,017 records; NCU reports the same 29,017 warp instructions. NCU gpc cycles
are 421983,422645,423139,422474,422190 (median 422474), while simulator lower
is 524608: lower-bound consistency is false by 102,134 cycles (24.18%). This
is a real local-memory falsifier; tails remain open and native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_local_volatile_array_direct_live_v1.json`.

### C7-S26 — register-spill local-memory slice — complete

The register_spill_local_mem build uses -maxrregcount=32 and reports STACK 424
bytes plus STL traffic. Host identity output passes with independent float-print
tolerance; replay has full 64-block coverage and 271,680 records, and NCU
reports the same instruction count. NCU gpc cycles are
23036,23073,23002,23206,23117 (median 23073), while simulator lower is 30010:
lower-bound consistency is false by 6,937 cycles (30.08%). This is a real
spill/local-memory falsifier; tails remain open and native dispatch is zero.
Evidence: `docs/sm86_c7_simbench_register_spill_local_mem_direct_live_v1.json`.

### C7-S27 — atomic max histogram slice — complete

The repaired global_atomic_max_histogram binary checks identity output and the
deterministic final atomicMax histogram; fresh SASS emits RED.MAX. Replay has
full one-block coverage and 42,874 records, and NCU reports the same
instructions. NCU gpc cycles are 568531,567884,568408,568964,568350 (median
568408), while simulator lower is 436096, so lower-bound consistency is true;
reduction RMW and dirty-writeback/coherence tails remain open and native dispatch
is zero. Evidence: docs/sm86_c7_simbench_global_atomic_max_histogram_direct_live_v1.json.

### C7-S28 — atomic max return slice — complete

The repaired global_atomic_max_return_used binary checks identity output and
final atomicMax histogram with disjoint finite return sinks; SASS contains
ATOMG.E.MAX. Replay has full one-block coverage and 46,882 records, and NCU
reports the same instruction count. NCU gpc cycles are
1060971,1060224,1060648,1060929,1060243 (median 1060648), while simulator
lower is 916220, so lower-bound consistency is true; tails remain open and
native dispatch is zero. Evidence: `docs/sm86_c7_simbench_global_atomic_max_return_used_direct_live_v1.json`.

### C7-S29 — full-shape DRAM streaming dot-product slice — complete

The hardened `dram_streaming_dot_product` timing-shape binary validates
`n=131072, passes=4, blocks=128, threads=256` with per-block floating-point
roundoff checks, launch/error checks, bounded allocation arithmetic, and zero
Compute Sanitizer errors. Fresh NVBit covers all 128 blocks; detailed replay
has 297,600 warp records and the five NCU rows report the same 297,600
`smsp__inst_executed.sum` instructions. NCU gpc cycles are
7714,7579,7548,7524,7562 (median 7562); simulator lower is 5345, so
lower-bound consistency is true. The completion upper remains open because
dirty-writeback and coherence are unobserved; fast-frontier candidates and
native macro dispatch remain zero. Evidence:
`docs/sm86_c7_simbench_dram_streaming_dot_product_direct_live_v1.json` and
its schema. This is not a point-accuracy or speedup claim.

### C7-S30 — full-shape RMSNorm BF16 slice — complete

The repaired `rmsnorm_fwd_bf16` driver now validates the actual BF16 inputs,
checks every output against a bounded BF16-relative tolerance, reports
allocation/argument/launch failures, and emits a `--summary` mode so a large
timing capture cannot be mistaken for an output dump. Shape metadata now
matches the implementation: one 256-thread block per row. The timing shape
`rows=1024, hidden=4096` passes host validation and Compute Sanitizer; fresh
NVBit covers all 1,024 blocks, replay has 4,235,264 warp records, and NCU
reports the same instruction count. Five application-replay NCU gpc cycles are
48453,48106,48931,50179,48481 (median 48481); simulator lower is 43214, so
lower-bound consistency is true. Dirty-writeback/coherence tails remain open,
and fast-frontier candidates/native dispatch remain zero. Evidence:
`docs/sm86_c7_simbench_rmsnorm_fwd_bf16_direct_live_v1.json` and its schema.
This is not a point-accuracy or speedup claim.

### C7-S31 — atomic-AND histogram slice — complete

The repaired `global_atomic_and_histogram` driver now uses checked integer
parsing, bounded index/allocation arithmetic, launch-error checks, host
allocation checks, and deterministic final histogram validation. Its former
billion-operation timing shape was reduced to `samples=16384,bins=256` in the
shape contract to avoid an OOM/impractical-trace path; the captured `trace_00`
row remains `samples=1024,bins=64`. Fresh host validation and Compute
Sanitizer pass; SASS contains `RED.E.AND`, NVBit/replay cover one block with
42,874 warp records, and NCU reports the same count. Five NCU gpc cycles are
568244,568570,568083,567453,568196 (median 568196); simulator lower is
436096, so lower-bound consistency is true. Dirty-writeback/coherence tails
remain open and native dispatch is zero. Evidence:
`docs/sm86_c7_simbench_global_atomic_and_histogram_direct_live_v1.json` and
its schema. This is not a point-accuracy or speedup claim.

### C7-S32 — vendor BF16 cuBLAS control disposition — complete

The repaired `vendor_gemm_cublas` driver passes the exact `32x32x32` BF16
reference check and Compute Sanitizer. Five same-binary application-replay NCU
captures observe the dynamic
`cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_32x1_nn_align8` kernel with
1,456 warp instructions and gpc cycles 5347,5360,5397,5245,5235 (median
5347). The repository binary's control sidecar has zero functions; NVBit fails
closed on the unsupported tensor-core shape before publishing a complete
trace, even with enhanced tracing disabled. Because the kernel originates in
cuBLASLt and no stable static control crosswalk exists, replay, simulator
timing, native dispatch, and accuracy comparison remain blocked. Evidence:
`docs/sm86_c7_simbench_vendor_gemm_cublas_blocked_live_v1.json` and its
schema. This is not a point-accuracy or speedup claim.

### C7-S33A — research-probe inventory correction — complete

The current tree contains 42 research-probe directories, not the stale 34-row
count: 31 have Makefiles for SM86 CUDA builds, six contain CUDA source or
preserved binaries without a Makefile, and five are analysis-only directories
with no native binary. The corrected inventory is
`docs/sm86_c7_research_probe_inventory_v1.json`; no build or live-capture
success is inferred from an existing executable.

### C7-S33B — research-probe build matrix — complete

All 31 current Makefile probes build successfully with the pinned CUDA 13.2,
GCC 14, and `sm_86` toolchain. The machine-readable result is
`docs/sm86_c7_research_probe_build_matrix_v1.json`; generated binaries are
not treated as live evidence. The six source/preserved-binary rows and five
analysis-only rows remain separate until they have a reproducible bounded
launch contract.

### C7-S33C — issue-port vertical slice — complete

The matched `issue_port_control_trace` cell `w1_p10_l16` (dependent IADD,
length 16, one warp, 64 generations) passes seven host correctness trials,
fresh NVBit tracing, full replay, independent tail extraction, and five
same-binary application-replay NCU captures. Replay has 3,034 warp records;
NCU gpc cycles are 9977,9987,9983,9975,9969 (median 9977), while the
simulator completion lower is 7159, so lower-bound consistency is true. The
earlier long 8,192-generation binary measurement was excluded because it was
not the traced executable. The auxiliary enhanced SASS parser skipped its
large input, but the primary payload and static control sidecar are present.
Evidence: `docs/sm86_c7_research_probe_issue_port_control_live_v1.json` and
its schema. This is a probe-slice timing comparison, not a point-accuracy or
speedup claim.

### C7-S33D — matched issue-port 68-cell matrix — complete

The campaign interface was repaired so correctness, NVBit, replay, static
control, and NCU all use the same 64-generation trace executable; the legacy
8,192-generation binary is no longer mixed into scored rows. All 68 cells pass
seven correctness trials, produce full NVBit coverage, replay deterministically,
and have five NCU repetitions with instruction-count crosswalks. Aggregate
result: 68/68 complete, 65 lower-bound-consistent cells, and 3 real
lower-bound falsifiers; native macro dispatch remains zero and all completion
uppers remain open. Evidence:
`docs/sm86_c7_research_probe_issue_port_control_matrix_live_v1.json` and its
schema. NVBit active-thread totals remain separate from warp-record counts.

### C7-S33E — async barrier vertical slice — complete

The async-barrier campaign was repaired to use its bounded 64-group trace
binary for correctness, NCU, NVBit, and replay; the legacy 16,384-group binary
is no longer mixed into scored rows. Cell `w4_c11_l1_g0` passes seven host
correctness trials, fresh NVBit, full deterministic replay, independent async
tail extraction, and five NCU repetitions. Replay and NCU both report 9,104
warp instructions; NCU gpc cycles are 21278,21246,21262,21689,21254 (median
21262), while simulator lower is 24386, a real lower-bound falsifier. Async
copy protocol facts are present, but hardware completion/writeback/coherence
tails remain open. The simulator readiness schema check was updated from a
stale literal v1 requirement to positive-version validation. Evidence:
`docs/sm86_c7_research_probe_async_barrier_overlap_live_v1.json` and its
schema. This is not a point-accuracy or speedup claim.

### C7-S33F — matched async-barrier 36-cell matrix — complete

The repaired campaign uses the same 64-group trace binary for all stages and
accepts the simulator's current positive readiness schema instead of requiring
stale schema version 1. All 36 cells pass seven host correctness trials,
produce full NVBit coverage, replay deterministically, and have five NCU
repetitions with instruction-count crosswalks. Aggregate result: 36/36
complete, 23 lower-bound-consistent cells, and 13 lower-bound falsifiers.
Async-copy lane facts remain explicit per cell; hardware completion,
writeback, and coherence uppers remain open, and native macro dispatch is zero.
Evidence: `docs/sm86_c7_research_probe_async_barrier_overlap_matrix_live_v1.json`
and its schema. No fixed barrier latency or point-accuracy claim is made.

### C7-S33G — store-ack vertical slice — complete

The `store_ack_frontier` store-none arm (`b1_d4_l32_line_unique_g0`) passes
seven host correctness trials, fresh NVBit tracing, full replay, independent
tail extraction, and five same-binary NCU captures. Replay and NCU both report
662 warp instructions; NCU gpc cycles are 6963,6890,6881,6892,6912 (median
6892), while simulator lower is 5238, so lower-bound consistency is true.
Store acknowledgement, writeback, and coherence completion remain unobserved;
native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_store_ack_frontier_live_v1.json` and its schema.
This is not a point-accuracy or speedup claim.

### C7-S33H — matched store-ack 30-arm matrix — complete

All ten declared `store_ack_frontier` cells and three explicit arms were
captured with one-arm trace-only launches, full NVBit coverage, deterministic
replay, and five NCU repetitions per arm (30 arms, 150 NCU rows). Instruction
counts crosswalk per arm with no failures; all 30 simulator lower bounds remain
below their NCU medians. The write-ack candidate remains a model intervention,
not a hardware completion oracle; completion/writeback/coherence uppers and
native dispatch remain open. Evidence:
`docs/sm86_c7_research_probe_store_ack_frontier_matrix_live_v1.json` and its
schema.

### C7-S33I — memory-retirement vertical slice — complete

The matched `memory_retirement_service` store/none/line_stream cell
`b1_store_none_line_stream_d4_l32` passes seven host correctness trials, fresh
NVBit, full replay, independent tail extraction, and five NCU captures. Replay
and NCU both report 1,178 warp instructions; NCU gpc cycles are
11464,11391,11370,11363,11402 (median 11391), while simulator lower is 9064,
so lower-bound consistency is true. Memory retirement/writeback/coherence
completion remains open and native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_memory_retirement_service_live_v1.json` and its
schema.

### C7-S33J — memory-retirement 102-cell matrix — complete

All 102 declared cells pass host correctness, fresh NVBit capture, and five
NCU repetitions. Replay now completes all 102 cells after the simulator parks
`CCTL.IVALL` behind the next external L1-fill response horizon when a pending
fill has no local timestamp. Eight system-fence cells retain explicit NCU/replay
instruction-count mismatches (NCU reports nine additional warp instructions
per CTA); these are crosswalk failures, not silently scored rows. Lower-bound
consistency is reported only where instruction identity matches. Evidence:
`docs/sm86_c7_research_probe_memory_retirement_service_matrix_live_v1.json`
and its schema. Completion uppers and native dispatch remain open.

### C7-S33K — CCTL pending-fill repair — complete

The replay seam now refuses to mutate L1 on an untimed pending fill. It parks
the SM at the system-provided response horizon, then retries `CCTL.IVALL`
after the response is consumed; timed fills use the cache's explicit due-time
frontier. Focused sim-mem/sim-sm tests pass (199 + 493 tests), and all four
previous CCTL replay blockers now complete. No invented completion token or
upper bound was added.

### C7-S33L — system-fence crosswalk diagnostic — complete

Investigate the eight system-fence NCU/replay instruction-count mismatches.
Static/dynamic opcode populations and NCU pipe/op counters were compared on a
matched no-fence/system-fence pair. NVBit adds three visible SASS instructions
(`MEMBAR`, `ERRBAR`, `CCTL`), while NCU reports twelve additional warp
instructions and deltas across branch/ALU/LSU/texture/global-store counters.
This is an unresolved NCU-versus-trace metric-semantics/counter-replay
mismatch; simulator padding and hidden-instruction synthesis are explicitly
rejected. Diagnostic evidence:
`docs/sm86_c7_memory_retirement_system_fence_crosswalk_diagnostic_v1.json` and
its schema. The eight rows remain crosswalk exclusions until an independent
instruction-identity explanation is found.

### C7-S33M — source-counter crosswalk check — corrected

An independent NCU SourceCounters export was compared for matched no-fence and
system-fence binaries. The source rows report one warp execution for each
visible `MEMBAR.SC.SYS`, `ERRBAR`, and `CCTL.IVALL` PC; their separate stall
sampling columns are zero. The full source-row sum is 1104 versus 1101, while
`smsp__inst_executed.sum` is 1113 versus 1101. The metric crosswalk remains
unresolved; no simulator padding or hidden-instruction synthesis is applied.
The corrected diagnostic is
`docs/sm86_c7_memory_retirement_system_fence_crosswalk_diagnostic_v1.json`.

### C7-S33N — barrier-pipeline vertical slice — complete

The matched 64-generation `barrier_pipeline` cell `w4_b10_s0` (one CTA
barrier, four warps, zero skew) passes seven host correctness trials, fresh
NVBit, full replay, independent tail extraction, and five NCU captures. Replay
and NCU both report 1,884 warp instructions; NCU gpc cycles are
5731,5732,5755,5728,5732 (median 5732), while simulator lower is 3545, so
lower-bound consistency is true. Barrier rendezvous/retirement tails remain
open and native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_barrier_pipeline_live_v1.json` and its schema.

### C7-S33O — matched barrier-pipeline 33-cell matrix — complete

All 33 declared `barrier_pipeline` cells use the matched 64-generation trace
binary for correctness, fresh NVBit, deterministic replay, and five NCU
repetitions. Every cell has an instruction-count crosswalk and remains
lower-bound-consistent; barrier protocol, warp width, and skew effects remain
mechanism-local observations with open completion tails. Native dispatch is
zero. Evidence:
`docs/sm86_c7_research_probe_barrier_pipeline_matrix_live_v1.json` and its
schema.

### C7-S33P — l1tex admission vertical slice — complete

The matched 64-generation `l1tex_admission_backpressure` store/line-unique
cell (`blocks=1, warps=4, depth=4, active_lanes=32, gap=0, all pattern`)
passes seven host correctness trials, fresh NVBit, full replay, independent
tail extraction, and five NCU captures. Replay and NCU both report 2,980 warp
instructions; NCU gpc cycles are 8898,8956,8887,8905,8976 (median 8905),
while simulator lower is 6524, so lower-bound consistency is true. The
post-burst MEMBAR/ERRBAR/CCTL sequence and writeback/coherence tails remain
open; native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_live_v1.json` and
its schema.

### C7-S33Q — matched l1tex admission 100-cell matrix — complete

All 100 configured `l1tex_admission_backpressure` cells pass host correctness,
fresh NVBit capture, deterministic replay, and five NCU repetitions with
per-cell instruction crosswalks. Every replay-complete cell is lower-bound
consistent; the post-burst MEMBAR/ERRBAR/CCTL sequence and writeback/coherence
uppers remain open, and native dispatch is zero. Evidence:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_matrix_live_v1.json`
and its schema.

### C7-S33R — l2 queue-frontier vertical slice — superseded

The original two-launch `l2_queue_frontier` capture is retained as historical
v1 evidence. Its shape and replay observations remain useful, but its host
binary did not independently validate the checksum and allocation contract.
The corrected same-binary capture is recorded under C7-S33S.

### C7-S33S — L2 probe safety and evidence repair — complete

The L2 queue-frontier driver now rejects malformed/overflowing shapes, checks
the aggregate allocation against available device memory before `cudaMalloc`,
initializes output sentinels, independently recomputes the lane-zero checksum,
and fails on invalid timestamp bounds or checksum results. A valid slice and a
reuse slice pass; an oversized allocation is rejected before launch, and
Compute Sanitizer reports zero errors. The fresh artifact is
`docs/sm86_c7_research_probe_l2_queue_frontier_live_v2.json`; it uses the same
rebuilt binary for correctness, NVBit, replay, and five NCU repetitions. Target
replay/NCU both report 70,560 warp instructions; NCU gpc cycles are
7714,7784,7729,7666,7663 (median 7714), while the simulator lower is 5751.
The lower remains consistent, but dirty-writeback/coherence completion is
unknown and no point-accuracy or speedup claim is made.

### C7-S33T — L1TEX arrival/replay vertical slice — complete

The driver now uses strict integer parsing, checked allocation arithmetic, a
512-MiB/half-free-memory budget, and explicit dependent-work observability.
The stale separate trace binary was removed from the campaign interface;
correctness, NVBit, replay, and NCU all bind to the rebuilt executable. The
`store,line,84,8,8,32,0,1` cell passes seven host trials, a reduced
Compute-Sanitizer run, fresh 84-block NVBit capture, deterministic replay, and
five application-replay NCU captures. Replay and NCU both report 809,088 warp
instructions; NCU gpc cycles are 32085,31590,32333,31748,31518 (median
31748), while the simulator lower is 8486. Output-wavefront, request, L2,
DRAM-write, and issue counters are retained as independent observations, not
simulator inputs. Dirty-writeback/coherence completion remains unknown and no
point-accuracy or speedup claim is made. Evidence:
`docs/sm86_c7_research_probe_l1tex_arrival_replay_service_live_v1.json`.

### C7-S33U — SFU dependent-chain vertical slice — complete with falsifier

The `sfu_frontier` driver now uses strict integer parsing, a 262,144-iteration
bound, checked allocation arithmetic, a 64-MiB/half-free-memory budget, output
sentinels, and timestamp/checksum-presence checks. The dependent cell
`84,4,1024,dependent` passes seven host trials, reduced Compute Sanitizer,
four-launch NVBit capture, target-4 replay, and five launch-3-skipped NCU
captures. Replay and NCU both report 1,389,024 target warp instructions; NCU
gpc cycles are 34274,34282,34292,34297,34274 (median 34282), but the
simulator lower is 38099. This is a real lower-bound falsifier, not a tuning
opportunity; the cell is retained as negative evidence. No SFU latency is
fitted into a scored workload, and dirty-writeback/coherence completion
remains unknown. Evidence:
`docs/sm86_c7_research_probe_sfu_frontier_live_v1.json`.

### C7-S33V — result-queue collision vertical slice — complete

The campaign no longer treats 64-generation trace cubins as equivalent to the
8,192-generation NCU cubins. It now uses one patched collision or separated
cubin for correctness, NVBit, replay, and NCU; the stale trace artifacts were
removed. The CUDA-driver harness uses strict parsing, bounded allocations,
sentinel output checks, elapsed-sample checks, and the deterministic sink
oracle. The collision cell (`warps=4, generations=8192`) passes seven host
trials, Compute Sanitizer, full NVBit coverage, deterministic replay, and five
NCU runs. Replay and NCU both report 524,416 warp instructions; NCU gpc cycles
are 307642,307646,307655,307628,307679 (median 307646), while simulator lower
is 287635. The result-queue hypothesis remains exploratory; no latency is
activated or fitted and dirty-writeback/coherence completion remains unknown.
Evidence: `docs/sm86_c7_research_probe_fixed_latency_result_queue_live_v1.json`.

### C7-S33W — result-queue matched collision/separated differential — complete with falsifier

The separated treatment was captured at the identical width and generation
count with the same driver, patched-cubin contract, replay, clocks, cache
policy, and five-NCU protocol. Both arms pass the host sink oracle, Compute
Sanitizer, full one-CTA NVBit/replay coverage, and instruction-count
crosswalks. Collision NCU median is 307646 with simulator lower 287635;
separated median is 340424 with simulator lower 353171, so separated is a real
lower-bound falsifier. The NCU treatment delta is 32778 cycles while the
simulator lower-bound delta is 65536; this differential is mechanism-local
evidence, not a fixed latency. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_pair_live_v1.json`.

### C7-S33X — next active milestone

Run one held-out width-8 collision/separated pair with the same 8,192-generation
single-cubin contract. Require fresh correctness, Compute Sanitizer, NVBit,
replay, and five NCU captures for both arms; compare the differential without
using NCU values to tune the model. If the held-out pair remains a falsifier,
retain it and move to the next mechanism rather than widening a latency term.
