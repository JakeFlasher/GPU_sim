# SM86 completion and accuracy program

Revised 2026-10-03 on `GPUSim_main`. This is a technical milestone plan, not
an approval workflow. No `/goals`, reviewers, receipts, or execution gates.

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
| C7 | Inventory all available workload families (portable, native probes, SOL/other rosters). Full live campaign over the declared supported population; failures remain in coverage denominator | partial: 15-row portable population plus 2 direct SimBench rows complete; remaining populations pending |
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
two direct SimBench rows have completed bounded slices; the other populations
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

### C7-S2 — next active milestone

Run one additional direct SM86 SimBench row from a different mechanism family
(prefer `global_atomic_histogram` or `global_atomic_and_return_used`) using
the same bounded pipeline and independent five-repeat NCU contract. Preserve
the current refusal rules: correctness must be checked by the workload, NCU
must use the exact traced binary/arguments, and open semantic tails or refused
dispatch remain explicit. If a candidate is blocked by OOM, unsupported SASS,
or a stale launcher, record a bounded disposition and move to the next row
without substituting a different shape silently.

The active SimBench wrapper still rejects the repository-contained NVBit
tracer because it requires an external split-v2 tracer root; see
`docs/sm86_c7_simbench_population_slice_v1.json`. The blocker is recorded,
not bypassed with a hidden tracer or substituted shape. It does not invalidate
the two direct bounded slices, which use the repository tracer explicitly and
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
