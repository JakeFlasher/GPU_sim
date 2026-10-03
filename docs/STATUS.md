# Current status

Updated 2026-10-02. The project develops analytical/statistical GPU performance
models and a runnable native SM86 replay path, with NCU as independent ground
truth for the declared measurement protocol.

## Research scope

Strictly forbidden: pursuing goals or exact proprietary mechanisms beyond normal academic PhD-level research; improve accuracy through analytical or statistical modeling and treat unobservable internals as black boxes.

The alpha branch permanently excludes M5/M6 authority, receipt, audit,
provenance, hashing, sealing, approval, freeze, admission, and validation
workflow artifacts. They must not be restored or required for execution or
commits; only local runtime safeguards needed for meaningful execution remain.
The obsolete M5/M6 and CUTLASS-successor campaign surfaces and dedicated tests
are deleted rather than skipped.

## Implemented and remaining work

| Surface | Current state |
|---|---|
| Native replay | SM86 instruction, memory, and event replay; separate frontiers |
| Custom frontends | Bounded functional/resource experiments, separate from native timing |
| Predictive extension | Opt-in resource/service intervals composed into a model-local frontier, cache-topology candidates, NCU candidate projection |
| Calibration | Profile loader and offline fitter exist; v2 requires explicit dependent latency and independent throughput, retains queue-depth/capacity interventions without fitting a queue equation, retains throughput interval endpoints, and records queue/occupancy/contention as explicitly unmodeled dimensions; supplied rates remain uncalibrated |
| Accuracy | R6 has a document-identity-consistent six-family holdout join labeled manifest-historical; live host identity is not independently verified; all 15 native intervals are open, so finite coverage is not applicable and no hardware-accuracy claim is made |
| Reification | Typed region contract and native-boundary comparator preserve only complete exact projections; incomplete canary boundaries fall back to replay |
| Selective coverage | Multi-warp/order and completion-multiplicity guards refuse conservatively; fast-frontier output includes full-trace candidate/reified/uncovered counts |
| Model macro seam | `sim-sm::ModelExecutor` consumes a certified scalar `MacroTransition`, emits explicit effects/pending completions, and resumes detailed descriptors; this is model-local |
| Native refinement | A closed two-instruction `SmCore` helper seam and a synthetic reference-feature event-loop fixture consume a complete scalar witness, reserve effect sequences, and resume a detailed suffix; latency/completion/front-end/pending/barrier/cache-tail classes refuse before mutation; real front-end/cache SM86 traces remain exact fallback after the canary's unresolved queue/response join |
| Service composition | Typed residual/replacement/independent-post-replay roles and a serial queue projection are explicit; grouped unknown tails keep queue uppers open and replacement admission is validated; full-trace demand is currently independent post-replay, so it does not silently alter the observer endpoint |
| Cache candidates | Warp-local instrumentation order can be observed when decoded identities are complete; exact history additionally requires temporal order, fence scope, line generation, owner/read-from, and hardware completion, while unresolved domains remain lower-only/unknown |
| Design evaluation | Fifteen-workload, six-family calibration/development/held-out NCU join runs post-prediction; v2 target contracts check metric/endpoint/clock/cache/launch/repetition and declared GPU identity plus family split semantics; predictions and manifest-built observations bind trace/control/card/launch identities when available; observations are labeled manifest-historical unless metadata is independently parsed; finite intervals, open refusals, accepted/refused coverage, model-local partial-order ties, and model-only step reduction are reported, while native front-end traces remain exact/refused |
| Payload/value work | Bounded model-local identity, ordering, and conservation checks |

Native replay is unchanged by the predictive flag. Service demand now yields an
explicit model-local service/queue composition; the live full-trace role is
independent post-replay, while only an explicit residual/replacement role may
feed an observer frontier. Unknown typed tails remove finite upper bounds. The
R6 corpus confirms that real native rows remain open/exact fallback and records
no macro dispatch or native step
reduction. Physical routing, compiler implementation, and internal cache state
are black-box variables, not milestones. Topology summaries are not causal
read-from histories. Endpoint projections require an explicit
statistical/modeling assumption rather than timer equality.

The P2 synthetic reference-feature fixture is a plumbing/differential result,
not native corpus compression or hardware speedup: its one dispatch is counted
in the reference event loop, while the retained SM86 corpus still reports zero
dispatches and exact fallback.

## Next work

The first-round history remains in the
[reification/refactor plan](../plans/sm86_reification_refactor_20261001.md).
The active second-round program is the
[second-round refinement plan](../plans/second_round_refinement_20261001.md),
with **R1 semantic closure and evidence hygiene complete**, **R2 executable
macro transition seam complete**, **R3 independent witness/order closure
complete**, **R4 service/queue semantics complete**, **R5 native state
refinement complete for the closed issue-model seam**, **R6
independent-corpus/observer evaluation complete**, and **R7 rebuttal/claim
discipline complete**. R1 removed unsound finite
bounds and evaluator ambiguities. R2 executes a model-local nonidentity
transition and exact continuation. R5 retains a closed native issue-model
helper transition, while P2 additionally exercises the production
reference-feature event loop on a synthetic complete scalar witness; real
front-end-enabled launches remain exact fallback. R6
expands the corpus and confirms that native compression, finite observer
uppers, and hardware accuracy remain unproven; no model-only step reduction is
relabeled as native compression.
Report finite intervals, open/refused cases, OOD behavior, actual execution
steps, memory, and runtime separately; a passing unit suite is not accuracy
evidence. The R6 corpus and third-round claim map are in
[`review_round3_20261002/README.md`](review_round3_20261002/README.md) and
[`third_round_rebuttal_20261002.md`](../plans/third_round_rebuttal_20261002.md).

The latest independent panel re-review is being addressed by the staged
[R8 refinement program](../plans/latest_review_refinement_20261002.md).
R8.0 review intake, R8.1 immediate soundness guards, R8.3 conservative native
eligibility, R8.4 service/queue identifiability, R8.5 measurement identity and
cost utility, and R8.6 synchronization are complete. R8.2 real-corpus native
dispatch remains **open**: a bounded native canary exposed a response-owner/
queue join failure before any real production macro could be consumed. P2
separately closes a synthetic complete-witness reference-feature event-loop
differential. The retained front-end-enabled SM86 corpus remains exact
fallback, with no native speedup or hardware causality claim. R8.3
exact-falls back for non-unit completion tails, pending tokens, or
untouched future-relevant state rather than calling the scalar recurrence
state-complete. R8.4 adds explicit residual/replacement service roles and a
model-local intervention test; the independent-post-replay default remains a
sidecar and queue-depth observations remain unmodeled.

R9.0 repaired the bounded reference-canary owner/queue projection failure by
widening unresolved joins to unknown ownership. The rerun terminates with
10,032 reference records and exact fallback throughout; it does not close R8.2
production macro dispatch.

R9.1 attempted the guarded witness/dispatcher path. The native canary has no
complete independent witness, so the run records zero native macro dispatches
and exact fallback throughout. P2 closes a separate synthetic reference
fixture, but does not promote it to a silicon or front-end completion claim.

R9.2 completed the two-pass witness audit negatively: no complete typed
pre/effect/post pair exists in the bounded canary, so self-derived witness
construction is rejected. The required positive falsifier is documented in the
R9 plan and remains open.

R10.0 completed a live 168-block SM86 development campaign: 10/10 cases ran
native correctness, NVBit trace, simulator replay, analyzer, and three fresh
post-prediction NCU repetitions. All intervals remained open, analyzer
candidate records and native macro dispatches were zero, and no speedup claim
is made. See [the live campaign result](review_round7_20261002/README.md).

R10.1 completed a live writeback campaign: 2/2 large dirty-stream cases ran
native correctness, NVBit trace, replay/analyzer, and three fresh NCU
repetitions. Writeback completion remains explicitly unavailable/open; no
parameter fit, native dispatch, or speedup claim is made. See
[the writeback result](review_round8_20261002/README.md).

R10.2 completed a live cache-policy contrast on two large cases with three
fresh `cache-control=none` repetitions after the all-cache measurements. The
contrast is observational only; no fitting, cache-causality, or speedup claim
is made. See [the contrast result](review_round9_20261002/README.md).

R11.0 completed detailed-versus-fast-frontier cost measurements on all 15
retained traces: 5,130,860 records, zero accepted candidates, and zero native
macro dispatches. The result reports wall/RSS and refusal coverage, not
speedup. See [the full cost result](review_round10_20261002/README.md).

The post-review P0–P6 completion program is synchronized and closed. Its final
real-path falsifier is the current bounded SM86 canary: 10,033 reference
records, exact fallback, and zero native dispatches. The synthetic
reference-feature fixture is a differential/plumbing result only. Full Cargo
workspace tests and all affected Python suites pass; repository-wide pytest is
not a valid check because the untracked review corpus and optional campaign
dependencies interrupt collection.

The accuracy-first NCU workstream has completed A0: the exact
metric/replay/cache/device/repetition contract, identity/build requirements,
artifact separation, and split rules are declared and schema-tested in
`docs/sm86_ncu_accuracy_protocol_v1.json`; no new measurements were collected.
A1 is complete: the metadata-only identity capture path validates GPU/driver/
clocks, NCU metric availability, tool/build/input identity, and replay/cache
contract. The current dirty checkout is labeled ineligible and no workload was
profiled. A2 is complete: the preflight/reporting path validates bounded
termination, five-repetition policy, dispersion, and prediction-before-
observation ordering without launching NCU. The current historical manifest is
correctly not ready for the new protocol. A3 is complete: source-path and
family/split closure pass, five expansion slots are declared, and missing
predictions keep preflight unready. A4 is complete: the protocol-aligned
source-bound 15-row prediction artifact is prepared before observations and
the service sidecar remains explicitly unvalidated/unmodeled where required.
No NCU observations were collected in A4. A5 is now complete: five fresh
application-replay/all-cache/no-clock-control repetitions ran for all 15
workloads after clean identity/preflight. The 75 observations are separate
from predictions; all intervals remain open, so finite accuracy is not
applicable (lower consistency 0.667/1.000/0.750 across the three splits). A6
is complete: cache `all→none` (15 workloads × five repetitions) and clock
`none→base` (two workloads × five repetitions) contrasts preserve separate
prediction/observation joins and matching mechanism counters. Simulator
projections were invariant; directions remain observational-only and all
uppers remain open. A7 is complete: the retained real SM86 trace produced
zero complete typed boundaries, zero native macro dispatches, and exact
fallback under the verified identity/endpoint contract. The primary outcome is NCU
accuracy; simulator wall-clock runtime remains secondary, with no
compression/speedup metric.

The A8 live-campaign foundation is complete under
[`accuracy_first_live_campaign_a8_20261002.md`](../plans/accuracy_first_live_campaign_a8_20261002.md).
A8.0–A8.3 are complete: the bounded runner classifies OOM/OOD/timeout and
launcher failures, all 15 portable-LCD binaries were rebuilt from source, and
all 15 correctness smoke rows passed without silent shape substitution. A8.4
and A8.5 are complete for one fresh `gemm_hmma_bf16_sm80` slice: native
correctness, NVBit trace, simulator replay, prediction-before-observation,
independently identified five-repetition NCU, and schema/contract validation
all pass. The prediction lower is 3023 cycles with an open upper; NCU target
cycles are 5684/5490/5514/5524/5517 (median 5517), so finite interval coverage
and point error are not applicable. Simulator dirty-writeback/coherence tails
remain open and native macro dispatch is zero. A8.6 handoff synchronization is
complete; A8 is complete and A9 is the next phase (not started).

## Completion program correction — 2026-10-03

The active [plan](../plans/sm86_completion_frontier_and_native_dispatch_20261002.md)
reopens C0/C1/C3. The previous tail-flag clearing used insufficient counters;
the 1600-cycle padding has no reproducible calibration; the async comparison
mixed 64- and 16384-group binaries; managed-memory message-passing timings
were not isolated coherence measurements. Previous finite-upper and C1
completion claims are withdrawn. Raw measurements remain historical data.
C0-R repairs soundness, then C1-R repairs matched live comparisons. Finite
statistical prediction and exact semantic completion are separate questions.
C4/C5 exploratory positive slice: calibration on group counts 16/32 predicted a finite 37737..38655 empirical NCU interval for a new group-128 binary before its NCU run; all 5 repeats (38353/38339/38348/38338/38395) were covered and point error was 0.396%. This is observer statistics only: simulator semantic lower 168898 and writeback/coherence remain open. C1-R matched async/control differential is a real negative model result: same 64-group binary target records match, NCU async-control delta is 13760 cycles while simulator lower-bound delta is 78852. C1-R matched async recapture is a real negative model result: same 64-group binary/arguments/control, target trace records equal NCU instruction counts, but simulator lower 84866 exceeds NCU median 20946. A resident-device message-passing recapture has simulator lower 1753 below NCU median 4321 and remains open. Both are retained as mismatch evidence; no finite upper or tuning claim. C0-R is verified on a live store trace; the same service configuration drains modeled queues but leaves dirty-writeback/coherence open. C1-R is active for matched async/fence recaptures. C4/C5 have one exploratory positive statistical slice: disjoint group16/32 calibration predicted group128 37737..38655 with 5/5 NCU coverage and 0.396% point error, while semantic completion remains open. C3 now refuses async timing lower bounds after the matched differential instead of emitting the false 84866-cycle lower. Full completion and real native selective dispatch remain unachieved.

The first A5 preflight correctly rejected the dirty checkout; after the
coherent implementation was committed and rebuilt, independent identity and
preflight passed. The first collection command retained a historical launch
skip and produced no profiled kernels; the corrected skip-zero campaign
completed all 75 repetitions and is the retained baseline.

### Completion-frontier continuation — 2026-10-03

The C3 async source-token repair is recorded in
`docs/sm86_completion_frontier_c3_async_source_token_repair_v1.json`. The
cacheable LDGSTS return ledger now charges one service token per coalesced
warp-level instruction rather than per 32-byte sector. A fresh post-change
group sweep gives model cycles 7,442/13,090/24,386/46,978 for groups
16/32/64/128 versus NCU medians 7,904/12,254/20,949/38,348; the mismatch is
reduced but remains. A fresh NVBit plus five-process NCU source-only capture is
retained in `docs/sm86_completion_frontier_c3_source_only_live_v1.json`: the
store-free case has finite model-local completion at 13,235 cycles versus NCU
median 12,616 (4.91% point error), while hardware completion tokens and fence
scope remain unobserved. This is a narrow positive model case, not a general
hardware-accuracy or writeback/coherence result.

C4/C5 now has a second declared holdout: compact/conflict calibration predicted
the severe destination-layout target at `24,476.2..24,902.2` cycles before
profiling. Five fresh severe NCU values are 24,822/24,855/24,875/24,891/24,840
(median 24,855); the interval covers the median and point APE is 0.667%. The
prediction and post-observation join are separated in
`docs/sm86_completion_frontier_c4_destination_layout_prediction_pre_ncu_v1.json`
and `docs/sm86_completion_frontier_c4_destination_layout_holdout_v1.json`.
This is one controlled layout family, not a semantic upper or global accuracy
claim.

C6 has a real same-shape scalar negative slice in
`docs/sm86_completion_frontier_c6_fixed_latency_live_v1.json`: the exact
64-generation trace cubin passes correctness and five NCU captures, but the
fast analyzer refuses all candidates for nonfull active/predicate populations
(zero native dispatches). Simulator endpoint 3,578 versus NCU median 6,812 is
47.5% low; prior 8,192-generation NCU rows are excluded.

C7’s portable population slice is complete: the full 15-row live campaign ran
native timing, NVBit tracing, two simulator replays, and five fresh NCU runs
per row in 389 seconds, with 15/15 completed and zero OOM/OOD/timeouts or
failed stages. Raw data is under
`results/alpha/c7_full_live_20261003_run/`; tracked summary:
`docs/sm86_c7_full_live_campaign_v1.json`. This is not yet the full repository
population: native probes, fixed-latency rosters, SOL/other suites, and
correctness-mode joins remain to be inventoried. Portable replay rows retain
open dirty-writeback/coherence tails; no native macro dispatch was requested.

The C7 inventory lists 15 portable rows, 31 SM86-capable rows among 92
SimBench rows, 34 research-probe directories, and 203 SOL baseline problems.
Portable timing paths and twenty-three direct SimBench slices have completed:
`fp32_fma_dep_chain`, `cp_async_shared_tile_copy`, `global_atomic_histogram`,
`global_vectorized_128b_copy`, `bf16_hmma_gemm_traceable`,
`global_scalar_32b_copy`, `global_atomic_return_used`,
`fp16_hmma_gemm_traceable`, `global_vectorized_64b_copy`,
`shared_bank_conflict_stride`, `l2_thrash_pointer_chase`,
`l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, and
`warm_cold_pair_relaunch`, and
`carveout_sensitivity_probe`, and `working_set_capacity_sweep`, and `dram_stream_stride_mix`, and `local_dynamic_index_array`, and `local_volatile_array`, `register_spill_local_mem`, and `global_atomic_max_histogram`, and `global_atomic_max_return_used`. These check outputs at the captured shape; the FP32
direct slice is timing-only (its separate correctness-shape run does not
validate that captured shape). All twenty-three retain open tails and zero native
macro dispatch. The inventory is
`docs/sm86_c7_workload_population_inventory_v1.json`.

Two tensor rows (INT8 IMMA and TF32 HMMA) pass host correctness, NVBit, and NCU
but are blocked before simulator timing because their SM86 pipeline models are
missing. Three L2 rows remain real lower-bound falsifiers: pointer chase
(59,545 vs NCU median 49,580), persisting window (31,173 vs 29,846 after the
sink repair), and prefetch hint (46,874 vs 40,598). The prefetch binary emits
five static `CCTL.E.PF2` instructions; cache effect and completion remain
unobservable. The corrected shared-bank shape uses SM86 dynamic shared-memory
opt-in (65,536 bytes against default 49,152 and opt-in 101,376), superseding
its old false OOD disposition.

Warm/cold multi-launch is complete with separate outputs/sinks, zero
Compute-Sanitizer errors, detailed three-launch replay, launch/stream-L2
separated simulator scopes, and cache-control=none NCU observations. First,
eviction, and repeat medians are 63,183 / 66,548 / 46,876 with large cache-state
dispersion; this is plumbing evidence, not point accuracy or completion proof.
See `docs/sm86_c7_simbench_warm_cold_pair_relaunch_direct_live_v1.json`.

Corrected active evidence:
- `docs/sm86_c7_simbench_shared_bank_conflict_stride_corrected_live_v1.json`
- `docs/sm86_c7_simbench_l2_persisting_window_sweep_corrected_live_v1.json`
- `docs/sm86_c7_simbench_l2_prefetch_hint_sweep_corrected_live_v1.json`
- `docs/sm86_c7_simbench_warm_cold_pair_relaunch_direct_live_v1.json`
- `docs/sm86_c7_simbench_carveout_sensitivity_probe_direct_live_v1.json`
Superseded artifacts remain historical and are not current evidence.

C7-S18 atomic-and-return is complete after repairing its shared-bin sink
race: disjoint return sinks and a deterministic final bitwise-AND histogram pass;
46,882 replay/NCU instructions match. C7-S19 carveout sensitivity is complete
and falsifies lower-bound consistency (211,002 simulator lower vs 178,521 NCU
median). C7-S20/S21 vendor SGEMM variants are explicit replay blockers: cuBLAS
kernels from libcublasLt.so have no stable static control sidecar, and traced
records cannot be silently crosswalked to NCU instruction counts. See
`docs/sm86_c7_simbench_vendor_sgemm_cublas_blocked_live_v1.json` and
`docs/sm86_c7_simbench_vendor_sgemm_strided_batched_cublas_blocked_live_v1.json`.

C7-S22 working-set capacity is complete: the repaired anchor identity checks
pass at a 1MiB working set, replay/NCU match at 789,888 instructions, and
simulator lower 143,555 exceeds NCU median 129,756 by 13,799 cycles (10.63%).
See `docs/sm86_c7_simbench_working_set_capacity_sweep_direct_live_v1.json`.

C7-S24 local dynamic-index is complete: STL/LDL local traffic is exposed, replay/NCU match at 72,655 instructions, and simulator lower 1,443,183 exceeds NCU median 1,243,714 by 199,469 cycles (16.04%). See `docs/sm86_c7_simbench_local_dynamic_index_array_direct_live_v1.json`.

C7-S25 local volatile-array is complete: STL local traffic is exposed, replay/NCU match at 29,017 instructions, and simulator lower 524,608 exceeds NCU median 422,474 by 102,134 cycles (24.18%). See `docs/sm86_c7_simbench_local_volatile_array_direct_live_v1.json`.

C7-S26 register-spill local memory is complete: -maxrregcount=32 exposes 424-byte STACK/STL traffic, replay/NCU match at 271,680 instructions, and simulator lower 30,010 exceeds NCU median 23,073 by 6,937 cycles (30.08%). See `docs/sm86_c7_simbench_register_spill_local_mem_direct_live_v1.json`.

C7-S27 atomic max histogram is complete: deterministic RED.MAX histogram checks pass, replay/NCU match at 42,874 instructions, and simulator lower 436,096 is below NCU median 568,408 while tails remain open. See `docs/sm86_c7_simbench_global_atomic_max_histogram_direct_live_v1.json`.

C7-S28 atomic max return is complete: disjoint return sinks and deterministic final max histogram checks pass, replay/NCU match at 46,882 instructions, and simulator lower 916,220 is below NCU median 1,060,648 while tails remain open. See `docs/sm86_c7_simbench_global_atomic_max_return_used_direct_live_v1.json`.

C7-S23 DRAM stream stride is complete: replay/NCU match at 1,716,864
instructions, while simulator lower 550,005 exceeds NCU median 419,040 by
130,965 cycles (31.25%). See
`docs/sm86_c7_simbench_dram_stream_stride_mix_direct_live_v1.json`.

The `async_visibility_phase` probe now independently recomputes and checks its
returned lane-0 token from the deterministic source initialization; it no
longer prints an unconditional correctness boolean. Fresh direct runs for
cache-all compact, cache-all conflict, and streaming severe all pass token and
interval-sum checks. Prior phase captures are historical because the binary
changed.

The corrected phase binary also completed a fresh NVBit/replay/five-NCU path;
its independently checked direct token is correct, simulator endpoint is
18,305 cycles, and NCU median is 16,658 (9.89% APE). The result-store’s
writeback/coherence tails remain open. See
`docs/sm86_completion_frontier_c3_async_phase_corrected_live_v1.json`.

The resident-device message trace has a same-input service differential in
`docs/sm86_completion_frontier_c3_message_service_differential_v1.json`:
explicit model-local queues complete 3 writeback sectors and 2 coherence
transactions, but both dirty-writeback/coherence tail flags remain true. This
demonstrates queue drain without manufacturing a semantic upper.

A matched one-warp `c00`/`c11` differential is retained in
`docs/sm86_completion_frontier_c3_async_endpoint_differential_warp1_v1.json`.
Five fresh NCU medians are 7,194 and 14,668 cycles (7,474-cycle increment),
while repaired simulator endpoints are 5,995 and 17,394 (11,399-cycle
increment); direct unprofiled `clock64` repeats give a 13,386-cycle
increment. This does not identify the missing async mechanism: the simulator
lies between the direct-clock and NCU differentials, so no scalar parameter is
changed from this comparison.

An independent cache-all source-only layout pair is now recorded in
`docs/sm86_completion_frontier_c3_async_destination_layout_pair_v1.json`.
Compact/conflict variants retain the same binary, depth, route, and launch;
destination wavefronts change 2,048→4,096. The simulator predicts a 4,096-cycle
delta and five fresh NCU medians show 4,070 cycles (26-cycle, 0.64% delta
error). Absolute endpoints remain 4.91%/3.87% high, so this is narrow
model-local differential evidence, not a general hardware upper or predictor.

## Retained measurements

2026-09-30 RTX 3090 Ti campaigns used driver 615.71.09, 1350/10501 MHz, P0,
and NCU clock control `none`:

| Cache policy | Workloads | Median NCU duration | Median GPC cycles |
|---|---:|---:|---:|
| `all` | 15/15 | 10,080 ns | 13,572 |
| `none` | 15/15 | 8,704 ns | 11,709 |

These are measurements, not prediction-error results; campaign raw outputs remain under results/.

## Navigation

[Model contract](PREDICTIVE_CLOSURE.md) · [Configuration](archprofile.md) ·
[Runbook](runbook_host.md) · [Checks](testing.md).
