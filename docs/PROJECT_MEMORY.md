# Continuation memory

Updated 2026-10-03. Start here, then [STATUS](STATUS.md); read only the
contract/source needed for the task.

## Research scope

Strictly forbidden: pursuing goals or exact proprietary mechanisms beyond normal academic PhD-level research; improve accuracy through analytical or statistical modeling and treat unobservable internals as black boxes.

**Permanent alpha policy:** M5/M6 authority, receipt, audit, provenance,
hashing, sealing, approval, freeze, admission, and validation workflow
artifacts are forbidden. Never restore them under a new name or make them a
precondition for a run or commit. Keep only local runtime safeguards that are
needed to avoid crashes or nonsensical model output.

The obsolete M5/M6 and CUTLASS-successor authority/receipt campaign surfaces,
schemas, records, and dedicated tests are deleted; do not recreate them.

Objective: fast SM86 performance prediction, evaluated against independent NCU
measurements. Never use a scored workload's NCU values as features, calibration
labels, or implementation-selection criteria. Keep simulator frontiers separate
from profiler measurements; black-box models are assumptions, not observations.

## Current state

- Branch: `GPUSim_main`. Native path: NVBit/SASS → enhanced protobuf →
  `sim-trace` → `sim-sm` / `sim-mem` / `sim-engine` → `sim-run`.
- `simt_u32_v1` and `simt_its_v1` are separate custom/model-relative frontends.
- `--latent-envelope --predictive-service-profile FILE` adds abstract resource
  demand/service intervals, composes a model-local service frontier, emits
  cache-topology summaries, and produces an NCU candidate projection without
  changing default replay.
- The service profile is an uncalibrated example. Prediction plumbing and unit
  tests do not establish cycle accuracy, empirical coverage, or generalization;
  active unknown completion tails now propagate unknown endpoint uppers.
- `sim-semantics` now has a typed region contract and native-boundary
  comparator. The opt-in SM86 reference trace records exact fallback when a
  macro does not provide complete typed state; a matching descriptor recurrence
  alone is not a native certificate.
- The v2 independent-probe fitter and 15-workload/six-family
  calibration/development/held-out NCU join now run post-prediction. All real
  native rows currently have open upper bounds; finite coverage, native
  preserved compression, and hardware accuracy remain unproven.
- Second-round R1 now uses v2 prediction/observation target contracts,
  rejects family-crossing splits, separates finite interval coverage from open
  refusal, and reports execution-step reduction separately from record
  coverage. These are evidence-hygiene changes, not accuracy evidence.
- R2 is complete at the model-executor level: `sim-sm::ModelExecutor` now has an explicit
  `MacroTransition`/`MacroEffect` seam. The model-local descriptor probe
  installs one independently projected scalar transition (16 detailed steps
  to one macro dispatch) and resumes a detailed suffix; pending pipeline work
  rejects the transition. This is not native `SmCore` execution.
- R3 is complete at the model contract: eligibility requires contiguous
  independent witness facts with an allowed origin; the compact native path
  supplies residency only and therefore refuses rather than self-certifying.
  Multi-warp/global-order cases remain conservative refusal, and same-cycle
  effect ordering is checked independently of global sequence monotonicity.
- R5 is complete for a closed native issue-model helper seam only: tests
  compare a scalar macro entry/exit with detailed replay, while real SM86
  comparator-selected dispatch remains open after a bounded native
  queue/response-owner join failure. The pure-scalar SM86 probe rejects
  front-end state as not closed and remains exact fallback.
- The reference issue actor has a closed scalar seam, and P2 exercises the
  same reference-feature event loop with a synthetic complete witness; the
  real SM86 reference probe/corpus still selects exact replay.
- The closed native issue-model seam is exercised with a two-instruction scalar
  macro and detailed suffix; front-end-enabled launches remain exact fallback.
- Service profiles now mark capacity-adjusted throughput slopes and carry
  explicit residual/replacement/independent service roles plus a serial queue
  projection; the v2 probe fitter requires explicit dependent latency and
  independent throughput, retains queue-depth/capacity interventions as
  explicitly unmodeled (no queue equation), and refuses aggregate-only
  latency identification; occupancy/contention rows remain explicitly
  unmodeled.
- P4 also removes aggregate latency inference from the v1 compatibility
  fitter and scores held-out throughput against the stored interval endpoints
  rather than only the median; interval coverage is explicit and queue
  depth/capacity remain unmodeled interventions.
- P5 strengthens evaluation identity: prediction documents require exact
  workload-to-trace/control/architecture-card/pchase-card/launch bindings;
  manifest-built NCU rows retain the same semantic binding when available, and
  the join reports whether both documents were binding-checked. Historical
  identity remains manifest-labeled rather than independently verified.
- The design evaluator now makes the negative boundary primary, reports
  accepted-region coverage separately from dispatch/step counters, and records
  when compatible prior-art cost inputs were not measured rather than inventing
  a comparison.
- Cache candidates preserve warp-local instrumentation order only when the
  producer supplies it; exact cache history now additionally requires temporal
  order, fence scope, line generation, owner/read-from, and completion.
  Unresolved domains keep the separate cache frontier lower-only/unknown
  rather than adding a bottleneck claim.
- R8.1 immediate soundness guards are complete: model certificates bind to the
  full consumed descriptor region before mutation; inconsistent grouped queue
  assignments cannot close an unknown tail; predicted rows must bind their
  interval to a typed observer projection; replacement admission and service
  arithmetic are checked; scalar sequence arithmetic checks intermediate
  additions. These are direct API/projection guards, not native speedup.
- R8.2 real-corpus native dispatch remains open. A bounded native canary
  exposed a response-owner/queue join failure before a real production macro
  could be consumed; separately, P2 has a synthetic complete-witness
  reference-feature event-loop differential. The retained front-end-enabled
  corpus remains exact fallback.
- R8.3 is complete as a conservative typed eligibility rule: latency >1,
  pending queue tokens, front-end/scalar-pipeline state, and nonempty
  instruction buffers refuse before any issue-only summary is treated as
  native. Model-local latency certificates remain valid at their own evidence
  level; native latency/completion modeling is not claimed.
- R6 is complete: `docs/sm86_reification_corpus_v1_20261002.json` defines a
  15-workload, six-family SM86 corpus with no family crossing; prediction,
  observation, and join artifacts are retained under
  `docs/review_round3_20261002/`. The pre-observation output uses the declared
  observer projection, records detailed replay plus a latent sidecar, and
  reports zero native macro dispatches/step reduction. The post-prediction join
  enforces row-level metric/endpoint/clock/declared-hardware identity and
  reports open intervals separately from finite coverage. Observation rows are
  explicitly `manifest_labeled_historical`; the builder does not independently
  verify live GPU identity.
- R9.0 repaired the bounded reference-canary response-owner/queue projection
  failure by marking due-owner/queued-response mismatches as unresolved
  ownership. The rerun terminates with 10,032 reference records and exact
  fallback throughout; R8.2 production macro dispatch remains open.
- R9.1 attempted the guarded witness/dispatcher path. The native canary has no
  complete independent witness, so it records zero macro dispatches and exact
  fallback throughout. P2 closes a separate synthetic reference-feature
  fixture, but it is not a silicon or front-end completion claim.
- R9.2 completed the two-pass witness audit negatively: no complete typed
  pre/effect/post pair exists in the bounded canary, so self-derived witness
  construction is rejected. The required positive falsifier remains open.
- R10.0 completed a live 168-block SM86 development campaign: 10/10 cases ran
  native correctness, NVBit tracing, simulator replay, analyzer, and three
  fresh post-prediction NCU repetitions. All intervals remained open, analyzer
  candidates and native macro dispatches were zero, and no speedup claim is
  made. Aggregate result: `docs/review_round7_20261002/`.
- R10.1 completed a live writeback campaign: 2/2 large dirty-stream cases ran
  native correctness, NVBit trace, replay/analyzer, and three fresh NCU
  repetitions. Writeback completion remains open/unavailable; no fitting or
  native speedup claim is made. Aggregate result:
  `docs/review_round8_20261002/`.
- R10.2 completed a live cache-policy contrast on two large cases with three
  fresh `cache-control=none` repetitions after all-cache measurements. It is
  observational only; no fitting, cache-causality, or speedup claim is made.
  Result: `docs/review_round9_20261002/`.
- R11.0 completed detailed-versus-fast-frontier cost measurements on all 15
  retained traces: 5,130,860 records, zero accepted candidates, zero native
  macro dispatches, and explicit wall/RSS/refusal accounting. Result:
  `docs/review_round10_20261002/`.
- P2 now has a positive **synthetic reference-feature event-loop fixture**:
  an independently supplied complete scalar witness reaches the real
  `SmWake` dispatch branch, reserves internal event sequence positions,
  updates the narrow issue/power projection, resumes a detailed suffix, and
  matches detailed cycles/instruction count. A perturbed typed exit falls back
  before mutation. This closes the event-loop plumbing seam only; the retained
  front-end/cache SM86 corpus still has no complete independent witness and
  records zero native dispatches.
- P3 closes the narrow native state boundary by refusal: latency >1, release or
  completion queues, front-end/cursor state, pending responses, barriers,
  local-completion history, unprojected timing tails, async/cache fills, and
  unmodeled tails are rejected before macro construction/dispatch. The actor
  validates every descriptor and continuation lookup before mutation; an
  independently detailed `SmCore` effect/exit ledger drives the positive
  synthetic witness.

## Next useful work

The first-round implementation plan is retained in
[`sm86_reification_refactor_20261001.md`](../plans/sm86_reification_refactor_20261001.md).
The active second-round program is
[`second_round_refinement_20261001.md`](../plans/second_round_refinement_20261001.md),
with **R1 semantic closure and evidence hygiene complete**, **R2 executable
macro transition seam complete**, and **R3 independent witness/order closure
complete**, **R4 service/queue semantics complete**, **R5 native state
refinement complete for the closed issue-model seam**, and **R6
corpus/observer evaluation complete**. R1 repaired unknown/zero-demand service
arithmetic, explicit-endpoint propagation, event ordering, compression
vocabulary, and split/NCU identity checks. R2 made a nonidentity macro
decision operational and resumes detailed replay; that model-local seam now
exists, while real native `SmCore` traces remain conservative fallback. R3 is separating witness
extraction from descriptor eligibility and now refuses stale/self-witnesses;
R4 now separates service roles and queue/assignment uncertainty. R5 validates
a closed native issue-model helper boundary, and P2 exercises the production
reference-feature event loop on a synthetic complete scalar witness; real
front-end-enabled traces still record exact-replay plans. Current
rankings, open
intervals, and descriptor/macro probe remain negative/diagnostic evidence and
black-box internals are not execution prerequisites. R7 rebuttal/
claim-discipline is complete; R8.0 review intake, R8.1 soundness guards, R8.3
native eligibility, R8.4 service identifiability, R8.5 measurement identity/
cost utility, and R8.6 synchronization are complete; R8.2 remains open only
for a real retained SM86 witness after the bounded queue/response failure. The
P0–P6 completion program is closed with the remaining native/hardware
falsifiers explicit; use
[`third_round_rebuttal_20261002.md`](../plans/third_round_rebuttal_20261002.md)
for the prior evidence map and
[`latest_review_refinement_20261002.md`](../plans/latest_review_refinement_20261002.md)
for the active stage ledger and remaining falsifiers.

The next workstream is the accuracy-first NCU completion program in the same
latest-review plan, beginning at A0 measurement-contract definition. Its
primary output is independent NCU agreement; simulator wall-clock runtime is
secondary and no compression/speedup metric is used.

A0 is complete: `docs/sm86_ncu_accuracy_protocol_v1.json` declares the exact
NCU endpoint/replay/cache/device/repetition contract, required identity/build
fields, prediction-before-observation artifact rules, and six-family split
map. The protocol validator and schema tests pass; no new NCU data was
collected. A1 is next: independently verify the actual device/replay/build
identity before the first new campaign.

A1 is complete: `tools/capture_sm86_ncu_identity.py` now performs metadata-only
GPU/NCU/tool/build/input checks and emits a schema-validated identity artifact.
The current host query finds the expected RTX 3090 Ti/SM86/NCU metric, but the
dirty worktree is correctly labeled ineligible; no workload profile was
collected. A2 is next: deterministic baseline repetitions and dispersion.

A2 is complete: `tools/preflight_sm86_ncu_campaign.py` validates the exact
replay/cache/clock/repetition/prediction-order contract and reports bounded
termination/dispersion summaries without launching NCU. The current historical
manifest is correctly not ready for the new five-repetition contract. A3 is
complete: source-path and family/split closure pass for the historical roster,
five expansion slots are declared, and preflight requires a valid prediction
before collection. A4 is complete: the protocol-aligned 15-row prediction
artifact is prepared before any observation, source-bound, five-repetition
targeted, and free of NCU values. The service fitter remains probe-only with
explicit latency/throughput intervals and queue/occupancy/contention
unmodeled; the supplied profile is explicitly unvalidated. No new NCU
observations were collected or scored. A5 is now complete: five fresh
application-replay/all-cache/no-clock-control repetitions ran for all 15
workloads after clean identity/preflight. The corrected launch-skip setting is
0; the first historical-skip attempt produced no profiled kernels and is
retained as failed raw execution evidence. The 75 observations are separate
from predictions and all intervals remain open: finite accuracy is not
applicable, with lower consistency 0.667/1.000/0.750 across
calibration/development/held-out. A6 is next: separate intervention
falsification.

A6 is complete: a predeclared cache `all→none` contrast ran five repetitions
for all 15 workloads and a clock `none→base` contrast ran five repetitions for
two cases, with matching mechanism counters and separate prediction/observation
joins. Simulator projections were invariant; directions are observational only
and all intervals remain open. Cache-none lower consistency was
0.000/1.000/0.625 and clock-base was 1.000 for its two rows. No causal claim
was made. A7 is next: real selective-path witness attempt.

A7 is complete: the current release reference path ran on a retained real
SM86 trace under the verified identity/endpoint contract without a self-derived
witness. It terminated with 10,033 records before the footer, zero complete
typed boundaries, zero native macro dispatches, and exact fallback. The retained
result is `docs/sm86_accuracy_a7_real_selective_result_v1.json`. The final
claim remains unknown-preserving typed closure with exact fallback; real native
selective dispatch and finite hardware accuracy remain explicit falsifiers.

A8.0–A8.5 of the accuracy-first live-campaign foundation are complete. The
new bounded workload runner (`tools/run_sm86_workload.py`) records explicit
pass/OOM/OOD/timeout/launcher outcomes, performs a free-memory preflight, and
terminates child process groups. Its seven focused tests pass. All 15 active
portable-LCD binaries were rebuilt from current source; the retired
`--enable-cupti-activity` interface is absent from rebuilt binary help, and
the architecture/SASS roster verifier passes. All 15 correctness smoke rows
passed sequentially with no OOM/OOD/timeout or silent shape substitution.

The fresh A8 vertical slice (`docs/sm86_a8_vertical_slice_v1.json`) completed
native correctness, NVBit tracing, simulator replay, prediction-before-NCU,
and five independently identified NCU repetitions. The trace has four CTA
payloads and 1,272 instructions; simulator replay sees 556 modeled records
and leaves dirty-writeback/coherence tails open. NCU GPC cycles are
5684/5490/5514/5524/5517 (median 5517); mechanism counters are retained
separately. The prediction contains no NCU values, has lower 3023 cycles and
an open upper, and the post-prediction join reports lower consistency 1.0 with
finite coverage not applicable. A8.6 handoff synchronization is complete; A8
is complete and A9 is not started. Real native dispatch and finite hardware
accuracy remain falsifiers.

## Current continuation (2026-10-03)

Use `plans/sm86_completion_frontier_and_native_dispatch_20261002.md`.
No goal tools; track milestones in the plan. **C0-R is complete**: repair the
2026-10-02 unsupported tail closure before any new finite-upper claim.
`close_explicit_hierarchy_tails` inferred completion from one timestamp and
coarse topology. It is being removed; unknown tails must stay open. The
hand-chosen 1600-cycle padding / store-2 result is not a valid calibrated
family-held-out accuracy result. C1 async compared a 64-group trace with
16384-group NCU; C1 message-passing used managed memory and polling. Neither
establishes the claimed matched visibility comparison. Raw results remain
retained; earlier C1/C3 completion claims are superseded by these corrections.

Separate model-completion state, empirical NCU prediction intervals, and
native model-transition equivalence. Private silicon cache tokens are not a
prerequisite to statistical prediction or a memory-free native differential;
a finite empirical interval must not clear semantic unknown tails.
C4/C5 exploratory slice now has a finite empirical interval: group16/32 calibration predicted group128 37737..38655, covering all 5 NCU repeats (median/point error 0.396%). The simulator semantic lower is 168898 and tails remain open, so this is not a completion upper. C1-R matched async/control differential now proves a model mismatch: NCU delta 13760 vs simulator lower-bound delta 78852; C1-R matched async recapture now proves a model mismatch (same 64-group binary: simulator lower 84866 vs NCU median 20946); resident-device message passing remains an open lower-bound case (1753 vs 4321). Neither is a finite-accuracy result. C3 now widens async timing frontiers and refuses predicted lower/upper after the matched mismatch; next: same-binary/shape fresh controls + NVBit/replay/NCU for C1-R; implement actual completion obligations before calibrated independent accuracy scoring.

The observed-fact layer of C2 is now complete: the independent raw-protobuf /
static-SASS extractor reports deterministic same-warp order edges, static
function-id/PC joins, protocol counts, masks, allocation ids, and explicit
absence of global instrumentation order. The C3 source-token repair charges
one cacheable LDGSTS return-service token per coalesced warp instruction,
rather than per sector. Fresh post-change group-sweep evidence is retained in
`docs/sm86_completion_frontier_c3_async_source_token_repair_v1.json`; the
group-128 model remains 46,978 versus NCU median 38,348, so C3 is still open.
A fresh store-free source-only NVBit/NCU capture is a narrow positive
model-completion case (13,235 modeled cycles versus NCU median 12,616) in
`docs/sm86_completion_frontier_c3_source_only_live_v1.json`; it does not expose
hardware ownership/read-from/fence-scope/completion tokens and does not close
the writeback/coherence problem.

A matched one-warp c00/c11 differential is retained in
`docs/sm86_completion_frontier_c3_async_endpoint_differential_warp1_v1.json`:
five fresh NCU medians are 7,194 and 14,668 cycles, repaired simulator
endpoints are 5,995 and 17,394, and direct unprofiled clock64 repeats give a
13,386-cycle increment. This separates endpoint disagreement from multi-warp
arbitration but does not identify a scalar async parameter; no further timing
fit is justified from it.

An independent cache-all source-only compact/conflict layout pair is now
retained in `docs/sm86_completion_frontier_c3_async_destination_layout_pair_v1.json`.
The same binary/depth/route/launch changes destination wavefronts 2,048→4,096;
the repaired simulator predicts a 4,096-cycle delta and five fresh NCU medians
show 4,070 cycles (26-cycle, 0.64% delta error). Absolute endpoints remain
4.91%/3.87% high and hardware completion tokens remain unknown, so this is a
narrow model-local differential result. A second C4/C5 layout holdout is now
separated into prediction and observation documents: compact/conflict NCU
calibration predicted severe at 24,476.2..24,902.2 cycles before profiling;
five severe NCU repeats have median 24,855, interval coverage true, and point
APE 0.667%. This remains a controlled observer result, not a semantic upper or
global accuracy claim. A real same-shape fixed-latency scalar slice now has
correctness plus five NCU captures, but nonfull active/predicate populations
force zero native dispatch; simulator 3,578 versus NCU median 6,812 is 47.5%
low. The old 8,192-generation NCU rows are excluded. The async visibility
phase probe was also repaired to
recompute its deterministic lane-0 token on the host before reporting
`correct`; compact/conflict/streaming-severe direct runs pass the check. Older
phase captures are historical because the binary changed.

The corrected phase binary has one fresh NVBit/replay/five-NCU path with an
independently checked direct token. Its simulator endpoint is 18,305 cycles
versus NCU median 16,658 (9.89% APE); result-store writeback/coherence tails
remain open.

The same resident-device message trace’s service differential completes 3
model-local writeback sectors and 2 coherence transactions but leaves both
dirty-writeback/coherence tail flags true, proving queue drain is not hardware
completion.

C7’s first population slice is complete for all 15 portable-LCD workloads:
native timing, NVBit, two simulator replays, and five NCU repetitions per row
finished in 389 seconds with zero OOM/OOD/timeouts/failed stages. Raw data is
`results/alpha/c7_full_live_20261003_run/` and the tracked summary is
`docs/sm86_c7_full_live_campaign_v1.json`. The full repository population is
not complete yet; probes/SOL/other suites and correctness joins remain to be
inventoried. Portable replay tails remain open and native macro dispatch was
not requested.

The C7 inventory lists 15 portable rows, 31 SM86-capable rows among 92
SimBench rows, 42 research-probe directories, and 203 SOL baseline problems.
Portable timing paths and twenty-six direct SimBench slices have completed:
`fp32_fma_dep_chain`, `cp_async_shared_tile_copy`, `global_atomic_histogram`,
`global_vectorized_128b_copy`, `bf16_hmma_gemm_traceable`,
`global_scalar_32b_copy`, `global_atomic_return_used`,
`fp16_hmma_gemm_traceable`, `global_vectorized_64b_copy`,
`shared_bank_conflict_stride`, `l2_thrash_pointer_chase`,
`l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, and
`warm_cold_pair_relaunch`, and
`carveout_sensitivity_probe`, and `working_set_capacity_sweep`, and `dram_stream_stride_mix`, and `local_dynamic_index_array`, and `local_volatile_array`, `register_spill_local_mem`, and `global_atomic_max_histogram`, and `global_atomic_max_return_used`, and `rmsnorm_fwd_bf16`, and `global_atomic_and_histogram`. These check outputs at the captured shape; the FP32
direct slice is timing-only (its separate correctness-shape run does not
validate that captured shape). All twenty-six retain open tails and zero native
macro dispatch. The inventory is
`docs/sm86_c7_workload_population_inventory_v1.json`.

The full-shape `dram_streaming_dot_product` slice is additionally recorded in
`docs/sm86_c7_simbench_dram_streaming_dot_product_direct_live_v1.json`:
argument/launch/allocation guards and a per-block roundoff-bounded reference
check pass, Compute Sanitizer reports zero errors, NVBit/replay cover all 128
blocks with 297,600 warp instructions, and five NCU gpc-cycle values are
7714/7579/7548/7524/7562 (median 7562). Its simulator lower bound is 5345,
so lower-bound consistency is true; dirty-writeback/coherence and native
dispatch remain open.

The bounded `global_atomic_and_histogram` slice is recorded in
`docs/sm86_c7_simbench_global_atomic_and_histogram_direct_live_v1.json`. The
repaired driver checks the identity output and deterministic bitwise-AND
histogram, emits RED.E.AND, and avoids the old billion-operation timing shape
that could cause OOM or impractical traces. Replay has 42,874 warp records;
NCU cycles are 568244/568570/568083/567453/568196 (median 568196), while the
simulator lower is 436096. Dirty-writeback/coherence and native dispatch remain
open.

The full-shape `rmsnorm_fwd_bf16` slice is recorded in
`docs/sm86_c7_simbench_rmsnorm_fwd_bf16_direct_live_v1.json`. Its repaired
driver validates BF16 inputs element-by-element, uses one 256-thread block per
row, and has explicit OOM/OOD/launch checks. The `rows=1024, hidden=4096`
capture passes Compute Sanitizer; NVBit/replay cover 1,024 blocks with
4,235,264 warp instructions, and five NCU gpc-cycle values are
48453/48106/48931/50179/48481 (median 48481). The simulator lower is 43214,
so lower-bound consistency is true; dirty-writeback/coherence and native
dispatch remain open.

Two tensor rows (INT8 IMMA and TF32 HMMA) pass host correctness, NVBit, and NCU
but are blocked before simulator timing because their SM86 pipeline models are
missing. Ten memory/control rows remain real lower-bound falsifiers; three L2
examples are pointer chase (59,545 vs NCU median 49,580), persisting window
(31,173 vs 29,846 after the sink repair), and prefetch hint (46,874 vs
40,598). The prefetch binary emits
five static `CCTL.E.PF2` instructions; cache effect and completion remain
unobservable. The corrected shared-bank shape uses SM86 dynamic shared-memory
opt-in (65,536 bytes against default 49,152 and opt-in 101,376), superseding
its old false OOD disposition.

The research-probe population was re-enumerated from the current tree: 42 directories are present (31 Makefile CUDA probes, 6 source/preserved-binary rows, and 5 analysis-only rows). The bounded inventory is `docs/sm86_c7_research_probe_inventory_v1.json`; all 31 Makefile probes build successfully with the pinned CUDA 13.2/GCC 14/sm_86 matrix in `docs/sm86_c7_research_probe_build_matrix_v1.json`. Matrix campaigns are complete for issue_port_control (68), async_barrier_overlap (36), store_ack_frontier (30 arms), memory_retirement_service (102 with eight crosswalk exclusions), barrier_pipeline (33), and l1tex_admission_backpressure (100, all lower-bound-consistent). An explicit warmup/target l2_queue_frontier vertical cell also passes correctness/NVBit/replay/five-NCU with lower-bound consistency in `docs/sm86_c7_research_probe_l2_queue_frontier_live_v1.json`; remaining probe families are not yet claimed.

The remaining BF16 cuBLAS control row is now also dispositioned in
`docs/sm86_c7_simbench_vendor_gemm_cublas_blocked_live_v1.json`: host BF16
reference validation and Compute Sanitizer pass, five NCU rows observe the
dynamic `cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_32x1_nn_align8`
kernel (1456 warp instructions; median 5347 cycles), but the repository
control sidecar has zero functions and NVBit refuses the tensor shape before
publishing a complete trace. All 31 SM86 SimBench roster rows therefore have
either a direct live artifact or an explicit replay blocker; this does not
close the separate research-probe or SOL populations.

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
and falsifies the lower-bound consistency (211,002 simulator lower vs 178,521
NCU median). C7-S20 vendor SGEMM is an explicit replay blocker: cuBLAS emits
ampere_sgemm_128x32_nn from libcublasLt.so without a stable static control
sidecar; 29,752 traced records cannot be crosswalked to 8,784 NCU instructions.

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

## Current probe continuation — 2026-10-04

C7-S33M was corrected: the NCU SourceCounters rows for `MEMBAR.SC.SYS`,
`ERRBAR`, and `CCTL.IVALL` each report one warp execution. Their separate
warp-stall sample columns are zero. The aggregate NCU instruction delta
(1113 versus 1101) therefore remains an unresolved counter/trace crosswalk
issue; the corrected diagnostic is
`docs/sm86_c7_memory_retirement_system_fence_crosswalk_diagnostic_v1.json`.

C7-S33S hardened `workloads/research_probes/l2_queue_frontier`: malformed or
overflowing shapes and allocations above half of free device memory are
rejected before launch, output sentinels are checked, and the host recomputes
the lane-zero checksum and timestamp bounds. A valid unique and reuse smoke
plus Compute Sanitizer pass completed. A fresh same-binary capture is recorded
in `docs/sm86_c7_research_probe_l2_queue_frontier_live_v2.json`: target replay
and NCU each report 70,560 warp instructions; five NCU gpc-cycle values are
7714/7784/7729/7666/7663 (median 7714), and the simulator lower is 5751.
Dirty-writeback/coherence completion remains unknown; simulator runtime is
recorded as 0.150273283 seconds for this replay, with no speedup claim.

C7-S33T completed a hardened L1TEX arrival/replay vertical slice. The driver
now rejects malformed or oversized shapes, checks allocation arithmetic and a
512-MiB/half-free-memory budget, and reports whether dependent work is
externally observed. The stale separate trace-twin interface was removed from
the campaign; one rebuilt executable served correctness, NVBit, replay, and
NCU. The `store,line,84,8,8,32,0,1` cell passed seven host trials, a reduced
Compute Sanitizer run, 84-block NVBit capture, target replay, and five NCU
repetitions. Replay and NCU both report 809,088 warp instructions; NCU gpc
cycles are 32085/31590/32333/31748/31518 (median 31748), while the simulator
lower is 8486. Dirty-writeback/coherence completion remains unknown and no
point-accuracy or speedup claim is made. Evidence:
`docs/sm86_c7_research_probe_l1tex_arrival_replay_service_live_v1.json`.

C7-S33U completed the hardened `sfu_frontier` dependent-MUFU vertical slice.
The driver now uses strict parsing, a 262,144-iteration bound, checked
allocation arithmetic, a 64-MiB/half-free-memory budget, output sentinels, and
timestamp/checksum-presence checks. The `84,4,1024,dependent` cell passed seven
host trials, reduced Compute Sanitizer, four-launch NVBit capture, target-4
replay, and five launch-3-skipped NCU captures. Replay and NCU both report
1,389,024 target warp instructions; NCU gpc cycles are
34274/34282/34292/34297/34274 (median 34282), but the simulator lower is
38099. This is a retained lower-bound falsifier, not a tuning opportunity; no
SFU latency is fitted into a scored workload. Evidence:
`docs/sm86_c7_research_probe_sfu_frontier_live_v1.json`.

C7-S33V completed a same-binary fixed-latency result-queue collision slice.
The campaign no longer mixes 64-generation trace cubins with 8,192-generation
NCU cubins; one patched collision/separated cubin is used across stages. The
CUDA-driver harness now has strict parsing, bounded allocations, sentinel output
checks, elapsed-sample checks, and a deterministic sink oracle. The collision
cell (`warps=4, generations=8192`) passed seven host trials, Compute
Sanitizer, full NVBit coverage, deterministic replay, and five NCU runs. Replay
and NCU both report 524,416 warp instructions; NCU gpc cycles are
307642/307646/307655/307628/307679 (median 307646), while simulator lower is
287635. No result-queue latency is activated or fitted; dirty-writeback/
coherence completion remains unknown. Evidence:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_live_v1.json`.

C7-S33W completed the matched result-queue collision/separated differential at
`warps=4, generations=8192`. Both treatments use one patched cubin each across
correctness, NVBit, replay, and NCU, with identical driver, clocks, cache
policy, and five-repeat protocol. Collision NCU median is 307646 with
simulator lower 287635; separated median is 340424 with simulator lower 353171,
so separated is a real lower-bound falsifier. The NCU treatment delta is 32778
cycles while the simulator lower-bound delta is 65536; this is mechanism-local
evidence, not a fixed latency. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_pair_live_v1.json`.

C7-S33X completed the held-out width-8 result-queue collision/separated pair.
The live trace exposed a finite simulator preflight ceiling: each block has
1,048,832 decoded instructions, so the wire-preflight limit was raised from
1,000,000 to 2,000,000 and the compact route index widened by one bit. The
same captures then replayed successfully. Both arms passed seven host trials,
Compute Sanitizer, full NVBit capture, deterministic replay, and five NCU runs
with matching 1,048,832 target warp instructions. Collision NCU median is
447092 with simulator lower 320466; separated median is 438935 with lower
369618. Both lowers are individually consistent, but hardware separated is
8157 cycles below collision while the simulator lower delta is +49152;
separated repeats are high-dispersion. No result-queue latency is activated.
Artifact: `docs/sm86_c7_research_probe_fixed_latency_result_queue_width8_pair_live_v1.json`.

C7-S33Y completed the width-12 result-queue collision/separated pair under
the two-million-instruction decoder ceiling. Each block has 1,573,248 decoded
instructions and replays without truncation. Both arms pass seven host trials,
Compute Sanitizer, full NVBit capture, deterministic replay, and five NCU runs
with matching target instructions. Collision NCU median is 589202 with lower
527551; separated median is 561568 with lower 533490. Both lowers are
individually consistent, but hardware separated is 27634 cycles below collision
while simulator lower delta is +5939; repeats remain dispersed and no latency
is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_width12_pair_live_v1.json`.

C7-S33Z completed the declared width-16 result-queue collision/separated
pair. Each block has 2,097,664 decoded instructions; the prior two-million
instruction/decoded-node ceilings rejected the exact trace, so they were raised
once to 3,000,000 instructions and 4,000,000 nodes, with a 22-bit compact
route index. Both arms then passed seven host trials, Compute Sanitizer, full
NVBit capture, deterministic replay, and five NCU runs with matching target
instructions. Collision NCU median is 708805 with lower 579430; separated median
is 693935 with lower 654998. Both lowers are individually consistent, but
hardware separated is 14870 cycles below collision while simulator lower delta
is +75568; no result-queue latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_width16_pair_live_v1.json`.

C7-S33AA completed a hardened completion-distance split vertical slice. The
stale 64-generation trace pair was removed; one patched 8,192-generation split
cubin serves correctness, NVBit, replay, and NCU. The driver uses strict
parsing, bounded allocations, sentinel output checks, elapsed-sample checks,
and the deterministic sink oracle. The `distance=1, split, warps=4` cell passed
seven host trials, Compute Sanitizer, full NVBit capture, deterministic replay,
and five NCU runs. Replay and NCU both report 622,832 target warp
instructions; NCU median is 586172 and simulator lower is 566204, so lower
consistency is true. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_live_v1.json`.

C7-S33AB completed the matched completion-distance bank3 arm. One patched
8,192-generation cubin serves correctness, NVBit, replay, and NCU. The
`distance=1, bank3, warps=4` cell passes seven host trials, Compute
Sanitizer, full NVBit capture, deterministic replay, and five NCU runs with
622,832 matching target warp instructions. Simulator lower is 566204 and NCU
median is 586129; the fresh split arm median is 586107, a +22-cycle bank3
differential. No completion latency is activated. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_pair_live_v1.json`.

C7-S33AC completed the held-out `distance=2, warps=4` completion pair. One
patched 8,192-generation cubin per layout serves all stages. Both split and
bank3 arms pass seven host trials, Compute Sanitizer, full NVBit capture,
deterministic replay, and five NCU runs with 622,832 matching target warp
instructions. Simulator lower is 336828 for both; split NCU median is 340415
and bank3 median 356791. Both lowers are consistent; the 16,376-cycle bank3
differential is mechanism-local and no latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance2_pair_live_v1.json`.

C7-S33AD is active: run the held-out `distance=4, warps=4` split/bank3 pair
under the same contract.

## Retained evidence

The 2026-09-30 cold/warm campaigns completed 15/15 workloads on RTX 3090 Ti / SM86,
UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09,
1350/10501 MHz, P0. This is historical host identity, not a current-device check.
Raw records: `results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/`.
Preserve datasets, source, tests, and scientific limitations; do not recreate
removed policy text or workflow machinery.
