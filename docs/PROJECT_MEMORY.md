# Continuation memory

Updated 2026-10-06. Start here, then [STATUS](STATUS.md); read only the
contract/source needed for the task.

## Current focus (read before historical results)

- M4 is closed at its scoped 42-directory research-probe population bar.
  The SOL source census is 52 custom CUDA, 14 CUTLASS, 137 pure ATen;
  source/capture/replay blockers remain, not successful executions.
- C7-S34CS reconciles all 14 CUTLASS rows: 11 live/replayable scoped rows
  and three explicit blockers (L1/003 and /037 capture timeouts, Quant/020
  SM120-only). L1/022 TF32 replay is resolved by the exact SM86 capability;
  FIB006 M=6 replays L1-L2 while M=12853 still times out.
- M5-F streaming async observer: pre-target interval 48680.6..49209.4
  covers 5/5 groups-64 repeats (median 48914, point APE 0.0634%); empirical
  only, completion lower 42437 and physical tails open.
- M5-G TF32: corrected numerical per-accumulator probe plus zero-error
  memcheck supports a distinct 32-cycle SM86 HMMA.1688.F32.TF32 initiation
  interval; standalone and all seven L1/022 launches replay in L1-L2/flat,
  with five post-fix NCU sets matching instruction/HMMA counts.
- M5-H/I: C1-R store-tail delta is 2886 modeled cycles versus NCU median
  delta 6080 without fitting a service cost; async source/visibility arms
  share 256 copies, 8192 lane operations, and 2048 committed/waited groups,
  but owner/read-from/generation/fence scope/hardware completion remain absent.
- M5-J: real two-warp continuation has typed entry but refuses before mutation;
  two reference-feature runs are byte-identical, footer-terminated, dispatch zero.
- M5-K: fresh FIB005 M=1/M=32 calibration freezes 242005.0..242187.4;
  untouched M=80 misses all five repeats (median 241722, point APE 0.155%),
  retained without refit; replay is byte-identical at 419104 instructions.
- M5-L/M: frontier reconciliation and blocker consolidation are complete; M5-O also resolves the exact INT8 IMMA replay capability with a shape-specific probe;
  the post-TF32 L1/037 retry again timed out before dynamic_trace.pb; M5-N4 found no additional local SM86 seam. **Next:
  M5-N3 — retain the objective open until external-evidence residuals change.**
- Native capture needs `.toolchain/pb/lib` in `LD_LIBRARY_PATH`; profiler-gated
  runners use `--profile-from-start off --launch-skip 0`. Retain parse warnings
  and actual failures; process exit alone is not capture evidence.
- Local checks: 495 sim-sm + 134 sim-semantics tests, format/diff checks,
  corrected TF32 oracle/memcheck, live M5-F/M5-K replays, and 40 focused Python tests / 703 subtests. Physical completion,
  broad accuracy, broad native continuation, two capture timeouts, and SM120
  execution remain open. Active plan:
  `plans/sm86_completion_frontier_and_native_dispatch_20261002.md`.

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

## Current continuation — M4 closed (2026-10-04)

M4/C7-S33BB is complete. The inventory status is
`48_live_probe_stages_reconciled_m4c_verticals_complete`. Seven previously
uncaptured Makefile families plus five source/preserved-binary families have
same-binary correctness, Compute Sanitizer, NVBit, deterministic replay, and
five independent NCU vertical records. Load-priority and corrected LDGSTS are
retained lower-bound falsifiers; async visibility, address mapping,
launch-cycle, value/control, bypass, scheduler, and coherence rows remain
observational with open semantic tails. LDGSTS process-start and wrong-launch
attempts remain retained. Python-only and host-only wrappers have explicit
non-native dispositions.

M4-E then independently rechecked the closure instead of trusting the prior
label. The barrier and L1TEX matrix campaigns were missing only their
declarative manifests; manifest-only regeneration restored them and the
retained keys match 33/33 and 100/100 cells. One bypass stdout reference was
stale and now points to the retained correctness output. The SFU Makefile had
inherited GCC 16, which CUDA 13.2 rejects; it now pins GCC 14 and a fresh SFU
v2 vertical passes seven correctness trials, full-shape Compute Sanitizer,
NVBit, tail-fact extraction, deterministic replay, and five NCU repetitions.
The fresh SFU lower-bound falsifier is 38099 modeled cycles versus 34257 NCU
median; no timing value was fitted. All 37 Makefile targets build under the
bounded rebuild loop, 42/42 inventory directories and live result paths
resolve, and 135 document/schema pairs validate. This is population closure,
not hardware completion or native dispatch.

C7-S33BD then synchronized the retained build matrix with the current
inventory. Six current Makefile families missing from the old 31-row matrix
received bounded make checks returning zero; the matrix now covers all 37
Makefile probes, with five analysis-only rows and no separate source/preserved
rows. This repairs bookkeeping drift only.

C7-S34A then inventoried the 203 SOL baseline problem directories without
execution: FIB 25, L1 94, L2 82, Quant 2; all have CUDA/host source and
`solution.json`, 201 have an SM86 marker, two do not, and none has a local
prebuilt artifact. This is source inventory only.

C7-S34B source-reviewed one SM86-marked FIB candidate. The existing
`tools/sol_execbench/.venv` supplies Torch 2.12.0+cu130 and SM86 CUDA; explicit
`nvcc -ccbin=/usr/bin/g++-14` produced the extension, BF16 correctness batches
1/2 passed, and Compute Sanitizer reported zero errors. A profiler-delimited
NVBit/control/replay/NCU vertical then passed with 3824 simulator/NCU warp
instructions and a 5374-cycle NCU median; raw tracer units and the enhanced
JIT parser limitation remain explicit. Evidence:
`docs/sm86_c7_sol_candidate_vertical_v1.json`.

C7-S34D repeated the vertical on FIB/problem_002 at hidden size 4096. The
SM86 build/correctness/memcheck smoke, NVBit/control extraction, deterministic
4562-record replay, and five corrected NCU runs passed; NCU median was 7922
cycles with 4562 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib002_vertical_v1.json`.

C7-S34F repeated the custom CUDA vertical on FIB/problem_003 at hidden size
7168. Build/correctness/memcheck, NVBit/control extraction, deterministic
7202-record replay, and five corrected NCU runs passed; NCU median was 10966
cycles with 7202 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib003_vertical_v1.json`.

C7-S34J repeated the custom CUDA vertical on FIB/problem_021 at hidden size
128. Build/correctness/memcheck, NVBit/control extraction, deterministic
154-record replay, and five corrected NCU runs passed; NCU median was 4341
cycles with 154 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib021_vertical_v1.json`.

C7-S34K repeated the custom CUDA vertical on FIB/problem_022 at hidden size
512. Build/correctness/memcheck, NVBit/control extraction, deterministic
562-record replay, and five corrected NCU runs passed; NCU median was 4433
cycles with 562 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib022_vertical_v1.json`.

C7-S34L repeated the custom CUDA vertical on FIB/problem_023 at hidden size
1536. Build/correctness/memcheck, NVBit/control extraction, deterministic
1442-record replay, and five corrected NCU runs passed; NCU median was 6365
cycles with 1442 instructions matching simulator records. Dirty-writeback and
coherence tails remain open; the enhanced JIT parser limitation, raw tracer
unit, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_candidate_fib023_vertical_v1.json`.

C7-S34M1 completed the source review for FIB/problem_024 RMSNorm at hidden
size 2048. The source contains an SM86 compile marker, bounded
`main.cpp::run` entry point, explicit BF16/FP32/EPS contracts, and a separate
FP32 reference; the permitted Torch 2.12.0+cu130/GCC14 environment sees the
RTX 3090 Ti at SM86. Evidence:
`docs/sm86_c7_sol_candidate_fib024_source_v1.json`.

C7-S34M2 then passed the fresh SM86 build, batch-1/16 correctness smoke, and
Compute Sanitizer memcheck with zero errors. Evidence:
`docs/sm86_c7_sol_candidate_fib024_smoke_v1.json`. C7-S34M3 then completed the
bounded profiler-delimited capture, independent 152-instruction control
sidecar, deterministic 1762-record replay, and five target NCU repetitions.
The NCU median was 7413 cycles with 1762 instructions matching replay; tails
remain open. Evidence: `docs/sm86_c7_sol_candidate_fib024_vertical_v1.json`.

C7-S34N is now active for source review of FIB/problem_025 RMSNorm at hidden
size 4096. C7-S34N1 verified the SM86 marker, bounded entry point, distinct
EPS `1e-5`/shape contract, independent FP32 reference, and permitted
Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_candidate_fib025_source_v1.json`. C7-S34N2 then passed the
fresh SM86 build, batch-1/16 correctness smoke, and Compute Sanitizer memcheck
with zero errors; evidence is
`docs/sm86_c7_sol_candidate_fib025_smoke_v1.json`. C7-S34N3 then completed the
bounded profiler-delimited capture, independent 152-instruction control
sidecar, deterministic 3042-record replay, and five target NCU repetitions.
The NCU median was 11164 cycles with 3042 instructions matching replay; tails
remain open. Evidence: `docs/sm86_c7_sol_candidate_fib025_vertical_v1.json`.

C7-S34O is now active for source review of FIB/problem_026 RMSNorm at hidden
size 7168, the final listed FIB RMSNorm shape. C7-S34O1 verified the SM86
marker, bounded entry point, 512-thread/shape contract, independent FP32
reference, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_candidate_fib026_source_v1.json`. C7-S34O2 then passed the
fresh SM86 build, batch-1/16 correctness smoke, and Compute Sanitizer memcheck
with zero errors; evidence is
`docs/sm86_c7_sol_candidate_fib026_smoke_v1.json`. C7-S34O3 then completed the
bounded profiler-delimited capture, independent 152-instruction control
sidecar, deterministic 5426-record replay, and five target NCU repetitions.
The NCU median was 10537 cycles with 5426 instructions matching replay; tails
remain open. Evidence: `docs/sm86_c7_sol_candidate_fib026_vertical_v1.json`.

C7-S34P is now active for a distinct unexecuted L1 custom-CUDA family,
`problem_002_vae_conv3x3_groupnorm_silu_residual_fused`. Source review found
that row is actually pure ATen: `kernel.cu` is a stub and the only `__global__`
token is comment text. C7-S34P corrected the census from 73/116 to 72 custom
CUDA / 117 pure ATen and retained the row as an explicit source blocker
without duplicate tracing. Evidence:
`docs/sm86_c7_sol_l1_problem002_source_class_correction_v1.json`. C7-S34Q is
now active for the next L1 row with an actual custom CUDA kernel,
`problem_006_hyena_depthwise_conv1d_split_gate`. C7-S34Q1 verified its two
custom kernels, FP32 shape/oracle contract, and permitted Torch/GCC14 runtime;
evidence is `docs/sm86_c7_sol_l1p006_source_v1.json`. C7-S34Q2 is active for a
fresh build/correctness/memcheck smoke. That smoke passed batch/sequence
shapes `[1,512]` and `[4,2048]` exactly against the independent reference and
Compute Sanitizer reported zero errors; evidence is
`docs/sm86_c7_sol_l1p006_smoke_v1.json`. C7-S34Q3 then completed the bounded
profiler-delimited capture: both kernels were retained, 72 static instructions
per kernel decoded, and replay produced 679936 records total. Five NCU
repetitions selected both target kernels with matching per-kernel instruction
counts; per-kernel lower frontiers were 5242 and 4359 cycles, with tails open.
Evidence: `docs/sm86_c7_sol_l1p006_vertical_v1.json`. C7-S34R1 then verified
the custom RoPE kernel, custom position/frequency input contract, and permitted
Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p011_source_v1.json`. C7-S34R2 then passed custom-input
RoPE shapes `[1,256]` and `[4,1024]` with max BF16 errors `0.00098` and
`0.00391`; Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p011_smoke_v1.json`. C7-S34R3 then completed the bounded
profiler-delimited capture, independent 56-instruction control sidecar,
deterministic 22016-record replay, and five target NCU repetitions. The NCU
median was 4432 cycles with 22016 instructions matching replay; tails remain
open. Evidence: `docs/sm86_c7_sol_l1p011_vertical_v1.json`. C7-S34S1 then
verified the custom cos/sin kernel, FP32/BF16 shape contract, and permitted
Torch/GCC14 runtime; evidence is `docs/sm86_c7_sol_l1p012_source_v1.json`.
C7-S34S2 then passed scaled cos/sin shapes `[1,512]` and `[4,1024]` with max
errors up to `0.0078125`; Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p012_smoke_v1.json`. C7-S34S3 then completed the bounded
profiler-delimited capture, independent 40-instruction control sidecar,
deterministic 61440-record replay, and five target NCU repetitions. The NCU
median was 4172 cycles with 61440 instructions matching replay; tails remain
open. Evidence: `docs/sm86_c7_sol_l1p012_vertical_v1.json`. C7-S34T is now
active for `problem_013_fused_residual_rms_norm_backward`; no timing or NCU
value is transferred from prior rows. C7-S34T1 verified the two custom
backward kernels, hidden-size/shape contract, independent RMSNorm backward
oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p013_source_v1.json`. C7-S34T2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed backward-gradient shapes
`[1,256]` and `[2,512]` against the independent reference, with BF16 max
error `0.0078125` and Compute Sanitizer reporting zero errors; evidence is
`docs/sm86_c7_sol_l1p013_smoke_v1.json`. C7-S34T3 then completed the
two-launch capture, independent 552/320 instruction control sidecars,
deterministic 2457600-record replay, and five per-kernel NCU repetitions. The
`grad_weight` lower-frontier falsifier is retained (`113069` modeled versus
`34972` NCU median); evidence is
`docs/sm86_c7_sol_l1p013_vertical_v1.json`. C7-S34U is now active for
`problem_016_rope_inverse_frequency_computation`; no timing or NCU value is
transferred from prior rows. C7-S34U1 verified its custom inverse-frequency
kernel, scalar/oracle contract, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p016_source_v1.json`. C7-S34U2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed all seven declared
`rope_theta` values with sub-`1e-7` max FP32 error and Compute Sanitizer
reported zero errors; evidence is
`docs/sm86_c7_sol_l1p016_smoke_v1.json`. C7-S34U3 then completed the bounded
capture, independent 24-instruction sidecar, deterministic 30-record replay,
and five target NCU repetitions. The NCU median was 2995 cycles with 30
instructions matching replay; tails remain open. Evidence:
`docs/sm86_c7_sol_l1p016_vertical_v1.json`. C7-S34V is now active for
`problem_018_fused_rope_with_qk_norm_and_kv_cache_update`. C7-S34V1 verified
the four custom kernels, cache/shape contract, independent custom-input
oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p018_source_v1.json`. C7-S34V2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed the bounded
`[B=1,S=1,cache_len=0]` custom-input case with zero output error and zero
Compute Sanitizer errors; evidence is
`docs/sm86_c7_sol_l1p018_smoke_v1.json`. C7-S34V3 then completed the
four-launch capture, independent 304-instruction sidecar, deterministic
48592-record replay, and five per-launch NCU repetitions; per-launch
instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p018_vertical_v1.json`. C7-S34W is now active for
`problem_019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward`.
C7-S34W1 verified the two custom kernels, FP32 Vision-RoPE backward contract,
independent oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p019_source_v1.json`. C7-S34W2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed Vision-RoPE backward
sequence lengths `[128,449]` against the independent oracle with FP32 errors
below `1e-5`, and Compute Sanitizer reported zero errors; evidence is
`docs/sm86_c7_sol_l1p019_smoke_v1.json`. C7-S34W3 then completed the
three-launch capture, independent 336-instruction sidecar, deterministic
764928-record replay, and five per-launch NCU repetitions; all per-launch
instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p019_vertical_v1.json`. C7-S34X is now active for
`problem_022_vision_mlp_gelu_backward`. C7-S34X1 verified the CUTLASS/Sm80
TensorOp source, MLP/GELU backward contract, independent oracle, and permitted
Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p022_source_v1.json`. C7-S34X2 is active for a fresh
build/correctness/memcheck smoke; replay may retain a CUTLASS-specific
crosswalk blocker rather than silently falling back. The repository CUTLASS
include path was required; build and memcheck passed, but independent FP32
correctness failed materially in TF32 GEMM-derived gradients, so no native
vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1p022_smoke_v1.json`. C7-S34Y is now active for
`problem_024_vision_rotary_position_embedding_generation_backward`. Source
review corrected this row from marker-only custom CUDA to pure ATen: its `.cu`
is a stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem024_source_class_correction_v1.json`. The census
is now 71 custom CUDA / 14 CUTLASS / 118 pure ATen; C7-S34Z is active for
`problem_025_video_latent_gelu_activation`. C7-S34Z1 verified its vectorized
and scalar-tail custom kernels, FP32 GELU contract, independent oracle, and
permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p025_source_v1.json`. C7-S34Z2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed vectorized/tail GELU
shapes `[2,128]` and `[4,541]` with max error `5.60e-6`; Compute Sanitizer
reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p025_smoke_v1.json`. C7-S34Z3 then completed the bounded
vector-kernel capture, independent 88-instruction sidecar, deterministic
368640-record replay, and five target NCU repetitions. The scalar tail kernel
is source-present but unreachable for fixed intermediate size 4096; evidence is
`docs/sm86_c7_sol_l1p025_vertical_v1.json`. C7-S34AA is now active for
`problem_029_mamba_conv1d_with_gating`. Source review corrected this row from
marker-only custom CUDA to pure ATen: its `.cu` is a stub and `__global__` is
comment text. Evidence:
`docs/sm86_c7_sol_l1_problem029_source_class_correction_v1.json`. The census
is now 70 custom CUDA / 14 CUTLASS / 119 pure ATen; C7-S34AB is active for
`problem_033_post_norm_residual`. C7-S34AB1 verified the custom post-norm
RMSNorm kernel, BF16/FP32 shape
contract, independent oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p033_source_v1.json`. C7-S34AB2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed post-norm residual shapes
`[1,128]` and `[2,512]` with max BF16 error `0.00390625`; Compute Sanitizer
reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p033_smoke_v1.json`. C7-S34AB3 is active for a bounded
profiler-delimited capture. That capture completed with an independent
160-instruction sidecar, deterministic 440576-record replay, and five target
NCU repetitions; instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p033_vertical_v1.json`. C7-S34AC is now active for
`problem_034_flux_multi_axis_rope_frequency_computation`. C7-S34AC2 passed all
16 declared workload shapes across three deterministic seeds (48 correctness
rows) at unchanged 1e-5 tolerances; max absolute error was `5.96e-7`, and
Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p034_smoke_v1.json`. C7-S34AC3 then completed the row-0
capture, independent 56-instruction sidecar, deterministic 160517-record
replay, and five target NCU repetitions. The NCU median was 4659 cycles with
160517 instructions matching replay; tails remain open. Evidence:
`docs/sm86_c7_sol_l1p034_vertical_v1.json`. C7-S34AD is now active for
`problem_035_flux_ada_layer_norm_zero_modulation_extraction`. C7-S34AD1
verified the custom AdaLN modulation kernel, 3072/18432 chunk contract,
independent matmul/chunk oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p035_source_v1.json`. C7-S34AD2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed B=1/16 AdaLN matmul/chunk
correctness at the declared 0.0021 tolerance with max error `7.25e-5`; Compute
Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p035_smoke_v1.json`. C7-S34AD3 is active for a bounded
profiler-delimited capture. That capture completed with an independent
232-instruction sidecar, deterministic 8607744-record replay, and five target
NCU repetitions; instruction counts matched. The typed unsupported-memory-path
count (18432) remains explicit. Evidence:
`docs/sm86_c7_sol_l1p035_vertical_v1.json`. C7-S34AE is now active for
`problem_036_flux_output_norm_projection_chain`. Source/build/memcheck passed,
but the independent FP32 smoke failed the declared tolerance on B=2/S=293
(max error `0.0045166` versus absolute `0.0031`); no native vertical was
admitted. Evidence:
`docs/sm86_c7_sol_l1p036_smoke_v1.json`. C7-S34AF is now active for
`problem_038_flux_multi_head_rmsnorm_qk`.
 C7-S34AF1 verified the custom per-head Q/K RMSNorm kernel and 48×128
contract; evidence is `docs/sm86_c7_sol_l1p038_source_v1.json`. C7-S34AF2
passed all five declared shapes against the independent FP32 reference with
max Q/K error `1.91e-6`; Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p038_smoke_v1.json`. C7-S34AF3 is active for bounded
profiler/control/replay/NCU capture. That capture used the pinned CUDA 13.2 /
GCC14 extension, retained both launches, and produced a 64-target-instruction
sidecar, 5013504 replay instructions, and five per-kernel NCU repetitions;
instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p038_vertical_v1.json`. C7-S34AG is now active for
`problem_039_adaptive_instance_normalization_2d_backward`. Source review
corrected this row from marker-only custom CUDA to pure ATen: its `.cu` is a
stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem039_source_class_correction_v1.json`. The census
is now 69 custom CUDA / 14 CUTLASS / 120 pure ATen; C7-S34AH is active for
`problem_040_conv2d_residual_block`.
 C7-S34AH1 verified the two custom convolution kernels and 3/32/3 residual
contract; evidence is `docs/sm86_c7_sol_l1p040_source_v1.json`. C7-S34AH2
passed all five declared shapes with max error `0.00042725`; Compute Sanitizer
reported zero errors. Evidence is `docs/sm86_c7_sol_l1p040_smoke_v1.json`. C7-S34AH3 then completed the pinned
two-launch capture, independent 3024-instruction sidecar, deterministic
10155520-record replay, and five per-kernel NCU repetitions; instruction counts
matched. Evidence: `docs/sm86_c7_sol_l1p040_vertical_v1.json`. C7-S34AI is now
active for `problem_041_strided_conv2d_downsampling`. Source review corrected
this row from marker-only custom CUDA to pure ATen: its `.cu` is an explicit
stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem041_source_class_correction_v1.json`. The census
is now 68 custom CUDA / 14 CUTLASS / 121 pure ATen; C7-S34AJ is active for
`problem_042_moe_expert_load_balancing_and_token_capacity_backward`.
 Source review corrected this row from marker-only custom CUDA to pure ATen:
its `.cu` is a stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem042_source_class_correction_v1.json`. The census
is now 67 custom CUDA / 14 CUTLASS / 122 pure ATen; C7-S34AK is active for
`problem_043_mla_fused_qkv_rope_split`.
 Source review corrected this row from marker-only custom CUDA to pure ATen:
its `.cu` is a stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem043_source_class_correction_v1.json`. The census
is now 66 custom CUDA / 14 CUTLASS / 123 pure ATen; C7-S34AL is active for
`problem_046_attention_softmax_with_softcapping_and_dropout`.

C7-S34AL completed the next genuine L1 custom-CUDA row. Source review recorded
the BF16 `[B,8,Q,K]` softcap/softmax contract and 16 declared workload shapes;
the pinned CUDA 13.2/GCC14 extension passed all 16 rows across three
deterministic seeds (48 correctness rows), with maximum absolute error
`0.005859375` and zero tolerance violations. Representative Compute
Sanitizer memcheck reported zero errors. A profiler-gated `[1,8,293,293]`
launch then produced 8,159,464 raw tracer instructions, an independent
248-instruction sidecar (232 target instructions plus a 16-instruction inert
CUB helper), deterministic `l1-l2` replay with 2,894,840 records/instructions,
and five NCU repetitions (cycle median `19158`, SM-active median `17492`,
2,894,840 instructions each). Simulator and NCU instruction counts match;
dirty-writeback/coherence tails and zero native macro dispatch remain explicit.
Evidence: `docs/sm86_c7_sol_l1p046_source_v1.json`,
`docs/sm86_c7_sol_l1p046_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p046_vertical_v1.json`. C7-S34AM is now active for
`problem_048_fused_gate_up_projection_with_swiglu`; no timing or NCU value is
transferred from prior rows.

C7-S34AM completed source review and a repaired build/smoke for
`problem_048_fused_gate_up_projection_with_swiglu`. The row is a genuine
custom elementwise kernel chained after two vendor cuBLAS BF16 matmuls. The
first independent smoke exposed that `--use_fast_math`'s float tanh saturated
the GELU tail. A multiply-only diagnostic did not repair the error and was
reverted; the source now calls double libdevice tanh for the narrow saturated
tail while retaining the BF16 activation and product path. The pinned build passed all 16 declared
rows across three deterministic seeds (48 rows), with exact output equality
in this run, and representative Compute Sanitizer reported zero errors.
Two bounded NVBit attempts then failed before trace emission on vendor
`HMNMX2.BF16_V2` static-SASS decoding; the same fatal remained with
`SKIP_ENHANCED_TRACER=1`, so this is not merely an enhanced-parser failure.
No control sidecar, replay, NCU repetition, custom-only substitute, or timing
claim is admitted. Evidence: `docs/sm86_c7_sol_l1p048_source_v1.json`,
`docs/sm86_c7_sol_l1p048_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p048_capture_blocked_v1.json`. C7-S34AN is now active for
`problem_052_altup_hidden_state_collapse_with_magnitude_normalization`.

C7-S34AN corrected the source census for `problem_052` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen/cuBLAS with no custom launcher. The corrected population is now
65 custom CUDA / 14 CUTLASS / 124 pure ATen. No build, trace, replay, or NCU
vertical was admitted for this row. Evidence:
`docs/sm86_c7_sol_l1_problem052_source_class_correction_v1.json`. C7-S34AO is
now active for `problem_053_gaussian_topk_sparse_activation`.

C7-S34AO completed the genuine L1 Gaussian top-k sparse activation row. The
pinned CUDA 13.2/GCC14 build passed all 12 declared shapes across three
deterministic seeds (36 rows; maximum absolute error `0.03125`, with zero
declared-tolerance violations), and representative Compute Sanitizer reported
zero errors. The `[1,512,12288]`, target-sparsity `0.2` capture passed the
profiler-gated NVBit path, independent 488-instruction control sidecar
(472 target plus 16 inert CUB helper), deterministic byte-identical replay,
and five application-replay NCU repetitions. Replay and NCU instruction
counts match (`2354176`); NCU cycle median is `58369`, SM-active median
`56623`; dirty-writeback/coherence tails, host-side `ndtri`, parser limitation,
raw tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p053_source_v1.json`,
`docs/sm86_c7_sol_l1p053_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p053_vertical_v1.json`. C7-S34AP is now active for
`problem_056_mamba_ssm_dt_projection_softplus_clamp_backward`.

C7-S34AP completed the genuine L1 Mamba softplus/clamp backward row. The
pinned CUDA 13.2/GCC14 build passed all 16 declared shapes across three
deterministic seeds (48 rows; maximum `grad_dt` absolute error
`0.0009765625`, with zero declared-tolerance violations), and representative
Compute Sanitizer reported zero errors. The `[2,1571,40]` capture passed the
profiler-gated NVBit path, independent 224-instruction control sidecar
(208 target plus 16 inert CUB helper), deterministic byte-identical replay,
and five application-replay NCU repetitions. Replay and NCU instruction
counts match (`111840`); NCU cycle median is `15654`, SM-active median
`13997`; the raw tracer unit (`310400`), FP32 bias reduction, parser
limitation, dirty-writeback/coherence tails, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_l1p056_source_v1.json`,
`docs/sm86_c7_sol_l1p056_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p056_vertical_v1.json`. C7-S34AQ is now active for
`problem_057_mtp_shifted_embedding_with_dual_rms_norm_fusion`.

C7-S34AQ corrected the source census for `problem_057` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen roll/embedding/RMSNorm/concatenation/cuBLAS with no custom launcher.
The corrected population is now 64 custom CUDA / 14 CUTLASS / 125 pure ATen;
no build, trace, replay, or NCU vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1_problem057_source_class_correction_v1.json`. C7-S34AR is
now active for `problem_058_moe_expert_token_radix_sort_with_prefix_sum`.

C7-S34AR corrected the source census for `problem_058` from marker-only custom
CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates to
stable argsort, bincount, cumsum, and destination copies with no custom
launcher. The corrected population is now 63 custom CUDA / 14 CUTLASS / 126
pure ATen; no build, trace, replay, or NCU vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1_problem058_source_class_correction_v1.json`. C7-S34AS is
now active for `problem_059_moe_group_score_aggregation_and_masking`.

C7-S34AS completed the genuine L1 MoE group-score aggregation/masking row.
The pinned CUDA 13.2/GCC14 build passed all 16 declared token counts across
three deterministic seeds (48 rows), with the declared matching-negative-
infinity exception explicit, and representative Compute Sanitizer reported
zero errors. The checked `[1163,256]` capture passed the profiler-gated NVBit
path, independent 312-instruction control sidecar (296 target plus 16 inert
CUB helper), deterministic byte-identical replay, and five application-replay
NCU repetitions. Replay and NCU instruction counts match (`711756`); NCU
cycle median is `11294`, SM-active median `9582`; the superseded inline-oracle
attempt is retained separately, while the checked official-reference capture
is authoritative. Dirty-writeback/coherence tails, parser limitation, raw
tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p059_source_v1.json`,
`docs/sm86_c7_sol_l1p059_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p059_vertical_v1.json`. C7-S34AT is now active for
`problem_061_tanh_gated_residual_add_backward`.

C7-S34AT completed the genuine L1 tanh-gated residual backward row. The
pinned CUDA 13.2/GCC14 build passed all 16 declared shapes across three
deterministic seeds (48 rows; maximum `grad_hidden_states` absolute error
`0.0625`, with zero declared-tolerance violations), and representative
Compute Sanitizer reported zero errors. The checked capture passed two
profiler-gated custom launches after fusing the identity residual write into
Kernel A and replacing the scalar `gate.to(float)` helper launch with a direct
`item<float>()` read. Independent control extraction decoded 400 instructions
(216 + 168 target plus 16 inert CUB helper), replay was byte-identical on a
repeat, and five application-replay NCU repetitions ran per kernel. Replay and
NCU instruction counts match (`231346` and `355`); cycle medians are `9795`
and `3287`, with SM-active medians `8115` and `1722`. Dirty-writeback/coherence
tails, parser limitation, raw tracer units, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_l1p061_source_v1.json`,
`docs/sm86_c7_sol_l1p061_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p061_vertical_v1.json`. C7-S34AU is now active for
`problem_062_kv_cache_update_with_rope_backward`.

C7-S34AU corrected the source census for `problem_062` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen index selection, BF16 RoPE algebra, reductions, clone/index-fill, and
destination copies with no custom launcher. The corrected population is now
62 custom CUDA / 14 CUTLASS / 127 pure ATen; no build, trace, replay, or NCU
vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1_problem062_source_class_correction_v1.json`. C7-S34AV is
now active for `problem_064_latent_kv_expansion_with_split`.

C7-S34AV corrected the source census for `problem_064` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen RMSNorm, cuBLAS projection, view/transpose/split, and destination
copies with no custom launcher. The corrected population is now 61 custom CUDA
/ 14 CUTLASS / 128 pure ATen; no build, trace, replay, or NCU vertical was
admitted. Evidence:
`docs/sm86_c7_sol_l1_problem064_source_class_correction_v1.json`. C7-S34AW is
now active for `problem_066_masked_softmax_with_attention_dropout_backward`.

C7-S34AW initially retained `problem_066` as a correctness blocker from a
custom-input/all-elements diagnostic. C7-S34BB corrected that disposition:
the actual SOL generator, reference, negative-infinity handling, and
`required_matched_ratio=0.99` criterion pass all 16 declared rows at seed 223
(minimum matched ratio 1.0; maximum absolute error `5.96e-8`). The raw
diagnostic remains retained but is explicitly superseded; no NCU value entered
the audit. Evidence: `docs/sm86_c7_sol_l1p066_correctness_recheck_v1.json` and
the superseded raw record `docs/sm86_c7_sol_l1p066_smoke_v1.json`. C7-S34AX is
now active for `problem_068_fused_rope_application_backward`.

C7-S34BB3 then completed a corrected problem066 native vertical using the
official generator/reference path. One profiler-gated custom launch produced
592 decoded control instructions (576 target + 16 CUB helper), 24,707,072
replay/NCU instructions, byte-identical replay repeat, and five application-
replay NCU repetitions (cycle median `174414`, SM-active median `172696`).
The modeled GPC lower (`173682`) exceeds two measured repeats; that is retained
as a lower-bound falsifier, not fitted or called completion. Evidence:
`docs/sm86_c7_sol_l1p066_vertical_v1.json`. C7-S34BC is now the active
remaining-population/source-execution seam.

C7-S34AX completed the genuine L1 FP32 RoPE backward row. The pinned CUDA
13.2/GCC14 build passed all 14 declared shapes across three deterministic
seeds (48 rows; maximum absolute error `0.0000190735`, with zero declared-
tolerance violations), and representative Compute Sanitizer reported zero
errors. The `[1,32,8,293]` capture passed three profiler-gated custom launches,
independent 768-instruction control extraction (40 + 712 target plus 16 inert
CUB helper), deterministic byte-identical replay, and five application-replay
NCU repetitions per launch. Replay and NCU instruction counts match; cycle
medians are `17355`, `6081`, and `20834`; SM-active medians are `15611`,
`4392`, and `19181`. Dirty-writeback/coherence tails, parser limitation, raw
tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p068_source_v1.json`,
`docs/sm86_c7_sol_l1p068_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p068_vertical_v1.json`. C7-S34AY is now active for
`problem_069_rms_norm`.

C7-S34AY completed the genuine L1 1024-thread BF16x2 RMSNorm row. The pinned
CUDA 13.2/GCC14 build passed all 16 declared shapes across three deterministic
seeds (48 rows; maximum absolute error `0.125`, with zero declared-tolerance
violations), and representative Compute Sanitizer reported zero errors. The
`[1,131,8192]` capture passed one profiler-gated launch, independent
192-instruction control extraction (176 target plus 16 inert CUB helper),
deterministic byte-identical replay, and five application-replay NCU
repetitions. Replay and NCU instruction counts match (`1192624`); NCU cycle
median is `17285`, SM-active median `15399`; dirty-writeback/coherence tails,
parser limitation, raw tracer units, and zero native macro dispatch remain
explicit. Evidence: `docs/sm86_c7_sol_l1p069_source_v1.json`,
`docs/sm86_c7_sol_l1p069_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p069_vertical_v1.json`. C7-S34AZ is now active for
`problem_071_kv_cache_update_with_rope`.

C7-S34AZ completed the genuine L1 KV-cache update with RoPE row. The pinned
CUDA 13.2/GCC14 build passed all 16 declared shapes across three deterministic
seeds (48 rows; exact output equality), and representative Compute Sanitizer
reported zero errors. The `[B=2,H=10,new=512,current=0]` capture passed one
profiler-gated launch, independent 152-instruction control extraction (136
target plus 16 inert CUB helper), deterministic byte-identical replay, and
five application-replay NCU repetitions. Replay and NCU instruction counts
match (`4136960`); NCU cycle median is `21576`, SM-active median `19847`;
dirty-writeback/coherence tails, parser limitation, raw tracer units, and zero
native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p071_source_v1.json`,
`docs/sm86_c7_sol_l1p071_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p071_vertical_v1.json`. C7-S34BA is now active for
`problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection`.

C7-S34BA completed the problem073 source/build/correctness bar but retained a
full-chain NVBit blocker. The pinned CUDA 13.2/GCC14 mixed custom-RMSNorm plus
cuBLAS/layout implementation passed all 16 declared rows × 3 seeds using the
actual SOL generator and `required_matched_ratio=0.99`; strict all-element
diagnostics retain small mismatches, and representative memcheck reported zero errors.
The full profiler-gated chain then failed before trace emission on vendor/ATen
`DEPBAR.LE SB0 0x0 {2 1}` decoding. No RMSNorm-only substitute, replay, or
NCU vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1p073_source_v1.json`,
`docs/sm86_c7_sol_l1p073_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p073_capture_blocked_v1.json`. The BA blocker is retained;
BB is complete and BC is now the active remaining-population/source-execution
seam.

C7-S34BC1 source-reviewed problem075 and corrected the prior marker census:
its `.cu` is a stub and `main.cpp` delegates the full grouped-query attention
chain to ATen/cuBLAS; no custom launcher exists. No build or native vertical
was admitted. The corrected population is now 60 custom CUDA / 14 CUTLASS /
129 pure ATen. Evidence:
`docs/sm86_c7_sol_l1_problem075_source_class_correction_v1.json`. C7-S34BD
is now active for `problem_078_group_norm_fusion`.

C7-S34BD1 source-reviewed problem078 as a genuine 32-group FP32 custom
GroupNorm kernel (not an ATen marker). The source contract and 16 declared
shape rows are recorded; build/correctness/memcheck is the next seam. Evidence:
`docs/sm86_c7_sol_l1p078_source_v1.json`.

C7-S34BD completed problem078. All 16 declared shapes × 3 seeds (48 rows)
passed the pinned CUDA 13.2/GCC14 build and independent FP32 reference
(`max_abs_error=2.861e-6`); representative memcheck reported zero errors.
The `[B=4,C=320,H=64,W=64]` capture passed one profiler-gated launch,
independent 448-instruction control extraction (432 target + 16 CUB helper),
byte-identical replay repeat, and five application-replay NCU runs. Replay and
NCU instruction counts match (`5542528`); NCU cycle median is `97482`,
SM-active median `95960`; retirement tails remain open. Evidence:
`docs/sm86_c7_sol_l1p078_source_v1.json`,
`docs/sm86_c7_sol_l1p078_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p078_vertical_v1.json`. C7-S34BE is now active for
`problem_080_adaptive_layernorm_continuous_with_modulation`.

C7-S34BF1 source-reviewed problem081 and corrected the marker census: its
`.cu` is a stub and its full grouped-query attention chain is ATen/cuBLAS, with
no custom launcher. No build or native vertical was admitted. The corrected
population is now 59 custom CUDA / 14 CUTLASS / 130 pure ATen. Evidence:
`docs/sm86_c7_sol_l1_problem081_source_class_correction_v1.json`. C7-S34BG
is now active for `problem_082_qk_norm_scaled_dot_product_attention`.

C7-S34BG1 source-reviewed problem082 and corrected the marker census: its
`.cu` is a stub and the full QK-normalized scaled attention chain is
ATen/cuBLAS with no custom launcher. No build or native vertical was admitted.
The corrected population is now 58 custom CUDA / 14 CUTLASS / 131 pure ATen.
Evidence: `docs/sm86_c7_sol_l1_problem082_source_class_correction_v1.json`.
C7-S34BH is now active for `problem_084_silu_activation_backward`.

C7-S34BE1 source-reviewed problem080 as a genuine two-kernel FP32 adaptive
LayerNorm/modulation path. The contract and 16 declared shape rows are
recorded; build/correctness/memcheck is the next seam. Evidence:
`docs/sm86_c7_sol_l1p080_source_v1.json`.

C7-S34BE completed problem080. All 16 declared shapes × 3 seeds (48 rows)
passed the pinned CUDA 13.2/GCC14 build and independent FP32 reference
(`max_abs_error=0.00036621`); representative memcheck reported zero errors.
The `[B=1,S=128]` capture passed two profiler-gated launches, independent
576-instruction control extraction (216 + 344 target plus 16 inert CUB helper),
byte-identical replay repeat, and five application-replay NCU runs per kernel.
Replay and NCU instruction counts match; cycle medians are `77856` and `9787`,
SM-active medians `76128` and `8109`; retirement tails remain open. Evidence:
`docs/sm86_c7_sol_l1p080_source_v1.json`,
`docs/sm86_c7_sol_l1p080_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p080_vertical_v1.json`. C7-S34BF is now active for
`problem_081_joint_attention_context_projection`.

C7-S34G source-reviewed an L2 pure-ATen/cuBLAS candidate. It has an explicit
SM86/shape contract but delegates to multiple GEMM and elementwise kernels; no
duplicate NVBit attempt was made after the adjacent L1 pure-ATen bounded hang.
The row remains an explicit source blocker with no native replay/NCU claim.
Evidence: `docs/sm86_c7_sol_l2_pure_aten_source_blocked_v1.json`.

C7-S34H then built and sanitizer-checked a CUTLASS TensorOp FIB candidate and
captured a real NVBit/control sidecar. The `l1-l2` replay refused on an
invalid inferred bypass-copy cohort projection; flat-mode replay completed and
five independent NCU runs measured a 73800-cycle median with a retained
four-instruction simulator/NCU crosswalk difference. Async/writeback/coherence
tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib004_flat_fallback_v1.json`.

C7-S34I initially classified the remaining 203 SOL rows from source markers;
C7-S34P1 corrected one comment-only `__global__` false positive, leaving 72
custom CUDA, 14 CUTLASS, and 117 pure ATen/cuBLAS-style rows. This census
changes candidate selection only; it does not claim compilation, replayability,
completion, timing, or SOL score. Evidence:
`docs/sm86_c7_sol_source_class_census_v1.json`.

C7-S34E source-reviewed an L1 pure-ATen candidate. It built and passed exact
analytic correctness plus memcheck, but a profiler-delimited NVBit attempt
hung before trace emission and was terminated after a bounded observation. No
replay/NCU vertical was admitted; the multi-kernel/static-module blocker is
retained explicitly. Evidence: `docs/sm86_c7_sol_l1_candidate_blocked_v1.json`.

C1-R-C/C6-M added a focused semantics regression: a structurally valid live
async-copy boundary is refused before scalar macro derivation/mutation. This
strengthens exact fallback; async visibility and hardware completion remain
open.

C3-F is closed only as a boundary-selection phase: the physical completion
upper remains unknown, and C4-A is the selected independent seam. The existing
async-group and destination-layout observer contracts are reconciled; the
endpoint remains model-local rather than physical completion.

C4-B is complete as one controlled observer slice: the dependent SFU warp
observer interval `34205.4..34521.3` covered all five warp-8 target repeats
with `0.150%` point APE, while its simulator lower bound remained separate at
`38126` cycles. C4-C, C4-D, and C4-E retain negative quad transfer, shape, and
block-count intervention results. C4-F's empirical batch feature covered four
of five 210-block target repeats, with one 12-cycle interval miss and
`0.028%` point APE; no designed-extrapolation coverage guarantee is claimed.
C4-G fixes the observer-domain boundary: broad hardware accuracy, semantic
completion, and native dispatch remain open.

C6-A made one fresh real-trace reference-feature attempt with a 10,000-record
bound. It reached the bound without a footer and returned the retained limit
error; this is not labeled complete fallback or a native witness. C6-B
preserves the prior footer-terminated exact fallback; C6-D/E now cover a
narrow default-front-end snapshot transition. C6-F preserves the multi-warp
boundary, and C6-G closes that boundary explicitly. C6-H now closes the
scoped real store/fence negative branch; broader completion remains open. The next implementation seam is C1-R/C3/C6: preserve the store/fence mismatch,
retain exact fallback, and seek independently typed ownership/visibility state
without synthetic promotion. The scoped C3-G/C6-I writeback audit is complete; C3-H/C6-J preserves the visibility exact-fallback boundary.

C6-B closes only the footer-terminated negative branch: a real reference run
has 33,290 typed boundaries, zero complete transitions, and 33,290 exact
fallbacks with zero native dispatch. C6-C now closes a narrow reference-only
positive real-trace dispatch; physical completion is still unknown.

C6-C now closes a narrow reference-only positive real-trace dispatch. The
memory-free scalar cubin produced a complete typed boundary after disabling
the reference-only front end; a separate detailed replay supplied six issue
effects and a terminal barrier/exit state, and the witness run dispatched one
typed macro with zero fallbacks while matching detailed counters, cycles,
frontiers, and SM activity. Readiness diagnostics differ because compressed
execution does not replay every issue-causal diagnostic increment. The same
witness falls back without the typed front-end snapshot; C6-D/E now provide a
narrow positive default-front-end snapshot transition.

C6-D/E now also have a narrow positive default-front-end transition: a typed
instruction-cache/stream exit snapshot enables one dispatch with zero
fallbacks, matching detailed counters, cycles, frontiers, terminal time, and
SM activity. Readiness diagnostics remain separate and differ by design. C6-F preserves the multi-warp boundary; C6-G closes the broader state
decision. C6-H now closes the scoped real store/fence negative branch. The snapshot
transition is not a general cache/front-end proof.
Hardware completion and native hardware scheduling remain open.

C6-H closes a scoped real store/device-fence negative branch. The one-block
SM86 probe passed correctness, NVBit capture, static control extraction, and
deterministic replay; tail extraction observed 17 stores and `MEMBAR.SC.GPU`
/ `ERRBAR` while fence scope and hardware completion remained unknown. The
reference-feature trace terminated with a footer after 34 reconstructed
boundaries (17 pre/post pairs); every boundary was incomplete for
`unmodeled_completion_tails`, exact replay was selected throughout, and native
macro dispatch remained zero. This is not a physical completion upper or a
native scheduling claim. Evidence: `docs/sm86_c6h_store_fence_completion_boundary_v1.json`.

C3-G/C6-I then audited the existing opt-in writeback/coherence service on the
same real probe. The writeback arm modeled completion for 512 dirty sectors
(16,384 bytes) at cycle 11,121; enabling the model-local L1 visibility/
resident-writeback path moved the modeled completion frontier to 11,146 while
the separate SM-active frontier stayed 11,140. `dirty_writeback`/`coherence`
remained unknown in both arms, so the typed boundary and native macro guards
retain exact fallback for incomplete tails. This is model-local service
progress, not a physical completion upper or NCU fit. Evidence:
`docs/sm86_c3g_optin_writeback_coherence_tail_audit_v1.json`.

C3-H/C6-J then ran the same visibility/resident-writeback path in reference
mode. A 256-record bound was retained as a limit failure; the 4096-record
run reached a footer with 1130 boundaries (565 pre/post pairs), including
1024 visibility accesses, two fences, two invalidations, and 32 L2 batches.
All boundaries remained incomplete for `unmodeled_completion_tails`, exact
replay was selected, and native dispatch stayed zero. Fence scope/order are
event observations, not a complete owner/generation/read-from witness.
Evidence: `docs/sm86_c3h_visibility_reference_boundary_v1.json`.

C3-I/C6-K then ran the independent raw-protobuf/static-SASS tail extractor
without simulator or NCU inputs. It confirmed the observed store/fence prefix
but found no owner/read-from, line-generation, global-order, fence-scope, or
hardware-completion relation. The visibility reference stream remains exact
fallback for all 1130 boundaries with zero native dispatch. This closes the
negative witness branch only; independent ownership/generation evidence remains
open. Evidence: `docs/sm86_c3i_owner_generation_negative_v1.json`.

C4-H then ran an independent constant-load dependent-chain observer protocol:
calibration blocks 8/16/32 produced a frozen `13805.3..13848.1` interval
before an untouched 64-block target. All five target repeats missed the
interval (`0/5`, point APE `0.434%`), so no refit or widening was performed.
C4-I synchronizes the domain: the positive SFU slice remains mechanism-local;
constant-load and quad transfers are falsified or partial, and broad accuracy
remains open. Evidence: `docs/sm86_c4i_observer_domain_reconciliation_v1.json`.

C5-A reports the corresponding held-out statistics: the frozen constant-load
interval covered `0/5` target repeats, width was `42.86` cycles, point APE was
`0.434%`, and target stdev was `64.67` cycles. The simulator lower bound
(`62287`) and open writeback/coherence tails remain separate. Broad C5
accuracy is still open. Evidence:
`docs/sm86_c5a_constant_load_holdout_evaluation_v1.json`.

C5-B then evaluated the disjoint independent-load mode-1 family. Calibration
blocks 8/16 produced a frozen `11003..11293` interval before an untouched
32-block target; all five target repeats missed (`0/5`, point APE `2.578%`).
The miss is retained without refit or widening, and simulator frontiers stay
separate. C5-C now synthesizes the positive/negative observer slices; broad
accuracy remains open. Evidence:
`docs/sm86_c5b_constant_load_mode1_reconciliation_v1.json`.

C5-C synthesizes the positive SFU slice with both constant-load transfer
falsifiers while keeping coverage, width, signed error, semantic lower bounds,
and completion status separate. No broad accuracy or completion claim is made.
The narrow scalar actor also gained a pre-mutation detailed-effect-span guard:
truncated or non-monotone supplied effects now refuse, while the model-local
empty-effect recurrence remains compatible.

M5-A through M5-D are now closed at their scoped bars. The M5-A
control/reduction differential is in
`docs/sm86_m5a_native_tail_differential_launch_cycle_v1.json` (simulator GPC
delta `671`, independent NCU median delta `740`, zero native dispatches, and
an unresolved reduction/dirty-writeback/coherence tail). M5-B reconciles the
pre-NCU async observer holdout in
`docs/sm86_m5b_calibration_holdout_reconciliation_v1.json` (interval coverage
`1.0`, point APE `0.396%`, semantic interval still open). M5-C reruns the
current reference build on a real trace and preserves exact fallback for all
`10,032` typed boundaries with zero complete transitions; see
`docs/sm86_m5c_native_continuation_fallback_v1.json`. M5-D synchronizes C1-C8
claims and records focused regressions in
`docs/sm86_m5d_claim_synchronization_v1.json`.

The subsequent C1-R store/fence recapture is retained in
`docs/sm86_c1r_store_fence_differential_live_v1.json`: simulator store deltas
are `2886` cycles versus NCU median deltas `6080`/`6228`, while the modeled
12-cycle device-fence increment misses NCU's `177`/`325` increments. It is a
fence-model mismatch, not a timing fit; store tails remain open.

The top M5 program remains open: physical completion tails, a positive real
native continuation witness, broader accuracy/generalization, and physical
C8 mechanism completion are not established by these scoped phases. No scored NCU value became a
model feature or tuning input.

C8-A is now complete at the active-interface seam: `sim-run` and the bounded
workload runner expose clean help surfaces, the retired cupti-activity flag is
absent from active simulator/workload source, historical SimBench references
remain retained, and all 42 inventory entries/48 stage links resolve. C8-B is
complete for corrected-workload failure dispositions: six concrete external
tracer/tensor-pipeline/vendor-module blockers remain retained without shape
substitution. C8-C is complete as a scoped final runbook synchronization;
`docs/sm86_c8c_runbook_synchronization_v1.json` now links the
M5/C8 records from `docs/README.md` and confirms that C1/C3/C4/C5/C6/C7 open
surfaces remain visible. C8 is complete only as a scoped runbook/claim phase;
physical mechanisms remain open.

The residual **C1-R matched resident-memory fence/visibility suite** now has a
fresh four-arm capture in
`docs/sm86_c1r_store_fence_differential_live_v1.json`. Control/store ×
no/device-fence arms passed seven host oracles, sanitizer, NVBit, deterministic
replay, and five NCU repetitions. Simulator store deltas are `2886` cycles;
NCU medians are `6080`/`6228`, and the simulator's 12-cycle fence increment
misses NCU's 177/325-cycle increments. This is a retained fence-model
mismatch, not a fitted parameter; store completion tails remain open and C1-R
stays active for repair. The follow-up system-fence arms reproduce the prior
crosswalk blocker (`756` extra NCU instructions per system arm), so those rows
are retained but not scored. C1-R-B is complete as a negative reconciliation;
the next technical seam is C3-F physical-upper/accuracy boundary. C3-A's
focused model-local regression passed 13 targeted tests
covering feedback acknowledgements, checked completion arithmetic, async
refusal, and explicit writeback-tail propagation; C3-B fixed the per-warp
acknowledgement readiness bug and the
SM-active projection bug, and the retained traces passed the `0` versus `4096`
model-only perturbation. C3-C independently extracted 1428 store facts per arm
and the device-fence opcode while retaining hardware completion, ownership,
generation, order, and fence scope as unknown. C3-D is complete for
async-copy/visibility tail reconciliation and records 8192 async lane copies
with 2048 committed/waited groups plus an unchanged carded-DRAM negative
perturbation. C3-E is complete for an independently witnessed
completion/visibility continuation and has now completed the async reference
branch with 33290 exact-fallback boundaries and zero complete transitions.
C3-F remains active because the physical completion upper is still unknown;
none of these local results close hardware completion.

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

TF32 HMMA and INT8 IMMA rows pass host correctness, NVBit, NCU, and exact
SM86 replay at their shape-specific capability seams; physical tails remain open. Ten memory/control rows remain real lower-bound falsifiers; three L2
examples are pointer chase (59,545 vs NCU median 49,580), persisting window
(31,173 vs 29,846 after the sink repair), and prefetch hint (46,874 vs
40,598). The prefetch binary emits
five static `CCTL.E.PF2` instructions; cache effect and completion remain
unobservable. The corrected shared-bank shape uses SM86 dynamic shared-memory
opt-in (65,536 bytes against default 49,152 and opt-in 101,376), superseding
its old false OOD disposition.

The research-probe population was re-enumerated from the current tree: 42 directories are present (37 Makefile-backed probes, no separate source/preserved-binary rows, and 5 analysis-only rows). The bounded inventory is `docs/sm86_c7_research_probe_inventory_v1.json`; the synchronized 37-row build matrix is `docs/sm86_c7_research_probe_build_matrix_v1.json`; all 37 Makefile probes build successfully with the pinned CUDA 13.2/GCC 14/sm_86 matrix in `docs/sm86_c7_research_probe_build_matrix_v1.json`. Matrix campaigns are complete for issue_port_control (68), async_barrier_overlap (36), store_ack_frontier (30 arms), memory_retirement_service (102 with eight crosswalk exclusions), barrier_pipeline (33), and l1tex_admission_backpressure (100, all lower-bound-consistent). An explicit warmup/target l2_queue_frontier vertical cell also passes correctness/NVBit/replay/five-NCU with lower-bound consistency in `docs/sm86_c7_research_probe_l2_queue_frontier_live_v1.json`; remaining probe families are not yet claimed.

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

C7-S33AD completed the held-out `distance=4, warps=4` completion pair. Both
split and bank3 arms use one patched 8,192-generation cubin and pass seven host
trials, Compute Sanitizer, full NVBit capture, deterministic replay, and five
NCU runs with 622,832 matching target warp instructions. Simulator lower is
336826 for both; split NCU median is 348610 and bank3 median 414145. Both
lowers are consistent; the 65,535-cycle differential is mechanism-local and no
latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance4_pair_live_v1.json`.

C7-S33AE completed the held-out `distance=8, warps=4` completion pair. Both
layouts pass seven host trials, Compute Sanitizer, full NVBit capture,
deterministic replay, and five NCU runs with 622,832 matching target warp
instructions. Simulator lower is 336826 for both; split NCU median is 356797
and bank3 median 414112. Both lowers are consistent; the 57,315-cycle bank3
differential is mechanism-local and no latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance8_pair_live_v1.json`.

C7-S33AL closed the completion-distance vertical family and reconciled the
probe inventory. All 23 live probe stages have schema-validated artifacts;
current manifests and focused selection runs pass; active paired-binary
campaigns no longer require stale trace twins or predecessor-sidecar
populations. Population entries now link live status/artifacts for completed
families, while 26 current research-probe entries remain uncaptured or
analysis-only and are not counted complete. Falsifiers, crosswalk exclusions,
open tails, unsupported rows, and VMM physical-identity limits remain explicit.
Evidence: `docs/sm86_c7_research_probe_inventory_v1.json`.

C7-S33AG completed one atomic completion serialization vertical slice. The
atomic campaign now uses one rebuilt 1,024-generation cubin per variant across
correctness, NVBit, replay, and NCU; stale 64-generation trace cubins were
removed. The driver enforces strict parsing, variant/layout consistency, bounded
allocations, output sentinels, elapsed-sample checks, returned-sum/data-sum
oracles, and cleanup. `ret_lane_word, warps=4` passed seven host trials,
Compute Sanitizer, full NVBit capture, deterministic replay, and five NCU runs.
Replay and NCU report 24,660 warp instructions; atomic requests are 4,096 per
run, NCU median 316109, simulator lower 267561. No atomic timing model is
activated. Artifact:
`docs/sm86_c7_research_probe_atomic_completion_serialization_live_v1.json`.

C7-S33AH completed one multi-SM atomic service vertical slice. One rebuilt
1,024-generation cubin per operation now serves correctness, NVBit, replay, and
NCU; stale 64-generation twins are removed. The driver rejects other generation
counts, bounds allocation size, validates elapsed samples and data/returned sums,
and cleans up CUDA resources. `ret32, blocks=4, spacing=0` passes seven host
trials, Compute Sanitizer, full NVBit capture, deterministic replay, and five
NCU runs. Replay/NCU report 24,692 target warp instructions; NCU observes 4,096
atomic and 131,072 LTS requests, median 355107, simulator lower 268019.
Routing/completion tails remain black-box and no timing model is activated.
Artifact:
`docs/sm86_c7_research_probe_atomic_multism_service_live_v1.json`.

C7-S33AI completed an observational atomic alias-adversarial vertical slice.
One rebuilt 1,024-generation cubin per operation/width replaces stale trace
twins; the driver validates generation/pattern/alias/width, bounded memory,
elapsed samples, data/returned sums, and cleanup. The ret32 dense alias-32
`warps=4, stride=4` cell passes seven host trials, Compute Sanitizer, full
NVBit capture, deterministic replay, and five NCU runs. Replay/NCU report
24,684 instructions; atomic requests are 4,096, LTS requests 131,072, NCU
median 382651, simulator lower 267996. No proprietary hash is reconstructed
and no timing model is activated. Artifact:
`docs/sm86_c7_research_probe_atomic_alias_adversarial_live_v1.json`.

C7-S33AJ completed a CUDA VMM alias diagnostic. The rebuilt VMM correctness
campaign passes 19/19 cells, including seven-trial exact alias/data oracles.
The selected offset-0 alias cell has five fresh NCU profiles and an NVBit
capture covering seven VMM launches. NCU read-sector medians are observations
only; VMM virtual mappings and physical allocation handles are outside the
ordinary native replay contract, so no simulator lower bound, hash, or timing
model is fabricated. Artifact:
`docs/sm86_c7_research_probe_atomic_vmm_alias_live_v2.json`.

C7-S33AK completed a task-done multi-kernel boundary slice. The same
`store_device` binary passes seven host correctness trials for footprint 16 MiB,
thrash 16 MiB, four blocks, and two warmups. NVBit captures all nine launches
with enhanced metadata enabled; sim-run replays all nine with full coverage,
and five application-replay NCU runs profile the same nine-launch application.
Per-kernel instruction counts match. Producer, thrash, consumer, copy, and host
endpoints remain separate; no fixed retirement tail or scalar upper is inferred.
Artifact: `docs/sm86_c7_research_probe_task_done_boundary_live_v1.json`.

C7-S33AM completed an SMSP scheduler-contention vertical slice. One rebuilt
8,192-generation binary replaces the stale trace twin. The `q20, phase=0,
warps=8` cell passes seven host trials, Compute Sanitizer, full NVBit capture,
deterministic replay, and five NCU runs. Its 69-MiB block payload required the
finite anchored-reader limit to rise from 64 MiB to 128 MiB; focused anchored
tests pass. Replay/NCU match at 2,556,096 instructions; simulator lower is
1,123,534 and NCU median 1,271,033. Phase labels are experimental controls,
not physical scheduler-slot truth; no scheduler timing model is activated.
Artifact:
`docs/sm86_c7_research_probe_smsp_scheduler_contention_live_v1.json`.

C7-S33AN is active: select the next uncaptured probe after source review, with
fixed-latency bypass canary safety checked first.

C7-S33AN completed a safe fixed-result bypass vertical slice. One rebuilt
1,025-generation cubin per safe variant replaces stale 65-generation twins; the
driver rejects stale generations, validates elapsed/sink outputs, and cleans up.
The `fixed_retired`, fixed-kind, warps-4 arm passes seven host trials, Compute
Sanitizer, metadata-enabled NVBit capture, deterministic replay, and five NCU
runs. Replay/NCU match at 28,788 instructions; simulator lower is 27505 and
NCU median 34930. The intentionally illegal-address global distance-4 canary
was not run. No bypass latency model is activated. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_bypass_live_v1.json`.

C7-S33AO and C7-S33AP are complete refreshes: the fixed-result-capacity arm
uses one 8,192-generation family and the immediate-constant anchor uses the
hand-authored PTX cubin through correctness, NVBit, replay, and NCU. Neither
activates a timing model.

## Retained evidence

The 2026-09-30 cold/warm campaigns completed 15/15 workloads on RTX 3090 Ti / SM86,
UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09,
1350/10501 MHz, P0. This is historical host identity, not a current-device check.
Raw records: `results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/`.
Preserve datasets, source, tests, and scientific limitations; do not recreate
removed policy text or workflow machinery.

C7-S33AO completed a fixed-result capacity live refresh. The stale 64-generation
trace twin was removed from the active Makefile/campaign/driver interface; one
8,192-generation b7/b8/b9 family now serves correctness, NVBit, replay, and
NCU. W8/b9 passed seven correctness trials, full NVBit capture, deterministic
replay, and five NCU runs. Replay/NCU match at 1,573,144 instructions; NCU
median is 851,475 cycles and simulator lower is 590,830. This is a canary only:
capacity and pop width remain unidentified and no timing model is enabled.
Artifact: `docs/sm86_c7_research_probe_fixed_latency_result_capacity_live_v1.json`.

C7-S33AP completed an immediate-constant live anchor. The hand-authored PTX
cubin passed direct correctness, NVBit capture, control-sidecar generation,
sim-run replay, and an independent NCU profile. At blocks84/iterations256,
replay and NCU match at 151,872 instructions; simulator lower is 6,037 cycles
and NCU is 10,709 GPC cycles. This remains observational; no IMC timing model
is activated. Artifact: `docs/sm86_c7_research_probe_immediate_constant_frontier_live_v1.json`.

C7-S33AQ completed a hardened `constant_load_path` vertical slice. The rebuilt
source now uses strict parsing, bounded input sizing, sentinel result buffers,
and a separate functional output oracle. Seven full-shape correctness runs,
reduced Compute Sanitizer, NVBit, deterministic replay, and five
application-replay NCU captures passed with replay/NCU instruction identity at
1,909,824. The simulator lower bound is 123,727 cycles versus NCU median
24,016, so this is a retained lower-bound falsifier; no constant-load/cache
parameter is fitted and dirty-writeback/coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_constant_load_path_live_v1.json`.

C7-S33AR completed the L1TEX admission same-binary refresh. The stale
trace-only executable interface was removed from its active Makefile/campaign;
one rebuilt `l1tex_admission_backpressure` binary served correctness, full-shape
Compute Sanitizer, NVBit, replay, and five application-replay NCU runs for the
store/line-unique 84-block/8-warp/depth-32 cell. Replay and NCU match at
1,255,968 instructions; simulator lower is 12,905 cycles and NCU median is
29,879. No L1TEX/cache parameter is fitted and dirty-writeback/coherence tails
remain open. Artifact:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_live_v2.json`.

C7-S33AS completed the predeclared same-binary control-versus-store contrast at
the same shape. Hardware adds 12,823 cycles while the simulator lower-bound
delta is 556; this mechanism-local mismatch is retained without fitting.
Artifact:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_pair_live_v1.json`.

C7-S33AT was active: reconcile the live-probe inventory and stale-interface
claims, then select one uncaptured non-memory source with a bounded oracle and
native replay path. Preserved binaries and old matrix rows are not new live
stages until the current same-binary contract passes.

C7-S33AT completed that reconciliation. The inventory records the current
same-binary L1TEX refresh and control/store differential, and active L1TEX and
Allocate interfaces no longer require historical trace-only executables.

C7-S33AU completed a hardened fixed-Allocate same-binary slice. One rebuilt
8,192-generation cubin served seven correctness trials with elapsed/sink
oracles, full-shape Compute Sanitizer, NVBit, replay, and five
application-replay NCU runs for `w8_r10_native`. Replay and NCU match at
1,245,536 instructions; simulator lower is 336,374 cycles and NCU median is
593,194. No allocation/register-bank timing is fitted and dirty-writeback/
coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_allocate_live_v1.json`.

C7-S33AV was active: review `fixed_latency_front_end_footprint` next. Prefer a
bounded U1 same-binary cell; repair any short trace-twin interface before
capture and do not fit front-end timing from one cell.

C7-S33AV completed the front-end footprint same-binary refresh. The active
Makefile/campaign no longer requires the historical short trace pair; one
rebuilt 1,024-generation U1 fixed_retired cubin served seven correctness
trials with elapsed/sink oracles, full-shape Compute Sanitizer, NVBit,
deterministic replay, and five application-replay NCU runs. Replay and NCU
match at 32,852 instructions; simulator lower is 32,403 cycles and NCU median
is 38,629. No front-end timing is fitted and dirty-writeback/coherence tails
remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_front_end_footprint_live_v1.json`.

C7-S33AW was active: review `fixed_latency_result_order` as the next
same-binary non-memory source; preserve falsifiers and do not fit result-order
or issue-queue timing from one cell.

C7-S33AW completed the result-order same-binary refresh. One rebuilt
8,192-generation cubin served the `w8_hf` cell through seven correctness
trials with elapsed/sink oracles, full-shape Compute Sanitizer, NVBit, replay,
and five application-replay NCU runs. Replay and NCU match at 458,968
instructions; simulator lower is 197,490 cycles and NCU median is 257,751. No
result-order/issue-queue timing is fitted and dirty-writeback/coherence tails
remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_order_live_v1.json`.

C7-S33AX was active: review `fixed_latency_result_same_bank` next. Preserve
falsifiers and do not infer physical bank mapping or latency from one contrast.

C7-S33AX completed the same-bank result-order same-binary refresh. One rebuilt
8,192-generation cubin served the W8 HF cell through seven correctness trials
with elapsed/sink oracles, full-shape Compute Sanitizer, NVBit, replay, and
five application-replay NCU runs. Replay and NCU match at 458,960
instructions; simulator lower is 197,489 cycles and NCU median is 233,169. No
physical bank mapping or latency is inferred and dirty-writeback/coherence
tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_same_bank_live_v1.json`.

C7-S33AY was active: review `fixed_latency_reuse` next; preserve falsifiers and
do not fit register-reuse or scoreboard timing from one cell.

C7-S33AY completed the register-reuse same-binary refresh. One rebuilt
8,192-generation variant-A cubin served the W8/U11-A cell through seven
correctness trials with elapsed/sink oracles, full-shape Compute Sanitizer,
NVBit, replay, and five application-replay NCU runs. Replay and NCU match at
1,245,824 instructions; simulator lower is 336,953 cycles and NCU median is
635,271. No reuse/scoreboard timing is fitted and dirty-writeback/coherence
tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_reuse_live_v1.json`.

C7-S33AZ completed a corrected front-end IBuffer same-binary slice. One rebuilt
baseline_s1 cubin passed seven timeline/sink-oracle trials, full-shape Compute
Sanitizer, NVBit, replay, and five application-replay NCU runs. Replay and NCU
match at 944 target warp instructions; simulator lower is 818 cycles and NCU
median is 3,678. No IBuffer depth is fitted. Artifact:
`docs/sm86_c7_research_probe_front_end_ibuffer_depth_live_v1.json`.

C7-S33BA is historical and superseded: its stale analyzer/link reconciliation
was incorporated into the completed M3/M4 ledger without making historical
outputs execution prerequisites.

M1 corrected `fixed_latency_result_capacity`: `docs/sm86_c7_research_probe_fixed_latency_result_capacity_live_v2.json`. Strict parsing, per-trial sentinel/elapsed checks, same b9 cubin, seven correctness, Compute Sanitizer, NVBit/replay, and five NCU runs pass; replay/NCU match at 1,573,144 instructions, lower 590,830, median 851,446; capacity remains unidentified.

M2 corrected `immediate_constant_frontier`: `docs/sm86_c7_research_probe_immediate_constant_frontier_live_v2.json`. Independent modulo-u32 result oracle, timestamp bounds, correct thread/warp accounting, executable-relative cubin lookup, seven correctness, reduced Compute Sanitizer, same-cubin NVBit/replay, and five NCU runs pass; replay/NCU match at 151,872 instructions, lower 6,037, median 10,315; IMC timing remains observational.

C7-S33AZ completed a corrected front-end IBuffer same-binary slice. One rebuilt
`baseline_s1` cubin passed seven timeline/sink-oracle trials, full-shape
Compute Sanitizer, NVBit, deterministic replay, and five application-replay
NCU runs. Replay and NCU match at 944 target warp instructions; simulator
lower is 818 cycles and NCU median is 3,678. No IBuffer depth is fitted.
Artifact:
`docs/sm86_c7_research_probe_front_end_ibuffer_depth_live_v1.json`.

M3 is complete: touched campaign analyzers use the active long-generation
populations, historical short-trace names are no longer execution prerequisites,
and inventory latest-artifact links point to corrected M1/M2 and AQ–AZ records.

C7-S33BB is complete after the M4-E independent closeout recheck: all current
source families have a live vertical or an explicit non-native disposition;
failing, unsupported, and physically unobservable rows remain preserved.
