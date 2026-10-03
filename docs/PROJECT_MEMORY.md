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

## Retained evidence

The 2026-09-30 cold/warm campaigns completed 15/15 workloads on RTX 3090 Ti / SM86,
UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09,
1350/10501 MHz, P0. This is historical host identity, not a current-device check.
Raw records: `results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/`.
Preserve datasets, source, tests, and scientific limitations; do not recreate
removed policy text or workflow machinery.
