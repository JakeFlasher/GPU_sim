# Continuation memory

Updated 2026-10-02. Start here, then [STATUS](STATUS.md); read only the
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
- R5 is complete for the closed native issue-model seam: native `SmCore`
  reference tests consume typed-macro/exact-replay decisions, compare macro
  entry/exit with detailed replay, and resume a detailed suffix. R5 also
  captures native reference pre/post typed boundaries and detailed issue
  effects. The pure-scalar SM86 probe has a complete pre/post pair with two
  effects; the strict independently derived scalar transition rejects the
  front-end state as not closed and therefore preserves exact fallback.
- The reference `SmCore` issue actor now consumes the comparator-selected
  typed-macro/exact-replay path; a native issue-model test compares macro entry,
  exit, and detailed continuation state. The real SM86 reference probe still
  selects exact replay because its front-end state is open.
- The closed native issue-model seam is exercised with a two-instruction scalar
  macro and detailed suffix; front-end-enabled launches remain exact fallback.
- Service profiles now mark capacity-adjusted throughput slopes and carry
  explicit residual/replacement/independent service roles plus a serial queue
  projection; the v2 probe fitter requires explicit dependent latency and
  independent throughput, retains queue-depth/capacity interventions as
  explicitly unmodeled (no queue equation), and refuses aggregate-only
  latency identification; occupancy/contention rows remain explicitly
  unmodeled.
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
- R8.2 is complete for a narrow independently witnessed issue-model seam. The
  reference event loop consumes a validated `TypedMacro`/`ExactReplay` result
  at `Ev::SmWake`, arms the next wake, resumes a detailed suffix, and refuses
  before mutation when the typed contract/effect ledger is perturbed. The
  retained front-end-enabled corpus has no such witness and remains exact
  fallback; dispatch counters in the reference footer are not speedup.
- R8.3 is complete as a conservative native closure rule: the dispatcher
  exact-falls back for non-unit completion tails, pending tokens, or changes in
  scalar/memory/cache/front-end/response/barrier/local-completion state that
  its actor does not install. Model-local latency certificates remain valid at
  their own evidence level; native latency/completion modeling is not claimed.
- R6 is complete: `docs/sm86_reification_corpus_v1_20261002.json` defines a
  15-workload, six-family SM86 corpus with no family crossing; prediction,
  observation, and join artifacts are retained under
  `docs/review_round3_20261002/`. The pre-observation output uses the declared
  observer projection, records detailed replay plus a latent sidecar, and
  reports zero native macro dispatches/step reduction. The post-prediction join
  enforces row-level metric/endpoint/clock/hardware identity and reports open
  intervals separately from finite coverage.

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
exists, but native `SmCore` remains untouched. R3 is separating witness
extraction from descriptor eligibility and now refuses stale/self-witnesses;
R4 now separates service roles and queue/assignment uncertainty. R5 validates
a consumed closed native issue-model boundary; full front-end-enabled traces
still record exact-replay plans. Current rankings, open
intervals, and descriptor/macro probe remain negative/diagnostic evidence and
black-box internals are not execution prerequisites. R7 rebuttal/
claim-discipline is complete; R8.0 review intake, R8.1 soundness guards, and
R8.2 witnessed production reference dispatch and R8.3 native state closure are
complete. R8.4 service identifiability is next; use
[`third_round_rebuttal_20261002.md`](../plans/third_round_rebuttal_20261002.md)
for the prior evidence map and
[`latest_review_refinement_20261002.md`](../plans/latest_review_refinement_20261002.md)
for the active stage ledger and remaining falsifiers.

## Retained evidence

The 2026-09-30 cold/warm campaigns completed 15/15 workloads on RTX 3090 Ti / SM86,
UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09,
1350/10501 MHz, P0. This is historical host identity, not a current-device check.
Raw records: `results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/`.
Preserve datasets, source, tests, and scientific limitations; do not recreate
removed policy text or workflow machinery.
