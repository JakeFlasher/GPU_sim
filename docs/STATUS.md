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
No new NCU observations were collected or scored. A5 is next, conditional on
clean A1/A2 eligibility. The primary outcome is NCU
accuracy; simulator wall-clock runtime remains secondary, with no
compression/speedup metric.

The current A5 preflight was run against that manifest/prediction: the
prediction is valid and no observation exists, but the identity is ineligible
because the worktree is dirty. No NCU workload command was launched.

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
