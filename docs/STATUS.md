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
| Calibration | Profile loader and offline fitter exist; v2 separates latency/throughput/queue probes, retains throughput interval endpoints, and records occupancy/contention as explicitly unmodeled dimensions; supplied rates remain uncalibrated |
| Accuracy | R6 has an identity-checked six-family holdout join; all 15 native intervals are open, so finite coverage is not applicable and no hardware-accuracy claim is made |
| Reification | Typed region contract and native-boundary comparator preserve only complete exact projections; incomplete canary boundaries fall back to replay |
| Selective coverage | Multi-warp/order and completion-multiplicity guards refuse conservatively; fast-frontier output includes full-trace candidate/reified/uncovered counts |
| Model macro seam | `sim-sm::ModelExecutor` consumes a certified scalar `MacroTransition`, emits explicit effects/pending completions, and resumes detailed descriptors; this is model-local |
| Native refinement | Closed native `SmCore` issue-model seam preserves a two-instruction region and resumes detailed continuation through comparator-selected paths; paired native reference boundaries/effects and exact fallback remain active for front-end-enabled probes |
| Service composition | Typed residual/replacement/independent-post-replay roles and a serial queue projection are explicit; full-trace demand is currently independent post-replay, so it does not silently alter the observer endpoint |
| Cache candidates | Warp-local instrumentation order can be observed when decoded identities are complete; cross-warp order, fence scope, line generation, owner/read-from, and hardware completion remain explicit unknowns |
| Design evaluation | Fifteen-workload, six-family calibration/development/held-out NCU join runs post-prediction; v2 target contracts check metric/endpoint/clock/cache/launch/repetition/GPU identity and family split semantics; finite intervals, open refusals, model-local partial-order ties, and model-only step reduction are reported, while native front-end traces remain exact/refused |
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
transition and exact continuation. R5 consumes a closed native issue-model
transition, while full front-end-enabled launches remain exact fallback. R6
expands the corpus and confirms that native compression, finite observer
uppers, and hardware accuracy remain unproven; no model-only step reduction is
relabeled as native compression.
Report finite intervals, open/refused cases, OOD behavior, actual execution
steps, memory, and runtime separately; a passing unit suite is not accuracy
evidence. The R6 corpus and third-round claim map are in
[`review_round3_20261002/README.md`](review_round3_20261002/README.md) and
[`third_round_rebuttal_20261002.md`](../plans/third_round_rebuttal_20261002.md).

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
