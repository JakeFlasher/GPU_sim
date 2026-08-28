# Rejected SM86 LDGSTS front-end composite

**Status:** complete 2026-08-28; `not_retained`; candidate surface retired

**Architecture:** `sm_86` / RTX 3090 Ti

**Frozen plan:**
[`plans/sm86_ldgsts_front_end_milestone.md`](../../../plans/sm86_ldgsts_front_end_milestone.md)

**Machine-readable result:** [`evaluation.json`](evaluation.json)

## Candidate tested

The tested model was this project's LDGSTS-only composition of published
Ampere measurements, not Huerta et al.'s complete multistage memory pipeline.
It modeled four physical queue slots plus a dispatch latch as five effective
outstanding positions per SMSP, a four-cycle SMSP initiation frontier, a
two-cycle SM-shared acceptance frontier, 13-cycle WAR visibility, and a
39-cycle source-completion floor.

For an effective LDGSTS issued on SMSP `s` at cycle `t`, the frozen equations
were:

```text
u       = max(t, L[s])
L[s]    = u + 4
v       = u + 13
a       = max(v, G)
G       = a + 2
delta   = a - (t + 13)
WAR     = a
source  = max(existing_response + delta, t + 39 + delta)
```

Only `delta` and the issue-anchored source floor were composed with the
existing async-copy response. No new delayed hierarchy event was inserted for
an already issued request, and the existing per-request hierarchy, payload,
destination-wavefront, and destination-service equations were reused. The
queue could still shift cross-instruction issue and cache order; production
int8 traffic drift demonstrates that boundary. The result therefore tested an
observational readiness composition, not physical request-arrival
backpressure.

## Frozen evaluation

The CPU-only evaluation ran 24 active v4 cells, all 18 coupled warm-L2 cells,
the W1-W8 guard, and six descriptive production cells. Eight c00/c01 controls
with no effective LDGSTS reused their retained values. Simulator last-issue
cycles were normalized by 64 groups and live active-SM medians by 16,384
groups. No GPU tool, fitting, adaptive parameter selection, or hash comparison
participated.

| Gate | Baseline | Candidate | Result |
|---|---:|---:|---|
| 30-cell transfer MAE, cycles/group | 43.803247 | 49.439591 | fail |
| W1-W8 guard MAE, cycles/group | 14.377197 | 15.037354 | fail |
| strict coupled route order | 6/6 live | 5/6 | fail |
| v4 corrected interaction signs | 10 nonzero live | 7/10 | fail |
| coupled c11-c10 signs | 9 positive live | 9/9 | pass |
| exact traffic/cardinality | baseline exact | 55/56 candidate replays exact | fail |

The transfer set contains 30 campaign-qualified observations even though
eight raw cell names overlap. Eleven candidate errors improve, eleven worsen,
and eight are unchanged. The largest regressions are the two independent W8
c11 L1 observations, each adding 47.671875 cycles/group of absolute error.
The complete per-cell values and errors are in `evaluation.json`.

One route group fails: coupled W8 c10 predicts equal L1 and L2 time
(391.140625 cycles/group), while live time is strictly ordered
409.141785 < 525.158630 < 625.804504 for L1/L2/DRAM. Three corrected
interactions reverse sign: W2 L1 gap 0 and W4 L1 gaps 0/64. Their candidate
values are +26.500000/+139.640625/+139.640625; live values are
-10.969971/-15.025513/-9.968933.

All 50 probe replays preserve instruction and traffic accounting. The
production int8 replay preserves instruction cardinality but changes aggregate
traffic, so structural accounting from all 56 executed replays correctly
fails the exactness gate. Production timing labels and MAE/MAPE remain
verdict-inert.

## Mechanical sanity

Each of the six W4 c10 L1/L2/DRAM gap-0/64 sanity cells issues 2,048 modeled
tokens, reaches effective occupancy five, reports four queue-blocked issues,
and accumulates 502 excess-delay cycles. This establishes that the candidate
executed as specified and that the no-go is scientific, not a dormant-mode or
cardinality failure. The route- and gap-identical pressure also exposes the
model's missing route/overlap coupling.

## Verdict-inert production reference

Descriptive production MAPE moves from 33.127762% to 32.900960%, and MAE from
2,559.0 to 2,536.67 cycles. Only bf16, fp16, and int8 move; three workloads are
unchanged, all-six error improvement is false, and int8 traffic drifts. These
labels selected no parameter and do not replace the 33.128% development
baseline.

## Decision and cleanup

The terminal verdict is `not_retained`, with five failed gates: transfer MAE,
guard MAE, strict route order, corrected interaction signs, and all-replay
accounting. The historical active implementation is in commits `044a350` and
`8d4a938`; the corrected evaluator/result checkpoint is `96007d6`. The
historical CLI token was `--ldgsts-front-end ampere-queue-v1` and its report
identity was `ampere_queue_v1`.

The selector, queue/source-timing state, schema-14 diagnostics, candidate-only
Rust tests, active workload runner, and derived binaries were removed after
this decision. The corrected evaluator and focused tests are archived beside
this result and require checkpoint `96007d6` for replay execution. The
pre-existing LDGSTS source routing, payload, destination projection, and
destination service remain. The frozen plan, archived evaluator/result, and
Git history are the reproducible negative evidence; current ordinary replay
has no LDGSTS front-end candidate command.

## Claim ceiling and next discriminator

This result rejects only the project's simplified LDGSTS-only composition on
the frozen GA102 cohorts. It does not falsify Huerta et al.'s full model, any
individual published constant, or the existence of a hardware front-end
resource. It supplies no validated-Ampere or production cycle model, P3 pass,
Jade milestone progress, M4/G1 authorization, or publication-readiness claim.

A future standalone GPU experiment may use a prospective, barrier-free
burst-release probe crossing depths 1/4/5/6/8, one- versus four-SMSP placement,
and L1/L2/DRAM routes while holding destination work fixed. Its question is
whether slots release at route-invariant shared acceptance or at a later
route-sensitive stage. Published constants remain frozen, and the rejected
composite must not be retuned on the consumed labels. This negative result
alone does not scientifically justify a fresh live campaign.
