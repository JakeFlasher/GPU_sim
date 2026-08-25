# Completed milestone: simulator event throughput and feedback hot paths

**Status:** complete 2026-08-19; absolute cost measured, two profile-supported optimizations retained  
**Scope:** release `sim-run` on retained full-coverage `sm_86` traces  
**Evidence:**
`results/captures/simulator_event_throughput_20260819_v1/README.md`

## Goal and method

The project previously described event-driven execution as an efficiency
hypothesis without measuring it. This milestone establishes absolute cost
before changing the queue design. Twelve retained cases span launch chains,
streaming and reused memory, arrival pressure, deep STORE/RED/ATOMG bursts, a
decode-heavy gap arm, and matched schema-10/11/13 feedback layers.

Clean timing pins release `sim-run` to performance CPU 8, warms every case once,
then runs seven deterministically shuffled complete rounds. Each run records
wall/user/system time, peak RSS, decoded records, modeled instructions, and
exact normal-JSON byte stability. Separate feature builds report exact event
counts and decode/replay/serialization phase times so instrumentation does not
contaminate the clean wall-time medians. Sampled profiles use `perf record` at
499 Hz with DWARF call graphs.

## Absolute optimized cost

Across the cohort:

- wall time ranges from 0.222 to 3.131 seconds;
- modeled instruction throughput ranges from 0.571 to 3.518 million/s;
- event throughput ranges from 0.001 to 0.813 million/s; and
- peak RSS ranges from 41.1 to 582.7 MiB.

These ranges describe different event mixes, not one universal throughput.
The eight-launch control chain executes 697,536 warp instructions through only
672 system events (1,038 instructions/event), while deep ATOMG processes
1,405,236 events for 4,094,496 instructions. The decode-heavy gap case processes
only 179,508 events but spends about 46.6% of its run in trace loading.

Deep ordinary RED schedules 695,604 events and reaches a 695,520-entry queue;
695,339 processed events immediately follow another event at the same modeled
cycle. Feedback-coupled RED processes 779,100 events, including 83,580 SM wakes,
but its yield/re-arm choreography caps the queue at 2,772 entries. This explains
why feedback replay is slower yet uses about 332 MiB instead of ordinary RED's
429 MiB. It also identifies deterministic same-cycle batching as a memory
optimization target without claiming that heap cost dominates every workload.

Phase timing reinforces the distinction. Replay consumes about 81% of
schema-10/11/13 runtime and 91% of streaming-load runtime; trace loading consumes
about 39% of ordinary deep RED and 47% of the gap-heavy case. Pretty JSON
serialization is negligible for these compact reports.

## Profile-driven changes

Baseline ordinary RED spent about 4.6% flat self time in binary-heap pop, so a
timing wheel was not justified as the first change. The feedback profiles found
larger removable costs:

- schema 11/13 spent roughly 18–20% in Unicode-aware iteration while repeatedly
  recognizing canonical static `CS2R ..., SR_CLOCKLO` and `MEMBAR.SC.GPU` text;
- schema 10/11/13 cloned the boxed feedback state and its `VecDeque`s at every
  issue transaction.

Static phase-marker booleans are now computed once when `IssueSlot`s are built.
The issue transaction now preflights a compact four-SMSP reservation/clock plan
and applies it only after all other issue work succeeds, instead of cloning the
whole feedback state. The plan retains same-cycle shared-capacity accounting,
typed overflow errors, and transactional failure behavior.

| replay layer | before | after | speedup | wall-time reduction |
|---|---:|---:|---:|---:|
| ordinary RED d128 | 1.517 s | 1.504 s | 1.008× | 0.8% |
| schema 10 | 3.263 s | 3.080 s | 1.059× | 5.6% |
| schema 11 | 4.342 s | 3.131 s | 1.387× | 27.9% |
| schema 13 | 4.396 s | 3.106 s | 1.416× | 29.4% |

All other ordinary-case medians improve by 0.7–1.3% in the retained run. Peak
RSS remains in the same class. Direct comparisons across ordinary, schema 10,
schema 11, and schema 13 preserve exact normal JSON bytes.

The final default binary passes the complete Rust workspace, both feature-build
test suites, Rust formatting, and 60 affected Python tests. All 5,519 active
replay pairs were regenerated after the optimization and pass freshness, exact
schema/configuration identity, deterministic-twin, and byte checks.

## Instrumentation boundary and successor

`sim-run/performance-counters` is an explicit build feature, additionally gated
at runtime by `SIM_RUN_PERFORMANCE_COUNTERS=1`. It emits event and phase
diagnostics on stderr and adds nothing to normal result JSON or the
default-build hot path. Exact counting and clean timing are separate runs.

The optimized final profiles are led by issue-cycle/readiness work and L1
mutation preflight, not phase parsing, feedback cloning, or a universal event
heap bottleneck. The next bounded milestone should reduce the extraordinary
same-cycle queue footprint on deep reply-less STORE/RED traffic while retaining
the exact `(time, insertion-sequence)` order. A same-time batch or ordered
per-source merge should be evaluated first; a timing wheel advances only if
timestamp-distance evidence supports it. Acceptance requires byte-identical
results, lower peak RSS on deep writes, and no cohort-wide time regression.

No GPU recapture or sudo was needed. Direct tracefs/uprobe registration was
unavailable under the local permission policy, so the completed implementation
uses feature-gated internal counters instead.
