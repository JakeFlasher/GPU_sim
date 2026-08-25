# Completed milestone: deterministic same-cycle L2-access batching

**Status:** complete 2026-08-20; narrow batching retained  
**Predecessor:** [simulator event throughput](simulator_event_throughput_milestone.md)  
**Evidence:**
`results/captures/simulator_event_throughput_20260819_v1/event_batch.md`

## Goal and representation

The measured ordinary deep-write path queued one physical event per reply-less
L2 request: 695,520 entries for STORE and RED depth 128, even though 99.96% of
processed events immediately followed another event at the same modeled cycle.
Heap pop was not a universal CPU hotspot, but this representation consumed
substantial memory and avoidable scheduling work.

The retained implementation batches only a consecutive run of requests that:

- came from one SM actor advance or response-retry emission;
- target one SM and carry one `issue_time`;
- are reply-less STORE or non-returning RED requests; and
- preserve their exact original emission/insertion order inside the batch.

Loads, response-bearing requests, fills, ATOMG, mixed timestamps, and cross-SM
ordering remain individual events. Timing-active schema-10/11/13 feedback also
keeps the legacy path because its yield/re-arm choreography already caps queue
depth; batching its two-request groups would add grouping work without
addressing the measured problem.

The batch handler walks the existing single-request L2 transaction in stored
order. Kernel-private hierarchy state still rolls back on any later request
error. Each queue event directly owns its boxed request slice, so metadata is
bounded by live queue depth rather than the historical number of batches.
Although one physical event represents multiple requests, the event engine
reserves the same number of logical insertion-sequence positions; FIFO order
and typed sequence exhaustion therefore match the unbatched representation.

## Exact event result

Logical modeled work is unchanged. Feature diagnostics now distinguish logical
events from physical queue entries and report batch/request counts separately.

| case | logical events | physical queue events | max queue before→after |
|---|---:|---:|---:|
| ordinary STORE d128 | 695,604 | 22,260 | 695,520→22,176 |
| ordinary RED d128 | 695,604 | 22,260 | 695,520→22,176 |
| RED gap64 | 179,508 | 6,132 | 179,424→6,048 |
| arrival STORE | 172,116 | 1,428 | 172,032→1,344 |
| reused STORE d64 | 172,116 | 5,460 | 172,032→5,376 |
| schema 10/11/13, each | 779,100 | 779,100 | 2,772→2,772 |

Deep STORE/RED physical events and maximum depth fall 96.8%. Arrival STORE falls
99.2%, and reused STORE falls 96.8%. Common instrumentation stores explain the
small batches observed in control and ATOMG cases; target ATOMG traffic remains
individual.

## Matched performance

One feature binary ran alternating legacy/batched conditions on pinned CPU 8.
Each condition received one warm-up and seven measured repetitions; case and
condition order were balanced, and normal JSON had to match byte-for-byte.

| case | wall-time change | peak-RSS change |
|---|---:|---:|
| ordinary STORE d128 | -10.9% | -9.8% |
| ordinary RED d128 | -10.3% | -9.6% |
| arrival STORE | -8.5% | -14.0% |
| reused STORE d64 | -12.9% | -19.0% |
| RED gap64 | -1.9% | -0.1% |

The maximum cohort wall-time regression is 0.5% (launch-chain control), inside
run noise and far below the predeclared 3% bound. Feedback uses no batches and
its paired median changes range from -0.31% to -0.14%. Every targeted case
improves both time and memory.

## Validation and successor

Focused tests pin same-time and cross-SM FIFO order, logical sequence-space
reservation, the narrow operation boundary, the legacy comparison path,
logical/physical counters, and rollback when a later request inside a batch
fails. The A/B harness proves the feature build is active, binds the binary's
stable file identity during capture, retains one canonical JSON output per
case, and reduces adjacent condition pairs rather than ratios of independent
medians. The complete Rust workspace, normal and feature builds, formatting,
and 60 affected Python tests pass. All 5,519 active replay pairs were
regenerated against the final default binary and pass freshness, nested
schema/configuration identity, deterministic-twin, byte, and no-extra-output
checks.

Batching is a representation optimization: it does not claim that logical GPU
memory work disappeared. The remaining largest memory case is the gap-heavy
trace at about 571 MiB, and trace loading consumes roughly half its runtime.
The next bounded milestone should reduce full-snapshot trace decode/storage
cost—prefer launch/block-at-a-time lifetime reduction—before further event-heap
work. Any decoder change must preserve exact replay bytes and all wire-admission
error precedence.

No sudo or GPU recapture was required.
