# Deterministic same-cycle event-batch milestone

**Completed:** 2026-08-20  
**Predecessor:** [measured simulator event throughput](../docs/simulator_event_throughput_milestone.md)  
**Goal:** reduce deep reply-less write queue memory without changing event order,
cache state, modeled timing, diagnostics, or public JSON

## Measured target

Ordinary RED depth 128 processes 695,604 system events, of which 695,520 are L2
accesses. Its event queue peaks at 695,520 entries and process RSS reaches about
429.5 MiB. Exactly 695,339 processed events immediately follow another event at
the same modeled cycle. STORE exhibits the same reply-less request shape.

The result does not justify a general timing wheel: sampled ordinary RED
attributes only about 4.6% flat self time to heap pop, and timestamp-distance
behavior differs sharply across the cohort. This milestone targets the measured
memory shape first.

## Candidate representation

Add a driver-internal event carrying a contiguous run of `MemRequest`s only
when all requests:

- come from one `SmCore::advance` or response-retry emission batch;
- have the same `issue_time` and target SM;
- are consecutive in original emission/insertion order; and
- are reply-less STORE or non-returning RED requests that cannot schedule a
  same-time response/fill event.

The batch handler walks requests in stored order through the existing single
request L2 transaction function. It must not reorder across SMs, timestamps,
pre-existing events, request groups, or feedback witnesses. Requests with a
reply, fill, intrinsic atomic completion, or mixed timestamp remain individual
events in the first implementation.

If factoring the L2 handler proves too invasive, evaluate an ordered per-source
merge heap with the same boundaries. Do not select a timing wheel or parallel
SM execution without new profile evidence.

## Correctness checks

- Differential unit tests compare batched and legacy scheduling for same-time
  STORE/RED streams, cross-SM ties, mixed timestamps, capacity feedback, and
  injected overflow/error cases.
- Every typed failure remains transactional at the same externally visible
  boundary; a partial batch may not commit earlier L2 mutations if the legacy
  kernel-level transaction would roll back.
- Normal and feature builds pass the full Rust suite.
- The 12-case benchmark produces byte-identical normal JSON before/after.
- The active 5,519-pair corpus is refreshed and passes schema/configuration,
  freshness, deterministic-twin, and byte checks.

## Performance acceptance

On the same pinned CPU and seven-round method:

- ordinary RED/STORE depth-128 maximum queue depth and peak RSS fall materially;
- ordinary RED/STORE median wall time does not regress by more than 3%;
- feedback, load/response, ATOMG, gap-heavy, and launch-chain medians show no
  cohort-wide regression; and
- exact feature counters explain changed event representation by reporting both
  logical requests and physical queue events.

The report must distinguish logical modeled events from physical queue entries;
batching is a representation optimization, not a claim that hardware events
disappeared.

## Outcome

Consecutive same-SM/same-time reply-less STORE/RED requests now share one
physical queue event and retain exact in-batch order. Timing-active feedback is
excluded because its queue was already bounded. Deep STORE/RED queue depth falls
96.8%, wall time 10.9%/10.3%, and peak RSS 9.8%/9.6%; the maximum matched cohort
regression is 0.48%. Batch payloads are bounded by live queue depth and reserve
their logical sequence span. All A/B outputs are byte-identical. See the
[completed milestone](../docs/simulator_same_cycle_event_batch_milestone.md).
The complete validation closes with 5,519 fresh deterministic replay pairs,
the full Rust/feature suites, and 60 affected Python tests.
