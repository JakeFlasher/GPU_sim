# Simulator event-throughput milestone

**Completed:** 2026-08-19  
**Predecessor:** `sm_86` readiness schemas 1–13 and the 5,519-pair active replay corpus  
**Goal:** measure absolute simulator cost, identify the dominant execution path,
and optimize it without changing modeled results

## Why this is next

The focused research notes explicitly withhold every simulator-speed claim: the
current binary-heap event queue is functional, but wall time, modeled
instructions/s, events/s, peak memory, and decode-versus-replay cost have never
been measured. Timing wheels, batching, and parallel execution are hypotheses,
not established improvements. With the Ampere causal path now implemented
through schema 13, cost measurement is the nearest independent milestone.

## Baseline cohort

Use only retained, full-coverage traces and the final release `sim-run`. The
cohort spans input size and event mix rather than selecting only fast cases:

- one short and one chained launch-cycle trace;
- streaming and reuse LOAD/STORE retirement traces at depth 64;
- arrival STORE and RED traces at high warp pressure;
- admission control, STORE, RED, and ATOMG at depth 128; and
- one gap-64 admission trace to expose decode-heavy input (about 180 MiB).

Record input bytes, decoded dynamic warp instructions, simulated cycles, mode,
and candidate configuration beside every timing. Run one filesystem warm-up
followed by seven measured executions on one pinned CPU. Report medians and the
complete min/max range; do not turn the range into a confidence interval.

## Measurements

1. End-to-end elapsed time and peak resident memory from `/usr/bin/time -v`.
2. User/system CPU time, cycles, retired host instructions, branches, and cache
   misses from `perf stat` where the local kernel permits them.
3. Opt-in benchmark-only counters for event pushes, pops, maximum queue depth,
   same-cycle batches, decoded records, and modeled warp instructions.
4. Decode, control-sidecar join, replay, and JSON-serialization phase time.
   Phase timers and event counters must stay outside normal result JSON.
5. A sampled `perf record` profile on the largest control, RED, and ATOMG cases.

Throughputs are derived from measured numerators and wall time: decoded
records/s, modeled warp instructions/s, and popped events/s. End-to-end and
replay-only rates are reported separately.

## Optimization order

Act on the profile rather than presupposing the event heap is dominant:

1. If protobuf reconstruction/allocation dominates, benchmark arena or
   launch-at-a-time decoding and drop decoded launch state before loading the
   next launch.
2. If readiness scans or diagnostic copies dominate, remove redundant work
   while preserving candidate-off and candidate-on behavior.
3. If event-heap operations dominate, first batch consecutive equal-time events
   behind the existing `(time, insertion-sequence)` API. Evaluate a timing
   wheel only if the measured timestamp-distance distribution supports it.
4. Consider SM-parallel execution only after proving that launch-global L2 and
   deterministic same-time ordering are not silently changed.

One material change is evaluated at a time. The optimized binary must produce
byte-identical normal JSON for the baseline cohort and pass the full Rust and
affected Python suites. Performance acceptance is a reproducible improvement
in median end-to-end time or peak memory on at least one dominant large case,
with no material cohort-wide regression; exact magnitude is reported rather
than selected in advance.

## Deliverables

- a machine-readable baseline and optimized measurement table;
- a short profile report naming dominant symbols and phase shares;
- benchmark-only event/phase instrumentation;
- the smallest profile-supported optimization; and
- refreshed active replay validation if shared simulator source changes.

This milestone needs CPU profiling, not a new GPU capture. NCU and NVBit remain
the independent hardware/reference and trace sources for accuracy; they are not
simulator-speed measurements.

## Outcome

The 12-case baseline, exact event counts, phase timing, and sampled profiles are
retained under
`results/captures/simulator_event_throughput_20260819_v1`. Static phase-marker
classification plus a transactional feedback issue plan reduce schema-10/11/13
wall time by 5.6/27.9/29.4% and preserve exact normal JSON. The binary heap is
not a universal dominant hotspot. The successor is bounded same-cycle queue
footprint reduction for deep reply-less writes; full results and limitations are
in [the completed milestone](../docs/simulator_event_throughput_milestone.md).
