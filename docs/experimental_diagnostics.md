# Diagnostic and functional-model boundaries

**Updated:** 2026-10-12

This page describes the bounded diagnostic surfaces that remain useful while
replaying the real SM120/RTX 5090 corpus. They are observers or explicitly
opt-in model experiments; they are not hardware certification, simulator
acceptance gates, or substitutes for whole-workload performance evidence.

## Native replay diagnostics

`sim-run --sm120-alpha --mode flat` is the primary native path. The replay
consumes the NVBit dynamic trace, its control sidecar, captured static metadata,
and typed TMA/synchronization sidecars. It reports issue, dependency, memory,
async, launch, and residual counters without inventing missing hardware events.
`--fast-frontier-only` is a complementary lower-cost view and is run only after
full replay.

The following outputs are intentionally model-relative:

- issue/readiness and dependency frontiers;
- logical cache/service and route counters;
- writeback/coherence/visibility scenario frontiers when their explicit opt-in
  configuration is supplied;
- typed TMA payload and mbarrier/phase/wait observations;
- per-launch coverage, unsupported mechanisms, and replay-process timing.

A completion frontier is not a measured silicon latency. Candidate route labels
are not physical issue-port identities. NCU/CUPTI values are joined after replay
as independent observations and never enter replay configuration or tuning.

## Optional bounded model seams

The Rust crates retain small, explicitly named mechanisms for research questions
that are visible in a real trace:

- carded L1/L2/DRAM service and finite writeback state;
- resident-L1 visibility/invalidation and typed memory-order observations;
- TMA/LDGSTS payload accounting and barrier-sidecar semantics;
- custom `simt_u32_v1` and `simt_its_v1` functional executors for API/semantic
  exploration, separate from native SASS timing.

These seams fail closed on malformed or unsupported inputs and use checked
arithmetic. A fail-closed runtime error is implementation behavior, not an
acceptance test. No synthetic program, fixture bundle, digest receipt, schema
approval, or small-shape simulator run is a performance claim for the native
model.

## Functional examples versus acceptance evidence

`arch_profiles/specs` and `arch_profiles/programs` contain bounded example
inputs for the custom functional executors. They are useful for manually
exercising a code path, but they are not native-trace coverage and must not be
used to select model constants or claim generalization. The acceptance
population is the complete SOLBench plus modern/SOTA trace corpus under
`/root/sm120_rtx_5090_capture`, compared with independent NCU ground truth.

The only retained Cargo integration probes use a real trace or real sidecar
provided through the environment:

- `sim/sm/tests/machine_local_replay.rs`;
- `sim/sm/tests/machine_local_sidecars.rs`;
- `sim/trace/tests/machine_local_smoke.rs`.

They are ignored unless a real capture path is supplied. The workspace no longer
contains inline synthetic `#[test]` modules or test-only simulator fixtures.

## Reading a result

Keep these domains separate when interpreting a report:

1. **Model cycles/frontiers:** simulator outputs under the declared card and
   mode; useful for mechanism-local comparison.
2. **Replay-process wall time:** host parse, sidecar join, scheduling, and JSON
   serialization time; this is not GPU execution time.
3. **Uninstrumented CUDA-event time:** the numerator for replay-speed claims
   when it covers the same launch population.
4. **NCU duration/counters:** independent profiler observations under their
   collection protocol; they are not replay inputs.

When a domain is unavailable, report it as unavailable rather than substituting
tracer-host or NCU duration. Report canonical workload scope, launch joins,
sidecar coverage, residual gaps, and timing units beside every aggregate.

See [`runbook_host.md`](runbook_host.md) for executable full-corpus commands and
[`STATUS.md`](STATUS.md) for current coverage and irreducible evidence limits.
