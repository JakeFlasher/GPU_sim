# Simulator event-throughput probe

This host-side probe measures absolute end-to-end `sim-run` cost on twelve cases
drawn from nine retained full-coverage `sm_86` traces. It spans launch chains, streaming/reuse memory,
arrival pressure, deep STORE/RED/ATOMG bursts, a decode-heavy gap case, and
matched schema-10/11/13 feedback layers.

The default run pins the simulator to performance CPU 8, warms each case once,
then executes seven deterministically shuffled complete rounds. It records wall
and CPU time, per-process peak RSS, page faults, context switches, decoded
records/s, modeled instructions/s, and exact normal-output byte stability.

```bash
.venv/bin/python workloads/research_probes/simulator_event_throughput/benchmark.py
```

The output is
`results/captures/simulator_event_throughput_20260819_v1/baseline.{json,csv}`.
This is an absolute-cost baseline, not yet an event/s result: exact event counts,
phase timing, and sampled `perf` attribution are the next instrumentation step.

The retained same-cycle representation study uses one feature binary for
alternating legacy/batched A/B runs:

```bash
throughput_target=$(mktemp -d)
cargo build --release --manifest-path sim/Cargo.toml -p sim-run \
  --features performance-counters --target-dir "$throughput_target"
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/compare_batching.py \
  --binary "$throughput_target/release/sim-run"
```

The harness clears ambient measurement controls, requires cross-condition JSON
byte equality, and records both condition orders. `analyze_event_batch.py`
combines this A/B result with logical/physical event counters and phase timing.

The direct trace-lifetime comparison uses the same matched method:

```bash
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/compare_trace_lifetime.py \
  --binary "$throughput_target/release/sim-run"
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/analyze_trace_lifetime.py
```

Raw-retaining and compact conditions run through one feature binary with
performance logging disabled; each pair must emit identical normal JSON.

The next lifetime study first compares full-map and exact-generation streaming
compaction, then compares the predecessor and compact-identity ordinary
binaries:

```bash
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/compare_streaming_loader.py \
  --binary "$throughput_target/release/sim-run" --repeats 3
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/analyze_streaming_loader.py
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/compare_compact_identity.py \
  --baseline-binary /path/to/predecessor/sim-run \
  --candidate-binary /path/to/candidate/sim-run
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/analyze_compact_identity.py
```

Streaming is feature-only and opt-in through the comparison harness. The
20-byte compact identity is the retained default representation.

Exact event composition uses the feature-gated internal counter:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run \
  --features performance-counters --target-dir "$throughput_target"
.venv/bin/python \
  workloads/research_probes/simulator_event_throughput/collect_event_counts.py \
  --binary "$throughput_target/release/sim-run"
```

The collectors set `SIM_RUN_PERFORMANCE_COUNTERS=1`; the feature build is quiet
without it. It emits diagnostics only on stderr, while default result JSON and
default-build hot paths are unchanged. Clean timing and instrumented counting
are separate executions.
