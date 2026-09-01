# SM86 NCU-grounded demo benchmark milestone

## Goal Description

Produce a compact, reproducible RTX 3090 Ti benchmark table analogous in shape
to published simulator-comparison tables, using this project's own functional
workloads and the current `NVBit -> trace -> sim-run` path. NCU is the sole
hardware ground-truth source; NCU values must never enter architecture cards,
simulator parameters, feature selection, or implementation choices.

The final report must distinguish simulator accuracy from simulator cost. It
must report raw GPU execution time, NCU profiler-process wall time, simulator
process wall time, `GPU execution / simulator wall` ratio (normally below one),
and `NCU profiler wall / simulator wall` ratio (tool-overhead diagnostic). It
must not call either ratio a speedup without naming numerator and denominator.

## Acceptance Criteria

- AC-1: Functional workload matrix
  - Positive: define a frozen 6-8 row matrix from existing short RTX-3090-Ti
    workloads, led by BF16 GEMM, BF16 attention, convolution, sparse GEMM,
    normalization, elementwise, and INT8 compute; every row has a correctness
    command, target-launch count, shape, rationale, and binary identity.
  - Positive: reuse existing workload implementations and shorten only when a
    row cannot complete NCU, NVBit, and replay within bounded demo time.
  - Negative: do not relabel a microbenchmark as GPT/BERT/BFS itself; describe
    it as a functional analogue or kernel snapshot.

- AC-2: Local SM86 profile and host identity
  - Positive: validate or create a local RTX 3090 Ti `sm_86` profile binding
    device index, name, UUID, compute capability, driver, L2 geometry, SM count,
    and observed clock state.
  - Positive: record pre/post GPU identity for every live phase.
  - Negative: do not invoke `sudo`; if clock locking is required, emit exact
    manual `nvidia-smi` commands and continue with an explicitly unlocked or
    observed-fixed status.

- AC-3: Fresh NCU ground truth
  - Positive: collect matched application-replay NCU rows for every target
    launch with finite watchdogs and repeated measurements.
  - Positive: collect at least GPC elapsed cycles, SM active cycles, GPU kernel
    duration, executed instructions, global load/store sectors, L1 read and
    unified hit rates, L2 hit rate, and DRAM read sectors where available.
  - Negative: reject missing, duplicate, nonfinite, or launch-mismatched NCU
    metrics rather than filling them with zero.

- AC-4: Fresh trace and deterministic replay
  - Positive: capture each selected launch with the retained NVBit runtime,
    exact binary/control/trace identity, full cardinality, and empty unsupported,
    unmapped, and cross-allocation sidecars.
  - Positive: replay the current release simulator twice and require byte-equal
    outputs, exact instruction conservation, and bounded wall-time records.
  - Negative: historical JSON may guide workload selection but cannot be
    reported as a current replay without current binary/card/trace binding.

- AC-5: Comparable simulator metrics
  - Positive: expose or derive simulator predictions for executed instructions,
    global load/store sectors, L1 read hit rate, unified L1 hit rate, L2 hit
    rate, DRAM read sectors, issue frontier, and terminal modeled-event frontier.
  - Positive: preserve availability/typed-tail status for every metric.
  - Negative: incomplete terminal frontiers remain named lower bounds; no
    NCU-derived correction, per-workload offset, or silent tail clearing.

- AC-6: Accuracy and timing formulas
  - Positive: per launch, compute APE for positive count/cycle targets; for hit
    rates report both MAPE (only when NCU rate is positive) and absolute
    percentage-point error.
  - Positive: per workload, report launch-mean MAPE; total rows use explicitly
    launch-weighted aggregation. Missing metrics carry coverage counts.
  - Positive: report `gpu_kernel_seconds / simulator_wall_seconds` and
    `ncu_profile_wall_seconds / simulator_wall_seconds` separately.
  - Negative: never average independent cycle frontiers into one selection
    scalar and never define MAPE on a zero denominator.

- AC-7: Final table and evidence
  - Positive: emit machine-readable JSON plus a concise Markdown report with a
    primary workload/launch/time/cycle-MAPE/counter-MAPE table and a secondary
    L1/L2/traffic accuracy table.
  - Positive: include workload and launch totals, metric coverage, commands,
    artifact identities, current revision, hardware identity, and caveats.
  - Positive: produce a smoke report before opening the full matrix.

- AC-8: Verification and durable handoff
  - Positive: run focused unit tests, existing affected tests, report
    regeneration/validate-only, and one complete executable smoke path.
  - Positive: commit each RLCR round with a clean worktree and write the required
    Humanize round summary.
  - Negative: do not add RLCR state, review receipts, or process artifacts to
    the benchmark result or simulator authority path.

## Path Boundaries

### Upper Bound

- Eight compact existing workloads, three NCU repeats, one exact trace per
  target launch, deterministic simulator twins, current SM86 cards, full metric
  and timing report, and any small output/metric plumbing required for honest
  comparison.

### Lower Bound

- Four representative rows (BF16 GEMM, BF16 attention, one memory-heavy or
  normalization kernel, one sparse/INT8 kernel), one NCU repeat for smoke, exact
  trace/replay twins, and the complete report schema. The full matrix opens only
  after this vertical slice passes.

### Allowed Choices

- May use existing `portable_lcd`, `simbench_bw`, live-cohort, NCU collector,
  NVBit capture, static-control, and simulator tools.
- May add a focused benchmark campaign/reducer and optional direct-run metric
  projection output.
- May use NCU without sudo.
- Must use `arch_profiles/cards/sm_86.json` and
  `data/pchase/sm_86_static_card.json`, or a content-bound local derivative that
  changes no value based on scored NCU workloads.
- Must not use JADE/HETR or CPU-GPU co-simulation.
- Must not compare against Accel-Sim in the headline; NCU is ground truth.

## Dependencies and Sequence

1. Inventory and freeze the workload/metric schema.
2. Implement the smallest missing metric/timing output and campaign runner.
3. Run correctness and a four-row smoke NCU/NVBit/replay slice.
4. Generate and validate the smoke report; decide whether the full matrix is
   executable and scientifically comparable.
5. Run remaining rows and independent NCU repeats.
6. Generate final JSON/Markdown tables, validate from retained artifacts, run
   tests, and commit the clean handoff.

## Implementation Notes

- Preserve raw per-launch values. Summaries never replace source rows.
- Time the whole external NCU and simulator processes with a monotonic host
  clock; use NCU `gpu__time_duration.sum` only for kernel execution time.
- A high MAPE is a valid result. Missing model coverage is reported, not hidden.
- Existing simulator dirty-writeback/coherence tails remain explicit unless an
  independently completed mechanism truly clears them.
