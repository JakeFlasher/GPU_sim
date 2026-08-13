# Historical sm_86 measurement regime v2.1

Use this card only to interpret or reproduce the July 2026 RTX 3090 Ti label
set. It is not a current project protocol and does not prevent experiments
with different tools, clocks, drivers, or capture settings.

## Recorded configuration

The historical v2.1 runs used an RTX 3090 Ti, driver 610.43.02, persistence
mode, a 1350 MHz core lock, a 10501 MHz memory lock, and these NCU options:

```text
--clock-control none --cache-control all --replay-mode kernel
```

Sidecars recorded the selected settings and post-capture host observations.
When comparing directly with that label set, reproduce these conditions or
state the difference. New experiments may choose a better configuration and
should record enough context to explain it.

## Why clock context matters

Across the recorded clock-control regimes:

- cycle measurements shifted by roughly 13–19% while within-regime repeats
  stayed below about 1% variation;
- observed L2 and DRAM pointer-chase latency in SM cycles changed with the
  core-to-memory clock ratio;
- observed L1 and shared-memory latency remained effectively constant in the
  tested cases; and
- some short-kernel percent-of-peak metrics exceeded 100% under NCU replay,
  consistent with their sensitivity to replay and non-steady-state behavior.

Do not pool differently configured timing labels into one claim without
checking for a regime effect. That is a scientific comparability issue, not a
permission gate.

## Recorded variability

The five-repeat reference set classified exact count metrics as deterministic,
many timing metrics as low-noise, and several cache-hit or composition metrics
as noisier. Notable cases included:

- a bimodal L2-residency pattern for `flash_attn_bwd_bf16_sm80`;
- L1 hit/miss composition variation for some `cp.async` HMMA GEMMs; and
- an occasional layout-correlated outlier for `gemm_imma_int8_sm80`.

Inspect distributions for these workloads instead of forcing every result into
one scalar target.

## Practical compatibility notes

- An NVBit tool linked to the repository protobuf runtime needs that runtime on
  `LD_LIBRARY_PATH`; otherwise injection may fail while the application still
  runs.
- Run NCU and NVBit as separate matched executions because their replay and
  instrumentation mechanisms can interfere.
- NVBit support varies by driver/toolkit combination; rerun a small injection
  smoke after an upgrade.
- C++ protobuf generated code and runtime versions must be compatible. The Rust
  `prost` reader implements the wire format independently.

For a different GPU, select settings appropriate to that device, run a small
repeatability probe, and document the observed regime. The direct commands in
`docs/runbook_host.md` are the current starting point.
