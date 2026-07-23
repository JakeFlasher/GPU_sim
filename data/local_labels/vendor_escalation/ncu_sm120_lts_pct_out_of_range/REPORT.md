# NVIDIA Bug Report: Nsight Compute `lts__t_sector_hit_rate.pct` Returns Values > 100% on sm_120 Blackwell Consumer

## Summary

Nsight Compute 2026.1.1.0 on the NVIDIA GeForce RTX 5060 Laptop GPU
(sm_120, Blackwell consumer, GB206) intermittently returns
`lts__t_sector_hit_rate.pct` values well above 100% for specific
kernels under repeated launches. By the Ratio metric's definition
("proportion of L2 sector lookups that hit"), the maximum admissible
value is 100% — any positive deviation indicates either ratio
aggregation, replay normalization, or collection-path overcounting
rather than ordinary counter-sampling quantization rounding.

Observed values: **115.6%**, **202.01%**, **217.66%** across at least
three independent kernel launches.

## Reproduction Surface

### Hardware

- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU
- **Architecture**: sm_120 (Blackwell consumer)
- **Die**: GB206
- **Driver version**: 595.71.05
- **CUDA Toolkit / Runtime**: 13.2 (per `nvcc --version`); driver
  reports CUDA Runtime 12.0 via `nvidia-smi`.

### Nsight Compute

- **Binary path**: `/usr/bin/ncu`
- **Version**: `Version 2026.1.1.0 (build 37634170) (public-release)`
- **Per `ncu --version`**: NVIDIA (R) Nsight Compute Command Line Profiler

### Kernel Under Test

- **Slug**: `silu_fwd_bf16_sm80`
- **Shape**: `correctness_shape` (small batch / single launch
  exercising the correctness path of the SiLU forward kernel)
- **Source**: `workloads/portable_lcd/silu_fwd_bf16_sm80/`
- **Multi-arch fatbin**: built with `nvcc -arch=sm_86 -arch=sm_90
  -arch=sm_120 -arch=compute_80` via `NVCC_CCBIN=g++-14 make`
- **Binary SHA-256**:
  `b6e9d7e11edb04c02ea7230b60ee89b9424975b9e95888847020764176349cb4`

### NCU Invocation

```
ncu --target-processes all \
    --replay-mode kernel \
    --metrics "lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct" \
    --section MemoryWorkloadAnalysis \
    --launch-skip <N> \
    --launch-count 1 \
    --clock-control none \
    --export <report>.ncu-rep \
    <binary> --correctness_shape
```

Where `<N>` is the warmup count (so the profiled launch is index N+1
in the sequence). Warmup counts observed to trigger the defect:
0, 1, 5 (i.e. the defect is not unique to a single warmup setting).

### CSV Extraction

The report is imported via `ncu --import <report> --page details
--csv --print-metric-name name --print-units base --print-fp` and
parsed by `tools/local_labels/parse_ncu_metrics.py`. The parser
selects the metric value from the documented `Metric Value` column
by exact `Metric Name` match. The `--launch-skip` / `--launch-count
1` arguments are present per CSV inspection (one launch identity
per report, no cross-launch aggregation regression).

## Observed Defect

For 5 of 30 launches across multiple (warmup, cache_control, repeat)
matrix configurations, the `.pct` value exceeded 100% by a large
margin:

| Sweep timestamp | warmup | cache | repeat | `.pct` value |
|---|---|---|---|---|
| `sweep_20260524T230930Z_dcaaffc789d6` | 0 | all | 2 | 202.01% |
| `silu_correctness_retry4_diag_20260524T232715Z` | 1 | all | 4 | 217.66% |
| `silu_correctness_retry4_diag_20260524T232715Z` | 1 | all | 5 | 115.6% |

The defect is intermittent — even within the same (warmup, cache,
repeat) config, the defect appears in only a subset of launches.
The same NCU invocation on the same binary on the same GPU produces
a value in [0, 100] on most launches and a value far above 100 on
occasional launches.

The defect was NOT observed at the (warmup=5, cache=none) config in
the round-5 retry sweep (10/10 launches produced values in
[0.129, 0.806] with mean 0.533 and stddev 0.222), but the kernel
itself has irreducibly high variance independent of the .pct
defect at this config.

## Why It Matters (Downstream Effect)

The repository's `tools/local_labels/parse_ncu_metrics.py` accepts
the `[100, 105]` saturation-rounding window (admitting 1.0 with a
typed `ncu_sector_hit_rate_clamped_to_100` audit diagnostic) for
quantization artifacts, but rejects values above 105 as out-of-range
with the existing `ncu_csv_metric_out_of_range` typed reason. This
caused 5 of 30 launches to fail the repeatability gate, blocking
the bounded sm_120 label admission loop's AC-6 closure (the
`silu_fwd_bf16_sm80/correctness_shape` cell could not produce 5
admitted sidecars at unchanged `stddev_threshold = 0.03`).

The repository's resolution (committed at `a6ccfdce`) is to
opportunistically also collect the matching raw counter pair
`lts__t_sectors_lookup_hit.sum` and `lts__t_sectors_lookup_miss.sum`
and derive a mathematically-bounded hit rate
`hit / (hit + miss) ∈ [0, 1]` as a fallback when the `.pct` value
exceeds 105%. The derived value is reported with a new typed
diagnostic `ncu_lts_pct_overcount_raw_counter_ratio_used` capturing
the original `.pct` value. This is a workaround for downstream
consumers; the upstream NCU defect remains and should be addressed
by NVIDIA.

## Hypotheses (Untested)

1. **Ratio aggregation across replays**: `--replay-mode kernel`
   replays the same kernel multiple times to collect different
   counter sections. The Ratio metric may aggregate hit/miss
   counts incorrectly across replays in certain section
   combinations on Blackwell consumer parts.
2. **Replay normalization on async-copy paths**: Blackwell adds
   new async-copy paths (TMA-like, `LDGSTS` variants) that may
   interact unexpectedly with the `MemoryWorkloadAnalysis`
   section's accounting on sm_120.
3. **Collection-path overcounting in the SiLU kernel's launch
   shape**: the kernel's correctness shape produces a small
   working set; the `.pct` ratio computation may double-count
   sector lookups when the working set fits inside a single
   tile.

The defect class is consistent across two warmup counts (0 and 1)
and one cache control (all), so it is not a function of
`--cache-control` alone. The defect does NOT appear at
(warmup=5, cache=none) in the round-5 retry sweep, suggesting the
warmup state and cache flush together avoid the overcounting path.

## Suggested Resolution

1. **Short-term**: an NCU CLI guide note documenting the `.pct`
   metric's behavior on sm_120 Blackwell consumer with
   `--replay-mode kernel`; ideally a typed diagnostic in
   `ncu --import` output flagging the out-of-range value.
2. **Long-term**: an NCU driver / Nsight Compute fix that ensures
   Ratio metrics never return values outside their definition's
   theoretical range; if the underlying counter pair shows the
   defect (rare), the Ratio metric should fail fast with a typed
   error rather than reporting a mathematically-invalid value.

## Repository Artifacts (For Reproduction)

- Patched parser: `tools/local_labels/parse_ncu_metrics.py` at
  commit `a6ccfdce` (raw-counter ratio fallback added).
- Pre-patch sweep with .pct overcounts:
  `data/local_labels/sweeps/sweep_20260524T230930Z_dcaaffc789d6/`
  and `data/local_labels/sweeps/silu_correctness_retry4_diag_20260524T232715Z/`.
- Post-patch sweep (variance-only failure mode):
  `data/local_labels/sweeps/silu_correctness_retry5_diag_20260525T001224Z/`.
- Kernel source:
  `workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80.cu`
  + `workloads/portable_lcd/silu_fwd_bf16_sm80/main.cu`.
- Round-4 RLCR diagnostic with full sweep history:
  `.humanize/rlcr/2026-05-25_05-35-54/round-4-summary.md`
  (host-local, not in git).

## Loop State

This bug report does not block the loop. The repository workaround
(raw-counter ratio fallback) is sufficient to extract bounded
hit-rate values for downstream consumers. The bounded sm_120 label
admission loop's AC-6 stall is now attributable to kernel-side L2
variance, not the NCU defect; the variance is being addressed via
a separate plan amendment proposing a variance-stable replacement
cell for the activation cohort.

This report is filed as evidence hygiene, not on the critical path.
