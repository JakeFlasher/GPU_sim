# SM86 direct DRAM-rate capture — 2026-09-26

The GA102 metric catalog exposes both the direct read-rate counter and the
DRAM throughput percentage. A bounded Nsight Compute run on the portable SM86
GEMM captured them independently:

```bash
ncu --force-overwrite --export /tmp/ncu_dram_crosswalk.qdWAYV/report \
  --launch-skip 1 --launch-count 1 --replay-mode application \
  --clock-control none --cache-control all \
  --metrics dram__bytes_read.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run \
  --timing_shape --warmup 1 --reps 1
```

Observed raw values:

| NCU metric | Value | Unit |
| --- | ---: | --- |
| `dram__bytes_read.sum.per_second` | 31,681,818,181.82 | byte/s |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 3.36 | % |

The Rust parser and `sim-run --ncu-csv` round trip preserve these fields as
`ncu_observation.dram_read_bytes_per_second` and
`ncu_observation.dram_throughput_pct`. They remain external labels and do not
configure replay or replace the native read-only model projection.

This capture used NCU only; no Accel-Sim/MGPUSim execution or simulator tuning
was performed.
