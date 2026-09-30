# SM86 release replay phase cohort — 2026-09-27

This is a reproducible event-driven turnaround cohort, not a claim against
GPU silicon or Accel-Sim.  It separates trace loading, schedule construction,
replay, serialization, and total process time using the existing
`SIM_RUN_PERFORMANCE_COUNTERS=1` instrumentation.  No NCU value is used as a
replay input.

## Command

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run \
  --features performance-counters,reference
SIM_RUN_PERFORMANCE_COUNTERS=1 sim/target/release/sim-run \
  --trace <trace> --control <control_bits> \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2 --cache-scope launch --ncu-shaped-metrics
```

The exact machine-readable observations are in
[`sm86_speed_cohort_20260927.csv`](sm86_speed_cohort_20260927.csv).  The
release binary was `ddc911987c9b1749f7ed119ef2322f6f1aeebc1977ced1d09fca7a9852715692`.

| Workload | Blocks | Instructions | Trace load | Replay | Total process | GPC frontier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMM BF16 | 512 | 197,120 | 122.895 ms | 773.864 ms | 898.707 ms | 12,949 |
| RMSNorm BF16 | 64 | 264,704 | 74.769 ms | 373.687 ms | 449.624 ms | 18,944 |
| SiLU BF16 | 1,024 | 417,792 | 89.971 ms | 449.545 ms | 540.452 ms | 5,642 |

These are single-run engineering observations.  They expose workload-dependent
trace/replay cost and retain frontiers for later event-vs-dense comparisons;
they do not establish a universal simulator speedup or a timing-accuracy
claim.
