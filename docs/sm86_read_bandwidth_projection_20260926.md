# SM86 read-bandwidth projection — 2026-09-26

Native replay now exposes two additional diagnostic rows when a shared-L2
DRAM observer, a card SM clock, and a modeled GPC frontier are available:

* `dram_read_bytes_per_second_estimate`
* `dram_read_bandwidth_pct_estimate`

They are explicitly read-only model-relative estimates.  The numerator is the
replay's modeled DRAM-read demand (including compulsory instruction sectors),
the denominator is the modeled GPC observation frontier, and the percentage
uses the architecture card's validated peak DRAM rate.  Neither row consumes
NCU values, and neither substitutes for `dram__bytes.sum.per_second` or NCU
achieved-bandwidth metrics.  Total DRAM throughput remains unavailable because
dirty writeback, coherence, and the profiler's kernel/flush interval are not
present in the fixed trace contract.

## Bounded executable check

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
sim/target/release/sim-run \
  --trace sim/run/tests/fixtures/addr_carry_canary_bundle/trace \
  --control sim/run/tests/fixtures/addr_carry_canary_bundle/control/canary_sm86.control_bits.json \
  --arch-card sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86.json \
  --pchase-card sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86_static_card.json \
  --mode l1-l2 --cache-scope launch --ncu-shaped-metrics
```

The current release binary (`5d2360a92a54fd8f0c8433072fbfc5227f667fb5e9db76c3bedbd4c44a95ca28`)
reported the following for the retained canary:

| Row | Value | Status |
| --- | ---: | --- |
| `dram_read_bytes` | 1,152 byte | modeled |
| `dram_read_bytes_per_second_estimate` | 607,262,787.97 byte/s | approximate |
| `dram_read_bandwidth_pct_estimate` | 0.0602443% | approximate |
| `dram_bytes_per_second` | unavailable | typed gap; total writes are not modeled |

The same binary also replayed the retained 512-block GEMM trace at
`results/captures/sm86_measurement_boundary_20260924_v1/.../gemm_hmma_bf16_sm80`:

| Row | Value | Status |
| --- | ---: | --- |
| `dram_read_bytes` | 264,384 byte | modeled |
| `dram_read_bytes_per_second_estimate` | 27,563,394,856.75 byte/s | approximate |
| `dram_read_bandwidth_pct_estimate` | 2.73446% | approximate |
| `dram_bytes_per_second` | unavailable | typed gap; total writes are not modeled |

This larger replay completed without changing the retained trace or using its
NCU labels as replay inputs.

The projection is a diagnostic addition, not an accuracy claim.  It should be
compared to NCU only as a separately labeled read-only demand hypothesis.
