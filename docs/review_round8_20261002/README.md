# Live SM86 writeback campaign (2026-10-02)

This is a live RTX 3090 Ti development campaign targeting the unresolved dirty
writeback/retirement tail. It is not a held-out accuracy score and does not fit
parameters from NCU.

## Protocol

- Two large store footprints were compiled and correctness-checked: 4 MiB and
  16 MiB-class streams (`large_store_4_b256_i128` and
  `large_store_4_b1024_i128`).
- The runner completed native execution, NVBit trace capture, prediction/replay,
  fast-frontier analysis, then three fresh post-prediction NCU repetitions per
  case (`clock-control none`, `cache-control all`).
- Host identity and all raw outputs remain under the ignored
  `results/captures/sm86_writeback_20261002_live_large_v2/` directory.

## Results

See [`writeback_large_summary.json`](writeback_large_summary.json).

- 2/2 cases completed; all observer uppers remained open.
- Mean lower-point APE against NCU medians: 32.936%; open lower consistency 1.0.
- Native macro dispatches and analyzer candidates: zero.
- The simulator reports terminal dirty-residency/writeback-demand gauges, but
  completed DRAM write traffic remains explicitly unavailable. The NCU write
  counters are retained as independent post-prediction observations only.
- Host clocks stayed 1350/10501 MHz; P-state changed P0→P2 after the run.

## Interpretation

The live writeback campaign strengthens the documented boundary: large dirty
streams expose unresolved terminal service, so finite observer uppers and
native selective compression remain refused. No writeback parameter was fitted
from these observations and no hardware scheduler/coherence claim is made.
