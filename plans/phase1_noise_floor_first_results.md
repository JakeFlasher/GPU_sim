# Phase-1C First Noise-Floor Results (2026-07-06, RTX 3090 Ti, host-locked 1350 MHz)

First 60-capture repeatability matrix (2 kernels × {clock-control none|base|boost} ×
{cache-control all|none} × 5 repeats; zero capture failures). Evidence:
`data/annex/repeatability_matrix_20260706/matrix_summary.json` (machine-local annex).
Runner: `tools/local_labels/repeatability_matrix_runner.py`.

## Headline findings (all three are load-bearing for the paper)

1. **T1 counts are exactly deterministic.** `smsp__inst_executed.sum` and
   `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` are bit-identical across every
   repeat and every protocol configuration (CV = 0.0000). Determinism class:
   *deterministic* — the simulator's count emissions can be gated at ~0% tolerance.

2. **T3 timing noise floor under the pinned protocol is ~0.7–1.0% CV.**
   `gpc__cycles_elapsed.max` at host-lock+cold-cache: CV 0.0067 (rmsnorm) / 0.0104
   (softmax); `gpu__time_duration.sum` similar. A ≤10% cycles-MAPE gate (D5) sits
   ~10× above this floor — the gate is measurable.

3. **Cycles are protocol-sensitive up to ~19% ACROSS clock-control modes** while
   staying <1% repeatable WITHIN each mode: rmsnorm 19,259 (host-lock 1350 + none) →
   21,270 (NCU base) → 22,934 (NCU boost); softmax 15,602 → 16,745 → 17,644 (+13%).
   "DVFS-stable" cycle counts depend on the core:mem clock *ratio*, so:
   - every accuracy claim must name its clock protocol;
   - the project's pinned reference = **host `-lgc 1350` + `--clock-control none`**;
   - `sim/calib` latency/bandwidth constants must be measured under the SAME pin.
   This cross-protocol spread (with per-protocol CVs) is the counter-determinism
   study's first quantified exhibit — the Weaver/McKee-analog section.

4. **Cold cache is the more repeatable label.** rmsnorm cycles at lock+`cache all`
   CV 0.0067 vs lock+`cache none` CV 0.0148 (mean shift only ~1%). Supports the
   cold-cache-per-kernel default (`--cache-control all` labels; sim/emit cold-cache
   parity).

## Next (Phase-1C continuation)

Scale the matrix to the portable_lcd 18 + simbench calibration tier with the full
61-metric registry; add replay-mode arms; publish per-metric determinism classes
(deterministic / clock-sensitive / replay-sensitive) + per-family noise floors as the
accuracy-gate denominators. Include the S7 Blackwell `lts` 115–217% anomaly when the
5060 opens (post-M2).
