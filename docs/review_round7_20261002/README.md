# Live SM86 large development campaign (2026-10-02)

This is a real CUDA/NVBit/NCU/replay campaign on the current RTX 3090 Ti host.
It is development evidence, not a held-out accuracy score or a native-speedup
claim.

## Protocol

- The runner was changed to complete the prediction pass for every case before
  starting any fresh post-prediction NCU repetition.
- Ten 168-block cases were executed: read, atom, reduction, and control
  variants. Each received three fresh NCU application-replay repetitions with
  `--clock-control none --cache-control all`.
- Each case also ran native correctness, NVBit tracing, simulator replay with
  the latent envelope, and `fast-frontier-only` analyzer timing/coverage.
- The current binary, CUDA/NVCC, NCU, host identity, trace/control/card paths,
  and process costs are retained under the ignored results directory:
  `results/captures/sm86_metric_domains_20261002_live_large_v2/`.

## Results

The machine-readable aggregate is
[`live_large_summary.json`](live_large_summary.json).

- 10/10 cases completed native execution, trace capture, replay, analyzer, and
  three NCU repetitions.
- All 10 simulator observer intervals remained open; finite coverage is not
  applicable.
- Mean lower-point APE against the post-prediction NCU medians: 43.793%.
- Open lower-bound consistency: 1.000.
- Native macro dispatches: 0; analyzer candidate records: 0; all traced records
  were conservative uncovered/refused candidates.
- Instruction counts matched across simulator and NCU domain checks for every
  case. This validates a bounded domain join, not cycle accuracy.
- Host clocks stayed at 1350/10501 MHz, while P-state changed from P0 before
  the campaign to P2 after it; that state change is retained as a limitation.

## Interpretation

This campaign supplies live termination, trace/replay, cost, and post-prediction
observation evidence. It confirms the current boundary: real memory/control
traces remain exact/open, the selective analyzer accepts no native candidates,
and no production macro dispatch or speedup should be claimed.
