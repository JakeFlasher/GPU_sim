# Live SM86 development probe (2026-10-02)

This result is a bounded live RTX 3090 Ti campaign, not a held-out accuracy
claim. It exercises the current CUDA/NVBit/NCU/replay path and keeps NCU
observations separate from prediction materialization.

## Host and campaign

- GPU: NVIDIA GeForce RTX 3090 Ti, SM86, UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`.
- Driver: 615.71.09; clocks observed by `nvidia-smi`: 1350/10501 MHz; device 0.
- Suite: `workloads/research_probes/sm86_metric_domains/run.py --size small --suite domains`.
- Ten live cases: read, atom, reduction, and control variants; each has native
  execution, NCU, NVBit trace, and simulator replay records under the ignored
  `results/captures/sm86_metric_domains_20261002_live_small/` directory.
- NCU was rerun after the pre-observation prediction document was materialized;
  the post-prediction observation set is `observations_post_ncu.json`.
- The observation identity status is `caller_labeled_unverified`: the host was
  queried live, but the CSV adapter does not independently prove device
  identity from profiler metadata.

## Measured result

- 10 predictions, 10 post-prediction observations, 10 joined rows.
- All simulator intervals are open because unresolved retirement/writeback or
  coherence tails remain active; finite interval coverage is not applicable.
- Open lower-bound consistency: 1.000; mean lower-point APE: 62.675%.
- Simulator native issue counts match the live probe's NCU instruction count
  for these small cases, but this is a diagnostic domain comparison, not
  hardware timing validation.
- Native macro dispatches: 0. No native compression or speedup is claimed.

The machine-readable row-level summary is
[`live_small_summary.json`](live_small_summary.json). Raw traces, profiler
reports, binaries, and replay outputs remain in the ignored results tree so the
source commit stays small.

## Interpretation

This campaign proves current live termination and the prediction/observation
separation on a small development suite. It does not close the production
witness falsifier: every native row remains exact/open, and NCU identity is
explicitly labeled rather than treated as an independent host proof.
