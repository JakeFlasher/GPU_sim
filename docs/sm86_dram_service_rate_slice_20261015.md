# SM86 effective DRAM service-rate scenario — 2026-10-15

## Scope and evidence

H10 adds an explicit `--dram-service-rate-bytes-per-sec` scenario to the
carded full replay. Independent repeated RTX 3090 Ti memtest runs (GPU
identity independently checked; unlocked protocol) measured forced-DRAM warm
throughput roughly 810–850 GB/s, with a repeated mean near 825 GB/s. The
vendor card peak remains 1008 GB/s and is retained as the reference, not
silently overwritten. Physical partition mapping, row state, and command
scheduling remain unknown.

The SM86 corpus runner now defaults full carded replay to the 825 GB/s
hypothesis, records `dram_service_rate_defaulted=true`, and permits explicit
scenario overrides. The unlocked-protocol bytes/s value is mapped onto the
simulator's locked-card SM clock for cycle pricing; it is not a hardware clock
crosswalk or physical service claim. The [810,850] interval remains the
uncertainty context.

## Representative full replay

Four real roots completed full replay with the explicit 825 GB/s scenario:
cp.async, scalar copy, DRAM streaming dot product, and RMSNorm. Non-DRAM roots
were unchanged; the streaming root's GPC frontier moved closer to its
independent NCU elapsed target (14.42%→11.55% GPC APE).

Artifact: `../sm86_replay_results/h10_dram_rate_825_20261015_v1/full.jsonl.zst`.

## Complete-corpus full-first evidence

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| `ok` roots | 286 |
| Mixed-code controls | 62 |
| Replay/input failures | 8 |
| Successful launches | 12,392 |
| Driver wall time | 35.094 s |
| Workers | 12 |
| Service-rate scenario | 825,000,000,000 byte/s (defaulted hypothesis) |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_h10_dram_rate_825_20261015_v1/full.jsonl.zst`.

## Exact-identity residual deltas

Compared with the H8 card-peak full corpus, the independent 825 GB/s scenario
slightly improves timing-domain residuals while leaving traffic/cache metrics
unchanged:

| Metric | Card peak mean APE | 825 GB/s mean APE | Delta |
|---|---:|---:|---:|
| GPC elapsed cycles | 13.14% | 13.05% | −0.08 pp |
| SM active cycles | 18.26% | 18.19% | −0.07 pp |
| SM active max cycles | 18.11% | 18.00% | −0.11 pp |
| SMSP active cycles | 17.76% | 17.69% | −0.07 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |

Report: `../sm86_replay_results/metric_report_h10_dram_rate_825_20261015_v2/`.

The conservative scenario is promoted as the SM86 full-replay default because
its rate is independently measured and the complete corpus moves timing
residuals in the expected direction. It remains an uncertainty-aware effective
model; no physical bandwidth or partition mapping claim is made.
