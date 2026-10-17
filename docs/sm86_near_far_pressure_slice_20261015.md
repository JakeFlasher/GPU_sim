# SM86 near/far partition-pressure uncertainty — 2026-10-15

## Scope and independent evidence

H9 adds a typed near/far pressure envelope over H8's candidate partition maps.
For modulo-12 and XOR-fold modulo-12 hypotheses, the six-versus-six half
controller groups form low/high candidate pressure edges and an explicit map
uncertainty width. The envelope is observer-only: it does not select a hash,
add latency, or alter the aggregate DRAM server.

NVIDIA's GA102 whitepaper supports twelve memory controllers and six 512-KiB
L2 slices, but not their swizzle. Luo et al. (2025) provides a partitioned-L2
near/far mechanism prior on A100/Hopper only; its latencies and counts are not
imported. Independent RTX 3090 Ti microbench runs (GPU identity independently
checked) measured forced-DRAM warm bandwidth roughly 810–850 GB/s across
unlocked-clock repeats, versus the 1008 GB/s vendor peak. This supports an
explicit service-rate interval for a future scenario, not a timing default.

## Representative full replay

Five real roots completed full replay 5/5. Candidate near/far intervals (in
source-derived sectors) were:

| Root | Low edge | High edge | Uncertainty width |
|---|---:|---:|---:|
| cp.async tile copy | 56 | 72 | 16 |
| FP16 HMMA | 96 | 96 | 0 |
| FP32 dependency chain | 56 | 72 | 16 |
| local/register spill | 0 | 16 | 16 |
| RMSNorm BF16 | 520 | 536 | 16 |

## Complete-corpus full-first evidence

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| `ok` roots | 286 |
| Mixed-code controls | 62 |
| Replay/input failures | 8 |
| Successful launches | 12,392 |
| Driver wall time | 35.059 s |
| Workers | 12 |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_h9_near_far_20261015_v1/full.jsonl.zst`.

## Residuals and next decision

Exact-identity DRAM/L1/L2/IPC residuals are unchanged from H8 (DRAM 40.91%,
L2 hit/miss/source 35.04%/27.00%/28.24%, active/elapsed IPC 18.26%/12.53%
mean APE). Report:
`../sm86_replay_results/metric_report_h9_near_far_20261015_v1/`.

The unlocked-clock microbench interval is retained as a future carded
service-rate scenario [810,850] GB/s, not silently substituted for the vendor
peak. Timing-active partition/row service remains deferred until a locked
3090 Ti probe identifies mapping or service behavior.
