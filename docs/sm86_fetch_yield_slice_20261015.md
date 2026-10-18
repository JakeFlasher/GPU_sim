# SM86 fetch-last-issued / Yield slice — 2026-10-15

## Scope

H4 separates fetch/decode continuation from issue hold in full replay. The
fetch picker now uses a persistent per-SMSP `fetch_last_issued` warp; the SASS
Yield bit still clears issue hold for the next issue cycle, but does not force
fetch to switch warps while the prior warp's instruction buffer has capacity.
The state is cleared only on warp retirement. No L1/L2/DRAM or issue timing
resource is changed directly.

## Real-trace evidence

Five real roots completed full replay 5/5, including divergence, FMA,
register-spill, local-volatile, and vendor-SGEMM control shapes. Front-end
observations included persistent continuations and Yield/hold separations; for
example, the divergence root recorded 271 continuations and 133 separations,
while FMA recorded 3,008 and 2,112 respectively. No deadlock, input
classification, or cache/DRAM counter changed.

The complete full-first corpus completed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 successful launches in 34.826 s
with 12 workers:

`../sm86_replay_results/full_corpus_h4_fetch_yield_20261015_v1/full.jsonl.zst`

## Exact-identity residual deltas

H4 changes fetch observability but does not alter the current timing frontier on
this corpus. H3 and H4 reports have identical exact-identity results:

| Metric | H3 mean APE | H4 mean APE | Delta |
|---|---:|---:|---:|
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| `sm__cycles_active.avg` | 18.26% | 18.26% | 0.00 pp |
| `sm__inst_executed.avg.per_cycle_elapsed` | 12.53% | 12.53% | 0.00 pp |

Report artifact: `../sm86_replay_results/metric_report_h4_fetch_yield_20261015_v1/`.

H4 is retained as a front-end coverage correction, not an IPC accuracy gain.
The next arm must target a timing-active residual (static `LDG.CONSTANT`/GCC
source classification or a separately evidenced denominator/front-end envelope)
without selecting parameters from NCU values.
