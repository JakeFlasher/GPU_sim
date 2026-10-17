# SM86 source-qualified pending-pressure ledger — 2026-10-15

## Research decision

Huerta *Modern GPU Cores* §5.4/§6 and the released modern-gpu-simulator
`ldst_unit_sm.{h,cc}` model finite memory-subcore queues and a pending-request
table (PRT) before coalescer selection. Upstream GPGPU-Sim likewise separates
MSHR waiter merging, fill service, and finite L2↔DRAM queues. These are useful
mechanism priors, not GA102 RTL truth: the local RTX 3090 Ti queue-frontier
probe shows pressure growth but no stable queue-latency differential after
occupancy and dirty/coherence tails are separated.

An attempted one-cycle-per-merged-sector response-drain arm (capped at four
sectors) was therefore treated as an ablation, not promoted. It changed 1,670
launch frontiers, improved GPC mean APE only 0.004 percentage points, and
worsened SM-active APE. The timing arm was removed before the retained run.

## Implemented observer

H37 retains H36's pending-merge/new-fill ownership and adds source-qualified
pending-sector pressure samples. Immediately before every shared-L2 event
request, the ledger records the number of pending fill sectors, indexed by
`L2Source::{L1Fill,Bypass,Untagged,Store}`. It emits per-source sample counts,
pressure sums, and bounded means. The ledger does not change cache state,
lookup classification, DRAM demand, response time, IPC, or frontiers; it does
not import a PRT capacity or hidden slice/hash mapping.

## Real-trace evidence

Five L2-thrash/high-stride/working-set/SGEMM roots passed representative
full-first replay (13 launches):
`../sm86_replay_results/h37_prt_pressure_observer_20261015_v1/full.jsonl.zst`.

Complete full-first replay then complementary fast replay passed 356 roots
(286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 launches
in 40.638 s with exactly twelve workers:

- `../sm86_replay_results/full_corpus_h37_prt_pressure_observer_20261015_v1/full.jsonl.zst`
- `../sm86_replay_results/full_corpus_h37_prt_pressure_observer_20261015_v1/fast.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h37_prt_pressure_observer_20261015_v1/`.

Across full launches, source pressure totals are:

| source | samples | pending-sector sum | launches with samples |
|---|---:|---:|---:|
| L1 fill | 2,727,211 | 3,253,767,441 | 12,021 |
| bypass | 31,433 | 1,933,553 | 37 |
| untagged | 32 | 4,666 | 2 |
| store | 3,279,482 | 198,743,493 | 12,383 |

The five-root pressure means are 2.5 sectors for the pointer-chase launch,
37.2 for vendor SGEMM, 28.91 for the working-set sweep, and 7,629.98 for the
streaming/DSMEM pressure roots.

## Exact non-regression

A keyed comparison against H36 found zero differences in all old L1/L2/DRAM
counters, lookup predictions, IPC/frontiers, timing fields, and trace fields
after removing only the new pressure rows and the nondeterministic sidecar
join wall-time field. H37 therefore adds a mechanism-local pressure surface,
not a timing or accuracy claim. NCU remains an independent post-replay target;
future timing promotion requires a locked pressure differential that isolates
queue service from occupancy and unmodeled retirement tails.
