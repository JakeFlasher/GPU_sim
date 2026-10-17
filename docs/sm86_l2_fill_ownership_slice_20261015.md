# SM86 sectored L2 fill-ownership observer — 2026-10-15

## Mechanism

H36 exposes the already-active SM86 event-mode coalescing rule as a checked
observer. A read sector either owns a new lower-level fill capability or joins
the matching allocation/sector whose fill is still pending. A pending merge
remains a modeled cache-state lookup miss, but reuses the original ready
frontier; the observer is therefore separate from cache-state hit/miss pricing
and does not fit NCU values.
Counters are cumulative, snapshot-safe, and included in checked launch deltas.
Full replay emits `l2_pending_merge_sectors`, `l2_new_fill_sectors`, and the
bounded `l2_pending_merge_fraction_pct` projection.

## Real-trace evidence

Five L2-thrash/high-stride/working-set/SGEMM roots passed representative
full-first replay (13 launches):
`../sm86_replay_results/h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`.

The required complete full-first replay then passed 356 roots (286 `ok`, 62
mixed-code controls, 8 replay/input failures), 12,392 launches in 40.379 s
with exactly twelve workers. The compressed artifacts are:

- `../sm86_replay_results/full_corpus_h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`
- `../sm86_replay_results/full_corpus_h36_fill_ownership_observer_20261015_v1/fast.jsonl.zst`

Post-replay comparison:
`../sm86_replay_results/metric_report_h36_fill_ownership_observer_20261015_v1/`.

## Corpus distribution and non-regression

Across all 12,392 full launches, the observer records 1,727,757 pending-merge
sectors and 4,823,652 new-fill sectors. Pending merges occur in 1,822 launches;
new fills occur in 12,044. Among launches with any fill ownership, the merge
fraction is 0% minimum, 0% median, 2.273% p90, and 97.196% maximum. The largest
pending-merge totals occur in real attention/vision/paged-MMA traces, while
streaming and working-set roots are new-fill dominated.

An exact launch-key comparison against H35 found zero differences in every old
counter, metric prediction, timing frontier, DRAM field, and trace field after
excluding the expected `sidecar_join_wall_ns` measurement and the three new
observer rows. H36 therefore changes no service time, cache state, DRAM demand,
IPC, or frontier; it only makes sectored fill ownership measurable for the
next independently justified conditional model.

NCU remains a post-replay comparison target. No proprietary GA102 pending-table
size, fill buffer count, or queue latency is claimed.
