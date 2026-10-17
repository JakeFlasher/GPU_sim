# SM86 integrated adversarial review — H86 — 2026-10-16

## Checks completed

* `cargo check --workspace` and `cargo build --workspace --release` pass.
* Python compilation for replay, reporter, and JSONL tools passes;
  `git diff --check` passes.
* Latest full/fast replay and metric-comparison `.jsonl.zst` artifacts pass
  `zstd -t`.
* Latest full artifact retains 286 `ok`, 62 controls, 8 replay errors, 12,392
  launches, 107 identity joins, 179 ordered fallback joins, and exactly twelve
  workers.
* All 12,392 full launches satisfy the promoted H80/H81 write conservation
  invariant: candidate write lookup hits ≤ candidate sectors and
  `hits + misses == sectors`.

## Adversarial findings

1. No NCU value is read by replay or used as a prediction selector. NCU values
   occur only in post-replay report joins and independent notes probes.
2. H80/H81 packet composition changes only operation-split write endpoint
   predictions; aggregate L2 state, DRAM read demand, writeback, frontiers,
   and IPC rows are unchanged bit-for-bit across H81/H83 reports.
3. H82 partition concentration and H83 readiness concentration are observer
   covariates, not timing laws. H64/H71 negative evidence still blocks queue,
   EXIT, partition, or IPC timing promotion.
4. H85 real NCU matrix falsifies a GCC-request-to-DRAM linear mapping; the P4
   multi-client arm remains correctly unpromoted.

## Remaining blockers/gates

The remaining >10% surfaces are genuine endpoint/causal gaps: L1 lookup form
scope, GCC/instruction/maintenance DRAM completion ownership, and SM/SMSP/GPC
retirement timing. The next slice must add independent evidence or a bounded
observer for one of these mechanisms; no closure claim is made from the checks
above.
