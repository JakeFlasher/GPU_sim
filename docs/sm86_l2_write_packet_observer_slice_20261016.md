# SM86 L2 write-packet coalescing — H80 — 2026-10-16

## Mechanism

H80 adds a bounded observer at the L1TEX→XBAR write boundary. For global
STORE requests on one SM, each `(128-B line, 32-B sector)` is admitted once
within an 8-cycle issue window. The candidate uses only traced event identity
and issue cycles; it does not alter L2 tags, dirty masks, source-sector
baseline counters, DRAM demand, write acknowledgements, or any timing frontier.
The original one-sector-per-request value remains available as
`l2_op_write_sectors_request_baseline`.

Only the NCU-shaped operation-sector endpoint
`lts__t_sectors_srcunit_tex_op_write.sum` selects the packet candidate. The
operation lookup-hit/miss rows remain on their established baseline because
the independent H75 probe showed lookup hits can exceed packet sectors and are
not derivable from packet count alone.

## Full-first evidence

Release build and full-first replay completed all 356 roots with twelve
workers:

* replay: `../sm86_replay_results/full_corpus_h80_write_packet_observer_20261016_v2/`
* report: `../sm86_replay_results/metric_report_h80_write_packet_observer_20261016_v2/`
* coverage: 286 `ok`, 62 controls, 8 replay errors; 12,392 full launches.

The selected operation-sector endpoint improves on all 107 identity joins:

| endpoint | mean APE | median | P90 | exact nonzero/rows |
|---|---:|---:|---:|---:|
| baseline request sectors | 39.86% | 0.00% | 100.00% | 77/107 |
| packet candidate (selected) | **9.37%** | 0.00% | 0.00% | **100/107** |

The candidate is exact on 100 of 106 nonzero targets; the remaining zero-target
row is retained in the denominator and no false nonzero target is introduced.
Aggregate L2 sector/lookup residuals and all cycle/IPC metrics are unchanged.

## Boundary

This is an effective operation-endpoint model, not a physical XBAR topology
claim. The 8-cycle window is a bounded card candidate and is not used for DRAM
service or writeback timing. Future promotion of write lookup-hit/miss timing
requires a same-binary request/tag witness; H80 does not infer one.
