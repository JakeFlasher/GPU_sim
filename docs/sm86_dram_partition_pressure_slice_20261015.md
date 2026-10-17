# SM86 DRAM partition-pressure envelope — 2026-10-15

## Scope and evidence boundary

H8 adds observer-only candidate pressure for GA102's documented 12 memory
controllers and six L2 slices. It applies line-ID modulo-12, an XOR-fold
modulo-12, and line-ID modulo-6 hypotheses to L2-miss candidate line IDs.
These are explicit effective hash candidates; physical controller/slice
swizzling is undisclosed and no candidate changes the G/D/1 byte server,
response timing, cache state, or IPC.

The primary Ampere GA102 architecture whitepaper documents twelve 32-bit memory
controllers and six 512-KiB L2 slices (6 MiB total), but not the address hash.
Public GPGPU-Sim address decoders were inspected as generic prior art, not
imported as GA102 truth. Independent real-trace probe notes found material
candidate imbalance on irregular traces but weak/unstable relation to NCU
`fbpa__dram_sectors.{avg,max,min}`; no mapping is selected.

## Representative full replay

Five real roots completed full replay 5/5. Candidate arrays expose distinct
pressure patterns: contiguous FP16/FP32 streams are nearly uniform, while
local/register and async shapes concentrate in selected candidate partitions.
No timing frontier changed.

## Complete-corpus full-first evidence

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| `ok` roots | 286 |
| Mixed-code controls | 62 |
| Replay/input failures | 8 |
| Successful launches | 12,392 |
| Driver wall time | 35.409 s |
| Workers | 12 |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_h8_dram_partition_20261015_v1/full.jsonl.zst`.

Aggregate candidate sector populations were 4,824,390 modulo-12 sectors,
4,824,390 XOR-fold sectors, and 4,824,390 modulo-6 sectors. These are
hypothesis-domain counts, not physical traffic or service completions.

## Residual deltas and verdict

Exact-identity residuals are unchanged from H6: DRAM read 40.91%,
`sm__cycles_active.avg` 18.26%, and elapsed IPC 12.53% mean APE; L1/L2 rows
remain 35.04%, 27.00%, and 28.24% for hit, miss, and source-total sectors.
Report: `../sm86_replay_results/metric_report_h8_dram_partition_20261015_v1/`.

H8 is accepted as the strongest current DRAM topology observability envelope,
not a timing-active partition model. The next slice must address a separate
service queue/row-buffer or IPC mechanism only if independent evidence can
identify it; otherwise keep expanding mechanism-local residual attribution.
