# sm_86 CUDA-VMM alias discriminator

The sequential RED result establishes that GA102 atomic LTS identity follows translated physical allocation state, not the virtual pointer alone.

## Live corpus

- 133 direct correctness executions.
- 121 cache-flushed read reports and 56 sequential-RED reports.
- 104,672 official profiled kernel actions; every RED action conserves 37,632 sectors.
- 3 read-sector total outliers are retained; medians, not filtered means, summarize the read diagnostic.

## Read diagnostic

A single pointer reaches only 8.05% of read sectors in the busiest LTS instance. Read requests are replicated or merged across instances, so `lts__t_sectors_op_read.max` is not a slice-identity oracle. The matched physical-layout A/B remains diagnostic only.

## Sequential RED discriminator

Each report first classifies all 48 non-aliased representatives and must decode exactly 0–47. Seven later kernels use one virtual alias at a time, with a synchronization boundary before switching proxies; this respects CUDA's alias coherence rule.

| physical offset | class | alias observations |
|---:|---:|---:|
| 0 | 5 | 49 |
| 128 | 5 | 49 |
| 512 | 5 | 49 |
| 4,096 | 13 | 49 |
| 65,536 | 42 | 49 |
| 524,288 | 3 | 49 |
| 1,048,576 | 39 | 49 |
| 2,096,128 | 40 | 49 |

All 392 of 392 alias observations agree. Virtual aliases are separated by 2 MiB over a 14 MiB VA range, yet each physical word keeps one class across all aliases and all seven independent reports.

## Allocation-relative transfer

A dense 16,384-offset VMM map covers the entire 2 MiB physical allocation. Its 48-way collision partition is exactly identical to the independent `cuMemAlloc` map up to a one-to-one class permutation: 100% collision recall and 100% specificity.

Every 2 MiB granule in the exhaustive 8 MiB extension also transfers exactly, but the three granules require different permutations. Treating the whole range as one phase yields only 55.51% collision recall. Conversely, the size-matched 32 MiB high holdout transfers exactly, showing that physical allocation layout controls the phase sequence.

## Cross-allocation phase

Seven reports classify eight distinct physical handles. All maps are repeat-stable and all 8 are distinct. The canonical representatives cover [48, 32, 48, 32, 32, 48, 32, 48] classes across the eight handles, and allocations 1–7 have zero same-offset class matches with allocation 0.

## Two-family phase model

All observed granules reduce to 2 canonical partition families. Standard granules alternate [0, 1, 0, 1]; the eight distinct handles use [0, 1, 0, 1, 1, 0, 1, 0] ([4, 4] counts). Phase 0/1 score 100.00%/75.00% collision recall on the retained pair corpus, while both reproduce all 220 local triplet occupancy classes. The latent phase is not present in the trace, so this remains an evaluator, not simulator timing state.

## Decision

Absolute virtual pointers are insufficient to compute exact GA102 atomic LTS identity. Allocation id and relative offset are valid ingredients, but not a universal key: each physical allocation/granule carries hidden phase. No hash or timing queue is promoted. Silently hashing either absolute trace VAs or relative offsets alone would encode the wrong domain.
