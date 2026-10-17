# SM86 L2 write-lookup split — negative decision — 2026-10-16

An independent same-binary L1TEX store probe reports cold line stores as zero
lookup-hit sectors and all requested sectors as lookup misses. However, replacing
the current corpus observer's conditional store-hit arm with
`lookup.present_sectors` was falsified by the complete real corpus.

H67 full-first replay covered 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches with 12 workers and compressed output:
`../sm86_replay_results/full_corpus_h67_l2_write_lookup_split_20261016_v1/`.
The operation was timing/state neutral but changed the aggregate source-unit
lookup observer from the H63 baseline as follows:

| Metric | H63 secondary mean APE | H67 experiment |
|---|---:|---:|
| `lts__t_sectors_srcunit_tex.sum` | 6.76% | 6.76% |
| `lts__t_sectors_srcunit_tex_lookup_hit.sum` | 11.77% | 33.67% |
| `lts__t_sectors_srcunit_tex_lookup_miss.sum` | 0.00% | 53.16% |

The independent store control and scored L2 source scope are not yet a
like-for-like endpoint; the probe cannot justify changing the corpus observer.
The source was restored to the H63 write-tag arm. No timing, DRAM, or cache-state
model was promoted. A future correction requires operation-split NCU/TEX scope
and NVBit trace crosswalk evidence before any write coalescing or lookup split.
