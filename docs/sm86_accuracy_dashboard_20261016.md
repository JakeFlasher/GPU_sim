# SM86 accuracy and coverage dashboard — 2026-10-16

Raw/all-row H41 APE is authoritative; H42 scope-compatible APE is a secondary like-for-like view. Scope exclusions are unresolved, not completed mechanisms.

| Metric | Raw compared | Raw mean APE | Secondary compared | Secondary mean APE | Excluded |
|---|---:|---:|---:|---:|---:|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 107 | 100.96% | 106 | 0.21% | 1 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` | 107 | 30.71% | 106 | 30.71% | 1 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum` | 107 | 58.44% | 106 | 32.39% | 1 |
| `lts__t_sectors_srcunit_tex.sum` | 107 | 28.24% | 104 | 6.76% | 3 |
| `lts__t_sectors_srcunit_tex_lookup_hit.sum` | 107 | 35.04% | 104 | 11.77% | 3 |
| `lts__t_sectors_srcunit_tex_lookup_miss.sum` | 107 | 27.0% | 104 | 0.0% | 3 |
| `dram__sectors_read.sum` | 107 | 40.91% | 58 | 26.84% | 49 |
| `sm__cycles_active.avg` | 107 | 18.02% | 107 | 18.02% | 0 |
| `gpc__cycles_elapsed.max` | 107 | 12.9% | 107 | 12.9% | 0 |

## Corpus and tails

- Full-first corpus: 356 roots, 286 `ok`, 62 controls, 8 replay/input failures, 12,392 launches, exactly 12 workers, `.jsonl.zst` artifacts.
- Exact independent joins: 107 identity rows; ordered fallback rows remain coverage metadata and are not primary APE.
- H37/H38 pressure and H48 constant-form observers are timing/traffic neutral.
- Dirty-writeback and coherence tails remain typed unmodeled obligations on the scored cycle/DRAM surfaces.

## Closure boundary

No overall sub-10% closure is claimed. The dashboard distinguishes scope compatibility from model completion; remaining scope-compatible DRAM/cycle residuals require independent probes and bounded mechanism evidence.

## H102 complete-corpus update (2026-10-16)

Authoritative acceptance artifact: `metric_report_h102_local_hybrid_20261016_v1`
from full-first replay `full_corpus_h102_local_hybrid_20261016_143358`.
Coverage is 356 roots (286 ok, 62 control, 8 replay errors), 12 workers, and
12,392 full launches. H102 local endpoints are 1.7972% load-sector APE,
0.3353% store-sector APE, 1.7972% local lookup-hit APE, and exact local lookup
misses. H99's LTC128B endpoint remains active: global L1 lookup hit 17.7036%
and miss 18.1662%; aggregate L2 TEX is 2.9180%/3.8618% with exact misses.
Elapsed IPC remains 12.3073%, so no sub-10% IPC claim is made.

## H115/H116 complete-corpus update (2026-10-16)

The latest accepted full-first artifact is
`metric_report_h116_uniform_constant_observer_20261016_v1`, based on 356 roots
(286 ok, 62 controls, 8 replay errors), 12,392 full launches, and twelve
workers. H115's mixed-client L1 endpoint is selected; H116's uniform-constant
classification is observer-only.

| Metric | Secondary mean APE |
|---|---:|
| `l1tex__...global_op_ld_lookup_hit.sum` | **9.155%** |
| `l1tex__...global_op_ld_lookup_miss.sum` | **9.828%** |
| `lts__t_sectors_srcunit_tex.sum` | 2.918% |
| `lts__t_sectors_srcunit_tex_lookup_hit.sum` | 3.862% |
| `lts__t_sectors_srcunit_tex_lookup_miss.sum` | 0.000% |
| `dram__sectors_read.sum` | **19.474%** (58 joins) |
| `sm__inst_executed.avg.per_cycle_elapsed` | 12.307% |
| `sm__inst_executed.avg.per_cycle_active` | 14.487% |
| `sm__cycles_active.avg` | 18.020% |
| `gpc__cycles_elapsed.max` | 12.901% |

The 1008 GB/s card-peak service-rate ablation was evaluated on the complete
corpus and rejected: it worsened GPC, SM-active, and both IPC endpoints versus
the retained 825 GB/s independent warm-rate hypothesis. No clock or NCU-derived
multiplier is promoted.
