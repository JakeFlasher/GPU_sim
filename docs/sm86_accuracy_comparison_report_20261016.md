# SM86 simulator vs NCU accuracy comparison — 2026-10-16

Latest H88 full-first report: `../sm86_replay_results/metric_report_h88_constant_reuse_20261016_v1/`.

## Coverage and interpretation

- Corpus roots: **356**; replay status: **286 ok**, **62 control_error**, **8 replay_error**.
- Full launches replayed: **12392**, exactly 12 workers; artifact is compressed `.jsonl.zst`.
- NCU identity joins: **107**; ordered fallback joins: **179** (coverage metadata only, not scored).
- The `Compared` column below counts exact identity NCU joins with a finite simulator prediction and target. It is not the 12,392 replay-launch count.
- Mean/median/P90 APE are computed over nonzero NCU targets. `Scope` counts explicit trace/NCU scope exclusions; they remain in raw authority but are excluded from the secondary like-for-like APE table.
- NCU values are never replay/model inputs; this is post-replay comparison only.

## SM/compute

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `gpc__cycles_elapsed.max` | 12.9008% | 10.2447% | 27.1431% | 107 | 0 | 1 |
| `sm__cycles_active.avg` | 18.0202% | 13.8670% | 40.5836% | 107 | 0 | 0 |
| `sm__cycles_active.max` | 17.8377% | 12.8127% | 42.0671% | 107 | 0 | 0 |
| `sm__inst_executed.avg.per_cycle_active` | 14.4865% | 12.7340% | 29.6113% | 107 | 0 | 0 |
| `sm__inst_executed.avg.per_cycle_elapsed` | 12.3073% | 11.7566% | 21.9743% | 107 | 0 | 0 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 2.6207% | 1.2016% | 5.2676% | 107 | 0 | 0 |
| `sm__warps_active.avg.per_cycle_active` | 2.6207% | 1.2016% | 5.2676% | 107 | 0 | 1 |
| `smsp__branch_targets_threads_divergent` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `smsp__cycles_active.avg` | 17.5137% | 12.9047% | 42.2589% | 107 | 0 | 0 |
| `smsp__inst_executed.avg` | 0.0073% | 0.0000% | 0.0000% | 107 | 0 | 6 |
| `smsp__inst_executed.sum` | 0.0073% | 0.0000% | 0.0000% | 107 | 0 | 106 |
| `smsp__inst_executed_op_branch.sum` | 2.1245% | 0.0000% | 0.0000% | 107 | 0 | 99 |
| `smsp__inst_executed_op_ldgsts.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 17.2182% | 16.6592% | 33.3926% | 107 | 0 | 0 |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | 0.0000% | 0.0000% | 0.0000% | 103 | 0 | 71 |
| `smsp__sass_inst_executed_op_global_ld.sum` | 95.1727% | 100.0000% | 100.0000% | 106 | 1 | 10 |
| `smsp__warps_active.avg.per_cycle_active` | 2.5095% | 1.1993% | 5.7621% | 107 | 0 | 0 |

## L1/L1TEX

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum` | N/A | N/A | N/A | 107 | 0 | 107 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | 5.6229% | 0.0000% | 32.9787% | 107 | 0 | 99 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | 2.4183% | 0.0000% | 0.0000% | 107 | 0 | 103 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 59.0862% | 0.0000% | 0.0000% | 107 | 0 | 105 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum` | 0.1671% | 0.0000% | 0.0000% | 107 | 0 | 106 |
| `l1tex__t_requests_pipe_lsu_mem_local_op_st.sum` | 22.3897% | 0.5025% | 0.5025% | 107 | 0 | 105 |
| `l1tex__t_sector_hit_rate.pct` | 29.0370% | 17.2388% | 71.4286% | 107 | 0 | 11 |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | 30.7137% | 4.1667% | 73.1801% | 107 | 0 | 20 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 0.2096% | 0.0000% | 0.0000% | 106 | 1 | 105 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` | 30.7137% | 4.1667% | 73.1801% | 106 | 1 | 22 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum` | 32.3914% | 38.5617% | 50.0000% | 106 | 1 | 22 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | 75.8471% | 75.1253% | 75.1253% | 107 | 0 | 104 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum` | 76.5029% | 76.1278% | 76.1278% | 107 | 0 | 104 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum` | N/A | N/A | N/A | 107 | 0 | 104 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum` | 78.1438% | 75.1256% | 75.1256% | 107 | 0 | 104 |

## L2

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `lts__t_sectors_srcunit_tex.sum` | 6.7551% | 0.0000% | 14.2857% | 104 | 3 | 74 |
| `lts__t_sectors_srcunit_tex_lookup_hit.sum` | 11.7747% | 0.0000% | 40.0000% | 104 | 3 | 72 |
| `lts__t_sectors_srcunit_tex_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 104 | 3 | 104 |
| `lts__t_sectors_srcunit_tex_op_atom_dot_alu.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 106 |
| `lts__t_sectors_srcunit_tex_op_atom_dot_alu_lookup_hit.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 106 |
| `lts__t_sectors_srcunit_tex_op_atom_dot_alu_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | 1.8998% | 0.0000% | 0.0000% | 104 | 3 | 99 |
| `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum` | 5.4357% | 0.0000% | 0.0000% | 104 | 3 | 97 |
| `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 104 | 3 | 104 |
| `lts__t_sectors_srcunit_tex_op_red.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `lts__t_sectors_srcunit_tex_op_red_lookup_hit.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `lts__t_sectors_srcunit_tex_op_red_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `lts__t_sectors_srcunit_tex_op_write.sum` | 9.3704% | 0.0000% | 0.0000% | 107 | 0 | 101 |
| `lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum` | 9.3631% | 0.0000% | 0.0000% | 107 | 0 | 101 |
| `lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum` | N/A | N/A | N/A | 107 | 0 | 107 |

## DRAM

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `dram__bytes_read.sum` | 40.9130% | 40.2542% | 70.0820% | 107 | 0 | 0 |
| `dram__sectors_read.sum` | 26.8371% | 28.5294% | 44.4444% | 58 | 49 | 0 |

## Launch/resource configuration

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `launch__block_size` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__grid_size` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__occupancy_limit_blocks` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__occupancy_limit_registers` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__occupancy_limit_shared_mem` | 430.7942% | 525.0000% | 528.5714% | 107 | 0 | 5 |
| `launch__occupancy_limit_warps` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__registers_per_thread` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__registers_per_thread_allocated` | 8.9369% | 0.0000% | 50.0000% | 107 | 0 | 92 |
| `launch__shared_mem_config_size` | 428.9136% | 525.0000% | 525.0000% | 107 | 0 | 5 |
| `launch__shared_mem_per_block_allocated` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__shared_mem_per_block_dynamic` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |
| `launch__shared_mem_per_block_static` | 0.0000% | 0.0000% | 0.0000% | 107 | 0 | 107 |

## Metrics with zero-target-only or no relative APE

| Metric | Compared | Exact | Note |
|---|---:|---:|---|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum` | 107 | 107 | All compared targets are zero; relative APE is undefined. |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum` | 107 | 104 | All compared targets are zero; relative APE is undefined. |
| `lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum` | 107 | 107 | All compared targets are zero; relative APE is undefined. |

## Unavailable inventory

The report contains **91** predicted metric names and **1647** target/prediction names unavailable in both retained modes. Exact family-grouped names/counts are in `summary.json` under `unavailable_by_family`; repeated N/A tables are intentionally omitted.

## Key current residuals

- L1 global lookup hit/miss: 30.7137% / 32.3914% secondary APE; H79 packet candidate is observer-only because LTC128B form scope regresses raw authority.
- L2 operation write sectors/write lookup hits: 9.3704% / 9.3631% after H80/H81 packet composition.
- DRAM read sectors: 26.8371% secondary APE, with 49 scope exclusions.
- GPC elapsed / SM-active / SMSP-active: 12.9008% / 18.0202% / 17.5137%.

## H102 acceptance note

Only complete 286-successful-root replay is authoritative. H102's local
frame/coalescing endpoint reaches 1.7972% local-load and 0.3353% local-store
sector APE, with 1.7972% local lookup-hit APE and exact local lookup misses.
All non-local metric rows are unchanged from H99. Global L1 and IPC remain
above the requested sub-10% threshold and are explicitly unresolved.
