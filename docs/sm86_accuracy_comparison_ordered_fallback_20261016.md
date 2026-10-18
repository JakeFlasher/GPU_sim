# SM86 expanded simulator vs NCU accuracy — ordered-fallback joins — 2026-10-16

This report includes the user-authorized 179 ordered-fallback joins in addition to 107 exact identity joins. The strict exact-only report remains at `../sm86_replay_results/metric_report_h88_constant_reuse_20261016_v1/`.

- Full replay launches: **12392**; successful roots: **286**; full joins scored: **286** (107 exact + 179 ordered fallback).
- Full scope-compatible metric comparison counts vary because scope exclusions are still applied: 150 predictions excluded from the raw/all-row comparison view.
- Ordered-fallback here means the user-authorized same-workload pairing by launch order when trace/NCU kernel-name strings differ; it is included in this expanded report.

## SM/compute

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `gpc__cycles_elapsed.max` | 12.4280% | 9.1402% | 29.4959% | 286 | 0 | 1 |
| `sm__cycles_active.avg` | 24.2196% | 20.3815% | 50.2016% | 286 | 0 | 0 |
| `sm__cycles_active.max` | 23.1099% | 20.0053% | 46.1525% | 286 | 0 | 0 |
| `sm__inst_executed.avg.per_cycle_active` | 18.5568% | 17.0304% | 33.7047% | 286 | 0 | 0 |
| `sm__inst_executed.avg.per_cycle_elapsed` | 11.8848% | 9.0684% | 22.9527% | 286 | 0 | 0 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 2.8875% | 1.4433% | 7.2101% | 286 | 0 | 0 |
| `sm__warps_active.avg.per_cycle_active` | 2.8875% | 1.4433% | 7.2101% | 286 | 0 | 1 |
| `smsp__branch_targets_threads_divergent` | 0.5888% | 0.0000% | 0.0000% | 286 | 0 | 285 |
| `smsp__cycles_active.avg` | 24.2470% | 20.9280% | 49.7242% | 286 | 0 | 0 |
| `smsp__inst_executed.avg` | 0.0817% | 0.0000% | 0.0001% | 286 | 0 | 10 |
| `smsp__inst_executed.sum` | 0.0817% | 0.0000% | 0.0000% | 286 | 0 | 283 |
| `smsp__inst_executed_op_branch.sum` | 3.2428% | 0.0000% | 11.1111% | 286 | 0 | 225 |
| `smsp__inst_executed_op_ldgsts.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 24.0314% | 23.5361% | 39.0975% | 286 | 0 | 0 |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | 0.0046% | 0.0000% | 0.0000% | 276 | 0 | 211 |
| `smsp__sass_inst_executed_op_global_ld.sum` | 44.8103% | 0.0000% | 100.0000% | 278 | 8 | 168 |
| `smsp__warps_active.avg.per_cycle_active` | 2.7082% | 1.6789% | 5.2056% | 286 | 0 | 1 |

## L1/L1TEX

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum` | 6.9627% | 6.9627% | 6.9627% | 286 | 0 | 285 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | 9.1958% | 0.0000% | 33.3333% | 286 | 0 | 260 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | 2.3095% | 0.0000% | 0.0000% | 286 | 0 | 280 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 25.4200% | 0.0000% | 0.0000% | 286 | 0 | 282 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum` | 0.1671% | 0.0000% | 0.0000% | 286 | 0 | 285 |
| `l1tex__t_requests_pipe_lsu_mem_local_op_st.sum` | 22.3897% | 0.5025% | 0.5025% | 286 | 0 | 284 |
| `l1tex__t_sector_hit_rate.pct` | 49.2537% | 33.3333% | 100.0000% | 286 | 0 | 35 |
| `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` | 290.1029% | 9.7561% | 150.0000% | 254 | 0 | 36 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 0.1036% | 0.0000% | 0.0000% | 285 | 1 | 278 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` | 290.0483% | 9.7561% | 150.0000% | 285 | 1 | 70 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum` | 38.5153% | 50.0000% | 50.0000% | 285 | 1 | 70 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | 75.8471% | 75.1253% | 75.1253% | 286 | 0 | 283 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum` | 76.5029% | 76.1278% | 76.1278% | 286 | 0 | 283 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum` | N/A | N/A | N/A | 286 | 0 | 283 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum` | 78.1438% | 75.1256% | 75.1256% | 286 | 0 | 283 |

## L2

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `lts__t_sectors_srcunit_tex.sum` | 3.3208% | 0.0000% | 14.2857% | 283 | 3 | 222 |
| `lts__t_sectors_srcunit_tex_lookup_hit.sum` | 20.3953% | 0.0000% | 25.0000% | 283 | 3 | 219 |
| `lts__t_sectors_srcunit_tex_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 283 | 3 | 283 |
| `lts__t_sectors_srcunit_tex_op_atom_dot_alu.sum` | 130.5092% | 0.0000% | 0.0000% | 286 | 0 | 284 |
| `lts__t_sectors_srcunit_tex_op_atom_dot_alu_lookup_hit.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 284 |
| `lts__t_sectors_srcunit_tex_op_atom_dot_alu_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | 1.0396% | 0.0000% | 0.0000% | 283 | 3 | 277 |
| `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum` | 4.3416% | 0.0000% | 0.0000% | 283 | 3 | 274 |
| `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 283 | 3 | 283 |
| `lts__t_sectors_srcunit_tex_op_red.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `lts__t_sectors_srcunit_tex_op_red_lookup_hit.sum` | 20.0000% | 0.0000% | 0.0000% | 286 | 0 | 285 |
| `lts__t_sectors_srcunit_tex_op_red_lookup_miss.sum` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `lts__t_sectors_srcunit_tex_op_write.sum` | 11.9647% | 0.0000% | 0.0000% | 286 | 0 | 260 |
| `lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum` | 11.9750% | 0.0000% | 0.0000% | 286 | 0 | 260 |
| `lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum` | N/A | N/A | N/A | 286 | 0 | 286 |

## DRAM

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `dram__bytes_read.sum` | 52.6937% | 51.2224% | 91.6667% | 286 | 0 | 0 |
| `dram__sectors_read.sum` | 39.0270% | 37.0000% | 90.3226% | 165 | 121 | 0 |

## launch

| Metric | Mean APE | Median APE | P90 APE | Compared | Scope | Exact |
|---|---:|---:|---:|---:|---:|---:|
| `launch__block_size` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__grid_size` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__occupancy_limit_blocks` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__occupancy_limit_registers` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__occupancy_limit_shared_mem` | 476.3325% | 525.0000% | 554.5455% | 286 | 0 | 16 |
| `launch__occupancy_limit_warps` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__registers_per_thread` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__registers_per_thread_allocated` | 3.3435% | 0.0000% | 0.0000% | 286 | 0 | 271 |
| `launch__shared_mem_config_size` | 473.2299% | 525.0000% | 525.0000% | 286 | 0 | 16 |
| `launch__shared_mem_per_block_allocated` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__shared_mem_per_block_dynamic` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |
| `launch__shared_mem_per_block_static` | 0.0000% | 0.0000% | 0.0000% | 286 | 0 | 286 |

## Notes

Metrics with all-zero targets show N/A relative APE but exact counts are retained. Full machine-readable data: `../sm86_replay_results/metric_report_h91_ordered_fallback_20261016_v1/summary.json` and `metric_comparisons.jsonl.zst`.
