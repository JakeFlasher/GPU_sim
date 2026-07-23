Assumption: “trace-derived” means lanes are re-aggregated by dynamic warp instruction and 32B sector before comparison. Raw lane counts are not comparable.

- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`: `trace_count_comparable`. Count one post-predicate warp LSU load instruction/request, not lanes; replay only affects NCU if passes see different work/cache state.
- `..._op_st.sum`: `trace_count_comparable`. Same request domain for global stores; do not infer from store bytes/sectors.
- `..._op_atom.sum`: `trace_count_comparable` for request count only. Atomics route through/pass L1 toward L2; lane-level atomic fanout/serialization is not the request count.
- `..._op_red.sum`: `trace_count_comparable` for request count only. Same atom/reduction routing caveat.

- `l1tex__t_requests.sum`: `engineering_closure_refuted`. Unified L1/TEX total includes non-global-LSU domains such as local/shared/texture/surface and all pipes; your trace target is global LSU only.

- `lts__t_requests_srcunit_tex.sum`: `engineering_closure_refuted`. L2 request packets from source unit TEX/L1TEX are post-L1, whole-srcunit, and packetized; not the same aggregation domain as per-instruction global memory trace misses.

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: `trace_count_comparable`. Compare coalesced unique 32B L1 sectors per global load request; raw per-lane addresses are not comparable.
- `..._op_st.sum`: `trace_count_comparable`. Compare coalesced 32B L1 store sectors; write-allocation/fill behavior matters later, not for this L1 access-sector count.

- `l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum`: `trace_count_comparable`. Comparable only after the L1 miss-stream filter; this is post-L1 return/fill traffic, not all load sectors.
- `l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum`: `trace_count_comparable` with caution. Must model `.ca/.cg` store forwarding/write policy at the L1-to-XBAR boundary; raw L1 store sectors may overcount/undercount if write allocation/coalescing differs.

- `dram__sectors_read.sum`: `engineering_closure_refuted`. DRAM is chip-global and after L2 fill/eviction/compression/system traffic effects; per-kernel NVBit addresses cannot defensibly close to this hardware domain.
- `dram__sectors_write.sum`: `engineering_closure_refuted`. Same, plus writeback/write-combine/allocation timing makes store trace sectors non-isomorphic to DRAM writes.

- `gmem ld sectors/request`: `trace_count_comparable`. Use comparable L1 load sectors divided by comparable L1 load requests; replay mismatch can perturb NCU ratios.
- `gmem st sectors/request`: `trace_count_comparable`. Same for stores at L1; do not substitute XBAR or DRAM sectors.

- `gmem_total_requests`: `trace_count_comparable` if defined only as LSU global `ld+st+atom+red` requests. Refuted if compared to unified `l1tex__t_requests.sum`.
- `gmem_total_sectors`: `trace_count_comparable` for LSU global `ld+st` L1 sectors. Refuted if it claims atom/red or unified L1 sectors without matching NCU domains.
- `l2_total_sectors`: `trace_count_comparable` only as a modeled post-L1 miss-stream total for tracked global traffic. Refuted against broad `lts__*srcunit_tex*` totals.
- `dram_total_sectors`: `engineering_closure_refuted`.

Defensibly comparable raw counts: the four global LSU request counts, global load/store L1 sector counts, global load/store L1 sectors/request ratios, and narrowly defined post-L1 XBAR sectors. NVIDIA documents L1 request/sector semantics, atom/red pass-through, replay/cache-control effects, and chip-global DRAM contamination in the Nsight Compute guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
