# SM86 L2 request-frontier probe

This is an independent, unscored microbenchmark for the local RTX 3090 Ti.
It keeps one fixed `.cg` global-load SASS family while varying resident warp
count, independent burst depth, address stride, and reuse.  The objective is
to identify whether the L1TEX-to-L2 request/response path has a finite service
frontier that is absent from the simulator's current latency-only L2 model.

The probe is deliberately separate from the 15-workload accuracy cohort.  No
NCU value, workload name, or fitted row correction is passed to the simulator.

Build and run a smoke point:

```bash
make -C workloads/research_probes/l2_queue_frontier
workloads/research_probes/l2_queue_frontier/l2_queue_frontier \
  unique 84 8 32 1 32 1
```

For NCU, collect the same binary with `--launch-skip 1 --launch-count 1
--replay-mode application --cache-control all` and the following observation
metrics:

```text
sm__cycles_active.max,smsp__cycles_active.max,
smsp__inst_issued.sum,smsp__inst_executed.sum,
l1tex__m_l1tex2xbar_req_cycles_active.sum,
l1tex__m_l1tex2xbar_req_cycles_stalled.sum,
l1tex__texin_sm2tex_req_cycles_active.sum,
l1tex__texin_sm2tex_req_cycles_stalled.sum,
l1tex__cycles_active.sum,lts__cycles_active.sum,
lts__xbar2lts_cycles_active.sum,lts__lts2xbar_cycles_active.sum,
dram__sectors_read.sum
```

The current experiment matrix is intentionally small: `warps ∈ {1,2,4,8}`,
`loads ∈ {1,4,8,16,32,64}`, `stride_lines ∈ {1,8,32}`, and both `unique` and
`reuse` modes at 84 blocks.  A point is retained only when correctness,
instruction-count conservation, and launch identity pass.  A candidate L2
queue is not promoted from this probe alone; it must also replay an NVBit trace
and preserve the cache/traffic gates on the held-out simulator cohort.
