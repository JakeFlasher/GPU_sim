# Memory-retirement service probe

This `sm_86` successor to the launch-cycle campaign varies the number,
lane topology, address reuse, and fence scope of final discarded loads,
global stores, and non-returning reductions. It measures the portion of NCU
GPC elapsed that remains after matched active-SM work has ended.

Build and run a correctness smoke:

```bash
make -C workloads/research_probes/memory_retirement_service
workloads/research_probes/memory_retirement_service/memory_retirement_service \
  red none same_stream 84 16 32 3
```

Arguments are `TAIL FENCE LAYOUT BLOCKS DEPTH ACTIVE_LANES TRIALS`.

The completed campaign is under
`results/captures/sm86_memory_retirement_service_20260818_v1`. Reproduce or
resume it with `campaign.py` actions `manifest`, `correctness`, `ncu`,
`source`, `nvbit`, `static`, `simulator`, and `simulator-carded-dram`, then run
`analyze.py`. The result promotes readiness schema 6 ideal work diagnostics,
not a timing service.
