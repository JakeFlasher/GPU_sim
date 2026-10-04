# L1TEX arrival/output-replay service probe

This `sm_86` probe holds trace-derived ideal L1TEX wavefront work fixed while
varying resident warps, CTA partitioning, arrival skew, lane topology, and
operation class. It separates input/ideal work from hardware output replay.

Build and run a fixed-512-wavefront correctness sample:

```bash
make -C workloads/research_probes/l1tex_arrival_replay_service
workloads/research_probes/l1tex_arrival_replay_service/l1tex_arrival_replay_service \
  store line 84 8 8 32 0 3
```

Arguments are
`TAIL LAYOUT BLOCKS WARPS DEPTH ACTIVE_LANES GAP_STEPS_PER_WARP TRIALS`.

The completed historical campaign is under
`results/captures/sm86_l1tex_arrival_replay_service_20260818_v1`. Resume it
with `campaign.py` actions `manifest`, `correctness`, `ncu`, `source`,
`l2-requests`, `nvbit`, `static`, `simulator`, and
`simulator-carded-dram`. The successor schema-7 study adds
`memory-service-candidates`, using `memory_service_candidates.json`, then runs
`analyze.py`. Correctness, NVBit, replay, and NCU use the one rebuilt
`l1tex_arrival_replay_service` executable; the former trace twin is removed so
the campaign cannot compare different binaries by accident. Its 75 deterministic candidate pairs select retrospective
SM-shared 10-cycle STORE-line and 16-cycle RED-line hypotheses, reject SMSP
scope, and leave both rates inactive pending finite-queue/backpressure tests.
