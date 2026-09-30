# SM86 live campaign continuation — 2026-09-29

This report records the two live performance campaigns run for the active
research goal. It intentionally contains no test, hashing, freshness, or
verification workflow. NCU is an independent comparison endpoint; its values
were not used to select simulator parameters or branches.

## Host and commands

The card was GPU index 0, RTX 3090 Ti, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
615.71.09, P0, 1350 MHz SM, and 10501 MHz memory.

```text
python3 workloads/research_probes/sm86_ncu_demo_benchmark/alpha_campaign.py \
  --output results/alpha/full_3090ti_20260929_goal_cold \
  --cache-control all --timeout 900

python3 workloads/research_probes/sm86_ncu_demo_benchmark/alpha_campaign.py \
  --output results/alpha/full_3090ti_20260929_goal_warm \
  --cache-control none --timeout 900
```

Each campaign ran the 15 base/large workloads with five native repeats, three
simulator replays, and seven NCU repeats. Both finished 15/15 without recorded
stage errors. All 375 process steps in each campaign used Linux pidfd
readiness rather than coarse wait polling.

## Across-workload medians

| endpoint | cold (`all`) | warm (`none`) |
|---|---:|---:|
| native CUDA-event mean | 0.009469 ms | 0.009450 ms |
| NCU duration | 9,952 ns | 8,608 ns |
| NCU GPC cycles | 13,374 | 11,562 |
| simulator event frontier | 11,780 cycles | 11,780 cycles |
| simulator GPC envelope | 13,596 cycles | 13,596 cycles |
| host replay wall | 0.629171 s | 0.635269 s |

The warm/cold median ratios are 0.816 for NCU duration and 0.818 for NCU GPC
cycles. This is direct evidence that the NCU endpoint is cache-state
sensitive. The simulator frontiers are unchanged because this campaign uses
the existing fixed replay/cache contract; this is not an elapsed-time accuracy
claim.

The per-workload warm NCU/GPC range is heterogeneous. The warm
simulator-event-to-NCU-GPC ratios range from 0.797 to 1.345 (median 1.029),
while the GPC-envelope-to-NCU-GPC ratios range from 0.930 to 1.489 (median
1.203). These ratios are descriptive endpoint comparisons only. They must not
be converted into a global timing multiplier or used to tune the card.

## What this campaign adds

* It refreshes the complete 15-workload NCU/native/replay population under the
  current driver and clock state for both cache controls.
* It strengthens the cache-domain observation: changing NCU cache control
  changes measured duration and GPC cycles while native CUDA-event and
  simulator values remain near their prior medians.
* It provides no new physical-pipe witness: the collected pipe metrics remain
  aggregate `smsp__inst_executed_pipe_*` counters.
* It provides no new cache-owner/read-from/DRAM-completion witness: NCU cache
  state sensitivity does not identify which cache or memory endpoint served a
  request.
* It provides no new ABI/compiler witness: the workloads use the existing
  NVIDIA compilation and bounded trace path.
* It provides no unified timing endpoint: NCU, CUDA event, simulator frontiers,
  and host replay still have different start/end semantics.

## Goal interpretation

The live performance deliverable is refreshed and complete for this pass. The
four mechanism blockers remain open under the boundary-constrained model in
`docs/sm86_blocker_modeling_20260929.md`. No result in this report is a
blocker-closure claim.

