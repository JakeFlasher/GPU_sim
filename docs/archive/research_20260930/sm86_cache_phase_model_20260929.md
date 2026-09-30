# SM86 cache-phase response model

Fresh live NCU profiles now include memory-sector and L2 lookup metrics for
independent flash-backward shapes `Sk=64,96,192,256`, both cold
(`--cache-control all`) and paired warm (`--cache-control none`). The target
`Sk=128` is score-only.

## Observed phase transition

| lane | duration | GPC cycles | DRAM read sectors | L2 hit sectors | L2 miss sectors |
|---|---:|---:|---:|---:|---:|
| target cold | 148,416 ns | 200,316 | 2,940 | 3,200 | 2,176 |
| target warm | 129,600 ns | 174,937 | 88 | 5,376 | 0 |

Across independent anchors, warm DRAM reads fall to 96--308 sectors and L2
misses are zero or 1,152 sectors, while cold profiles retain 1,916--4,988
DRAM reads and 1,152--4,224 L2 misses. The target transition is consistent
with this phase family, but these aggregate counters do not expose cache owner,
read-from source, eviction, or DRAM completion order.

The retained machine-readable summary is
`results/alpha/sm86_flash_bwd_cache_phase_goal_20260929/cache_phase_summary.json`.
The endpoint model uses the paired warm duration/GPC response; the memory
metrics are independent phase evidence and are not model features for the
scored target.
