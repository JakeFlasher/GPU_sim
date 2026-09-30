# NCU-local clock endpoint audit (2026-09-30)

The fresh campaign
`results/alpha/full_3090ti_20260930_clock_context_warm_live/` collected
per-kernel NCU frequency counters alongside duration and GPC cycles. The
observation-only tool `tools/sm86_ncu_clock_endpoint_audit.py` compares:

```text
implied frequency = NCU GPC cycles / NCU duration
observed frequency = NCU gpc__cycles_elapsed.avg.per_second
```

All 15 rows expose the fields. Results:

| measure | residual |
|---|---:|
| median absolute residual | **0.134%** |
| p90 absolute residual | **0.190%** |
| maximum absolute residual | **0.277%** |
| median NCU GPC frequency | 1.8778025 GHz |

This is the strongest timing-domain relation obtained in the current pass, but
it is explicitly **NCU-local**. It does not establish equivalence to CUDA
events, CUPTI activity timestamps, host timers, simulator frontiers, or final
retirement. NCU fields remain score-only and are not model features or tuning
inputs. It also says nothing about pipe tags, ABI generation, cache ownership,
read-from source, or complete DRAM writeback.

Nsight Compute documents that profiling can adjust clocks and serialize/replay
applications, so comparing endpoint semantics requires stable capture settings
and independent clock context ([profiling reproducibility guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility)).
