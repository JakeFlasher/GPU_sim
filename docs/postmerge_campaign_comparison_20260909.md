# Post-merge SM86 campaign comparison

Revision `9df50c47760b590c525b3a3a6723225eb2460f84` was rebuilt on the pinned
RTX 3090 Ti host and run through the complete small and large campaign. The
durable evidence roots are:

- `results/captures/sm86_postmerge_campaign_small_20260909_v1` — 8 workloads.
- `results/captures/sm86_postmerge_campaign_large_20260909_v1` — 7 workloads.

Both runs used `--ncu-repeats 7`, `--simulator-repeats 1`, and
`--require-reference-clocks`. Every row completed correctness, NCU, NVBit,
static-control, and simulator stages. The simulator command for every row
includes `--ncu-csv` and the matching rep-1 CSV, so the merged observation
parser is exercised without allowing NCU values to configure replay.

| Cohort | Rows | Correctness | NCU | NVBit | Static | Simulator | Clock mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 0 |
| Large | 7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 0 |

The native causal-frontier probe is recorded in each manifest. It reports 26
exact events versus 19 causal events, one collapsed safe region, two reified
contended regions, and the same final cycle (9). The probe is a component
integration signal; the production trace producer still does not expose all
dependency and child-completion facts required to change replay scheduling.

## Non-regression result

The current binary was replayed against the pre-merge retained trace/control
pairs from `sm86_regenerated_workloads_{small,large}_20260905_v1`. The compared
fields were instruction count, global load/store sectors, DRAM read sectors,
all cache-observer hit/miss pairs, raw issue frontier, and modeled terminal
frontier.

| Cohort | Inputs | Identical comparison rows | Scored-field regressions |
|---|---:|---:|---:|
| Small | 8 | 8 | 0 |
| Large | 7 | 7 | 0 |

Some fresh captures move internal readiness timestamps and expose corrected
non-scored `bandwidth_delayed_responses` gauges. No scored field changed on the
same trace, so no regression fix was warranted. Fresh NCU medians remain
diagnostic until a new frozen scorer binds them; per-workload fitting was not
performed.

The largest fresh comparison-only DRAM gaps were small flash-attention forward
(1,280 simulator versus 1,596 NCU median, 19.8%), large flash-attention
backward (2,176 versus 2,940, 26.0%), and small sparse 2:4 GEMM (1,536 versus
1,716, 10.5%). The same-input differential shows no corresponding simulator
change, so these are unresolved measurement/request-mapping differences rather
than post-merge regressions.

The presentation-ready state report, with the commit timeline and ASCII
diagrams, is [`gpu_simulator_current_state_20260909.md`](../../GPU_sims_notes/gpu_simulator_presentation/gpu_simulator_current_state_20260909.md).
