# Causal-frontier prototype integration

The merged prototype surfaces are intentionally opt-in. `sim-sm::causal_frontier`
accepts explicit dynamic descriptors and joins each record to the native
`FunctionSchedule::issue_slot(pc)`. It preserves the simulator's
`PipelineClass`, decoded control bits, four-SMSP mapping, initiation interval,
in-flight limit, and completion count. A region is summarizable only when the
producer says its dependencies are resolved, its controls are clean, it has no
external side effect, and it has one completion unit; otherwise the analyzer
reports the region as reified.

`sim-run` accepts an observation-only Nsight Compute raw CSV through
`--ncu-csv PATH`, with `--ncu-launch ID` required for multi-launch exports. The
result places raw timing and resource/dependency projections under
`ncu_observation`. NCU values never enter replay configuration or timing.

The direct SM86 probe is in
[`workloads/prototype_sector_probe`](../workloads/prototype_sector_probe/).
Its contiguous, stride-eight, and half-active cases match structural sector
counts exactly. An offset-one case intentionally exposes cache reuse: NCU
traffic is lower than the uncached structural footprint. This is retained as a
model boundary.

The descriptor bridge does not invent register operands, CTA-wide barriers,
unsupported memory roles, or child completion timestamps. A future producer
must supply those facts before causal summaries are used to change native
replay scheduling. No top-tier novelty or silicon cycle-accuracy claim follows
from this integration alone.
