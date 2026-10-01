> **Dated opt-in prototype note.** This analyzer is retained engineering context and does not alter the native replay completion bar or create a workflow gate.

# Model-local frontier prototype integration

> Compatibility note: the source module and v1 JSON fields retain the
> historical `causal_frontier`/`causal_*` names. They currently certify only
> descriptor-model self-equivalence. They do not certify equivalence to native
> `SmCore`, hardware scheduling, or silicon timing. A later schema revision
> will reserve “causal” for independently derived native order/state evidence.

The merged prototype surfaces are intentionally opt-in. The retained
`sim-sm::causal_frontier` module
accepts explicit dynamic descriptors and joins each record to the native
`FunctionSchedule::issue_slot(pc)`. It preserves the simulator's
`PipelineClass`, decoded control bits, four-SMSP mapping, initiation interval,
in-flight limit, and completion count. A region is summarizable only when the
producer says its dependencies are resolved, its controls are clean, it has no
external side effect, and it has one completion unit; otherwise the analyzer
reports the region as reified.

The current analyzer executes the exact and summarized recurrences separately.
It admits only homogeneous latency/issue/capacity intervals with an exclusive
SMSP and no incoming pipeline completions. The summary uses a closed-form
capacity recurrence, materializes the bounded pending-completion state, and
compares terminal state and issue frontiers with the detailed execution. Mixed
timing, incoming work, non-contiguous sequences, controls, dependencies, and
effects remain detailed. Certified regions are emitted individually; their
frontier values are not summed into elapsed time.

Descriptor recurrence equality is a hard success condition for this analyzer.
A mismatch in the complete modeled recurrence state or issue/completion
frontier returns a typed `StateMismatch` error. The serialized
`terminal_state_matches` field is a positive **model-local** diagnostic witness;
it is not a native-state or hardware-timing certificate.

Summary eligibility now also requires an explicit complete resident-warp roster
through `summarize_with_residency`. The legacy `summarize` entry point treats
residency as unknown and reifies all candidate regions. A candidate list cannot
prove that a stalled or non-candidate warp is absent; adding an extra resident
warp to the release probe removes the collapse, while the complete roster keeps
the original one-region collapse.

The event-order contract also follows the MGPUSim/Akita review: equal-time
events need an explicit class/phase, and an `L2AccessBatch` sequence reservation
must not hide logical positions from a future transformer. Retry notifications
are future state transitions, so timestamp-only endpoint checks cannot certify
them.
The typed queue projection preserves batch sequence spans and the next unused
insertion position; summaries cannot infer hidden ordering from an anchor
sequence alone.

`sim-run` accepts an observation-only Nsight Compute raw CSV through
`--ncu-csv PATH`, with `--ncu-launch ID` required for multi-launch exports. The
result places raw timing and resource/dependency projections under
`ncu_observation`. NCU values never enter replay configuration or timing.
Kernel identity falls back to the raw export's `Kernel Name` column when
`launch__kernel_name` is empty. FMA and LSU instruction counts remain separate
optional observations and are never substituted for total issue activity.
The smsp__inst_executed.sum total and .avg per-SMSP roll-up are also kept as
distinct observations.
Wide NCU exports are normalized from their per-column units row, so automatic
time scaling remains expressed in the observation's nanosecond field.

The direct SM86 probe is in
[`workloads/prototype_sector_probe`](../workloads/prototype_sector_probe/).
Its contiguous, stride-eight, and half-active cases match structural sector
counts exactly. An offset-one case intentionally exposes cache reuse: NCU
traffic is lower than the uncached structural footprint. This is retained as a
model boundary.

The descriptor bridge does not invent register operands, CTA-wide barriers,
unsupported memory roles, or child completion timestamps. A future producer
must supply those facts before causal summaries are used to change native
replay scheduling. Planner wall-clock telemetry is runtime-only and omitted
from serialized reports so identical analyses remain byte-deterministic. No
top-tier novelty or silicon cycle-accuracy claim follows
from this integration alone.

The normal SM86 campaign now exercises both merged observation surfaces. After
the NCU stage, `campaign.py` passes the matching raw CSV to `sim-run`, where it
is serialized under `ncu_observation` without entering replay configuration.
The `--stage all` path also runs the release `causal-frontier-probe` and records
its collapse/reification report in the campaign manifest. This makes the
merged components executable and regression-visible while preserving the
production timing boundary until a real descriptor producer is available.
