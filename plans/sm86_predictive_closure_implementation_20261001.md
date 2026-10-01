# SM86 predictive modeling implementation plan

Updated 2026-10-01. Improve held-out NCU accuracy with analytical/statistical
models and public black-box measurements. Follow [STATUS](../docs/STATUS.md)
and [the model contract](../docs/PREDICTIVE_CLOSURE.md); hidden internals are
not prerequisites. Existing filenames and runtime schemas are retained.

## Implemented

- `sim-semantics::latent`: demand/profile types and service/observer projections.
- `sim-sm::metrics`: candidate demand and operand-bank pressure summaries.
- `sim-run --latent-envelope --predictive-service-profile FILE`: opt-in output;
  default native replay remains unchanged.
- `tools/calibrate_sm86_service_profile.py`: preliminary offline fitter.

These are runnable interfaces, not demonstrated accuracy. The example service
rates and observer padding are uncalibrated; cache summaries describe topology,
not actual read-from events.

## Remaining vertical slices

1. **Interval correctness.** Check empty demand, unknown classes, capacity
   domains, checked arithmetic, and dependencies. An issue/event upper bound
   cannot stand in for an unknown completion upper bound. Keep frontiers and
   their clock/endpoint meanings separate; never reduce them to one ranking score.
2. **Probe calibration.** Characterize dependency latency, initiation rate,
   bank contention, occupancy, and memory service on separate probe kernels.
   Account for timer/loop overhead and parameter identifiability. Record units,
   operating point, input identity, and uncertainty; do not fit score workloads.
3. **Cache effects.** Use address/sector/reuse and queueing models. Include cache
   history and declared synchronization where available. Treat absent physical
   ownership as a black-box variable; test traffic and causal ordering separately.
4. **Observer model.** State the NCU metric, replay/cache/clock policy, and work
   included by the endpoint. Learn/measure nuisance terms on separate data;
   report missing support instead of substituting another timer or a zero tail.
5. **Held-out evaluation.** Save predictions before joining NCU targets. Report
   MAPE, absolute log error, rank correlation, interval coverage/width, per-family
   failures, and OOD cases. Split by workload family, not repeated launches.
6. **Cost and integration.** Run the real trace path after each change. Compare
   default output, termination, counters, memory use, and total trace/replay/model
   runtime before making a speed or accuracy claim.

## Completion bar

A declared workload population has independently evaluated NCU predictions with
reported error, calibrated uncertainty, and reproducible cost. Select numeric
acceptance targets before inspecting score labels. Do not call the work complete
from schema checks, self-authored examples, or wide uncalibrated intervals.

When progress stalls, inspect the failing model assumption and the relevant
public paper/repository; run a small discriminating experiment rather than
adding a new framework or searching for inaccessible telemetry.
