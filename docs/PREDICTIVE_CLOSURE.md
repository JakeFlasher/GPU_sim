# Predictive modeling contract

Improve cycle-count and duration prediction with analytical or statistical
models calibrated from public observations; treat unobservable internals as
black boxes. [STATUS](STATUS.md) defines scope and implementation maturity.
The filename is retained for existing links.

## Inputs and outputs

Inputs: native trace and static SASS/control sidecars, launch resources,
architecture cards, and separately calibrated model parameters. Custom semantic
frontends remain distinct from native replay.

```text
trace + card + independent calibration
  -> resource/cache models -> composed model-local service frontier
  -> endpoint-specific prediction -> independent NCU comparison
```

Report a point or interval for the named metric, model assumptions, uncertainty,
and OOD/unsupported cases. `gpc__cycles_elapsed.max` is the primary cycle target;
duration requires its own unit/endpoint contract. Keep issue, modeled-event,
completion, launch-envelope, and typed-tail values separate. Do not choose a
frontier, route, or coefficient using the same workload's scored NCU result.

The opt-in service projection now composes its abstract service interval after
the modeled-event frontier before constructing the observer candidate. An
active unknown completion tail propagates an unknown upper endpoint; finite
padding cannot manufacture a bound. This remains model-local composition, not
native-state refinement or a hardware endpoint identity.

Service composition is now typed as an explicit admission/service overlap
policy. Pool rates expose their queue capacities and whether the fitted
initiation interval already includes capacity; capacity-adjusted throughput
probes are never divided by capacity a second time. The v2 probe fitter
requires separate dependent-latency, independent-throughput, queue-capacity,
occupancy, and contention observations; occupancy and contention remain
unmodeled when the current profile schema cannot identify them.

The staged Phase 6 evaluation uses three current SM86 families, joins NCU only
after prediction, and keeps open upper bounds explicit. Its conclusion is
model-local service ranking plus exact fallback for native memory-heavy
regions; it does not establish a silicon design ranking.

## Abstractions

- Resource demand and queue contention replace physical route reconstruction.
- Native binaries/traces are inputs; existing CUDA tools provide compilation.
- Sector/reuse/service models represent cache effects. Physical ownership is
  not required; topology alone does not establish causal read-from edges.
  The cache candidate summary now keeps warp-local instrumentation order separate from
  fence scope, line generation, physical owner, read-from, and completion;
  only supplied temporal witnesses narrow those domains. An unresolved
  cross-SM history adds no positive delay claim, but it removes a finite cache
  history upper endpoint instead of inventing a protocol completion time.
- An explicit observer model predicts the chosen profiler endpoint. CUDA event,
  CUPTI, device timer, and simulator time are not interchangeable measurements.

## Evaluation

Calibrate on independent probes/development data. Keep final workload families
separate from fitting and model selection; retain predictions before joining
score labels. Report cycle/duration error, interval coverage and width, ranking,
per-family failures, and end-to-end cost. Repeated use of a development cohort
is not a new independent holdout.

The checked-in service profile remains an example, not a silicon-calibrated
parameter set. The v2 fitter and three-family staged evaluation use independent
probe inputs and post-prediction NCU joins; finite/open intervals remain model
outputs, not guaranteed confidence bounds. Broader workload-family accuracy,
native-preserved compression, and hardware design ranking remain unproven.
See the [active reification/refactor plan](../plans/sm86_reification_refactor_20261001.md)
and [configuration](archprofile.md) rather than older report task lists.
