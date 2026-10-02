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

The opt-in path constructs an explicit abstract service composition and queue
frontier after the modeled-event frontier. The live full-trace role is
`independent_post_replay`, so that service envelope is not silently fed into
the native observer endpoint; residual or replacement timing must be selected
explicitly. An active unknown completion tail propagates an unknown upper
endpoint; finite padding cannot manufacture a bound. This remains model-local
composition, not native-state refinement or a hardware endpoint identity.

The current full-trace demand path is explicitly labeled
`independent_post_replay`: its service/queue envelope is reported separately
and is not silently added to the observer endpoint. Residual and replacement
timing are separate typed roles with separate tests. The queue report retains
an open completion upper for unclassified demand and labels pool intervals as
marginal assignments rather than a jointly feasible route schedule.

Service composition is now typed as an explicit admission/service overlap
policy. Pool rates expose their queue capacities and whether the fitted
initiation interval already includes capacity; capacity-adjusted throughput
probes are never divided by capacity a second time. The v2 probe fitter
requires separate dependent-latency, independent-throughput, and queue-capacity
observations. Occupancy and contention rows are accepted as holdout
diagnostics but are emitted as explicitly unmodeled dimensions; their values do
not silently change the fitted rate profile. Independent throughput probes
retain lower/upper initiation-interval endpoints so service uncertainty widens
an upper bound instead of collapsing to one median slope.

The R6 evaluation uses 15 retained SM86 workloads across six operation families
with a checked family-held-out split, joins NCU only after prediction, and keeps
open upper bounds explicit. All real native rows currently remain detailed
replay plus a latent sidecar: the observer projection is a model-local
candidate, not a hardware endpoint identity, and no native compression/runtime
benefit is claimed. The result is deliberately negative where the boundary is
not closed.

Second-round R1 adds an explicit active/inactive completion requirement witness:
an entirely unknown field is inactive only when it is outside the declared
projection; an active unknown service/visibility/writeback/cache/tail field
keeps the observer upper open. The v2 prediction/observation documents carry
the metric, endpoint, clock, cache, launch, repetition, and independently
checked GPU identity. Finite interval coverage is reported separately from
open-tail refusal, and record coverage is not called execution compression.

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
parameter set. The v2 fitter and six-family staged evaluation use independent
probe inputs and post-prediction NCU joins; the current native rows are all
open intervals, so finite interval coverage is not applicable. Open-tail
consistency and lower-point APE are diagnostics, not confidence bounds or
hardware accuracy. Native-preserved compression and hardware design ranking
remain unproven. See the [R6 corpus result](review_round3_20261002/README.md)
and the [third-round rebuttal](../plans/third_round_rebuttal_20261002.md).
See the [active second-round refinement program](../plans/second_round_refinement_20261001.md),
the [first-round implementation history](../plans/sm86_reification_refactor_20261001.md),
and [configuration](archprofile.md) rather than older report task lists.
