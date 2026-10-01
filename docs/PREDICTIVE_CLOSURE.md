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
  -> resource/cache models -> separate simulator frontiers
  -> endpoint-specific prediction -> independent NCU comparison
```

Report a point or interval for the named metric, model assumptions, uncertainty,
and OOD/unsupported cases. `gpc__cycles_elapsed.max` is the primary cycle target;
duration requires its own unit/endpoint contract. Keep issue, modeled-event,
completion, launch-envelope, and typed-tail values separate. Do not choose a
frontier, route, or coefficient using the same workload's scored NCU result.

## Abstractions

- Resource demand and queue contention replace physical route reconstruction.
- Native binaries/traces are inputs; existing CUDA tools provide compilation.
- Sector/reuse/service models represent cache effects. Physical ownership is
  not required; topology alone does not establish causal read-from edges.
- An explicit observer model predicts the chosen profiler endpoint. CUDA event,
  CUPTI, device timer, and simulator time are not interchangeable measurements.

## Evaluation

Calibrate on independent probes/development data. Keep final workload families
separate from fitting and model selection; retain predictions before joining
score labels. Report cycle/duration error, interval coverage and width, ranking,
per-family failures, and end-to-end cost. Repeated use of a development cohort
is not a new independent holdout.

The current opt-in projection and fitter are preliminary. The supplied service
profile is uncalibrated; finite intervals are model outputs, not guaranteed
bounds or confidence intervals. Calibration and held-out accuracy are unfinished.
See the [active plan](../plans/sm86_predictive_closure_implementation_20261001.md)
and [configuration](archprofile.md) rather than older report task lists.
