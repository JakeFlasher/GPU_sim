# Predictive closure contract

**Updated 2026-10-01.** This is the active replacement for the blocked exact-
mechanism objective. It does not claim access to NVIDIA's hidden pipe, cache,
compiler, or completion state.

## Objective

Predict the declared Nsight Compute result for a retained SM86 trace using
public architecture-card inputs, the dynamic trace, and separately calibrated
black-box model parameters:

```text
trace capsule + architecture card
  -> route/cache latent quotient
  -> simulator frontiers
  -> endpoint-specific NCU projection
  -> held-out NCU comparison
```

NCU values are score labels only. They are not features, tuning inputs, route
choices, or training data for the same scored workload.

## Replacement mechanisms

| Hidden mechanism | Model replacement |
|---|---|
| Physical `(SM,SMSP,pipe,slot)` route | Candidate abstract resource pools, service demand, queue contention, and intervals |
| Arbitrary SM86 compiler/ABI/linker | Opaque native trace capsule; bounded semantic IR for custom programs |
| Cache owner/read-from/DRAM receipt | Sectorized cache replay, reuse profiles, causal write/read candidate sets, model-local service completion |
| Common hardware/simulator endpoint | Separate issue/event/tail frontiers plus an explicit NCU observer descriptor |

A singleton route, owner, compiler realization, or endpoint is never created by
balancing pressure or choosing a deterministic tie-breaker.

## Active inputs and outputs

Inputs are trace identity, static opcode/control sidecars, masks, dependencies,
addresses, launch resources, architecture-card constants, and a calibration
profile. Outputs are:

- predicted `gpc__cycles_elapsed.max` or declared NCU duration;
- simulator-native frontiers and endpoint descriptor;
- candidate/interval route and cache summaries;
- uncertainty coverage and unsupported/OOD reasons.

The NCU observer projection models the selected cache-control, serialization,
clock-control, and start/end semantics. It does not convert CUDA-event,
CUPTI, device-timer, or replay-wall values into simulator cycles by assumption.

## Validation contract

1. Calibrate resource and cache service parameters on independent probe kernels.
2. Freeze the model before the score cohort.
3. Predict from trace/card inputs before loading score labels.
4. Score by family-held-out NCU rows, reporting log error, MAPE, rank
   correlation, interval coverage, and OOD behavior.
5. Keep exact external closure blocked unless a new supported endpoint supplies
   physical route, cache completion, unrestricted codegen, or synchronized
   completion evidence.

The first implementation seam is the `sim-semantics::latent` resource-demand
contract and its `sim-sm::TraceMetrics` adapter. The detailed plan is
[`../plans/sm86_predictive_closure_implementation_20261001.md`](../plans/sm86_predictive_closure_implementation_20261001.md).

The first predictive slice is implemented: `ResourceDemandEnvelope`,
`ServiceRateProfile`, and `ResourceServiceProjection` validate/project
abstract demand; `TraceMetrics::resource_demand_envelope()` preserves
candidate bounds; the opt-in latent report emits service intervals, cache
candidate summaries, and an NCU observer candidate interval. The checked-in
profile is explicitly an abstract/pending-probe profile; it does not claim
silicon calibration or alter default replay. Independent probe data can be
fit into the same profile format with
`tools/calibrate_sm86_service_profile.py`; the fitter accepts probe service
observations, not score-workload NCU rows.
