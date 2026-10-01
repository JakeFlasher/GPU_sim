# Current status

Updated 2026-10-01. The project develops analytical/statistical GPU performance
models and a runnable native SM86 replay path, with NCU as independent ground
truth for the declared measurement protocol.

## Research scope

Strictly forbidden: pursuing goals or exact proprietary mechanisms beyond normal academic PhD-level research; improve accuracy through analytical or statistical modeling and treat unobservable internals as black boxes.

## Implemented and remaining work

| Surface | Current state |
|---|---|
| Native replay | SM86 instruction, memory, and event replay; separate frontiers |
| Custom frontends | Bounded functional/resource experiments, separate from native timing |
| Predictive extension | Opt-in resource/service intervals, cache-topology candidates, NCU candidate projection |
| Calibration | Profile loader and offline fitter exist; supplied rates/padding are uncalibrated examples |
| Accuracy | This extension has no demonstrated family-held-out cycle accuracy or interval coverage yet |
| Payload/value work | Bounded model-local identity, ordering, and conservation checks |

Native replay is unchanged by the predictive flag. Physical routing, compiler
implementation, and internal cache state are black-box variables, not milestones.
Topology summaries are not causal read-from histories. Endpoint projections
require an explicit statistical/modeling assumption rather than timer equality.

## Next work

Use the [active plan](../plans/sm86_predictive_closure_implementation_20261001.md):
correct unsupported interval assumptions, calibrate on separate probe data, and
score untouched workload families. Report error, coverage, OOD behavior, and
runtime together; no accuracy claim follows from a passing unit suite.

## Retained measurements

2026-09-30 RTX 3090 Ti campaigns used driver 615.71.09, 1350/10501 MHz, P0,
and NCU clock control `none`:

| Cache policy | Workloads | Median NCU duration | Median GPC cycles |
|---|---:|---:|---:|
| `all` | 15/15 | 10,080 ns | 13,572 |
| `none` | 15/15 | 8,704 ns | 11,709 |

These are measurements, not prediction-error results. See [campaign details](archive/research_20260930/sm86_final_performance_handoff_20260930.md).

## Navigation

[Model contract](PREDICTIVE_CLOSURE.md) · [Configuration](archprofile.md) ·
[Runbook](runbook_host.md) · [Checks](testing.md) · [History](HISTORY.md).
