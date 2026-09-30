# Native CUDA-event endpoint ratio (2026-09-29)

## Model

The warm/cache-none campaign exposes a native CUDA-event mean for every row.
`tools/sm86_native_endpoint_ratio.py` fits `log(NCU/native_event)` on one
family and applies it to the other family in the same campaign, separately for
duration and GPC. The native event is an observable endpoint feature, not an
NCU feature and not a physical clock-equivalence witness.

This follows the endpoint-separation principle already used by the project’s
CUDA-event→NCU transform. It is also motivated by frequency-scaling work that
uses fast runtime observations and component-specific response instead of
forcing a cycle simulator to discover every hidden clock/cache mechanism.

## Warm live result

Campaign: `results/alpha/full_3090ti_20260929_stage_clock_sampled_warm_live/`.
Calibration and score families are disjoint within the same clock/cache regime.

| scope | GPC MAPE | duration MAPE |
|---|---:|---:|
| large holdout | 11.287% | 12.632% |
| base holdout | 9.620% | 10.889% |
| combined 15 rows | **10.398%** | **11.702%** |

This is substantially below the frozen warm duration projection on the same
high-clock campaign, but it remains a related-family endpoint crosswalk. The
reports are under `results/alpha/sm86_native_endpoint_ratio_goal_20260929/`.

## Boundary mapping

* It does not infer physical pipes or issue slots.
* It does not infer ABI/control-word semantics.
* It does not identify cache ownership/read-from state.
* It keeps native-event and NCU endpoint semantics distinct.

The replay and promoted defaults are unchanged. No tests, hashing, or
verification workflow was run.
