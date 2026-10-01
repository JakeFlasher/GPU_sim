# Continuation memory

Updated 2026-10-01. Start here, then [STATUS](STATUS.md); read only the
contract/source needed for the task. [HISTORY](HISTORY.md) is optional navigation.

## Research scope

Strictly forbidden: pursuing goals or exact proprietary mechanisms beyond normal academic PhD-level research; improve accuracy through analytical or statistical modeling and treat unobservable internals as black boxes.

Objective: fast SM86 performance prediction, evaluated against independent NCU
measurements. Never use a scored workload's NCU values as features, calibration
labels, or implementation-selection criteria. Keep simulator frontiers separate
from profiler measurements; black-box models are assumptions, not observations.

## Current state

- Branch: `GPUSim_main`. Native path: NVBit/SASS → enhanced protobuf →
  `sim-trace` → `sim-sm` / `sim-mem` / `sim-engine` → `sim-run`.
- `simt_u32_v1` and `simt_its_v1` are separate custom/model-relative frontends.
- `--latent-envelope --predictive-service-profile FILE` adds abstract resource
  demand/service intervals, cache-topology summaries, and an NCU candidate
  projection without changing default replay.
- The service profile is an uncalibrated example. Prediction plumbing and unit
  tests do not establish cycle accuracy, empirical coverage, or generalization.
- Independent probe calibration and family-held-out NCU scoring remain work to do.

## Next useful work

Review the [implementation plan](../plans/sm86_predictive_closure_implementation_20261001.md):
check interval/endpoint arithmetic, calibrate on separate probes, then evaluate
on untouched workload families. Do not turn black-box internals into prerequisites.

## Retained evidence

The 2026-09-30 cold/warm campaigns completed 15/15 workloads on RTX 3090 Ti / SM86,
UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09,
1350/10501 MHz, P0. This is historical host identity, not a current-device check.
Raw records: `results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/`.
Read [the campaign summary](archive/research_20260930/sm86_final_performance_handoff_20260930.md)
only when those measurements are needed. Preserve datasets, source, tests, and
scientific limitations; do not recreate removed policy text or workflow machinery.
