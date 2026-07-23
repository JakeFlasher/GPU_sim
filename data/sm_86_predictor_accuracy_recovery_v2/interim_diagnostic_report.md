# Interim Diagnostic Report

**Cohort**: 13-cell HEADLINE_COHORT_13 (9 admitted + 4 excluded under the operator partial-closure override).
**Closure status**: `aborted_unrecoverable_per_user_decision`.
**Variance precondition verdict**: `satisfied` (cross_cell_stddev=0.1475 vs documented_floor_stddev=0.1000).

## Hard Targets

- HARD-1 (friendly_l2 geomean < 0.15): PASS (observed geomean=0.1010).
- HARD-2 (friendly_l1 geomean < 0.20): MISS (observed geomean=1.9470).
- HARD-3 (cross_cell_stddev >= 0.10): PASS (observed stddev=0.1475).

## Per-Bucket Headline (independent L1 / L2 geomeans)

| Bucket-Layer | n_cells | n_scalar | n_typed_unavailable | geomean_rel_err |
|--------------|---------|----------|---------------------|-----------------|
| friendly_l1 | 4 | 3 | 1 | 1.9470 |
| friendly_l2 | 4 | 4 | 0 | 0.1010 |
| hostile_l1 | 4 | 3 | 1 | 0.3172 |
| hostile_l2 | 4 | 1 | 3 | 0.0911 |

## Candidate-Feature Catalog Top Recommendations (quoted)

- `per_warp_occupancy` → `inconclusive`
- `blockDim` → `inconclusive`
- `gridDim` → `predictor_input`

## Per-Cell Residual Decomposition

| slug | shape_point | admission_status | bucket | ncu_truth_l1 | predicted_l1 | rel_err_l1 | l1_status | ncu_truth_l2 | predicted_l2 | rel_err_l2 | l2_status | signed_residual_l2 | plan_2_candidate |
|------|-------------|-----------------|--------|-------------|--------------|------------|-----------|-------------|--------------|------------|-----------|--------------------|------------------|
| carveout_sensitivity_probe | trace_l2_accuracy_xl | admitted | predictor_hostile | 0.0014 | 0.0013 | 0.0801 | scalar_present | 0.9135 | 0.9967 | 0.0911 | scalar_present | 0.0832 | already_meets_target_no_action_needed |
| dram_stream_stride_mix | trace_l2_accuracy_xl | excluded_label_repeatability | predictor_friendly | n/a | n/a | n/a | typed_unavailable_persistent | n/a | n/a | n/a | typed_unavailable_persistent | n/a | typed_unavailable_persistent |
| l2_persisting_window_sweep | trace_l2_accuracy_xl | admitted | predictor_hostile | 0.6565 | 0.9052 | 0.3788 | scalar_present | n/a | n/a | n/a | typed_unavailable_persistent | n/a | predictor_hostile_physics_challenge_plan_2 |
| l2_prefetch_hint_sweep | trace_l2_accuracy_xl | admitted | predictor_hostile | 0.4752 | 0.9750 | 1.0518 | scalar_present | n/a | n/a | n/a | typed_unavailable_persistent | n/a | predictor_hostile_physics_challenge_plan_2 |
| l2_thrash_pointer_chase | trace_l2_accuracy_xl | excluded_label_repeatability | predictor_friendly | n/a | n/a | n/a | typed_unavailable_persistent | n/a | n/a | n/a | typed_unavailable_persistent | n/a | typed_unavailable_persistent |
| layernorm_fwd_bf16_sm80 | timing_shape | excluded_label_repeatability | predictor_friendly | n/a | n/a | n/a | typed_unavailable_persistent | n/a | n/a | n/a | typed_unavailable_persistent | n/a | typed_unavailable_persistent |
| register_spill_local_mem | trace_l2_accuracy_xl | excluded_label_repeatability | predictor_hostile | n/a | n/a | n/a | typed_unavailable_persistent | n/a | n/a | n/a | typed_unavailable_persistent | n/a | typed_unavailable_persistent |
| rmsnorm_fwd_bf16_sm80 | timing_shape | admitted | predictor_friendly | 0.3614 | 1.0000 | 1.7670 | scalar_present | 0.6795 | 0.9685 | 0.4254 | scalar_present | 0.2890 | predictor_friendly_plan_2_accuracy_target |
| silu_fwd_bf16_sm80 | correctness_shape | excluded_label_repeatability | predictor_friendly | n/a | n/a | n/a | typed_unavailable_persistent | n/a | n/a | n/a | typed_unavailable_persistent | n/a | typed_unavailable_persistent |
| silu_fwd_bf16_sm80 | timing_shape | admitted | predictor_friendly | 0.2028 | 1.0000 | 3.9300 | scalar_present | 0.5390 | 0.5407 | 0.0030 | scalar_present | 0.0016 | predictor_friendly_plan_2_accuracy_target |
| softmax_safe_bf16_sm80 | timing_shape | admitted | predictor_friendly | 0.4848 | 1.0000 | 1.0628 | scalar_present | 0.5726 | 0.9583 | 0.6737 | scalar_present | 0.3857 | predictor_friendly_plan_2_accuracy_target |
| warm_cold_pair_relaunch | trace_l2_accuracy_xl | admitted | predictor_hostile | 0.0417 | n/a | n/a | typed_unavailable_persistent | n/a | n/a | n/a | typed_unavailable_persistent | n/a | typed_unavailable_persistent |
| working_set_capacity_sweep | trace_l2_accuracy_xl | admitted | predictor_friendly | 0.0000 | 0.0000 | n/a | typed_unavailable_persistent | 0.8885 | 0.9963 | 0.1214 | scalar_present | 0.1078 | predictor_friendly_plan_2_accuracy_target |

## Operator Partial-Closure Override

- Accepted: True
- Date: 2026-05-18
- Decided admitted count: 9

**Reason (verbatim):**

> Operator accepted the 9-cell admitted cohort as this plan's de-facto admitted state. The plan-pinned >=11 admission floor cannot be reached under the contract-pinned 0.03 stddev repeatability gate because 4 of the 6 candidate cells in the strengthening matrix are predictor_hostile-by-design L2-stressors (dram_stream_stride_mix / l2_persisting_window_sweep / l2_prefetch_hint_sweep / warm_cold_pair_relaunch) whose inherent NCU-replay variance is structurally above the gate. The bottleneck is label-side measurement-repeatability, not a predictor-accuracy gap; the gate is not loosened. Set-associative promotion was rejected as L1-only and not an L2 admission path.
