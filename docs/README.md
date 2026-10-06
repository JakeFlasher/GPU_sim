# Documentation map

Start with [PROJECT_MEMORY](PROJECT_MEMORY.md) and [STATUS](STATUS.md), then
open only the relevant technical page.

M4/C7-S33BB is closed at the scoped 42-directory research-probe bar. The SOL
source census is 52 custom CUDA, 14 CUTLASS, and 137 pure ATen. CUTLASS
closure is 11 live/replayable rows plus three explicit blockers; L1/022 TF32
replay is repaired on SM86, while L1/003 and L1/037 capture and Quant/020
SM120 remain blocked. M5-F through M5-O record bounded observer holdouts,
TF32 capability/replay, physical-tail and async negatives, a multiwarp exact
fallback, and blocker consolidation. M5-N1/N4 synchronized the residual dependencies and found no additional local SM86 seam; next: M5-N3 external-evidence residuals;
physical completion, broad accuracy, and broad native continuation remain open.

| Need | Document |
|---|---|
| Modeling scope and evaluation | [Predictive contract](PREDICTIVE_CLOSURE.md) |
| Next implementation | [Latest-review R8 refinement program](../plans/latest_review_refinement_20261002.md) · [Second-round history](../plans/second_round_refinement_20261001.md) · [first-round history](../plans/sm86_reification_refactor_20261001.md) |
| Build, trace, replay | [Runbook](runbook_host.md) |
| Focused checks | [Testing](testing.md) |
| Review history | [Review/refactor history](REVIEW_HISTORY.md) |
| Architecture and source map | [Architecture](ARCHITECTURE.md), [components](COMPONENTS.md) |
| Cards and predictive profiles | [Configuration](archprofile.md) |
| Prediction/observation schemas | [`reification_prediction_manifest_v1`](reification_prediction_manifest_v1.schema.json), [`reification_prediction_v2`](reification_prediction_v2.schema.json), [`ncu_observation_v2`](ncu_observation_v2.schema.json), [`reification_prediction_evaluation_v2`](reification_prediction_evaluation_v2.schema.json), [`reification_native_macro_witness_v1`](reification_native_macro_witness_v1.schema.json) |
| Accuracy-first NCU protocol | [Declared SM86 NCU accuracy protocol](sm86_ncu_accuracy_protocol_v1.json) · [Protocol schema](sm86_ncu_accuracy_protocol_v1.schema.json) |
| A8 live-campaign plan | [Accuracy-first A8 plan](../plans/accuracy_first_live_campaign_a8_20261002.md) · [bounded-run schema](sm86_workload_run_v1.schema.json) |
| Completion frontier program | [Writeback/coherence and native-dispatch plan](../plans/sm86_completion_frontier_and_native_dispatch_20261002.md) |
| Completion frontier contract | `sm86_completion_frontier_contract_v1.json` · `tools/report_sm86_completion_frontier.py` |
| Completion frontier C1 live evidence | `sm86_completion_frontier_c1_small_summary_v1.json` · `sm86_completion_frontier_c1_visibility_summary_v1.json` |
| Completion frontier C3 probe | `sm86_completion_frontier_c3_service_probe_v1.json` |
| Completion frontier C3 holdout | `sm86_completion_frontier_c3_store2_holdout_v1.json` |
| M4 research-probe population | `sm86_c7_research_probe_inventory_v1.json` |
| M4-E closeout recheck | `sm86_c7_research_probe_sfu_frontier_live_v2.json` plus regenerated barrier/L1TEX matrix manifests |
| M5 scoped continuation | `sm86_m5a_native_tail_differential_launch_cycle_v1.json` · `sm86_m5b_calibration_holdout_reconciliation_v1.json` · `sm86_m5c_native_continuation_fallback_v1.json` · `sm86_m5d_claim_synchronization_v1.json` |
| C8 interface/failure/runbook reconciliation | `sm86_c8a_interface_reconciliation_v1.json` · `sm86_c8b_corrected_failure_dispositions_v1.json` · `sm86_c8c_runbook_synchronization_v1.json` |
| Tail witness report | `sm86_tail_witness_report_v1.schema.json` · `tools/report_sm86_tail_witness.py` |
| C1-R matched live negatives | `sm86_completion_frontier_c1r_async_matched_v1.json` · `sm86_completion_frontier_c1r_message_device_v1.json` |
| C1-R store/fence differential | `sm86_c1r_store_fence_differential_live_v1.json` · `sm86_c1r_store_fence_scope_reconciliation_live_v1.json` |
| C3-A completion obligations | `sm86_c3a_completion_obligation_reconciliation_v1.json` |
| C3-B/C completion and tail facts | `sm86_c3b_completion_feedback_live_perturbation_v1.json` · `sm86_c3c_store_fence_tail_facts_v1.json` |
| C3-D async visibility tails | `sm86_c3d_async_visibility_tail_reconciliation_v1.json` |
| C3-E async continuation | `sm86_c3e_async_reference_continuation_v1.json` |
| C4/C5 observer holdout | `sm86_async_observer_calibration_holdout_v1.json` |
| C4-A observer contract | `sm86_c4a_observer_contract_reconciliation_v1.json` |
| C4-B SFU observer holdout | `sm86_c4b_sfu_warp_observer_reconciliation_v1.json` |
| C4-C SFU OOD transfer | `sm86_c4c_sfu_ood_observer_reconciliation_v1.json` |
| C4-D quad-family observer | `sm86_c4d_sfu_quad_observer_reconciliation_v1.json` |
| C4-E quad occupancy intervention | `sm86_c4e_sfu_quad_occupancy_reconciliation_v1.json` |
| C4-F quad batch-feature observer | `sm86_c4f_sfu_quad_wave_reconciliation_v1.json` |
| C4-G observer domain boundary | `sm86_c4g_observer_domain_boundary_v1.json` |
| C6-A native continuation bound | `sm86_c6a_native_continuation_bound_probe_v1.json` |
| C6-B native fallback | `sm86_c6b_native_continuation_fallback_v1.json` |
| C6-C positive scalar dispatch | `sm86_c6c_positive_real_scalar_dispatch_v1.json` |
| C6-D default front-end refusal | `sm86_c6d_default_frontend_refusal_v1.json` |
| C6-D positive default front-end dispatch | `sm86_c6d_positive_default_frontend_dispatch_v1.json` |
| C6-E front-end snapshot transition | `sm86_c6e_frontend_snapshot_transition_v1.json` |
| C6-F multi-warp boundary | `sm86_c6f_multiwarp_frontend_boundary_v1.json` |
| C6-G multiwarp domain | `sm86_c6g_multiwarp_domain_boundary_v1.json` |
| C6-H store/fence completion boundary | `sm86_c6h_store_fence_completion_boundary_v1.json` |
| C3-G/C6-I opt-in tail audit | `sm86_c3g_optin_writeback_coherence_tail_audit_v1.json` |
| C3-H/C6-J visibility reference boundary | `sm86_c3h_visibility_reference_boundary_v1.json` |
| C3-I/C6-K owner-generation negative | `sm86_c3i_owner_generation_negative_v1.json` |
| C4-H constant-load prediction | `sm86_c4h_constant_load_prediction_pre_ncu_v1.json` |
| C4-H constant-load reconciliation | `sm86_c4h_constant_load_reconciliation_v1.json` |
| C4-I observer domain reconciliation | `sm86_c4i_observer_domain_reconciliation_v1.json` |
| C5-A constant-load holdout evaluation | `sm86_c5a_constant_load_holdout_evaluation_v1.json` |
| C5-B mode-1 prediction | `sm86_c5b_constant_load_mode1_prediction_pre_ncu_v1.json` |
| C5-B mode-1 reconciliation | `sm86_c5b_constant_load_mode1_reconciliation_v1.json` |
| C5-C cross-family boundary | `sm86_c5c_cross_family_boundary_v1.json` |
| C7-S33BD inventory/build synchronization | `sm86_c7_research_probe_inventory_sync_v1.json` |
| C7-S34A SOL source inventory | `sm86_c7_sol_population_source_inventory_v1.json` |
| C7-S34B SOL candidate blocker | `sm86_c7_sol_candidate_blocked_v1.json` |
| C7-S34B SOL candidate smoke | `sm86_c7_sol_candidate_smoke_v1.json` |
| C7-S34B SOL candidate vertical | `sm86_c7_sol_candidate_vertical_v1.json` |
| C7-S34D FIB002 vertical | `sm86_c7_sol_candidate_fib002_vertical_v1.json` |
| C7-S34F FIB003 vertical | `sm86_c7_sol_candidate_fib003_vertical_v1.json` |
| C7-S34J FIB021 vertical | `sm86_c7_sol_candidate_fib021_vertical_v1.json` |
| C7-S34K FIB022 vertical | `sm86_c7_sol_candidate_fib022_vertical_v1.json` |
| C7-S34L FIB023 vertical | `sm86_c7_sol_candidate_fib023_vertical_v1.json` |
| C7-S34M1 FIB024 source review | `sm86_c7_sol_candidate_fib024_source_v1.json` |
| C7-S34M2 FIB024 build/smoke | `sm86_c7_sol_candidate_fib024_smoke_v1.json` |
| C7-S34M FIB024 vertical | `sm86_c7_sol_candidate_fib024_vertical_v1.json` |
| C7-S34N1 FIB025 source review | `sm86_c7_sol_candidate_fib025_source_v1.json` |
| C7-S34N2 FIB025 build/smoke | `sm86_c7_sol_candidate_fib025_smoke_v1.json` |
| C7-S34N FIB025 vertical | `sm86_c7_sol_candidate_fib025_vertical_v1.json` |
| C7-S34O1 FIB026 source review | `sm86_c7_sol_candidate_fib026_source_v1.json` |
| C7-S34O2 FIB026 build/smoke | `sm86_c7_sol_candidate_fib026_smoke_v1.json` |
| C7-S34O FIB026 vertical | `sm86_c7_sol_candidate_fib026_vertical_v1.json` |
| C7-S34P L1 problem002 source-class correction | `sm86_c7_sol_l1_problem002_source_class_correction_v1.json` |
| C7-S34Q1 L1 problem006 source review | `sm86_c7_sol_l1p006_source_v1.json` |
| C7-S34Q2 L1 problem006 build/smoke | `sm86_c7_sol_l1p006_smoke_v1.json` |
| C7-S34Q L1 problem006 vertical | `sm86_c7_sol_l1p006_vertical_v1.json` |
| C7-S34R1 L1 problem011 source review | `sm86_c7_sol_l1p011_source_v1.json` |
| C7-S34R2 L1 problem011 build/smoke | `sm86_c7_sol_l1p011_smoke_v1.json` |
| C7-S34R L1 problem011 vertical | `sm86_c7_sol_l1p011_vertical_v1.json` |
| C7-S34S1 L1 problem012 source review | `sm86_c7_sol_l1p012_source_v1.json` |
| C7-S34S2 L1 problem012 build/smoke | `sm86_c7_sol_l1p012_smoke_v1.json` |
| C7-S34S L1 problem012 vertical | `sm86_c7_sol_l1p012_vertical_v1.json` |
| C7-S34T1 L1 problem013 source review | `sm86_c7_sol_l1p013_source_v1.json` |
| C7-S34T2 L1 problem013 build/smoke | `sm86_c7_sol_l1p013_smoke_v1.json` |
| C7-S34T L1 problem013 vertical | `sm86_c7_sol_l1p013_vertical_v1.json` |
| C7-S34U1 L1 problem016 source review | `sm86_c7_sol_l1p016_source_v1.json` |
| C7-S34U2 L1 problem016 build/smoke | `sm86_c7_sol_l1p016_smoke_v1.json` |
| C7-S34U L1 problem016 vertical | `sm86_c7_sol_l1p016_vertical_v1.json` |
| C7-S34V1 L1 problem018 source review | `sm86_c7_sol_l1p018_source_v1.json` |
| C7-S34V2 L1 problem018 build/smoke | `sm86_c7_sol_l1p018_smoke_v1.json` |
| C7-S34V L1 problem018 vertical | `sm86_c7_sol_l1p018_vertical_v1.json` |
| C7-S34W1 L1 problem019 source review | `sm86_c7_sol_l1p019_source_v1.json` |
| C7-S34W2 L1 problem019 build/smoke | `sm86_c7_sol_l1p019_smoke_v1.json` |
| C7-S34W L1 problem019 vertical | `sm86_c7_sol_l1p019_vertical_v1.json` |
| C7-S34X1 L1 problem022 source review | `sm86_c7_sol_l1p022_source_v1.json` |
| C7-S34X2 L1 problem022 build/smoke blocker | `sm86_c7_sol_l1p022_smoke_v1.json` |
| C7-S34Y L1 problem024 source-class correction | `sm86_c7_sol_l1_problem024_source_class_correction_v1.json` |
| C7-S34Z1 L1 problem025 source review | `sm86_c7_sol_l1p025_source_v1.json` |
| C7-S34Z2 L1 problem025 build/smoke | `sm86_c7_sol_l1p025_smoke_v1.json` |
| C7-S34Z L1 problem025 vertical | `sm86_c7_sol_l1p025_vertical_v1.json` |
| C7-S34AA L1 problem029 source-class correction | `sm86_c7_sol_l1_problem029_source_class_correction_v1.json` |
| C7-S34AB1 L1 problem033 source review | `sm86_c7_sol_l1p033_source_v1.json` |
| C7-S34AB2 L1 problem033 build/smoke | `sm86_c7_sol_l1p033_smoke_v1.json` |
| C7-S34AB L1 problem033 vertical | `sm86_c7_sol_l1p033_vertical_v1.json` |
| C7-S34AC2 L1 problem034 build/smoke | `sm86_c7_sol_l1p034_smoke_v1.json` |
| C7-S34AC L1 problem034 vertical | `sm86_c7_sol_l1p034_vertical_v1.json` |
| C7-S34AD1 L1 problem035 source review | `sm86_c7_sol_l1p035_source_v1.json` |
| C7-S34AD2 L1 problem035 build/smoke | `sm86_c7_sol_l1p035_smoke_v1.json` |
| C7-S34AD L1 problem035 vertical | `sm86_c7_sol_l1p035_vertical_v1.json` |
| C7-S34AE1 L1 problem036 source review | `sm86_c7_sol_l1p036_source_v1.json` |
| C7-S34AE2 L1 problem036 correctness blocker | `sm86_c7_sol_l1p036_smoke_v1.json` |
| C7-S34AF1 L1 problem038 source review | `sm86_c7_sol_l1p038_source_v1.json` |
| C7-S34AF2 L1 problem038 build/smoke | `sm86_c7_sol_l1p038_smoke_v1.json` |
| C7-S34AF L1 problem038 vertical | `sm86_c7_sol_l1p038_vertical_v1.json` |
| C7-S34AH1 L1 problem040 source review | `sm86_c7_sol_l1p040_source_v1.json` |
| C7-S34AG L1 problem039 source-class correction | `sm86_c7_sol_l1_problem039_source_class_correction_v1.json` |
| C7-S34AH1 L1 problem040 source review | `sm86_c7_sol_l1p040_source_v1.json` |
| C7-S34AH2 L1 problem040 build/smoke | `sm86_c7_sol_l1p040_smoke_v1.json` |
| C7-S34AH L1 problem040 vertical | `sm86_c7_sol_l1p040_vertical_v1.json` |
| C7-S34AI L1 problem041 source-class correction | `sm86_c7_sol_l1_problem041_source_class_correction_v1.json` |
| C7-S34AJ L1 problem042 source-class correction | `sm86_c7_sol_l1_problem042_source_class_correction_v1.json` |
| C7-S34AK L1 problem043 source-class correction | `sm86_c7_sol_l1_problem043_source_class_correction_v1.json` |
| C7-S34AL L1 problem046 custom-CUDA vertical | `sm86_c7_sol_l1p046_source_v1.json`, `sm86_c7_sol_l1p046_smoke_v1.json`, `sm86_c7_sol_l1p046_vertical_v1.json` |
| C7-S34AM L1 problem048 custom-CUDA-plus-cuBLAS row | `sm86_c7_sol_l1p048_source_v1.json`, `sm86_c7_sol_l1p048_smoke_v1.json`, `sm86_c7_sol_l1p048_capture_blocked_v1.json` |
| C7-S34AN L1 problem052 source-class correction | `sm86_c7_sol_l1_problem052_source_class_correction_v1.json` |
| C7-S34AO L1 problem053 custom-CUDA vertical | `sm86_c7_sol_l1p053_source_v1.json`, `sm86_c7_sol_l1p053_smoke_v1.json`, `sm86_c7_sol_l1p053_vertical_v1.json` |
| C7-S34AP L1 problem056 custom-CUDA vertical | `sm86_c7_sol_l1p056_source_v1.json`, `sm86_c7_sol_l1p056_smoke_v1.json`, `sm86_c7_sol_l1p056_vertical_v1.json` |
| C7-S34AQ L1 problem057 source-class correction | `sm86_c7_sol_l1_problem057_source_class_correction_v1.json` |
| C7-S34AR L1 problem058 source-class correction | `sm86_c7_sol_l1_problem058_source_class_correction_v1.json` |
| C7-S34AS L1 problem059 custom-CUDA vertical | `sm86_c7_sol_l1p059_source_v1.json`, `sm86_c7_sol_l1p059_smoke_v1.json`, `sm86_c7_sol_l1p059_vertical_v1.json` |
| C7-S34AT L1 problem061 custom-CUDA vertical | `sm86_c7_sol_l1p061_source_v1.json`, `sm86_c7_sol_l1p061_smoke_v1.json`, `sm86_c7_sol_l1p061_vertical_v1.json` |
| C7-S34AU L1 problem062 source-class correction | `sm86_c7_sol_l1_problem062_source_class_correction_v1.json` |
| C7-S34AV L1 problem064 source-class correction | `sm86_c7_sol_l1_problem064_source_class_correction_v1.json` |
| C7-S34AW L1 problem066 correctness blocker | `sm86_c7_sol_l1p066_source_v1.json`, `sm86_c7_sol_l1p066_smoke_v1.json`, `sm86_c7_sol_l1p066_correctness_blocked_v1.json` |
| C7-S34BB problem066 official criterion audit | `sm86_c7_sol_l1p066_correctness_recheck_v1.json`; prior raw diagnostic superseded |
| C7-S34BB problem066 corrected native vertical | `sm86_c7_sol_l1p066_vertical_v1.json` |
| C7-S34BC remaining population/source-execution seam | plan milestone; active |
| C7-S34CD L2 problem066 source review | `sm86_c7_sol_l2p066_source_v1.json` |
| C7-S34CD L2 problem066 build/smoke | `sm86_c7_sol_l2p066_smoke_v1.json` |
| C7-S34CD L2 problem066 capture blocker | `sm86_c7_sol_l2p066_capture_blocked_v1.json` |
| C7-S34CE Quant problem014 source review | `sm86_c7_sol_quantp014_source_v1.json` |
| C7-S34CE Quant problem014 build/smoke | `sm86_c7_sol_quantp014_smoke_v1.json` |
| C7-S34CE Quant problem014 native vertical | `sm86_c7_sol_quantp014_vertical_v1.json` |
| C7-S34CF Quant problem020 architecture blocker | `sm86_c7_sol_quantp020_source_v1.json`, `sm86_c7_sol_quantp020_architecture_blocked_v1.json` |
| C7-S34CG FIB problem005 CUTLASS source review | `sm86_c7_sol_fibp005_source_v1.json` |
| C7-S34CG FIB problem005 CUTLASS build/smoke | `sm86_c7_sol_fibp005_smoke_v1.json` |
| C7-S34CG FIB problem005 CUTLASS native vertical | `sm86_c7_sol_fibp005_vertical_v1.json` |
| C7-S34CH FIB problem006 CUTLASS source review | `sm86_c7_sol_fibp006_source_v1.json` |
| C7-S34CH FIB problem006 CUTLASS build/smoke | `sm86_c7_sol_fibp006_smoke_v1.json` |
| C7-S34CH FIB problem006 declared-row flat fallback | `sm86_c7_sol_fibp006_row0_vertical_v1.json`, `sm86_c7_sol_fibp006_capture_blocked_v1.json` |
| C7-S34CI FIB problem007 CUTLASS source review | `sm86_c7_sol_fibp007_source_v1.json` |
| C7-S34CI FIB problem007 CUTLASS build/smoke | `sm86_c7_sol_fibp007_smoke_v1.json` |
| C7-S34CI FIB problem007 CUTLASS native vertical | `sm86_c7_sol_fibp007_vertical_v1.json` |
| C7-S34CJ FIB problem008 CUTLASS source review | `sm86_c7_sol_fibp008_source_v1.json` |
| C7-S34CJ FIB problem008 CUTLASS build/smoke | `sm86_c7_sol_fibp008_smoke_v1.json` |
| C7-S34CJ FIB problem008 CUTLASS native vertical | `sm86_c7_sol_fibp008_vertical_v1.json` |
| C7-S34CH post-fix FIB problem006 L1-L2 vertical | `sm86_c7_sol_fibp006_row0_l1_l2_vertical_v2.json` |
| C7-S34CI post-fix FIB problem007 L1-L2 vertical | `sm86_c7_sol_fibp007_l1_l2_vertical_v2.json` |
| C7-S34CJ post-fix FIB problem008 L1-L2 vertical | `sm86_c7_sol_fibp008_l1_l2_vertical_v2.json` |
| C7-S34CK FIB problem009 source review | `sm86_c7_sol_fibp009_source_v1.json` |
| C7-S34CK FIB problem009 build/smoke | `sm86_c7_sol_fibp009_smoke_v1.json` |
| C7-S34CK post-fix FIB problem009 L1-L2 vertical | `sm86_c7_sol_fibp009_l1_l2_vertical_v2.json` |
| C7-S34CL FIB problem010 CUTLASS source review | `sm86_c7_sol_fibp010_source_v1.json` |
| C7-S34CL FIB problem010 CUTLASS build/smoke | `sm86_c7_sol_fibp010_smoke_v1.json` |
| C7-S34CL FIB problem010 CUTLASS native vertical | `sm86_c7_sol_fibp010_vertical_v1.json` |
| C7-S34CM FIB problem011 CUTLASS source review | `sm86_c7_sol_fibp011_source_v1.json` |
| C7-S34CM FIB problem011 CUTLASS build/smoke | `sm86_c7_sol_fibp011_smoke_v1.json` |
| C7-S34CM FIB problem011 CUTLASS native vertical | `sm86_c7_sol_fibp011_vertical_v1.json` |
| C7-S34CN L1 problem003 CUTLASS source review | `sm86_c7_sol_l1p003_source_v1.json` |
| C7-S34CN L1 problem003 CUTLASS build/smoke | `sm86_c7_sol_l1p003_smoke_v1.json` |
| C7-S34CN L1 problem003 capture blocker | `sm86_c7_sol_l1p003_capture_blocked_v1.json` |
| C7-S34CO L1 problem022 CUTLASS source review | `sm86_c7_sol_l1p022_source_v1.json` |
| C7-S34CO L1 problem022 CUTLASS build/smoke | `sm86_c7_sol_l1p022_smoke_v1.json` |
| C7-S34CO L1 problem022 TF32 replay/capability | `sm86_m5g_tf32_pipeline_replay_v1.json` · historical blocker `sm86_c7_sol_l1p022_capture_blocked_v1.json` |
| C7-S34CP L1 problem030 CUTLASS source review | `sm86_c7_sol_l1p030_source_v1.json` |
| C7-S34CP L1 problem030 CUTLASS build/smoke | `sm86_c7_sol_l1p030_smoke_v1.json` |
| C7-S34CP L1 problem030 CUTLASS native vertical | `sm86_c7_sol_l1p030_vertical_v1.json` |
| C7-S34CQ L1 problem037 CUTLASS source review | `sm86_c7_sol_l1p037_source_v1.json` |
| C7-S34CQ L1 problem037 CUTLASS build/smoke | `sm86_c7_sol_l1p037_smoke_v1.json` |
| C7-S34CQ L1 problem037 capture blocker | `sm86_c7_sol_l1p037_capture_blocked_v1.json` |
| C7-S34CR L1 problem077 CUTLASS source review | `sm86_c7_sol_l1p077_source_v1.json` |
| C7-S34CR L1 problem077 CUTLASS build/smoke | `sm86_c7_sol_l1p077_smoke_v1.json` |
| C7-S34CR L1 problem077 CUTLASS native vertical | `sm86_c7_sol_l1p077_vertical_v1.json` |
| C7-S34CS CUTLASS/source-population closure audit | `sm86_c7_sol_cutlass_population_closure_v1.json` |
| M5-E retained embedding continuation reconciliation | `sm86_m5e_completion_continuation_reconciliation_v1.json` |
| M5-F streaming async observer | `sm86_m5f1_async_dram_prediction_pre_ncu_v1.json` · [result](../results/alpha/m5f_async_dram_20261006/observer_result.json) |
| M5-G TF32 execution-pipeline capability | `sm86_m5g1_tf32_pipeline_probe_v1.json` · `sm86_m5g_tf32_pipeline_replay_v1.json` |
| M5-H store-tail differential | `sm86_m5h_store_tail_differential_v1.json` |
| M5-I async visibility/ownership boundary | `sm86_m5i_async_visibility_ownership_boundary_v1.json` |
| M5-J multiwarp continuation boundary | `sm86_m5j_multiwarp_continuation_boundary_v1.json` |
| M5-K FIB005 M80 holdout | `sm86_m5k_fibp005_m80_prediction_pre_ncu_v1.json` · `sm86_m5k_fibp005_m80_holdout_v1.json` |
| M5-L frontier/retry reconciliation | `sm86_m5l_frontier_reconciliation_v1.json` · `sm86_m5l_c7_s34cq_retry_v1.json` |
| M5-M blocker consolidation | `sm86_m5m_blocker_consolidation_v1.json` |
| M5-O exact INT8 IMMA capability | `sm86_m5o_imma16832_pipeline_replay_v1.json` |
| M5-N external residual audit | `sm86_m5n_external_residual_audit_v1.json` |
| M5-N3 external-evidence residuals | active plan milestone |
| C7-S34BC1 problem075 source-class correction | `sm86_c7_sol_l1_problem075_source_class_correction_v1.json` |
| C7-S34BD L1 problem078 custom-CUDA vertical | `sm86_c7_sol_l1p078_source_v1.json`, `sm86_c7_sol_l1p078_smoke_v1.json`, `sm86_c7_sol_l1p078_vertical_v1.json` |
| C7-S34BE L1 problem080 custom-CUDA vertical | `sm86_c7_sol_l1p080_source_v1.json`, `sm86_c7_sol_l1p080_smoke_v1.json`, `sm86_c7_sol_l1p080_vertical_v1.json` |
| C7-S34BF1 problem081 source-class correction | `sm86_c7_sol_l1_problem081_source_class_correction_v1.json` |
| C7-S34BG1 problem082 source-class correction | `sm86_c7_sol_l1_problem082_source_class_correction_v1.json` |
| C7-S34BH next L1 custom-CUDA vertical | plan milestone; evidence pending |
| C7-S34AX L1 problem068 custom-CUDA vertical | `sm86_c7_sol_l1p068_source_v1.json`, `sm86_c7_sol_l1p068_smoke_v1.json`, `sm86_c7_sol_l1p068_vertical_v1.json` |
| C7-S34AY L1 problem069 custom-CUDA vertical | `sm86_c7_sol_l1p069_source_v1.json`, `sm86_c7_sol_l1p069_smoke_v1.json`, `sm86_c7_sol_l1p069_vertical_v1.json` |
| C7-S34AZ L1 problem071 custom-CUDA vertical | `sm86_c7_sol_l1p071_source_v1.json`, `sm86_c7_sol_l1p071_smoke_v1.json`, `sm86_c7_sol_l1p071_vertical_v1.json` |
| C7-S34BA L1 problem073 mixed custom/ATen row | `sm86_c7_sol_l1p073_source_v1.json`, `sm86_c7_sol_l1p073_smoke_v1.json`, `sm86_c7_sol_l1p073_capture_blocked_v1.json` |
| C7-S34BB correctness-contract reconciliation | plan milestone; problem066 criterion audit active |
| C7-S34E L1 pure-ATen blocker | `sm86_c7_sol_l1_candidate_blocked_v1.json` |
| C7-S34G L2 pure-ATen blocker | `sm86_c7_sol_l2_pure_aten_source_blocked_v1.json` |
| C7-S34H CUTLASS replay blocker | `sm86_c7_sol_candidate_fib004_replay_blocked_v1.json` |
| C7-S34H CUTLASS flat fallback | `sm86_c7_sol_candidate_fib004_flat_fallback_v1.json` |
| C7-S34I SOL source class census | `sm86_c7_sol_source_class_census_v1.json` |
| NCU identity capture | [Identity schema](sm86_ncu_identity_v1.schema.json) · `tools/capture_sm86_ncu_identity.py` |
| NCU campaign preflight | [Preflight schema](sm86_ncu_campaign_preflight_v1.schema.json) · `tools/preflight_sm86_ncu_campaign.py` |
| First accuracy campaign | `sm86_accuracy_campaign_v1_{manifest,predictions_pre_ncu,observations_post_ncu,evaluation_join,repetition_summary}.json` |
| Intervention contrasts | `sm86_ncu_intervention_plan_v1.json` · `sm86_accuracy_intervention_comparison_v1.json` |
| Real selective-path result | `sm86_accuracy_a7_real_selective_result_v1.json` · [schema](sm86_accuracy_a7_real_selective_result_v1.schema.json) |
| A8 vertical live slice | [summary](sm86_a8_vertical_slice_v1.json) · [prediction](sm86_a8_vertical_predictions_pre_ncu_v1.json) · [observation](sm86_a8_vertical_observations_post_ncu_v1.json) · [evaluation](sm86_a8_vertical_evaluation_v1.json) |
| Third-round result | [R6 corpus and observer evaluation](review_round3_20261002/README.md) · [corpus manifest](sm86_reification_corpus_v1_20261002.json) |
| R9.0 native boundary probe | [Canary termination and unknown-join result](review_round4_20261002/README.md) |
| R9.1 production witness attempt | [Negative native-dispatch result](review_round5_20261002/README.md) |
| Live SM86 development probe | [RTX 3090 Ti small-suite result](review_round6_20261002/README.md) |
| Live SM86 large campaign | [168-block RTX 3090 Ti result](review_round7_20261002/README.md) |
| Live SM86 writeback campaign | [Large dirty-stream result](review_round8_20261002/README.md) |
| Live cache-policy contrast | [All-cache versus none-cache result](review_round9_20261002/README.md) |
| Full retained-trace cost campaign | [Detailed versus analyzer utility result](review_round10_20261002/README.md) |
| Latent and payload APIs | [Contract](sm86_factorized_latent_contract_20260930.md) |
| Experimental semantics | [Diagnostics](experimental_diagnostics.md) |
| Prior measurements | preserved under results/ and experiment inputs |

`AGENTS.md` governs execution. Experiment outputs are data, not queued work.
SOL/campaign source, datasets, environments, schemas, and knowledge cards remain
in place; this index does not require loading them for unrelated development.
