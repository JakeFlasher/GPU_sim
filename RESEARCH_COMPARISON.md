# Exhaustive source comparison (allowed simulator scope)

- Source root: `/home/jakeshea/5090_branch` (read-only; no source files were modified).
- Compared commits: A `0c8791f5d2b87c45d409f23ce6f0ad67b708e50b`, B `818a1dbd7583c273ee1996caa75a18c3a4ac90a5`, common base `f3ca5b2f0d3b12ec9c720df408de7908810d8617`.
- Excluded from merge inputs: `paper/`, manuscript/thesis writing, virtualenvs, and generated build products. Those paths were not read for architectural decisions.

## Tracked path comparison (file by file)

After exclusions: 8,481 paths; A-only=691, different=42, same=7,748.

| Top-level | same | different | A-only | B-only |
|---|---:|---:|---:|---:|
| `"references` | 1 | 0 | 0 | 0 |
| `.gitattributes` | 1 | 0 | 0 | 0 |
| `.gitignore` | 1 | 0 | 0 | 0 |
| `.python-version` | 1 | 0 | 0 | 0 |
| `AGENTS.md` | 0 | 1 | 0 | 0 |
| `README.md` | 0 | 1 | 0 | 0 |
| `arch_profiles` | 41 | 3 | 2 | 0 |
| `data` | 2 | 0 | 0 | 0 |
| `docs` | 8 | 5 | 99 | 0 |
| `eval` | 5 | 0 | 0 | 0 |
| `gpgpu_inst_stats.txt` | 1 | 0 | 0 | 0 |
| `perf_counter.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_02-54-27.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_03-27-19.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_03-56-48.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-04-34.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-04-35.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-07-10.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-28-26.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-28-38.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-28-39.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_04-28-40.csv.gz` | 1 | 0 | 0 | 0 |
| `perf_counter_2026-09-28_05-04-22.csv.gz` | 1 | 0 | 0 | 0 |
| `plans` | 1 | 1 | 1 | 0 |
| `references` | 3933 | 0 | 0 | 0 |
| `requirements.txt` | 1 | 0 | 0 | 0 |
| `rust-toolchain.toml` | 1 | 0 | 0 | 0 |
| `sim` | 74 | 27 | 0 | 0 |
| `third_party` | 1691 | 0 | 0 | 0 |
| `tools` | 303 | 4 | 0 | 0 |
| `workloads` | 1671 | 0 | 589 | 0 |

### Every non-identical tracked path

| Status | Path |
|---|---|
| different | `AGENTS.md` |
| different | `README.md` |
| different | `arch_profiles/build_arch_families.json` |
| A-only | `arch_profiles/candidates/sm86_store_fence_candidate_v1.json` |
| A-only | `arch_profiles/candidates/sm86_store_fence_xbar_drain_candidate_v1.json` |
| different | `arch_profiles/cards/sm_120-rtx-5090.json` |
| different | `arch_profiles/loader.py` |
| different | `docs/PROJECT_MEMORY.md` |
| different | `docs/README.md` |
| different | `docs/STATUS.md` |
| different | `docs/runbook_host.md` |
| different | `docs/sm120_replay_bench.md` |
| A-only | `docs/sm86_accuracy_comparison_ordered_fallback_20261016.md` |
| A-only | `docs/sm86_accuracy_comparison_report_20261016.md` |
| A-only | `docs/sm86_accuracy_dashboard_20261016.md` |
| A-only | `docs/sm86_atomic_conflict_slice_20261015.md` |
| A-only | `docs/sm86_atomic_service_slice_20261015.md` |
| A-only | `docs/sm86_barrier_rearm_slice_20261015.md` |
| A-only | `docs/sm86_branch_redirect_slice_20261015.md` |
| A-only | `docs/sm86_constant_client_interval_slice_20261016.md` |
| A-only | `docs/sm86_constant_demand_slice_20261015.md` |
| A-only | `docs/sm86_constant_dependency_probe_slice_20261015.md` |
| A-only | `docs/sm86_constant_form_ncu_probe_slice_20261016.md` |
| A-only | `docs/sm86_constant_form_observer_slice_20261016.md` |
| A-only | `docs/sm86_constant_form_probe_slice_20261016.md` |
| A-only | `docs/sm86_constant_form_residual_ledger_slice_20261016.md` |
| A-only | `docs/sm86_constant_form_sweep_slice_20261016.md` |
| A-only | `docs/sm86_constant_pressure_slice_20261015.md` |
| A-only | `docs/sm86_constant_reuse_observer_slice_20261016.md` |
| A-only | `docs/sm86_critical_frontier_slice_20261015.md` |
| A-only | `docs/sm86_dram_demand_completion_negative_20261016.md` |
| A-only | `docs/sm86_dram_demand_envelope_slice_20261015.md` |
| A-only | `docs/sm86_dram_demand_interval_slice_20261015.md` |
| A-only | `docs/sm86_dram_dirty_tail_ledger_slice_20261015.md` |
| A-only | `docs/sm86_dram_partition_pressure_slice_20261015.md` |
| A-only | `docs/sm86_dram_residual_ledger_slice_20261015.md` |
| A-only | `docs/sm86_dram_route_concentration_diagnostic_slice_20261016.md` |
| A-only | `docs/sm86_dram_service_rate_slice_20261015.md` |
| A-only | `docs/sm86_dram_sota_queue_prior_slice_20261016.md` |
| A-only | `docs/sm86_dynamic_cbank_tracer_slice_20261016.md` |
| A-only | `docs/sm86_exit_drain_slice_20261016.md` |
| A-only | `docs/sm86_external_consult_prompt_20261016.txt` |
| A-only | `docs/sm86_fetch_yield_slice_20261015.md` |
| A-only | `docs/sm86_form_qualified_lookup_slice_20261016.md` |
| A-only | `docs/sm86_full_replay_slice_20261015.md` |
| A-only | `docs/sm86_gcc_demand_matrix_slice_20261016.md` |
| A-only | `docs/sm86_gcc_demand_observer_slice_20261016.md` |
| A-only | `docs/sm86_gcc_scope_slice_20261015.md` |
| A-only | `docs/sm86_gcc_tex_crosswalk_slice_20261016.md` |
| A-only | `docs/sm86_gpc_envelope_residual_slice_20261016.md` |
| A-only | `docs/sm86_integrated_accuracy_dashboard_20261016.md` |
| A-only | `docs/sm86_integrated_adversarial_review_slice_20261016.md` |
| A-only | `docs/sm86_integrated_audit_slice_20261016.md` |
| A-only | `docs/sm86_integration_audit_slice_20261016.md` |
| A-only | `docs/sm86_ipc_denominator_negative_20261015.md` |
| A-only | `docs/sm86_ipc_readiness_concentration_slice_20261016.md` |
| A-only | `docs/sm86_ipc_residual_ledger_slice_20261015.md` |
| A-only | `docs/sm86_l0c_scan_slice_20261015.md` |
| A-only | `docs/sm86_l0c_switch_slice_20261015.md` |
| A-only | `docs/sm86_l1_client_form_slice_20261016.md` |
| A-only | `docs/sm86_l1_lookup_form_scope_review_slice_20261016.md` |
| A-only | `docs/sm86_l1_lookup_retry_negative_20261015.md` |
| A-only | `docs/sm86_l1_lookup_slice_20261015.md` |
| A-only | `docs/sm86_l1_mshr_pressure_slice_20261015.md` |
| A-only | `docs/sm86_l1_packet_aware_lookup_slice_20261016.md` |
| A-only | `docs/sm86_l1_scope_audit_slice_20261015.md` |
| A-only | `docs/sm86_l1_setway_slice_20261015.md` |
| A-only | `docs/sm86_l1_wavefront_lookup_negative_20261016.md` |
| A-only | `docs/sm86_l1tex_observer_slice_20261015.md` |
| A-only | `docs/sm86_l1tex_write_coalescing_probe_slice_20261016.md` |
| A-only | `docs/sm86_l2_concentration_slice_20261015.md` |
| A-only | `docs/sm86_l2_dram_scope_audit_slice_20261015.md` |
| A-only | `docs/sm86_l2_fill_ownership_slice_20261015.md` |
| A-only | `docs/sm86_l2_frontier_observer_slice_20261016.md` |
| A-only | `docs/sm86_l2_hash_uncertainty_slice_20261015.md` |
| A-only | `docs/sm86_l2_operation_split_slice_20261016.md` |
| A-only | `docs/sm86_l2_packet_conservation_slice_20261016.md` |
| A-only | `docs/sm86_l2_pressure_observer_slice_20261015.md` |
| A-only | `docs/sm86_l2_pressure_strata_slice_20261015.md` |
| A-only | `docs/sm86_l2_prt_pressure_slice_20261015.md` |
| A-only | `docs/sm86_l2_queue_frontier_negative_20261016.md` |
| A-only | `docs/sm86_l2_route_correlation_slice_20261015.md` |
| A-only | `docs/sm86_l2_route_strata_slice_20261015.md` |
| A-only | `docs/sm86_l2_semantic_residual_ledger_slice_20261016.md` |
| A-only | `docs/sm86_l2_slice_observer_slice_20261015.md` |
| A-only | `docs/sm86_l2_write_lookup_negative_20261016.md` |
| A-only | `docs/sm86_l2_write_lookup_packet_slice_20261016.md` |
| A-only | `docs/sm86_l2_write_packet_observer_slice_20261016.md` |
| A-only | `docs/sm86_l2_write_policy_slice_20261015.md` |
| A-only | `docs/sm86_ldg_constant_slice_20261015.md` |
| A-only | `docs/sm86_ldgsts_ltc128b_scope_slice_20261016.md` |
| A-only | `docs/sm86_local_frame_coalesced_slice_20261016.md` |
| A-only | `docs/sm86_local_width_slice_20261015.md` |
| A-only | `docs/sm86_ltc128b_l1_slice_20261016.md` |
| A-only | `docs/sm86_ltc128b_probe_blocker_slice_20261016.md` |
| A-only | `docs/sm86_ltc128b_ptx_probe_slice_20261016.md` |
| A-only | `docs/sm86_metric_unit_crosswalk_slice_20261016.md` |
| A-only | `docs/sm86_near_far_pressure_slice_20261015.md` |
| A-only | `docs/sm86_plan_consistency_audit_slice_20261016.md` |
| A-only | `docs/sm86_priority_queue_audit_slice_20261015.md` |
| A-only | `docs/sm86_register_read_slice_20261015.md` |
| A-only | `docs/sm86_rejected_l1_local_experiments_20261016.md` |
| A-only | `docs/sm86_repaired_cbank_crosswalk_slice_20261016.md` |
| A-only | `docs/sm86_rf_allocate_slice_20261015.md` |
| A-only | `docs/sm86_scope_aware_report_slice_20261015.md` |
| A-only | `docs/sm86_scope_threshold_sensitivity_slice_20261016.md` |
| A-only | `docs/sm86_sfu_dependent_slice_20261015.md` |
| A-only | `docs/sm86_store_feedback_no_fence_negative_20261016.md` |
| A-only | `docs/sm86_store_fence_slice_20261015.md` |
| A-only | `docs/sm86_unresolved_gap_tracker_20261016.md` |
| A-only | `docs/sm86_writeback_scenario_negative_20261015.md` |
| A-only | `plans/3090ti_full_replay_modeling_plan.md` |
| different | `plans/README.md` |
| different | `sim/arch/src/lib.rs` |
| different | `sim/direct_run/src/emit.rs` |
| different | `sim/direct_run/src/lib.rs` |
| different | `sim/direct_run/src/main.rs` |
| different | `sim/direct_run/src/metrics.rs` |
| different | `sim/engine/src/lib.rs` |
| different | `sim/mem/src/access.rs` |
| different | `sim/mem/src/cache_op.rs` |
| different | `sim/mem/src/l1.rs` |
| different | `sim/mem/src/l2.rs` |
| different | `sim/mem/src/lib.rs` |
| different | `sim/semantics/src/boundary.rs` |
| different | `sim/sm/src/evidence.rs` |
| different | `sim/sm/src/issue.rs` |
| different | `sim/sm/src/issue/candidates.rs` |
| different | `sim/sm/src/issue/config.rs` |
| different | `sim/sm/src/issue/readiness.rs` |
| different | `sim/sm/src/metrics.rs` |
| different | `sim/sm/src/program.rs` |
| different | `sim/sm/src/replay.rs` |
| different | `sim/sm/src/system.rs` |
| different | `sim/sm/tests/machine_local_replay.rs` |
| different | `sim/trace/src/async_sidecars.rs` |
| different | `sim/trace/src/bin/submission_obligation_report.rs` |
| different | `sim/trace/src/exec_info.rs` |
| different | `sim/trace/src/obligations.rs` |
| different | `sim/trace/src/reader.rs` |
| different | `tools/nvbit_tracer/tracer_tool/tracer_tool.cu` |
| different | `tools/sm120_corpus_replay.py` |
| different | `tools/sm120_metric_report.py` |
| different | `tools/sm120_replay_bench.py` |
| A-only | `workloads/portable_lcd/.gitignore` |
| A-only | `workloads/portable_lcd/Makefile` |
| A-only | `workloads/portable_lcd/Makefile.template` |
| A-only | `workloads/portable_lcd/PRESERVED_BINARIES.md` |
| A-only | `workloads/portable_lcd/README.md` |
| A-only | `workloads/portable_lcd/ac4_mixed_long_sm80/Makefile` |
| A-only | `workloads/portable_lcd/ac4_mixed_long_sm80/PRESERVED_BINARY.md` |
| A-only | `workloads/portable_lcd/ac4_mixed_long_sm80/ac4_mixed_long_sm80.cu` |
| A-only | `workloads/portable_lcd/common.mk` |
| A-only | `workloads/portable_lcd/conv2d_implicit_gemm_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/conv2d_implicit_gemm_bf16_sm80/conv2d_implicit_gemm_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/conv2d_implicit_gemm_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/cross_entropy_fwd_fp32_sm80/Makefile` |
| A-only | `workloads/portable_lcd/cross_entropy_fwd_fp32_sm80/cross_entropy_fwd_fp32_sm80.cu` |
| A-only | `workloads/portable_lcd/cross_entropy_fwd_fp32_sm80/main.cu` |
| A-only | `workloads/portable_lcd/cutlass/LICENSE` |
| A-only | `workloads/portable_lcd/cutlass/NOTES.md` |
| A-only | `workloads/portable_lcd/cutlass/VERSION` |
| A-only | `workloads/portable_lcd/cutlass/tools/recompute_subset.py` |
| A-only | `workloads/portable_lcd/cutlass/upstream/FILES.txt` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/axpby.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/clear.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/cooperative_copy.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/cooperative_gemm.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/copy.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/fill.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/functional.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/gemm.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/prefer.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/prefetch.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/tensor_algorithms.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/algorithm/tuple_algorithms.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/cluster_sm90.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/config.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm100.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm100_tma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm50.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm75.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm80.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm90.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm90_desc.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/copy_sm90_tma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm100.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm100_desc.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm100_umma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm61.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm70.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm75.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm80.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm90.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm90_desc.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm90_gmma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm90_gmma_ext.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm90_gmma_sparse.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/mma_sm90_gmma_sparse_ext.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/arch/util.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_atom.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm100.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm100_tma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm50.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm75.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm80.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm90.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm90_tma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/copy_traits_sm90_tma_swizzle.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_atom.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm100.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm61.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm70.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm75.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm80.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm90.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm90_gmma.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm90_gmma_ext.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm90_gmma_sparse.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/mma_traits_sm90_gmma_sparse_ext.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/atom/partitioner.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/config.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/alignment.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/array.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/array_aligned.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/array_subbyte.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/bit_field.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/cuda_types.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/tuple.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/container/type_list.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/int_tuple.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/layout.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/layout_composed.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/arithmetic_tuple.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/complex.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/int.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/integer_sequence.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/integral_constant.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/integral_ratio.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/math.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/numeric_types.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/numeric/real.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/pointer.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/pointer_base.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/pointer_flagged.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/pointer_sparse.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/pointer_swizzle.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/stride.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/swizzle.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/swizzle_layout.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/tensor.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/tensor_impl.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/tensor_predicate.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/tensor_zip.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/underscore.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/util/debug.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/util/print.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cute/util/type_traits.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/aligned_buffer.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/arch.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/cache_operation.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/config.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/memory.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/memory_sm75.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/memory_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm50.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm60.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm61.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm75.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm89.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sm90.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sparse_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/mma_sparse_sm89.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/synclog.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/wmma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/wmma_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/wmma_sm72.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/arch/wmma_sm75.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/array.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/array_planar_complex.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/array_subbyte.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/bfloat16.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/block_striped.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/complex.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/constants.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/conv2d_problem_size.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/conv3d_problem_size.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/convolution.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/device/implicit_gemm_convolution.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/kernel/default_conv2d.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/kernel/default_conv2d_fprop.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/kernel/implicit_gemm_convolution.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/kernel/implicit_gemm_convolution_fusion.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/kernel/implicit_gemm_convolution_strided_dgrad.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_few_channels.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_fixed_channels.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_params.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/conv2d_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/implicit_gemm_fprop_fusion_multistage.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/implicit_gemm_multistage.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/implicit_gemm_pipelined.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/implicit_gemm_wgrad_fusion_multistage.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/threadblock/threadblock_swizzle.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/conv/warp/scale_bias_relu_transform.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/coord.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/core_io.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/cuda_host_adapter.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/cutlass.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/detail/collective.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/detail/helper_macros.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/detail/layout.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/device_kernel.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/activation.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/conversion_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_clamp.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_gelu.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_generic.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_hardswish.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_params.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_planar_complex.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_relu.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_relu0.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/linear_combination_sigmoid.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/reduction_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/thread/scale_type.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_epilogue_simt.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_epilogue_with_broadcast.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_epilogue_with_reduction.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_thread_map_simt.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_thread_map_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/default_thread_map_volta_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue_base.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue_base_streamk.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue_depthwise.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue_streamk_with_broadcast.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue_with_broadcast.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/epilogue_with_reduction.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/interleaved_epilogue.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/output_iterator_parameter.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/output_tile_thread_map.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/predicated_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/predicated_tile_iterator_direct_conv.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/predicated_tile_iterator_params.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/shared_load_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/shared_load_iterator_mixed.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/threadblock/shared_load_iterator_pitch_linear.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/fragment_iterator_simt.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/fragment_iterator_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/fragment_iterator_volta_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/simt_policy.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/tensor_op_policy.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/tile_iterator_simt.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/tile_iterator_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/tile_iterator_volta_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/epilogue/warp/volta_tensor_op_policy.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/exmy_base.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/fast_math.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/float8.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/float_subbyte.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/floating_point_nvrtc.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/functional.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/collective/sm80_mma_multistage.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/dispatch_policy.hpp` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/gemm.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/gemm_enumerated_types.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/thread/mma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/thread/mma_sm50.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/thread/mma_sm60.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/thread/mma_sm61.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma_core.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma_core_simt.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma_core_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma_core_sm75.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma_core_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_mma_core_wmma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/index_remat.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/mma_base.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/mma_multistage.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/mma_pipelined.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/mma_singlestage.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/threadblock_swizzle.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/default_mma_complex_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/default_mma_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/default_mma_tensor_op_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_complex_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_complex_tensor_op_fast_f32.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_gaussian_complex_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_gaussian_complex_tensor_op_tile_iterator_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_mixed_input_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_simt.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_simt_policy.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_simt_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_fast_f32.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_policy.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_tile_iterator_wmma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/mma_tensor_op_wmma.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm/warp/scale_bias_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/gemm_coord.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/half.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/integer_subbyte.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/matrix.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/permute.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/pitch_linear.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/tensor.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/tensor_op_multiplicand_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/tensor_op_multiplicand_sm75.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/tensor_op_multiplicand_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/layout/vector.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/matrix.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/matrix_coord.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/matrix_shape.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/numeric_conversion.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/numeric_size.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/numeric_types.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/pitch_linear_coord.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/platform/platform.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/predicate_vector.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/quaternion.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/real.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/reduction/thread/reduce.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/semaphore.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/subbyte_reference.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/tensor_coord.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/tensor_ref.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/tensor_view.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/tfloat32.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/trace.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/pitch_linear_thread_map.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/thread/transpose.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/thread/unary_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/predicated_tile_access_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/predicated_tile_access_iterator_params.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/predicated_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_access_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_iterator.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/uint128.h` |
| A-only | `workloads/portable_lcd/cutlass/upstream/cutlass/wmma_array.h` |
| A-only | `workloads/portable_lcd/embedding_gather_int32_sm80/Makefile` |
| A-only | `workloads/portable_lcd/embedding_gather_int32_sm80/embedding_gather_int32_sm80.cu` |
| A-only | `workloads/portable_lcd/embedding_gather_int32_sm80/main.cu` |
| A-only | `workloads/portable_lcd/flash_attn/LICENSE` |
| A-only | `workloads/portable_lcd/flash_attn/NOTES.md` |
| A-only | `workloads/portable_lcd/flash_attn/VERSION` |
| A-only | `workloads/portable_lcd/flash_attn/lcd_glue/bwd_sm80.cuh` |
| A-only | `workloads/portable_lcd/flash_attn/lcd_glue/fwd_sm80.cuh` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/AUTHORS` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/LICENSE` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/alibi.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/block_info.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/dropout.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/flash_bwd_kernel.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/flash_fwd_kernel.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/kernel_traits.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/mask.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/namespace_config.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/philox.cuh` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/philox_unpack.cuh` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/rotary.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/softmax.h` |
| A-only | `workloads/portable_lcd/flash_attn/upstream/utils.h` |
| A-only | `workloads/portable_lcd/flash_attn_bwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/flash_attn_bwd_bf16_sm80/flash_attn_bwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/flash_attn_bwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/flash_attn_fwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/flash_attn_fwd_bf16_sm80/flash_attn_fwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/flash_attn_fwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/gemm_hmma_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/gemm_hmma_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/gemm_hmma_fp16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/gemm_hmma_fp16_sm80/gemm_hmma_fp16_sm80.cu` |
| A-only | `workloads/portable_lcd/gemm_hmma_fp16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/gemm_imma_int8_sm80/Makefile` |
| A-only | `workloads/portable_lcd/gemm_imma_int8_sm80/gemm_imma_int8_sm80.cu` |
| A-only | `workloads/portable_lcd/gemm_imma_int8_sm80/main.cu` |
| A-only | `workloads/portable_lcd/gemm_sparse_2to4_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/gemm_sparse_2to4_bf16_sm80/gemm_sparse_2to4_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/gemm_sparse_2to4_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/layernorm_fwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/layernorm_fwd_bf16_sm80/layernorm_fwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/layernorm_fwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/preserved_binaries_manifest.json` |
| A-only | `workloads/portable_lcd/rmsnorm_bwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/rmsnorm_bwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/rmsnorm_bwd_bf16_sm80/rmsnorm_bwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/rmsnorm_fwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/rmsnorm_fwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/rmsnorm_fwd_bf16_sm80/rmsnorm_fwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/roster.toml` |
| A-only | `workloads/portable_lcd/runner_common.h` |
| A-only | `workloads/portable_lcd/silu_fwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/silu_fwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/softmax_safe_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/softmax_safe_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/softmax_safe_bf16_sm80/softmax_safe_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/swiglu_fwd_bf16_sm80/Makefile` |
| A-only | `workloads/portable_lcd/swiglu_fwd_bf16_sm80/main.cu` |
| A-only | `workloads/portable_lcd/swiglu_fwd_bf16_sm80/swiglu_fwd_bf16_sm80.cu` |
| A-only | `workloads/portable_lcd/tracer_runtime_signal_demo/NOTES.md` |
| A-only | `workloads/portable_lcd/tracer_runtime_signal_demo/cutlass_sm86_diagnostic_fires.log` |
| A-only | `workloads/portable_lcd/tracer_runtime_signal_demo/inline_ptx_diagnostic_quiet.log` |
| A-only | `workloads/portable_lcd/verify_kernel_contracts.py` |
| A-only | `workloads/sm86_adaptations/README.md` |
| A-only | `workloads/sm86_adaptations/flashinfer_reference.py` |
| A-only | `workloads/sm86_adaptations/runner.py` |
| A-only | `workloads/sm86_adapted_l2_014_audio_varlen_attention/definition.json` |
| A-only | `workloads/sm86_adapted_l2_014_audio_varlen_attention/reference.py` |
| A-only | `workloads/sm86_adapted_l2_014_audio_varlen_attention/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/006_multimodal_rope_position_calculation/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/006_multimodal_rope_position_calculation/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/006_multimodal_rope_position_calculation/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/011_rotary_position_embedding/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/011_rotary_position_embedding/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/011_rotary_position_embedding/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/018_fused_rope_with_qk_norm_and_kv_cache_update/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/018_fused_rope_with_qk_norm_and_kv_cache_update/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/018_fused_rope_with_qk_norm_and_kv_cache_update/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/020_vision_patch_merger_spatial_shuffle_mlp/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/020_vision_patch_merger_spatial_shuffle_mlp/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/020_vision_patch_merger_spatial_shuffle_mlp/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/022_vision_mlp_gelu_backward/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/022_vision_mlp_gelu_backward/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/022_vision_mlp_gelu_backward/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/023_multimodal_rope_position_computation_with_grid_based_indexing/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/023_multimodal_rope_position_computation_with_grid_based_indexing/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/023_multimodal_rope_position_computation_with_grid_based_indexing/workload.jsonl` |
| A-only | `workloads/sm86_official_missing_controls/058_moe_expert_token_radix_sort_with_prefix_sum/definition.json` |
| A-only | `workloads/sm86_official_missing_controls/058_moe_expert_token_radix_sort_with_prefix_sum/reference.py` |
| A-only | `workloads/sm86_official_missing_controls/058_moe_expert_token_radix_sort_with_prefix_sum/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/002_vae_conv3x3_groupnorm_silu_residual_fused/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/002_vae_conv3x3_groupnorm_silu_residual_fused/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/002_vae_conv3x3_groupnorm_silu_residual_fused/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/003_lm_head_projection_with_logit_slicing/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/003_lm_head_projection_with_logit_slicing/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/003_lm_head_projection_with_logit_slicing/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/004_attention_output_projection_with_reshape_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/004_attention_output_projection_with_reshape_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/004_attention_output_projection_with_reshape_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/004_fused_residual_rms_mlp/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/004_fused_residual_rms_mlp/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/004_fused_residual_rms_mlp/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/005_conv_gated_projection_with_causal_conv/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/005_conv_gated_projection_with_causal_conv/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/005_conv_gated_projection_with_causal_conv/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/005_swiglu_mlp_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/005_swiglu_mlp_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/005_swiglu_mlp_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/006_hyena_depthwise_conv1d_split_gate/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/006_hyena_depthwise_conv1d_split_gate/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/006_hyena_depthwise_conv1d_split_gate/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/007_hyena_fft_size_padding_rfft/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/007_hyena_fft_size_padding_rfft/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/007_hyena_fft_size_padding_rfft/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/008_expert_output_weighted_index_add_accumulation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/008_expert_output_weighted_index_add_accumulation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/008_expert_output_weighted_index_add_accumulation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/008_moe_sparse_routing_and_dispatch/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/008_moe_sparse_routing_and_dispatch/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/008_moe_sparse_routing_and_dispatch/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/010_attention_value_projection_with_transpose/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/010_attention_value_projection_with_transpose/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/010_attention_value_projection_with_transpose/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/012_fused_cos_sin_embedding_generation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/012_fused_cos_sin_embedding_generation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/012_fused_cos_sin_embedding_generation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/013_fused_residual_rms_norm_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/013_fused_residual_rms_norm_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/013_fused_residual_rms_norm_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/014_rotary_embedding_with_attention_scaling_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/014_rotary_embedding_with_attention_scaling_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/014_rotary_embedding_with_attention_scaling_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/016_rope_inverse_frequency_computation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/016_rope_inverse_frequency_computation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/016_rope_inverse_frequency_computation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/025_video_latent_gelu_activation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/025_video_latent_gelu_activation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/025_video_latent_gelu_activation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/027_video_spatial_attention_with_rope_3d/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/027_video_spatial_attention_with_rope_3d/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/027_video_spatial_attention_with_rope_3d/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/028_hybrid_attention_mask_preparation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/028_hybrid_attention_mask_preparation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/028_hybrid_attention_mask_preparation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/029_mamba_conv1d_with_gating/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/029_mamba_conv1d_with_gating/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/029_mamba_conv1d_with_gating/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/030_attention_output_projection_with_residual/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/030_attention_output_projection_with_residual/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/030_attention_output_projection_with_residual/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/031_flux_timestep_guidance_projection_embedding/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/031_flux_timestep_guidance_projection_embedding/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/031_flux_timestep_guidance_projection_embedding/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/031_repeat_kv_attention_matmul/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/031_repeat_kv_attention_matmul/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/031_repeat_kv_attention_matmul/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/032_attention_weights_matmul_with_value_projection/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/032_attention_weights_matmul_with_value_projection/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/032_attention_weights_matmul_with_value_projection/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/033_multi_scale_feature_pyramid/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/033_multi_scale_feature_pyramid/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/033_multi_scale_feature_pyramid/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/033_post_norm_residual/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/033_post_norm_residual/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/033_post_norm_residual/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/034_flux_multi_axis_rope_frequency_computation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/034_flux_multi_axis_rope_frequency_computation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/034_flux_multi_axis_rope_frequency_computation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/035_convnextv2_block_with_grn/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/035_convnextv2_block_with_grn/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/035_convnextv2_block_with_grn/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/036_convnextv2_layer_with_nhwc_persistence_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/036_convnextv2_layer_with_nhwc_persistence_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/036_convnextv2_layer_with_nhwc_persistence_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/037_flux_feedforward_gelu_approximate/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/037_flux_feedforward_gelu_approximate/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/037_flux_feedforward_gelu_approximate/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/038_flux_multi_head_rmsnorm_qk/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/038_flux_multi_head_rmsnorm_qk/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/038_flux_multi_head_rmsnorm_qk/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/039_adaptive_instance_normalization_2d_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/039_adaptive_instance_normalization_2d_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/039_adaptive_instance_normalization_2d_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/040_conv2d_residual_block/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/040_conv2d_residual_block/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/040_conv2d_residual_block/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/041_strided_conv2d_downsampling/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/041_strided_conv2d_downsampling/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/041_strided_conv2d_downsampling/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/045_fused_linear_gelu_grn_linear/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/045_fused_linear_gelu_grn_linear/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/045_fused_linear_gelu_grn_linear/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/049_attention_qk_matmul_with_gqa_repeat_and_scaling/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/049_attention_qk_matmul_with_gqa_repeat_and_scaling/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/049_attention_qk_matmul_with_gqa_repeat_and_scaling/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/050_attention_qkv_projection_with_bias_and_reshape/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/050_attention_qkv_projection_with_bias_and_reshape/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/050_attention_qkv_projection_with_bias_and_reshape/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/051_attention_qkv_with_qk_norm_single_kernel_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/051_attention_qkv_with_qk_norm_single_kernel_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/051_attention_qkv_with_qk_norm_single_kernel_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/053_gaussian_topk_sparse_activation/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/053_gaussian_topk_sparse_activation/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/053_gaussian_topk_sparse_activation/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/056_mamba_ssm_dt_projection_softplus_clamp_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/056_mamba_ssm_dt_projection_softplus_clamp_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/056_mamba_ssm_dt_projection_softplus_clamp_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/059_moe_group_score_aggregation_and_masking/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/059_moe_group_score_aggregation_and_masking/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/059_moe_group_score_aggregation_and_masking/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/061_tanh_gated_residual_add_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/061_tanh_gated_residual_add_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/061_tanh_gated_residual_add_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/064_latent_kv_expansion_with_split/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/064_latent_kv_expansion_with_split/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/064_latent_kv_expansion_with_split/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/065_fused_statistics_projection_and_split/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/065_fused_statistics_projection_and_split/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/065_fused_statistics_projection_and_split/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/068_fused_rope_application_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/068_fused_rope_application_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/068_fused_rope_application_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/069_rms_norm/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/069_rms_norm/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/069_rms_norm/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/071_kv_cache_update_with_rope/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/071_kv_cache_update_with_rope/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/071_kv_cache_update_with_rope/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/075_sam_hq_mask_decoder_two_way_transformer/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/075_sam_hq_mask_decoder_two_way_transformer/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/075_sam_hq_mask_decoder_two_way_transformer/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/078_group_norm_fusion/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/078_group_norm_fusion/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/078_group_norm_fusion/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/083_attention_score_value_matmul/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/083_attention_score_value_matmul/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/083_attention_score_value_matmul/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/084_silu_activation_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/084_silu_activation_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/084_silu_activation_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/088_rotary_position_embedding_application/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/088_rotary_position_embedding_application/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/088_rotary_position_embedding_application/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/090_batched_2d_rope_position_encoding_backward/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/090_batched_2d_rope_position_encoding_backward/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/090_batched_2d_rope_position_encoding_backward/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/094_time_decay_exponential_stabilization/definition.json` |
| A-only | `workloads/sm86_snapshot_controls/094_time_decay_exponential_stabilization/reference.py` |
| A-only | `workloads/sm86_snapshot_controls/094_time_decay_exponential_stabilization/workload.jsonl` |
| A-only | `workloads/sm86_snapshot_controls/manifest.json` |
| A-only | `workloads/sm86_substitutes/quant_substitute_runner.py` |

## Branch-specific commit summaries

### A / SM86

- 4575d436 Advance SM86 full-replay accuracy modeling
- fbb564df Allow sequential corpus replay timing mode
- 9ac99a6e Bundle SM86 workload sources and control runners
- bf069ab0 Clarify launch-scope and frontier-figure claims
- c451a8bf Add vivid SOTA and contribution figures
- a44ec6b9 Expand SOTA comparison and figure documentation
- 6f299e89 Add simulator architecture and causal timeline figures
- 45110839 Explain SIMT execution and interleaving tradeoffs
- 0c8791f5 Clarify full-only evidence and frontier semantics

### B / SM120

- 7471be0b sm120: align SMSP branch projection with CBU metric
- efea4707 sm120: expose pending-merge L2 read diagnostics
- e8933eef docs: record Q57-Q59 side-metric continuation
- 22052b3c docs: record Q60 IPC residual audit
- 9cb48b11 sm120: subtract ideal shared wavefronts from bank conflicts
- 3641433e sm120: separate matrix operations from shared conflicts
- eb227837 l2: scope pending merge classification to preexisting fills
- 0f916eae docs: record Q63-Q65 shared metric closure
- 687712f2 docs: record Round23 post-Q65 adversarial closure
- 18f762cb docs: clarify Q57-Q65 evidence naming
- 3dc7c25b sm120: use audited L1 pending-merge lookup counts
- cf79b9cc Revert "sm120: use audited L1 pending-merge lookup counts"
- 9ba700e8 docs: record Q66 L1 audit rejection
- c9dd4b9a sm120: preserve bounded L1 request mask audit
- 9ee5cafd sm120: add bounded L1 request mask audit
- f7b7eb26 docs: record Q67 L1 request mask audit
- 4c85d816 sm120: clarify L1 mask audit report semantics
- 430f9686 docs: record Round25 Q67 audit closure
- 5ec92d90 sm120: retain ordered TMA shadow diagnostic
- 41a3f92e docs: record Q68 Q69 shadow and IPC review
- 1864eb34 docs: record Round27 final adversarial resolution
- 7b01297c trace: retain typed TMA barrier generation cohorts
- fa9c5b7c trace: report TMA generation relation counts
- 8bd9274c docs: record Q70 TMA generation relation
- 4e7c4054 trace: bound TMA generation event references
- 6b37f5cb sm120: add bounded joint TMA L2 shadow diagnostic
- 99f656f5 docs: record Q71 joint TMA L2 diagnostic
- 783b3e13 docs: record Q70 full replay and Round28 resolution
- f5ce8015 docs: mark effective SM120 plan complete
- 4e35cdeb sm120: add conditional control-stall active IPC arm
- 0f615098 docs: record Q73 active IPC continuation
- c3b8442b docs: record Q73 full-corpus IPC closure
- 4c15d7cc sm120: add effective active IPC lifetime envelope
- d2427a4b docs: record Q74 active IPC envelope
- 99308f5f docs: record Q74 full-corpus IPC promotion
- b8558f0b sm120: add guarded pure TMA IPC envelope
- 5abe8740 docs: record Q76 L1 L2 residual diagnosis
- 05442f3e sm120: preserve partial TMA source TEX sectors
- 8d162479 docs: record Q79 partial TMA source projection
- 6c1b5a05 trace: quarantine typed TMA sidecar provenance
- 2529b412 docs: record Q80 Q81 rejection and Q82 provenance
- 75268420 docs: record L1 carveout hard stop
- 408381b1 trace: add reuse and clock sandwich diagnostics
- 3b1f7b04 docs: record Q83 diagnostics
- 84200733 tools: tag join and client scope provenance
- cc18e183 docs: record Q84 provenance tags
- 5dc6827a trace: add directional TMA reuse diagnostics
- 6ecd8b06 docs: record Q85 directional diagnostics
- 9d259968 docs: record B2 B3 offline closure
- 57ec4e63 cleanup: consolidate SM120 contracts and defaults
- 027523b7 docs: record latest complete-corpus replay
- 0ceb740c fix shared opcode direction and conflict aggregation
- 3bd7e869 docs: record shared-memory metric improvements
- 3ed08318 model SM120 global-atomic wavefront observer alias
- c90d8e22 docs: record atomic observer improvement
- 898d2727 refine elapsed IPC launch-boundary observation
- a9638f3e docs: record elapsed IPC observation improvement
- 6c56ea5a model partial-store read-modify-write demand
- cf8f8310 extend partial-store RMW to local stores
- 1f45543a docs: record local-store RMW refinement
- 4f9db588 Preserve per-lane conflicts for shared atomics
- 138d337e Decouple phased shared observers from scheduler timing
- f29ba8ce Record Nsight Compute executable path
- 2b082725 Project dynamic local-memory interleave in L1 observer
- 5373b8d1 Project dynamic local stores in mixed L2 observer
- 4b5de0bb Refine local-memory width interleave observer
- 0b2eeb49 Pack sparse active shared loads by phase capacity
- 5e7a3264 model SM120 no-allocate global loads
- fe4dcaa3 project bounded constant-bank DRAM reads
- afa73c87 complete store-free resident L1 hits
- bd14d64f extend resident L1 correction to tiny stores
- ccb6b6f3 widen global-read resident L1 scope
- ce328f5c use trace store scope for resident L1 correction
- 8fde2f8a extend resident global-read small-store scope
- b00cf5f3 scope resident global reads by block topology
- 3285579e scope resident reads to trace topology
- 4743c1f2 scope adjacent full-warp store lookup coalescing
- 9c7a6147 refine full warp lds128 observer projection
- 2847a9b7 partition generic store observer lookups
- 6af860d1 add generic store l1 observer
- b4e57268 Allow sequential SM120 replay and enable latent envelope
- 818a1dbd Enable prototype carded DRAM by default

## Working-tree/capture inventory (from bounded file inventories)

- `GPU_simulator`: ~7.9 GiB working tree, 49,444 files; `GPU_simulator_5090`: ~9.0 GiB, 38,293 files. Most size is generated `target/`, virtualenv, or workload artifacts.
- `sm86_rtx_3090_capture`: ~58 GiB, 206,831 files; `sm120_rtx_5090_capture`: ~88 GiB, 274,192 files. Large protobuf/Nsight reports are evidence artifacts, not source-tree inputs.
- `final_results`: ~7.4 GiB, 5,186 files; baseline JSONL/compressed logs are retained as external references and are not copied wholesale.

## Focused architectural inventory

- `sim/`: shared Rust trace → SM → memory → engine → direct-run pipeline; both branches modify the same core modules listed in `consultation_prompt.txt`.
- `arch_profiles/`: A adds SM86 candidates and retains measured SM86 card values; B changes only the SM120 DRAM card block from typed gap to an explicitly unvalidated prototype.
- `tools/`: A generalizes `sm120_corpus_replay.py` for SM86 and sequential runs; B keeps SM120-specific defaults and sequential option. `nvbit_tracer` differs only in a one-line receive-thread fix at B tip.
- `workloads/`: A adds SM86 adaptation/substitute and portable-LCD trees; B adds SM120/Blackwell and newer low-precision/TMA rows. Workload source is retained only when needed by the corresponding capture flow; no generated captures are copied into the source tree.
- `docs/`: A contains SM86 continuation/evidence notes; B contains compact SM120 contract/status notes. Manuscript/paper paths are excluded.

## Capture and baseline interface inventory

- SM86 target contract: `sm86_rtx_3090_capture/config/target.json`; controls in `controls/`; replay metadata in `replay/`; authoritative full/fast outputs in `final_results/3090ti*`.
- SM120 target contract: `sm120_rtx_5090_capture/config/target.json`; dynamic traces/profiles and sidecars in `captures/` and `profiles/`; authoritative outputs in `final_results/5090*`.
- `final_results/NCU_profiling_diff.md` states NCU policies differ (SM86 one launch vs SM120 all matching launches), so campaign wall times are not interchangeable.
- `final_results/runtime_note.md` records SM86 356 attempted / 286 full successes and `final_results/5090_replay_runtime_note.md` records SM120 339/339 success. Both explicitly state NCU was not replay input.

## Diff artifacts

- `research_a_sim.patch` and `research_b_sim.patch` contain the complete common-base-to-tip patches for simulator/architecture/replay tooling paths (generated here, never written to source).
- `consultation_prompt.txt` records the non-obvious merge conflicts and questions before structural changes.
