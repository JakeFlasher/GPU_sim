# sm_86 Submission Roster Task3 Review

## Invocation

- Tool: `/home/jakeshea/.codex/skills/humanize/scripts/ask-codex.sh`
- Model: `gpt-5.5`
- Effort: `high`
- Timeout: `5400s`
- Exit code: `0`
- Duration: `264s`
- Saved output: `.humanize/skill/2026-06-01_16-26-58-1738715-3c13f234/output.md`

## Verdict

Task3 verdict: AC-2 remains valid. No blocking roster defects were found. Codex did not edit files and did not run NCU. The committed validator reports 47 primary cells, 35 primary unique slugs, 10 replacements, all family minima met, and 6 depth-first slugs.

## Unsupported-sm_86 Blockers

- BLOCKING: none. Compatible rows are slug-specific, have existing workload roots and shape sources, and pass `workload_root_not_slug_specific`, Makefile `ARCH_GATE`, source `// arch_gate:`, and forbidden-token checks.
- BLOCKING: none for portable-LCD `lcd` expansion. The validator expands `lcd` to `sm_86 sm_90 sm_120`, and `workloads/portable_lcd/roster.toml` declares active LCD-portable rows as `sm_86 sm_90 sm_120`.
- BLOCKING: none for regression coverage. Negative tests cover unsupported tokens, non-sm_86 Makefiles, stale source gates, and shape-catalog workload roots.

## Family/Depth/Replacement Blockers

- BLOCKING: none. Current counts pass: primary cells `47 >= 42`, unique primary slugs `35 >= 35`, replacements `10 >= 10`.
- BLOCKING: none. Family unique slug counts are `dl_primitives=8`, `gemm_family=8`, `attention_conv=3`, `memory_stress=9`, `irregular_hostile=4`, `vendor_baselines=3`.
- BLOCKING: none. Depth-first slugs are `conv2d_implicit_gemm_bf16_sm80`, `dram_streaming_dot_product`, `flash_attn_fwd_bf16_sm80`, `gemm_hmma_bf16_sm80`, `gemm_hmma_fp16_sm80`, and `global_atomic_histogram`.
- QUEUED: Capacity slack is thin. Unique primary slugs, replacement count, depth-first slug count, and `irregular_hostile` are at exact floors, so task4 substitutions must preserve these dimensions.

## Queued Repeatability/Preflight Risks

- QUEUED: Three roster-referenced run binaries are currently absent from the filesystem: `shared_bank_conflict_stride_run`, `cp_async_shared_tile_copy_run`, and `vendor_gemm_cublas_run`. These are task4 preflight issues, not AC-2 blockers, because AC-2 validates roster breadth and compatibility rather than binary readiness.
- QUEUED: `data/sm_86_predictor_accuracy_recovery_v2/expansion_candidates.json` still records binary paths and hashes for those missing binaries. Task4 should refresh binary existence, hash, and provenance before manifest staging.
- QUEUED: Task6 should watch repeatability on cache-state and contention-heavy rows: `working_set_capacity_sweep`, `warm_cold_pair_relaunch`, `l2_thrash_pointer_chase`, `global_atomic_histogram`, and vendor cuBLAS rows.

## Tracker Recommendation

Move task3 to completed with evidence: independent Codex review found no blocking AC-2 roster defects; validator summary is `47 primary / 35 unique / 10 replacements / 6 depth-first`; unsupported-sm_86 gate audit found zero compatible-row issues. Add queued side issues for task4 binary rebuild/substitution and task6 repeatability monitoring.
