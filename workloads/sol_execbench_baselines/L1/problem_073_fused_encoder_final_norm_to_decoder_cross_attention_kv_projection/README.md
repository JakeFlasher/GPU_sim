# problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Encoder final RMSNorm fused with decoder cross-attention K/V projection (FP16).             |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection/` |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local speedup vs reference  | <diagnostic only on sm_86; expected < 1 vs the pure-cuBLAS reference path>                  |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | sm_86 the calibration workstream: nvbit_metrics + cupti_activity + sanitizer + nsys + static all ok.             |

## Approach

Custom RMSNorm CUDA kernel + `at::matmul` for the K/V projection
matmuls. Phase 1 (custom kernel) computes the FP16-quantised
RMSNorm-normalized intermediate (matching the PyTorch reference's
`(norm_weight * hidden_states).to(input_dtype)` precision profile).
Phase 2 (host-side `at::matmul`) routes the two matmuls through
the same cuBLAS HGEMM path PyTorch's `F.linear` uses, including
the `allow_fp16_reduced_precision_reduction` setting that defaults
to `True` on Ampere+ hosts.

This two-phase shape (R11 refactor) closes the cross-arch
divergence documented in
`BL-20260528-delegate-matmul-to-at-matmul-for-reference-hgemm-
precision-parity`. Earlier kernel iterations used a pure FP32
accumulator across the matmul reduction, which diverged from the
reduced-precision cuBLAS reference by ~1 FP16 ULP on near-zero
outputs and failed `workload_index=13`'s `max_atol=2.9e-5`
tolerance on sm_86.

## sm_86 the calibration workstream result (R12)

All 16 sol-execbench workloads PASSED on this 3090 Ti host after
the R11 refactor. Sidecar at
`scripts/baselines_harness/.harness_traces/problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection__cuda_cpp__sm86/full_correctness_v1.json`;
SHA-256 cross-stamped in bundle manifest + ledger.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles + runs this baseline correctly
after the R11 refactor; Phase-2 profiling runs on B200 (sm_100a).
Cross-arch divergence axes:

- HBM3e vs GDDR6X / GDDR7 bandwidth: bandwidth dominates the
  output K/V tensor write-back at small batch sizes.
- DSMEM and cluster-cooperative reduction (sm_100a-only): out of
  scope for sm_86; the RMSNorm reduction stays in a single CTA.
- L2 carveout: B200's larger L2 may absorb the K/V projection
  weight matrix (128×1024 FP16 = 256 KiB) entirely.
- TCGen05 / TMEM divergence: the at::matmul delegation routes
  through PyTorch's matmul backend which on B200 will use 5th-gen
  tensor cores via cuBLAS; on sm_86 it uses the Ampere tensor
  cores.
