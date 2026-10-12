# problem_001_fused_add_rmsnorm_h2048

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | FlashInfer fused add RMSNorm                                                                |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_001_fused_add_rmsnorm_h2048.md`        |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/001_fused_add_rmsnorm_h2048/`                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                           |
| Local speedup vs reference  | 4.01 – 9.99× on sm_120 (diagnostic only)                                                    |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

### Local sm_120 harness result (all 7 workloads PASSED)

| Workload | Status | latency_ms | reference_latency_ms | speedup_factor |
|---|---|---|---|---|
| 0 | PASSED | 0.00306 | 0.02526 | 8.27 |
| 1 | PASSED | 0.00363 | 0.02413 | 6.64 |
| 2 | PASSED | 0.00586 | 0.02942 | 5.02 |
| 3 | PASSED | 0.00910 | 0.03850 | 4.23 |
| 4 | PASSED | 0.79530 | 7.94890 | 9.99 |
| 5 | PASSED | 0.60963 | 5.50321 | 9.03 |
| 6 | PASSED | 0.00821 | 0.03295 | 4.01 |

Hardware: NVIDIA GeForce RTX 5090 GPU (sm_120). Evaluation ran
through the native `sol-execbench` CLI (no Docker shim). The cache /
manifest now key on `native_env_hash` (the SHA-256 over python /
torch / nvcc / cuda-runtime / cutlass-tree / cudnn /
sol_execbench-package versions, see
`tools/sol_baselines_harness/native_env.py`), so captured traces are
identified by that hash rather than by any container digest. The
benchmark `definition.json` was staged at
`.tmp_staged_benchmarks/FlashInfer-Bench/001_fused_add_rmsnorm_h2048/definition.json`
with the upstream-incompatible empty `hf_id` normalised to `null` (a
localised fix for an upstream-vs-data Pydantic divergence).

## Approach

One CTA per row (batch element); BF16x2 vectorised loads of `hidden_states` and `residual`; FP32 accumulator for the row's sum-of-squares; intra-warp reduction via `__shfl_xor_sync`, inter-warp reduction via shared memory; `rsqrtf` in FP32; final `(h + r) * inv_rms * weight` computed in FP32 and downcast to BF16 on store. `hidden_size=2048` is baked in as a `constexpr`. Launch on `at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

## Architecture-coupling caveats

The local 5090 (sm_120) compiles and smoke-runs this baseline; final SOL scoring
is on B200 (sm_100a). Differences worth noting for this pattern:

- HBM3e vs GDDR7 bandwidth: fused_add_rmsnorm is memory-bound; the 5090's bandwidth ceiling makes local wall-time a weak proxy for B200 SOL Score. sm_120 timing primarily validates correctness and launch-config sanity.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are not used here; for very large hidden_size, B200 has additional headroom that this template does not exploit.
- L2 carveout: B200's larger L2 absorbs the `weight` tensor across batch rows; the 5090's smaller L2 may not.
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or the dedicated TMEM register surface, so the kernel is not exposed to sm_100a-only TCGen05 / TMEM codegen differences. A future GEMM-fused variant of this op (which would touch TCGen05 / TMEM on B200) is out of scope for the current rmsnorm template.
- `__launch_bounds__` is the same on both archs but register pressure is workload-specific; verify occupancy via `nvcc -Xptxas=-v` if regression observed.

Evidence-label conventions per `CLAUDE.md` § Architecture Portability Discipline.
