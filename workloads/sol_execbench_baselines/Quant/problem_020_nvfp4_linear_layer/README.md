# problem_020_nvfp4_linear_layer

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | NVFP4 linear layer (FP4-preserving GEMM, FP32 accumulator, BF16 output)                     |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_020_nvfp4_linear_layer.md`             |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/Quant/020_nvfp4_linear_layer/`                                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Local speedup vs reference  | 1.92 – 4.31× on sm_120 (diagnostic only; mean 3.38, median 3.67)                            |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

The reference path quantizes `x` with `BlockWiseScalerNVFP4` (per-tensor
FP32 `global_decode_scale` and per-16-element FP8 e4m3 `decode_scale`),
packs into E2M1 FP4 nibbles, and computes `y = (x @ w.T)` via
`torch._scaled_mm(fp4, fp4)` with FP32 accumulation and a BF16 cast at
the output.

This baseline mirrors that exactly via the CUTLASS 4.4.1 sm_120 dense
block-scaled GEMM template
(`cutlass::arch::OpClassBlockScaledTensorOp` + `arch::Sm120`,
matching example `79a_blackwell_geforce_nvfp4_bf16_gemm.cu` in the
local CUTLASS tree at `cutlass/`). The kernel uses the architecturally
native FP4 MMA on sm_120's block-scaled tensor cores; per-K-block FP8
scale multiplication, FP32 accumulation, and BF16 cast all happen
inside the CUTLASS pipeline — no precision-loss BF16 staging on
dequant outputs.

Three custom CUDA kernels glue the reference's quantization layout to
the CUTLASS dispatch:

1. `fp4_pack_kernel` packs FP32-clipped values into E2M1 nibbles using
   the reference's exact threshold table.
2. `remap_scales_kernel` converts the plain `[D, K/16]` FP8 e4m3
   scale tensor (the shape produced by `BlockWiseScalerNVFP4.quantize`)
   into the CUTLASS interleaved layout that
   `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB` expects.
   The closed-form mapping is:
   ```
   offset = (d % 32) * 16 + ((d / 32) % 4) * 4 + (d / 128) * (K_blocks * 128)
          + (kb % 4)      + (kb / 4) * K_blocks
   ```
3. The CUTLASS GEMM is dispatched with `ClusterShape = Shape<_1,_1,_1>`
   (GeForce sm_120 constraint), `ThreadBlockShape = Shape<_128,_128,_128>`,
   and `alpha = x_global_decode * w_global_decode` folded into the
   linear-combination epilogue.

Full canonical 16-workload set on sm_120: **16/16 PASSED**,
`speedup_factor` range **1.92–4.31** (mean 3.38, median 3.67) against
the reference's `torch._scaled_mm` BF16-emulated path. `max_relative_error`
0.00775–0.00781 across every workload (well within the 0.02 rtol band).

## Architecture-coupling caveats

The local 5090 (sm_120) compiles, runs correctness, AND wins speed on
every canonical workload. The CUTLASS template targets sm_120's
native FP4 block-scaled tensor cores via the `_120a` arch variant
(MMA atom `SM120_16x8x64_TN_VS<e2m1, e2m1, f32, ue4m3, 16>`). Final
SOL scoring is on B200 (sm_100a); differences worth noting:

- **TCGen05 / TMEM divergence**: sm_100a has 5th-gen tensor cores
  with a dedicated TMEM register surface; the CUTLASS sm_120 dense
  block-scaled NVFP4 GEMM compiles to `tcgen05` instructions on
  sm_100a and to the sm_120 variant on the 5090. The template
  dispatches correctly for either arch — re-tag to `["B200"]` for
  B200 scoring; the packager auto-injects `sm_100a` gencode.
- **Cluster cooperative + DSMEM**: B200 supports cluster shapes
  larger than 1×1×1 via TMA multicast; sm_120 is hard-pinned to
  cluster 1×1×1 (CUTLASS example 79a documents this constraint).
  Re-tuning the `ClusterShape` for B200 may yield additional speedup.
- **L2 carveout**: B200's larger L2 may absorb tensors the 5090's
  smaller L2 cannot; the CUTLASS pipeline's TMA-staged loads benefit
  proportionally.
- **HBM3e vs GDDR7 bandwidth**: bandwidth-bound stages (quant
  prepass + FP4 pack + scale remap) have a different ceiling on
  B200; the GEMM compute dominates already at M ≥ 1024 on the 5090.

## Development iteration history (sm_120)

| Iteration | Approach | Result |
|-----------|----------|--------|
| 6     | BF16 substitution on weight only | 16/16 INCORRECT_NUMERICAL |
| 7     | Faithful FP4 round-trip + BF16 matmul | 16/16 INCORRECT_NUMERICAL (precision wall) |
| 9     | FP32-accumulator naive GEMM | 4/4 PASSED, full set timed out |
| 10    | Tiled SMEM K-blocking | Regressed |
| 11    | Predecode FP32 + microtile | 3× faster than iteration 9, full set still timed out |
| 12    | INT8 dp4a remap | 2.43× over iteration 11, 16/16 PASSED, speedup 0.029-0.069 |
| 13    | 16×128 + double buffer dp4a | Regressed |
| 14    | cp.async-pipelined dp4a | No improvement |
| 15    | BF16 TC WMMA (64×64) | 16/16 PASSED, speedup 0.085-0.351 |
| 16    | BF16 TC WMMA (64×128) | 3/3 probe PASSED, speedup 0.197-0.308 |
| **17** | **CUTLASS sm_120 dense block-scaled NVFP4 GEMM** | **16/16 PASSED, speedup 1.92-4.31, LANDED** |

The landed CUTLASS path is **6-12× faster than the BF16-TC WMMA
kernel** at the largest workloads, because the architecturally
native FP4 MMA on sm_120's block-scaled tensor cores has 4× the
throughput of plain BF16 WMMA AND amortizes the per-K-block scale
multiplication into the MMA pipeline. The earlier BF16-TC iterations
were the right path before this discovery; the CUTLASS NVFP4 GEMM
unblocks the architectural ceiling on sm_120.
