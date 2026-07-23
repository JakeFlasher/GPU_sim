# problem_014_fp8_yarn_rope_embedding

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | FP8-quantized YaRN rotary position embedding, qk_rope_head_dim=64                           |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_014_fp8_yarn_rope_embedding.md`        |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/Quant/014_fp8_yarn_rope_embedding/`                           |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Local speedup vs reference  | 8.19 – 19.84× on sm_120 (diagnostic only)                                                   |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

YaRN inv-frequencies are constants of the YaRN config (qk_rope_head_dim,
rope_theta, scaling_factor, beta_fast, beta_slow, original_max_pos), so
they are precomputed on the host once per kernel invocation and staged
into a 128-byte device buffer.

The CUDA kernel uses one CTA per sequence position with a single warp
of 32 threads (one per frequency `k ∈ [0, 32)`). Each thread:

1. Computes `freq[k] = position_ids[row] * inv_freq[k]` in fp32.
2. Reduces `|freq|` across the warp via `__shfl_xor_sync` to obtain
   `row_amax`. The reference's BlockWise1x128 scale operates on a row
   padded to 128 with zeros; the zeros don't change the amax, so we
   skip materialising the padded vector entirely and reduce over the
   32 real values directly.
3. Computes `scale = max(row_amax / 448, 1e-12)` and applies it: divide
   `freq[k]` by scale, clamp to `[-448, 448]`, cast to `__nv_fp8_e4m3`
   (round-to-nearest), cast back to fp32, multiply by scale. This is
   the faithful BlockWise1x128 quantize/dequantize that the reference
   uses inside the inlined `BlockwiseScaler` (NOT a precision
   substitution).
4. Computes `cos(freq_final) * mscale` and `sin(freq_final) * mscale`,
   downcasts to bf16, and writes the result to two slots per row:
   `cos_emb[row, k]` and `cos_emb[row, k + 32]`. The reference does
   `emb = concat([freqs_final, freqs_final], -1)` so each lane's
   value populates the lower half AND the upper half of the 64-wide
   row.

`mscale_combined = yarn_get_mscale(scaling, mscale) / yarn_get_mscale(scaling, mscale_all_dim)`
is computed on the host (one ratio per kernel call) and passed as a
single fp32 to the kernel.

### Local sm_120 harness result (16 / 16 workloads PASSED)

| Workload | Status | speedup_factor |
|---|---|---|
| 0  | PASSED | 8.2-19.8 (per-workload spread; min 8.189, max 19.836) |

All 16 workloads pass with `speedup_factor` ranging from 8.189x to
19.836x. Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120).
Evaluation ran through the native `sol-execbench` CLI (no Docker); the
cache / manifest identify this run by `native_env_hash` (see
`scripts/baselines_harness/native_env.py`).

## Architecture-coupling caveats

The local 5060 (sm_120) compiles and smoke-runs this baseline; final SOL
scoring is on B200 (sm_100a). Differences worth noting for this pattern:

- HBM3e vs GDDR7 bandwidth: this kernel is compute-bound on the
  per-position cos/sin transcendentals, NOT bandwidth-bound (output
  size is `seq_len * 64 * 2 bytes * 2` for cos + sin; input is just
  `seq_len * 8 bytes`). B200's higher HBM3e bandwidth does not change
  the operation's cost meaningfully; the speedup vs the reference is
  driven by avoiding the reference's intermediate
  `[seq_len, 128]` padded buffer and the FP8 round-trip allocation.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are not used
  here; per-row reduction over 32 lanes fits trivially within a single
  warp.
- L2 carveout: irrelevant for this kernel — total working set per row
  is 128 B of frequencies plus the 128-B inv_freq table, all of which
  fits in L1.
- TCGen05 / TMEM divergence: this kernel does not touch 5th-generation
  tensor cores or the TMEM register surface; the math is pure SIMT
  FP32 transcendentals + FP8 e4m3 cast. The faithful FP8
  quantize/dequantize uses `__nv_fp8_e4m3` casts in CUDA (not tensor
  cores), so sm_100a vs sm_120 code-gen is identical.

## Forbidden-surface compliance

The kernel does not use NCU, clock-locking, user CUDA streams,
`torch.jit.fork`, `data_ptr`-keyed caches, monkey-patched
`Event.elapsed_time`, or CUDA Graphs. Launches go through
`at::cuda::getCurrentCUDAStream()` per the default-stream discipline.
