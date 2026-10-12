# problem_019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 backward pass for Vision 3D RoPE with spatial-merge indexing. Three DPS outputs (`grad_q`, `grad_k`, `grad_embeddings`). cos/sin tables are derived in-kernel from a single `embeddings[seq_len, head_dim]` tensor via `sinf` / `cosf`. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward/` |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements the backward pass for
Vision 3D RoPE with spatial-merge indexing as two kernels
driven by one DPS entry. Reuses the
`problem_068_fused_rope_application_backward` structure
with the twist that cos/sin tables are derived in-kernel
from a single `embeddings[seq_len, head_dim]` tensor via
`sinf` / `cosf` (rather than being passed as separate
`cos` / `sin` inputs).

`head_dim = 128` and `num_heads = 16` are pinned by the
problem definition; `half_head_dim = 64`. `seq_len` is
workload-variable.

### Kernel A — `problem_019_grad_x_kernel` (per-element)

Computes `grad_q` or `grad_k`. Grid: `(num_heads, seq_len)`.
`BLOCK_THREADS = head_dim = 128` (one thread per `d`).

Per element:
- Resolve `(partner_d, partner_sign_fwd)`:
  - `d < 64`: `partner_d = d + 64`, `partner_sign_fwd = -1`.
  - `d >= 64`: `partner_d = d - 64`, `partner_sign_fwd = +1`.
- Read `g_d = grad_x_embed[s, h, d]`,
  `g_partner = grad_x_embed[s, h, partner_d]`,
  `emb_d = embeddings[s, d]`,
  `emb_par = embeddings[s, partner_d]`.
- Compute `cos_d = cosf(emb_d)`, `sin_par = sinf(emb_par)`.
- Write
  `grad_x[s, h, d] = g_d * cos_d - partner_sign_fwd * g_partner * sin_par`.

Invoked TWICE from `run()` — once for the Q-side, once for
the K-side — both sharing the same launcher template.

### Kernel B — `problem_019_grad_emb_kernel` (per-(s) reduction)

Grid: `(seq_len,)`. One CTA per `s`.
`BLOCK_THREADS = 256` stride-loops over `head_dim = 128`.

Per `(s, d)`:
- Maintain two fp32 accumulators `acc_cos`, `acc_sin`.
- Iterate `h in [0, num_heads)`:
  - Read `g_q`, `g_k`, `q_d`, `k_d`, `q_par`, `k_par` from
    the corresponding tensor positions.
  - `acc_cos += g_q * q_d + g_k * k_d`
  - `acc_sin += g_q * q_par + g_k * k_par`
- Scale `acc_sin *= partner_sign_fwd` (factor out from the
  per-head loop).
- Read `emb_d = embeddings[s, d]`, compute
  `cos_e_d = cosf(emb_d)`, `sin_e_d = sinf(emb_d)`.
- Write
  `grad_embeddings[s, d] = -acc_cos * sin_e_d + acc_sin * cos_e_d`.

Invoked ONCE from `run()`.

`main.cpp` pins `head_dim == 128` and `num_heads == 16`
via `TORCH_CHECK`; validates 3D q/k/grads, 2D embeddings,
shape consistency, fp32 dtype, CUDA + contiguous. DPS entry
signature
`run(grad_q_embed, grad_k_embed, q, k, embeddings,
grad_q, grad_k, grad_embeddings)` — 5 inputs and 3 outputs.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on.
Final SOL scoring is on B200 (sm_100a). Differences worth
noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  Kernel A is bandwidth-bound on the grad_x_embed and
  grad_x streams. Kernel B reads grad_q_embed + q +
  grad_k_embed + k per-head, so its read traffic scales
  linearly with num_heads — at num_heads=16 the read
  pattern is moderate.
- DSMEM and cluster-cooperative reduction (sm_100a-only)
  are out of scope.
- L2 carveout / partition: the embeddings tile is
  `seq_len * 128 * 4 B` — small enough to fit comfortably
  in L1 for typical seq_len values.
- TCGen05 / TMEM divergence: this kernel does not use
  tensor cores.
