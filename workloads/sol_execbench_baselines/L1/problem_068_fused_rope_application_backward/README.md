# problem_068_fused_rope_application_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 backward pass for fused RoPE application: computes `grad_q` and `grad_k` per-element via the inverse rotation, plus `grad_cos` and `grad_sin` via reduction over the head axis. Four DPS outputs with mixed shapes. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_068_fused_rope_application_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/068_fused_rope_application_backward/`                       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements the backward pass for
fused RoPE application as two kernels driven by one DPS
entry.

Algorithm:
- Forward (reference):
  `output[..., d] = x[..., d] * cos[..., d] + rotate_half(x)[..., d] * sin[..., d]`
- Backward gradients:
  - `grad_x = grad_output * cos - rotate_half(grad_output) * sin`
  - `grad_cos = sum_h(grad_q_embed * q) + sum_h_kv(grad_k_embed * k)`
  - `grad_sin = sum_h(grad_q_embed * rotate_half(q)) + sum_h_kv(grad_k_embed * rotate_half(k))`

`rotate_half(x)[d] = -x[d + 64]` for `d ∈ [0, 64)` and
`x[d - 64]` for `d ∈ [64, 128)`. `head_dim = 128` and
`half_head_dim = 64` are pinned. `num_heads`,
`num_kv_heads`, `batch_size`, and `seq_len` are
workload-variable.

Two-kernel implementation:

### Kernel A — `problem_068_grad_x_kernel` (per-element)

Computes `grad_x` for either the query or key gradient stream.
- Grid: `(seq_len, num_heads_axis, batch_size)` where
  `num_heads_axis` is taken from the input tensor's
  `size(1)`.
- `BLOCK_THREADS = head_dim = 128` (one thread per `d`).
- Each thread reads `g = grad_x_embed[b, h, s, d]` and the
  rotate-half partner element
  `g_partner = grad_x_embed[b, h, s, partner_d]` (with
  `partner_d` and `partner_sign` derived from `d` via the
  pinned half_head_dim split).
- Writes
  `grad_x[b, h, s, d] = g * cos[b, s, d] - partner_sign * g_partner * sin[b, s, d]`.

Invoked twice from the DPS entry:
1. `(grad_q_embed, cos, sin, grad_q)` — num_heads_axis = num_heads.
2. `(grad_k_embed, cos, sin, grad_k)` — num_heads_axis = num_kv_heads.

### Kernel B — `problem_068_grad_cos_sin_kernel` (per-(b, s) reduction-over-heads)

Computes `grad_cos[b, s, d]` and `grad_sin[b, s, d]` for the
non-head-axis tensors.
- Grid: `(seq_len, batch_size)`. One CTA per `(b, s)` row.
- `BLOCK_THREADS = head_dim = 128` (one thread per `d`).
- Each thread maintains two fp32 accumulators
  (`acc_cos`, `acc_sin`) and stride-loops over the head
  axis:
  - For `h in [0, num_heads)`:
    - Read `g = grad_q_embed[b, h, s, d]`,
      `x = q[b, h, s, d]`,
      `x_partner = q[b, h, s, partner_d]`.
    - `acc_cos += g * x`.
    - `acc_sin += g * partner_sign * x_partner`.
  - For `h_kv in [0, num_kv_heads)`:
    - Same accumulation pattern using `grad_k_embed` and
      `k` with `num_kv_heads` strides.
- Writes
  `grad_cos[b, s, d] = acc_cos`,
  `grad_sin[b, s, d] = acc_sin`.

Invoked once from the DPS entry.

`main.cpp` pins `head_dim == 128` via `TORCH_CHECK`,
validates 4D q/k/grad tensors, 3D cos/sin/grad_cos/grad_sin
tensors, shape consistency across the (b, s) and (b, s, d)
projections, fp32 dtype, CUDA + contiguous. DPS entry
signature has 6 inputs and 4 outputs.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  Kernel A is bandwidth-bound on grad_x_embed and grad_x
  streams (write traffic dominates). Kernel B's read
  traffic scales linearly with `num_heads + num_kv_heads`,
  so its bandwidth requirement grows with the head count
  while the per-CTA work stays bounded.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope; both kernels use only per-thread register
  state.
- L2 carveout / partition: the cos/sin tables are
  `[B, S, 128] fp32`. For batch_size=1 the L2 footprint is
  modest. The Q and K input streams dominate Kernel B's
  read traffic: per-CTA H+H_kv loads at each `d`, summing
  to `(num_heads + num_kv_heads) * head_dim` per (b, s).
  This is workload-variable and exposes a per-head reuse
  pattern to the NCU labels.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel uses
  no tensor cores.
