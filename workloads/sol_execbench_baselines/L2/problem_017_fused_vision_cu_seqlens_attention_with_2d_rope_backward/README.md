# problem_017_fused_vision_cu_seqlens_attention_with_2d_rope_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Fused vision cu_seqlens variable-length attention backward with 2D rotary position embeddings (5 gradient outputs) |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_017_fused_vision_cu_seqlens_attention_with_2d_rope_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/017_fused_vision_cu_seqlens_attention_with_2d_rope_backward/`         |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Profiled-local-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                       |
| Local speedup vs reference  | <to be filled after native run; diagnostic only>                                                       |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The
backward computes five outputs through four fp32 2D
`cublasSgemm` projection-matmul dispatches plus `N` (=
`num_sequences`) groups of fp32 4D `cublasSgemmStridedBatched`
per-sequence attention matmuls, plus a 2D RoPE backward
(via `rotate_half_inverse`) and softmax/scale backwards
expressed as ATen elementwise / reduction ops. The
cu_seqlens-driven variable-length attention is the load-bearing
structural feature: a brief host-sync reads the int32
cumulative-length tensor to derive per-sequence lengths, then
each sequence's Q/K/V slice runs independently through the
attention chain.

1. **Forward recomputation** (mirrors the reference exactly):
   - `qkv = at::linear(hidden_states, qkv_weight, qkv_bias)`.
   - Reshape to `[seq_len, 3, num_heads, head_dim]`, permute,
     unbind to `(query, key, value)`.
   - `cos_expanded = cos.unsqueeze(-2)` ;
     `sin_expanded = sin.unsqueeze(-2)`.
   - 2D RoPE on Q and K:
     `q_rope = q * cos_expanded + rotate_half(q) * sin_expanded`;
     analogous for K. V passes through (no RoPE on V).
   - Transpose + unsqueeze: `[seq_len, H, d] -> [1, H, seq_len, d]`.
   - Host-sync: read `cu_seqlens` to derive `lengths[i] =
     cu_seqlens[i+1] - cu_seqlens[i]`.
   - Per-sequence loop (`for i in range(num_sequences)`):
     `attn_weights = softmax((q @ k^T) * scaling, dim=-1)`;
     `attn_output = attn_weights @ v`. Save each
     `attn_weights` for the backward.
   - Concatenate `attn_outputs` along the seq dim; reshape to
     `[seq_len, hidden_size]`.
2. **Output projection backward**:
   - `grad_proj_bias = grad_output.sum(dim=0)`.
   - `grad_proj_weight = grad_output.t() @ attn_output_reshaped`
     (cublasSgemm 2D fp32 OP_T OP_N).
   - `grad_attn_output = grad_output @ proj_weight`
     (cublasSgemm 2D fp32 OP_N OP_N — verified-family
     canonical binding).
   - Reshape `[seq_len, hidden_size] -> [seq_len, H, d] ->
     [1, H, seq_len, d]`.
3. **Per-sequence attention backward** (same `lengths` split):
   - `grad_attn_weights = grad_attn_out @ v.transpose(2, 3)`
     (4D cublasSgemmStridedBatched).
   - `grad_v = attn_weights.transpose(2, 3) @ grad_attn_out`
     (same family).
   - Softmax backward (standard formula):
     `sum_grad = (grad_attn_weights * attn_weights)
     .sum(dim=-1, keepdim=true)`;
     `grad_attn_scores = attn_weights *
     (grad_attn_weights - sum_grad)`.
   - Scale backward: `grad_attn_scores *= scaling`.
   - `grad_q = grad_attn_scores @ k`;
     `grad_k = grad_attn_scores.transpose(2, 3) @ q`.
4. **Concatenate** per-sequence q/k/v gradients;
   `squeeze(0).transpose(0, 1) -> [seq_len, H, d]`.
5. **RoPE backward** (using `rotate_half_inverse`):
   - `grad_q_pre_rope = grad_q * cos_expanded
     + rotate_half_inverse(grad_q * sin_expanded)`.
   - `grad_k_pre_rope = grad_k * cos_expanded
     + rotate_half_inverse(grad_k * sin_expanded)`.
   - `grad_v` passes through.
6. **Stack** into `[seq_len, 3, H, d]`; reshape to
   `[seq_len, qkv_out_size = 10752]`.
7. **QKV projection backward**:
   - `grad_hidden_states = grad_qkv @ qkv_weight`
     (cublasSgemm 2D fp32 OP_N OP_N — verified family).
   - `grad_qkv_weight = grad_qkv.t() @ hidden_states`
     (cublasSgemm 2D fp32 OP_T OP_N — verified family).
   - `grad_qkv_bias = grad_qkv.sum(dim=0)`.

Const axes pinned as `constexpr`: `hidden_size = 3584`,
`num_heads = 28`, `head_dim = 128`,
`qkv_out_size = 3 * hidden_size = 10752`. Variable axes
`total_seq_len` and `num_sequences` are workload-driven
and recovered from input tensor shapes at runtime
(`num_seqlens = num_sequences + 1`). Every input AND every
DPS output is explicitly shape-checked against the const +
recovered axes (15 TORCH_CHECK rank+size assertions total).
CUDA guard from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The dispatch-family pin is `cublasSgemm_basic_fp32`
(verified in `.humanize/corpus-expansion/verified_dispatch_families.toml`).
All four projection 2D matmul dispatches bind the verified
family. The 4D per-sequence batched matmuls (4 per
sequence: `q @ k^T`, `attn_weights @ v` on the forward;
`grad_attn_out @ v^T`, `attn_weights^T @ grad_attn_out`,
`grad_attn_scores @ k`, `grad_attn_scores^T @ q` on the
backward) bind `cublasSgemmStridedBatched` with
`batch = num_heads = 28`.

The cu_seqlens host-sync is a known design point: the
reference uses `lengths.tolist()` which forces a CPU
read. The C++ implementation calls
`cu_seqlens.to(torch::kCPU).data_ptr<int32_t>()` once per
launch — the same shape as the reference. This is the
load-bearing site for the per-sequence iteration count.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the cu_seqlens-split attention chain reads
  large QKV / proj weight matrices (`qkv_weight 10752x3584` = 144 MiB;
  `proj_weight 3584x3584` = 49 MiB) and per-sequence `attn_weights`
  tensors that total `sum(len_i^2) * num_heads` fp32 entries (workload 11
  with `total_seq_len = 8192` and `num_sequences = 32` averaging
  `len_i ~ 256` produces ~232 MiB of attention activations); the 3090 Ti's
  bandwidth ceiling makes local wall-time a weak proxy for the B200 SOL
  Score.
- The per-sequence Python-style loop in C++ launches `num_sequences * 4`
  individual 4D batched matmuls (per launch: 1 forward attn_scores +
  1 forward attn_output + 4 backward matmuls). A fused
  variable-length-attention kernel (FlashAttention-style) on B200 with
  TMA + TCGen05 could absorb the per-sequence boundary into one launch,
  but the template path stays at the conservative ATen dispatch.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used here;
  the softmax-backward sum could exploit cluster-coop on B200 in a
  future tuned variant.
- L2 carveout: B200's larger L2 absorbs the two big weight matrices
  more effectively; the 3090 Ti's smaller L2 evicts between dispatches
  at long sequences. GDDR7 does not apply here (the 3090 Ti uses
  GDDR6X).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the fp32 sgemm path is the
  conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences.
- cu_seqlens host-sync overhead: the brief `cu_seqlens.to(cpu)` read is
  a synchronization point. On the 3090 Ti at 16-32 sequence-per-batch
  it is negligible; on B200 with much faster compute it may show up
  proportionally larger and motivate a future fused-launch variant.
