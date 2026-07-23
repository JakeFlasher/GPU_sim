# problem_021_cross_attention_text_video_conditioning_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Cross-attention text-video conditioning backward (10 gradient outputs)                                 |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_021_cross_attention_text_video_conditioning_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/021_cross_attention_text_video_conditioning_backward/`                |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Profiled-local-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                       |
| Local speedup vs reference  | 0.85 - 1.15x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes ten outputs through eight fp32 matmul dispatches (4 2D
`cublasSgemm` grad_input pairs + 4 4D `cublasSgemmStridedBatched`
attention-backward pairs) plus a softmax backward and a scale
backward expressed as ATen elementwise / reduction ops. Queries
come from the video latents; keys and values come from the text
embeddings; the gradient w.r.t. text_embeddings accumulates from
both the K and V projection backwards.

1. **Forward recomputation** (mirrors the reference exactly):
   - `queries = F.linear(video_latents, query_weight, query_bias).view(B, N_v, H, d).transpose(1, 2)`
   - `keys = F.linear(text_embeddings, key_weight, key_bias).view(B, N_t, H, d).transpose(1, 2)`
   - `values = F.linear(text_embeddings, value_weight, value_bias).view(B, N_t, H, d).transpose(1, 2)`
   - `attn_scores = (queries @ keys.transpose(-2, -1)) * scale`
   - `attn_probs = softmax(attn_scores, dim=-1)`
   - `context = (attn_probs @ values).transpose(1, 2).contiguous().view(B, N_v, D)`
2. **Output projection backward**:
   - `grad_context = at::matmul(grad_output, output_weight)` (cublasSgemm 2D fp32 OP_N OP_N — the verified-family canonical binding).
   - `grad_output_weight = grad_output_2d.t() @ context_2d` (cublasSgemm 2D fp32 OP_T OP_N).
   - `grad_output_bias = grad_output.sum(kBatchSeqDims)`.
3. **Head merge + attention-application backward**:
   - `grad_attention_probs = grad_context_heads @ values.transpose(-2, -1)` (4D cublasSgemmStridedBatched).
   - `grad_values_heads = attn_probs.transpose(-2, -1) @ grad_context_heads` (same family).
4. **Softmax backward** (standard formula):
   - `sum_grad = (grad_attention_probs * attn_probs).sum(dim=-1, keepdim=true)`.
   - `grad_attention_scores = attn_probs * (grad_attention_probs - sum_grad)`.
5. **Scale backward**: `grad_attention_scores *= scale`.
6. **Scaled-dot-product backward**:
   - `grad_queries_heads = grad_attention_scores @ keys` (4D batched).
   - `grad_keys_heads = grad_attention_scores.transpose(-2, -1) @ queries` (4D batched).
7. **Head split + Q/K/V projection backward**:
   - Each `grad_q/k/v_heads.transpose(1, 2).contiguous().view(B, *, D)`.
   - `grad_video_latents = at::matmul(grad_queries, query_weight)` (cublasSgemm 2D fp32 OP_N OP_N).
   - `grad_query_weight = grad_queries_2d.t() @ video_latents_2d` (cublasSgemm 2D fp32 OP_T OP_N).
   - `grad_query_bias = grad_queries.sum(kBatchSeqDims)`.
   - Analogous K and V projection backwards on `text_embeddings`.
8. **Accumulate** `grad_text_embeddings = grad_text_from_keys + grad_text_from_values`.

Const axes pinned as `constexpr`: `num_heads = 16`,
`head_dim = 64`, `hidden_size = 1024`. The variable axes
`N_v = num_video_tokens` and `N_t = num_text_tokens` are
workload-driven and recovered from input tensor shapes at
runtime. Every input AND every DPS output is shape-checked
against the const axes plus the recovered `B`, `N_v`, `N_t`
(twenty-one TORCH_CHECK rank+size assertions in total).
CUDA guard from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The dispatch-family pin is `cublasSgemm_basic_fp32` (verified
in `.humanize/corpus-expansion/verified_dispatch_families.toml`).
The four projection grad_input dispatches use `at::matmul(g, W)`
directly — binding cuBLAS as OP_N OP_N, the canonical
verified-family signature. The four grad_weight dispatches use
`at::matmul(g_2d.transpose(0, 1), x_2d)` — binding cuBLAS as
OP_T OP_N (also in the verified family). The 4D batched matmuls
dispatch through `cublasSgemmStridedBatched` (fp32).

The axis-list reduction args (`grad_output.sum`,
`grad_queries.sum`, `grad_keys.sum`, `grad_values.sum`) are
wrapped in `std::vector<int64_t> kBatchSeqDims = {0, 1}` so
the int-axis overload of `Tensor::sum` binds unambiguously
against the named-axis overload under current Torch headers.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the cross-attention backward chain reads four
  large weight matrices (each 1024x1024 = 4 MiB fp32, total 16 MiB) plus
  the 4D attention activations (`attn_probs` at `[B, 16, N_v, N_t]`
  becomes large at long video-token sequences -- workload 11 has
  B=2, N_v=4096, N_t=77 giving ~40 MiB); the 3090 Ti's bandwidth
  ceiling makes local wall-time a weak proxy for the B200 SOL Score.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used here;
  the softmax-backward sum reduction could exploit cluster-coop on B200
  in a future tuned variant.
- L2 carveout: B200's larger L2 absorbs the four 4 MiB projection weights
  more effectively across the chain; the 3090 Ti's smaller L2 evicts
  between dispatches at long sequences. GDDR7 does not apply here (the
  3090 Ti uses GDDR6X).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the fp32 sgemm path is the
  conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences.
