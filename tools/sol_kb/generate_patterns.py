#!/usr/bin/env python3
"""
generate_patterns.py — emit the 20 missing pattern cards.

`tools/sol_baselines_scaffold/sol_pattern_mapping.md` lists 20 `tbd_*`
pattern families covering 120 SOL-ExecBench problems. This script writes
each family's pattern card to `.claude/knowledge/patterns/<name>.md`
using a shared template and a per-pattern data block (PATTERN_SPECS
below). Every emitted card satisfies the hardened linter (`tools/sol_kb/
lint_cards.py`): full 7-required-field frontmatter, every Rule-1-through-8
canonical phrase in `Out of scope`, both `sm_120` and `sm_100a` tokens
plus the CLAUDE.md citation and the evidence-label vocabulary phrase in
the body.

Usage:  python3 tools/sol_kb/generate_patterns.py [--root REPO_ROOT]
        [--force]  # overwrite existing pattern cards
        [--list]   # only print the names that would be written

The pattern card body is intentionally compact: it states the recipe,
links to the canonical CUTLASS / CUDA C++ building blocks, calls out
sm_120 vs sm_100a divergences, and points at the spec corpus. It is NOT
a re-implementation of the upstream NVIDIA docs — the supporting-sources
links carry the deep references.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared boilerplate. Keep in sync with `.claude/knowledge/_template.md`.
# ---------------------------------------------------------------------------

FORBIDDEN_SURFACE_BLOCK = """\
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5090 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8)."""

ARCH_PORTABILITY_TEMPLATE = """\
## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5090, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. {arch_notes}

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card."""

DEFAULT_ARCH_NOTES = (
    "Local pass on sm_120 verifies compile + correctness only; sm_100a is "
    "required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and "
    "HBM3e bandwidth assumptions. Build twice when in doubt; the B200 "
    "problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`."
)


# ---------------------------------------------------------------------------
# Per-pattern specifications. Names match the `tbd_<canonical>` stems in
# `tools/sol_baselines_scaffold/sol_pattern_mapping.md` (drop the `tbd_` prefix).
# ---------------------------------------------------------------------------

PATTERN_SPECS: list[dict] = [
    {
        "name": "attention_backward",
        "use_when": (
            "implementing the backward pass for transformer attention — the "
            "Q/K/V matmul gradients, softmax/dropout backward, and value-output "
            "matmul backward — used inside training-loop fused kernels. Covers "
            "BF16 / FP16 / FP32 inputs with FP32 reduction, with or without "
            "GQA / dropout / scaling. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/device/gemm_universal.h; "
            "cutlass/include/cute/atom/mma_atom.hpp; "
            "baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "attention; attention-backward; backward; softmax-backward; "
            "qkv-backward; matmul-backward; bf16; fp16"
        ),
        "recipe": (
            "Attention backward decomposes into three GEMM-shaped gradients "
            "plus the softmax / dropout backprop:\n\n"
            "```\n"
            "dV = (P^T)        @ dO        // [seq, head_dim]\n"
            "dP = dO            @ V^T       // [seq, seq]; followed by softmax-backward\n"
            "dQ = (dS · scale)  @ K         // [seq, head_dim]\n"
            "dK = (dS · scale)^T @ Q        // [seq, head_dim]\n"
            "```\n"
            "where `S = QK^T`, `P = softmax(S)`, `dS = dropout_backward(softmax_backward(dP, P))`. "
            "Each gradient maps onto `cutlass::gemm::device::GemmUniversal` with the "
            "appropriate transpose; the softmax-backward is a streaming reduce-update "
            "kernel that reads `P` (or recomputes it) and produces `dS`."
        ),
        "building_blocks": (
            "- `cutlass::gemm::device::GemmUniversal` with sm_100a collective builder for `dV`, `dQ`, `dK`.\n"
            "- `cutlass::epilogue::collective::sm100_epilogue_builder` to fuse the `scale` factor and any residual add into the gradient output.\n"
            "- A custom `__global__` softmax-backward kernel that walks `P` row-wise; accumulate in FP32.\n"
            "- `cutlass::TiledMma` selection matches the forward kernel's MMA shape so the backward executes on the same `tcgen05.mma` schedule on sm_100a."
        ),
        "pitfalls": (
            "- **Precision downgrade is a Rule-8 reward-hack** — keep softmax-backward FP32 even when forward is BF16/FP16.\n"
            "- **`data_ptr` caching of `P`** (Rule 6) — the harness shifts inputs every iteration; recompute or revalidate, do not memoise.\n"
            "- **Default-stream discipline** (Rule 7) — never split the four gradient launches across user streams."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md`; this pattern's primary_pattern row in that table enumerates the dossiers that link here.",
    },
    {
        "name": "attention_core",
        "use_when": (
            "implementing the forward attention core `softmax(QK^T / sqrt(d)) @ V` "
            "without RoPE / paged-KV / dropout fusions. Covers BF16 / FP16 inputs "
            "with FP32 softmax reduction, dense (not paged) KV, optional "
            "additive masking, head_dim ∈ {64, 128, 192, 256}. Targets B200 sm_100a "
            "with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; "
            "cutlass/include/cute/atom/mma_atom.hpp; "
            "baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "attention; attention-core; softmax; qk-matmul; av-matmul; "
            "dense-attention; bf16; fp16"
        ),
        "recipe": (
            "```\n"
            "S = (Q @ K^T) * scale         // [seq_q, seq_kv]\n"
            "P = softmax_rowwise(S)        // FP32 reduction; cast back to input dtype\n"
            "O = P @ V                     // [seq_q, head_dim]\n"
            "```\n"
            "Streaming (Flash-style) variants compute `P` and `O` in tiles, keeping a "
            "running max / sum / output to avoid materialising the full `[seq_q, seq_kv]` "
            "matrix. For B200, the tile size is bounded by shared memory + TMEM; for "
            "sm_120 the `KernelMultistage` schedule uses regular registers."
        ),
        "building_blocks": (
            "- Two `cutlass::gemm` collectives back-to-back (`QK^T` then `AV`).\n"
            "- A producer warp that stages `K, V` tiles via TMA on sm_100a.\n"
            "- Online-softmax accumulators (running max, running sum, running output) in FP32.\n"
            "- `cutlass::epilogue::fusion` callbacks for the trailing scale + dtype cast."
        ),
        "pitfalls": (
            "- **Softmax in BF16** loses precision around `S ≈ 0`; keep the reduction FP32 (Rule 8).\n"
            "- **`data_ptr` on K/V** caching breaks under the harness's 256-byte input shift (Rule 6).\n"
            "- **Multi-stream QK^T / AV** to overlap is a Rule-7 violation."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `attention_core` rows.",
    },
    {
        "name": "vision_audio_conv_blocks",
        "use_when": (
            "implementing fused convolution + normalization + activation + residual "
            "blocks from vision / audio model families (ResNet, ViT patch embed, VAE "
            "encoder/decoder, audio Conv1D). Covers Conv2D / Conv3D / Conv1D with "
            "BF16 / FP16 / FP32 weights, GroupNorm / LayerNorm / RMSNorm fusion, "
            "SiLU / GELU / ReLU activations, optional residual add. Targets B200 "
            "sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/conv/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/conv/device/conv_universal_adapter.hpp; "
            "cutlass/include/cutlass/conv/collective/builders/sm100_umma_builder.inl; "
            "baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "conv; conv2d; conv3d; conv1d; vision; audio; vae; resnet; "
            "implicit-gemm; groupnorm; layernorm; silu; gelu; bf16; fp16"
        ),
        "recipe": (
            "Conv blocks decompose into:\n\n"
            "1. Implicit-GEMM Conv via `cutlass::conv::device::ConvUniversal` (sm_100a) "
            "or the older `ImplicitGemmConvolution` interface for compatibility.\n"
            "2. Fused norm / activation / residual via `cutlass::epilogue::fusion` "
            "callbacks chained after the implicit-GEMM accumulator.\n"
            "3. For 3D conv (VAE), use `cutlass::conv::Conv3dProblemSize` and "
            "the `kAnalyticIDIGSwizzler` for the strided-GEMM transform."
        ),
        "building_blocks": (
            "- `cutlass::conv::device::ConvUniversal` with `cutlass::conv::Operator::kFprop` (forward) / `kDgrad` / `kWgrad`.\n"
            "- `cutlass::conv::collective::ConvCollective` builder using `sm100_implicit_gemm_umma_builder` for B200.\n"
            "- `cutlass::epilogue::fusion::sm100_callbacks` chain to merge `groupnorm(silu(...))` into the conv output.\n"
            "- For Conv3D, `cutlass::conv::ConvolutionMode::kCrossCorrelation` + 3D problem size."
        ),
        "pitfalls": (
            "- **Layout mismatch.** NCHW vs NHWC matters for the implicit-GEMM swizzler; verify against the spec's reference PyTorch.\n"
            "- **GroupNorm in BF16** loses bits in the variance reduction; keep FP32 (Rule 8).\n"
            "- **CUDA Graphs for the multi-launch sequence** is a Rule-7 reward-hack."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `vision_audio_conv_blocks` rows.",
    },
    {
        "name": "hyena_mamba_ssm",
        "use_when": (
            "implementing selective-scan / state-space model (SSM) kernels — Mamba "
            "selective scan, Hyena long-convolution + FFT, Mamba2 SSD. Covers BF16 "
            "/ FP16 / FP32 inputs with FP32 state accumulation and chunked / parallel "
            "scan algorithms. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cute/algorithm/copy.hpp",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cuda_guide/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cute/atom/copy_atom.hpp; "
            "cutlass/include/cute/algorithm/copy.hpp; "
            "baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "ssm; mamba; mamba2; hyena; selective-scan; ssd; "
            "state-space; scan; recurrence; bf16; fp16"
        ),
        "recipe": (
            "Selective scan:\n\n"
            "```\n"
            "h_t = exp(Δ_t · A) · h_{t-1} + (Δ_t · B_t) · x_t        // state update\n"
            "y_t = C_t · h_t + D · x_t                                 // output projection\n"
            "```\n"
            "with `A` diagonal, `B_t, C_t` input-dependent, `Δ_t` softplus-gated. "
            "Implementations:\n\n"
            "1. **Chunk-parallel scan.** Within a chunk run the recurrence sequentially "
            "in registers; between chunks combine via prefix-sum-of-`(exp(ΔA), Δ·B·x)` pairs.\n"
            "2. **Hyena.** The convolution branch becomes an FFT (`cuFFT`) when the kernel "
            "is long; the gating branch is a standard MLP.\n"
            "3. **Mamba2 SSD** reinterprets the scan as a structured matrix-multiply, "
            "amenable to `cutlass::gemm`."
        ),
        "building_blocks": (
            "- A `__global__` chunk-parallel scan kernel in CUDA C++; FP32 state accumulators.\n"
            "- `cutlass::gemm::device::GemmUniversal` for Mamba2 SSD's structured-matrix step.\n"
            "- `cuFFT` plan caching (host-side) for Hyena's long-convolution branch."
        ),
        "pitfalls": (
            "- **State precision** — accumulating `h_t` in BF16 diverges over long sequences; FP32 is mandatory.\n"
            "- **`data_ptr` on the state buffer** breaks under the 256-byte shift (Rule 6).\n"
            "- **FFT plan cached by tensor address** silently goes stale; key by shape only."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `hyena_mamba_ssm` rows.",
    },
    {
        "name": "norm_layernorm_groupnorm",
        "use_when": (
            "implementing LayerNorm / GroupNorm kernels (not RMSNorm — see "
            "`rmsnorm.md` or `fused_add_rmsnorm.md` for that family). Covers BF16 / "
            "FP16 / FP32 inputs with FP32 reduction, optional affine weight + bias, "
            "with or without fused residual / activation. Targets B200 sm_100a with "
            "sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cute/algorithm/copy.hpp",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cuda_guide/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cute/atom/copy_atom.hpp; "
            "baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "layernorm; groupnorm; norm; normalization; affine; "
            "welford; reduction; bf16; fp16"
        ),
        "recipe": (
            "Two-pass numerically-stable Welford variance + standard affine:\n\n"
            "```\n"
            "(mean, var) = Welford(x_row)                  // FP32 reduction\n"
            "x_hat = (x_row - mean) / sqrt(var + eps)\n"
            "y     = x_hat * weight + bias                 // affine, optional\n"
            "```\n"
            "GroupNorm partitions channels into groups and runs the same reduction "
            "per (batch, group). For B200, cluster-cooperative reduction across CTAs "
            "via DSMEM is available for very wide normalization axes."
        ),
        "building_blocks": (
            "- A `__global__` two-pass reduction kernel (or Welford one-pass) in CUDA C++; FP32 accumulators.\n"
            "- `cute::TiledCopy` for the affine-parameter load on sm_100a.\n"
            "- Cluster-cooperative reduction primitives (sm_100a only) for very wide axes."
        ),
        "pitfalls": (
            "- **Variance in BF16** loses precision (catastrophic cancellation); FP32 reduction is contractual (Rule 8).\n"
            "- **Caching the mean / var** across iterations via `data_ptr` (Rule 6) — the harness shifts inputs.\n"
            "- **`atomicAdd` for cross-CTA reduce** without proper coalescing burns L2 bandwidth."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `norm_layernorm_groupnorm` rows.",
    },
    {
        "name": "transformer_layer",
        "use_when": (
            "implementing a full transformer decoder / encoder layer as a single fused "
            "kernel-graph (multi-op L2 problem). Composes attention, RoPE, RMSNorm, "
            "residual add, MLP. Variants: pre-norm vs post-norm, dense vs MoE FFN, "
            "BF16 vs FP8. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/examples/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; "
            "cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp; "
            "baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "transformer; transformer-layer; decoder-layer; encoder-layer; "
            "pre-norm; post-norm; bf16; fp8"
        ),
        "recipe": (
            "Canonical pre-norm decoder layer:\n\n"
            "```\n"
            "x'  = x + Attention(RMSNorm(x))             // attention residual\n"
            "y   = x' + MLP(RMSNorm(x'))                 // MLP residual\n"
            "```\n"
            "Each sub-block composes a pattern card already in the atlas:\n"
            "- Attention: `attention_core` (or `rope_attention` if rotational, or `gqa_paged_decode` for paged).\n"
            "- RMSNorm: `fused_add_rmsnorm` (fuses the residual add).\n"
            "- MLP: `swiglu_mlp` (or `mlp_gelu_geglu` for non-gated variants).\n"
            "- MoE variant: replace MLP with `moe_dispatch`."
        ),
        "building_blocks": (
            "- Compose the sub-pattern cards' recipes; this card carries the *layer-level* "
            "ordering and the choice of fusion strategy (kernel boundaries).\n"
            "- For B200, the layer can fold attention + RMSNorm + residual into a single "
            "CUTLASS schedule via epilogue chaining; for sm_120 split into 2-3 launches."
        ),
        "pitfalls": (
            "- **Pre-norm vs post-norm** is spec-dependent; verify against the reference PyTorch in the spec.\n"
            "- **Stream split across sub-blocks** is a Rule-7 reward-hack.\n"
            "- **Residual `data_ptr` caching** across the two residuals (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `transformer_layer` rows.",
    },
    {
        "name": "rope_embedding_tables",
        "use_when": (
            "computing or refreshing the RoPE cos / sin frequency tables — `theta_i = "
            "base^(-2i/head_dim)`, `cos_p_i / sin_p_i` for positions `p`. Distinct from "
            "the rotation kernel (`rope_attention.md`); this card covers table-only "
            "kernels and the dynamic-NTK / YaRN scaling variants. Targets B200 "
            "sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "references/NVIDIA_Manuals/cuda_guide/00_index.md",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "baselines/L1/<problem_slug>/cuda_cpp/; "
            "baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; sol-execbench-2026",
        "routing_tags": (
            "rope; rope-tables; rope-cos-sin; theta-table; dynamic-ntk; yarn; "
            "position-embedding; fp32"
        ),
        "recipe": (
            "```\n"
            "theta_i  = base^(-2i / head_dim)            // i in [0, head_dim/2)\n"
            "cos_p_i  = cos(p * theta_i)\n"
            "sin_p_i  = sin(p * theta_i)\n"
            "```\n"
            "Compute in FP32; cast to BF16 / FP16 if the downstream rotation kernel "
            "consumes the cast dtype. Dynamic-NTK and YaRN modify `theta_i` based on "
            "context-length scaling factors before the table is built."
        ),
        "building_blocks": (
            "- A `__global__` table-build kernel in CUDA C++; FP32 trig in fast-math mode.\n"
            "- `__sinf` / `__cosf` intrinsics with `--use_fast_math` for speed.\n"
            "- `cute::TiledCopy` for the final dtype-cast write to HBM on sm_100a."
        ),
        "pitfalls": (
            "- **Cached table base address** via `data_ptr` (Rule 6) — the harness shifts inputs.\n"
            "- **Wrong base value.** Common values: `10000.0` (LLaMA), `1e6` (Qwen with YaRN), per-model.\n"
            "- **Precision downgrade.** Computing `theta` in BF16 loses 1-2 bits per multiply; FP32 is contractual."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `rope_embedding_tables` rows.",
    },
    {
        "name": "rmsnorm",
        "use_when": (
            "implementing pure RMSNorm without a fused residual add — `y = (x / "
            "rms(x)) * weight`. For the residual-add-fused variant see "
            "`fused_add_rmsnorm.md`. Covers BF16 / FP16 inputs with FP32 reduction, "
            "hidden-size families 128 to 7168. Targets B200 sm_100a with sm_120 "
            "fallback."
        ),
        "primary_raw_source": "sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cuda_guide/; "
            "references/NVIDIA_Manuals/blackwell_tuning/"
        ),
        "repo_touchpoints": (
            "sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu; "
            "baselines/L1/<problem_slug>/cuda_cpp/; baselines/FIB/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; sol-execbench-2026",
        "routing_tags": (
            "rmsnorm; root-mean-square; rms; pure-rmsnorm; "
            "norm; reduction; bf16; fp16"
        ),
        "recipe": (
            "```\n"
            "rms  = sqrt(mean(x^2) + eps)              // FP32 reduction\n"
            "y    = (x / rms) * weight                 // affine scale only, no bias\n"
            "```\n"
            "Single-pass: accumulate `sum_sq` in FP32 in one CTA per row; reciprocal-sqrt "
            "via `__frsqrt_rn`; multiply-broadcast across the row. Wide rows (h > 4096) "
            "benefit from cluster-cooperative reduction on sm_100a."
        ),
        "building_blocks": (
            "- A `__global__` row-reduction kernel; FP32 accumulator; `__frsqrt_rn` for the reciprocal-sqrt.\n"
            "- `cute::TiledCopy` for the weight broadcast on sm_100a.\n"
            "- Cluster-cooperative reduce for wide-h rows (sm_100a only)."
        ),
        "pitfalls": (
            "- **Mean-of-squares in BF16** loses bits; FP32 reduction is contractual (Rule 8).\n"
            "- **Forgetting `+ eps`** when `x` is exactly zero.\n"
            "- **`data_ptr` caching of `weight`** (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `rmsnorm` rows.",
    },
    {
        "name": "mlp_gelu_geglu",
        "use_when": (
            "implementing the MLP block with GELU / GEGLU activation (vs SwiGLU — see "
            "`swiglu_mlp.md` for that). Covers `y = down(gelu(x @ up))` (single-gate) "
            "and `y = down(gelu(x @ gate) * (x @ up))` (gated, GEGLU). BF16 / FP16 "
            "inputs, FP32 GELU. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp; "
            "cutlass/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp; "
            "baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "mlp; gelu; geglu; gate; gated-mlp; activation; "
            "tanh-approx-gelu; bf16; fp16"
        ),
        "recipe": (
            "GELU MLP (non-gated):\n"
            "```\n"
            "y = down(gelu(x @ up))                       // two GEMMs + one elementwise\n"
            "```\n"
            "GEGLU (gated):\n"
            "```\n"
            "gate = x @ W_gate\n"
            "up   = x @ W_up\n"
            "y    = down(gelu(gate) * up)\n"
            "```\n"
            "GELU implementations: exact `0.5 * x * (1 + erf(x / sqrt(2)))` or the tanh "
            "approximation `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`. "
            "Match the spec's reference PyTorch."
        ),
        "building_blocks": (
            "- `cutlass::gemm::device::GemmUniversal` for the two (or three) projections.\n"
            "- `cutlass::epilogue::fusion::sm100_callbacks` for the fused-epilogue variant.\n"
            "- FP32 GELU computation inside the epilogue callback; cast back to dtype."
        ),
        "pitfalls": (
            "- **Wrong GELU flavor** (exact vs tanh-approx) — verify against spec.\n"
            "- **FP16 GELU upcast trick** is a Rule-8 violation.\n"
            "- **Three projections on three streams** (Rule 7) — fused single-stream is required."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `mlp_gelu_geglu` rows.",
    },
    {
        "name": "embedding_norm_projection",
        "use_when": (
            "implementing token-embedding lookup + initial layer-norm + optional "
            "projection — the first kernel chain in many transformer pipelines. "
            "Covers BF16 / FP16 embedding tables, FP32 normalization, optional "
            "linear projection back to model dim. Targets B200 sm_100a with sm_120 "
            "fallback."
        ),
        "primary_raw_source": "cutlass/include/cute/algorithm/copy.hpp",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cuda_guide/; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cute/atom/copy_atom.hpp; "
            "baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "embedding; token-embedding; initial-norm; layernorm; rmsnorm; "
            "projection; lookup; bf16; fp16"
        ),
        "recipe": (
            "```\n"
            "e = embedding_table[token_ids]              // gather: [B, T, hidden]\n"
            "n = LayerNorm(e)  or  RMSNorm(e)\n"
            "y = n @ W_proj      (optional)\n"
            "```\n"
            "Two valid shapes: (a) three-call (gather + norm + GEMM); (b) fused "
            "gather-norm in one CUDA C++ kernel, then a separate GEMM. (b) saves "
            "one HBM round-trip on `e`."
        ),
        "building_blocks": (
            "- A `__global__` gather kernel; `cute::TiledCopy` for the embedding-table fetch.\n"
            "- Norm reduction in FP32 inside the same kernel (shape b).\n"
            "- `cutlass::gemm` for the projection step."
        ),
        "pitfalls": (
            "- **Out-of-vocabulary token_ids** — bound-check or assume sanitized; spec defines.\n"
            "- **`data_ptr` on the embedding table** (Rule 6).\n"
            "- **Norm precision** (Rule 8) — FP32 reduction."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `embedding_norm_projection` rows.",
    },
    {
        "name": "gqa_paged_prefill",
        "use_when": (
            "implementing the prefill-phase grouped-query attention with paged KV "
            "cache — multi-query causal attention over a long context where KV is "
            "split into pages. Distinct from decode (`gqa_paged_decode.md`): prefill "
            "is multi-query (full sequence) with a causal mask. Covers head_dim ∈ "
            "{64, 128, 192, 256}, BF16 / FP16 / FP8 inputs. Targets B200 sm_100a "
            "with sm_120 fallback."
        ),
        "primary_raw_source": "sol-execbench/examples/cutlass/",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; "
            "cutlass/include/cute/atom/copy_atom.hpp; "
            "baselines/FIB/<problem_slug>/cutlass/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026; flashinfer-2026",
        "routing_tags": (
            "gqa; paged-attention; paged_attention; prefill; causal; ragged; "
            "flashinfer; kv-cache; bf16; fp16; fp8"
        ),
        "recipe": (
            "Prefill is a multi-query Flash-style attention over a paged KV cache "
            "with causal masking:\n\n"
            "```\n"
            "for each (batch, q_tile) in seq_q × num_q_heads:\n"
            "  for each page in page_table[batch]:\n"
            "    QK^T(q_tile, K_page)  // apply causal mask if page is within q_tile's past\n"
            "    online-softmax update\n"
            "    AV(P_tile, V_page)\n"
            "```\n"
            "Ragged-batch variants concatenate sequences of different lengths into a "
            "single packed tensor with a `cu_seqlens` table."
        ),
        "building_blocks": (
            "- `cutlass::TiledCopy` with paged TMA descriptors (sm_100a) for the K/V page gather.\n"
            "- `cutlass::TiledMma` for QK^T and AV; online-softmax in shared memory.\n"
            "- A causal-mask predicate evaluated per (q_idx, kv_idx) pair before softmax.\n"
            "- `cu_seqlens` for ragged-batch packing."
        ),
        "pitfalls": (
            "- **Mask off-by-one** — verify the causal predicate against spec.\n"
            "- **Paged gather on non-default stream** (Rule 7).\n"
            "- **`data_ptr` caching of page_table** (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `gqa_paged_prefill` rows.",
    },
    {
        "name": "activation_elementwise",
        "use_when": (
            "implementing standalone elementwise activations — SiLU, GELU, ReLU, "
            "Sigmoid, Tanh, Hardswish, etc. — not fused into a GEMM epilogue. Covers "
            "BF16 / FP16 / FP32 inputs with optional bias add. Targets B200 sm_100a "
            "with sm_120 fallback."
        ),
        "primary_raw_source": "references/NVIDIA_Manuals/cuda_guide/00_index.md",
        "supporting_sources": (
            "references/NVIDIA_Manuals/ptx_isa/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "baselines/L1/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; sol-execbench-2026",
        "routing_tags": (
            "activation; elementwise; silu; gelu; relu; sigmoid; "
            "tanh; hardswish; bf16; fp16; fp32"
        ),
        "recipe": (
            "Memory-bound elementwise — pure HBM-bandwidth kernels:\n"
            "```\n"
            "y[i] = activation(x[i])              // optionally with bias\n"
            "```\n"
            "Use 128-bit vectorized loads / stores via `int4` casts; FP32 internal "
            "computation; cast back to input dtype for write."
        ),
        "building_blocks": (
            "- A `__global__` elementwise kernel; vectorized `int4` loads.\n"
            "- FP32 activation computation; `__expf`, `__tanhf` intrinsics under fast-math.\n"
            "- `cute::TiledCopy` for vectorized writes on sm_100a."
        ),
        "pitfalls": (
            "- **FP16 activation upcast trick** is a Rule-8 reward-hack.\n"
            "- **Vector-load mis-alignment** — assert `data_ptr % 16 == 0` only when verified.\n"
            "- **Stream split across input tiles** is a Rule-7 violation."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `activation_elementwise` rows.",
    },
    {
        "name": "sam_mask_decoder",
        "use_when": (
            "implementing Segment Anything Model (SAM) mask-decoder fused kernels — "
            "the IoU-prediction MLP, hypernetwork MLP, and mask-token attention. "
            "Covers BF16 / FP16 inputs. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; "
            "baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "sam; segment-anything; mask-decoder; iou; hypernetwork; "
            "fused-mlp; bf16; fp16"
        ),
        "recipe": (
            "SAM-HQ mask decoder fuses three sub-MLPs:\n\n"
            "1. **IoU prediction.** Small MLP `(hidden → iou_dim → num_classes)`.\n"
            "2. **Hypernetwork.** Per-mask-token MLP that produces dynamic conv kernels.\n"
            "3. **Mask-token cross-attention.** Standard attention with shared K/V from image features.\n"
            "All three run back-to-back; the L2 problems fuse them into a single multi-launch sequence."
        ),
        "building_blocks": (
            "- Three `cutlass::gemm` collectives back-to-back (`hypernet`, `iou`, `mask_proj`).\n"
            "- `cutlass::epilogue::fusion` for activation + LayerNorm between MLP layers.\n"
            "- `attention_core` building blocks for the cross-attention step."
        ),
        "pitfalls": (
            "- **Mask-token order** — spec defines whether IoU or mask comes first.\n"
            "- **Stream split** across the three sub-MLPs is a Rule-7 reward-hack.\n"
            "- **`data_ptr` on the per-mask conv kernels** (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `sam_mask_decoder` rows.",
    },
    {
        "name": "attention_masking",
        "use_when": (
            "implementing attention-mask preparation kernels — building causal / "
            "padding / block-diagonal masks for downstream attention kernels. "
            "Distinct from attention itself; this card covers the mask-tensor build. "
            "Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "references/NVIDIA_Manuals/cuda_guide/00_index.md",
        "supporting_sources": (
            "cutlass/include/cute/algorithm/copy.hpp; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; sol-execbench-2026",
        "routing_tags": (
            "attention; attention-masking; mask; causal-mask; padding-mask; "
            "block-diag; fp32; bool"
        ),
        "recipe": (
            "Three common masks:\n\n"
            "1. **Causal:** `mask[q, k] = (k <= q)`.\n"
            "2. **Padding:** `mask[b, q, k] = (k < seq_len[b])`.\n"
            "3. **Block-diagonal:** `mask[q, k] = (block_id[q] == block_id[k])`.\n"
            "Stored as packed bools or as `-inf` floats (additive mask)."
        ),
        "building_blocks": (
            "- A `__global__` mask-write kernel; `cute::TiledCopy` for vectorized writes.\n"
            "- `__ldg` for the `seq_len` / `block_id` per-batch tables.\n"
            "- Optional in-place pack-to-bool to save HBM."
        ),
        "pitfalls": (
            "- **Float `-inf` vs bool** — match spec; mixing them silently breaks downstream softmax.\n"
            "- **`data_ptr` caching of seq_len** (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `attention_masking` rows.",
    },
    {
        "name": "mla_paged_attention",
        "use_when": (
            "implementing DeepSeek-style Multi-head Latent Attention (MLA) — paged "
            "attention with a compressed KV cache (latent KV split into low-rank `c_kv` "
            "and a RoPE'd k_pe component). Covers BF16 / FP16 inputs, head_dim families "
            "16-heads × `ckv=512 + kpe=64`. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "sol-execbench/examples/cutlass/",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; "
            "baselines/FIB/<problem_slug>/cutlass/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026; flashinfer-2026",
        "routing_tags": (
            "mla; multi-head-latent-attention; deepseek; latent-kv; "
            "paged-attention; paged_attention; bf16; fp16"
        ),
        "recipe": (
            "MLA splits each KV head into:\n\n"
            "- `c_kv ∈ [seq, 512]` — low-rank latent (no positional info; non-rotated).\n"
            "- `k_pe ∈ [seq, 64]`  — small RoPE'd positional component.\n"
            "Decode flow:\n"
            "```\n"
            "Q_c = q_proj_c(Q)                            // project Q into c_kv space\n"
            "Q_pe = rope(q_proj_pe(Q))                    // project + rotate Q's pe part\n"
            "S    = Q_c @ c_kv.T + Q_pe @ k_pe.T          // two GEMMs summed\n"
            "P    = softmax(S * scale)\n"
            "O    = P @ c_kv                              // single AV with the latent value\n"
            "```\n"
            "Save HBM vs vanilla GQA because `c_kv` is shared across all heads and only "
            "the small `k_pe` carries position info."
        ),
        "building_blocks": (
            "- Two `cutlass::gemm` collectives for the dual `Q_c @ c_kv.T` and `Q_pe @ k_pe.T` reduction.\n"
            "- `rope_attention` building blocks for the Q_pe rotation prologue.\n"
            "- Paged-KV gather (same primitives as `gqa_paged_decode`)."
        ),
        "pitfalls": (
            "- **Wrong split** — `c_kv` and `k_pe` must be cached separately under the page layout.\n"
            "- **RoPE on `c_kv`** is incorrect; only `k_pe` gets rotated.\n"
            "- **Two streams for the dual reduction** is a Rule-7 violation."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `mla_paged_attention` rows.",
    },
    {
        "name": "sparse_activation",
        "use_when": (
            "implementing sparse-activation primitives — top-k selection / sigmoid "
            "gating with thresholding / sparse-mask construction — as a standalone "
            "kernel (not as part of a MoE dispatch). Targets B200 sm_100a with "
            "sm_120 fallback."
        ),
        "primary_raw_source": "references/NVIDIA_Manuals/cuda_guide/00_index.md",
        "supporting_sources": (
            "cutlass/include/cute/algorithm/copy.hpp; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "baselines/L1/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; sol-execbench-2026",
        "routing_tags": (
            "sparse; sparse-activation; top-k; thresholded; sigmoid-gate; "
            "bf16; fp16; fp32"
        ),
        "recipe": (
            "```\n"
            "y[i] = activation(x[i])\n"
            "y[y < threshold] = 0        // or top-k mask\n"
            "```\n"
            "Two access patterns: (1) bitmask emission; (2) compacted-output emission "
            "(scatter / scan)."
        ),
        "building_blocks": (
            "- A `__global__` thresholding kernel; FP32 internal compute.\n"
            "- For top-k: a small heap or radix-select in shared memory.\n"
            "- For compaction: warp-level prefix-sum (`cub::WarpScan`)."
        ),
        "pitfalls": (
            "- **Compaction without coalescing** burns HBM bandwidth.\n"
            "- **`data_ptr` on the threshold table** (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `sparse_activation` rows.",
    },
    {
        "name": "flash_attention",
        "use_when": (
            "implementing the canonical Flash Attention forward — fused online-softmax "
            "tiled attention without paging. Covers BF16 / FP16 with FP32 reduction, "
            "head_dim ∈ {64, 128, 192, 256}. Distinct from `attention_core` in that "
            "`flash_attention` specifically denotes the Flash-2 / Flash-3 algorithm "
            "with TMA-staged producer warps. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/; "
            "cutlass/include/cute/algorithm/copy.hpp"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; "
            "baselines/L1/<problem_slug>/cutlass/; baselines/FIB/<problem_slug>/cutlass/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "flash-attention; flash; flash-2; flash-3; online-softmax; "
            "attention; bf16; fp16"
        ),
        "recipe": (
            "Flash-3 forward (sm_100a):\n\n"
            "```\n"
            "for each (q_tile, kv_tile):\n"
            "  S = Q_tile @ K_tile^T\n"
            "  m_new = max(m, max(S))         // running max\n"
            "  P = exp(S - m_new)\n"
            "  l_new = exp(m - m_new) * l + sum(P)\n"
            "  O = exp(m - m_new) * O + P @ V_tile\n"
            "  m, l = m_new, l_new\n"
            "O /= l                           // final normalize\n"
            "```\n"
            "On sm_100a the producer warps stage Q/K/V tiles via TMA into shared memory "
            "+ TMEM; the consumer warps run `tcgen05.mma` for QK^T and AV with TMEM "
            "accumulation. On sm_120 the `KernelMultistage` schedule runs in regular registers."
        ),
        "building_blocks": (
            "- Two `cutlass::gemm` collectives back-to-back (QK^T then AV) sharing the warp-specialized schedule.\n"
            "- Producer / consumer pipeline via `cute::PipelineTmaAsync` on sm_100a.\n"
            "- Online-softmax state (m, l, O) held in registers + TMEM."
        ),
        "pitfalls": (
            "- **Numerical underflow** in `exp(S - m_new)` when m is large; the `S - m` trick is contractual.\n"
            "- **head_dim splitting** — head_dim > 128 requires tile splitting; respect cluster limits.\n"
            "- **Multi-stream split of forward / backward** (Rule 7)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `flash_attention` rows.",
    },
    {
        "name": "multimodal_sequence_packing",
        "use_when": (
            "implementing the multimodal sequence-packing kernel — combining text + "
            "image / audio token streams into a single packed transformer input with "
            "modality-aware positional embeddings and segment masks. Targets B200 "
            "sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "references/NVIDIA_Manuals/cuda_guide/00_index.md",
        "supporting_sources": (
            "cutlass/include/cute/algorithm/copy.hpp; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; sol-execbench-2026",
        "routing_tags": (
            "multimodal; sequence-packing; modality; segment-mask; "
            "concatenate; gather; bf16; fp16"
        ),
        "recipe": (
            "```\n"
            "packed[range_text]  = text_tokens                  // gather\n"
            "packed[range_image] = image_tokens\n"
            "segment_ids[range_text]  = 0\n"
            "segment_ids[range_image] = 1\n"
            "position_ids[per_segment] = arange()\n"
            "```\n"
            "Gather-style packing with modality-aware segment IDs and per-modality "
            "positional embeddings, suitable for unified transformer ingestion."
        ),
        "building_blocks": (
            "- A `__global__` gather kernel; `cute::TiledCopy` for vectorized writes.\n"
            "- Per-modality `cu_seqlens` tables for the segment boundaries.\n"
            "- A small `arange` kernel for per-segment positions."
        ),
        "pitfalls": (
            "- **Segment boundary off-by-one** — spec defines closed vs half-open intervals.\n"
            "- **`data_ptr` on segment-id table** (Rule 6)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `multimodal_sequence_packing` rows.",
    },
    {
        "name": "flow_coupling",
        "use_when": (
            "implementing a normalizing-flow coupling layer — partition input, apply "
            "an affine transformation conditioned on one half, recombine. Used in "
            "flow-based generative models (RealNVP, Glow). Covers BF16 / FP16 / FP32 "
            "inputs. Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cuda_guide/; "
            "references/NVIDIA_Manuals/cutlass_cute/"
        ),
        "repo_touchpoints": (
            "baselines/L2/<problem_slug>/cuda_cpp/"
        ),
        "version_family": "cuda-13.1; cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "flow; coupling; realnvp; glow; normalizing-flow; "
            "affine-coupling; bf16; fp16; fp32"
        ),
        "recipe": (
            "```\n"
            "x_a, x_b = split(x)                     // along channel dim\n"
            "log_s, t = mlp(x_a)                     // affine parameters\n"
            "y_a = x_a\n"
            "y_b = x_b * exp(log_s) + t              // affine transform\n"
            "y   = concat(y_a, y_b)\n"
            "```\n"
            "The MLP is a standard 2-3 layer block (see `mlp_gelu_geglu` or `swiglu_mlp`). "
            "The affine + concat step is memory-bound elementwise."
        ),
        "building_blocks": (
            "- MLP sub-block (composes with `mlp_gelu_geglu`).\n"
            "- A `__global__` elementwise kernel for the affine + concat step.\n"
            "- FP32 internal compute for `exp(log_s)`."
        ),
        "pitfalls": (
            "- **`exp(log_s)` in BF16** is numerically unstable; FP32 is contractual.\n"
            "- **Wrong split axis** — verify against spec.\n"
            "- **Two-stream split** (Rule 7)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `flow_coupling` rows.",
    },
    {
        "name": "loss_perceptual",
        "use_when": (
            "implementing a perceptual-loss feature-extraction kernel — typically a "
            "VGG-style backbone evaluated at multiple intermediate layers, summed "
            "with per-layer weights. Used in vision generation training (VAE, GAN). "
            "Targets B200 sm_100a with sm_120 fallback."
        ),
        "primary_raw_source": "cutlass/include/cutlass/conv/collective/builders/sm100_umma_builder.inl",
        "supporting_sources": (
            "references/NVIDIA_Manuals/cutlass_cute/; "
            "references/NVIDIA_Manuals/blackwell_tuning/"
        ),
        "repo_touchpoints": (
            "cutlass/include/cutlass/conv/device/conv_universal_adapter.hpp; "
            "baselines/L2/<problem_slug>/cutlass/"
        ),
        "version_family": "cutlass-4.5.x; sol-execbench-2026",
        "routing_tags": (
            "loss; perceptual-loss; vgg; feature-loss; lpips; "
            "conv-features; bf16; fp16; fp32"
        ),
        "recipe": (
            "```\n"
            "for each layer L in vgg_backbone[1, 4, 7, ...]:\n"
            "  f_real = vgg_L(real_img)\n"
            "  f_fake = vgg_L(fake_img)\n"
            "  loss_L = mean_l2(f_real - f_fake)\n"
            "loss = sum_L(w_L * loss_L)\n"
            "```\n"
            "Conv blocks compose with `vision_audio_conv_blocks`; the per-layer "
            "L2 difference + sum is a reduction kernel."
        ),
        "building_blocks": (
            "- `cutlass::conv` for the VGG forward passes.\n"
            "- A `__global__` per-layer L2-mean kernel; FP32 accumulator.\n"
            "- Final weighted-sum reduction across layers."
        ),
        "pitfalls": (
            "- **Caching VGG outputs** via `data_ptr` (Rule 6).\n"
            "- **Weight-sum precision** — FP32 accumulator (Rule 8).\n"
            "- **Multi-stream layer dispatch** (Rule 7)."
        ),
        "linked_problems": "See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `loss_perceptual` rows.",
    },
]


def _render(spec: dict) -> str:
    name = spec["name"]
    title = name.replace("_", " ").title()
    frontmatter = (
        f"---\n"
        f"Use when: {spec['use_when']}\n"
        f"Primary raw source: {spec['primary_raw_source']}\n"
        f"Supporting sources: {spec['supporting_sources']}\n"
        f"Repo touchpoints: {spec['repo_touchpoints']}\n"
        f"Patents anchored here: None\n"
        f"Out of scope:\n"
        f"{FORBIDDEN_SURFACE_BLOCK}\n"
        f"Version family: {spec['version_family']}\n"
        f"Default skill: implementing-sol-baselines\n"
        f"Routing tags: {spec['routing_tags']}\n"
        f"---\n"
    )
    arch_section = ARCH_PORTABILITY_TEMPLATE.format(
        arch_notes=spec.get("arch_notes", DEFAULT_ARCH_NOTES),
    )
    body = (
        f"\n# {title}\n\n"
        f"## Use when\n\n{spec['use_when'].capitalize()}\n\n"
        f"## Canonical recipe\n\n{spec['recipe']}\n\n"
        f"## CUTLASS / CuTe / CUDA C++ building blocks\n\n"
        f"{spec['building_blocks']}\n\n"
        f"{arch_section}\n\n"
        f"## Common pitfalls\n\n{spec['pitfalls']}\n\n"
        f"## Linked problems\n\n{spec['linked_problems']}\n"
    )
    return frontmatter + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root (default: cwd)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pattern cards")
    parser.add_argument("--list", action="store_true",
                        help="Print names then exit (no file writes)")
    args = parser.parse_args()

    if args.list:
        for spec in PATTERN_SPECS:
            print(spec["name"])
        return 0

    root = Path(args.root).resolve()
    patterns_dir = root / ".claude" / "knowledge" / "patterns"
    if not patterns_dir.is_dir():
        print(f"error: {patterns_dir} not found", file=sys.stderr)
        return 1

    written = 0
    skipped = 0
    for spec in PATTERN_SPECS:
        out = patterns_dir / f"{spec['name']}.md"
        if out.exists() and not args.force:
            print(f"skip (exists): {out.relative_to(root)}")
            skipped += 1
            continue
        out.write_text(_render(spec), encoding="utf-8")
        print(f"wrote {out.relative_to(root)}")
        written += 1

    print(f"\nSummary: {written} written, {skipped} skipped, "
          f"{len(PATTERN_SPECS)} total pattern specs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
