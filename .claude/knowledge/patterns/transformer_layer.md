---
Use when: implementing a full transformer decoder / encoder layer as a single fused kernel-graph (multi-op L2 problem). Composes attention, RoPE, RMSNorm, residual add, MLP. Variants: pre-norm vs post-norm, dense vs MoE FFN, BF16 vs FP8. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/examples/
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp; baselines/L2/<problem_slug>/cutlass/
Patents anchored here: None
Out of scope:
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8).
Version family: cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: transformer; transformer-layer; decoder-layer; encoder-layer; pre-norm; post-norm; bf16; fp8
---

# Transformer Layer

## Use when

Implementing a full transformer decoder / encoder layer as a single fused kernel-graph (multi-op l2 problem). composes attention, rope, rmsnorm, residual add, mlp. variants: pre-norm vs post-norm, dense vs moe ffn, bf16 vs fp8. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Canonical pre-norm decoder layer:

```
x'  = x + Attention(RMSNorm(x))             // attention residual
y   = x' + MLP(RMSNorm(x'))                 // MLP residual
```
Each sub-block composes a pattern card already in the atlas:
- Attention: `attention_core` (or `rope_attention` if rotational, or `gqa_paged_decode` for paged).
- RMSNorm: `fused_add_rmsnorm` (fuses the residual add).
- MLP: `swiglu_mlp` (or `mlp_gelu_geglu` for non-gated variants).
- MoE variant: replace MLP with `moe_dispatch`.

## CUTLASS / CuTe / CUDA C++ building blocks

- Compose the sub-pattern cards' recipes; this card carries the *layer-level* ordering and the choice of fusion strategy (kernel boundaries).
- For B200, the layer can fold attention + RMSNorm + residual into a single CUTLASS schedule via epilogue chaining; for sm_120 split into 2-3 launches.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Pre-norm vs post-norm** is spec-dependent; verify against the reference PyTorch in the spec.
- **Stream split across sub-blocks** is a Rule-7 reward-hack.
- **Residual `data_ptr` caching** across the two residuals (Rule 6).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `transformer_layer` rows.
