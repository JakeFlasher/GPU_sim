---
Use when: implementing the SwiGLU-activated MLP block `output = (silu(x @ W_gate) * (x @ W_up)) @ W_down` used by LLaMA-family, Qwen, Mistral, DeepSeek decoders. Covers BF16 / FP16 inputs with FP32 SiLU and accumulation, gate / up / down projection sizes, and the fused-elementwise vs three-pass tradeoff. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp; cutlass/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp; baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/
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
Version family: cuda-13.1; cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: swiglu; silu; mlp; gate; gate-up; gated-mlp; fused-mul; activation; bf16; fp16
---

# SwiGLU MLP (Gated Activation)

## Use when

You're implementing an MLP block whose activation is gated SwiGLU rather than GELU / ReLU. Specifically:

```
gate    = x @ W_gate           // [B, T, hidden] @ [hidden, ffn_dim] -> [B, T, ffn_dim]
up      = x @ W_up             // [B, T, hidden] @ [hidden, ffn_dim] -> [B, T, ffn_dim]
hidden  = silu(gate) * up      // elementwise; silu(z) = z * sigmoid(z)
output  = hidden @ W_down      // [B, T, ffn_dim] @ [ffn_dim, hidden] -> [B, T, hidden]
```

This is the canonical SOL-ExecBench MLP shape for modern LLMs. Common hidden sizes are 4096 / 5120 / 7168 with `ffn_dim` ≈ 2.67 × hidden (rounded to a multiple of 256 / 1024).

## Three implementation shapes

1. **Three-call.** Two GEMMs (`gate`, `up`) → one elementwise kernel for `silu(gate) * up` → one GEMM (`down`). Simplest; pays two extra HBM round-trips on the intermediate tensors. Author this first.
2. **Fused-epilogue.** Use CUTLASS's `EpilogueVisitor` to fuse `silu(gate) * up` into the *output* of the second GEMM (the `up` GEMM consumes its own GEMM output and the previously computed `gate` as auxiliary input). This is the standard `sol-execbench/examples/cutlass/` fused pattern.
3. **Two-GEMM concatenated (`gate_up`).** Concatenate `W_gate` and `W_up` along the output dim into one weight matrix and run a single fused GEMM that emits both, then apply the SwiGLU elementwise in a tiny follow-on kernel. Reduces address calculation cost; popular in production inference engines.

Pick (1) for a Round 1 baseline; (2) for a perf-tuning pass; (3) once the layout invariants are stable.

## CUTLASS / CuTe building blocks

- `cutlass::gemm::device::GemmUniversal` with a sm_100a collective builder (see `gemm_baseline.md`).
- `cutlass::epilogue::collective::sm100_epilogue_builder` for shape (2) with a `cutlass::epilogue::fusion` callback chain implementing `silu(aux) * acc` where `aux` is the `gate` tensor passed via auxiliary load.
- `cute::TiledCopy` for the auxiliary `gate` load when shape (2) is used.
- FP32 SiLU computation. **Do NOT compute SiLU in BF16 / FP16** — `silu(z) = z * sigmoid(z)` is numerically sensitive around `z ≈ 0` and BF16 sigmoid loses precision. Cast to FP32, compute, cast back. Skipping this is a Rule 8 precision-downgrade reward-hack.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Concrete divergences:

- **TCGen05 fused epilogue.** On sm_100a the fused-epilogue shape (2) uses `tcgen05.mma` with `EpilogueVisitor` chaining; on sm_120 the `KernelMultistage` schedule runs the epilogue in regular-register accumulators with reduced throughput. Local sm_120 timing under-represents the B200 fused-epilogue win.
- **Auxiliary tensor TMA load.** Shape (2)'s `gate` auxiliary load benefits from B200 TMA descriptors; sm_120 falls back to `cp.async` and pays more L1 / shared-memory pressure.
- **HBM3e vs GDDR7 bandwidth.** B200's higher HBM bandwidth makes the three-call shape (1) less penalized than on sm_120; the relative cost of shapes (1) vs (2) is arch-dependent.
- **`ffn_dim` cache fit.** B200's larger L2 may keep parts of `W_down` resident across batches; sm_120's smaller L2 evicts more aggressively.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **SiLU in BF16** (CLAUDE.md Rule 8). `sigmoid` in BF16 around small values loses bits; the harness's tolerance check catches it. Use FP32 SiLU.
- **`data_ptr` caching of `W_gate` / `W_up`** (CLAUDE.md Rule 6). The harness shifts pointers by 256 B every iteration; memoised weight bases fail.
- **Multi-stream gate / up dispatch** (CLAUDE.md Rule 7). Tempting to run the two GEMMs on separate streams to overlap; the harness's stream-injection detector rejects this. Use the default PyTorch stream and let the scheduler order them.
- **Wrong rounding in concatenated weight layout** (shape 3). Concatenating `W_gate` and `W_up` along the output dim only works if the output stride is the concatenated dim's stride; double-check with a small numeric test before committing the shape.

## Linked problems

- L1 / L2 SwiGLU-bearing problems link here via the matching-pattern field of their dossiers. SwiGLU MLP is a frequent component of both L1 single-op (`silu_and_mul` style) and L2 fused decoder-layer problems.
- Composes with `.claude/knowledge/patterns/fused_add_rmsnorm.md` in the typical decoder block (RMSNorm → attention → residual → RMSNorm → SwiGLU MLP → residual).
