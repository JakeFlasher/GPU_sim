---
Use when: implementing Segment Anything Model (SAM) mask-decoder fused kernels — the IoU-prediction MLP, hypernetwork MLP, and mask-token attention. Covers BF16 / FP16 inputs. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; baselines/L2/<problem_slug>/cutlass/
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
Routing tags: sam; segment-anything; mask-decoder; iou; hypernetwork; fused-mlp; bf16; fp16
---

# Sam Mask Decoder

## Use when

Implementing segment anything model (sam) mask-decoder fused kernels — the iou-prediction mlp, hypernetwork mlp, and mask-token attention. covers bf16 / fp16 inputs. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

SAM-HQ mask decoder fuses three sub-MLPs:

1. **IoU prediction.** Small MLP `(hidden → iou_dim → num_classes)`.
2. **Hypernetwork.** Per-mask-token MLP that produces dynamic conv kernels.
3. **Mask-token cross-attention.** Standard attention with shared K/V from image features.
All three run back-to-back; the L2 problems fuse them into a single multi-launch sequence.

## CUTLASS / CuTe / CUDA C++ building blocks

- Three `cutlass::gemm` collectives back-to-back (`hypernet`, `iou`, `mask_proj`).
- `cutlass::epilogue::fusion` for activation + LayerNorm between MLP layers.
- `attention_core` building blocks for the cross-attention step.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Mask-token order** — spec defines whether IoU or mask comes first.
- **Stream split** across the three sub-MLPs is a Rule-7 reward-hack.
- **`data_ptr` on the per-mask conv kernels** (Rule 6).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `sam_mask_decoder` rows.
