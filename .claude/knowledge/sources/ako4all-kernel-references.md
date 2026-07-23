---
Use when: writing a *.cu hand-port and needing per-language kernel reference patterns — CUDA C++ or CUTLASS/CuTe C++ — or architecture-specific optimization notes covering solbench's full local-test → cloud-scoring path. Kept: sm_100/sm_100a (B200 cloud scoring target), sm_103 (Blackwell datacenter variant context), sm_120 (RTX 5060 laptop local proxy). For sm_80/sm_86 (Ampere — 3090Ti, A100) local-test, route to solbench's `references/NVIDIA_Manuals/ampere_compatibility/` (11 files) + `ampere_tuning/` (11 files) — kernel-pilot did not ship Ampere guides. Source: BBuf/AKO4ALL CUDA kernel skill set, vendored into kernel-pilot then imported into solbench with NCU references stripped and the Triton + CuTe DSL Python subtrees dropped (Rule 5 forbids them as submission surfaces). sm_89 (Ada — RTX 4090, L40S) and sm_90 (Hopper H100) dropped: not in solbench hardware.
Primary raw source: references/kernel_pilot_imported/references/ako4all/
Supporting sources: references/kernel_pilot_imported/references/ako4all/source-attribution.md; references/kernel_pilot_imported/references/index.md
Repo touchpoints: references/kernel_pilot_imported/references/ako4all/
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
Version family: ako4all-trimmed-2026-05
Default skill: implementing-sol-baselines
Routing tags: kernel-pilot; ako4all; cuda-cpp; cutlass-cpp; kernel-templates; kernel-loop; architecture-reference; troubleshooting; sm100; blackwell; b200
---

# AKO4ALL Kernel References (Imported, Trimmed)

## Use when

You need a per-language kernel reference for CUDA C++ or CUTLASS/CuTe C++ (the only submittable surfaces for solbench), or sm_100/sm_100a (B200) optimization context.

## Approach / Content

**15 files kept after the trim:**

| File | Purpose |
|---|---|
| `ako4all-kernel-loop.md` | CUDA kernel optimization loop discipline (NCU step replaced with counter-free surfaces) |
| `cuda-cpp-kernel-reference.md` | CUDA C++/PTX reference, examples (stream-overlap + CUDA Graphs marked NOT FOR SOLBENCH per Rule 7) |
| `cutlass-cpp-kernel-reference.md` | CUTLASS/CuTe C++ reference (the **submission target**) |
| `kernel-templates.md` | Template families (Stream Overlap Pipeline section marked NOT FOR SOLBENCH per Rule 7) |
| `nvidia-architecture-reference.md` | GPU architecture (SM, memory hierarchy, execution model) — sm_89 / sm_90 / sm_120 retained as cross-arch background context |
| `source-attribution.md` | Attribution + license notes |
| `troubleshooting.md` | Common issues, solutions, optimization tips |
| `cuda-cpp/cuda-cpp-overview.md` | CUDA C++ kernel work overview |
| `cuda-cpp/sm100-optimization-guide.md` | B200 sm_100 optimization (**cloud scoring target**) |
| `cuda-cpp/sm103-optimization-guide.md` | Blackwell variant context |
| `cuda-cpp/sm120-optimization-guide.md` | **RTX 5060 laptop (local proxy)** |
| `cutlass-cpp/cutlass-cpp-overview.md` | CUTLASS/CuTe C++ overview |
| `cutlass-cpp/sm100-optimization-guide.md` | B200 sm_100 CUTLASS/CuTe optimization (**cloud scoring target**) |
| `cutlass-cpp/sm103-optimization-guide.md` | Blackwell variant context, CUTLASS surface |
| `cutlass-cpp/sm120-optimization-guide.md` | **RTX 5060 laptop (local proxy)**, CUTLASS surface |

## What is INTENTIONALLY NOT imported / dropped during the trim

- `profiling-debugging-reference.md` — entirely NCU-focused (dropped during NCU strip).
- `cuda-cpp/vendored-docs/` (~860 files) — covered by solbench's `references/NVIDIA_Manuals/` more comprehensively.
- `cuda-cpp/vendored-docs/ncu-docs/` (7 files) — NCU documentation, forbidden by Rule 1.
- `architectures/sm89-optimization-guide.md` (Ada — RTX 4090, L40S) — not in solbench hardware path.
- `architectures/sm90-optimization-guide.md` (Hopper H100) — not in solbench hardware path.
- (sm_100, sm_103, sm_120 CUDA C++ + CUTLASS C++ guides RESTORED for solbench's local-test → cloud-scoring workflow)
- `cute-dsl/sm103-optimization-guide.md`, `cute-dsl/sm120-optimization-guide.md` — dropped with entire cute-dsl/ subtree (CuTe DSL Python forbidden by Rule 5).
- Entire `cute-dsl/` subtree (18 files) — CuTe DSL Python forbidden by Rule 5; dropped post-pivot.
- Entire `triton/` subtree (10 files) — Triton forbidden by Rule 5; dropped post-pivot.
- `triton-kernel-reference.md`, `cute-dsl-kernel-reference.md` — top-level forbidden-DSL references; dropped post-pivot.

## Architecture coverage for solbench's local-test → cloud-scoring workflow

| Hardware | Architecture | Coverage | Where |
|---|---|---|---|
| 3090Ti / A100 | sm_86 / sm_80 (Ampere) | local-test refs | solbench's `references/NVIDIA_Manuals/ampere_compatibility/` (11 files) + `ampere_tuning/` (11 files). kernel-pilot did not ship Ampere guides. |
| RTX 5060 laptop | sm_120 (consumer Blackwell) | local dev proxy | `cuda-cpp/sm120-optimization-guide.md` + `cutlass-cpp/sm120-optimization-guide.md` |
| B200 (cloud) | sm_100a (datacenter Blackwell) | cloud scoring target | `cuda-cpp/sm100-optimization-guide.md` + `cutlass-cpp/sm100-optimization-guide.md` + solbench's `references/NVIDIA_Manuals/blackwell_tuning/` (12 authoritative files) |
| B-series variant | sm_103 (Blackwell variant) | datacenter Blackwell context | `cuda-cpp/sm103-optimization-guide.md` + `cutlass-cpp/sm103-optimization-guide.md` |

## License / attribution

See `references/kernel_pilot_imported/references/ako4all/source-attribution.md` for the AKO4ALL vendoring license. The original AKO4ALL pack is at [BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS); kernel-pilot vendored a subset, and this trimmed import keeps the B200-relevant CUDA C++ / CUTLASS C++ subset (sm_100 + sm_103 + sm_120 covering solbench's local-test → cloud-scoring workflow) minus NCU / Triton / CuTe-DSL-Python / sm_89 (Ada) / sm_90 (Hopper) pieces.
