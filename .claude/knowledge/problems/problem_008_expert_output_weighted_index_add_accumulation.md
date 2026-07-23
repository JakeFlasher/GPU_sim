---
Use when: starting the L1 baseline for `problem_008_expert_output_weighted_index_add_accumulation` -- MoE routing or expert dispatch. Spec summary: Critical atomic MoE operation that performs weighted expert output accumulation using index_add_ with element-wise routing weight multiplication. Expert outputs are multiplied by routing weights and scattered back to original token posit...
Primary raw source: references/NVIDIA_Manuals/solbench_problems/problem_008_expert_output_weighted_index_add_accumulation.md
Supporting sources: .claude/knowledge/patterns/moe_dispatch.md; .claude/knowledge/sources/cutlass-cute-docs.md
Repo touchpoints: baselines/L1/problem_008_expert_output_weighted_index_add_accumulation/cutlass/; baselines/L1/problem_008_expert_output_weighted_index_add_accumulation/cuda_cpp/
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
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: expert; output; weighted; index; add; accumulation; moe_dispatch
---

# problem_008_expert_output_weighted_index_add_accumulation

## Use when

starting the L1 baseline for `problem_008_expert_output_weighted_index_add_accumulation` -- MoE routing or expert dispatch. Spec summary: Critical atomic MoE operation that performs weighted expert output accumulation using index_add_ with element-wise routing weight multiplication. Expert outputs are multiplied by routing weights and scattered back to original token posit...

## Workload summary

| Field | Value |
| --- | --- |
| Category | L1 |
| Primary pattern | `moe_dispatch` |
| Secondary patterns | none |
| Source spec | `references/NVIDIA_Manuals/solbench_problems/problem_008_expert_output_weighted_index_add_accumulation.md` |
| Notes | MoE routing or expert dispatch |

For shapes, dtypes, reference PyTorch `run()`, and per-workload hardware-target latencies, consult the source spec listed above. This dossier intentionally does not duplicate the spec's numeric workload table (which carries paper-authority latencies) to keep the dossier free of latency-authority ambiguity.

## Matching pattern

This problem instantiates `.claude/knowledge/patterns/moe_dispatch.md`. The pattern card carries the canonical CUTLASS / CuTe / CUDA C++ recipe; this dossier records the problem-specific shape and dtype constraints.

## Implementation entry point

Per CLAUDE.md Rule 5 the submission surface is CUTLASS C++ or CUDA C++. Author the baseline under one (or both) of:

- `baselines/L1/problem_008_expert_output_weighted_index_add_accumulation/cutlass/{main.cpp, kernel.cu, kernel.h, solution.json}` for a CUTLASS-template variant.
- `baselines/L1/problem_008_expert_output_weighted_index_add_accumulation/cuda_cpp/{kernel.cu, solution.json}` for a from-scratch CUDA C++ variant.

Entry-point signature per `sol-execbench/docs/solution.md` (by-value `torch::Tensor` for inputs and outputs; the eval driver pre-allocates the outputs and passes them in destination-passing style):

```cpp
void run(/* input torch::Tensor args */,
         /* output torch::Tensor args */) {
    // launch kernel(s); write into the output tensors in-place; no return.
}
```

Dtype contract: BF16 / FP16 / FP8 inputs, FP32 softmax + accumulator.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

The 5060 (sm_120) compiles and smoke-runs this baseline for correctness verification; final SOL scoring is on B200 (sm_100a). Treat the two targets as different: TCGen05 / TMA / DSMEM / TMEM behaviors are sm_100a-only, and HBM3e bandwidth / L2 size differences mean local timing is not a proxy for the cloud B200 score. The B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a` for `target_hardware: ["B200"]`; build twice when in doubt.

Evidence-label vocabulary per `CLAUDE.md` lives in `baselines/L1/problem_008_expert_output_weighted_index_add_accumulation/README.md`, NOT in this dossier.

## Cross-references

- Pattern recipe: `.claude/knowledge/patterns/moe_dispatch.md`.
- Source spec (authoritative): `references/NVIDIA_Manuals/solbench_problems/problem_008_expert_output_weighted_index_add_accumulation.md`.
- CUTLASS submission-format precedent: `sol-execbench/examples/cutlass/` and `sol-execbench/examples/cuda_cpp/`.
