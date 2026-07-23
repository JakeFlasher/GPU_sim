---
Use when: looking up the canonical mirror of all 235 SOL-ExecBench problem specs as `problem_NNN_<slug>.md`. Authoritative source for per-problem shape tables, dtype contracts, reference PyTorch `run()` functions, and per-workload latency anchors.
Primary raw source: references/NVIDIA_Manuals/solbench_problems/
Supporting sources: SOL-Execbench.pdf; sol-execbench/docs/definition.md; sol-execbench/docs/workload.md
Repo touchpoints: references/NVIDIA_Manuals/solbench_problems/; tools/sol_baselines_scaffold/sol_pattern_mapping.md
Patents anchored here: None
Out of scope: This stub is a citation gateway. For implementation guidance see the per-problem dossiers under `.claude/knowledge/problems/<slug>.md`; for kernel-family recipes see `.claude/knowledge/patterns/`. The specs are read-only mirrors; do NOT edit them locally.
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: sol-execbench; problem-specs; solbench-problems; problem_NNN; shape-tables; reference-pytorch; gemm; attention; rmsnorm; rope; moe; fp8; nvfp4
Latency authority: paper
---

# SOL-ExecBench Problem Specs (Mirror)

## Use when

Looking up the canonical filename for a SOL-ExecBench problem, or extracting the shape table / reference PyTorch / per-workload latency baselines from a spec. Every per-problem dossier under `.claude/knowledge/problems/<slug>.md` cites a single spec from this mirror via its `Primary raw source` field.

## Pointer

The mirror lives at `references/NVIDIA_Manuals/solbench_problems/` and contains exactly 235 markdown files of the form `problem_NNN_<kernel_slug>.md`. Each file has:

1. A `## Description` paragraph (one or two sentences).
2. An input-dtype table (`| Name | Shape | Dtype |`).
3. An output-dtype table (same shape).
4. A workload table (`| # | N | K | M | Baseline (ms) | SOL (ms) |` or similar).
5. A reference PyTorch `run()` function in a fenced ```python block.
6. A user-results table with `Reference Implementation` latency.

## Out of scope for this stub

This card is a citation gateway. The crawled mirror is read-only and refreshed by a separate agent; do not modify files here. For routing into specific specs, the gate's `\bproblem_[0-9]{3,}` Group A anchor takes any `problem_NNN_*` slug fragment directly to the matching dossier.
