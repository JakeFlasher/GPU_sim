---
name: implementing-sol-baselines
description: Scoped SOL-ExecBench baseline implementation guidance for the evaluation subtree. It does not govern the active Rust simulator or root measurement policy. Retain its B200/sm_100a and local-sm_120 benchmark context only where the SOL subtree explicitly requires it; consult root CLAUDE.md for current NCU/architecture boundaries.
---

# implementing-sol-baselines

> **Subtree scope.** This skill owns only SOL-ExecBench baseline/evaluation
> authoring. Its historical local/cloud device assumptions and forbidden-surface
> list must not override root `CLAUDE.md`, the active NCU label protocol, or the
> simulator's architecture-local capability rules.

Scoped skill for the SOL-ExecBench baseline-implementation workflow; it is not a
root simulator or measurement contract.

## Owns

- `.claude/knowledge/sources/` — citation stubs pointing to technical source materials (SOL-ExecBench paper, AVO paper, NVIDIA reference manuals for CUTLASS / CuTe / Blackwell / Ampere).
- `.claude/knowledge/patterns/` — kernel-pattern recipe cards. ~25 at full Upper Bound; 3 in the Round 1 Lower-Bound thin slice (`gemm_baseline`, `fused_add_rmsnorm`, `fp8_blockwise_quant`).
- `.claude/knowledge/problems/` — per-problem dossiers. 235 at full Upper Bound; 4 in the Round 1 Lower-Bound thin slice covering L1 / L2 / Quant / FIB.

## When to use

- Selecting and starting a SOL-ExecBench problem (e.g. "I want to start `problem_004_gemm_n128_k2048`").
- Choosing a CUTLASS / CuTe recipe for a given kernel pattern (GEMM, attention, RMSNorm, RoPE, MoE dispatch, FP8 / NVFP4 quant, GQA, SwiGLU, paged-attention).
- Historical reasoning about portability between local sm_120 (RTX 5060) and cloud
  sm_100a (B200), only where the SOL subtree requires it.
- Verifying the scoped SOL submission format and its retained forbidden-surface
  records.

## What it does NOT cover

- **Measurement and host policy** are outside this skill. NCU labels, clock policy,
  CUPTI surfaces, and privilege boundaries must follow root `CLAUDE.md`, active
  schemas, `tools/local_labels/`, and the host runbook; this subtree does not
  authorize or forbid them.
- **Python DSL submissions** are governed, if at all, by the scoped SOL-ExecBench
  contract in `docs/sol_execbench/CLAUDE.md`, not by a retired root Rule 5.
- **Profiler-era simulator design** — see the active Rust simulator documentation.
- **Driver / host work** — consult the current root charter and host runbook.

## Authority anchors

- Root `CLAUDE.md`, `docs/README.md`, and `docs/STATUS.md` — current project
  policy, status, and documentation precedence. The historical SOL subtree may
  not override them.
- `docs/sol_execbench/CLAUDE.md` and `docs/sol-execbench.md` — scoped historical
  benchmark context and retained workflow references.
- `references/NVIDIA_Manuals/solbench_problems/` — 235 hand-curated problem
  specifications and technical source material, not root policy.
- `workloads/sol_execbench_baselines/{L1,L2,Quant,FIB}/` — retained hand-port
  locations; per-problem status belongs in the corresponding scoped README.
- [`references/papers/SOL-Execbench.pdf`](../../../references/papers/SOL-Execbench.pdf) (arXiv:2603.19173) — benchmark paper and historical metric definition.
- [`references/papers/AVO.pdf`](../../../references/papers/AVO.pdf) — agent-optimizer companion paper and historical comparison context.

## Routing-time behaviour

`scripts/kb/route_query.py` imports the shared ownership contract from
`scripts/kb/routing_contract.py`, which maps `sources`, `patterns`,
`problems`, and `cutlass_constructs` to this skill. SOL-pattern anchors in
`SKILL_ANCHORS` (`gemm`, `attention`, `rmsnorm`, `rope`, `moe`, `gqa`,
`swiglu`, `paged_attention`, `fp8`, `nvfp4`, `problem_[0-9]{3,}`) boost the
`patterns` and `problems` subcorpora when the prompt contains those tokens.

`scripts/kb/_relevance.py` Group A includes the same SOL anchors, so problem-start prompts pass the relevance gate and reach this skill's cards.
