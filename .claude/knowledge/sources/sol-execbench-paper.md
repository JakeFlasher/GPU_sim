---
Use when: looking up the SOL-ExecBench paper for the SOL Score formula, the 235-problem split (L1 / L2 / Quant / FIB), reward-hacking taxonomy (Table 3), eval-harness parameters, or the SOLAR `T_SOL` derivation pipeline. Authority anchor for paper-level claims about the benchmark.
Primary raw source: SOL-Execbench.pdf
Supporting sources: AVO.pdf; references/papers/
Repo touchpoints: SOL-Execbench.pdf; AVO.pdf; .humanize/plans/KB_update_plan.md; CLAUDE.md
Patents anchored here: None
Out of scope: This stub is a citation pointer only — read `SOL-Execbench.pdf` directly for benchmark internals. For per-problem starting points see `.claude/knowledge/problems/`; for kernel-pattern recipes see `.claude/knowledge/patterns/`.
Version family: sol-execbench-2026 (arXiv:2603.19173)
Default skill: implementing-sol-baselines
Routing tags: sol-execbench; paper; arxiv; sol-score; benchmark; reward-hacking; cuda-events; b200; sm_100a; gemm; attention; rmsnorm; rope; moe; fp8; nvfp4
Latency authority: paper
---

# SOL-ExecBench Paper (Citation Stub)

## Use when

Looking up the SOL-ExecBench paper as the authoritative source for: the SOL Score formula `S(Tk) = (Tb - T_SOL) / ((Tk - T_SOL) + (Tb - T_SOL))`; problem-set composition; eval-harness behaviour (CUDA events, 10 warmup + 50 timed x 3 trials, L2-cache clear, 1500 MHz B200 clock-lock); the reward-hacking exploit / defense pairs in Table 3.

## Pointer

The benchmark paper lives at `./SOL-Execbench.pdf` (18 pages, NVIDIA, arXiv:2603.19173, March 2026). Lead authors: Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang.

The companion agent-optimizer paper (whose internal scoring baseline `Tb` we aim to beat: median S=0.732) is at `./AVO.pdf` -- "Agentic Variation Operators for Autonomous Evolutionary Search".

## Anchored values (paper authority)

- Overall median SOL score from the paper's agent baseline: S=0.732.
- Per-category medians: L1=0.688, L2=0.761, Quant=0.757, FIB=0.789.
- 14.5% of agent submissions were flagged for reward hacking and rejected (paper Section 5.3).

When citing these numbers elsewhere in the KB, set `Latency authority: paper` to satisfy the latency-authority rule.

## Out of scope for this stub

This card is a citation gateway. For implementation guidance see the pattern recipes under `patterns/` and the per-problem dossiers under `problems/`. For the eval-harness CLI surface see the official `sol-execbench/` repo cloned at the repo root.
