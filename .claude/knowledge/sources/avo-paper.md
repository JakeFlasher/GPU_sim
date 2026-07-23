---
Use when: looking up the AVO (Agentic Variation Operators) paper — the NVIDIA companion paper that almost certainly describes the agent-optimizer producing the SOL-ExecBench scoring-baseline `Tb` anchor (paper median S=0.732). Authoritative source for the agent baseline our submissions aim to beat.
Primary raw source: AVO.pdf
Supporting sources: SOL-Execbench.pdf; references/papers/
Repo touchpoints: AVO.pdf; SOL-Execbench.pdf
Patents anchored here: None
Out of scope: This stub is a citation pointer; do NOT re-implement the AVO search loop locally. The paper describes the SOTA agent we want to beat on the public SOL-ExecBench leaderboard — its mutation / selection mechanics inform our understanding of `Tb`, not our submission code.
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: sol-execbench; avo; agent-optimizer; scoring-baseline; tb-anchor; evolutionary-search; paper; nvidia; gemm; attention; moe; fp8
Latency authority: paper
---

# AVO Paper (Citation Stub)

## Use when

Looking up AVO ("Agentic Variation Operators for Autonomous Evolutionary Search") for context on the SOL-ExecBench scoring-baseline `Tb`. The AVO author set overlaps with the SOL-ExecBench paper's; AVO is the most likely identity of the agent that produced `Tb` (which our submissions must beat to score `S > 0.5`).

## Pointer

The paper lives at `./AVO.pdf` (7 pages, NVIDIA). Note the SOL-ExecBench paper's overall median agent score is **S=0.732** with per-category medians L1=0.688, L2=0.761, Quant=0.757, FIB=0.789 — those are the bars to beat.

## Out of scope for this stub

This card is a context-only citation. Do not attempt to re-create the AVO search loop locally — the SOL-ExecBench leaderboard is the only place where the agent-baseline comparison is made, and submissions are manually uploaded per `CLAUDE.md`. Forbidden surfaces (Python DSL submissions, NCU, clock-locking, driver work) are off-limits regardless of how the agent baseline approached them.
