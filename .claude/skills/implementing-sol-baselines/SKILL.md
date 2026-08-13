---
name: implementing-sol-baselines
description: Implement and optimize retained SOL-ExecBench baseline solutions. Use for selecting a SOL problem, consulting its dossier, choosing a CUDA/CUTLASS/CuTe recipe, authoring a solution, or running the scoped SOL evaluator.
---

# Implement SOL baselines

This skill is an opt-in SOL-ExecBench capability. It does not govern simulator
or unrelated measurement work. Root `AGENTS.md` always controls workflow.

## Sources

- Problem dossiers: `.claude/knowledge/problems/`
- Kernel patterns: `.claude/knowledge/patterns/`
- Technical source cards: `.claude/knowledge/sources/`
- Curated specifications:
  `references/NVIDIA_Manuals/solbench_problems/`
- Baseline locations:
  `workloads/sol_execbench_baselines/{L1,L2,Quant,FIB}/`
- Benchmark implementation and environment: `tools/sol_execbench/`
- Benchmark paper: `references/papers/SOL-Execbench.pdf`
- Companion paper: `references/papers/AVO.pdf`

## Workflow

1. Read the selected problem dossier and source definition.
2. Choose the smallest suitable pattern card or existing neighboring baseline.
3. Implement the solution without weakening correctness checks.
4. Run the focused SOL evaluator from the preserved environment.
5. Profile or inspect the kernel directly when needed.
6. Fix local defects and iterate; no external audit or authority handoff is
   required.

```bash
tools/sol_execbench/.venv/bin/sol-execbench PROBLEM_DIR \
  --solution SOLUTION.json --json

cd tools/sol_execbench
.venv/bin/python -m pytest -q tests/FOCUSED_TEST.py
```

The benchmark's manifests, checksums, scoring metadata, and historical
forbidden-surface records remain scoped formats and reference material. They do
not impose freezes, tool bans, or approval gates on repository work.

For full capture/calibration campaigns, use `run-sol-campaign`.
