# references/

Local reference material supporting the SOL-ExecBench baseline project.

- `NVIDIA_Manuals/` — NVIDIA documentation mirror (CUDA, PTX, SASS,
  binary utilities, tuning guides, CUTLASS / CuTe / cuTile / Blackwell /
  Ampere, NCU metric definitions). NCU docs are retained as a
  metric-semantics dictionary; NCU itself is NOT an allowed measurement
  surface (see `CLAUDE.md` Critical Rule 1).
- `NVIDIA_Manuals/solbench_problems/` — 235 SOL-ExecBench problem specs
  (`problem_NNN_<slug>.md`), each with shape table, reference PyTorch
  `run()`, and per-workload SOL + scoring-baseline + reference-implementation
  latencies.
- `NVIDIA_Manuals/scripts/` — crawler toolchain
  (`crawler_multiTOC.py`, `crawler_singlefile.py`,
  `sol_execbench_crawler.py`).
- `NVIDIA_patents/` — NVIDIA patents relevant to GPU architecture and
  compiler techniques.
- `papers/` — supporting papers (performance modeling, microbenchmarking,
  CUTLASS / CuTe / FlashAttention / FlashInfer / Orojenesis literature,
  and related benchmarking work).
