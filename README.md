# GPU simulator

This repository contains the Rust trace/replay engine, architecture cards,
NVBit producer, and SOLBench workload sources for the SM120/RTX 5090 alpha
lane.

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
NCU/CUPTI observations ---------------------------------------> independent join
```

## Start here

1. Read [`AGENTS.md`](AGENTS.md) for the permanent workflow boundaries.
2. Read [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) and
   [`docs/STATUS.md`](docs/STATUS.md).
3. Follow `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
4. Use [`docs/runbook_host.md`](docs/runbook_host.md) for complete-corpus
   replay commands.

Acceptance is performance and generalization on the complete SOLBench plus
modern/SOTA traces in `/root/sm120_rtx_5090_capture`, compared with independent
NCU ground truth.  Full replay is primary; fast replay is complementary.  No
fixture/synthetic simulator tests or procedural hash/check/validation gates are
part of the alpha workflow.

## Repository map

- `sim/` — Rust trace frontend, scheduler, memory model, event engine, runners.
- `arch_profiles/` — active RTX 5090 card and architecture schemas.
- `tools/nvbit_tracer/` — real trace and sidecar producer.
- `tools/sm120_*.py` — replay and corpus reporting tools.
- `workloads/sol_execbench_baselines/` — SOLBench kernel sources.
- `docs/` and `plans/` — current contracts and plan pointer.
