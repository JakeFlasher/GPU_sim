# GPU simulator

Rust trace/replay engine and NVBit producer for the SM120/RTX 5090 alpha lane.

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
NCU/CUDA observations ---------------------------------------> post-replay join
```

## Start here

1. Read [`AGENTS.md`](AGENTS.md) for repository boundaries.
2. Read [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) and
   [`docs/STATUS.md`](docs/STATUS.md).
3. Follow `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
4. Use [`docs/runbook_host.md`](docs/runbook_host.md) for complete-corpus replay.

The canonical real-trace scope is `/root/sm120_rtx_5090_capture`, with 339
capture/profile pairs. Full replay is primary and fast frontier replay is
complementary. NCU remains an independent comparison surface. No synthetic
fixture acceptance path or Accel-Sim baseline is part of this alpha lane.

## Repository map

- `sim/` — Rust trace frontend, scheduler, memory model, event engine, runners.
- `arch_profiles/` — active RTX 5090 card and architecture schemas.
- `tools/nvbit_tracer/` — real trace and sidecar producer.
- `tools/sm120_corpus_replay.py` — full/fast corpus driver.
- `tools/sm120_replay_bench.py` — sidecar adapter and NCU join.
- `tools/sm120_corpus_compare.py` — endpoint comparison.
- `workloads/` — SOLBench kernel sources and real capture launchers.
- `docs/` — current implementation notes only.
