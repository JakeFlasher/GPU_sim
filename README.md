# GPU simulator

Rust trace/replay engine and NVBit producer for the SM120/RTX 5090 effective
model, with an SM86 compatibility lane.

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
NCU/CUDA observations ---------------------------------------> post-replay join
```

## Start here

1. Read [`AGENTS.md`](AGENTS.md) for local execution boundaries.
2. Read [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) and
   [`docs/STATUS.md`](docs/STATUS.md) for the current contract.
3. Read the external implementation plan at
   `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
4. Use [`docs/runbook_host.md`](docs/runbook_host.md) for a complete-corpus run.

The canonical real-trace scope is `/root/sm120_rtx_5090_capture`, with 339
capture/profile pairs. Full replay is primary and the fast frontier is
complementary. NCU remains an independent comparison surface. There is no
synthetic fixture acceptance path or Accel-Sim baseline in this lane.

## Repository map

- `sim/` — Rust trace frontend, scheduler, memory model, event engine, runners.
- `arch_profiles/` — architecture cards and schemas.
- `tools/nvbit_tracer/` — real trace and sidecar producer.
- `tools/sm120_corpus_replay.py` — complete-corpus full/fast driver.
- `tools/sm120_replay_bench.py` — sidecar adapter and NCU join.
- `tools/sm120_corpus_compare.py` — endpoint comparison.
- `workloads/` — SOLBench kernels and real capture launchers.
- `docs/` — current architecture, status, and runbook documentation.
