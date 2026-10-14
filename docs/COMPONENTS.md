# Components

| Component | Responsibility |
| --- | --- |
| `sim/trace` | NVBit protobuf reading, trace identity, and semantic sidecars |
| `sim/arch` | Architecture cards and functional executors |
| `sim/sm` | Warp/CTA replay, issue/readiness, barriers, and async state |
| `sim/mem` | Cache, memory request, and completion state |
| `sim/engine` | Dependency/resource event queue |
| `sim/direct_run` | Direct `sim-run` binary and prepared replay |
| `sim/run` | Runner and result emission |
| `tools/nvbit_tracer` | Real dynamic trace and sidecar producer |
| `tools/sm120_corpus_replay.py` | Complete-corpus full/fast driver |
| `tools/sm120_replay_bench.py` | Static-sidecar adapter and independent NCU parser/join |
| `tools/sm120_corpus_compare.py` | Post-replay endpoint comparison |
| `arch_profiles` | Active cards and architecture schemas |
| `workloads/sol_execbench_baselines` | SOLBench source kernels |

NCU/CUDA observations remain independent. Unsupported behavior is reported as a
conservative typed gap; no workload-specific delay is added to force a match.
