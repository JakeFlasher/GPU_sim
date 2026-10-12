# Components

| Component | Responsibility |
| --- | --- |
| `sim/trace` | NVBit protobuf reading, trace identity/coverage, semantic adapter |
| `sim/arch` | Native architecture cards and functional executors |
| `sim/sm` | Native warp/CTA replay, issue, readiness, barriers, async state |
| `sim/mem` | Native cache/memory request and completion state |
| `sim/engine` | Generic dependency/resource event queue |
| `sim/direct_run` | Direct `sim-run` binary and prepared replay |
| `sim/run` | Campaign runner, manifest handling, and emitters |
| `tools/nvbit_tracer` | Native dynamic trace and sidecar producer |
| `tools/sm120_*.py` | SM120 replay, feature, service, and corpus reporting tools |
| `arch_profiles` | Native cards, custom specs, programs, and schemas |
| `workloads/sol_execbench_baselines` | SOLBench kernel sources used for real captures |
| `eval` | Small non-acceptance comparison utilities |

NCU/CUPTI values remain independent observations.  Unavailable behavior is
reported as an explicit error or typed gap; no workload-specific delay is added
to manufacture a profiler match.
