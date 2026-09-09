# Components

| Component | Responsibility |
| --- | --- |
| `sim/trace` | Decode NVBit protobuf traces and control sidecars. |
| `sim/arch` | Load architecture cards and capability selections. |
| `sim/sm` | Replay warps and CTAs, dependency readiness, issue, barriers, and async operations. |
| `sim/mem` | Model L1/TEX, shared L2, request ordering, and completion state. |
| `sim/engine` | Advance deterministic time ordered events. |
| `sim/run` | Compose the manifest-driven `sim-run-campaign` CLI and JSON output. |
| `sim/direct_run` | Direct replay convenience entry point. |
| `tools/nvbit_tracer` | Produce the dynamic trace consumed by `sim-trace`. |
| `tools/local_labels` | Collect and parse independent NCU observations. |
| `workloads/research_probes` | Focused CUDA inputs and capture helpers. |
| `arch_profiles` | Device-specific geometry and capability data. |
| `eval/direct_score.py` | Direct numeric comparison of JSON observations and predictions. |

Keep architecture values in cards and keep NCU observations outside replay.
When a trace cannot identify an endpoint, return typed incomplete state instead
of adding a workload-specific delay.
