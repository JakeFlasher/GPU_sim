# Components

| Component | Responsibility |
| --- | --- |
| `sim/trace` | NVBit protobuf reading, trace identity/coverage, semantic adapter |
| `sim/arch` | Native architecture capabilities/cards, custom specs and functional executors |
| `sim/arch::simt_pipeline` | Fixed custom ISA -> register/memory dependency and resource tasks |
| `sim/sm` | Native warp/CTA replay, issue, readiness, barriers, async state |
| `sim/sm::custom` | Custom group residency, address-space state, bounded barrier ordering |
| `sim/mem` | Native L1/TEX/L2, memory requests and completion state |
| `sim/engine` | Event queue; separate generic dependency/resource pipeline |
| `sim/direct_run` | Package/binary `sim-run`, library `sim_direct_run`, prepared replay |
| `sim/run` | Retained package/binary `sim-run-campaign`, library `sim_run`, shared emitters |
| `tools/nvbit_tracer` | Produce native dynamic trace evidence |
| `tools/traceg_adapter_probe` | Fail-closed SM86 common-subset protobuf-to-legacy-`.traceg` representation probe; not a timing frontend |
| `tools/local_labels` | Collect/parse independent NCU observations |
| `tools/custom_simt_differential.py` | Separately implemented bounded functional/timing reference; not hardware truth |
| `arch_profiles` | Native cards, hypothetical specs, custom programs, schemas |
| `workloads`, `eval` | Executable workloads, probes, and comparison tools |

The native and custom paths share infrastructure but not a complete hardware
model. Unavailable behavior remains an explicit error or typed gap; do not add
workload-specific delays to manufacture a profiler match.
