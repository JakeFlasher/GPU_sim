# GPU simulator

This repository contains a trace driven NVIDIA GPU simulator. NVBit produces
SASS and memory traces, the Rust engine replays issue, synchronization, cache,
and memory events, and the runner emits JSON metrics for comparison with Nsight
Compute measurements.

The SM crate also exposes an opt-in causal-frontier descriptor analysis. It
uses the native `PipelineClass` and control schedule, keeps resource and
completion envelopes explicit, and refuses missing dependency/completion facts;
it does not alter the established replay path.

The active target is Ampere SM86 (RTX 3090 Ti). SM120 support is exploratory
only when a local trace/card exists. NCU observations are comparison data only;
they are never simulator inputs. RTL, Verilator, FPGA, physical-design, and
silicon-internal simulation are outside the project scope.

The [research and positioning memo](docs/gpu_simulator_research_positioning_20260923.md)
records the literature boundary and paper agenda. The [general GPU simulator
roadmap](plans/cross_architecture_gpu_simulator_research_plan_20260923.md)
adds a staged user-defined architecture path alongside this SM86 validation
anchor. The CurryGPU functional-architecture study and bounded per-lane-PC ITS
example are documented in
[docs/curry_gpu_its_study_20260923.md](docs/curry_gpu_its_study_20260923.md).
Configurability, functional execution, and cross-architecture validation are
work in progress, not current claims of arbitrary GPU support.

## Data path

```text
CUDA workload -> NVBit trace -> sim-trace -> sim-sm -> sim-mem -> sim-run
                                      \-> direct NCU comparison in eval/
```

## Repository map

| Path | Role |
| --- | --- |
| `sim/` | Rust workspace: architecture cards, trace reader, SM replay, memory model, runner |
| `tools/nvbit_tracer/` | NVBit producer and protobuf support |
| `tools/local_labels/` | Direct NCU capture and metric parsing |
| `workloads/` | CUDA workloads and focused probes |
| `arch_profiles/` | Architecture inputs |
| `eval/direct_score.py` | Small JSON metric comparator |
| `docs/` | Current architecture, status, and run commands |
| `plans/` | Current execution roadmaps |
| `data/`, `references/` | Retained source data and research material |

## Fast path

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
python -m eval.direct_score --observed observed.json --predicted predicted.json
```

For a GPU run, use the commands in [docs/runbook_host.md](docs/runbook_host.md).
The current engine status and retained SM86 comparison roots are in
[docs/STATUS.md](docs/STATUS.md).
The historical cross-architecture review and bounded residuals are in
[docs/adversarial_code_review_20260923.md](docs/adversarial_code_review_20260923.md);
it is not an active scope or validation gate.

Historical clock-qualified small/large campaigns and same-input regression
comparisons are in
[docs/postmerge_campaign_comparison_20260909.md](docs/postmerge_campaign_comparison_20260909.md).

The prototype integration and its NCU sector probe are documented in
[docs/prototype_causal_frontier.md](docs/prototype_causal_frontier.md).
