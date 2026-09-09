# GPU simulator

This repository contains a trace driven NVIDIA GPU simulator. NVBit produces
SASS and memory traces, the Rust engine replays issue, synchronization, cache,
and memory events, and the runner emits JSON metrics for comparison with Nsight
Compute measurements.

The active target is Ampere SM86 (RTX 3090 Ti). SM120 support is exploratory.
NCU observations are comparison data only; they are never simulator inputs.

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
