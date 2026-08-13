# Event-driven NVIDIA GPU simulator

This repository develops a fast, trace-driven simulator for modern NVIDIA
GPUs. The Rust simulator consumes NVBit SASS/address traces and static SASS
control information. Nsight Compute (NCU) measurements from real GPUs provide
independent comparison targets.

The active target is Ampere `sm_86` on an RTX 3090 Ti. Blackwell `sm_120`
support remains exploratory until the Ampere path is useful end to end.

## Working principles

Development is research-first. CUDA, NCU, NVBit, GPU execution, clock controls,
and host administration commands may be used directly when they help answer the
technical question. Ordinary tests, schema validation, numerical sanity checks,
and repeated measurements are welcome. They report facts; they do not grant
permission to continue.

Project workflow policy is in [AGENTS.md](AGENTS.md). In particular, this
repository does not use review swarms, external-advisor loops, artifact
approval ceremonies, cryptographic workflow checks, or phase locks.

The complete SOL-ExecBench and historical campaign stack remains available as
an opt-in research capability. Its legacy manifests and checksums are scoped
experiment data, not repository-wide permission gates.

## Data path

```text
CUDA workload
  ├─ NVBit tracer -> dynamic protobuf trace -> sim-trace
  │                                      -> sim-sm
  │                                      -> sim-mem
  │                                      -> sim-engine
  │                                      -> sim-run predictions
  └─ NCU capture --------------------------------------> eval comparison
```

NCU values are comparison targets, not simulator inputs.

## Quick start

```bash
cd sim
cargo test --all-targets
cd ..
.venv/bin/python -m pytest -q
```

Build and run the GPU tools with the commands in
[docs/runbook_host.md](docs/runbook_host.md). The technical roadmap is
[plans/micro2027_ncu_sim_execution_plan.md](plans/micro2027_ncu_sim_execution_plan.md).

## Repository map

| Path | Purpose |
|---|---|
| `sim/` | Rust event engine, trace reader, SM replay, memory model, and runner |
| `tools/nvbit_tracer/` | NVBit trace producer and protobuf format |
| `tools/local_labels/` | Direct NCU collection and metric parsing |
| `tools/sol_execbench/` | Retained SOL-ExecBench source, environment, and data |
| `tools/sol_baselines_harness/` | Retained optional SOL analytics/campaign stack |
| `workloads/` | CUDA workloads used for capture and experiments |
| `arch_profiles/` | Architecture and SKU parameters |
| `eval/` | Direct numerical comparison of predictions and observations |
| `docs/` | Current architecture, status, and host commands |
| `plans/` | Current simulator and capture roadmaps |
