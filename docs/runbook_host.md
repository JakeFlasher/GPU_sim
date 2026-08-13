# Host runbook

Commands below are direct. Set paths for the installed tools on the current
host; exact patch versions are not workflow requirements.

## Inspect the host

```bash
nvidia-smi
nvcc --version
ncu --version
protoc --version
```

Administrative access may be used when needed:

```bash
sudo -n true
```

## Build the Rust simulator

```bash
cd sim
cargo build
cargo test --all-targets
```

## Build the NVBit tracer

The repository includes the NVBit source drop and enhanced protobuf sources.

```bash
make -C tools/nvbit_tracer -j"$(nproc)"
```

Override tool locations in the environment when the defaults are not suitable:

```bash
make -C tools/nvbit_tracer \
  CUDA_INSTALL_PATH=/opt/cuda-13.2 \
  PROTOC=/path/to/protoc \
  NVCC_CCBIN=/usr/bin/g++-14
```

## Capture an NVBit trace

```bash
mkdir -p traces/my_run
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
TRACES_FOLDER="$PWD/traces/my_run" \
  /absolute/path/to/workload [workload arguments]
```

Inspect it with:

```bash
tools/nvbit_tracer/tracer_tool/trace_printer traces/my_run
```

## Collect NCU metrics

```bash
.venv/bin/python tools/local_labels/collect_ncu.py \
  --output results/my_run \
  --metrics gpc__cycles_elapsed.max,smsp__inst_executed.sum \
  -- /absolute/path/to/workload [workload arguments]
```

The command preserves the NCU report, exported CSV, command, return code, and
captured stdout/stderr. It does not require a pre-existing descriptor or
approval document.

## Compare predictions and observations

```bash
.venv/bin/python -m eval.direct_score \
  --observed observed.json \
  --predicted predicted.json
```

Use repeated runs when variance matters. Record tool versions and GPU state in
experiment notes when they affect interpretation.
