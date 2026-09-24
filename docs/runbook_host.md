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
cargo build --release -p sim-run
```

Replay the release binary directly after a change and inspect its JSON and
stderr. Run a package-specific test only when the changed behavior needs a
contract check.

## Build the NVBit tracer

The repository includes the NVBit source drop and enhanced protobuf sources.

```bash
make -C tools/nvbit_tracer -j"$(nproc)"
```

Override tool locations in the environment when the defaults are not suitable:

```bash
make -C tools/nvbit_tracer \
  CUDA_INSTALL_PATH=/usr/local/cuda-13.2 \
  PROTOC="$PWD/.toolchain/pb/bin/protoc" \
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

For a model-relative issue-resource sweep, append:

```bash
--hypothetical-issue-ports 2
```

The accepted range is `1..=4`. This changes only the modeled issue transaction
width, labels the JSON output as hypothetical, and does not modify the
architecture card or claim a physical GPU resource.

For the narrow production resource-profile bridge, use the explicit
trace-safe schema instead of a scalar override:

```bash
sim/target/release/sim-run \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --architecture-spec arch_profiles/specs/sm86_trace_issue_profile.json \
  --mode l1-l2
```

The bridge currently consumes only matching `alu`/`memory` issue widths from a
`trace_sass_v1` profile. Functional custom ISAs and multi-group specs fail
closed rather than being applied to a fixed SASS trace; latency, initiation,
in-flight, extra-resource, and operation-binding fields are rejected because
this bridge does not consume them.
