# Standalone NVBit tracer

This tree contains the NVBit instrumentation tool used to produce the enhanced
dynamic SASS/address traces consumed by the Rust simulator. The vendored NVBit
SDK, protobuf definitions, C++ producer, Python readers, metric extractor, and
host-side tests are retained.

## Layout

- `tracer_tool/`: full dynamic trace producer and `trace_printer`.
- `traces_enhanced/`: enhanced protobuf schema and C++ serialization code.
- `nvbit_release/`: vendored NVBit SDK.
- `isa_def/`: architecture opcode tables.
- `metric_extractor/`: compact per-launch metric extractor.
- `smoketest/`: small CUDA workloads and runtime checks.

## Build

A compatible CUDA toolkit, host C++ compiler, and protobuf installation are
required. A repository-local compatibility environment is available when
useful, but it is optional:

```bash
# Optional: provision and select the repository's compatibility toolchain.
bash tools/nvbit_tracer/pinned_build_env.sh --ensure
source tools/nvbit_tracer/pinned_build_env.sh

# Direct in-tree build.
make -C tools/nvbit_tracer -j"$(nproc)"
```

Tool paths can also be supplied directly:

```bash
make -C tools/nvbit_tracer \
  CUDA_INSTALL_PATH=/opt/cuda-13.2 \
  PROTOC=/path/to/protoc \
  NVCC_CCBIN=/usr/bin/g++-14
```

Useful targets:

```bash
make -C tools/nvbit_tracer test
make -C tools/nvbit_tracer clean
make -C tools/nvbit_tracer/metric_extractor ARCH=sm_86
```

The main outputs are:

- `tracer_tool/tracer_tool.so`
- `tracer_tool/trace_printer`
- `metric_extractor/metric_extractor.so`

## Capture

Run NCU and NVBit in separate executions because both instrument the target.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT="$REPO_ROOT/results/nvbit_capture/manual"
mkdir -p "$OUT"

LD_LIBRARY_PATH="$REPO_ROOT/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$OUT" \
  /absolute/path/to/cuda-program [arguments]
```

Common outputs include `dynamic_trace.pb`, `stats.csv`,
`extra_info/enhanced_execution_info.json`, allocation metadata, and
per-thread-block protobuf payloads.

Inspect a trace:

```bash
tools/nvbit_tracer/tracer_tool/trace_printer "$OUT"
```

Historical campaign consumers have additional compatibility parsers routed by
the opt-in `run-sol-campaign` skill. They are not part of direct capture.

## Runtime notes

- `CUDA_INJECTION64_PATH` is the supported injection mechanism.
- `TRACES_FOLDER` should name a writable output directory when
  `USER_DEFINED_FOLDERS=1`.
- The producer contains bounded phase-channel handshakes and writes partial
  phase records under noncanonical names if a flush times out.
- The module-origin sidecar records loaded-module paths and content identity for
  campaign joins. It is not a build or execution gate.
- The producer supports both `sm_86` and `sm_120`; opcode and architecture
  behavior must still be tested on the target GPU.
- Python protobuf readers under `workloads/simbench_bw/_pb_generated/` and
  Rust prost bindings under `sim/trace/src/pb/` consume the same wire schema.
  Update all consumers when the schema changes.

## Smoke test

```bash
bash tools/nvbit_tracer/smoketest/no_op_kernel_idempotency.sh
```

The script compiles a one-kernel probe when a CUDA compiler and GPU are
available, runs it under the current tracer, and checks that a nonempty trace is
produced.

For a two-GPU check, use [CROSS_ARCH_RUNBOOK.md](CROSS_ARCH_RUNBOOK.md).
