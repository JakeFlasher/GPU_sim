# NVBit tracer

This tree builds the NVBit instrumentation producer used by the Rust trace
reader. It contains the tracer, protobuf schema, opcode tables, and metric
extractor.

## Build

For the repository-pinned CUDA/protobuf stack, initialize the validated host
toolchain first. This is important when the shell exports an incompatible
`NVCC_CCBIN` (for example GCC 16 with CUDA 13.2):

```bash
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer -j"$(nproc)"
```

The environment script checks CUDA 13.2, GCC 14, protoc/protobuf 34.x, and
exports the protobuf runtime path needed by `tracer_tool.so` and
`trace_printer`. If using a different toolchain, override the variables
explicitly rather than inheriting an ambient `NVCC_CCBIN`.

Without the pinned environment, the equivalent SM86 build is:

```bash
make -C tools/nvbit_tracer \
  CUDA_INSTALL_PATH=/usr/local/cuda-13.2 \
  PROTOC="$PWD/.toolchain/pb/bin/protoc" \
  NVCC_CCBIN=/usr/bin/g++-14 \
  -j"$(nproc)"
```

Override `CUDA_INSTALL_PATH`, `PROTOC`, or `NVCC_CCBIN` when the host uses a
different toolkit. The main outputs are
`tracer_tool/tracer_tool.so` and `tracer_tool/trace_printer`.
Build the metric extractor separately with
`make -C tools/nvbit_tracer/metric_extractor`.

## Capture

Keep the pinned environment active in the capture shell so protobuf/Abseil
runtime libraries resolve for both the injected tracer and `trace_printer`:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT="/root/sm120_rtx_5090_capture/captures/manual"
mkdir -p "$OUT"
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$OUT" \
  /absolute/path/to/cuda-program [arguments]
```

A capture normally contains `dynamic_trace.pb`, `stats.csv`, launch metadata,
allocation metadata, and per-thread-block protobuf payloads. Inspect it with
`tracer_tool/trace_printer "$OUT"`.

Set `SIMBENCH_NVBIT_ASYNC_ORDER=1` to retain host creation order for selected
asynchronous-copy instructions. It is an ordering token, not a GPU timestamp.

Synchronization-after sidecar rows emitted by the SM120 tracer carry
`query_clock`, a `%clock64` sample at the instrumented query site. The tracer
does not label this as acquire visibility because relaxed and acquire
`mbarrier` queries can lower to the same `PHASECHK` SASS mnemonic. Older
captures may contain `visibility_clock`; consumers must treat that legacy field
as a query-site sample unless the query semantics are independently identified.

The Rust reader and Python consumers share the protobuf definitions. Update both
generated surfaces when the wire schema changes.
