# NVBit tracer

This tree builds the NVBit instrumentation producer used by the Rust trace
reader. It contains the tracer, protobuf schema, opcode tables, metric
extractor, and small CUDA smoke workloads.

## Build

```bash
make -C tools/nvbit_tracer -j"$(nproc)"
```

Override `CUDA_INSTALL_PATH`, `PROTOC`, or `NVCC_CCBIN` when the host uses a
different toolkit. The main outputs are
`tracer_tool/tracer_tool.so` and `tracer_tool/trace_printer`.
Build the metric extractor separately with
`make -C tools/nvbit_tracer/metric_extractor`.

## Capture

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT="$REPO_ROOT/results/captures/manual"
mkdir -p "$OUT"
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$OUT" \
  /absolute/path/to/cuda-program [arguments]
```

A capture normally contains `dynamic_trace.pb`, `stats.csv`, enhanced launch
metadata, allocation metadata, and per-thread-block protobuf payloads. Inspect
it with `tracer_tool/trace_printer "$OUT"`.

Set `SIMBENCH_NVBIT_ASYNC_ORDER=1` to retain the optional host creation order
for selected asynchronous-copy instructions. It is an ordering token, not a
GPU timestamp.

The Rust reader and Python consumers share the protobuf definitions. Update
both generated surfaces when the wire schema changes.
