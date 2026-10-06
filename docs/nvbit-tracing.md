# NVBit tracing

The tracer in `tools/nvbit_tracer/` records dynamic SASS, per-lane addresses,
launch metadata, thread-block information, and optional phase information in
the enhanced protobuf format.

Build:

```bash
make -C tools/nvbit_tracer -j"$(nproc)"
```

Capture:

```bash
mkdir -p traces/example
LD_LIBRARY_PATH="$PWD/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
TRACES_FOLDER="$PWD/traces/example" \
ACTIVE_FROM_START=1 \
  /absolute/path/to/cuda-program [arguments]
```

Use `ACTIVE_FROM_START=1` for ordinary binaries that do not call
`cuProfilerStart`/`cuProfilerStop`. Otherwise NVBit can exit with status 0 but
accept no launches. Set it to `0` only when the workload deliberately emits
profiler markers and the capture is meant to be marker-gated.

The in-tree tracer needs the protobuf/Abseil libraries in `.toolchain/pb/lib`.
If they cannot be loaded, a CUDA application can exit successfully without
loading NVBit or producing a trace. Check the NVBit banner, accepted launches,
and instruction-bearing trace output rather than the application exit code alone.

Read:

```bash
tools/nvbit_tracer/tracer_tool/trace_printer traces/example
```

NCU and NVBit should normally be run in separate executions because both
instrument the target. Match them by workload, arguments, build, GPU, and
launch order recorded in ordinary experiment metadata.

Wire-format compatibility tests and bounds checks are normal correctness
checks. They should describe malformed data clearly, never create a lifecycle
state that prevents unrelated experiments.
