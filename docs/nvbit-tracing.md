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
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
TRACES_FOLDER="$PWD/traces/example" \
  /absolute/path/to/cuda-program [arguments]
```

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
