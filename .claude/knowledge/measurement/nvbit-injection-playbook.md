---
Use when: building or modifying the NVBit tracer, debugging injection, planning SASS instrumentation, or running clock/phase probes.
Primary source: `tools/nvbit_tracer/README.md`.
---

# NVBit injection playbook

NVBit is a user-space dynamic instrumentation framework. This repository builds
and runs it directly; no H3 hold, temporary-copy ritual, artifact pin, or
reviewer handoff applies.

## Build

```bash
# Optional repository-local CUDA/protobuf environment.
bash tools/nvbit_tracer/pinned_build_env.sh --ensure
source tools/nvbit_tracer/pinned_build_env.sh

make -C tools/nvbit_tracer -j"$(nproc)"
make -C tools/nvbit_tracer test
```

The Makefiles also accept ordinary `CUDA_INSTALL_PATH`, `PROTOC`,
`PKG_CONFIG`, and `NVCC_CCBIN` overrides.

## Inject

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
TRACE_ROOT="$REPO_ROOT/results/nvbit_capture/manual"
mkdir -p "$TRACE_ROOT"

LD_LIBRARY_PATH="$REPO_ROOT/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$TRACE_ROOT" \
  ./your_cuda_app
```

Use `CUDA_INJECTION64_PATH` as the injection mechanism. If the tracer links
against the repository-local protobuf runtime, keep that library directory on
`LD_LIBRARY_PATH`.

A missing NVBit banner and empty trace usually mean the loader could not resolve
a dependency. Check with `ldd tracer_tool.so`, inspect stderr, and confirm the
output directory is writable.

## Inspect

```bash
tools/nvbit_tracer/tracer_tool/trace_printer "$TRACE_ROOT"
```

Historical campaign compatibility parsers are routed by `run-sol-campaign` and
are not part of ordinary capture.

## Measurement discipline

Run NCU, Nsight Systems, CUPTI injection, and NVBit in separate matched
executions. Each instrument can perturb or conflict with the others. Match
workload build, arguments, GPU, and launch order when comparing outputs.

NVBit per-instruction tracing can be very slow. Prefer bounded launches,
sampling, or short microbenchmarks when that answers the question.

## Architecture notes

The tracer includes an `sm_120` binary-version fallback. Exercise
architecture-specific opcode and wire changes on the relevant GPU rather than
assuming `sm_86` behavior transfers unchanged.

Relevant sources:

- `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`
- `tools/nvbit_tracer/tracer_tool/inject_funcs.cu`
- `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`
- `workloads/simbench_bw/`
