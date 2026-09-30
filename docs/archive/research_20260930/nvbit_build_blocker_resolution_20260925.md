# NVBit tracer build blocker resolution — 2026-09-25

## Failure reproduced

The host shell exported `NVCC_CCBIN=/usr/bin/g++`, which resolved to GCC
16.2.1. CUDA 13.2 rejects host compilers newer than GCC 15 before compiling
the tracer. Retrying with `-allow-unsupported-compiler` is not a valid fix:
the CUDA 13.2 front end then rejects GCC 16 C++23 library headers
(`this`-deducing members and `__builtin_is_virtual_base_of`).

## Supported build

The repository already carries a pinned environment contract. This follows
NVIDIA's NVCC guidance that the host compiler is selected through `NVCC_CCBIN`
and must be compatible with the toolkit ([CUDA 13.2 NVCC documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html)). It validates:

* CUDA/NVCC 13.2;
* `/usr/bin/g++-14` (14.3.1 on the reviewed host);
* protoc/protobuf 34.1 and the repository `.toolchain/pb` prefix; and
* the protobuf/Abseil runtime search path.

Use:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer -j"$(nproc)"
```

An equivalent explicit invocation is:

```bash
make -C tools/nvbit_tracer \
  CUDA_INSTALL_PATH=/usr/local/cuda-13.2 \
  PROTOC="$PWD/.toolchain/pb/bin/protoc" \
  NVCC_CCBIN=/usr/bin/g++-14 \
  -j"$(nproc)"
```

## Evidence

The checked-in tracer changes compiled successfully with the pinned command:

* `tools/nvbit_tracer/tracer_tool/tracer_tool.so` produced as an ELF shared
  object;
* `tools/nvbit_tracer/tracer_tool/trace_printer` produced as an ELF executable;
* `make -C tools/nvbit_tracer/traces_enhanced` regenerated the protobuf object
  surface; and
* `ldd` resolved protobuf/Abseil dependencies after sourcing the pinned
  environment.

The CUDA/NVBit no-op idempotency smoke also passed under the same environment:
the probe exited 0 and produced both `traces/dynamic_trace.pb` and a nonempty
threadblock payload. This validates injection/runtime loading in addition to
host compilation.

Running `ldd` without that environment reports the expected protobuf/Abseil
libraries as not found; this is a runtime environment issue, not a CUDA source
compile failure. The README and host runbook now make the required setup
explicit.

The build emits compatibility warnings from the pinned protobuf/Abseil headers;
no source compile or link error remains under the supported GCC 14/CUDA 13.2
pair. No simulator or trace semantics were changed by this resolution.
