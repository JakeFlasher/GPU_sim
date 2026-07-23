# Host NVBit build and runtime runbook

This runbook covers the certified `sm_86` development host and the split
toolchain required when CUDA 13.3 is the system default but the vendored NVBit
tracer must be built with CUDA 13.2.

## Certified host matrix

| Surface | Pinned/observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti, `sm_86` |
| Driver | 610.43.02 |
| Default CUDA compiler | 13.3 under `/opt/cuda` |
| Pinned NVBit build CUDA | 13.2 under `/usr/local/cuda-13.2` |
| NVBit | 1.8 vendored core |
| Project Python | 3.12.12 in `.venv` |
| Protobuf compiler | pinned protoc 34.1 under `.toolchain/pb` |

NVBit 1.8 carries a CUDA 13.2 header/API snapshot; this repository's custom
tracer is also built with nvcc 13.2 because nvcc 13.3 fails on its pinned
protobuf/Abseil headers. That build pin does not require target workloads to be
compiled with CUDA 13.2: CUDA 13.3 remains the system default, and a CUDA
13.3-built sm_86 target passed the live trace contract on this host. NVBit's
published `driver <=575.xx` line has not been formally corrected, so driver
610.43.02 is locally verified outside the published envelope, not claimed as
official NVBit support.

Check the live host before a capture:

```bash
nvidia-smi
nvcc --version
/usr/local/cuda-13.2/bin/nvcc --version
.venv/bin/python --version
```

Any mismatch in the pinned build tools is a build refusal, not an invitation to
weaken the version checks.

## Compiler-environment hazard

`/etc/profile.d/cuda.sh` exports `NVCC_CCBIN=/usr/bin/g++-15`. The certified
campaign uses g++-14, and capture preflight treats ambient compiler overrides
as hazardous. Always enter capture commands through:

```bash
tools/local_labels/run_sm86_capture_clean_env.sh COMMAND [ARG ...]
```

The wrapper removes `NVCC_CCBIN`, `NVCC_PREPEND_FLAGS`, and
`NVCC_APPEND_FLAGS` before the campaign constructs its compiler environment.
Do not replace the wrapper with a shell alias or hand-edited profile.

## Pinned NVBit/protobuf build

The checked-in `.pc` metadata was created in an older checkout. The Makefiles
now relocate it through `pkg-config --define-prefix`, and the sourced
environment refuses a version or checkout-prefix mismatch:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
mamba create -y -p "$REPO_ROOT/.toolchain/pb" \
  -c conda-forge libprotobuf=7.34.1  # one time per checkout
cd "$REPO_ROOT"
source tools/nvbit_tracer/pinned_build_env.sh
pkg-config --define-prefix --variable=prefix protobuf
```

The printed prefix must resolve under this repository's `.toolchain/pb`, not
`/home/jakeshea/non_root_profiler/`.

Rebuild the trace wire library and tracer on `sm_86`. Clean first; the single
final build recursively regenerates every protobuf binding with the pinned
compiler:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer clean
make -C tools/nvbit_tracer ARCH=sm_86
make -C workloads/simbench_bw nvbit-pb-python
```

Rebuild active metric-extractor instrumentation from source:

```bash
make -C tools/nvbit_tracer/metric_extractor clean
make -C tools/nvbit_tracer/metric_extractor \
  ARCH=sm_86 NVCC_CCBIN=/usr/bin/g++-14
make -C tools/nvbit_tracer/metric_extractor run-test-classifier
make -C tools/nvbit_tracer/metric_extractor run-test-saturating-counter
make -C tools/nvbit_tracer/metric_extractor run-test-tier2-sha256
make -C tools/nvbit_tracer/metric_extractor run-test-tier2-marker-metadata
cuobjdump --list-elf tools/nvbit_tracer/metric_extractor/metric_extractor.so
```

The final listing must contain the intended host architecture. A missing,
stale, or wrong-architecture `.so` is a hard stop. Generated `.so`, `.o`, and
host-test executables are not source artifacts and should not be committed.

At runtime, the tracer needs the pinned protobuf libraries. Sourcing the build
environment supplies them; a fresh CUDA 13.3 workload shell may instead add
only `$PWD/.toolchain/pb/lib` to `LD_LIBRARY_PATH`:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
ldd tools/nvbit_tracer/tracer_tool/tracer_tool.so
```

An `ldd` pass alone does not prove injection. The campaign preflight must also
produce a non-empty, contract-valid smoke trace.
