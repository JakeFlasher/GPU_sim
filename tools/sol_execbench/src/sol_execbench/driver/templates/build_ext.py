# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
from pathlib import Path

import torch.utils.cpp_extension as ext

from sol_execbench.core import Solution

HERE = Path.cwd().resolve()

# Parse solution — validates sources (e.g. forbidden keywords) at compile time.
solution = Solution(**json.loads((HERE / "solution.json").read_text()))
compile_options = solution.spec.compile_options

CERTIFIED_ENVIRONMENT = {
    "CUDA_HOME": "/usr/local/cuda-13.2",
    "NVCC_CCBIN": "g++-14",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
}


def _validated_compiler_binding() -> str:
    observed = {name: os.environ.get(name) for name in CERTIFIED_ENVIRONMENT}
    if observed != CERTIFIED_ENVIRONMENT:
        raise RuntimeError(
            "SOL compiler environment is not certified: "
            f"expected={CERTIFIED_ENVIRONMENT!r} observed={observed!r}"
        )
    cuda_home = Path(CERTIFIED_ENVIRONMENT["CUDA_HOME"]).resolve()
    expected_nvcc = (cuda_home / "bin" / "nvcc").resolve()
    observed_nvcc = shutil.which("nvcc")
    if (
        observed_nvcc is None
        or Path(observed_nvcc).resolve() != expected_nvcc
        or not expected_nvcc.is_file()
        or not os.access(expected_nvcc, os.X_OK)
        or ext.CUDA_HOME is None
        or Path(ext.CUDA_HOME).resolve() != cuda_home
    ):
        raise RuntimeError(
            "SOL nvcc does not resolve from certified CUDA_HOME: "
            f"expected={expected_nvcc} observed={observed_nvcc!r} "
            f"torch_cuda_home={ext.CUDA_HOME!r}"
        )
    host_cxx = CERTIFIED_ENVIRONMENT["NVCC_CCBIN"]
    observed_host_cxx = shutil.which(host_cxx)
    if (
        observed_host_cxx is None
        or not Path(observed_host_cxx).resolve().is_file()
        or not os.access(Path(observed_host_cxx).resolve(), os.X_OK)
    ):
        raise RuntimeError(f"certified SOL host compiler is absent: {host_cxx}")
    return host_cxx


def _certified_cuda_cflags(
    requested_flags, *, host_cxx: str | None = None,
) -> list[str]:
    flags = list(requested_flags)
    if any(
        flag in {"-ccbin", "--compiler-bindir"}
        or flag.startswith("-ccbin=")
        or flag.startswith("--compiler-bindir=")
        for flag in flags
    ):
        raise RuntimeError("solution cuda_cflags may not override certified -ccbin")
    validated_host = _validated_compiler_binding()
    if host_cxx is not None and host_cxx != validated_host:
        raise RuntimeError(
            "SOL host-compiler binding changed between validation and assembly"
        )
    flags.extend(["-ccbin", validated_host])
    return flags


def _prepare_torch_extension_environment(*, host_cxx: str) -> None:
    """Prevent PyTorch from injecting a second, conflicting NVCC ``-ccbin``.

    PyTorch translates the ambient ``CC`` variable into its own leading
    ``-ccbin`` even when ``extra_cuda_cflags`` already contains the certified
    binding.  The exact seven-key campaign environment is validated first;
    then only this build-process-local ``CC`` copy is removed.  ``CXX`` and
    ``CUDAHOSTCXX`` remain pinned to g++-14, while the explicit NVCC flag below
    is the sole host-compiler binding in the generated command.
    """
    if host_cxx != CERTIFIED_ENVIRONMENT["NVCC_CCBIN"]:
        raise RuntimeError("SOL certified host compiler changed before build")
    if os.environ.get("CC") != CERTIFIED_ENVIRONMENT["CC"]:
        raise RuntimeError("SOL certified CC changed before build")
    if (
        os.environ.get("CXX") != host_cxx
        or os.environ.get("CUDAHOSTCXX") != host_cxx
    ):
        raise RuntimeError("SOL certified C++ compiler changed before build")
    os.environ.pop("CC")


def _validated_cutlass_dir() -> Path:
    raw = os.environ.get("CUTLASS_DIR")
    if not raw:
        raise RuntimeError(
            "CUTLASS_DIR is required; the build harness must provide the "
            "hash-pinned CUTLASS tree"
        )
    cutlass_dir = Path(raw).expanduser().resolve()
    required = (
        cutlass_dir / "include" / "cutlass" / "version.h",
        cutlass_dir / "tools" / "util" / "include",
    )
    if not cutlass_dir.is_dir() or any(not path.exists() for path in required):
        raise RuntimeError(
            "CUTLASS_DIR does not identify a usable pinned tree: "
            f"{cutlass_dir}"
        )
    return cutlass_dir


# set flags
host_cxx = _validated_compiler_binding()
cuda_cflags = _certified_cuda_cflags(
    compile_options.cuda_cflags if compile_options else [], host_cxx=host_cxx,
)
_prepare_torch_extension_environment(host_cxx=host_cxx)
cflags = compile_options.cflags if compile_options else []
ld_flags = compile_options.ld_flags if compile_options else []

# Collect C/C++/CUDA source files from current directory
sources = [
    str(p)
    for p in HERE.iterdir()
    if p.suffix in (".cu", ".cpp", ".cc", ".cxx", ".c") and p.is_file()
]
if not sources:
    raise RuntimeError("No CUDA/C++ source files found in working directory")

cutlass_dir = _validated_cutlass_dir()
extra_include_paths = [
    str(HERE),
    f"{cutlass_dir}/include",
    f"{cutlass_dir}/tools/util/include",
]

ext.load(
    name="benchmark_kernel",
    sources=sources,
    extra_cuda_cflags=cuda_cflags,
    extra_cflags=cflags,
    extra_ldflags=ld_flags,
    extra_include_paths=extra_include_paths,
    build_directory=str(HERE),
    verbose=True,
)

# Rename platform-suffixed .so → benchmark_kernel.so
so_files = [
    f for f in HERE.glob("benchmark_kernel*.so") if f.name != "benchmark_kernel.so"
]
if so_files:
    so_files[0].rename("benchmark_kernel.so")
elif not (HERE / "benchmark_kernel.so").exists():
    raise FileNotFoundError("benchmark_kernel.so not produced by compilation")

print("benchmark_kernel.so ready", flush=True)
