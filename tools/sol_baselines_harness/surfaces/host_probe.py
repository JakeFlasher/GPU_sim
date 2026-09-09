"""Build a ``HostInfo`` for the orchestrator from local probes.

Read-only end-to-end: ``nvidia-smi`` for device identity (name,
compute capability, driver version); ``nvcc --version`` for the CUDA
toolchain version; ``cudaRuntimeGetVersion`` for the loaded runtime;
and filesystem checks for NVTX header + library presence (no link
attempted — the header/library probe is a directory scan only).

Used by the surfaces CLI to populate the orchestrator's host
fingerprint before driving any surface plugin.
"""

from __future__ import annotations

import datetime
import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from sol_baselines_harness.surfaces.base import HostInfo


def _run(cmd: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    """Run a subprocess and return ``(exit_code, stdout, stderr)``."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out: {' '.join(cmd)}"
    return proc.returncode, proc.stdout, proc.stderr


def _probe_nvcc_version() -> str:
    rc, stdout, _ = _run(["nvcc", "--version"])
    if rc != 0:
        return "unknown"
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", stdout)
    if match:
        return match.group(1)
    match = re.search(r"release (\d+\.\d+)", stdout)
    return match.group(1) if match else "unknown"


def _probe_cuda_runtime_version() -> str:
    """Read the CUDA runtime's linked version via ``cudaRuntimeGetVersion``.

    ``nvcc --version`` reports the compiler toolchain, which can differ from
    the runtime loaded by the evaluator. Keep this host-fingerprint field
    tied to the runtime API and return ``unknown`` when no usable cudart is
    available instead of substituting the compiler version.
    """
    candidates: List[str] = []
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(env_var)
        if base:
            candidates.extend(
                [
                    str(Path(base) / "lib64" / "libcudart.so"),
                    str(
                        Path(base)
                        / "targets"
                        / "x86_64-linux"
                        / "lib"
                        / "libcudart.so"
                    ),
                ]
            )
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_root = Path(nvcc).resolve().parent.parent
        candidates.append(
            str(nvcc_root / "targets" / "x86_64-linux" / "lib" / "libcudart.so")
        )
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        candidates.append(discovered)
    candidates.extend(["libcudart.so", "libcudart.so.13"])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            cudart = ctypes.CDLL(candidate)
            get_version = cudart.cudaRuntimeGetVersion
            get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
            get_version.restype = ctypes.c_int
            value = ctypes.c_int()
            if get_version(ctypes.byref(value)) != 0 or value.value <= 0:
                continue
            return f"{value.value // 1000}.{(value.value % 1000) // 10}"
        except (AttributeError, OSError):
            continue
    return "unknown"


def _probe_device() -> Tuple[str, str, str]:
    """Read device_name, compute_capability, driver_version from nvidia-smi.

    Returns ``("unknown", "unknown", "unknown")`` when no GPU is
    visible. The orchestrator still proceeds — every surface probe
    is responsible for its own availability check, and the
    ``unknown`` fingerprint surfaces in the manifest as a clear
    no-device signal rather than aborting the run.
    """
    rc, stdout, _ = _run([
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    ])
    if rc != 0 or not stdout.strip():
        return "unknown", "unknown", "unknown"
    line = stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return "unknown", "unknown", "unknown"
    return parts[0], parts[1], parts[2]


def _probe_nvtx_paths() -> Tuple[bool, bool]:
    """Detect NVTX header + library presence.

    Header probe scans ``$CUDA_HOME/include`` and the standard
    ``/usr/local/cuda*/include`` paths for ``nvtx3/`` (NVTX3) or
    ``nvToolsExt.h`` (legacy). Library probe scans the corresponding
    ``lib64`` and ``/usr/lib/x86_64-linux-gnu`` paths for
    ``libnvToolsExt`` (legacy) or ``libnvtx``.

    Returns ``(header_present, library_present)``. The orchestrator
    records both bits separately in the manifest (negative test
    requires that a host missing either bit produces a build without
    NVTX symbols).
    """
    include_roots: List[Path] = []
    lib_roots: List[Path] = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        include_roots.append(Path(cuda_home) / "include")
        lib_roots.append(Path(cuda_home) / "lib64")
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_root = Path(nvcc).resolve().parent.parent
        include_roots.append(nvcc_root / "include")
        lib_roots.append(nvcc_root / "lib64")
    include_roots.extend([
        Path("/usr/local/cuda/include"),
        Path("/usr/include"),
        Path("/usr/local/include"),
    ])
    lib_roots.extend([
        Path("/usr/local/cuda/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
    ])
    header_present = False
    for root in include_roots:
        if not root.is_dir():
            continue
        if (root / "nvtx3").is_dir():
            header_present = True
            break
        if (root / "nvToolsExt.h").is_file():
            header_present = True
            break
        if (root / "nvtx3" / "nvToolsExt.h").is_file():
            header_present = True
            break
    library_present = False
    for root in lib_roots:
        if not root.is_dir():
            continue
        for candidate in (
            "libnvToolsExt.so",
            "libnvToolsExt.so.1",
        ):
            if (root / candidate).exists():
                library_present = True
                break
        if library_present:
            break
    return header_present, library_present


def _classify_host_kind(compute_capability: str) -> str:
    """Translate a compute capability into the manifest's ``host.kind``."""
    if compute_capability.startswith("10."):
        return "b200_sm_100a"
    if compute_capability.startswith("12."):
        return "local_sm_120"
    if compute_capability.startswith("9."):
        return "local_sm_90"
    if compute_capability == "unknown":
        return "unknown"
    return f"local_sm_{compute_capability.replace('.', '_')}"


def probe_host() -> HostInfo:
    """Assemble a ``HostInfo`` from local read-only probes."""
    device_name, compute_capability, driver_version = _probe_device()
    nvcc_version = _probe_nvcc_version()
    runtime_version = _probe_cuda_runtime_version()
    nvtx_header, nvtx_library = _probe_nvtx_paths()
    return HostInfo(
        kind=_classify_host_kind(compute_capability),
        device_name=device_name,
        compute_capability=compute_capability,
        driver_version=driver_version,
        cuda_runtime_version=runtime_version,
        captured_at_utc=datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        nvtx_header_present=nvtx_header,
        nvtx_library_present=nvtx_library,
    )
