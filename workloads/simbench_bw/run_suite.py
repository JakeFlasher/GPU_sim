#!/usr/bin/env python3
"""SimBench-BW orchestrator.

Dispatches per-kernel runs across the roster in one of four modes:

    correctness  Run the kernel binary + its correctness reference; compare outputs
                 under the tolerance triplet keyed on (dtype, reduction_kind,
                 kernel_family).
    timing       Run the kernel binary with warm-up + repeat; record median,
                 CV, and a measurement_manifest.json.
    trace        Invoke the NVBit tracer pipeline, pre-gated by trace_budget_mib
                 and the per-shape predicted_bytes. A run that would exceed budget
                 exits with code 3 ("sampling_required") and writes a
                 trace_volume_exceeded.json sidecar.
    regression   Selective rerun based on a PR diff or explicit --kernel list;
                 compares against workloads/simbench_bw/golden/.

Startup is gated by a device-identity check via cudaGetDeviceProperties (through
the local libcudart.so). The identity fields are written into every
measurement_manifest.json so downstream tooling can trace a run back to the
exact GPU, driver, and CUDA toolkit that produced it.

Exit codes:
    0   success
    2   usage error (invalid mode, missing kernel, etc.)
    3   sampling_required (trace would exceed declared budget)
    4   unsupported (kernel's arch_gate not satisfied on the current device)
    5   correctness_failed (reference comparison outside tolerance)
    6   build_failed (nvcc failure while compiling the kernel)
    7   golden_mismatch (regression mode: output differs from committed golden)
    8   unknown_target_device (startup identity check refused the device)
    9   dry_run_not_supported_for_mode (--dry-run-cpu used with a GPU-only mode)
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import csv
import hashlib
import json
import os
import sys
import tomllib
from ctypes import c_char, c_int, c_size_t, POINTER, byref, c_uint
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = SUITE_ROOT / "schemas"
REPO_ROOT = SUITE_ROOT.parents[1]

# Allow `from arch_resolver import ...` below whether the module is executed
# as a script (Python auto-adds the script dir to sys.path) or imported via
# importlib from a test harness (which does not).
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    TracerRuntimeError,
    tracer_runtime_env,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmissionError,
    require_trace_admission,
)

# C-AC-H budget constants come from `tools/cloud/trace_budgets.py`
# (Loop 2 R11). Add the cloud directory to sys.path so the import
# works whether run_suite.py is executed as a script or imported by
# pytest. The import is deliberately not at top-of-file because the
# `tools/cloud/` directory is not on sys.path by default; this
# follows the same pattern as the SUITE_ROOT insertion above.
_TOOLS_CLOUD_DIR = SUITE_ROOT.parent.parent / "tools" / "cloud"
if str(_TOOLS_CLOUD_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_CLOUD_DIR))
import trace_budgets as _trace_budgets  # type: ignore[import-not-found]  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_SAMPLING_REQUIRED = 3
EXIT_UNSUPPORTED = 4
EXIT_CORRECTNESS_FAILED = 5
EXIT_BUILD_FAILED = 6
EXIT_GOLDEN_MISMATCH = 7
EXIT_UNKNOWN_TARGET_DEVICE = 8
EXIT_DRY_RUN_NOT_SUPPORTED = 9

KB_CAPABILITY_CARD = "references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md"

MODES = (
    "correctness",
    "timing",
    "trace",
    "regression",
    "emit-dual-run-manifest",
)
GPU_ONLY_MODES = ("timing", "trace")
CPU_OK_MODES = ("correctness", "regression", "emit-dual-run-manifest")


# --- device identity via libcudart -------------------------------------------


class DeviceProps:
    """Holder for the identity-relevant subset of cudaDeviceProp."""

    def __init__(self) -> None:
        self.device_name: str = ""
        self.compute_capability_major: int = 0
        self.compute_capability_minor: int = 0
        self.uuid: str = ""
        self.sm_count: int = 0
        self.total_global_mem_bytes: int = 0
        self.cuda_runtime_version: str = ""
        self.driver_version: str = ""
        # task11 / the HRD-profiler clause: live L2 cache capacity from
        # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE. Used by the HRD
        # profiler's cache-model spec resolver (CUDA-attribute-wins
        # rule per R16 task10 artifact Section 7).
        self.l2_cache_size_bytes: int = 0
        # Slice-geometry-aware effective capacity (the Round B
        # sub-slice 2.1 refinement): the CUDA driver API does not
        # publish a stable attribute for the L2 slice count, so we
        # leave this at 0 (treated as `None` by the resolver) and
        # rely on the static-card fallback
        # (`_static_card_num_l2_slices_for_props`).
        self.num_l2_slices: int = 0


# Note: we deliberately avoid mirroring the full cudaDeviceProp struct because
# its layout is unstable across CUDA versions. Instead we read the identity
# fields individually through cuDeviceGetName / cuDeviceGetAttribute /
# cuDeviceGetUuid / cuDeviceTotalMem_v2, which have stable ABIs.


def _load_cuda_driver_lib() -> ctypes.CDLL | None:
    """Find and load libcuda.so (the driver API) without failing hard."""
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _load_cudart_lib() -> ctypes.CDLL | None:
    """Find and load libcudart.so (runtime API) without failing hard."""
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so.11", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


# Driver API CUdevice_attribute enum values we need (values are CUDA-version-stable).
_CU_DEV_ATTR_MULTIPROCESSOR_COUNT = 16
_CU_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEV_ATTR_COMPUTE_CAPABILITY_MINOR = 76
# task11 / the HRD-profiler clause: CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE.
# The driver-API enum value is 38 per
# `tools/nvbit_tracer/nvbit_release/core/cuda.h:854`. R17 picked 87
# (the runtime API's `cudaDevAttrL2CacheSize` mis-remembered as the
# driver API value); R18 unmasked the bug because the controller now
# threads the live CUDA value through, and attribute 87 returns
# `64` on sm_120 + CUDA 13.2 (likely a different driver attribute
# entirely).
_CU_DEV_ATTR_L2_CACHE_SIZE = 38


def probe_device_identity(device_ordinal: int = 0) -> DeviceProps:
    """Populate a DeviceProps from the live libcuda / libcudart.

    Raises RuntimeError if the driver or runtime is unreachable.
    """
    cuda = _load_cuda_driver_lib()
    if cuda is None:
        raise RuntimeError("libcuda.so not found; install the NVIDIA driver runtime")

    cuInit = cuda.cuInit
    cuInit.argtypes = [c_uint]
    cuInit.restype = c_int
    rc = cuInit(0)
    if rc != 0:
        raise RuntimeError(f"cuInit failed with rc={rc}")

    cuDeviceGet = cuda.cuDeviceGet
    cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    cuDeviceGet.restype = c_int
    cu_dev = c_int()
    rc = cuDeviceGet(byref(cu_dev), c_int(device_ordinal))
    if rc != 0:
        raise RuntimeError(f"cuDeviceGet({device_ordinal}) failed with rc={rc}")

    props = DeviceProps()

    cuDeviceGetName = cuda.cuDeviceGetName
    cuDeviceGetName.argtypes = [POINTER(c_char), c_int, c_int]
    cuDeviceGetName.restype = c_int
    name_buf = (c_char * 256)()
    rc = cuDeviceGetName(name_buf, c_int(256), cu_dev)
    if rc != 0:
        raise RuntimeError(f"cuDeviceGetName failed with rc={rc}")
    props.device_name = name_buf.value.decode("utf-8", errors="replace")

    cuDeviceGetAttribute = cuda.cuDeviceGetAttribute
    cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
    cuDeviceGetAttribute.restype = c_int

    def _attr(k: int) -> int:
        v = c_int()
        rc_ = cuDeviceGetAttribute(byref(v), c_int(k), cu_dev)
        if rc_ != 0:
            raise RuntimeError(f"cuDeviceGetAttribute({k}) failed with rc={rc_}")
        return v.value

    props.compute_capability_major = _attr(_CU_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR)
    props.compute_capability_minor = _attr(_CU_DEV_ATTR_COMPUTE_CAPABILITY_MINOR)
    props.sm_count = _attr(_CU_DEV_ATTR_MULTIPROCESSOR_COUNT)
    props.l2_cache_size_bytes = _attr(_CU_DEV_ATTR_L2_CACHE_SIZE)

    # UUID via cuDeviceGetUuid (fills 16-byte buffer)
    cuDeviceGetUuid = cuda.cuDeviceGetUuid
    cuDeviceGetUuid.argtypes = [POINTER(c_char), c_int]
    cuDeviceGetUuid.restype = c_int
    uuid_buf = (c_char * 16)()
    rc = cuDeviceGetUuid(uuid_buf, cu_dev)
    if rc != 0:
        raise RuntimeError(f"cuDeviceGetUuid failed with rc={rc}")
    raw = bytes(uuid_buf)
    props.uuid = f"GPU-{raw[:4].hex()}-{raw[4:6].hex()}-{raw[6:8].hex()}-{raw[8:10].hex()}-{raw[10:].hex()}"

    cuDeviceTotalMem = cuda.cuDeviceTotalMem_v2
    cuDeviceTotalMem.argtypes = [POINTER(c_size_t), c_int]
    cuDeviceTotalMem.restype = c_int
    mem = c_size_t()
    rc = cuDeviceTotalMem(byref(mem), cu_dev)
    if rc != 0:
        raise RuntimeError(f"cuDeviceTotalMem_v2 failed with rc={rc}")
    props.total_global_mem_bytes = mem.value

    cuDriverGetVersion = cuda.cuDriverGetVersion
    cuDriverGetVersion.argtypes = [POINTER(c_int)]
    cuDriverGetVersion.restype = c_int
    drv = c_int()
    rc = cuDriverGetVersion(byref(drv))
    if rc != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with rc={rc}")
    # driver version is packed as major*1000 + minor*10
    props.driver_version = f"{drv.value // 1000}.{(drv.value % 1000) // 10}"

    cudart = _load_cudart_lib()
    if cudart is not None:
        cudaRuntimeGetVersion = cudart.cudaRuntimeGetVersion
        cudaRuntimeGetVersion.argtypes = [POINTER(c_int)]
        cudaRuntimeGetVersion.restype = c_int
        rv = c_int()
        rc = cudaRuntimeGetVersion(byref(rv))
        if rc == 0:
            props.cuda_runtime_version = f"{rv.value // 1000}.{(rv.value % 1000) // 10}"
    return props


# --- roster loading -----------------------------------------------------------


def load_roster() -> dict[str, dict[str, Any]]:
    with open(SUITE_ROOT / "roster.toml", "rb") as f:
        data = tomllib.load(f)
    return {r["slug"]: r for r in data["rows"]}


def _binary_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --- measurement manifest -----------------------------------------------------


def build_measurement_manifest(
    *,
    slug: str,
    mode: str,
    warm_up_iters: int,
    repeat_iters: int,
    cv_pct: float,
    props: DeviceProps,
    binary_path: Path | None,
    compiler_flags: list[str],
    shape_point: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "slug": slug,
        "mode": mode,
        "warm_up_iters": warm_up_iters,
        "repeat_iters": repeat_iters,
        "cv_pct": cv_pct,
        "cuda_version": props.cuda_runtime_version,
        "driver_version": props.driver_version,
        "gpu_sku": _infer_gpu_sku(props),
        "gpu_uuid": props.uuid,
        "sm_count": props.sm_count,
        "total_global_mem_bytes": props.total_global_mem_bytes,
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "binary_sha256": _binary_sha256(binary_path) if binary_path else "",
        "compiler_flags": list(compiler_flags),
        "kb_capability_card": KB_CAPABILITY_CARD,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shape_point": shape_point or "",
    }
    return manifest


class ArchitectureIdentityUnavailable(ValueError):
    """Typed fail-closed result for unresolved device identity."""

    TOKEN = "architecture_identity_missing"


def _infer_gpu_sku(props: DeviceProps) -> str:
    cc = (props.compute_capability_major, props.compute_capability_minor)
    if cc == (9, 0):
        return "h100"
    if cc == (12, 0):
        name = props.device_name.lower()
        if "pro 6000" in name or "rtx pro 6000" in name:
            return "rtx_pro_6000_blackwell"
        if "5060" in name:
            return "rtx_5060_laptop"
        return "blackwell_consumer_unknown"
    if cc == (8, 6):
        name = props.device_name.lower()
        if "3090 ti" in name or "3090ti" in name:
            return "rtx_3090_ti"
        return "ampere_consumer_unknown"
    return f"sm_{props.compute_capability_major}{props.compute_capability_minor}_unknown"


def _divergence_proxies_device_token_for_props(props: DeviceProps) -> str:
    """Map live `DeviceProps` to the divergence-proxies producer's
    device token. The producer at
    `tools/divergence_proxies/bundle_from_trace.py` accepts only
    `sm_86` (Ampere desktop, CC 8.6) and `sm_120-laptop` (Blackwell
    consumer laptop, CC 12.0); any other compute capability raises a
    typed `ValueError` so the controller-side `except` records a
    `divergence_proxies_skip_reason` instead of dispatching the wrong
    token. The lockstep partner is the producer's argparse `choices`
    tuple at `bundle_from_trace.py:255-256`; both sides MUST move
    together when a third local arch is added (see
    BL-20260507-arch-token-multi-surface-lockstep)."""
    cc = (props.compute_capability_major, props.compute_capability_minor)
    if cc == (8, 6):
        return "sm_86"
    if cc == (12, 0):
        return "sm_120-laptop"
    raise ValueError(
        f"unknown_target_device_for_divergence_proxies:"
        f"cc_{cc[0]}_{cc[1]}"
    )


def _perturbation_target_arch_for_props(props: DeviceProps) -> str:
    """Map live `DeviceProps` to the perturbation-harness producer's
    target_arch token. The producer at
    `tools/perturbation_harness/build.py::run_perturbation_harness`
    accepts only the values in
    `tools/perturbation_harness/ledger.py::SUPPORTED_TARGET_ARCHES`
    (`sm_86` for CC 8.6 Ampere desktop, `sm_120-laptop` for CC 12.0
    Blackwell consumer laptop); any other compute capability raises
    a typed `ValueError` so the controller-side `except` records a
    `perturbation_run_ledger_skip_reason` instead of building binaries
    with a wrong -gencode flag set. The multi-surface-lockstep
    partners are the schema's `$defs.target_arch_enum`, the producer
    helper `nvcc_gencode_for_target_arch`, and the producer-side
    `_device_block_for_target_arch`; all four MUST move together when
    a third local arch is added (BL-20260507-arch-token-multi-surface-
    lockstep)."""
    cc = (props.compute_capability_major, props.compute_capability_minor)
    if cc == (8, 6):
        return "sm_86"
    if cc == (12, 0):
        return "sm_120-laptop"
    raise ValueError(
        f"unknown_target_arch_for_perturbation_harness:"
        f"cc_{cc[0]}_{cc[1]}"
    )


# --- startup identity gate ----------------------------------------------------


KNOWN_COMPUTE_CAPS = {(8, 6), (9, 0), (12, 0)}


def startup_identity_check(allow_any_device: bool) -> DeviceProps | None:
    """Probe device identity. Returns DeviceProps on success, exits otherwise.

    Returns None if allow_any_device is True and probing raised (caller should
    proceed with a synthetic identity record)."""
    try:
        props = probe_device_identity(0)
    except RuntimeError as e:
        if allow_any_device:
            print(f"run_suite: --allow-any-device set; identity probe failed: {e}", file=sys.stderr)
            return None
        print(f"run_suite: device identity probe failed: {e}", file=sys.stderr)
        print("run_suite: pass --allow-any-device to proceed without a GPU", file=sys.stderr)
        sys.exit(EXIT_UNKNOWN_TARGET_DEVICE)

    cc = (props.compute_capability_major, props.compute_capability_minor)
    if cc not in KNOWN_COMPUTE_CAPS and not allow_any_device:
        print(
            f"run_suite: unknown_target_device: device reports CC {cc[0]}.{cc[1]}; "
            f"known targets are {sorted(KNOWN_COMPUTE_CAPS)}. Pass --allow-any-device "
            f"to bypass.",
            file=sys.stderr,
        )
        sys.exit(EXIT_UNKNOWN_TARGET_DEVICE)
    return props


def synthetic_dry_run_props() -> DeviceProps:
    p = DeviceProps()
    p.device_name = "DRY-RUN-CPU"
    p.compute_capability_major = 0
    p.compute_capability_minor = 0
    p.uuid = "GPU-dry-run-cpu-00000000-0000-0000-0000-000000000000"
    p.sm_count = 0
    p.total_global_mem_bytes = 0
    p.cuda_runtime_version = "n/a"
    p.driver_version = "n/a"
    return p


# --- per-mode handlers --------------------------------------------------------


def _emit_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "measurement_manifest.json"
    with open(dest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    return dest


def _load_reference_family(family: str):
    """Import the per-family Python module under references/; return None on miss.

    Adds the references directory to sys.path for the duration of the import
    so intra-family `from _common import ...` works under importlib spec loading.
    """
    import importlib.util
    ref_dir = SUITE_ROOT / "references"
    ref_path = ref_dir / f"{family}.py"
    if not ref_path.exists():
        return None
    if str(ref_dir) not in sys.path:
        sys.path.insert(0, str(ref_dir))
    spec = importlib.util.spec_from_file_location(f"simbench_bw_references_{family}", ref_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tolerance_triplet(dtype: str, reduction_kind: str, kernel_family: str) -> dict | None:
    """Look up the tolerance entry matching the row's triplet."""
    path = SUITE_ROOT / "tolerances.toml"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for entry in data.get("entries", []):
        if (entry.get("dtype") == dtype
                and entry.get("reduction_kind") == reduction_kind
                and entry.get("kernel_family") == kernel_family):
            return entry
    return None


def _infer_reduction_kind(row: dict[str, Any]) -> str:
    """Mirror tolerances.toml inference: nondeterministic rows use atomic_add;
    otherwise fall back by family."""
    if row["deterministic"] == "ND":
        return "atomic_add"
    family = row["reference_family"]
    if family in ("attention", "paged_attention", "norm", "activation", "sampling",
                   "conv", "diffusion", "vision_encoder", "audio", "tts", "multimodal"):
        return "tree_sum"
    if family in ("grouped_gemm", "mma_microbench", "ssm", "rope", "control"):
        return "fma_chain"
    if family in ("embedding", "cluster_dsmem"):
        return "warp_shuffle_sum"
    if family == "tma":
        return "none"
    return "none"


def _first_correctness_shape(slug: str) -> dict | None:
    shape_path = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not shape_path.exists():
        return None
    with open(shape_path, "rb") as f:
        shape = tomllib.load(f)
    shapes = shape.get("correctness_shapes", [])
    if not shapes:
        return None
    return shapes[0]


def _emit_candidate(row: dict[str, Any], shape_point: dict, output: Any, out_dir: Path) -> Path:
    """Persist a correctness candidate under out_dir keyed by shape_point."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"candidate_{shape_point['name']}.json"
    dest.write_text(json.dumps({
        "schema_version": 1,
        "slug": row["slug"],
        "shape_set": "correctness_shapes",
        "shape_point": shape_point["name"],
        "output": output,
    }, indent=2))
    return dest


args_out_dir_root: str = "results"


def _current_arch_tag(props: DeviceProps) -> str:
    """CC-family fallback arch (NOT the row's build arch). Reserved for the
    DeviceProps-level identity record and for the synthetic/unknown-CC path.
    Artifact routing for non-dry-run correctness / regression / measurement
    MUST use `_resolve_build_arch_for_row` instead — CC 12.0 alone is
    ambiguous between sm_120, sm_120a, sm_120f."""
    cc = (props.compute_capability_major, props.compute_capability_minor)
    if cc == (9, 0): return "sm_90a"
    if cc == (12, 0): return "sm_120"
    return f"sm_{cc[0]}{cc[1]}"


# Row-to-arch resolution is owned by `arch_resolver.py` so this orchestrator
# and validators share a single source of truth. The SUITE_ROOT sys.path insert
# above makes this import work whether run_suite.py is executed as a script or
# imported via importlib.
from arch_resolver import resolve_build_arch_for_row as _shared_resolve_build_arch  # pyright: ignore[reportMissingImports]


def _resolve_build_arch_for_row(row: dict[str, Any], props: DeviceProps) -> str:
    """Return the row's declared build arch that applies on the current device.

    Thin wrapper around `arch_resolver.resolve_build_arch_for_row` that adds
    the `run_suite.py`-specific fallback policy: when the row declares no
    `arch_gate`, or when no declared arch matches the device's CC family
    (e.g. unknown-CC hosts under `--allow-any-device`), fall back to
    `_current_arch_tag(props)`. The shared helper returns `None` in those
    cases so the caller's fallback policy is explicit."""
    cc = (props.compute_capability_major, props.compute_capability_minor)
    resolved = _shared_resolve_build_arch(row, cc)
    if resolved is not None:
        return resolved
    return _current_arch_tag(props)


# Arch-token → -gencode flag map, mirroring `workloads/simbench_bw/common.mk`'s
# `NVCC_ARCH_TOKEN_FLAG`. Kept in sync by convention — a new arch token added
# to common.mk MUST be added here too (and vice versa). The correctness
# measurement manifest derives `compiler_flags` from the row's `arch_gate`
# via `_gencode_flags_for_arch_gate` rather than hardcoding, so rows built
# for sm_120f / sm_120a / sm_90a-only report their actual build arch in
# provenance. Closes the R19-review / R20-review blocking side
# issue that AC27 accuracy for Blackwell-conditional rows depends on.
_ARCH_TOKEN_TO_GENCODE: dict[str, list[str]] = {
    "sm_86":   ["-gencode", "arch=compute_86,code=sm_86"],
    "sm_90a":  ["-gencode", "arch=compute_90a,code=sm_90a"],
    "sm_120":  ["-gencode", "arch=compute_120,code=sm_120"],
    "sm_120f": ["-gencode", "arch=compute_120f,code=sm_120f"],
    "sm_120a": ["-gencode", "arch=compute_120a,code=sm_120a"],
}


def _gencode_flags_for_arch_gate(arch_gate: list[str]) -> list[str]:
    """Return the `-gencode` flag sequence that `common.mk` would emit for the
    row's `arch_gate` tokens (in declaration order). Unknown tokens are
    skipped, matching `common.mk`'s strict-token policy at the Makefile
    level."""
    flags: list[str] = []
    for token in arch_gate or ():
        if token in _ARCH_TOKEN_TO_GENCODE:
            flags.extend(_ARCH_TOKEN_TO_GENCODE[token])
    return flags


def _find_kernel_executable(slug: str) -> Path | None:
    """Search the per-level layout for a built executable named <slug>_run."""
    for level in ("l1", "l2", "l3", "l4"):
        cand = SUITE_ROOT / level / slug / f"{slug}_run"
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    return None


def _stdout_wall_time_s(stdout: str) -> float | None:
    """Read positive numeric ``wall_time_ms`` from a subprocess's
    stdout JSON, return seconds.

    The simbench workload binaries emit a JSON record on stdout that
    carries ``wall_time_ms`` as the device-side kernel timing
    (matching what `handle_timing` reads at lines 1659-1669). This
    helper centralizes the parse so `handle_trace` can use the same
    stdout-preferred semantic for both baseline and NVBit timing.

    Returns ``None`` when stdout is not valid JSON, the field is
    absent, the value is non-numeric, or the value is non-positive.
    Callers fall back to host ``perf_counter`` in those cases. The
    fallback is intentional: a workload that does not report a
    self-timed value is still a valid baseline; the controller
    just loses the device-side accuracy advantage.
    """
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    val = result.get("wall_time_ms")
    if not isinstance(val, (int, float)):
        return None
    if val <= 0:
        return None
    return float(val) / 1000.0


_HRD_SECTION10_FIELDS: tuple[str, ...] = (
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "launch_index",
    "kernel_name",
    "device_name",
    "compute_capability_major",
    "compute_capability_minor",
    "sm_count",
    "total_global_mem_bytes",
    "l2_cache_size_bytes",
    "cuda_l2_cache_size_bytes",
    "static_card_l2_cache_size_bytes",
    "cache_model_capacity_source",
    "cache_model_capacity_typed_reason",
    "sector_size_bytes",
    "line_size_bytes",
    "sectors_per_line",
    "geometry_source",
    "geometry_typed_reason",
    "associativity_policy",
    "fill_policy",
    "write_policy",
    "l1_bypass_cases",
    "ncu_cache_hit_rate_l2_sector",
    "ncu_cache_hit_rate_l1tex_sector",
    "hrd_profile_path",
    "hrd_schema_version",
    "bootstrap_resamples",
    "ci_half_width_target",
    "minimum_bin_sample_count",
    "trace_access_count",
    "trace_drop_count",
    "trace_drop_rate",
    "typed_unavailable_count",
    "underconfigured_bin_count",
    "low_confidence_bin_count",
    "bundle_underconfigured",
    "cache_model_underconfigured_typed_reason",
)


def _hrd_bundle_underconfigured_typed_reason(
    bundle: dict[str, Any],
) -> str | None:
    """Read the bundle-level scalar-refusal evidence record's
    `typed_reason` from a bundle whose `bundle_underconfigured`
    is True. Returns the typed reason
    (`cache_model_underconfigured` or
    `hrd_compute_budget_exceeded`) or None when no matching
    evidence is present.

    Matches the semantics
    `verify_artifacts.py::_recompute_hrd_bundle_section10`
    expects: the cached field MUST equal whichever typed reason
    the bundle's `evidence[]` array carries in its bundle-level
    typed-unavailable record.
    """
    for ev in bundle.get("evidence", []) or []:
        if (
            isinstance(ev, dict)
            and ev.get("result_kind") == "typed_unavailable"
            and ev.get("scope") == "bundle"
            and ev.get("typed_reason") in (
                "cache_model_underconfigured",
                "hrd_compute_budget_exceeded",
            )
        ):
            return ev.get("typed_reason")
    return None


def _hrd_skip_run_summary(
    artifact_record: dict[str, Any],
    *,
    skip_reason: str,
) -> None:
    """Populate the Section 10 fields with None and record the
    typed `hrd_skip_reason`. Used by the typed-unavailable arms
    (sector-stream consumer wire-contract failure, schema invalid,
    or bundle emission failure)."""
    for field in _HRD_SECTION10_FIELDS:
        artifact_record[field] = None
    artifact_record["hrd_skip_reason"] = skip_reason


def _emit_byproduct_manifest_pairing_index(
    traces_folder: Path,
    manifest_pairing_index: dict[str, Any],
) -> list[Path]:
    """Emit a directory-level `manifest_pairing_index.json` sidecar
    to each trace-byproduct subtree (`sector_stream/`, `extra_info/`)
    under ``traces_folder``. Returns the list of paths written.

    Trace byproducts (the JSONL stream + the tracer's per-execution
    JSON sidecars under ``extra_info/``) do not carry
    `manifest_pairing_index` fields in their own wire formats — the
    JSONL is a per-event stream and the extra_info JSONs are
    payload-only arrays / dicts. Without this sidecar, byproducts
    would be paired with the parent trace JSON purely by
    filesystem location, which Codex's round-21 review correctly
    flagged as weaker provenance than the manifest-indexed feature
    bundles' wire-format MPI.

    The sidecar carries the same identity tuple (`slug`,
    `shape_point`, `binary_sha256`, `source_commit`,
    `kernel_signature_sha256`) as the feature bundles' MPI dict.
    Emitted only when the byproduct directory exists; missing
    directories are silently skipped (e.g., a trace that didn't
    produce sector_stream output).

    Call-order invariant (round-23 repair of Codex round-22 BSI #2):
    callers MUST invoke this helper ONLY on the success branch of
    `emit_hrd_bundle(...)` (i.e., after both `try/except` and the
    `result["contract_ok"]` gate). The parent trace JSON's
    Section 10 identity fields are emitted on the same success
    branch; calling this helper on an HRD-skip path would create
    parent/sidecar drift, since the parent JSON nulls out
    Section 10 on skip while the sidecar would still carry the
    pre-skip identity tuple.
    """
    written: list[Path] = []
    for byproduct_dir in ("sector_stream", "extra_info"):
        target_dir = traces_folder / byproduct_dir
        if not target_dir.is_dir():
            continue
        target = target_dir / "manifest_pairing_index.json"
        with target.open("w", encoding="utf-8") as fh:
            json.dump(manifest_pairing_index, fh, indent=2, sort_keys=True)
            fh.write("\n")
        written.append(target)
    return written


def _ensure_sector_stream_jsonl(
    traces_folder: Path,
) -> tuple[bool, str | None]:
    """Generate the sector-stream JSONL from the threadblock proto
    if it is missing. Returns ``(ok, typed_skip_reason)``: when
    ``ok`` is True, the JSONL exists and the wire contract +
    schema were re-validated; when False, the second element is
    the typed-skip-reason string the controller persists into the
    run summary's ``hrd_skip_reason`` field."""
    sector_stream_jsonl = (
        traces_folder / "sector_stream" / "sector_stream.jsonl"
    )
    if sector_stream_jsonl.is_file():
        return True, None
    sector_stream_dir = SUITE_ROOT.parent.parent / "tools" / "sector_stream"
    if str(sector_stream_dir) not in sys.path:
        sys.path.insert(0, str(sector_stream_dir))
    try:
        from consume_trace import (  # type: ignore[import-not-found]
            consume_trace_dir as _consume_trace_dir,
        )
    except Exception as exc:
        return False, f"sector_stream_consumer_import_failed:{exc}"
    try:
        result = _consume_trace_dir(traces_folder, write_jsonl=True)
    except Exception as exc:
        return False, f"sector_stream_consumer_failed:{exc}"
    if not result["contract_ok"]:
        return False, (
            f"trace_contract:{result['contract_typed_reason']}"
        )
    if not result["schema_admits_all"]:
        return False, (
            f"sector_stream_schema_invalid:{result['schema_first_failure']}"
        )
    return True, None


def _bundle_capacity_typed_reason(
    bundle: dict[str, Any], spec_field_prefix: str,
) -> str | None:
    """Inspect the bundle's evidence + spec source to surface the
    typed-reason for capacity / geometry resolution. Returns
    ``None`` when CUDA or static-card resolved successfully (the
    common live-trace case); the typed-reason string when the
    field is unresolved or the resolver emitted a static-card
    disagreement note."""
    spec = bundle.get("cache_model_spec", {})
    source = spec.get(spec_field_prefix)
    if source == "typed_unavailable":
        return f"cache_model_{spec_field_prefix.replace('_source', '')}_missing"
    for ev in bundle.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        note_kind = ev.get("note_kind") or ev.get("typed_reason")
        if (
            spec_field_prefix == "capacity_source"
            and note_kind == "cache_model_capacity_static_card_disagreement"
        ):
            return note_kind
        if (
            spec_field_prefix == "geometry_source"
            and note_kind == "cache_model_geometry_static_card_disagreement"
        ):
            return note_kind
    return None


def _stats_csv_kernel_name(traces_folder: Path) -> str | None:
    stats_csv = traces_folder / "stats.csv"
    if not stats_csv.is_file():
        return None
    try:
        with stats_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [h.strip() for h in rows[0]]
    cells = [c.strip() for c in rows[1]]
    if len(cells) != len(header):
        return None
    mapping = dict(zip(header, cells))
    return mapping.get("kernel mangled name") or None


def _emit_hrd_bundle_for_run_summary(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    slug: str,
    shape_point: dict[str, Any],
    executable: Path,
    props: "DeviceProps",
) -> None:
    """task11 / the HRD-profiler clause: emit the HRD bundle and
    persist the R16 task10 §10 evidence fields into the run
    summary.

    R19 closure (Codex's R18 review directive #5): the helper
    generates the sector-stream JSONL from the fresh
    `dynamic_trace.pb` via
    `tools/sector_stream/consume_trace.consume_trace_dir` when
    the JSONL is missing, requires `contract_ok` and
    `schema_admits_all`, and surfaces typed
    `hrd_skip_reason` strings on contract / schema failures so a
    fresh-trace `handle_trace` capture cannot silently skip HRD.
    The HRD bundle producer
    (`tools/hrd_profiler.bundle_from_trace.emit_hrd_bundle`)
    walks the same threadblock proto records via
    `import_trace_contract._reconstruct_lane_vector` (Codex's R18
    review directive #1) and emits per-touched-sector
    observations with byte-address-derived identities.
    """
    ok, skip_reason = _ensure_sector_stream_jsonl(traces_folder)
    if not ok:
        assert skip_reason is not None
        _hrd_skip_run_summary(
            artifact_record, skip_reason=skip_reason,
        )
        return
    # BL-20260505 (tools-subdir-module-name-collision): both
    # `tools/hrd_profiler/bundle_from_trace.py` and
    # `tools/divergence_proxies/bundle_from_trace.py` exist with
    # the same module basename. A bare `from bundle_from_trace
    # import emit_hrd_bundle` would return whichever module was
    # loaded first during the same Python process (e.g., a long-
    # lived test harness that has already imported the divergence
    # module would receive the wrong sibling here). Load HRD by
    # absolute path via importlib so the HRD module is
    # unambiguously resolved when nothing has pre-installed it.
    # Mirrors the divergence-side pattern at the
    # `_divergence_proxies_emit_for_handle_trace` site below.
    #
    # Test-side patchability: tests that exercise the skip paths
    # (`emit_failed:` and `trace_contract:<reason>`) install a
    # fake `bundle_from_trace` module into `sys.modules` via
    # `monkeypatch.setitem(sys.modules, "bundle_from_trace",
    # fake_mod)`. Honor that pre-installed module so the
    # skip-path regression coverage continues to work. The
    # pre-install precedence does NOT reintroduce the
    # divergence-side collision: divergence's spec_from_file_location
    # registers under the qualified name
    # `tools.divergence_proxies.bundle_from_trace`, not the
    # bare basename `bundle_from_trace`, so divergence loading
    # does not populate `sys.modules["bundle_from_trace"]`.
    if "bundle_from_trace" in sys.modules:
        emit_hrd_bundle = sys.modules["bundle_from_trace"].emit_hrd_bundle
    else:
        hrd_module_path = (
            SUITE_ROOT.parent.parent
            / "tools" / "hrd_profiler" / "bundle_from_trace.py"
        )
        hrd_dir = hrd_module_path.parent
        if str(hrd_dir) not in sys.path:
            sys.path.insert(0, str(hrd_dir))
        import importlib.util as _importlib_util
        _spec = _importlib_util.spec_from_file_location(
            "tools.hrd_profiler.bundle_from_trace",
            str(hrd_module_path),
        )
        if _spec is None or _spec.loader is None:
            raise ImportError(
                f"could not build spec for {hrd_module_path}"
            )
        _hrd_module = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_hrd_module)
        emit_hrd_bundle = _hrd_module.emit_hrd_bundle

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": _sha256_of_file(executable),
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": _kernel_signature_sha256(executable, slug),
    }
    device_block = {
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "sm_count": props.sm_count,
        "total_global_mem_bytes": props.total_global_mem_bytes,
        "l2_cache_size_bytes": props.l2_cache_size_bytes,
    }
    static_l2 = _static_card_l2_bytes_for_props(props)
    static_num_l2_slices = (
        _static_card_num_l2_slices_for_props(props)
    )
    cuda_num_l2_slices = (
        props.num_l2_slices if props.num_l2_slices > 0 else None
    )
    try:
        result = emit_hrd_bundle(
            trace_dir=traces_folder,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
            cuda_l2_cache_size_bytes=(
                props.l2_cache_size_bytes
                if props.l2_cache_size_bytes > 0 else None
            ),
            static_card_l2_cache_size_bytes=static_l2,
            cuda_num_l2_slices=cuda_num_l2_slices,
            static_card_num_l2_slices=static_num_l2_slices,
        )
    except Exception as exc:
        _hrd_skip_run_summary(
            artifact_record, skip_reason=f"emit_failed:{exc}",
        )
        return
    if not result.get("contract_ok", False):
        _hrd_skip_run_summary(
            artifact_record,
            skip_reason=(
                f"trace_contract:"
                f"{result.get('contract_typed_reason') or 'unknown'}"
            ),
        )
        return
    # R23 byproduct-manifest emission: write a directory-level
    # manifest_pairing_index sidecar to each trace-byproduct subtree
    # (sector_stream/, extra_info/) so byproducts carry their own
    # identity tuple rather than being paired purely by filesystem
    # location with the parent trace JSON. Emission is gated on
    # `emit_hrd_bundle(...)` succeeding AND `result["contract_ok"]
    # == True` — both preconditions for the parent trace JSON's
    # Section 10 identity fields being non-null. Pre-R23 the
    # emission happened before these gates, so an HRD-skip path
    # would null out the parent Section 10 but leave the sidecars
    # with non-null fields, producing exactly the parent/sidecar
    # drift R22's regression test was supposed to prevent.
    _emit_byproduct_manifest_pairing_index(
        traces_folder, manifest_pairing_index,
    )
    bundle = result["bundle"]
    spec = bundle["cache_model_spec"]
    bundle_path = Path(result["bundle_path"])
    rel_path = (
        bundle_path.relative_to(artifact_dir)
        if bundle_path.is_relative_to(artifact_dir)
        else bundle_path
    )
    expected_record_count = artifact_record.get("expected_record_count")
    admitted_record_count = artifact_record.get("admitted_record_count")
    drop_count = (
        int(expected_record_count) - int(admitted_record_count)
        if (
            isinstance(expected_record_count, int)
            and isinstance(admitted_record_count, int)
        )
        else None
    )
    section10: dict[str, Any] = {
        # manifest_pairing_index expansion
        "binary_sha256": manifest_pairing_index["binary_sha256"],
        "source_commit": manifest_pairing_index["source_commit"],
        "kernel_signature_sha256": (
            manifest_pairing_index["kernel_signature_sha256"]
        ),
        # launch / kernel identity
        "launch_index": 0,
        "kernel_name": _stats_csv_kernel_name(traces_folder),
        # device block
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "sm_count": props.sm_count,
        "total_global_mem_bytes": props.total_global_mem_bytes,
        "l2_cache_size_bytes": spec["capacity_bytes"],
        # cache-model resolver inputs
        "cuda_l2_cache_size_bytes": (
            props.l2_cache_size_bytes
            if props.l2_cache_size_bytes > 0 else None
        ),
        "static_card_l2_cache_size_bytes": static_l2,
        "cache_model_capacity_source": spec["capacity_source"],
        "cache_model_capacity_typed_reason": _bundle_capacity_typed_reason(
            bundle, "capacity_source"
        ),
        # geometry block
        "sector_size_bytes": spec["sector_size_bytes"],
        "line_size_bytes": spec["line_size_bytes"],
        "sectors_per_line": spec["sectors_per_line"],
        "geometry_source": spec["geometry_source"],
        "geometry_typed_reason": _bundle_capacity_typed_reason(
            bundle, "geometry_source"
        ),
        # Slice-geometry-aware effective capacity. The resolver
        # follows the same CUDA-wins / static-card-fallback /
        # constant-default rule used by the L2 capacity field.
        # The four mirror fields keep the cached run summary
        # auditable against the on-disk bundle's
        # `cache_model_spec` values.
        "cuda_num_l2_slices": cuda_num_l2_slices,
        "static_card_num_l2_slices": static_num_l2_slices,
        "cache_model_num_l2_slices": spec["num_l2_slices"],
        "cache_model_num_l2_slices_source": (
            spec["num_l2_slices_source"]
        ),
        # policy block
        "associativity_policy": spec["associativity_policy"],
        "fill_policy": spec["fill_policy"],
        "write_policy": spec["write_policy"],
        "l1_bypass_cases": list(spec["l1_bypass_cases"]),
        # NCU label placeholders (populated by the labels pipeline,
        # not by the HRD path; today the run summary surfaces them
        # as None until the local-labels pipeline reconciles).
        "ncu_cache_hit_rate_l2_sector": None,
        "ncu_cache_hit_rate_l1tex_sector": None,
        # HRD bundle reference
        "hrd_profile_path": str(rel_path),
        "hrd_schema_version": bundle["schema_version"],
        # bootstrap parameters (the binning constants the bundle
        # was computed under)
        "bootstrap_resamples": bundle["binning"]["bootstrap_resamples"],
        "ci_half_width_target": bundle["binning"]["ci_half_width_target"],
        "minimum_bin_sample_count": (
            bundle["binning"]["minimum_bin_sample_count"]
        ),
        # trace access / drop accounting
        "trace_access_count": result.get("total_observations", 0),
        "trace_drop_count": drop_count,
        "trace_drop_rate": artifact_record.get("actual_drop_rate"),
        # bin / underconfigured counts
        "typed_unavailable_count": result["typed_unavailable_count"],
        "underconfigured_bin_count": result["underconfigured_bin_count"],
        "low_confidence_bin_count": result["low_confidence_bin_count"],
        # bundle-level annotation. R55 / Codex's R54 review: the
        # cached `cache_model_underconfigured_typed_reason` field
        # mirrors whichever bundle-level scalar-refusal typed
        # reason actually fired (may be
        # `cache_model_underconfigured` OR
        # `hrd_compute_budget_exceeded`); the field name is
        # preserved for backwards compat with existing
        # cached summaries. Reading the actual value from the
        # bundle's evidence ensures the cache matches what
        # `_recompute_hrd_bundle_section10` will compute, so
        # the verifier's cached-vs-recomputed comparison passes.
        "bundle_underconfigured": bool(bundle["bundle_underconfigured"]),
        "cache_model_underconfigured_typed_reason": (
            _hrd_bundle_underconfigured_typed_reason(bundle)
            if bundle["bundle_underconfigured"] else None
        ),
    }
    artifact_record.update(section10)


def _sha256_of_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "0" * 64


def _resolve_source_commit() -> str:
    """Best-effort source-commit resolver: read HEAD from
    `.git/HEAD` and resolve the ref, falling back to all-zero
    when the repo state is unreadable.

    The resolver handles two on-disk ref storage layouts:
    (a) loose refs at `.git/refs/heads/<branch>` (the default
    after `git init` / `git checkout`); (b) packed refs in
    `.git/packed-refs` (the canonical layout after `git gc`
    or `git pack-refs --all`, and on freshly-cloned repos
    that ship with pre-packed refs). A bare
    `ref_file.read_text()` raises on case (b) because the
    loose-ref file does not exist; the packed-refs fallback
    scans `.git/packed-refs` for the matching ref line.
    Without the fallback, the resolver returns all-zero on
    packed-refs checkouts, polluting `source_commit` fields
    in manifest pairing indexes and run summaries."""
    try:
        git_dir = SUITE_ROOT.parent.parent / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref_path = head[len("ref: "):]
        ref_file = git_dir / ref_path
        try:
            return ref_file.read_text().strip()
        except OSError:
            packed = git_dir / "packed-refs"
            for line in packed.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "^")):
                    continue
                parts = stripped.split(" ", 1)
                if len(parts) == 2 and parts[1] == ref_path:
                    return parts[0]
            return "0" * 64
    except OSError:
        return "0" * 64


def _kernel_signature_sha256(executable: Path, slug: str) -> str:
    """Stable placeholder derived from slug + repo-relative executable
    path.

    The canonical form is repo-relative so the hash is host-independent
    and joins byte-identically with the label-side fallback emitted by
    `tools/local_labels/label_driver.sh:530`, which serializes
    ``$SLUG|$BINARY_PATH`` with a repo-relative ``$BINARY_PATH``.
    Pre-round-21 this function serialized the absolute executable path,
    so the trace-side placeholder diverged from the label-side
    placeholder whenever the repo lived at different absolute paths on
    different hosts (or whenever the label sweep was invoked from the
    repo root with a relative path while the trace runner resolved an
    absolute Path). The carveout cell exhibited that divergence at the
    end of round-20 (Codex round-20 mainline finding); R21 closes it by
    canonicalizing to the repo-relative form here.

    If `executable` is not inside the repo root, fall back to the bare
    `Path.name` (basename) so the hash stays canonical regardless of
    where an out-of-tree binary lives on disk.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        canonical = executable.resolve().relative_to(repo_root)
    except ValueError:
        canonical = Path(executable.name)
    return hashlib.sha256(
        f"{slug}|{canonical}".encode("utf-8"),
    ).hexdigest()


def _static_card_l2_bytes_for_props(props: "DeviceProps") -> int | None:
    """Static device-card L2 fallback per arch:
    sm_86 → 6 MiB (RTX 3090 Ti, Ampere consumer card);
    sm_120 (laptop) → 32 MiB (RTX 5060 Laptop GPU, Blackwell GB206;
    confirmed via the live `cuDeviceGetAttribute(L2_CACHE_SIZE)`
    on driver 595.71.05 + CUDA 13.2 — 33554432 bytes)."""
    if props.compute_capability_major == 8 and props.compute_capability_minor == 6:
        return 6 * 1024 * 1024
    if props.compute_capability_major == 12:
        return 32 * 1024 * 1024
    return None


def _static_card_num_l2_slices_for_props(
    props: "DeviceProps",
) -> int | None:
    """Static device-card L2 slice count fallback per arch.

    The CUDA driver API does not publish a stable attribute for
    the L2 slice count, so the resolver leans on this static
    fallback when no CUDA value is available.

    sm_86 (RTX 3090 Ti, Ampere GA102) → 6 L2 slices per the GA102
    architecture (6 MiB L2 split across 6 slices of 1 MiB each;
    each sector maps to slice `sector_id % 6`). This is the
    documented GA102 partitioning behind the
    `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE` value.

    sm_120 (RTX 5060 Laptop, Blackwell GB206) → returns None
    today; the Blackwell L2 slice count is not yet captured in
    this static card. Cross-arch sm_120 verification is
    explicitly deferred per CLAUDE.md's
    Two-Device Local Development Model.
    """
    if props.compute_capability_major == 8 and props.compute_capability_minor == 6:
        return 6
    return None


_ORDER_PROJECTIONS_SECTION_FIELDS: tuple[str, ...] = (
    "order_projections_path",
    "order_projections_schema_version",
    "order_projections_source_commit",
    "order_projections_admitted_count",
    "order_projections_typed_unavailable_count",
    "order_projections_disagreement_variance_result_kind",
    "order_projections_disagreement_variance_value",
    "order_projections_disagreement_variance_typed_reason",
    "order_projections_total_access_events",
)


def _order_projections_skip_run_summary(
    artifact_record: dict[str, Any],
    *,
    skip_reason: str,
) -> None:
    """Populate the order-projections section fields with None
    and record the typed `order_projections_skip_reason`. The
    typed-unavailable arm: a producer / contract / sidecar
    failure leaves the bundle absent and the run summary
    flagged so the verifier skips enforcement."""
    for field in _ORDER_PROJECTIONS_SECTION_FIELDS:
        artifact_record[field] = None
    artifact_record["order_projections_skip_reason"] = skip_reason


def _emit_order_projections_for_run_summary(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    slug: str,
    shape_point: dict[str, Any],
    executable: Path,
    props: "DeviceProps",
) -> None:
    """task13 / the order-projections clause: emit the
    six-projections bundle alongside the HRD bundle and persist
    the order-projections evidence fields into the run summary.

    R26 closure of Codex's R25 review action item #2: the
    controller path now emits the order-projections bundle
    during the normal trace flow rather than only via the
    post-commit operator probe. The bundle is written to
    `<trace_dir>/order_projections/order_projections.json`;
    the run summary persists the bundle path, schema version,
    source commit, admitted-projection count,
    typed-unavailable count, disagreement-variance result
    kind / value / typed reason, and total access events. On
    producer failure (wire contract or allocator-sidecar
    error), all section fields land None and a typed
    `order_projections_skip_reason` is recorded.
    """
    op_dir = traces_folder / "order_projections"
    op_path = op_dir / "order_projections.json"

    projector_dir = SUITE_ROOT.parent.parent / "tools" / "order_projections"
    if str(projector_dir) not in sys.path:
        sys.path.insert(0, str(projector_dir))
    try:
        from projector import (  # type: ignore[import-not-found]
            compute_order_projections,
        )
    except Exception as exc:
        _order_projections_skip_run_summary(
            artifact_record,
            skip_reason=f"projector_import_failed:{exc}",
        )
        return

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": _sha256_of_file(executable),
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": _kernel_signature_sha256(executable, slug),
    }
    static_l2 = _static_card_l2_bytes_for_props(props)
    cuda_l2 = (
        props.l2_cache_size_bytes
        if props.l2_cache_size_bytes > 0 else None
    )
    capacity_bytes = (
        cuda_l2 if cuda_l2 is not None
        else (static_l2 if static_l2 is not None else 0)
    )
    if capacity_bytes <= 0:
        _order_projections_skip_run_summary(
            artifact_record,
            skip_reason="cache_model_capacity_unresolved",
        )
        return
    device_block = {
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "sm_count": props.sm_count,
        "l2_cache_size_bytes": capacity_bytes,
    }
    try:
        bundle = compute_order_projections(
            trace_dir=traces_folder,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
            capacity_bytes=capacity_bytes,
        )
    except Exception as exc:
        _order_projections_skip_run_summary(
            artifact_record,
            skip_reason=f"compute_failed:{exc}",
        )
        return

    op_dir.mkdir(parents=True, exist_ok=True)
    op_path.write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )
    rel_path = (
        op_path.relative_to(artifact_dir)
        if op_path.is_relative_to(artifact_dir)
        else op_path
    )
    admitted = sum(
        1 for p in bundle["projections"] if p["result_kind"] == "ok"
    )
    typed_unavail = len(bundle["projections"]) - admitted
    dv = bundle["disagreement_variance"]
    if dv["result_kind"] == "ok":
        dv_value = dv["value"]
        dv_typed_reason = None
    else:
        dv_value = None
        dv_typed_reason = dv["typed_reason"]
    total_events = 0
    for ev in bundle.get("evidence", []):
        if (
            isinstance(ev, dict)
            and ev.get("note_kind") == "order_projections_access_summary"
        ):
            total_events = int(ev.get("total_access_events", 0))
            break
    section: dict[str, Any] = {
        "order_projections_path": str(rel_path),
        "order_projections_schema_version": bundle["schema_version"],
        "order_projections_source_commit": (
            manifest_pairing_index["source_commit"]
        ),
        "order_projections_admitted_count": admitted,
        "order_projections_typed_unavailable_count": typed_unavail,
        "order_projections_disagreement_variance_result_kind": (
            dv["result_kind"]
        ),
        "order_projections_disagreement_variance_value": dv_value,
        "order_projections_disagreement_variance_typed_reason": (
            dv_typed_reason
        ),
        "order_projections_total_access_events": total_events,
    }
    artifact_record.update(section)


# =============================================================
# R53: divergence_proxies + sharing_locality canonical emission
# =============================================================
# Codex's R52 review (mainline gap #2) flagged that
# `handle_trace`'s canonical capture path was not emitting two
# per-shape feature sidecars (sector_stream / hrd_profile / divergence_proxies / sharing_locality / order_projections): `divergence_proxies/` and
# `sharing_locality/`. The producers exist
# (`tools/divergence_proxies/bundle_from_trace.py::consume_trace_dir`
# and `tools/sharing_locality/classifier.py::compute_sharing_locality`)
# and are exercised by legacy `dram_stream_stride_mix` evidence,
# but were never wired into the controller. R53 adds the two
# emitter helpers here, mirroring the
# `_emit_hrd_bundle_for_run_summary` /
# `_emit_order_projections_for_run_summary` pattern: typed skip
# reason on producer failure, schema-strict admission required
# for success, run-summary cache fields persisted.

_DIVERGENCE_PROXIES_SECTION_FIELDS: tuple[str, ...] = (
    "divergence_proxies_path",
    "divergence_proxies_schema_version",
    "divergence_proxies_source_commit",
    "divergence_proxies_admitted_count",
    "divergence_proxies_typed_unavailable_count",
    "divergence_proxies_total_instruction_count",
    "divergence_proxies_sector_stream_records_consumed",
)


def _divergence_proxies_skip_run_summary(
    artifact_record: dict[str, Any],
    *,
    skip_reason: str,
) -> None:
    """Populate divergence-proxies fields with None and record the
    typed `divergence_proxies_skip_reason`. Producer failure is a
    typed outcome (not silent omission) so the verifier sees an
    explicit skip rather than a missing sidecar."""
    for field in _DIVERGENCE_PROXIES_SECTION_FIELDS:
        artifact_record[field] = None
    artifact_record["divergence_proxies_skip_reason"] = skip_reason


def _emit_divergence_proxies_for_run_summary(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    slug: str,
    shape_point: dict[str, Any],
    executable: Path,
    props: "DeviceProps",
) -> None:
    """Canonical emission of the per-shape `divergence_proxies/`
    sidecar.

    Derives the producer-side device token from the live
    `DeviceProps` via
    `_divergence_proxies_device_token_for_props(props)` (CC 8.6 →
    `sm_86`; CC 12.0 → `sm_120-laptop`; other CCs raise a typed
    `ValueError` that the outer `except` translates into a
    `divergence_proxies_skip_reason`). The
    producer-side argparse choices at
    `tools/divergence_proxies/bundle_from_trace.py` are the
    lockstep partner; both surfaces MUST move together when a third
    local arch is added (BL-20260507-arch-token-multi-surface-lockstep).

    Builds the canonical 5-field `manifest_pairing_index`
    (slug, shape_point, binary_sha256, source_commit,
    kernel_signature_sha256) and passes it through to the producer
    so the on-disk bundle carries the same provenance shape sibling
    sidecars (HRD, sharing locality, order projections, perturbation
    ledger) emit. The verifier's
    `_verify_divergence_proxies_run_summaries` binds the bundle's
    `manifest_pairing_index.{slug, shape_point, source_commit}` to
    the run-summary's cached fields and fails-closed on mismatch.

    Calls `tools.divergence_proxies.bundle_from_trace.consume_trace_dir`
    with the derived `device` token and the manifest_pairing_index.
    The producer reads the per-shape
    `sector_stream/sector_stream.jsonl` and `stats.csv` (already
    materialized by HRD's upstream `_ensure_sector_stream_jsonl`
    call), computes the 8 named proxies, writes
    `trace_<shape>/divergence_proxies/divergence_proxies.json`, and
    re-validates the on-disk bundle against
    `workloads/simbench_bw/schemas/divergence_proxies.schema.json`.
    Schema-rejected output → typed
    `divergence_proxies_skip_reason`; success → run-summary cache
    fields below.
    """
    # BL-20260505 (tools-subdir-module-name-collision): both
    # `tools/hrd_profiler/bundle_from_trace.py` and
    # `tools/divergence_proxies/bundle_from_trace.py` exist with
    # the same module basename. A bare
    # `from bundle_from_trace import consume_trace_dir` would
    # return the cached HRD module (whichever was loaded first
    # during the same handle_trace flow), pointing the divergence
    # producer at the wrong bundle. Load by absolute path via
    # importlib so the divergence module is unambiguously
    # resolved no matter what sys.modules already cached.
    div_module_path = (
        SUITE_ROOT.parent.parent
        / "tools" / "divergence_proxies" / "bundle_from_trace.py"
    )
    div_dir = div_module_path.parent
    if str(div_dir) not in sys.path:
        sys.path.insert(0, str(div_dir))
    try:
        import importlib.util as _importlib_util
        _spec = _importlib_util.spec_from_file_location(
            "tools.divergence_proxies.bundle_from_trace",
            str(div_module_path),
        )
        if _spec is None or _spec.loader is None:
            raise ImportError(
                f"could not build spec for {div_module_path}"
            )
        _div_module = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_div_module)
        _div_consume_trace_dir = _div_module.consume_trace_dir
    except Exception as exc:
        _divergence_proxies_skip_run_summary(
            artifact_record,
            skip_reason=f"producer_import_failed:{exc}",
        )
        return

    try:
        device_token = _divergence_proxies_device_token_for_props(props)
    except ValueError as exc:
        _divergence_proxies_skip_run_summary(
            artifact_record,
            skip_reason=f"compute_failed:{exc}",
        )
        return

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": _sha256_of_file(executable),
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": _kernel_signature_sha256(
            executable, slug
        ),
    }

    try:
        result = _div_consume_trace_dir(
            traces_folder, device_token,
            manifest_pairing_index=manifest_pairing_index,
        )
    except Exception as exc:
        _divergence_proxies_skip_run_summary(
            artifact_record,
            skip_reason=f"compute_failed:{exc}",
        )
        return
    if not result.get("schema_admits", False):
        _divergence_proxies_skip_run_summary(
            artifact_record,
            skip_reason=(
                f"schema_rejected:"
                f"{result.get('schema_first_failure') or 'unknown'}"
            ),
        )
        return

    bundle_path = Path(result["bundle_path"])
    rel_path = (
        bundle_path.relative_to(artifact_dir)
        if bundle_path.is_relative_to(artifact_dir)
        else bundle_path
    )
    on_disk = json.loads(bundle_path.read_text(encoding="utf-8"))
    # The divergence_proxies bundle is FLAT: each of the 8 proxy
    # names is a top-level field carrying `result_kind`. Iterate
    # over the proxy-name set rather than expecting a nested
    # `proxies` dict.
    _proxy_names = (
        "unique_sectors_per_warp_inst",
        "unique_lines_per_warp_inst",
        "sectors_per_request",
        "md_loads_per_kilo_inst",
        "outstanding_load_density_by_phase",
        "pc_repeated_miss_likely_streams",
        "stride_entropy",
        "throughput_saturation_slope_under_resident_warp_sweep",
    )
    # Per the divergence-proxies producer + schema
    # (`tools/divergence_proxies/proxies.py:89`), admitted
    # records carry `result_kind == "proxy_value"`, NOT "ok"
    # (Codex's R53 review mainline gap #2). The pre-R54 emitter
    # counted `"ok"` and reported 0 admitted on every canonical
    # capture despite the producer admitting 4 proxies on real
    # traces. R54 fix: use `"proxy_value"` for the admitted arm
    # to match the actual schema.
    admitted_count = 0
    typed_unavailable_count = 0
    for name in _proxy_names:
        record = on_disk.get(name)
        if not isinstance(record, dict):
            continue
        kind = record.get("result_kind")
        if kind == "proxy_value":
            admitted_count += 1
        elif kind == "typed_unavailable":
            typed_unavailable_count += 1
    artifact_record.update({
        "divergence_proxies_path": rel_path.as_posix(),
        "divergence_proxies_schema_version": int(
            on_disk.get("schema_version", 0)
        ),
        "divergence_proxies_source_commit": (
            manifest_pairing_index["source_commit"]
        ),
        "divergence_proxies_admitted_count": admitted_count,
        "divergence_proxies_typed_unavailable_count": (
            typed_unavailable_count
        ),
        "divergence_proxies_total_instruction_count": (
            int(result.get("total_instruction_count", 0))
        ),
        "divergence_proxies_sector_stream_records_consumed": (
            int(result.get("sector_stream_records_consumed", 0))
        ),
    })


_SHARING_LOCALITY_SECTION_FIELDS: tuple[str, ...] = (
    "sharing_locality_path",
    "sharing_locality_schema_version",
    "sharing_locality_source_commit",
    "sharing_locality_grounded_count",
    "sharing_locality_projected_count",
    "sharing_locality_total_accesses",
)


def _sharing_locality_skip_run_summary(
    artifact_record: dict[str, Any],
    *,
    skip_reason: str,
) -> None:
    """Populate sharing-locality fields with None and record a typed
    `sharing_locality_skip_reason`."""
    for field in _SHARING_LOCALITY_SECTION_FIELDS:
        artifact_record[field] = None
    artifact_record["sharing_locality_skip_reason"] = skip_reason


def _emit_sharing_locality_for_run_summary(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    slug: str,
    shape_point: dict[str, Any],
    executable: Path,
    props: "DeviceProps",
) -> None:
    """R53 canonical emission of the per-shape feature sidecars
    `sharing_locality/` sidecar.

    Calls `tools.sharing_locality.classifier.compute_sharing_locality`
    with the same `manifest_pairing_index` shape that HRD uses,
    strict-validates the bundle against
    `workloads/simbench_bw/schemas/sharing_locality.schema.json`,
    writes the JSON to
    `trace_<shape>/sharing_locality/sharing_locality.json`, and
    caches identity + grounded-count + projected-count metadata
    in the run summary. Producer or schema failure becomes a
    typed `sharing_locality_skip_reason`.
    """
    sl_dir = SUITE_ROOT.parent.parent / "tools" / "sharing_locality"
    if str(sl_dir) not in sys.path:
        sys.path.insert(0, str(sl_dir))
    try:
        from classifier import (  # type: ignore[import-not-found]
            compute_sharing_locality as _compute_sharing_locality,
        )
    except Exception as exc:
        _sharing_locality_skip_run_summary(
            artifact_record,
            skip_reason=f"producer_import_failed:{exc}",
        )
        return

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": _sha256_of_file(executable),
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": _kernel_signature_sha256(executable, slug),
    }
    # The sharing_locality schema's `device` block uses
    # `additionalProperties: false` with exactly four fields
    # (device_name, compute_capability_major,
    # compute_capability_minor, sm_count) — `l2_cache_size_bytes`
    # is not part of this contract because the sharing-locality
    # classifier doesn't depend on cache capacity. Keeping the
    # block minimal so strict admission passes.
    device_block = {
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "sm_count": props.sm_count,
    }
    try:
        bundle = _compute_sharing_locality(
            trace_dir=traces_folder,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
        )
    except Exception as exc:
        _sharing_locality_skip_run_summary(
            artifact_record,
            skip_reason=f"compute_failed:{exc}",
        )
        return

    sl_subdir = traces_folder / "sharing_locality"
    sl_subdir.mkdir(parents=True, exist_ok=True)
    sl_path = sl_subdir / "sharing_locality.json"
    sl_path.write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )

    # R7-style discipline (validate the bytes the producer wrote,
    # not the in-process bundle): re-read the on-disk JSON before
    # strict validation so a serialization discrepancy cannot
    # silently slip through.
    schema_path = (
        SUITE_ROOT / "schemas" / "sharing_locality.schema.json"
    )
    try:
        import jsonschema as _jsonschema  # type: ignore[import-not-found]
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = _jsonschema.Draft202012Validator(schema)
        on_disk = json.loads(sl_path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(on_disk))
    except Exception as exc:
        _sharing_locality_skip_run_summary(
            artifact_record,
            skip_reason=f"schema_load_failed:{exc}",
        )
        return
    if errors:
        first = errors[0]
        _sharing_locality_skip_run_summary(
            artifact_record,
            skip_reason=(
                f"schema_rejected:{list(first.absolute_path)}: "
                f"{first.message}"
            ),
        )
        return

    # The schema declares `grounded` and `projected` as arrays of
    # records (each carrying `class_name` + `result_kind`). Count
    # the admitted ones; typed-unavailable records show up here
    # too but are not counted as `*_count`.
    grounded = on_disk.get("grounded") or []
    projected = on_disk.get("projected") or []
    grounded_count = sum(
        1 for r in grounded
        if isinstance(r, dict) and r.get("result_kind") == "ok"
    )
    projected_count = sum(
        1 for r in projected
        if isinstance(r, dict) and r.get("result_kind") == "ok"
    )
    # The sharing-locality schema declares `evidence` as an
    # array of records, with the
    # `sharing_locality_pair_count_summary` record carrying an
    # `addresses_observed` field (NOT `total_accesses` — Codex's
    # R53 review mainline gap #3). R53's emitter read the wrong
    # key and persisted 0 on every canonical summary despite
    # the producer reporting nonzero values (warm/cold has
    # `addresses_observed = 1064962`). R54 fix: sum
    # `addresses_observed` across the
    # `sharing_locality_pair_count_summary` evidence records.
    evidence_records = on_disk.get("evidence") or []
    total_accesses = 0
    for ev in evidence_records:
        if (
            isinstance(ev, dict)
            and ev.get("note_kind") == "sharing_locality_pair_count_summary"
        ):
            v = ev.get("addresses_observed")
            if isinstance(v, int):
                total_accesses += v
    rel_path = (
        sl_path.relative_to(artifact_dir)
        if sl_path.is_relative_to(artifact_dir)
        else sl_path
    )
    artifact_record.update({
        "sharing_locality_path": rel_path.as_posix(),
        "sharing_locality_schema_version": int(
            on_disk.get("schema_version", 0)
        ),
        "sharing_locality_source_commit": (
            manifest_pairing_index["source_commit"]
        ),
        "sharing_locality_grounded_count": grounded_count,
        "sharing_locality_projected_count": projected_count,
        "sharing_locality_total_accesses": total_accesses,
    })


_PERTURBATION_RUN_LEDGER_SECTION_FIELDS: tuple[str, ...] = (
    "perturbation_run_ledger_path",
    "perturbation_run_ledger_schema_version",
    "perturbation_run_ledger_source_commit",
    "perturbation_admitted_count",
    "perturbation_typed_unavailable_count",
    "perturbation_per_family_admitted_count",
    "perturbation_typed_unavailable_breakdown",
    "sass_confirmation_pass_count",
    "correctness_equivalence_pass_count",
)


def _perturbation_run_ledger_skip_run_summary(
    artifact_record: dict[str, Any],
    *,
    skip_reason: str,
) -> None:
    """Populate the perturbation-run-ledger section fields with None
    and record the typed `perturbation_run_ledger_skip_reason`. The
    typed-unavailable arm: a producer / build / sidecar failure
    leaves the bundle absent and the run summary flagged so the
    verifier skips enforcement."""
    for field in _PERTURBATION_RUN_LEDGER_SECTION_FIELDS:
        artifact_record[field] = None
    artifact_record["perturbation_run_ledger_skip_reason"] = skip_reason


def _load_module_from_file_without_basename_collision(
    *,
    module_name: str,
    module_path: Path,
    transient_basenames: tuple[str, ...],
    import_root: Path,
):
    import importlib.util as _importlib_util

    spec = _importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec for {module_path}")
    module = _importlib_util.module_from_spec(spec)
    sentinel = object()
    old_modules = {
        name: sys.modules.get(name, sentinel)
        for name in transient_basenames
    }
    old_named_module = sys.modules.get(module_name, sentinel)
    old_path = list(sys.path)
    try:
        for name in transient_basenames:
            sys.modules.pop(name, None)
        sys.modules[module_name] = module
        sys.path.insert(0, str(import_root))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        if old_named_module is sentinel:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_named_module
        for name, old in old_modules.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


def _load_perturbation_harness_symbols(harness_dir: Path):
    build_module = _load_module_from_file_without_basename_collision(
        module_name="tools.perturbation_harness.build",
        module_path=harness_dir / "build.py",
        transient_basenames=(
            "build",
            "ledger",
            "sass_confirm",
            "correctness_check",
        ),
        import_root=harness_dir,
    )
    ledger_module = _load_module_from_file_without_basename_collision(
        module_name="tools.perturbation_harness.ledger",
        module_path=harness_dir / "ledger.py",
        transient_basenames=("ledger",),
        import_root=harness_dir,
    )
    return (
        build_module.run_perturbation_harness,
        ledger_module.per_family_admitted_count,
        ledger_module.per_typed_reason_count,
    )


def _emit_perturbation_run_ledger_for_run_summary(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    slug: str,
    shape_point: dict[str, Any],
    executable: Path,
    props: "DeviceProps",
) -> None:
    """task15 / the perturbation-harness clause: build the
    perturbation_run_ledger bundle alongside the HRD and
    order-projection bundles, persist the run-summary fields the
    verifier recomputes against.

    The bundle is written to
    `<trace_dir>/perturbation_run_ledger/perturbation_run_ledger.json`;
    the run summary persists the bundle path, schema version, source
    commit, per-family admitted-count map, per-typed-reason
    breakdown map, and pass-count fields. On producer failure (build
    error, missing baseline, missing nvcc / cuobjdump), all section
    fields land None and a typed
    `perturbation_run_ledger_skip_reason` is recorded.
    """
    ledger_dir = traces_folder / "perturbation_run_ledger"
    ledger_path = ledger_dir / "perturbation_run_ledger.json"

    try:
        (
            run_perturbation_harness,
            per_family_admitted_count,
            per_typed_reason_count,
        ) = _load_perturbation_harness_symbols(
            SUITE_ROOT.parent.parent / "tools" / "perturbation_harness"
        )
    except Exception as exc:
        _perturbation_run_ledger_skip_run_summary(
            artifact_record,
            skip_reason=f"perturbation_harness_import_failed:{exc}",
        )
        return

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": _sha256_of_file(executable),
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": _kernel_signature_sha256(executable, slug),
    }
    static_l2 = _static_card_l2_bytes_for_props(props)
    cuda_l2 = (
        props.l2_cache_size_bytes
        if props.l2_cache_size_bytes > 0 else None
    )
    capacity_bytes = (
        cuda_l2 if cuda_l2 is not None
        else (static_l2 if static_l2 is not None else 0)
    )
    device_block = {
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "sm_count": props.sm_count,
        "l2_cache_size_bytes": capacity_bytes,
    }
    repo_root = SUITE_ROOT.parent.parent
    perturbation_build_dir = (
        repo_root / "workloads" / "simbench_bw" / "l1" / slug
        / "perturbation_build"
    )
    try:
        target_arch = _perturbation_target_arch_for_props(props)
    except ValueError as exc:
        _perturbation_run_ledger_skip_run_summary(
            artifact_record,
            skip_reason=f"perturbation_harness_unknown_target_arch:{exc}",
        )
        return
    try:
        bundle = run_perturbation_harness(
            repo_root=repo_root,
            slug=slug,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
            perturbation_build_dir=perturbation_build_dir,
            target_arch=target_arch,
        )
    except Exception as exc:
        _perturbation_run_ledger_skip_run_summary(
            artifact_record,
            skip_reason=f"perturbation_harness_failed:{exc}",
        )
        return

    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )
    rel_path = (
        ledger_path.relative_to(artifact_dir)
        if ledger_path.is_relative_to(artifact_dir)
        else ledger_path
    )
    perturbations = bundle["perturbations"]
    admitted = sum(
        1 for p in perturbations if p["result_kind"] == "ok"
    )
    typed_unavail = len(perturbations) - admitted
    sass_pass = sum(
        1 for p in perturbations
        if p.get("result_kind") == "ok"
        and p.get("sass_confirmation_status") == "pass"
    )
    correctness_pass = sum(
        1 for p in perturbations
        if p.get("result_kind") == "ok"
        and p.get("correctness_equivalence_status") == "pass"
    )
    section: dict[str, Any] = {
        "perturbation_run_ledger_path": str(rel_path),
        "perturbation_run_ledger_schema_version": bundle["schema_version"],
        "perturbation_run_ledger_source_commit": (
            manifest_pairing_index["source_commit"]
        ),
        "perturbation_admitted_count": admitted,
        "perturbation_typed_unavailable_count": typed_unavail,
        "perturbation_per_family_admitted_count": (
            per_family_admitted_count(perturbations)
        ),
        "perturbation_typed_unavailable_breakdown": (
            per_typed_reason_count(perturbations)
        ),
        "sass_confirmation_pass_count": sass_pass,
        "correctness_equivalence_pass_count": correctness_pass,
    }
    artifact_record.update(section)


_TRACE_CONVERGENCE_SECTION_FIELDS: tuple[str, ...] = (
    "trace_convergence_confidence_path",
    "trace_convergence_schema_version",
    "trace_convergence_admitted",
    "trace_convergence_typed_reason",
    "trace_convergence_windows_observed",
    "trace_convergence_windows_converged",
    "trace_convergence_final_residual",
    "trace_convergence_mrc_quantile_set",
    "trace_convergence_source_commit",
    "trace_convergence_windows_digest",
    "trace_convergence_manifest_slug",
    "trace_convergence_manifest_shape_point",
    "trace_convergence_manifest_binary_sha256",
    "trace_convergence_manifest_source_commit",
    "trace_convergence_manifest_kernel_signature_sha256",
)


def _trace_convergence_skip_run_summary(
    artifact_record: dict[str, Any],
    *,
    skip_reason: str,
) -> None:
    """Populate the trace-convergence section fields with None and
    record the typed `trace_convergence_skip_reason`. The typed-
    unavailable arm: a producer failure or missing upstream HRD bundle
    leaves the bundle absent and the run summary flagged so the
    verifier skips enforcement on partial sweeps."""
    for field in _TRACE_CONVERGENCE_SECTION_FIELDS:
        artifact_record[field] = None
    artifact_record["trace_convergence_skip_reason"] = skip_reason


def _emit_trace_convergence_confidence_for_run_summary(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    slug: str,
    shape_point: dict[str, Any],
    props: "DeviceProps",
    trace_convergence_enabled: bool = True,
) -> None:
    """task17 / the adaptive-sampler clause: build the
    trace_convergence_confidence bundle alongside the HRD,
    order-projection, and perturbation bundles.

    The helper inherits `manifest_pairing_index` from the upstream
    `hrd_profile.json`'s `manifest_pairing_index` so the convergence
    bundle's identity matches the trace's identity (the convergence
    sidecar is a downstream artifact computed from the trace + HRD;
    its identity tuple is the trace's, not the converger's).

    Wires the five typed-unavailable admission arms in priority
    order:
    1. `trace_convergence_disabled` if `trace_convergence_enabled`
       config flag is False (explicit user opt-out).
    2. `hrd_layer_unavailable` if either sector or line layer has 0
       admitted (`result_kind == "ok"`) HRD bins.
    3. `trace_convergence_not_reached_budget_cap` if
       `<trace_dir>/trace_volume_exceeded.json` exists (the
       convergence sidecar references the matching budget sidecar).
    4. `trace_convergence_window_count_underconfigured` if the trace
       is admittedly sized but the actual cumulative trace bytes is
       below `(K + 1) * TRACE_WINDOW_SIZE_BYTES` so K trailing
       converged windows cannot be populated.
    5. `mrc_quantile_underconfigured` if iteration could be performed
       but a quantile's cumulative observation count is below the
       pinned minimum.

    The previous R37 synthetic `[cumulative] * 4` admission path is
    removed (Codex R37 review directive #2); the helper now refuses
    to fabricate convergence by duplicating one HRD bundle into
    multiple windows. Real cumulative-window iteration over raw
    trace evidence is queued as a future-round expansion.
    """
    sampler_dir = SUITE_ROOT.parent.parent / "tools" / "adaptive_sampler"
    if str(sampler_dir) not in sys.path:
        sys.path.insert(0, str(sampler_dir))
    try:
        from sampler import (  # type: ignore[import-not-found]
            budget_cap_admission,
            disabled_admission,
            hrd_unavailable_admission,
            window_count_underconfigured_admission,
        )
        from trace_convergence_ledger import (  # type: ignore[import-not-found]
            HIERARCHY_LAYERS,
            K_CONSECUTIVE_WINDOWS,
            MRC_QUANTILES,
            TRACE_WINDOW_SIZE_BYTES,
            build_bundle,
            usable_admitted_bins,
            windows_digest as _windows_digest,
        )
    except Exception as exc:
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=f"adaptive_sampler_import_failed:{exc}",
        )
        return

    bundle_dir = traces_folder / "convergence_confidence"
    bundle_path = bundle_dir / "convergence_confidence.json"

    hrd_bundle_path = traces_folder / "hrd_profile" / "hrd_profile.json"
    if not hrd_bundle_path.is_file():
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=f"hrd_profile_missing:{hrd_bundle_path}",
        )
        return

    try:
        hrd_bundle = json.loads(hrd_bundle_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=f"hrd_profile_unreadable:{exc}",
        )
        return

    # Inherit the manifest_pairing_index from the upstream HRD bundle
    # verbatim. The convergence bundle's identity tuple binds to the
    # trace that produced the HRD, not to the converger's HEAD or the
    # current on-disk executable hash.
    hrd_manifest = hrd_bundle.get("manifest_pairing_index")
    if not isinstance(hrd_manifest, dict):
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason="hrd_profile_manifest_missing",
        )
        return
    required_manifest_keys = (
        "slug", "shape_point", "binary_sha256",
        "source_commit", "kernel_signature_sha256",
    )
    missing = [k for k in required_manifest_keys if not hrd_manifest.get(k)]
    if missing:
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=f"hrd_profile_manifest_incomplete:{','.join(missing)}",
        )
        return
    manifest_pairing_index = {
        k: hrd_manifest[k] for k in required_manifest_keys
    }
    # Sanity-check that the HRD bundle the caller is consuming
    # actually identifies the trace the caller is processing (cheap
    # mis-routing guard — the upstream caller provides slug +
    # shape_point.name from the run-suite's own dispatch).
    if manifest_pairing_index["slug"] != slug:
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=(
                f"hrd_profile_slug_mismatch:"
                f"hrd={manifest_pairing_index['slug']}:caller={slug}"
            ),
        )
        return
    if manifest_pairing_index["shape_point"] != shape_point["name"]:
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=(
                f"hrd_profile_shape_point_mismatch:"
                f"hrd={manifest_pairing_index['shape_point']}:"
                f"caller={shape_point['name']}"
            ),
        )
        return
    source_commit = manifest_pairing_index["source_commit"]
    static_l2 = _static_card_l2_bytes_for_props(props)
    cuda_l2 = (
        props.l2_cache_size_bytes
        if props.l2_cache_size_bytes > 0 else None
    )
    capacity_bytes = (
        cuda_l2 if cuda_l2 is not None
        else (static_l2 if static_l2 is not None else 0)
    )
    device_block = {
        "device_name": props.device_name,
        "compute_capability_major": props.compute_capability_major,
        "compute_capability_minor": props.compute_capability_minor,
        "sm_count": props.sm_count,
        "l2_cache_size_bytes": capacity_bytes,
    }

    rel_hrd = hrd_bundle_path.relative_to(traces_folder).as_posix()
    bundle_underconfigured = bool(hrd_bundle.get("bundle_underconfigured", False))

    # Filter HRD bins through the shared admitted-bin adapter so the
    # controller helper, the operator CLI (`run_sampler.py`), and the
    # producer module (`compute_mrc_at_quantile` / `iterate_windows`)
    # all use the same filter. R37 / R38 / R39 stalled on duplicated
    # filter logic across these three call sites; the shared adapter
    # is the single source of truth for the admitted-bin contract.
    raw_bins = hrd_bundle.get("bins", [])
    sector_bins = usable_admitted_bins(raw_bins, layer="sector")
    line_bins = usable_admitted_bins(raw_bins, layer="line")

    # Estimate the actual trace bytes from `dynamic_trace.pb` size if
    # present (the canonical raw-trace volume measure). Fall back to
    # 0 when the file is absent — the producer treats 0 as
    # "below-window" and emits the window-count-underconfigured arm.
    dynamic_trace_path = traces_folder / "dynamic_trace.pb"
    actual_trace_bytes = (
        dynamic_trace_path.stat().st_size
        if dynamic_trace_path.is_file() else 0
    )
    required_trace_bytes = (
        (K_CONSECUTIVE_WINDOWS + 1) * TRACE_WINDOW_SIZE_BYTES
    )

    # Trace volume budget sidecar binding: the convergence
    # `trace_convergence_not_reached_budget_cap` arm requires a paired
    # `trace_volume_exceeded.json` sidecar pinning the cap and the
    # actual record counts at cap-firing time.
    trace_volume_exceeded_path = (
        traces_folder / "trace_volume_exceeded.json"
    )

    # Decision tree: priority order matters. Disabled wins over all
    # because it expresses an explicit operator/config opt-out.
    if not trace_convergence_enabled:
        admission = disabled_admission()
    elif bundle_underconfigured or not sector_bins or not line_bins:
        unavailable = "sector" if not sector_bins else "line"
        admission = hrd_unavailable_admission(
            hrd_profile_path=rel_hrd,
            bundle_underconfigured=bundle_underconfigured,
            requested_layers=HIERARCHY_LAYERS,
            unavailable_layer=unavailable,
            usable_bin_count=(
                len(sector_bins) if unavailable == "sector" else len(line_bins)
            ),
        )
    elif trace_volume_exceeded_path.is_file():
        rel_volume_sidecar = trace_volume_exceeded_path.relative_to(
            traces_folder,
        ).as_posix()
        try:
            volume_sidecar = json.loads(
                trace_volume_exceeded_path.read_text(encoding="utf-8"),
            )
            max_trace_bytes = int(
                volume_sidecar.get("expected_record_count", required_trace_bytes),
            )
            actual_record_count = int(
                volume_sidecar.get("admitted_record_count", actual_trace_bytes),
            )
        except Exception:
            max_trace_bytes = required_trace_bytes
            actual_record_count = actual_trace_bytes
        admission = budget_cap_admission(
            windows=[],
            max_trace_bytes=max_trace_bytes,
            actual_trace_bytes=actual_record_count,
            trace_volume_exceeded_sidecar_path=rel_volume_sidecar,
        )
    elif actual_trace_bytes < required_trace_bytes:
        # Not enough cumulative trace bytes for K trailing converged
        # windows — emit the typed-unavailable arm naming the missing
        # evidence rather than fabricating convergence.
        admission = window_count_underconfigured_admission(
            observed_window_count=(
                1 if actual_trace_bytes > 0 else 0
            ),
            actual_trace_bytes=actual_trace_bytes,
        )
    else:
        # Trace is large enough in raw bytes; real cumulative-window
        # iteration would be performed here. The R38 implementation
        # does not yet ship the iterative HRD profiler driver, so the
        # producer falls through to the typed-unavailable arm with the
        # honest `observed_window_count = 1` (one final HRD bundle
        # exists; per-window iteration is not yet wired).
        admission = window_count_underconfigured_admission(
            observed_window_count=1,
            actual_trace_bytes=actual_trace_bytes,
        )

    # The convergence-confidence bundle requires `windows[]` with at
    # least one record. For typed-unavailable admissions emit a single
    # baseline window (window_index = 0, distance_to_previous = null).
    windows: list[dict[str, Any]] = [
        {
            "window_index": 0,
            "observation_count_cumulative": sum(
                int(b["observation_count"])
                for b in sector_bins + line_bins
                if isinstance(b.get("observation_count"), int)
            ),
            "trace_bytes_cumulative": actual_trace_bytes,
            "mrc_at_quantile": [
                {
                    "hierarchy_layer": layer,
                    "quantile": float(q),
                    "mrc_hit_rate_fraction": 0.0,
                    "quantile_observation_count": 0,
                }
                for layer in HIERARCHY_LAYERS
                for q in MRC_QUANTILES
            ],
            "distance_to_previous": None,
            "converged": False,
        },
    ]

    try:
        bundle = build_bundle(
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
            hrd_profile_path=rel_hrd,
            windows=windows,
            admission=admission,
            source_commit=source_commit,
        )
    except Exception as exc:
        _trace_convergence_skip_run_summary(
            artifact_record,
            skip_reason=f"trace_convergence_schema_invalid:{exc}",
        )
        return

    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8",
    )
    rel_bundle = (
        bundle_path.relative_to(artifact_dir)
        if bundle_path.is_relative_to(artifact_dir)
        else bundle_path
    )

    admitted = admission["result_kind"] == "ok"
    typed_reason = (
        admission.get("typed_reason") if not admitted else None
    )
    final_residual = admission.get("final_residual")
    windows_converged = (
        admission["windows_converged"] if admitted
        else admission.get("trailing_converged_windows", 0)
    )

    section: dict[str, Any] = {
        "trace_convergence_confidence_path": str(rel_bundle),
        "trace_convergence_schema_version": bundle["schema_version"],
        "trace_convergence_admitted": admitted,
        "trace_convergence_typed_reason": typed_reason,
        "trace_convergence_windows_observed": len(windows),
        "trace_convergence_windows_converged": windows_converged,
        "trace_convergence_final_residual": final_residual,
        "trace_convergence_mrc_quantile_set": list(MRC_QUANTILES),
        "trace_convergence_source_commit": source_commit,
        "trace_convergence_windows_digest": _windows_digest(windows),
        "trace_convergence_manifest_slug": manifest_pairing_index["slug"],
        "trace_convergence_manifest_shape_point": (
            manifest_pairing_index["shape_point"]
        ),
        "trace_convergence_manifest_binary_sha256": (
            manifest_pairing_index["binary_sha256"]
        ),
        "trace_convergence_manifest_source_commit": (
            manifest_pairing_index["source_commit"]
        ),
        "trace_convergence_manifest_kernel_signature_sha256": (
            manifest_pairing_index["kernel_signature_sha256"]
        ),
    }
    artifact_record.update(section)


def _kernel_args_from_shape(shape_point: dict, row: dict[str, Any]) -> list[str]:
    """Flatten shape axes + seed + eps into CLI args for the host driver."""
    args: list[str] = []
    axes = shape_point.get("axes", {})
    for k, v in axes.items():
        if isinstance(v, int):
            args.extend([f"--{k.replace('_', '-')}", str(v)])
    seed = int(row.get("seed_policy", {}).get("seed", 42))
    args.extend(["--seed", str(seed)])
    args.extend(["--shape-point", shape_point.get("name", "<?>")])
    return args


def handle_correctness(row: dict[str, Any], props: DeviceProps, dry_run: bool,
                       out_dir: Path | None, component_strict: bool = False) -> int:
    slug = row["slug"]
    family = row["reference_family"]
    ref_type = row["reference_type"]
    mod = _load_reference_family(family)
    if mod is None:
        print(f"reference_missing: references/{family}.py (needed by slug={slug}, reference_type={ref_type})", file=sys.stderr)
        return EXIT_USAGE
    covers = getattr(mod, "FAMILY_COVERS_SLUGS", ())
    if slug not in covers:
        print(f"reference_family_slug_mismatch: slug={slug} not in {family}.FAMILY_COVERS_SLUGS", file=sys.stderr)
        return EXIT_USAGE

    # Load the first correctness shape point + produce the reference output.
    shape_point = _first_correctness_shape(slug)
    if shape_point is None:
        # R32: --component-strict on an L4 row without correctness_shapes
        # must refuse loudly instead of silently returning EXIT_OK. Codex's
        # R31 review caught this false-green: both unshaped L4 rows
        # exited 0 under `--component-strict --dry-run-cpu` even though no
        # checkpoint could possibly be emitted. Refuse up-front with a typed
        # reason so the missing-L4-shape condition is observable.
        if component_strict and slug in L4_PIPELINE_SLUGS:
            print(
                f"correctness_shapes_missing_on_l4_row: --component-strict "
                f"requires shapes/{slug}.toml to declare at least one "
                f"correctness_shapes entry so the L4 reference can emit a "
                f"canonical checkpoint; this row has none.",
                file=sys.stderr,
            )
            return EXIT_UNSUPPORTED
        if dry_run:
            print(f"run_suite: dry-run correctness for {slug}: no correctness_shapes defined (dispatch verified without execution)")
            return EXIT_OK
        print(f"correctness_shapes_missing: shapes/{slug}.toml has no correctness_shapes", file=sys.stderr)
        return EXIT_UNSUPPORTED

    seed = int(row.get("seed_policy", {}).get("seed", 42))
    try:
        reference_output = mod.build_reference(slug, shape_point, seed)
    except NotImplementedError as e:
        print(f"reference_unimplemented: {e}", file=sys.stderr)
        return EXIT_UNSUPPORTED

    if dry_run:
        # Dry-run-cpu: treat the reference itself as the "actual" and self-compare.
        # This exercises the full dispatch + build + compare path without a GPU.
        tol = _load_tolerance_triplet(row["primary_precision"], _infer_reduction_kind(row), family)
        if tol is None:
            print(f"tolerance_unmapped: no tolerances.toml entry for ({row['primary_precision']}, {_infer_reduction_kind(row)}, {family})", file=sys.stderr)
            return EXIT_UNSUPPORTED
        ok = mod.compare(reference_output, reference_output, tol)
        if out_dir is not None:
            candidate_dir = out_dir / "dry_run"
            _emit_candidate(row, shape_point, reference_output, candidate_dir)
        if not ok:
            print(f"run_suite: dry-run correctness self-compare FAILED for {slug} — tolerance contract bug in family {family}", file=sys.stderr)
            return EXIT_CORRECTNESS_FAILED
        if component_strict and out_dir is not None:
            # R34: the dry-run-correctness path is the only
            # `_emit_l4_component_strict` caller that doesn't already emit
            # a sibling measurement_manifest.json before the call. Emit
            # it here so the suite_results' `measurement_manifest_path`
            # resolves to a real manifest carrying the correctness shape.
            dry_run_dir = out_dir / "dry_run"
            dry_run_dir.mkdir(parents=True, exist_ok=True)
            _emit_manifest(
                build_measurement_manifest(
                    slug=slug, mode="correctness",
                    warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
                    props=props, binary_path=None,
                    compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
                    shape_point=shape_point.get("name", ""),
                ),
                dry_run_dir,
            )
            rc = _emit_l4_component_strict(
                row=row, props=props, shape_point=shape_point, seed=seed, mod=mod,
                artifact_dir=dry_run_dir,
                mode="correctness",
                dry_run=True,
            )
            if rc != EXIT_OK:
                return rc
        print(f"run_suite: dry-run correctness for {slug}: reference_family={family} reference built + self-compare passed")
        return EXIT_OK

    # Non-dry-run path: subprocess-invoke the host driver with shape args,
    # parse its JSON output, compare against the Python reference.
    executable = _find_kernel_executable(slug)
    if executable is None:
        print(f"build_failed: no {slug}_run executable; run `make` in the kernel's directory first", file=sys.stderr)
        return EXIT_BUILD_FAILED
    args = _kernel_args_from_shape(shape_point, row)
    import subprocess as _sp
    proc = _sp.run([str(executable), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"build_failed: {executable} exited with rc={proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    try:
        kernel_result = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"correctness_failed: {slug} driver output is not valid JSON ({e})", file=sys.stderr)
        return EXIT_CORRECTNESS_FAILED
    # R48: a host driver may report that the device does not advertise the
    # arch feature it is exercising (e.g. cudaDevAttrMaxAccessPolicyWindowSize
    # not advertised on a non-Blackwell device). The driver emits a structured
    # JSON document with `result_kind == "typed_unavailable"` and an
    # `arch_feature_unavailable_*` typed_reason instead of a normal `output`
    # field; the harness persists the typed-unavailable record as the
    # candidate JSON so the run is observable as unsupported, not silently
    # admitted as a deterministic correctness pass. See
    # tools/perturbation_harness/ledger.PERTURBATION_RUN_LEDGER_TYPED_REASONS
    # for the canonical typed_reason list shared across surfaces.
    if kernel_result.get("result_kind") == "typed_unavailable":
        typed_reason = kernel_result.get("typed_reason")
        candidate_dir = Path(args_out_dir_root) / slug / _infer_gpu_sku(props) / _resolve_build_arch_for_row(row, props)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"candidate_{shape_point['name']}.json"
        with open(candidate_path, "w") as f:
            json.dump(kernel_result, f, indent=2)
        manifest = build_measurement_manifest(
            slug=slug, mode="correctness",
            warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
            props=props, binary_path=executable,
            compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
            shape_point=shape_point["name"],
        )
        _emit_manifest(manifest, candidate_dir)
        print(
            f"run_suite: correctness for {slug}: typed_unavailable "
            f"({typed_reason}); candidate at {candidate_path}"
        )
        return EXIT_OK
    kernel_output = kernel_result.get("output")
    if kernel_output is None:
        print(f"correctness_failed: {slug} driver JSON missing 'output' field", file=sys.stderr)
        return EXIT_CORRECTNESS_FAILED

    tol = _load_tolerance_triplet(row["primary_precision"], _infer_reduction_kind(row), family)
    if tol is None:
        print(f"tolerance_unmapped: no entry for ({row['primary_precision']}, {_infer_reduction_kind(row)}, {family})", file=sys.stderr)
        return EXIT_UNSUPPORTED
    if not mod.compare(kernel_output, reference_output, tol):
        print(f"correctness_failed: {slug} kernel output differs from reference beyond tolerance", file=sys.stderr)
        return EXIT_CORRECTNESS_FAILED

    # Emit the candidate JSON next to the measurement manifest. Platform + arch
    # come from the device identity recorded at startup; the arch is taken
    # from the row's declared `arch_gate` so Blackwell-only rows land at
    # `<platform>/sm_120a/` or `<platform>/sm_120f/` rather than collapsing
    # into the baseline `sm_120` path.
    candidate_dir = Path(args_out_dir_root) / slug / _infer_gpu_sku(props) / _resolve_build_arch_for_row(row, props)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"candidate_{shape_point['name']}.json"
    with open(candidate_path, "w") as f:
        json.dump(kernel_result, f, indent=2)
    manifest = build_measurement_manifest(
        slug=slug, mode="correctness",
        warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
        props=props, binary_path=executable,
        compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
        shape_point=shape_point["name"],
    )
    _emit_manifest(manifest, candidate_dir)
    if component_strict:
        rc = _emit_l4_component_strict(
            row=row, props=props, shape_point=shape_point, seed=seed, mod=mod,
            artifact_dir=candidate_dir,
            mode="correctness",
        )
        if rc != EXIT_OK:
            return rc
    print(f"run_suite: correctness for {slug}: native match under tolerance (candidate at {candidate_path})")
    return EXIT_OK


# L4 pipeline slugs that carry a canonical component-level checkpoint schema
# under `schemas/l4_checkpoints/<slug>.schema.json`. Kept in sync with
# `validate.py::L4_SCORED_SLUGS` — any future L4 row added to the roster
# must appear in both sets.
L4_PIPELINE_SLUGS = frozenset({
    "moe_layer_residual_norm_pipeline",
    "prefill_then_decode_pagedkv_pipeline",
    "speculative_decode_draft_verify_pipeline",
})


# Per-pipeline component stage lists used for `component_health` in
# `suite_results.json`. The stage whose name matches the checkpoint
# schema's boundary gets the `checkpoint_artifact` reference; other
# stages are reported pass/fail only.
_L4_COMPONENT_STAGES: dict[str, list[tuple[str, bool]]] = {
    "moe_layer_residual_norm_pipeline": [
        ("router",       True),  # this stage owns the checkpoint boundary
        ("grouped_gemm", False),
        ("combine",      False),
        ("residual",     False),
        ("rmsnorm",      False),
    ],
    "prefill_then_decode_pagedkv_pipeline": [
        ("prefill",      True),  # prefill stage owns the prefill-to-decode KV boundary
        ("decode",       False),
    ],
    "speculative_decode_draft_verify_pipeline": [
        ("draft",        True),  # draft stage owns the draft-to-verifier boundary
        ("verify",       False),
    ],
}


def _build_l4_component_health(slug: str, checkpoint_artifact_name: str) -> list[dict]:
    """Build the component_health list for a successful L4 correctness run.
    Every stage is marked passed (the reference + compare succeeded); the
    stage owning the checkpoint boundary carries `checkpoint_artifact`."""
    stages = _L4_COMPONENT_STAGES.get(slug, [("reference", True)])
    out: list[dict] = []
    for name, has_artifact in stages:
        entry = {"component": name, "passed": True}
        if has_artifact:
            entry["checkpoint_artifact"] = checkpoint_artifact_name
        out.append(entry)
    return out


def _l4_checkpoint_diff(emitted: dict, reference: dict) -> list[str]:
    """Compare two L4 checkpoint dicts field-by-field. Used in R37
    to enforce that the kernel-emitted checkpoint matches the Python
    reference's `build_l4_checkpoint()` output byte-for-byte. Returns
    a list of human-readable diff descriptions; empty list means the
    two are equivalent for the schema-significant fields.

    Schema-significant fields per the L4 checkpoint schemas: every
    leaf object with `digest` (must be hex-equal) or `values` (must
    be ordered-list-equal). `shape` and `dtype` fields are also
    compared. Top-level `slug` / `checkpoint_name` / `schema_version`
    must match. Optional `rope_kind` field on the prefill_then_decode
    schema is also compared when present."""
    diffs: list[str] = []
    for top in ("schema_version", "slug", "checkpoint_name"):
        if emitted.get(top) != reference.get(top):
            diffs.append(f"top-level {top}: emitted={emitted.get(top)!r} reference={reference.get(top)!r}")
    # Recurse into per-field objects.
    ref_keys = {k for k in reference if k not in ("schema_version", "slug", "checkpoint_name")}
    emt_keys = {k for k in emitted if k not in ("schema_version", "slug", "checkpoint_name")}
    if ref_keys != emt_keys:
        diffs.append(f"field set differs: emitted-only={emt_keys - ref_keys} reference-only={ref_keys - emt_keys}")
    for field in sorted(ref_keys & emt_keys):
        ref_obj = reference[field]
        emt_obj = emitted[field]
        if not isinstance(ref_obj, dict) or not isinstance(emt_obj, dict):
            if ref_obj != emt_obj:
                diffs.append(f"{field}: not a dict; emitted={emt_obj!r} reference={ref_obj!r}")
            continue
        for sub in ("shape", "dtype", "rope_kind"):
            if sub in ref_obj or sub in emt_obj:
                if ref_obj.get(sub) != emt_obj.get(sub):
                    diffs.append(f"{field}.{sub}: emitted={emt_obj.get(sub)!r} reference={ref_obj.get(sub)!r}")
        if "digest" in ref_obj or "digest" in emt_obj:
            if ref_obj.get("digest") != emt_obj.get("digest"):
                diffs.append(
                    f"{field}.digest: emitted={emt_obj.get('digest')} reference={ref_obj.get('digest')}"
                )
        if "values" in ref_obj or "values" in emt_obj:
            if ref_obj.get("values") != emt_obj.get("values"):
                diffs.append(
                    f"{field}.values(): emitted={emt_obj.get('values')!r} reference={ref_obj.get('values')!r}"
                )
    return diffs


def _emit_l4_component_strict(
    *,
    row: dict[str, Any],
    props: DeviceProps,
    shape_point: dict[str, Any],
    seed: int,
    mod: Any,
    artifact_dir: Path,
    mode: str,
    measured_timing_ms: dict | None = None,
    dry_run: bool = False,
    measurement_manifest_relpath: str = "measurement_manifest.json",
) -> int:
    """Emit an L4 component checkpoint + suite_results.json under
    `--component-strict`. Mode is either `"correctness"` (emits
    suite_results with `status="unsupported"`, `reason="pending_timing"`
    — no placeholder native timing) or `"timing"` (emits with
    `status="native"` + `measured_timing_ms` from the real kernel
    timing loop). Returns EXIT_OK on success, EXIT_CORRECTNESS_FAILED
    when the checkpoint or suite_results fails its canonical schema.

    When `dry_run=True`, the checkpoint is synthesized from the
    Python reference (`build_l4_checkpoint`) rather than invoking
    the binary. This preserves the pre-Round-37 CPU-only smoke-test
    path; `--mode correctness --dry-run-cpu` explicitly bypasses
    the kernel, so requiring a built binary here would turn the
    dry-run path into a no-op on machines without CUDA.

    When `dry_run=False` (the default for non-dry-run correctness
    and for timing mode), the checkpoint is emitted by invoking
    the kernel binary with `--emit-l4-checkpoint` and validated
    against the Python reference as an ORACLE (byte-for-byte
    equality on digest + values fields). Missing binary returns
    EXIT_BUILD_FAILED (fail-closed; no silent fallback).

    The caller is responsible for emitting the sibling
    `measurement_manifest.json` with the correct `shape_point`
    (timing shape on the timing path; correctness shape on the
    correctness path). R34 hoisted the manifest emit out of
    here so the timing-path manifest stops being overwritten with
    the checkpoint shape."""
    slug = row["slug"]
    if slug not in L4_PIPELINE_SLUGS:
        # Non-L4 row under --component-strict: no-op.
        return EXIT_OK
    build_l4_checkpoint = getattr(mod, "build_l4_checkpoint", None)
    if build_l4_checkpoint is None:
        print(
            f"l4_checkpoint_unimplemented: slug={slug} is an L4 pipeline, but "
            f"references/{row['reference_family']}.py does not export "
            f"build_l4_checkpoint",
            file=sys.stderr,
        )
        return EXIT_UNSUPPORTED

    # ------------------------------------------------------------------------
    # R37 (Codex R36-review BLOCKING fix): the checkpoint artifact
    # is now produced by the kernel binary, not synthesized from the Python
    # reference. The Python reference's `build_l4_checkpoint()` becomes the
    # comparison ORACLE: we invoke the binary with `--emit-l4-checkpoint`,
    # validate the schema, and assert byte-equality between the binary's
    # digest fields and the reference's digest fields. Pre-Round-37 code
    # path called `build_l4_checkpoint(...)` and wrote the result directly,
    # which silently bypassed any kernel-side proof of contract compliance.
    # ------------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / f"l4_checkpoint_{slug}.json"

    if dry_run:
        # CPU-only smoke-test path: synthesize the checkpoint from the
        # Python reference (pre-Round-37 behavior). Honest because
        # `--dry-run-cpu` explicitly bypasses the kernel; the
        # non-dry-run path (`dry_run=False`) still requires a
        # kernel-emitted checkpoint per Codex R36-review.
        checkpoint = build_l4_checkpoint(slug, shape_point, seed)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
    else:
        binary = _find_kernel_executable(slug)
        if binary is None:
            print(
                f"l4_checkpoint_binary_missing: slug={slug} requires a built "
                f"binary at l4/{slug}/{slug}_run to emit its kernel-backed "
                f"checkpoint; run `make` in that directory first.",
                file=sys.stderr,
            )
            return EXIT_BUILD_FAILED

        import subprocess as _sp
        args = _kernel_args_from_shape(shape_point, row)
        proc = _sp.run(
            [str(binary), *args, "--emit-l4-checkpoint", str(checkpoint_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(
                f"l4_checkpoint_emit_failed: binary {binary} exited rc={proc.returncode} "
                f"under --emit-l4-checkpoint:\n{proc.stderr.strip()}",
                file=sys.stderr,
            )
            return EXIT_CORRECTNESS_FAILED
        if not checkpoint_path.exists():
            print(
                f"l4_checkpoint_emit_missing: binary {binary} returned 0 but did not "
                f"write {checkpoint_path}",
                file=sys.stderr,
            )
            return EXIT_CORRECTNESS_FAILED

    # Schema-validate the binary's output against the row's canonical schema.
    import importlib.util
    val_spec = importlib.util.spec_from_file_location("simbench_bw_validate_inproc", SUITE_ROOT / "validate.py")
    assert val_spec is not None and val_spec.loader is not None
    val = importlib.util.module_from_spec(val_spec)
    val_spec.loader.exec_module(val)
    errors = val.validate_l4_checkpoint(checkpoint_path)
    if errors:
        print(f"l4_checkpoint_schema_failed: {checkpoint_path} (kernel-emitted) failed its canonical schema:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return EXIT_CORRECTNESS_FAILED

    # Compare against the Python reference (oracle). Every digest field
    # and every `values` field must match byte-for-byte. Tolerance bands
    # do not apply to checkpoint contents — the binary must reproduce
    # the canonical state exactly per BL-20260424-kernel-contract-vs-shape-match.
    reference = build_l4_checkpoint(slug, shape_point, seed)
    with open(checkpoint_path, "r") as f:
        emitted = json.load(f)
    diffs = _l4_checkpoint_diff(emitted, reference)
    if diffs:
        print(
            f"l4_checkpoint_oracle_mismatch: {checkpoint_path} (kernel-emitted) "
            f"diverged from Python reference oracle:",
            file=sys.stderr,
        )
        for d in diffs:
            print(f"  {d}", file=sys.stderr)
        return EXIT_CORRECTNESS_FAILED

    # Emit suite_results.json with mode-appropriate status + timing.
    # R34: the manifest emit was hoisted out — `_build_single_row_suite_results`
    # no longer writes the sibling manifest, so it doesn't need
    # binary_path / shape_point / artifact_dir. The caller is
    # responsible for emitting the matching `measurement_manifest.json`
    # with the correct shape (timing shape on the timing path,
    # correctness shape on the correctness path).
    suite_results = _build_single_row_suite_results(
        row=row,
        props=props,
        mode=mode,
        measured_timing_ms=measured_timing_ms,
        checkpoint_artifact_name=checkpoint_path.name,
        measurement_manifest_relpath=measurement_manifest_relpath,
    )
    suite_results_path = artifact_dir / "suite_results.json"
    with open(suite_results_path, "w") as f:
        json.dump(suite_results, f, indent=2)

    sr_errors = val.validate_suite_results(suite_results_path)
    sr_errors.extend(val._l4_scoring_integrity(suite_results))
    if sr_errors:
        print(f"suite_results_schema_failed: {suite_results_path} failed validation:", file=sys.stderr)
        for e in sr_errors:
            print(f"  {e}", file=sys.stderr)
        return EXIT_CORRECTNESS_FAILED

    print(
        f"run_suite: component-strict L4 checkpoint emitted at {checkpoint_path} "
        f"(suite_results.json at {suite_results_path}, mode={mode})"
    )
    return EXIT_OK


def _build_single_row_suite_results(
    *,
    row: dict[str, Any],
    props: DeviceProps,
    mode: str,
    measured_timing_ms: dict | None,
    checkpoint_artifact_name: str,
    measurement_manifest_relpath: str = "measurement_manifest.json",
) -> dict:
    """Per-row suite_results.json payload for L4 runs under
    `--component-strict`.

    R33 (Codex R32-review gap #2):
    - mode="correctness" → status="unsupported" + reason="pending_timing".
      No `timing_ms` field; the L4 scoring integrity check skips
      unsupported rows. The row still carries `component_health` so
      a downstream timing-mode run can merge it in.
    - mode="timing" → status="native" + `measured_timing_ms` from the
      real timing loop. suite_results.schema.json requires
      `timing_ms.median > 0` when status=native.

    R34 (Codex R33-review blocker): the
    `measurement_manifest.json` emit was hoisted out of this builder.
    The timing path's manifest (with `shape_point="timing_00"`) was
    being overwritten by the checkpoint shape ("correctness_00")
    here, breaking the schema's manifest-describes-the-measurement
    contract. Manifest emission now lives at the call sites
    (handle_timing emits its own; handle_correctness emits at
    candidate_dir; the dry-run-correctness path emits explicitly
    before calling `_emit_l4_component_strict`).
    """
    slug = row["slug"]
    tb_hash = hashlib.sha256((SUITE_ROOT / "trace_budgets.toml").read_bytes()).hexdigest()
    cc = (props.compute_capability_major, props.compute_capability_minor)
    platform = _infer_gpu_sku(props)
    if cc == (0, 0) or platform.endswith("_unknown"):
        platform = "dry_run_cpu"
    component_health = _build_l4_component_health(slug, checkpoint_artifact_name)

    row_payload: dict[str, Any] = {
        "slug": slug,
        "platform": platform,
    }
    if mode == "timing":
        if measured_timing_ms is None:
            raise ValueError("mode='timing' requires measured_timing_ms")
        row_payload["status"] = "native"
        row_payload["timing_ms"] = measured_timing_ms
        row_payload["component_health"] = component_health
        row_payload["measurement_manifest_path"] = measurement_manifest_relpath
    elif mode == "correctness":
        row_payload["status"] = "unsupported"
        row_payload["reason"] = "pending_timing"
        row_payload["component_health"] = component_health
    else:
        raise ValueError(f"mode must be 'correctness' or 'timing', got {mode!r}")

    return {
        "schema_version": 1,
        "git_commit": _suite_results_git_commit(),
        "suite_tag_candidate": None,
        "cuda_version": props.cuda_runtime_version or "n/a",
        "driver_version": props.driver_version or "n/a",
        "trace_budgets_sha256": tb_hash,
        "rows": [row_payload],
        "vendor_control_only": [],
    }


def _suite_results_git_commit() -> str:
    """Return the HEAD commit hash for suite_results.json provenance.
    Falls back to a 7-char placeholder that still passes the schema
    `^[0-9a-f]{7,40}$` pattern when git is unavailable."""
    import subprocess as _sp
    try:
        proc = _sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SUITE_ROOT.parent.parent,
            capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "0000000"  # 7 hex chars, still schema-valid


def _artifact_dir_for_row(row: dict[str, Any], props: DeviceProps) -> Path:
    """Canonical per-(slug, platform, arch) artifact directory. Matches the
    layout used by correctness, timing, trace, and regression modes so their
    artifacts land next to each other under
    `results/<slug>/<gpu_sku>/<arch>/`. Sourced from `args_out_dir_root`
    so `--out` overrides (used by tests) flow through."""
    return (Path(args_out_dir_root)
            / row["slug"]
            / _infer_gpu_sku(props)
            / _resolve_build_arch_for_row(row, props))


def handle_timing(row: dict[str, Any], props: DeviceProps, out_dir: Path,
                  component_strict: bool = False,
                  shape_point_name: str | None = None) -> int:
    """Execute the built kernel binary against the row's first `timing_shapes`
    entry with warm-up + repeat iterations and emit a timing artifact plus a
    sibling measurement manifest. Refuses cleanly with a typed reason when the
    kernel binary is missing or the shape file lacks a timing point.

    R33 AC32 integration: when `component_strict` is set AND the row
    is an L4 pipeline slug, also emit `l4_checkpoint_<slug>.json` and
    `suite_results.json` with measured `timing_ms` + `component_health`
    (the real kernel timing, not the Python reference build-time
    placeholder). This is the canonical scored-L4 path.

    """
    _ = out_dir
    slug = row["slug"]

    executable = _find_kernel_executable(slug)
    if executable is None:
        print(f"executable_missing: {slug} has no {slug}_run executable (build the kernel first)", file=sys.stderr)
        return EXIT_UNSUPPORTED

    shape_path = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not shape_path.exists():
        print(f"shapes_missing: {shape_path}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    with open(shape_path, "rb") as f:
        shape = tomllib.load(f)
    timing_shapes = shape.get("timing_shapes", [])
    if not timing_shapes:
        print(f"timing_shapes_missing: shapes/{slug}.toml has no timing_shapes", file=sys.stderr)
        return EXIT_UNSUPPORTED
    if shape_point_name is not None:
        named_matches = [s for s in timing_shapes if s.get("name") == shape_point_name]
        if not named_matches:
            print(
                f"shape_point_not_found_in_timing_shapes: {shape_point_name!r} not in "
                f"{[s.get('name') for s in timing_shapes]} (shapes/{slug}.toml)",
                file=sys.stderr,
            )
            return EXIT_USAGE
        shape_point = named_matches[0]
    else:
        shape_point = timing_shapes[0]
    warm_up = int(shape_point.get("warm_up_iters", 5))
    repeat = int(shape_point.get("repeat_iters", 20))

    args = _kernel_args_from_shape(shape_point, row)
    import subprocess as _sp
    import time as _time
    import statistics as _stats

    def _stdout_json_dict(stdout: str) -> dict[str, Any] | None:
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            return None
        return result if isinstance(result, dict) else None

    def _persist_typed_unavailable_timing(result: dict[str, Any]) -> int:
        typed_reason = result.get("typed_reason")
        artifact_dir = _artifact_dir_for_row(row, props)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timing_path = artifact_dir / f"timing_{shape_point['name']}.json"
        with open(timing_path, "w") as f:
            json.dump(result, f, indent=2)
        manifest = build_measurement_manifest(
            slug=slug, mode="timing",
            warm_up_iters=warm_up, repeat_iters=repeat, cv_pct=0.0,
            props=props, binary_path=executable,
            compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
            shape_point=shape_point["name"],
        )
        _emit_manifest(manifest, artifact_dir)
        print(
            f"run_suite: timing for {slug}: typed_unavailable "
            f"({typed_reason}); artifact at {timing_path}"
        )
        return EXIT_OK

    for warm_up_index in range(warm_up):
        proc = _sp.run([str(executable), *args], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"build_failed: {executable} warm-up rc={proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
            return EXIT_BUILD_FAILED
        # R49: typed-unavailable propagation for timing mode. If the
        # R49: typed-unavailable propagation for timing mode.
        # Any warm-up stdout reporting `result_kind == "typed_unavailable"`
        # is a typed outcome, not a timing sample.
        warmup_result = _stdout_json_dict(proc.stdout)
        if (warmup_result is not None
                and warmup_result.get("result_kind") == "typed_unavailable"):
            return _persist_typed_unavailable_timing(warmup_result)

    artifact_dir = _artifact_dir_for_row(row, props)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    wall_times_ms: list[float] = []
    for _repeat_index in range(repeat):
        t0 = _time.perf_counter()
        proc = _sp.run(
            [str(executable), *args],
            capture_output=True, text=True,
        )
        t1 = _time.perf_counter()
        if proc.returncode != 0:
            print(f"build_failed: {executable} timed run rc={proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
            return EXIT_BUILD_FAILED
        # Prefer the kernel's self-reported wall time when the host driver
        # emits one; fall back to subprocess wall time. A typed-unavailable
        # stdout is not a timing sample, including zero-warm-up shapes where
        # the first chance to observe it is a measured repeat.
        reported_ms: float | None = None
        result = _stdout_json_dict(proc.stdout)
        if (result is not None
                and result.get("result_kind") == "typed_unavailable"):
            return _persist_typed_unavailable_timing(result)
        if result is not None:
            val = result.get("wall_time_ms") if isinstance(result, dict) else None
            if isinstance(val, (int, float)) and val > 0:
                reported_ms = float(val)
        repeat_wall_time_ms = (
            reported_ms if reported_ms is not None else (t1 - t0) * 1000.0
        )
        wall_times_ms.append(repeat_wall_time_ms)

    if wall_times_ms:
        median_ms = float(_stats.median(wall_times_ms))
        mean_ms = float(_stats.mean(wall_times_ms))
        if len(wall_times_ms) > 1 and mean_ms > 0:
            cv_pct = (float(_stats.stdev(wall_times_ms)) / mean_ms) * 100.0
        else:
            cv_pct = 0.0
    else:
        median_ms = 0.0
        cv_pct = 0.0

    timing_artifact: dict[str, Any] = {
        "schema_version": 1,
        "slug": slug,
        "shape_set": "timing_shapes",
        "shape_point": shape_point["name"],
        "timing_ms": {
            "median": median_ms,
            "cv_pct": cv_pct,
            "warm_up_iters": warm_up,
            "repeat_iters": repeat,
        },
        "samples_ms": wall_times_ms,
    }
    artifact_path = artifact_dir / f"timing_{shape_point['name']}.json"
    artifact_path.write_text(json.dumps(timing_artifact, indent=2))

    manifest = build_measurement_manifest(
        slug=slug, mode="timing",
        warm_up_iters=warm_up, repeat_iters=repeat, cv_pct=cv_pct,
        props=props, binary_path=executable,
        compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
        shape_point=shape_point["name"],
    )
    _emit_manifest(manifest, artifact_dir)

    # R33 AC32: L4 scored-row emission on the timing path.
    if component_strict and slug in L4_PIPELINE_SLUGS:
        mod = _load_reference_family(row["reference_family"])
        if mod is None:
            print(f"reference_missing: references/{row['reference_family']}.py (needed for L4 checkpoint)", file=sys.stderr)
            return EXIT_USAGE
        # Use the correctness_shapes[0] point for the L4 checkpoint
        # (the checkpoint encodes the boundary state, not the timing
        # workload). If the shape file has no correctness shape, the
        # component-strict contract cannot be satisfied.
        checkpoint_shape = _first_correctness_shape(slug)
        if checkpoint_shape is None:
            print(
                f"correctness_shapes_missing_on_l4_row: --component-strict on "
                f"handle_timing requires shapes/{slug}.toml to declare at "
                f"least one correctness_shapes entry so the L4 boundary state "
                f"can be constructed for the checkpoint.",
                file=sys.stderr,
            )
            return EXIT_UNSUPPORTED
        seed = int(row.get("seed_policy", {}).get("seed", 42))
        measured = {
            "median": median_ms,
            "cv_pct": cv_pct,
            "warm_up_iters": warm_up,
            "repeat_iters": repeat,
        }
        rc = _emit_l4_component_strict(
            row=row, props=props, shape_point=checkpoint_shape, seed=seed,
            mod=mod, artifact_dir=artifact_dir, mode="timing",
            measured_timing_ms=measured,
            measurement_manifest_relpath="measurement_manifest.json",
        )
        if rc != EXIT_OK:
            return rc

    print(f"run_suite: timing for {slug}: median={median_ms:.3f}ms cv={cv_pct:.2f}% (artifact at {artifact_path})")
    return EXIT_OK


# Absolute path to the H3-preserved, SHA-bound NVBit tracer .so. Tests patch
# this module attribute directly to exercise the missing-artifact refusal path.
# Canonical checkout rebuild/clean/repin is prohibited while H3 holds.
NVBIT_TRACER_SO = SUITE_ROOT.parent.parent / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"


def _emit_nvbit_trace_summaries_for_trace_artifact(
    *,
    traces_folder: Path,
    trace_pb: Path,
    slug: str,
    shape_point: str,
    executable: Path,
    tracer_so_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Per-launch NVBit trace summary emission, factored out of
    `handle_trace` so the orchestrator's interaction with
    `nvbit_trace_summary_adapter` is host-independently testable
    via `tests/test_nvbit_trace_summary_adapter.py`.

    Returns a dict with:
      - `status`: one of `ok`, `adapter_import_failed`,
        `tracked_protobuf_corrupt`, `protobuf_runtime_prerequisite_missing`,
        `adapter_error`.
      - `paths`: list of repo-relative summary paths (relative to
        `artifact_dir`); empty when status is non-ok.
      - `failure_reason`: human-readable diagnostic when status is
        non-ok; None on success.
      - `failed_hard`: True for every non-ok dependency or adapter result.

    The adapter and all nine `_pb2.py` files are tracked authorities. Missing
    source/binding bytes are repository corruption; a missing protobuf runtime
    is a prerequisite failure. Neither condition is a graceful trace result.
    """
    result: dict[str, Any] = {
        "status": "not_attempted",
        "paths": [],
        "failure_reason": None,
        "failed_hard": False,
    }
    try:
        import nvbit_trace_summary_adapter as _nvbit_adapter  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        result["status"] = "adapter_import_failed"
        result["failure_reason"] = (
            "tracked adapter source is unavailable; restore "
            "workloads/simbench_bw/nvbit_trace_summary_adapter.py from "
            f"the committed tree: {exc}"
        )
        result["failed_hard"] = True
        return result
    try:
        launch_count = _nvbit_adapter.enumerate_launches(trace_pb)
    except _nvbit_adapter.NvbitTrackedProtobufCorruption as exc:
        result["status"] = "tracked_protobuf_corrupt"
        result["failure_reason"] = str(exc)
        result["failed_hard"] = True
        return result
    except _nvbit_adapter.NvbitProtobufRuntimePrerequisiteMissing as exc:
        result["status"] = "protobuf_runtime_prerequisite_missing"
        result["failure_reason"] = str(exc)
        result["failed_hard"] = True
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "adapter_error"
        result["failure_reason"] = (
            f"enumerate_launches: {type(exc).__name__}: {exc}"
        )
        result["failed_hard"] = True
        return result

    paths: list[str] = []
    for launch_index in range(launch_count):
        summary_out_path = (
            artifact_dir / f"nvbit_trace_summary_0_{launch_index}.json"
        )
        try:
            _nvbit_adapter.emit_nvbit_trace_summary(
                traces_dir=traces_folder,
                slug=slug,
                shape_point=shape_point,
                repeat_index=0,
                launch_index=launch_index,
                executable_path=executable,
                tracer_so_path=tracer_so_path,
                output_path=summary_out_path,
            )
            paths.append(str(summary_out_path.relative_to(artifact_dir)))
        except _nvbit_adapter.NvbitTrackedProtobufCorruption as exc:
            result["status"] = "tracked_protobuf_corrupt"
            result["failure_reason"] = str(exc)
            result["paths"] = []
            result["failed_hard"] = True
            return result
        except _nvbit_adapter.NvbitProtobufRuntimePrerequisiteMissing as exc:
            result["status"] = "protobuf_runtime_prerequisite_missing"
            result["failure_reason"] = str(exc)
            result["paths"] = []
            result["failed_hard"] = True
            return result
        except Exception as exc:  # noqa: BLE001
            result["status"] = "adapter_error"
            result["failure_reason"] = (
                f"emit_nvbit_trace_summary launch_index={launch_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            result["failed_hard"] = True
            result["paths"] = []
            return result

    result["status"] = "ok"
    result["paths"] = paths
    return result


def handle_trace(row: dict[str, Any], props: DeviceProps, out_dir: Path,
                 shape_point_name: str | None = None,
                 phase_inject_ab: bool = False,
                 ab_repeat_count: int = 3) -> int:
    slug = row["slug"]
    shape_path = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not shape_path.exists():
        print(f"run_suite: trace for {slug} requires shapes/{slug}.toml (missing)", file=sys.stderr)
        return EXIT_UNSUPPORTED
    # Delegate to the trace_size_estimator for per-point validation. This
    # recomputes the static predictor and flags trace_prediction_stale when
    # the committed predicted_bytes has drifted, AND catches trace_shape_overbudget
    # before any tracer invocation would occur.
    import importlib.util
    est_path = SUITE_ROOT / "trace_size_estimator.py"
    spec = importlib.util.spec_from_file_location("simbench_bw_trace_estimator", est_path)
    assert spec is not None and spec.loader is not None
    est = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(est)
    estimator_errors = est.validate_slug(slug)
    with open(shape_path, "rb") as f:
        shape = tomllib.load(f)
    budget_bytes = row["trace_budget_mib"] * 1024 * 1024
    stale = [e for e in estimator_errors if "trace_prediction_stale" in e]
    overbudget = [e for e in estimator_errors if "trace_shape_overbudget" in e]
    if stale:
        for e in stale:
            print(e, file=sys.stderr)
        return EXIT_UNSUPPORTED
    if overbudget:
        import hashlib
        out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = out_dir / "trace_volume_exceeded.json"
        # Pick the worst offender for the sidecar
        worst_point = None
        worst_bytes = 0
        for point in shape.get("trace_shapes", []):
            pred = int(point.get("predicted_bytes") or 0)
            if pred > worst_bytes:
                worst_bytes = pred
                worst_point = point.get("name", "<?>")
        tb_hash = hashlib.sha256((SUITE_ROOT / "trace_budgets.toml").read_bytes()).hexdigest()
        sidecar.write_text(json.dumps({
            "schema_version": 1,
            "slug": slug,
            "shape_point": worst_point or "<?>",
            "sampling_policy": "off",
            "max_trace_bytes": budget_bytes,
            "actual_trace_bytes": worst_bytes,
            "max_wall_time_s": 3600,
            "actual_wall_time_s": 0.0,
            "exceeded_trace_bytes": True,
            "exceeded_wall_time": False,
            "trace_budgets_sha256": tb_hash,
        }, indent=2))
        for e in overbudget:
            print(e, file=sys.stderr)
        print(f"trace_shape_overbudget: sidecar at {sidecar}", file=sys.stderr)
        return EXIT_SAMPLING_REQUIRED

    # Pre-admission passed. Now dispatch the NVBit tracer against the built
    # kernel binary. The tracer is the H3-preserved .so; a missing artifact is
    # not an invitation to rebuild in the canonical checkout.
    tracer_so = NVBIT_TRACER_SO
    if not tracer_so.exists():
        print(
            f"tracer_preserved_artifact_missing: {tracer_so} is missing; "
            "restore the byte-exact SHA-bound artifact through the H3 "
            "operator workflow; canonical rebuild is prohibited",
            file=sys.stderr,
        )
        return EXIT_UNSUPPORTED
    executable = _find_kernel_executable(slug)
    if executable is None:
        print(f"executable_missing: {slug} has no {slug}_run executable (build the kernel first)", file=sys.stderr)
        return EXIT_UNSUPPORTED
    trace_shapes = shape.get("trace_shapes", [])
    if not trace_shapes:
        print(f"trace_shapes_missing: shapes/{slug}.toml has no trace_shapes", file=sys.stderr)
        return EXIT_UNSUPPORTED
    if shape_point_name is not None:
        named_matches = [s for s in trace_shapes if s.get("name") == shape_point_name]
        if not named_matches:
            print(
                f"shape_point_not_found_in_trace_shapes: {shape_point_name!r} not in "
                f"{[s.get('name') for s in trace_shapes]} (shapes/{slug}.toml)",
                file=sys.stderr,
            )
            return EXIT_USAGE
        shape_point = named_matches[0]
    else:
        shape_point = trace_shapes[0]

    artifact_dir = _artifact_dir_for_row(row, props)
    traces_folder = artifact_dir / f"trace_{shape_point['name']}"
    # R51: pre-clean any prior per-shape trace folder so a fresh capture
    # cannot inherit a stale `dynamic_trace.pb`, stale `stats.csv`, or
    # stale per-kernel `.trace` files from a previous run on the same
    # shape point. Without the cleanup the post-capture
    # `trace_missing` check (which only re-checks file existence) and
    # `require_trace_admission` would otherwise inspect a mixed tree from two
    # captures and admit a tree whose `dynamic_trace.pb` was produced
    # by a different binary, different injected library, or different
    # kernel arguments — exactly the false-positive admission shape
    # that R50's bypass-orchestrator capture hid.
    if traces_folder.exists():
        import shutil as _shutil
        _shutil.rmtree(traces_folder)
    traces_folder.mkdir(parents=True, exist_ok=True)
    args = _kernel_args_from_shape(shape_point, row)

    import subprocess as _sp
    import time as _time
    # C-AC-H baseline timing (Loop 2 R11). Measure a non-instrumented
    # baseline run of the same executable + shape args immediately
    # before the NVBit run. The baseline sets the denominator for the
    # slowdown-ratio admissibility check; without it the verifier
    # cannot independently decide that a run is within budget. The
    # baseline run uses a fresh env that scrubs CUDA_INJECTION64_PATH
    # so the host executable runs without NVBit injection. A baseline
    # failure does not block the NVBit run; it leaves
    # `baseline_wall_time_s = None` in the run summary, and the
    # verifier treats absent baseline as the typed-unavailable arm
    # (no enforcement on that run).
    baseline_env = {
        k: v for k, v in os.environ.items()
        if k != "CUDA_INJECTION64_PATH"
    }
    baseline_env["TRACES_FOLDER"] = str(traces_folder)
    baseline_t0 = _time.perf_counter()
    baseline_proc = _sp.run(
        [str(executable), *args],
        capture_output=True, text=True, env=baseline_env,
    )
    baseline_t1 = _time.perf_counter()
    # C-AC-H R13: prefer the workload's self-reported `wall_time_ms`
    # (matching the existing `handle_timing` stdout-preferred pattern)
    # over the host `perf_counter` measurement. The self-reported
    # device-side timing excludes process-launch overhead and host
    # scheduling jitter, which is what the admissibility-budget
    # enforcement should measure. When stdout lacks the field, fall
    # back to the host timer.
    if baseline_proc.returncode == 0:
        reported = _stdout_wall_time_s(baseline_proc.stdout)
        baseline_wall_time_s: float | None = (
            reported if reported is not None
            else baseline_t1 - baseline_t0
        )
    else:
        baseline_wall_time_s = None
    # R49: typed-unavailable propagation for trace mode. If the
    # baseline (uninstrumented) workload run reports
    # `result_kind == "typed_unavailable"`, the kernel never
    # launched a CUDA kernel — the host driver detected an
    # unsupported arch feature and exited cleanly. Skipping the
    # NVBit-injected run is required: a typed-unavailable workload
    # produces no `dynamic_trace.pb`, so the tracer-side check at
    # the bottom of this function would (correctly) fire
    # `trace_missing`. We persist a typed-unavailable trace
    # artifact + manifest so the unsupported-device run is
    # observable as a typed outcome rather than dropped.
    if baseline_proc.returncode == 0:
        try:
            baseline_result = json.loads(baseline_proc.stdout)
        except json.JSONDecodeError:
            baseline_result = None
        if (isinstance(baseline_result, dict)
                and baseline_result.get("result_kind") == "typed_unavailable"):
            typed_reason = baseline_result.get("typed_reason")
            trace_artifact = artifact_dir / f"trace_{shape_point['name']}.json"
            with open(trace_artifact, "w") as f:
                json.dump(baseline_result, f, indent=2)
            manifest = build_measurement_manifest(
                slug=slug, mode="trace",
                warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
                props=props, binary_path=executable,
                compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
                shape_point=shape_point["name"],
            )
            _emit_manifest(manifest, artifact_dir)
            print(
                f"run_suite: trace for {slug}: typed_unavailable "
                f"({typed_reason}); artifact at {trace_artifact}"
            )
            return EXIT_OK
    try:
        env = tracer_runtime_env(tracer_so)
    except TracerRuntimeError as exc:
        print(f"tracer_so_deps_unresolved: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    env["CUDA_INJECTION64_PATH"] = str(tracer_so)
    # R51: USER_DEFINED_FOLDERS=1 is the gate that unlocks the tracer's
    # TRACES_FOLDER lookup; without it the tracer ignores TRACES_FOLDER
    # and writes EVERY artifact (`dynamic_trace.pb`, `stats.csv`, the
    # per-kernel `.trace` files, `extra_info/`, `threadblocks/`) to
    # `<cwd>/traces/`. The post-capture admission below — and every
    # downstream feature-sidecar producer (sector_stream, hrd_profile,
    # divergence_proxies, sharing_locality, order_projections,
    # perturbation_run_ledger, convergence_confidence) — looks for
    # those artifacts under `traces_folder = artifact_dir /
    # f"trace_{shape}"`, so missing this gate is what produced R50's
    # raw-only trees with no per-run sidecars.
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(traces_folder)
    t0 = _time.perf_counter()
    proc = _sp.run([str(executable), *args], capture_output=True, text=True, env=env)
    t1 = _time.perf_counter()
    if proc.returncode != 0:
        print(f"build_failed: {executable} under tracer rc={proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    # R13: prefer the workload's stdout `wall_time_ms` over host
    # `perf_counter` for the NVBit-instrumented run's wall time, so
    # the C-AC-H slowdown evidence comes from the device-side
    # measurement rather than host scheduling jitter. Both the
    # post-capture `trace_volume_exceeded.json` `actual_wall_time_s`
    # and the per-shape run summary's wall-time fields use the
    # selected value.
    nvbit_reported = _stdout_wall_time_s(proc.stdout)
    actual_wall_time_s = (
        nvbit_reported if nvbit_reported is not None
        else t1 - t0
    )
    trace_pb = traces_folder / "dynamic_trace.pb"
    if not trace_pb.exists():
        # NVBit dispatch appeared to succeed but produced no trace artifact.
        # Treat this as a hard failure rather than silently recording success.
        print(f"trace_missing: tracer ran but {trace_pb} was not written (CUDA_INJECTION64_PATH=LD path wrong, or binary does not launch any CUDA kernels)", file=sys.stderr)
        return EXIT_BUILD_FAILED

    # Post-capture artifact-integrity enforcement (R24 / R23-review
    # blocking fix). The pre-admission check uses the committed
    # `predicted_bytes`, but the actual trace emitted by NVBit can still
    # exceed the row budget for kernels where the static predictor under-
    # estimates (dynamic control flow, cache-dependent access patterns, etc.).
    # AC7 wants the orchestrator to refuse such traces — not record them —
    # so write the sidecar, delete the oversize .pb, and return
    # EXIT_SAMPLING_REQUIRED. This closes the `BudgetViolationUntracked`
    # class Codex reproduced on the R23 dispatch.
    trace_bytes = trace_pb.stat().st_size
    budget_bytes = row["trace_budget_mib"] * 1024 * 1024
    if trace_bytes > budget_bytes:
        import hashlib
        sidecar = artifact_dir / "trace_volume_exceeded.json"
        tb_hash = hashlib.sha256((SUITE_ROOT / "trace_budgets.toml").read_bytes()).hexdigest()
        sidecar.write_text(json.dumps({
            "schema_version": 1,
            "slug": slug,
            "shape_point": shape_point["name"],
            "sampling_policy": "off",
            "max_trace_bytes": budget_bytes,
            "actual_trace_bytes": trace_bytes,
            "max_wall_time_s": 3600,
            "actual_wall_time_s": actual_wall_time_s,
            "exceeded_trace_bytes": True,
            "exceeded_wall_time": False,
            "trace_budgets_sha256": tb_hash,
        }, indent=2))
        # Remove the oversize trace so the `results/` tree never contains an
        # untracked BudgetViolation artifact. Use missing_ok=True so a racy
        # concurrent removal does not re-fail the handler.
        trace_pb.unlink(missing_ok=True)
        print(f"trace_shape_overbudget: post-capture trace_bytes={trace_bytes} > budget_bytes={budget_bytes}", file=sys.stderr)
        print(f"trace_shape_overbudget: sidecar at {sidecar}", file=sys.stderr)
        return EXIT_SAMPLING_REQUIRED

    try:
        trace_admission = require_trace_admission(traces_folder)
    except TraceAdmissionError as exc:
        print(
            f"trace_capture_not_admitted: {exc.typed_reason}: {exc.detail}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED

    # C-AC-H slowdown / drop-rate sidecar emission (Loop 2 R13).
    # Two pinned thresholds (`NVBIT_SLOWDOWN_BUDGET_MAX_RATIO`,
    # currently 1200.0 post-R4 via the sm120_l2_observability loop's
    # user-sanctioned widening: 400.0 -> 1200.0 (historical chain
    # 200.0 -> 400.0 at R25 -> 1200.0 at R4); and
    # `NVBIT_DROP_RATE_BUDGET_MAX = 0.05`) are imported from
    # `tools/cloud/trace_budgets.py`; the producer honors whichever
    # value the runtime constant holds rather than pinning a literal
    # here. The slowdown ratio uses the baseline run timed at the
    # top of this function; the drop-rate numerator + denominator
    # come from `stats.csv`'s `total_insts` and
    # `total_reported_insts` summed across all kernel launches in
    # the trace. The `actual_wall_time_s` selected above (stdout
    # `wall_time_ms` preferred over host `perf_counter`) drives the
    # slowdown-ratio numerator.
    actual_slowdown_ratio: float | None = None
    if baseline_wall_time_s is not None and baseline_wall_time_s > 0:
        actual_slowdown_ratio = actual_wall_time_s / float(baseline_wall_time_s)
    expected_record_count = trace_admission.expected_record_count
    admitted_record_count = trace_admission.admitted_record_count
    actual_drop_rate: float | None = None
    if expected_record_count is not None and expected_record_count > 0:
        admitted_for_rate = admitted_record_count or 0
        actual_drop_rate = 1.0 - admitted_for_rate / expected_record_count
    sidecar = traces_folder / "nvbit_slowdown_budget_exceeded.json"
    (artifact_dir / "nvbit_slowdown_budget_exceeded.json").unlink(
        missing_ok=True
    )
    if (
        actual_slowdown_ratio is not None
        and baseline_wall_time_s is not None
        and float(baseline_wall_time_s)
        >= _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S
        and actual_slowdown_ratio > _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO
    ):
        import hashlib
        tb_hash = hashlib.sha256(
            (SUITE_ROOT / "trace_budgets.toml").read_bytes()
        ).hexdigest()
        sidecar.write_text(json.dumps({
            "schema_version": 1,
            "slug": slug,
            "shape_point": shape_point["name"],
            "max_slowdown_ratio": _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO,
            "actual_slowdown_ratio": actual_slowdown_ratio,
            "baseline_wall_time_s": float(baseline_wall_time_s),
            "actual_wall_time_s": actual_wall_time_s,
            "trace_budgets_sha256": tb_hash,
        }, indent=2))
        print(
            f"nvbit_slowdown_budget_exceeded: actual_ratio="
            f"{actual_slowdown_ratio:.3f} > max_ratio="
            f"{_trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO}",
            file=sys.stderr,
        )
        print(
            f"nvbit_slowdown_budget_exceeded: sidecar at {sidecar}",
            file=sys.stderr,
        )
    else:
        # Reruns into an existing artifact_dir can leave a stale sidecar
        # from a prior run when the producer's emission predicate
        # suppresses (baseline below the min-baseline-time floor, or
        # ratio at/below MAX). The verifier's binding check would then
        # mismatch the stale sidecar's cached fields against the new
        # run summary; remove the stale sidecar to keep on-disk state
        # in lockstep with the producer's intent for THIS run.
        sidecar.unlink(missing_ok=True)
    sidecar = traces_folder / "nvbit_drop_rate_exceeded.json"
    (artifact_dir / "nvbit_drop_rate_exceeded.json").unlink(missing_ok=True)
    if (
        actual_drop_rate is not None
        and expected_record_count is not None
        and actual_drop_rate > _trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX
    ):
        import hashlib
        tb_hash = hashlib.sha256(
            (SUITE_ROOT / "trace_budgets.toml").read_bytes()
        ).hexdigest()
        # The R11 producer surface populates only the residual
        # source counter (`expected_record_mismatch_dropped`) with
        # `expected - admitted`; the other three named source
        # counters stay at 0 until the per-source NVBit producer
        # surface lands. The schema requires all four fields so a
        # future per-source landing fails the C-AC-C parity scanner
        # without simultaneously updating the schema + this writer.
        admitted_int = int(admitted_record_count or 0)
        residual = int(expected_record_count) - admitted_int
        sidecar.write_text(json.dumps({
            "schema_version": 1,
            "slug": slug,
            "shape_point": shape_point["name"],
            "max_drop_rate": _trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX,
            "actual_drop_rate": actual_drop_rate,
            "expected_record_count": int(expected_record_count),
            "admitted_record_count": admitted_int,
            "channel_overflow_dropped": 0,
            "ring_buffer_loss_dropped": 0,
            "decoder_gap_dropped": 0,
            "expected_record_mismatch_dropped": residual,
            "trace_budgets_sha256": tb_hash,
        }, indent=2))
        print(
            f"nvbit_drop_rate_exceeded: actual_rate="
            f"{actual_drop_rate:.3f} > max_rate="
            f"{_trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX}",
            file=sys.stderr,
        )
        print(
            f"nvbit_drop_rate_exceeded: sidecar at {sidecar}",
            file=sys.stderr,
        )
    else:
        # Reruns into an existing artifact_dir can leave a stale
        # drop-rate sidecar from a prior failing run. The verifier
        # at `_verify_canonical_trace_pb_admission` treats the mere
        # presence of `nvbit_drop_rate_exceeded.json` as a
        # budget-exceeded carve-out and skips the feature-sidecar
        # checks; remove the stale sidecar to keep on-disk state
        # in lockstep with the producer's intent for THIS run.
        # Parallels the unlink in the slowdown branch above.
        sidecar.unlink(missing_ok=True)

    # Serialize the tracer .so path as repo-relative when possible (readable
    # in committed manifests); fall back to absolute for paths outside the
    # repo tree (monkeypatched in tests, or externally installed tracers).
    try:
        tracer_so_serialized = str(NVBIT_TRACER_SO.relative_to(SUITE_ROOT.parent.parent))
    except ValueError:
        tracer_so_serialized = str(NVBIT_TRACER_SO)

    # Per-launch NVBit trace summary emission (consumed by
    # workloads/simbench_bw/merge_evidence.py join-cupti-nvbit). The
    # adapter enumerates kernels in dynamic_trace.pb and emits one
    # summary per launch; the offline two-run join requires one
    # summary per CUPTI sidecar. Tracked-binding corruption, a missing runtime
    # prerequisite, and every other adapter exception are hard producer
    # failures so non-joinable trace evidence is never reported as success.
    nvbit_emit_result = _emit_nvbit_trace_summaries_for_trace_artifact(
        traces_folder=traces_folder,
        trace_pb=trace_pb,
        slug=slug,
        shape_point=shape_point["name"],
        executable=executable,
        tracer_so_path=NVBIT_TRACER_SO,
        artifact_dir=artifact_dir,
    )
    nvbit_summary_status = nvbit_emit_result["status"]
    nvbit_summary_paths = nvbit_emit_result["paths"]
    nvbit_summary_failure_reason = nvbit_emit_result["failure_reason"]
    nvbit_emit_failed_hard = nvbit_emit_result["failed_hard"]
    if nvbit_emit_failed_hard:
        print(
            f"trace_nvbit_summary_emit_failed: status={nvbit_summary_status} "
            f"reason={nvbit_summary_failure_reason}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED

    artifact_path = artifact_dir / f"trace_{shape_point['name']}.json"
    artifact_record: dict[str, Any] = {
        "schema_version": 1,
        "slug": slug,
        "shape_set": "trace_shapes",
        "shape_point": shape_point["name"],
        "trace_pb_path": str(trace_pb.relative_to(artifact_dir)),
        "trace_bytes": trace_bytes,
        "wall_time_s": actual_wall_time_s,
        "tracer_so_path": tracer_so_serialized,
        "nvbit_trace_summary_emit_status": nvbit_summary_status,
        "nvbit_trace_summary_paths": nvbit_summary_paths,
    }
    # C-AC-H evidence the post-collection verifier reads back to
    # recompute slowdown / drop-rate against the pinned budgets
    # (Loop 2 R12). When a raw-evidence value is None the verifier
    # treats it as the typed-unavailable arm and skips enforcement
    # on that value; cached ratio fields (`actual_slowdown_ratio`,
    # `actual_drop_rate`) are persisted as a consistency-check hint
    # but are not authoritative — the verifier recomputes from raw
    # evidence and rejects any disagreement. The four drop-source
    # counter fields are persisted to mirror the
    # `nvbit_drop_rate_exceeded.json` schema; today only the
    # residual (`expected_record_mismatch_dropped`) carries the
    # gap, the other three sources stay 0 until the upstream NVBit
    # producer surfaces them.
    artifact_record["baseline_wall_time_s"] = baseline_wall_time_s
    artifact_record["actual_wall_time_s"] = actual_wall_time_s
    artifact_record["actual_slowdown_ratio"] = actual_slowdown_ratio
    artifact_record["expected_record_count"] = expected_record_count
    artifact_record["admitted_record_count"] = admitted_record_count
    artifact_record["actual_drop_rate"] = actual_drop_rate
    if expected_record_count is not None and admitted_record_count is not None:
        residual = int(expected_record_count) - int(admitted_record_count)
    else:
        residual = None
    artifact_record["channel_overflow_dropped"] = 0
    artifact_record["ring_buffer_loss_dropped"] = 0
    artifact_record["decoder_gap_dropped"] = 0
    artifact_record["expected_record_mismatch_dropped"] = residual

    # task11 / the HRD-profiler clause: emit the HRD bundle and
    # persist Section 10 evidence fields. The bundle producer
    # reads the sector-stream JSONL the consumer wrote alongside
    # the trace; if the consumer hasn't run yet (or the proto
    # walk failed), the HRD bundle is skipped with a typed-
    # unavailable annotation in the run summary.
    _emit_hrd_bundle_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        executable=executable,
        props=props,
    )

    # R53 / Codex's R52 review mainline gap #2: the canonical
    # capture path was missing the `divergence_proxies/` and
    # `sharing_locality/` per-shape feature sidecars sidecars. Wired here so a
    # canonical `run_suite.py --mode trace` capture emits the
    # full sidecar set the verifier requires.
    _emit_divergence_proxies_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        executable=executable,
        props=props,
    )
    _emit_sharing_locality_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        executable=executable,
        props=props,
    )

    # task13 / the order-projections clause: emit the six-
    # projection bundle and persist its evidence into the run
    # summary. The producer reuses the same threadblock proto
    # walk the HRD profiler already validated; no additional
    # sidecar is required. Skip path: a producer failure leaves
    # all order_projections_* fields None and records a typed
    # `order_projections_skip_reason`.
    _emit_order_projections_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        executable=executable,
        props=props,
    )

    # task15 / the perturbation-harness clause: build the 9
    # loop-scope perturbation records (cache_modifier_ca,
    # cache_modifier_cg, working_set_sweep MiB ladder), confirm SASS,
    # check correctness equivalence, and persist the bundle + run-
    # summary fields. The producer is best-effort; build failures
    # leave a typed `perturbation_run_ledger_skip_reason` so the
    # verifier skips enforcement on partial sweeps.
    _emit_perturbation_run_ledger_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        executable=executable,
        props=props,
    )

    # task17 / the adaptive-sampler clause: build the
    # trace_convergence_confidence bundle from the upstream HRD bundle
    # produced earlier in this same handle_trace flow. The producer is
    # best-effort; missing HRD evidence leaves a typed
    # `trace_convergence_skip_reason` so the verifier skips
    # enforcement on partial sweeps.
    _emit_trace_convergence_confidence_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        props=props,
    )

    if nvbit_summary_failure_reason is not None:
        artifact_record["nvbit_trace_summary_failure_reason"] = (
            nvbit_summary_failure_reason
        )
    artifact_path.write_text(json.dumps(artifact_record, indent=2))

    manifest = build_measurement_manifest(
        slug=slug, mode="trace",
        warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
        props=props, binary_path=executable,
        compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
        shape_point=shape_point["name"],
    )
    _emit_manifest(manifest, artifact_dir)
    print(f"run_suite: trace for {slug}: {trace_bytes} bytes at {trace_pb} (artifact at {artifact_path})")

    # Non-root measurement NVBit phase-injection contract: when --phase-inject-ab
    # is set, run a paired no-injection / with-injection trace pair to
    # compute the A/B Heisenberg-overhead record, then decode the
    # per-launch phase_records_<repeat>_<launch>.bin files emitted by
    # the consolidated tracer extension and emit
    # phase_cycle_ledger_<repeat>_<launch>.json sidecars.
    if phase_inject_ab:
        rc_ab = _run_phase_inject_ab_for_handle_trace(
            slug=slug,
            shape_point_name=shape_point["name"],
            executable=executable,
            args=args,
            tracer_so_path=tracer_so,
            traces_folder=traces_folder,
            artifact_dir=artifact_dir,
            ab_repeat_count=ab_repeat_count,
            row=row,
        )
        if rc_ab != EXIT_OK:
            return rc_ab

    return EXIT_OK


def _run_phase_inject_ab_for_handle_trace(
    *,
    slug: str,
    shape_point_name: str,
    executable: Path,
    args: list[str],
    tracer_so_path: Path,
    traces_folder: Path,
    artifact_dir: Path,
    ab_repeat_count: int,
    row: dict[str, Any],
) -> int:
    """Run the A/B overhead pair + emit per-launch phase ledgers.
    Helper extracted from handle_trace to keep the trace handler's
    main path readable. Returns EXIT_OK on success or
    EXIT_BUILD_FAILED on producer failure."""
    import phase_inject_orchestrator as _pinj  # type: ignore[import-not-found]
    import phase_cycle_ledger_consumer as _pcons  # type: ignore[import-not-found]
    try:
        base_env = tracer_runtime_env(tracer_so_path)
    except TracerRuntimeError as exc:
        print(f"tracer_so_deps_unresolved: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILED
    # Isolate the A/B subprocesses' NVBit outputs from the canonical
    # trace folder. handle_trace has already emitted dynamic_trace.pb,
    # stats.csv, threadblocks/, extra_info/, and the per-shape feature
    # sidecars under traces_folder; the run summary and verify_artifacts
    # admission describe THOSE bytes. With USER_DEFINED_FOLDERS=1, each
    # A/B subprocess writes a fresh dynamic_trace.pb / stats.csv /
    # threadblocks/ / extra_info/ to its TRACES_FOLDER; routing those
    # to the canonical traces_folder would overwrite the canonical
    # evidence. Drop A/B output into a phase_inject_ab/ sub-directory
    # so producer phase records co-locate with the canonical capture
    # but never clobber it.
    phase_traces_folder = traces_folder / "phase_inject_ab"
    phase_traces_folder.mkdir(parents=True, exist_ok=True)
    inputs = _pinj.AbOverheadInputs(
        executable=executable,
        args=list(args),
        tracer_so_path=tracer_so_path,
        traces_folder=phase_traces_folder,
        repeat_count=ab_repeat_count,
        base_env=base_env,
    )
    admission = _pcons.load_admission_registry()
    admission_threshold_pct = _pcons.admission_threshold_for_slug(
        admission, slug
    )
    try:
        ab_overhead_record = _pinj.run_ab_overhead_pair(
            inputs, admission_threshold_pct=admission_threshold_pct
        )
    except _pinj.PhaseInjectAbRunFailed as e:
        print(
            f"phase_inject_ab_run_failed: side={e.side!r} rc={e.returncode}: "
            f"{e.stderr.strip()[:200]}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except _pinj.PhaseInjectAbRunTimeout as e:
        print(
            f"phase_inject_ab_run_timeout: side={e.side!r} timeout_s={e.timeout_s}: "
            f"{e.stderr.strip()[:200]}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except _pinj.PhaseInjectRuntimeUnavailable as e:
        print(f"phase_inject_runtime_unavailable: {e.detail}", file=sys.stderr)
        return EXIT_BUILD_FAILED

    ledger_dir = artifact_dir / f"phase_cycle_ledgers_{shape_point_name}"
    binary_sha = _binary_sha256(executable)
    # Provenance: derive a stable kernel_signature_sha256 placeholder
    # from the binary's sha + shape_point. Real per-launch kernel
    # signatures land when the kernel-feature-vector emitter for the
    # six-axis kernel-feature-vector contract is wired in (sequenced
    # alongside that producer surface).
    import hashlib as _hashlib
    kernel_sig = _hashlib.sha256(
        f"{binary_sha}:{shape_point_name}".encode("utf-8")
    ).hexdigest()
    nvbit_tracer_build_hash = _hashlib.sha256(
        tracer_so_path.read_bytes()
    ).hexdigest()
    import phase_cycle_ledger_aggregator as _agg  # type: ignore[import-not-found]
    # Catch typed aggregator/consumer diagnostics so a real producer-side
    # phase-pairing anomaly (e.g. a kernel whose SASS has multiple EXIT
    # terminators causing duplicate function_entry close records) surfaces
    # to stderr as a stable diagnostic token + EXIT_BUILD_FAILED, mirroring
    # the PhaseInjectAbRunFailed / PhaseInjectAbRunTimeout pattern above.
    # The token + payload are recoverable from the captured stderr; the
    # caller can decide whether to treat the typed-failure as evidence
    # of a producer regression or as an expected per-arch outcome.
    _typed_ledger_excs = (
        _agg.PhaseRecordPairingFailed,
        _agg.PhaseRecordCycleDeltaNegative,
        _agg.PhaseLedgerSchemaInvalid,
        _agg.PhaseRecordDuplicateOpen,
        _agg.PhaseRecordDuplicateClose,
        _agg.PhaseRecordLaunchIndexMismatch,
        _agg.PhaseDefinitionInconsistent,
        _pinj.PhaseLedgerEvidenceMissing,
    )
    try:
        summary = _pinj.emit_per_launch_phase_ledgers(
            traces_folder=phase_traces_folder,
            ledger_output_dir=ledger_dir,
            slug=slug,
            shape_point=shape_point_name,
            kernel_signature_sha256=kernel_sig,
            binary_sha256=binary_sha,
            nvbit_tracer_build_hash=nvbit_tracer_build_hash,
            claim_scope=row.get("claim_scope", "measured_local"),
            ab_overhead_record=ab_overhead_record,
            dual_run_manifest_ref=str(
                artifact_dir / "dual_run_manifest.json"
            ),
            admission=admission,
            admission_threshold_pct=admission_threshold_pct,
        )
    except _typed_ledger_excs as e:
        print(
            f"phase_inject_ab_ledger_typed_failure: token={e.TOKEN} "
            f"slug={slug} shape_point={shape_point_name}: "
            f"{str(e).strip()[:400]}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    print(
        "run_suite: phase_inject_ab "
        f"slug={slug} shape_point={shape_point_name} "
        f"overhead_pct={ab_overhead_record['overhead_pct']:.3f} "
        f"admitted={ab_overhead_record['admitted']} "
        f"emitted_admitted={summary['n_admitted']} "
        f"emitted_rejected={summary['n_rejected']}"
    )
    return EXIT_OK


def handle_emit_dual_run_manifest(
    *,
    cupti_run_dir: str | None,
    nvbit_run_dir: str | None,
    output_path: str | None,
    drift_tolerance_ns: int | None,
) -> int:
    """Non-root measurement dual-run manifest emission entry point: pair a
    CUPTI run directory with an NVBit run directory and emit a
    `dual_run_manifest.json` via
    `dual_run_manifest_emitter.emit_dual_run_manifest`. The join
    layer's drift-exceeded / preflight-disagreement / NVML-
    alignment-missing diagnostics surface as typed exceptions
    that are caught here and reported via stderr + a non-zero
    exit code."""
    if cupti_run_dir is None or nvbit_run_dir is None or output_path is None:
        print(
            "run_suite: emit-dual-run-manifest requires "
            "--cupti-run-dir, --nvbit-run-dir, and "
            "--dual-run-manifest-output",
            file=sys.stderr,
        )
        return EXIT_USAGE
    import dual_run_manifest_emitter as _drm  # type: ignore[import-not-found]
    import phase_cycle_ledger_consumer as _pcons  # type: ignore[import-not-found]
    if drift_tolerance_ns is None:
        admission = _pcons.load_admission_registry()
        drift_tolerance_ns = admission.drift_tolerance_ns_default
    try:
        _drm.emit_dual_run_manifest(
            cupti_run_dir=Path(cupti_run_dir),
            nvbit_run_dir=Path(nvbit_run_dir),
            output_path=Path(output_path),
            drift_tolerance_ns=int(drift_tolerance_ns),
        )
    except (
        _drm.DualRunPreflightDisagreement,
        _drm.DualRunManifestUnmatchedLaunches,
        _drm.DualRunManifestEmptyInputDir,
        _drm.DualRunNvbitNvmlMissing,
        _drm.DualRunManifestSchemaInvalid,
        _drm.DualRunInputSchemaInvalid,
        _drm.DualRunInputDuplicateLaunchKey,
        _drm.DualRunNvmlAlignmentMissing,
    ) as e:
        print(
            f"dual_run_manifest_emit_failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    print(
        f"run_suite: emit_dual_run_manifest -> {output_path}",
        file=sys.stderr,
    )
    return EXIT_OK




# --- selective rerun -----------------------------------------------------------


def _git_diff_files(base: str) -> list[str]:
    """Return the list of repo-relative paths changed between `base` and the
    current worktree (HEAD). Silently returns an empty list on any git
    failure so the caller can fall back gracefully."""
    import subprocess as _sp
    try:
        out = _sp.run(
            ["git", "diff", "--name-only", base],
            cwd=SUITE_ROOT.parent.parent,
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


SHARED_ASSET_PATTERNS = (
    "workloads/simbench_bw/common.mk",
    "workloads/simbench_bw/run_suite.py",
    "workloads/simbench_bw/validate.py",
    "workloads/simbench_bw/verify_coverage.py",
    "workloads/simbench_bw/trace_size_estimator.py",
    "workloads/simbench_bw/update_goldens.py",
)


def _select_regression_targets(roster: dict[str, dict], base: str) -> list[dict]:
    """Pick the set of rows affected by git diff against `base`:

    - A change under `l{1,2,3,4}/<slug>/` or `shapes/<slug>.toml` adds <slug>
      to the target set.
    - A change under `references/<family>.py` adds every row with that
      `reference_family`.
    - A change to any file in SHARED_ASSET_PATTERNS adds every benchmark row.
    - A change anywhere else under `workloads/simbench_bw/` does nothing (it
      is suite-infrastructure, not kernel-level).
    """
    changed = _git_diff_files(base)
    if not changed:
        # Empty diff means no kernel-relevant changes; the CLI caller short-
        # circuits with "no kernels affected". Do NOT fall through to full
        # roster (the R5 bug Codex flagged).
        return []

    affected: set[str] = set()
    family_to_slugs: dict[str, list[str]] = {}
    for slug, row in roster.items():
        family_to_slugs.setdefault(row["reference_family"], []).append(slug)

    shared_set = set(SHARED_ASSET_PATTERNS)
    for path in changed:
        # Exact repo-relative match only. A deep file that happens to share a
        # basename with a shared asset (e.g. `scratch/notes/run_suite.py`)
        # does not broadcast to every row.
        if path in shared_set:
            return list(roster.values())
        if path.startswith("workloads/simbench_bw/l") and "/" in path[len("workloads/simbench_bw/"):]:
            # e.g. workloads/simbench_bw/l2/<slug>/<slug>.cu
            parts = path.split("/")
            if len(parts) >= 4:
                slug = parts[3]
                if slug in roster:
                    affected.add(slug)
        elif path.startswith("workloads/simbench_bw/shapes/"):
            base_name = path.split("/")[-1]
            if base_name.endswith(".toml"):
                slug = base_name[:-len(".toml")]
                if slug in roster:
                    affected.add(slug)
        elif path.startswith("workloads/simbench_bw/references/"):
            base_name = path.split("/")[-1]
            if base_name.endswith(".py"):
                family = base_name[:-len(".py")]
                for slug in family_to_slugs.get(family, []):
                    affected.add(slug)

    return [roster[slug] for slug in sorted(affected)]


# regression-mode:real-dispatch
def handle_regression(row: dict[str, Any], props: DeviceProps, dry_run: bool,
                      out_dir: Path | None, strict_golden_origin: bool = False,
                      allow_missing_golden: bool = False) -> int:
    _ = out_dir
    slug = row["slug"]
    golden_dir = SUITE_ROOT / "golden" / slug
    if not golden_dir.exists():
        if allow_missing_golden:
            print(f"run_suite: regression for {slug}: golden_absent (--allow-missing-golden in effect)")
            return EXIT_OK
        print(f"golden_absent: {golden_dir} has no committed golden (pass --allow-missing-golden to override)", file=sys.stderr)
        return EXIT_UNSUPPORTED

    # Non-dry-run mode REQUIRES the golden matching the current platform + arch
    # (computed from the startup identity probe). Dry-run mode falls back to any
    # committed golden since there is no "current" platform to align to.
    if dry_run:
        golden_files = list(golden_dir.rglob("results.json"))
        if not golden_files:
            if allow_missing_golden:
                print(f"run_suite: regression for {slug}: golden_absent (no results.json; --allow-missing-golden in effect)")
                return EXIT_OK
            print(f"golden_absent: {golden_dir} has no results.json (pass --allow-missing-golden to override)", file=sys.stderr)
            return EXIT_UNSUPPORTED
        golden_path = golden_files[0]
    else:
        gpu_sku = _infer_gpu_sku(props)
        arch_tag = _resolve_build_arch_for_row(row, props)
        expected_path = golden_dir / gpu_sku / arch_tag / "results.json"
        if not expected_path.exists():
            if allow_missing_golden:
                print(f"run_suite: regression for {slug}: golden_platform_mismatch (--allow-missing-golden in effect; expected {expected_path.relative_to(SUITE_ROOT)})")
                return EXIT_OK
            print(f"golden_platform_mismatch: {slug} has no golden at {expected_path.relative_to(SUITE_ROOT)} for current platform={gpu_sku} arch={arch_tag}", file=sys.stderr)
            return EXIT_UNSUPPORTED
        golden_path = expected_path

    with open(golden_path) as f:
        golden = json.load(f)

    if strict_golden_origin and "_golden_update_rationale" not in golden:
        print(f"golden_bypassed: {golden_path} lacks _golden_update_rationale marker", file=sys.stderr)
        return EXIT_GOLDEN_MISMATCH

    family = row["reference_family"]
    mod = _load_reference_family(family)
    if mod is None:
        print(f"reference_missing: references/{family}.py", file=sys.stderr)
        return EXIT_USAGE
    shape_point_name = golden.get("shape_point")
    shape_path = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not shape_path.exists():
        print(f"shapes_missing: {shape_path}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    with open(shape_path, "rb") as f:
        shape = tomllib.load(f)
    shape_point = next((p for p in shape.get("correctness_shapes", []) if p.get("name") == shape_point_name), None)
    if shape_point is None:
        print(f"shape_point_missing: {shape_point_name!r} not in shapes/{slug}.toml correctness_shapes", file=sys.stderr)
        return EXIT_UNSUPPORTED
    seed = int(row.get("seed_policy", {}).get("seed", 42))
    tol = _load_tolerance_triplet(row["primary_precision"], _infer_reduction_kind(row), family)
    if tol is None:
        print(f"tolerance_unmapped: no entry for ({row['primary_precision']}, {_infer_reduction_kind(row)}, {family})", file=sys.stderr)
        return EXIT_UNSUPPORTED
    golden_output = golden.get("output")
    if golden_output is None:
        print(f"golden_mismatch: {golden_path} missing 'output' field", file=sys.stderr)
        return EXIT_GOLDEN_MISMATCH

    if not dry_run:
        # Non-dry-run demands a real kernel executable. Compile-only kernels
        # MUST NOT pass through the Python-reference fallback; that was the
        # false-green Codex flagged.
        executable = _find_kernel_executable(slug)
        if executable is None:
            print(f"executable_missing: {slug} has no <slug>_run executable for a native regression gate (build the kernel first or pass --dry-run-cpu)", file=sys.stderr)
            return EXIT_UNSUPPORTED
        args_cli = _kernel_args_from_shape(shape_point, row)
        import subprocess as _sp
        proc = _sp.run([str(executable), *args_cli], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"build_failed: {executable} exited rc={proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
            return EXIT_BUILD_FAILED
        try:
            kernel_result = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            print(f"golden_mismatch: {slug} driver output is not valid JSON ({e})", file=sys.stderr)
            return EXIT_GOLDEN_MISMATCH
        kernel_output = kernel_result.get("output")
        if kernel_output is None:
            print(f"golden_mismatch: {slug} driver JSON missing 'output' field", file=sys.stderr)
            return EXIT_GOLDEN_MISMATCH
        if not mod.compare(kernel_output, golden_output, tol):
            print(f"golden_mismatch: slug={slug} native kernel output differs from committed golden beyond tolerance", file=sys.stderr)
            return EXIT_GOLDEN_MISMATCH
        print(f"run_suite: regression for {slug}: native kernel output matches committed golden under tolerance")
        return EXIT_OK

    # Dry-run-only fallback: rebuild the Python reference and compare against
    # the golden. This is intentionally a reference-only smoke test and NEVER
    # substitutes for a native regression gate.
    try:
        reference_output = mod.build_reference(slug, shape_point, seed)
    except NotImplementedError as e:
        print(f"reference_unimplemented: {e}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    if not mod.compare(reference_output, golden_output, tol):
        print(f"golden_mismatch: slug={slug} reference output differs from committed golden beyond tolerance", file=sys.stderr)
        return EXIT_GOLDEN_MISMATCH
    print(f"run_suite: dry-run regression for {slug}: reference matches committed golden under tolerance")
    return EXIT_OK


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SimBench-BW orchestrator",
        epilog=(
            "Exit codes: 0 ok, 2 usage, 3 sampling_required, 4 unsupported, "
            "5 correctness_failed, 6 build_failed, 7 golden_mismatch, "
            "8 unknown_target_device, 9 dry_run_not_supported_for_mode."
        ),
    )
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--kernel", metavar="SLUG", help="Target a single roster row by slug")
    p.add_argument("--wave", metavar="N", help="Run only rows with wave=N")
    p.add_argument("--dry-run-cpu", action="store_true",
                   help="Skip GPU, validate dispatch + reference presence; cpu-ok modes only")
    p.add_argument("--allow-any-device", action="store_true",
                   help="Skip the known-CC gate in startup identity check")
    p.add_argument("--out", metavar="DIR", default=str(SUITE_ROOT / "results"),
                   help="Output directory for measurement artifacts (default: "
                        "workloads/simbench_bw/results/ so validate.py --check-l4-scoring "
                        "discovery works from the default CLI flow per R31-review)")
    p.add_argument("--full", action="store_true",
                   help="Regression mode: ignore PR diff, rerun every kernel")
    p.add_argument("--strict-golden-origin", action="store_true",
                   help="Regression mode: refuse goldens that lack _golden_update_rationale (detects direct text edits)")
    p.add_argument("--allow-missing-golden", action="store_true",
                   help="Regression mode: treat missing golden as OK (smoke-test convenience)")
    p.add_argument("--component-strict", action="store_true",
                   help="For L4 pipeline rows under --mode correctness, emit the "
                        "canonical component-level checkpoint matching "
                        "schemas/l4_checkpoints/<slug>.schema.json alongside the "
                        "candidate + measurement_manifest, re-validate it in-process "
                        "via validate.validate_l4_checkpoint, and emit a minimal "
                        "suite_results.json with component_health so "
                        "validate.py --check-l4-scoring is non-vacuous. No-op for "
                        "non-L4 rows.")
    p.add_argument("--base-branch", metavar="REF", default=None,
                   help="Regression mode: base git ref for --full-less selective rerun (default: HEAD~1)")
    p.add_argument("--shape-point", metavar="NAME", default=None,
                   help="Run a single named shape from the row's mode-specific shape list "
                        "(timing_shapes for --mode timing; trace_shapes for --mode trace). "
                        "When omitted, both handlers default to index 0 of their respective "
                        "shape list.")
    p.add_argument("--phase-inject-ab", action="store_true",
                   help="Non-root measurement NVBit phase-injection contract: when --mode trace is "
                        "set, run a paired no-injection / with-injection trace pair and "
                        "compute the A/B Heisenberg-overhead record. Decode the per-launch "
                        "phase_records_<repeat>_<launch>.bin files emitted by the consolidated "
                        "tracer extension and emit phase_cycle_ledger_<repeat>_<launch>.json "
                        "sidecars admitted via the consumer's `admit_phase_cycle_ledger`.")
    p.add_argument("--ab-repeat-count", metavar="N", type=int, default=3,
                   help="Number of repeats per A/B side when --phase-inject-ab is set "
                        "(default: 3 -- the median wall-clock per side is used).")
    p.add_argument("--cupti-run-dir", metavar="DIR", default=None,
                   help="Non-root measurement dual-run manifest emission: directory containing "
                        "the CUPTI run's per-launch sidecars + NVML covariate.")
    p.add_argument("--nvbit-run-dir", metavar="DIR", default=None,
                   help="Non-root measurement dual-run manifest emission: directory containing "
                        "the NVBit run's per-launch phase_cycle_ledger sidecars + NVML covariate.")
    p.add_argument("--dual-run-manifest-output", metavar="PATH", default=None,
                   help="Non-root measurement dual-run manifest emission target path for the "
                        "emitted dual_run_manifest.json. Required when --mode "
                        "emit-dual-run-manifest is set.")
    p.add_argument("--drift-tolerance-ns", metavar="N", type=int, default=None,
                   help="Override the per-launch drift tolerance for dual-run manifest "
                        "emission. Default loads from `nvbit_phase_injection.toml`'s "
                        "`defaults.drift_tolerance_ns`.")
    args = p.parse_args(argv)

    if args.dry_run_cpu and args.mode in GPU_ONLY_MODES:
        print(f"run_suite: --dry-run-cpu cannot be combined with --mode {args.mode}", file=sys.stderr)
        return EXIT_DRY_RUN_NOT_SUPPORTED

    roster = load_roster()

    # Startup identity check (skipped in dry-run mode)
    if args.dry_run_cpu:
        props = synthetic_dry_run_props()
    else:
        props = startup_identity_check(args.allow_any_device)
        if props is None:
            props = synthetic_dry_run_props()

    # Select rows
    if args.kernel:
        if args.kernel not in roster:
            print(f"unknown kernel slug: {args.kernel!r}", file=sys.stderr)
            return EXIT_USAGE
        targets = [roster[args.kernel]]
    elif args.wave:
        targets = [r for r in roster.values() if r["wave"] == args.wave]
        if not targets:
            print(f"no rows with wave={args.wave!r}", file=sys.stderr)
            return EXIT_USAGE
    elif args.mode == "regression" and not args.full:
        # AC9 selective rerun. Empty target set means nothing to run — exit OK.
        base = args.base_branch or "HEAD~1"
        targets = _select_regression_targets(roster, base)
        if not targets:
            print(f"run_suite: regression selective-rerun: no kernels affected by diff vs {base}")
            return EXIT_OK
    else:
        targets = list(roster.values())

    global args_out_dir_root
    args_out_dir_root = args.out
    worst: int = EXIT_OK
    for row in targets:
        out_dir = Path(args.out) / row["slug"]
        if args.mode == "correctness":
            rc = handle_correctness(row, props, args.dry_run_cpu, out_dir,
                                    component_strict=args.component_strict)
        elif args.mode == "timing":
            rc = handle_timing(
                row, props, out_dir,
                component_strict=args.component_strict,
                shape_point_name=args.shape_point,
            )
        elif args.mode == "trace":
            rc = handle_trace(
                row, props, out_dir,
                shape_point_name=args.shape_point,
                phase_inject_ab=args.phase_inject_ab,
                ab_repeat_count=args.ab_repeat_count,
            )
        elif args.mode == "emit-dual-run-manifest":
            rc = handle_emit_dual_run_manifest(
                cupti_run_dir=args.cupti_run_dir,
                nvbit_run_dir=args.nvbit_run_dir,
                output_path=args.dual_run_manifest_output,
                drift_tolerance_ns=args.drift_tolerance_ns,
            )
            # The dual-run manifest emit path is run-pair-scoped, not
            # per-row, so terminate the loop after the first dispatch
            # regardless of how many rows the targets list contains.
            if rc != EXIT_OK:
                worst = rc
            return worst
        elif args.mode == "regression":
            rc = handle_regression(row, props, args.dry_run_cpu, out_dir,
                                    args.strict_golden_origin, args.allow_missing_golden)
        else:
            return EXIT_USAGE
        # Aggregate worst exit code, preserving sampling_required signal semantics
        if rc != EXIT_OK:
            worst = rc
    return worst


if __name__ == "__main__":
    sys.exit(main())
