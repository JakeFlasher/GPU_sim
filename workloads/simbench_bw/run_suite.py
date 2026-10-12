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
                 compares against the captured workload corpus.

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
from contextlib import ExitStack, contextmanager, nullcontext
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from ctypes import c_char, c_int, c_size_t, POINTER, byref, c_uint
from pathlib import Path
from typing import Any, Callable


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

from tools.nvbit_tracer import capture_process as _nvbit_process  # noqa: E402
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ExclusiveDirectoryStage,
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableDirectoryGeneration,
    ImmutableFile,
    PinnedDirectoryReader,
    PinnedTreeSnapshot,
    RetainedExecutable,
    RetainedRegular,
    create_exclusive_directory_stage,
    ensure_directory,
    inspect_immutable_directory_generation,
    open_retained_executable,
    open_retained_regular,
    publish_directory,
    publish_regular_file,
    publish_retained_regular_file,
    read_regular,
    replace_regular_bytes,
    retire_regular_file,
    sha256_file as anchored_sha256_file,
)
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
    ExternalTracerAuthority,
    require_external_tracer_authority,
    verify_retained_tracer_authority,
)
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    TracerRuntimeError,
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
    sanitized_capture_environment,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    AdmittedTraceSources,
    TraceAdmission,
    TraceAdmissionError,
    open_admitted_trace_sources,
    raw_trace_generation_limits,
    require_trace_admission,
)

from tools.sector_stream.consume_trace import AdmittedSectorStream  # noqa: E402

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

_TRACE_CHILD_TIMEOUT_SECONDS = 60 * 60
_TRACE_CHILD_MAX_STDOUT_STDERR_BYTES = 8 * 1024 * 1024

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
    binary_sha256: str | None = None,
) -> dict[str, Any]:
    if binary_sha256 is not None:
        if (
            len(binary_sha256) != 64
            or any(character not in "0123456789abcdef" for character in binary_sha256)
        ):
            raise ValueError("binary_sha256 override must be lowercase 64-hex")
        binary_digest = binary_sha256
    else:
        binary_digest = _binary_sha256(binary_path) if binary_path else ""
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
        "binary_sha256": binary_digest,
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
        if "rtx 5090" in name or "5090" in name:
            return "rtx_5090_blackwell"
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
    `tools/divergence_proxies/bundle_from_trace.py` accepts `sm_86`,
    historical `sm_120`, and active RTX 5090 `sm_120`; any other
    compute capability raises a
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
        if "rtx 5090" in props.device_name.lower():
            return "sm_120"
        return "sm_120"
    raise ValueError(
        f"unknown_target_device_for_divergence_proxies:"
        f"cc_{cc[0]}_{cc[1]}"
    )


def _perturbation_target_arch_for_props(props: DeviceProps) -> str:
    """Map live `DeviceProps` to the perturbation-harness producer's
    target_arch token. The producer at
    `tools/perturbation_harness/build.py::run_perturbation_harness`
    accepts only the historical values in
    `tools/perturbation_harness/ledger.py::SUPPORTED_TARGET_ARCHES`. The
    active RTX 5090 is intentionally refused by this still-device-specific
    producer rather than mislabeled; historical `sm_120` records remain
    supported. Any other compute capability raises
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
        # The perturbation producer still has no Pro-specific target token;
        # return the supported SM120 token for the active RTX 5090 and fail closed for
        # the active RTX 5090 instead of emitting a mislabeled artifact.
        if "rtx 5090" in props.device_name.lower():
            raise ValueError("RTX 5090 perturbation target is not yet parameterized")
        return "sm_120"
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


def _emit_manifest(
    manifest: dict[str, Any],
    out_dir: Path,
    *,
    filename: str = "measurement_manifest.json",
    preserve_previous: bool = False,
    retain_capability: bool = False,
) -> Path | RetainedRegular:
    """Publish one manifest without following destination path components.

    Ordinary timing/correctness callers retain the historical replace-in-place
    behavior.  Trace activation uses a shape-specific filename with
    ``preserve_previous=True``: the prior regular file is atomically retired as
    byte-exact evidence, then the new file is committed with no-replace
    semantics so a concurrent destination cannot be overwritten. Activation
    callers also set ``retain_capability=True`` so the publication returns the
    exact committed inode with its path monitor already armed.
    """
    if (
        type(filename) is not str
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise ImmutableArtifactError(
            "manifest filename must be one portable path component"
        )
    if type(retain_capability) is not bool:
        raise ImmutableArtifactError("manifest retain selector must be boolean")
    if retain_capability and not preserve_previous:
        raise ImmutableArtifactError(
            "retained manifest publication requires no-replace preservation"
        )
    ensure_directory(out_dir)
    dest = out_dir / filename
    payload = (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode(
        "utf-8"
    )
    if preserve_previous:
        retire_regular_file(dest)
        if retain_capability:
            return publish_retained_regular_file(out_dir, filename, payload)
        publish_regular_file(out_dir, filename, payload)
    else:
        replace_regular_bytes(dest, payload)
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


def _stdout_wall_time_s(stdout: str | bytes) -> float | None:
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
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
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
        publish_regular_file(
            target_dir,
            target.name,
            (
                json.dumps(
                    manifest_pairing_index, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8"),
        )
        written.append(target)
    return written


def _ensure_sector_stream_jsonl(
    traces_folder: Path,
    *,
    trace_sources: AdmittedTraceSources,
) -> tuple[bool, str | None, AdmittedSectorStream | None]:
    """Generate one admission-bound sector result for the feature batch."""
    try:
        from tools.sector_stream.consume_trace import (
            consume_admitted_trace_sources as _consume_trace_sources,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
    except Exception as exc:
        return False, f"sector_stream_consumer_import_failed:{exc}", None
    try:
        result = _consume_trace_sources(
            sources=trace_sources,
            output_root=traces_folder,
            write_jsonl=True,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
    except Exception as exc:
        return False, f"sector_stream_consumer_failed:{exc}", None
    if not isinstance(result, AdmittedSectorStream):
        return False, "sector_stream_result_invalid", None
    if not result.contract_ok:
        return False, (
            f"trace_contract:{result.contract_typed_reason}"
        ), None
    if not result.schema_admits_all:
        return False, (
            "sector_stream_schema_invalid:"
            f"{result.schema_first_failure}"
        ), None
    result.verify_for(trace_sources)
    trace_sources.verify_unchanged()
    return True, None, result


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
    binary_sha256: str,
    kernel_signature_sha256: str,
    props: "DeviceProps",
    trace_sources: AdmittedTraceSources,
    sector_stream: AdmittedSectorStream,
) -> None:
    """Emit the single-launch HRD bundle and run-summary identity."""
    admission = trace_sources.admission
    if admission.launch_count != 1:
        raise TraceAdmissionError(
            "trace_launch_count_unsupported",
            "run-summary HRD identity requires exactly one admitted launch; "
            f"observed launch_count={admission.launch_count}",
        )
    sector_stream.verify_for(trace_sources)
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
        emit_hrd_bundle_from_sources = getattr(
            sys.modules["bundle_from_trace"],
            "emit_hrd_bundle_from_admitted_sources",
        )
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
        emit_hrd_bundle_from_sources = (
            _hrd_module.emit_hrd_bundle_from_admitted_sources
        )

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": binary_sha256,
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": kernel_signature_sha256,
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
        result = emit_hrd_bundle_from_sources(
            sources=trace_sources,
            sector_stream=sector_stream,
            output_root=traces_folder,
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
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
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
        # launch / kernel identity (single-launch gate above)
        "launch_index": admission.launches[0].launch_index,
        "kernel_name": admission.launches[0].kernel_name,
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
    the NVBit/CUPTI sidecar emitter, which serializes
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
    historical sm_120 → 32 MiB (RTX 5090 GPU, Blackwell GB206;
    confirmed via the retained live query). The active RTX 5090 path has
    no static fallback here; it uses a live CUDA value when available and
    otherwise stays unresolved.
    """
    if props.compute_capability_major == 8 and props.compute_capability_minor == 6:
        return 6 * 1024 * 1024
    if props.compute_capability_major == 12:
        if "rtx 5090" in props.device_name.lower():
            return None
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

    sm_120 (active RTX 5090 Workstation, Blackwell GB202) → returns None
    today; the RTX 5090 L2 slice count is not yet captured in
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
    binary_sha256: str,
    kernel_signature_sha256: str,
    props: "DeviceProps",
    trace_sources: AdmittedTraceSources,
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
            compute_order_projections_from_admitted_sources,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
    except Exception as exc:
        _order_projections_skip_run_summary(
            artifact_record,
            skip_reason=f"projector_import_failed:{exc}",
        )
        return

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": binary_sha256,
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": kernel_signature_sha256,
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
        bundle = compute_order_projections_from_admitted_sources(
            sources=trace_sources,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
            capacity_bytes=capacity_bytes,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
    except Exception as exc:
        _order_projections_skip_run_summary(
            artifact_record,
            skip_reason=f"compute_failed:{exc}",
        )
        return

    ensure_directory(op_dir)
    publish_regular_file(
        op_dir,
        op_path.name,
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
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
    binary_sha256: str,
    kernel_signature_sha256: str,
    props: "DeviceProps",
    trace_sources: AdmittedTraceSources,
    sector_stream: AdmittedSectorStream,
) -> None:
    """Emit single-launch divergence proxies from admission-bound sectors."""
    if trace_sources.admission.launch_count != 1:
        raise TraceAdmissionError(
            "trace_launch_count_unsupported",
            "run-summary divergence identity requires exactly one admitted "
            f"launch; observed launch_count="
            f"{trace_sources.admission.launch_count}",
        )
    sector_stream.verify_for(trace_sources)
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
        _div_consume_trace_sources = (
            _div_module.consume_admitted_trace_sources
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
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
        "binary_sha256": binary_sha256,
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": kernel_signature_sha256,
    }

    try:
        result = _div_consume_trace_sources(
            sources=trace_sources,
            sector_stream=sector_stream,
            output_root=traces_folder,
            device=device_token,
            manifest_pairing_index=manifest_pairing_index,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
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
    binary_sha256: str,
    kernel_signature_sha256: str,
    props: "DeviceProps",
    trace_sources: AdmittedTraceSources,
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
            compute_sharing_locality_from_admitted_sources as _compute_sharing_locality_from_sources,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
    except Exception as exc:
        _sharing_locality_skip_run_summary(
            artifact_record,
            skip_reason=f"producer_import_failed:{exc}",
        )
        return

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point["name"],
        "binary_sha256": binary_sha256,
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": kernel_signature_sha256,
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
        bundle = _compute_sharing_locality_from_sources(
            sources=trace_sources,
            manifest_pairing_index=manifest_pairing_index,
            device=device_block,
        )
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
    except Exception as exc:
        _sharing_locality_skip_run_summary(
            artifact_record,
            skip_reason=f"compute_failed:{exc}",
        )
        return

    sl_subdir = traces_folder / "sharing_locality"
    ensure_directory(sl_subdir)
    sl_path = sl_subdir / "sharing_locality.json"
    publish_regular_file(
        sl_subdir,
        sl_path.name,
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
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
        schema = json.loads(read_regular(schema_path))
        validator = _jsonschema.Draft202012Validator(schema)
        on_disk = json.loads(read_regular(sl_path))
        errors = list(validator.iter_errors(on_disk))
    except (
        TraceAdmissionError,
        ImmutableArtifactLimitError,
        ImmutableArtifactResourceError,
        ImmutableArtifactError,
    ):
        raise
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
    binary_sha256: str,
    kernel_signature_sha256: str,
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
        "binary_sha256": binary_sha256,
        "source_commit": _resolve_source_commit(),
        "kernel_signature_sha256": kernel_signature_sha256,
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

    ensure_directory(ledger_dir)
    publish_regular_file(
        ledger_dir,
        ledger_path.name,
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
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
    trace_admission: TraceAdmission | None = None,
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

    # Canonical callers pass the admitted size and never reopen/stat the raw
    # trace path.  The path-compatible fallback remains for standalone tests and
    # older call sites that do not yet own a Task235 capability.
    if trace_admission is not None:
        actual_trace_bytes = trace_admission.dynamic_trace_size_bytes
    else:
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

    ensure_directory(bundle_dir)
    publish_regular_file(
        bundle_dir,
        bundle_path.name,
        (json.dumps(bundle, indent=2) + "\n").encode("utf-8"),
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


def _write_trace_volume_exceeded(
    *,
    sidecar: Path,
    slug: str,
    shape_point: str,
    budget_bytes: int,
    actual_trace_bytes: int,
    actual_wall_time_s: float,
) -> None:
    trace_budgets_sha256, _size = anchored_sha256_file(
        SUITE_ROOT / "trace_budgets.toml"
    )
    record = {
        "schema_version": 1,
        "slug": slug,
        "shape_point": shape_point,
        "sampling_policy": "off",
        "max_trace_bytes": budget_bytes,
        "actual_trace_bytes": actual_trace_bytes,
        "max_wall_time_s": 3600,
        "actual_wall_time_s": actual_wall_time_s,
        "exceeded_trace_bytes": True,
        "exceeded_wall_time": False,
        "trace_budgets_sha256": trace_budgets_sha256,
    }
    ensure_directory(sidecar.parent)
    retire_regular_file(sidecar)
    publish_regular_file(
        sidecar.parent,
        sidecar.name,
        (json.dumps(record, indent=2) + "\n").encode("utf-8"),
    )


def _trace_artifact_record_payload(record: dict[str, Any]) -> bytes:
    """Serialize the exact canonical locator bytes used for publication."""
    return (json.dumps(record, indent=2) + "\n").encode("utf-8")


def _publish_trace_artifact_record(
    path: Path,
    record: dict[str, Any],
    *,
    before_commit: Callable[[], None] | None = None,
    after_commit: Callable[[], None] | None = None,
) -> None:
    """Commit the sole trace-activation leaf with no-replace semantics.

    The caller must retire any prior active record before beginning the rerun.
    Publication therefore refuses every destination race rather than replacing
    a concurrently created regular file.  The committed leaf is read-only and
    can be retired byte-exact by the next attempt. ``before_commit`` is the
    final authority check immediately before the no-replace rename;
    ``after_commit`` repeats it while rollback guards can still retire a bad
    commit.
    """
    publish_regular_file(
        path.parent,
        path.name,
        _trace_artifact_record_payload(record),
        before_commit=before_commit,
        after_commit=after_commit,
    )


def _admitted_trace_exceeds_budget(
    *,
    trace_admission: TraceAdmission,
    budget_bytes: int,
    sidecar: Path,
    slug: str,
    shape_point: str,
    actual_wall_time_s: float,
) -> bool:
    actual_trace_bytes = trace_admission.dynamic_trace_size_bytes
    if actual_trace_bytes <= budget_bytes:
        retire_regular_file(sidecar)
        return False
    _write_trace_volume_exceeded(
        sidecar=sidecar,
        slug=slug,
        shape_point=shape_point,
        budget_bytes=budget_bytes,
        actual_trace_bytes=actual_trace_bytes,
        actual_wall_time_s=actual_wall_time_s,
    )
    return True


def _emit_nvbit_budget_sidecars(
    *,
    output_root: Path,
    artifact_dir: Path,
    slug: str,
    shape_point: str,
    actual_slowdown_ratio: float | None,
    baseline_wall_time_s: float | None,
    actual_wall_time_s: float,
    actual_drop_rate: float | None,
    expected_record_count: int | None,
    admitted_record_count: int | None,
) -> dict[str, Path | None]:
    """Publish attempt-scoped slowdown/drop evidence without stale reuse."""
    for legacy_name in (
        "nvbit_slowdown_budget_exceeded.json",
        "nvbit_drop_rate_exceeded.json",
    ):
        retire_regular_file(artifact_dir / legacy_name)
    trace_budgets_sha256, _size = anchored_sha256_file(
        SUITE_ROOT / "trace_budgets.toml"
    )
    result: dict[str, Path | None] = {
        "slowdown": None,
        "drop_rate": None,
    }
    if (
        actual_slowdown_ratio is not None
        and baseline_wall_time_s is not None
        and float(baseline_wall_time_s)
        >= _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S
        and actual_slowdown_ratio
        > _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO
    ):
        payload = (
            json.dumps({
                "schema_version": 1,
                "slug": slug,
                "shape_point": shape_point,
                "max_slowdown_ratio": (
                    _trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO
                ),
                "actual_slowdown_ratio": actual_slowdown_ratio,
                "baseline_wall_time_s": float(baseline_wall_time_s),
                "actual_wall_time_s": actual_wall_time_s,
                "trace_budgets_sha256": trace_budgets_sha256,
            }, indent=2)
            + "\n"
        ).encode("utf-8")
        result["slowdown"] = publish_regular_file(
            output_root,
            "nvbit_slowdown_budget_exceeded.json",
            payload,
        )
        print(
            f"nvbit_slowdown_budget_exceeded: actual_ratio="
            f"{actual_slowdown_ratio:.3f} > max_ratio="
            f"{_trace_budgets.NVBIT_SLOWDOWN_BUDGET_MAX_RATIO}",
            file=sys.stderr,
        )
        print(
            "nvbit_slowdown_budget_exceeded: sidecar at "
            f"{result['slowdown']}",
            file=sys.stderr,
        )

    if (
        actual_drop_rate is not None
        and expected_record_count is not None
        and actual_drop_rate > _trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX
    ):
        admitted_int = int(admitted_record_count or 0)
        residual = int(expected_record_count) - admitted_int
        payload = (
            json.dumps({
                "schema_version": 1,
                "slug": slug,
                "shape_point": shape_point,
                "max_drop_rate": _trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX,
                "actual_drop_rate": actual_drop_rate,
                "expected_record_count": int(expected_record_count),
                "admitted_record_count": admitted_int,
                "channel_overflow_dropped": 0,
                "ring_buffer_loss_dropped": 0,
                "decoder_gap_dropped": 0,
                "expected_record_mismatch_dropped": residual,
                "trace_budgets_sha256": trace_budgets_sha256,
            }, indent=2)
            + "\n"
        ).encode("utf-8")
        result["drop_rate"] = publish_regular_file(
            output_root,
            "nvbit_drop_rate_exceeded.json",
            payload,
        )
        print(
            f"nvbit_drop_rate_exceeded: actual_rate="
            f"{actual_drop_rate:.3f} > max_rate="
            f"{_trace_budgets.NVBIT_DROP_RATE_BUDGET_MAX}",
            file=sys.stderr,
        )
        print(
            "nvbit_drop_rate_exceeded: sidecar at "
            f"{result['drop_rate']}",
            file=sys.stderr,
        )
    return result


RAW_TRACE_GENERATION_NAMESPACE = "raw_trace_generations"
TRACE_DERIVED_GENERATION_NAMESPACE = "trace_derived_generations"
PHASE_LEDGER_GENERATION_NAMESPACE = "phase_cycle_ledger_generations"
TRACE_CAPTURE_ATTEMPT_NAMESPACE = "trace_capture_attempts"


def _trace_generation_root(
    *, artifact_dir: Path, namespace: str, shape_point: str,
) -> Path:
    if namespace not in {
        RAW_TRACE_GENERATION_NAMESPACE,
        TRACE_DERIVED_GENERATION_NAMESPACE,
        PHASE_LEDGER_GENERATION_NAMESPACE,
    }:
        raise ImmutableArtifactError("unknown trace generation namespace")
    if (
        type(shape_point) is not str
        or not shape_point
        or shape_point in {".", ".."}
        or "/" in shape_point
        or "\\" in shape_point
        or len(os.fsencode(shape_point)) > 255
    ):
        raise ImmutableArtifactError(
            "shape_point must be one nonempty portable path component"
        )
    return artifact_dir / namespace / shape_point


def _new_trace_attempt_id() -> str:
    """Return a no-reuse generation name independent of trace content."""
    return f"attempt-{os.urandom(16).hex()}"


def _trace_generation_commit_limits() -> dict[
    str, int | dict[str, int]
]:
    return raw_trace_generation_limits(DEFAULT_TRACE_ADMISSION_LIMITS)


def _derived_generation_commit_limits() -> dict[str, int]:
    limits = DEFAULT_TRACE_ADMISSION_LIMITS
    return {
        "max_files": limits.tree_files,
        "max_directories": limits.tree_directories,
        "max_total_file_bytes": (
            limits.trace_root_bytes
            + limits.allocations_sidecar_bytes
            + limits.stats_bytes
            + limits.total_threadblock_bytes
        ),
        "max_total_relative_path_bytes": limits.tree_relative_path_bytes,
    }


def _directory_generation_record(
    *,
    generation: ImmutableDirectoryGeneration,
    artifact_dir: Path,
    attempt_id: str,
) -> dict[str, Any]:
    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    generation_path = Path(os.path.abspath(os.fspath(generation.path)))
    try:
        relative = generation_path.relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ImmutableArtifactError(
            f"trace generation escapes artifact directory: {generation_path}"
        ) from exc
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "path": relative,
        "content_manifest_sha256": generation.content_manifest_sha256,
        "filesystem_generation_sha256": (
            generation.filesystem_generation_sha256
        ),
        "file_count": generation.file_count,
        "directory_count": generation.directory_count,
        "total_file_bytes": generation.total_file_bytes,
    }


class _TraceActivationAuthorityError(ImmutableArtifactError):
    """A required activation role is absent or structurally invalid."""

    def __init__(self, typed_reason: str, detail: str) -> None:
        self.typed_reason = typed_reason
        self.detail = detail
        super().__init__(f"{typed_reason}: {detail}")


class _RetainedDirectoryGenerationAuthority:
    """One descriptor-bound immutable tree retained through activation."""

    def __init__(
        self,
        *,
        role: str,
        generation: ImmutableDirectoryGeneration,
        reader: PinnedDirectoryReader,
        snapshot: PinnedTreeSnapshot,
        max_total_relative_path_bytes: int,
    ) -> None:
        self.role = role
        self.generation = generation
        self.reader = reader
        self.snapshot = snapshot
        self.max_total_relative_path_bytes = max_total_relative_path_bytes

    def verify_unchanged(self) -> None:
        """Rehash exact bytes and rebind all metadata through retained roots."""
        self.reader.verify_path()
        observed = self.reader.snapshot_tree(
            require_readonly_files=True,
            require_readonly_directories=True,
            max_files=max(1, self.generation.file_count),
            max_directories=max(1, self.generation.directory_count),
            max_total_file_bytes=self.generation.total_file_bytes,
            max_total_relative_path_bytes=self.max_total_relative_path_bytes,
        )
        if observed != self.snapshot:
            raise ImmutableArtifactError(
                f"retained {self.role} tree metadata generation changed"
            )
        files = dict(observed.files)
        directories = dict(observed.directories)
        expected_files = {
            identity.relative_path: identity
            for identity in self.generation.files
        }
        if (
            set(files) != set(expected_files)
            or len(directories) != self.generation.directory_count
            or "." not in directories
        ):
            raise ImmutableArtifactError(
                f"retained {self.role} tree inventory changed"
            )
        for relative_path, metadata in directories.items():
            if (
                not stat.S_ISDIR(metadata[2])
                or stat.S_IMODE(metadata[2]) != 0o555
            ):
                raise ImmutableArtifactError(
                    f"retained {self.role} directory is not read-only: "
                    f"{relative_path}"
                )
        for relative_path, identity in sorted(expected_files.items()):
            metadata = files[relative_path]
            if (
                not stat.S_ISREG(metadata[2])
                or stat.S_IMODE(metadata[2]) != 0o444
                or metadata[3] != 1
                or metadata[4] != identity.size_bytes
            ):
                raise ImmutableArtifactError(
                    f"retained {self.role} file metadata changed: {relative_path}"
                )
            payload = self.reader.read_snapshot(
                relative_path,
                max_bytes=identity.size_bytes,
                expected_metadata=metadata,
            ).payload
            if (
                len(payload) != identity.size_bytes
                or hashlib.sha256(payload).hexdigest() != identity.sha256
            ):
                raise ImmutableArtifactError(
                    f"retained {self.role} file bytes changed: {relative_path}"
                )
        self.reader.verify_path()
        if self.reader.snapshot_tree(
            require_readonly_files=True,
            require_readonly_directories=True,
            max_files=max(1, self.generation.file_count),
            max_directories=max(1, self.generation.directory_count),
            max_total_file_bytes=self.generation.total_file_bytes,
            max_total_relative_path_bytes=self.max_total_relative_path_bytes,
        ) != self.snapshot:
            raise ImmutableArtifactError(
                f"retained {self.role} tree changed during byte verification"
            )


class _TraceActivationAuthoritySet:
    """Explicit complete role set held from the final gate through commit."""

    def __init__(self, expected_roles: set[str]) -> None:
        if not expected_roles or any(
            type(role) is not str or not role for role in expected_roles
        ):
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_invalid",
                "activation authority roles must be nonempty strings",
            )
        self._expected_roles = frozenset(expected_roles)
        self._verifiers: dict[str, Callable[[], None]] = {}
        self._stack = ExitStack()
        self._closed = False

    @property
    def stack(self) -> ExitStack:
        if self._closed:
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_closed",
                "activation authority stack is already closed",
            )
        return self._stack

    def add(self, role: str, verifier: Callable[[], None]) -> None:
        if role not in self._expected_roles:
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_unexpected",
                f"unexpected activation authority role: {role}",
            )
        if role in self._verifiers:
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_duplicate",
                f"duplicate activation authority role: {role}",
            )
        if not callable(verifier):
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_invalid",
                f"activation authority verifier is not callable: {role}",
            )
        self._verifiers[role] = verifier

    def verify_unchanged(self) -> None:
        if self._closed:
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_closed",
                "activation authority set closed before commit",
            )
        observed_roles = frozenset(self._verifiers)
        if observed_roles != self._expected_roles:
            missing = sorted(self._expected_roles - observed_roles)
            unexpected = sorted(observed_roles - self._expected_roles)
            raise _TraceActivationAuthorityError(
                "trace_activation_authority_missing",
                f"missing_roles={missing} unexpected_roles={unexpected}",
            )
        for role in sorted(self._verifiers):
            try:
                self._verifiers[role]()
            except BaseException as exc:
                try:
                    exc.add_note(
                        f"trace activation authority role failed: {role}"
                    )
                except (AttributeError, TypeError):
                    pass
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stack.close()



def _retain_exact_directory_generation(
    *,
    authorities: _TraceActivationAuthoritySet,
    role: str,
    generation: ImmutableDirectoryGeneration,
    require_nonempty: bool,
    max_total_relative_path_bytes: int,
) -> _RetainedDirectoryGenerationAuthority:
    """Arm a tree monitor and bind it to one previously admitted generation."""
    reader = authorities.stack.enter_context(
        PinnedDirectoryReader(generation.path)
    )
    snapshot = authorities.stack.enter_context(
        reader.retain_tree_generation(
            require_readonly_files=True,
            require_readonly_directories=True,
            max_files=max(1, generation.file_count),
            max_directories=max(1, generation.directory_count),
            max_total_file_bytes=generation.total_file_bytes,
            max_total_relative_path_bytes=max_total_relative_path_bytes,
        )
    )
    observed = inspect_immutable_directory_generation(
        generation.path,
        require_nonempty=require_nonempty,
        max_files=max(1, generation.file_count),
        max_directories=max(1, generation.directory_count),
        max_total_file_bytes=generation.total_file_bytes,
        max_total_relative_path_bytes=max_total_relative_path_bytes,
    )
    if observed != generation:
        raise ImmutableArtifactError(
            f"{role} generation changed before retained authority acquisition"
        )
    retained = _RetainedDirectoryGenerationAuthority(
        role=role,
        generation=generation,
        reader=reader,
        snapshot=snapshot,
        max_total_relative_path_bytes=max_total_relative_path_bytes,
    )
    retained.verify_unchanged()
    authorities.add(role, retained.verify_unchanged)
    return retained



def _retained_capability_identity(
    *,
    retained: RetainedRegular,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Project one live retained file into the portable active-record identity."""
    retained.verify_unchanged()
    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    try:
        relative = retained.path.relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ImmutableArtifactError(
            f"evidence path escapes artifact directory: {retained.path}"
        ) from exc
    return {
        "path": relative,
        "sha256": retained.sha256,
        "size_bytes": retained.size_bytes,
    }



def _summary_generation_from_record(
    *,
    record: dict[str, Any],
    artifact_dir: Path,
) -> ImmutableDirectoryGeneration:
    """Recover and authenticate the exact tree named by a summary descriptor."""
    if type(record) is not dict:
        raise _TraceActivationAuthorityError(
            "trace_activation_authority_invalid",
            "summary generation descriptor is not an object",
        )
    path_text = record.get("path")
    file_count = record.get("file_count")
    directory_count = record.get("directory_count")
    total_file_bytes = record.get("total_file_bytes")
    if (
        type(path_text) is not str
        or not path_text
        or type(file_count) is not int
        or file_count <= 0
        or type(directory_count) is not int
        or directory_count <= 0
        or type(total_file_bytes) is not int
        or total_file_bytes < 0
    ):
        raise _TraceActivationAuthorityError(
            "trace_activation_authority_invalid",
            "summary generation descriptor counts or path are invalid",
        )
    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    generation_path = artifact_root / path_text
    try:
        if generation_path.relative_to(artifact_root).as_posix() != path_text:
            raise ValueError(path_text)
    except ValueError as exc:
        raise _TraceActivationAuthorityError(
            "trace_activation_authority_invalid",
            "summary generation descriptor escapes the artifact directory",
        ) from exc
    generation = inspect_immutable_directory_generation(
        generation_path,
        require_nonempty=True,
        max_files=file_count,
        max_directories=directory_count,
        max_total_file_bytes=total_file_bytes,
        max_total_relative_path_bytes=max(4096, file_count * 4096),
    )
    for field, observed in (
        ("content_manifest_sha256", generation.content_manifest_sha256),
        (
            "filesystem_generation_sha256",
            generation.filesystem_generation_sha256,
        ),
        ("file_count", generation.file_count),
        ("directory_count", generation.directory_count),
        ("total_file_bytes", generation.total_file_bytes),
    ):
        if record.get(field) != observed:
            raise ImmutableArtifactError(
                f"summary generation descriptor changed: {field}"
            )
    expected_files = record.get("files")
    observed_files = [
        {
            "path": (Path(path_text) / identity.relative_path).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in generation.files
    ]
    if expected_files != observed_files:
        raise ImmutableArtifactError(
            "summary generation descriptor file inventory changed"
        )
    return generation



def _publish_trace_attempt_event(
    *,
    artifact_dir: Path,
    slug: str,
    shape_point: str,
    attempt_id: str,
    event_name: str,
    status: str,
    typed_reason: str | None,
    raw_trace_generation: dict[str, Any] | None = None,
    raw_staging_path: Path | None = None,
    raw_unresolved_committed_path: Path | None = None,
    derived_generation: dict[str, Any] | None = None,
    derived_staging_path: Path | None = None,
    phase_ledger_generation: dict[str, Any] | None = None,
    phase_ledger_staging_path: Path | None = None,
    summary_generation: dict[str, Any] | None = None,
    measurement_manifest: dict[str, Any] | None = None,
    process_failure_evidence: list[dict[str, Any]] | None = None,
    observed_output_bytes_lower_bound: int | None = None,
    retained_authorities: _TraceActivationAuthoritySet | None = None,
) -> dict[str, Any]:
    """Publish one immutable, non-active event for an attempt.

    Events never act as canonical authority.  They make handled failures typed
    and preserve exact staging/final references without replacing earlier event
    bytes.  A successful activation references its ``activation_ready`` event
    and is still the last publication.
    """
    if event_name not in {
        "stage_creation_failed",
        "capture_failed",
        "typed_unavailable",
        "raw_commit_failed",
        "admission_failed",
        "budget_rejected",
        "preactivation_failed",
        "activation_ready",
        "activation_failed",
    }:
        raise ImmutableArtifactError("unknown trace attempt event name")
    attempt_dir = (
        artifact_dir
        / TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / shape_point
        / attempt_id
    )
    ensure_directory(attempt_dir)

    def relative_staging(path: Path | None) -> str | None:
        if path is None:
            return None
        artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            return absolute.relative_to(artifact_root).as_posix()
        except ValueError as exc:
            raise ImmutableArtifactError(
                f"trace attempt staging path escapes artifact directory: {absolute}"
            ) from exc

    event = {
        "schema_version": 1,
        "slug": slug,
        "shape_point": shape_point,
        "attempt_id": attempt_id,
        "status": status,
        "typed_reason": typed_reason,
        "activation_state": "non_active",
        "raw_trace_generation": raw_trace_generation,
        "raw_staging_path": relative_staging(raw_staging_path),
        "raw_unresolved_committed_path": relative_staging(
            raw_unresolved_committed_path
        ),
        "trace_derived_generation": derived_generation,
        "derived_staging_path": relative_staging(derived_staging_path),
        "phase_cycle_ledger_generation": phase_ledger_generation,
        "phase_ledger_staging_path": relative_staging(
            phase_ledger_staging_path
        ),
        "nvbit_trace_summary_generation": summary_generation,
        "measurement_manifest": measurement_manifest,
    }
    if process_failure_evidence is not None:
        event["process_failure_evidence"] = process_failure_evidence
    if observed_output_bytes_lower_bound is not None:
        if (
            type(observed_output_bytes_lower_bound) is not int
            or observed_output_bytes_lower_bound < 0
        ):
            raise ImmutableArtifactError(
                "trace attempt output-byte lower bound is invalid"
            )
        event["observed_output_bytes_lower_bound"] = (
            observed_output_bytes_lower_bound
        )
    event_payload = (json.dumps(event, indent=2) + "\n").encode("utf-8")
    if retained_authorities is None:
        event_path = publish_regular_file(
            attempt_dir,
            f"{event_name}.json",
            event_payload,
        )
        assert isinstance(event_path, Path)
        return _retained_artifact_identity(
            path=event_path, artifact_dir=artifact_dir
        )
    if event_name != "activation_ready":
        raise _TraceActivationAuthorityError(
            "trace_activation_authority_invalid",
            "only activation_ready may become a retained attempt event",
        )
    retained_event = publish_retained_regular_file(
        attempt_dir,
        f"{event_name}.json",
        event_payload,
    )
    retained_event = retained_authorities.stack.enter_context(retained_event)
    retained_authorities.add(
        "activation_ready_event", retained_event.verify_unchanged
    )
    return _retained_capability_identity(
        retained=retained_event, artifact_dir=artifact_dir
    )


def _rebase_trace_feature_paths(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    staged_root: Path,
    committed_root: Path,
) -> None:
    """Move explicit feature references from the hidden stage to its final name."""
    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    staged = Path(os.path.abspath(os.fspath(staged_root)))
    committed = Path(os.path.abspath(os.fspath(committed_root)))
    try:
        staged_relative = staged.relative_to(artifact_root)
        committed_relative = committed.relative_to(artifact_root)
    except ValueError as exc:
        raise ImmutableArtifactError(
            "trace feature generation escapes artifact directory"
        ) from exc
    for field in (
        "sector_stream_path",
        "hrd_profile_path",
        "divergence_proxies_path",
        "sharing_locality_path",
        "order_projections_path",
        "perturbation_run_ledger_path",
        "trace_convergence_confidence_path",
        "nvbit_slowdown_budget_sidecar_path",
        "nvbit_drop_rate_sidecar_path",
    ):
        value = artifact_record.get(field)
        if value is None:
            continue
        if type(value) is not str or not value:
            raise ImmutableArtifactError(
                f"trace feature path field is invalid: {field}"
            )
        path = Path(value)
        try:
            suffix = path.relative_to(staged_relative)
        except ValueError as exc:
            raise ImmutableArtifactError(
                f"trace feature path does not reference its staging generation: {field}"
            ) from exc
        artifact_record[field] = (committed_relative / suffix).as_posix()


NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE = (
    "nvbit_trace_summary_generations"
)


def _nvbit_trace_summary_generation_root(
    *,
    artifact_dir: Path,
    shape_point: str,
    trace_admission_sha256: str,
) -> Path:
    """Return the parent whose final child is one filesystem generation.

    The artifact directory already scopes ``slug``.  The remaining directory
    components scope the batch by shape and portable content admission; the
    final no-replace directory name is the local filesystem-generation seal.
    Shape names are committed roster inputs, but still fail closed unless they
    are one portable path component so they cannot escape the namespace.
    """
    if (
        type(shape_point) is not str
        or not shape_point
        or shape_point in {".", ".."}
        or "/" in shape_point
        or "\\" in shape_point
        or len(os.fsencode(shape_point)) > 255
    ):
        raise ImmutableArtifactError(
            "shape_point must be one nonempty portable path component"
        )
    if (
        len(trace_admission_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in trace_admission_sha256
        )
    ):
        raise ImmutableArtifactError(
            "trace_admission_sha256 must be lowercase 64-hex"
        )
    return (
        artifact_dir
        / NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
        / shape_point
        / trace_admission_sha256
    )


def _emit_nvbit_trace_summaries_for_trace_artifact(
    *,
    trace_admission: TraceAdmission,
    slug: str,
    shape_point: str,
    binary_sha256: str,
    tracer_build_sha256: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Per-launch NVBit trace summary emission, factored out of
    `handle_trace` so the orchestrator's interaction with
    `nvbit_trace_summary_adapter` is host-independently testable
    via `tests/test_nvbit_trace_summary_adapter.py`.

    Returns a dict with:
      - `status`: one of `ok`, `adapter_import_failed`,
        `tracked_protobuf_corrupt`, `protobuf_runtime_prerequisite_missing`,
        `resource_unavailable`, `adapter_error`.
      - `paths`: exact paths relative to `artifact_dir` inside the committed
        immutable generation; empty when status is non-ok.
      - `failure_reason`: human-readable diagnostic when status is
        non-ok; None on success.
      - `failed_hard`: True for every non-ok dependency or adapter result.

    The adapter and all nine `_pb2.py` files are tracked authorities. Missing
    source/binding bytes are repository corruption; a missing protobuf runtime
    is a prerequisite failure. Neither condition is a graceful trace result.
    Launch metadata, counts, and all trace identities come only from
    `trace_admission`. Executable and tracer digests must come from the retained
    descriptors used for execution and injection. This producer reopens none of
    those source paths.  Every launch is rendered outside the artifact namespace,
    converted to an in-memory ``ImmutableFile``, and committed together by the
    final generation-directory rename.  Prior generations are never removed.
    """
    result: dict[str, Any] = {
        "status": "not_attempted",
        "paths": [],
        "failure_reason": None,
        "failed_hard": False,
    }

    def fail(status: str, reason: str) -> dict[str, Any]:
        result["status"] = status
        result["failure_reason"] = reason
        result["paths"] = []
        result["failed_hard"] = True
        return result

    try:
        import nvbit_trace_summary_adapter as _nvbit_adapter  # pyright: ignore[reportMissingImports]
    except MemoryError as exc:
        return fail(
            "resource_unavailable",
            "trace_summary_resource_unavailable: NVBit summary adapter import "
            f"exhausted memory: {type(exc).__name__}: {exc}",
        )
    except ImportError as exc:
        return fail(
            "adapter_import_failed",
            "tracked adapter source is unavailable; restore "
            "workloads/simbench_bw/nvbit_trace_summary_adapter.py from "
            f"the committed tree: {exc}",
        )
    launch_count = trace_admission.launch_count
    if launch_count != len(trace_admission.launches):
        return fail(
            "adapter_error",
            "admitted launch count disagrees with retained summary inputs",
        )

    material: list[ImmutableFile] = []
    try:
        filesystem_generation = (
            trace_admission.trace_filesystem_generation_sha256
        )
        if (
            len(filesystem_generation) != 64
            or any(
                character not in "0123456789abcdef"
                for character in filesystem_generation
            )
        ):
            raise ImmutableArtifactError(
                "trace_filesystem_generation_sha256 must be lowercase 64-hex"
            )
        generation_root = _nvbit_trace_summary_generation_root(
            artifact_dir=artifact_dir,
            shape_point=shape_point,
            trace_admission_sha256=(
                trace_admission.trace_admission_sha256
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix="nvbit-trace-summary-batch-"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for launch_index in range(launch_count):
                filename = f"nvbit_trace_summary_0_{launch_index}.json"
                staged_path = temporary_root / filename
                _nvbit_adapter.emit_admitted_nvbit_trace_summary(
                    admission=trace_admission,
                    slug=slug,
                    shape_point=shape_point,
                    repeat_index=0,
                    launch_index=launch_index,
                    binary_sha256=binary_sha256,
                    tracer_build_sha256=tracer_build_sha256,
                    output_path=staged_path,
                )
                with open_retained_regular(staged_path) as staged:
                    payload = staged.read_bytes()
                material.append(ImmutableFile.from_bytes(filename, payload))

            published_directory = publish_directory(
                generation_root,
                filesystem_generation,
                material,
                accept_existing=False,
            )

        artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
        generation_relative = published_directory.relative_to(
            artifact_root
        )
    except (ImmutableArtifactResourceError, MemoryError) as exc:
        return fail(
            "resource_unavailable",
            "trace_summary_resource_unavailable: transactional NVBit summary "
            f"emission failed: {type(exc).__name__}: {exc}",
        )
    except _nvbit_adapter.NvbitTrackedProtobufCorruption as exc:
        return fail(
            "tracked_protobuf_corrupt",
            str(exc),
        )
    except _nvbit_adapter.NvbitProtobufRuntimePrerequisiteMissing as exc:
        return fail(
            "protobuf_runtime_prerequisite_missing",
            str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return fail(
            "adapter_error",
            "transactional NVBit summary emission failed: "
            f"{type(exc).__name__}: {exc}",
        )

    result["status"] = "ok"
    result["paths"] = [
        (generation_relative / file.relative_path).as_posix()
        for file in material
    ]
    return result


def _expected_admitted_nvbit_summary_record(
    *,
    trace_admission: TraceAdmission,
    slug: str,
    shape_point: str,
    launch_index: int,
    binary_sha256: str,
    tracer_build_sha256: str,
) -> dict[str, Any]:
    """Rebuild the exact V2 body allowed to become active."""
    import nvbit_trace_summary_adapter as adapter  # pyright: ignore[reportMissingImports]

    launch = trace_admission.launches[launch_index]
    return {
        "schema_version": 2,
        "slug": slug,
        "shape_point": shape_point,
        "repeat_index": 0,
        "launch_index": launch_index,
        "kernel_name": launch.kernel_name,
        "kernel_signature_sha256": adapter.compute_kernel_signature_sha256(
            launch.kernel_signature_inputs()
        ),
        "inst_count": launch.inst_count,
        "binary_sha256": binary_sha256,
        "nvbit_trace_pb_path": trace_admission.dynamic_trace_path,
        "nvbit_trace_pb_sha256": trace_admission.dynamic_trace_sha256,
        "nvbit_tracer_build_hash": tracer_build_sha256,
        "nvbit_threadblock_payload_size_bytes": (
            trace_admission.threadblock_payload_size_bytes
        ),
        "nvbit_threadblock_payload_manifest_sha256": (
            trace_admission.threadblock_payload_manifest_sha256
        ),
        "trace_admission_sha256": trace_admission.trace_admission_sha256,
    }


def _require_one_committed_nvbit_summary_generation(
    *,
    emit_result: dict[str, Any],
    trace_admission: TraceAdmission,
    slug: str,
    shape_point: str,
    binary_sha256: str,
    tracer_build_sha256: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Close the exact immutable summary generation before activation.

    The active record is allowed to name exactly one complete, read-only
    generation.  This gate inventories that generation through a pinned
    no-follow reader, rejects extra/missing launches and hard links, binds every
    summary body to the admitted trace, and returns the file identities recorded
    by the activation leaf.
    """
    if emit_result.get("status") != "ok" or emit_result.get("failed_hard"):
        raise ImmutableArtifactError(
            "NVBit summary generation is not eligible for activation"
        )
    paths = emit_result.get("paths")
    if type(paths) is not list or not paths:
        raise ImmutableArtifactError(
            "NVBit summary generation has no committed launch references"
        )

    generation_relative = (
        Path(NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE)
        / shape_point
        / trace_admission.trace_admission_sha256
        / trace_admission.trace_filesystem_generation_sha256
    )
    expected_paths = [
        (
            generation_relative
            / f"nvbit_trace_summary_0_{launch_index}.json"
        ).as_posix()
        for launch_index in range(trace_admission.launch_count)
    ]
    if paths != expected_paths:
        raise ImmutableArtifactError(
            "NVBit summary references do not name exactly one admitted generation"
        )

    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    generation_dir = artifact_root.joinpath(*generation_relative.parts)
    expected_names = {Path(path).name for path in expected_paths}
    identities: list[dict[str, Any]] = []
    with PinnedDirectoryReader(generation_dir) as reader:
        snapshot = reader.snapshot_tree(
            require_readonly_files=True,
            require_readonly_directories=True,
            max_files=max(1, trace_admission.launch_count),
            max_directories=1,
            max_total_file_bytes=max(1, trace_admission.launch_count)
            * 16
            * 1024
            * 1024,
            max_total_relative_path_bytes=max(
                4096, trace_admission.launch_count * 4096
            ),
        )
        files = dict(snapshot.files)
        directories = dict(snapshot.directories)
        if set(files) != expected_names or set(directories) != {"."}:
            raise ImmutableArtifactError(
                "committed NVBit summary generation inventory mismatch"
            )
        if stat.S_IMODE(directories["."][2]) != 0o555:
            raise ImmutableArtifactError(
                "committed NVBit summary generation is not read-only"
            )

        for launch_index, relative_path in enumerate(expected_paths):
            filename = Path(relative_path).name
            metadata = files[filename]
            if stat.S_IMODE(metadata[2]) != 0o444 or metadata[3] != 1:
                raise ImmutableArtifactError(
                    "committed NVBit summary is not a single-link read-only file: "
                    f"{filename}"
                )
            payload = reader.read_snapshot(
                filename, expected_metadata=metadata
            ).payload
            try:
                summary = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ImmutableArtifactError(
                    f"committed NVBit summary is invalid JSON: {filename}: {exc}"
                ) from exc
            expected_summary = _expected_admitted_nvbit_summary_record(
                trace_admission=trace_admission,
                slug=slug,
                shape_point=shape_point,
                launch_index=launch_index,
                binary_sha256=binary_sha256,
                tracer_build_sha256=tracer_build_sha256,
            )
            if type(summary) is not dict or summary != expected_summary:
                raise ImmutableArtifactError(
                    "committed NVBit summary body disagrees with exact admitted "
                    f"launch and execution identity: {filename}"
                )
            identities.append({
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            })
        reader.verify_path()

    sealed_generation = inspect_immutable_directory_generation(
        generation_dir,
        require_nonempty=True,
        max_files=len(identities),
        max_directories=1,
        max_total_file_bytes=sum(
            identity["size_bytes"] for identity in identities
        ),
        max_total_relative_path_bytes=max(
            4096, trace_admission.launch_count * 4096
        ),
    )
    sealed_files = [
        {
            "path": (
                generation_relative / identity.relative_path
            ).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in sealed_generation.files
    ]
    if sealed_files != identities:
        raise ImmutableArtifactError(
            "committed NVBit summary generation seal disagrees with file identities"
        )

    generation_manifest_sha256 = hashlib.sha256(
        json.dumps(
            identities,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "path": generation_relative.as_posix(),
        "content_manifest_sha256": (
            sealed_generation.content_manifest_sha256
        ),
        "filesystem_generation_sha256": (
            sealed_generation.filesystem_generation_sha256
        ),
        "file_count": len(identities),
        "directory_count": sealed_generation.directory_count,
        "total_file_bytes": sealed_generation.total_file_bytes,
        "files": identities,
        "manifest_sha256": generation_manifest_sha256,
    }


def _retained_artifact_identity(
    *, path: Path, artifact_dir: Path,
) -> dict[str, Any]:
    """Return a no-follow identity for one already-published evidence file."""
    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ImmutableArtifactError(
            f"evidence path escapes artifact directory: {absolute}"
        ) from exc
    with open_retained_regular(absolute) as retained:
        retained.read_bytes()
        return {
            "path": relative,
            "sha256": retained.sha256,
            "size_bytes": retained.size_bytes,
        }


def _collect_trace_feature_evidence_identities(
    *, artifact_record: dict[str, Any], artifact_dir: Path,
) -> list[dict[str, Any]]:
    """Bind every successfully published feature path carried by the record."""
    path_fields = (
        "sector_stream_path",
        "hrd_profile_path",
        "divergence_proxies_path",
        "sharing_locality_path",
        "order_projections_path",
        "perturbation_run_ledger_path",
        "trace_convergence_confidence_path",
        "nvbit_slowdown_budget_sidecar_path",
        "nvbit_drop_rate_sidecar_path",
    )
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in path_fields:
        relative_path = artifact_record.get(field)
        if relative_path is None:
            continue
        if type(relative_path) is not str or not relative_path:
            raise ImmutableArtifactError(
                f"trace feature path field is invalid: {field}"
            )
        if relative_path in seen:
            raise ImmutableArtifactError(
                f"duplicate trace feature evidence path: {relative_path}"
            )
        seen.add(relative_path)
        identity = _retained_artifact_identity(
            path=artifact_dir / relative_path,
            artifact_dir=artifact_dir,
        )
        identity["kind"] = field.removesuffix("_path")
        identities.append(identity)
    return sorted(identities, key=lambda identity: identity["path"])


def _require_same_trace_admission(
    expected: TraceAdmission,
    observed: TraceAdmission,
    *,
    stage: str,
) -> None:
    if observed != expected:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            f"{stage} trace admission disagrees with the captured generation",
        )


def _emit_admitted_trace_features(
    *,
    artifact_record: dict[str, Any],
    artifact_dir: Path,
    traces_folder: Path,
    trace_admission: TraceAdmission,
    slug: str,
    shape_point: dict[str, Any],
    binary_sha256: str,
    kernel_signature_sha256: str,
    props: DeviceProps,
    trace_sources: AdmittedTraceSources,
) -> None:
    """Emit every feature under one admitted-source capability boundary.

    Task235's capability is the integration point for downstream producers.
    Until every producer accepts it directly, path-compatible producer calls are
    permitted only inside this continuously monitored capability and cannot be
    activated unless the final capability gate also closes successfully.
    """
    if trace_sources.admission != trace_admission:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            "feature capability admission disagrees with captured generation",
        )
    trace_sources.verify_unchanged()
    if trace_admission.launch_count != 1:
        raise TraceAdmissionError(
            "trace_launch_count_unsupported",
            "coherent run-suite feature batch requires exactly one admitted "
            f"launch; observed launch_count={trace_admission.launch_count}",
        )
    sector_ok, sector_skip_reason, sector_stream = (
        _ensure_sector_stream_jsonl(
            traces_folder,
            trace_sources=trace_sources,
        )
    )
    if not sector_ok:
        reason = sector_skip_reason or "sector_stream_generation_failed"
        artifact_record["sector_stream_path"] = None
        artifact_record["sector_stream_skip_reason"] = reason
        _hrd_skip_run_summary(artifact_record, skip_reason=reason)
        _divergence_proxies_skip_run_summary(
            artifact_record, skip_reason=reason,
        )
        _sharing_locality_skip_run_summary(
            artifact_record, skip_reason=reason,
        )
        _order_projections_skip_run_summary(
            artifact_record, skip_reason=reason,
        )
        trace_sources.verify_unchanged()
        return
    if sector_stream is None:
        raise TraceAdmissionError(
            "sector_stream_result_missing",
            "sector-stream producer reported success without a typed result",
        )
    sector_stream.verify_for(trace_sources)
    artifact_record["sector_stream_path"] = (
        traces_folder / "sector_stream" / "sector_stream.jsonl"
    ).relative_to(artifact_dir).as_posix()
    _emit_hrd_bundle_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha256,
        kernel_signature_sha256=kernel_signature_sha256,
        props=props,
        trace_sources=trace_sources,
        sector_stream=sector_stream,
    )
    _emit_divergence_proxies_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha256,
        kernel_signature_sha256=kernel_signature_sha256,
        props=props,
        trace_sources=trace_sources,
        sector_stream=sector_stream,
    )
    _emit_sharing_locality_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha256,
        kernel_signature_sha256=kernel_signature_sha256,
        props=props,
        trace_sources=trace_sources,
    )
    _emit_order_projections_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha256,
        kernel_signature_sha256=kernel_signature_sha256,
        props=props,
        trace_sources=trace_sources,
    )
    _emit_perturbation_run_ledger_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha256,
        kernel_signature_sha256=kernel_signature_sha256,
        props=props,
    )
    _emit_trace_convergence_confidence_for_run_summary(
        artifact_record=artifact_record,
        artifact_dir=artifact_dir,
        traces_folder=traces_folder,
        slug=slug,
        shape_point=shape_point,
        props=props,
        trace_admission=trace_admission,
    )
    trace_sources.verify_unchanged()


class _TraceChildExecutionError(RuntimeError):
    """One bounded trace child failed after synchronous termination/reaping."""

    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        process_id: int | None = None,
        observed_output_bytes_lower_bound: int | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.stdout = stdout
        self.stderr = stderr
        self.process_id = process_id
        if (
            observed_output_bytes_lower_bound is not None
            and (
                type(observed_output_bytes_lower_bound) is not int
                or observed_output_bytes_lower_bound < 0
            )
        ):
            raise ValueError(
                "trace child output-byte lower bound is invalid"
            )
        self.observed_output_bytes_lower_bound = observed_output_bytes_lower_bound
        self.failure_evidence: tuple[dict[str, Any], ...] = ()
        super().__init__(f"{reason}: {detail}")


_TRACE_CAPTURE_RESOURCE_ERRNOS = frozenset({
    errno.EAGAIN,
    errno.EMFILE,
    errno.ENFILE,
    errno.ENOMEM,
    errno.ENOSPC,
})


def _trace_capture_resource_failure(error: BaseException) -> bool:
    """Recognize resource exhaustion across an exception chain."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        if (
            isinstance(current, OSError)
            and current.errno in _TRACE_CAPTURE_RESOURCE_ERRNOS
        ):
            return True
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


def _trace_child_presentation_text(payload: bytes) -> str:
    """Decode child output for diagnostics without changing retained bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("trace child output must remain bytes")
    return payload.decode("utf-8", errors="replace").strip()


def _retain_trace_child_failure_streams(
    directory: Path,
    *,
    role: str,
    stdout: bytes,
    stderr: bytes,
) -> tuple[dict[str, Any], ...]:
    """Publish bounded partial streams in the already-preserved raw stage."""

    evidence: list[dict[str, Any]] = []
    for stream, payload in (("stdout", stdout), ("stderr", stderr)):
        name = f"{role}_failure.{stream}.bin"
        publish_regular_file(directory, name, payload)
        evidence.append({
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        })
    return tuple(evidence)


def _bind_trace_child_failure_evidence(
    error: BaseException,
    *,
    directory: Path,
    role: str,
    stdout: bytes,
    stderr: bytes,
) -> tuple[dict[str, Any], ...]:
    try:
        evidence = _retain_trace_child_failure_streams(
            directory,
            role=role,
            stdout=stdout,
            stderr=stderr,
        )
    except BaseException as preservation_error:
        try:
            error.add_note(
                "partial trace process evidence preservation failed: "
                f"{type(preservation_error).__name__}: {preservation_error}"
            )
        except (AttributeError, TypeError):
            pass
        evidence = ()
    try:
        error.failure_evidence = evidence
    except (AttributeError, TypeError):
        pass
    return evidence


def _run_bounded_trace_child(
    command: tuple[str, ...],
    *,
    environment: dict[str, str],
    working_directory: str,
    inherited_descriptors: tuple[int, ...],
    role: str,
) -> subprocess.CompletedProcess[bytes]:
    """Run one trace child with a wall clock, output cap, and typed cleanup."""

    timeout_code = f"{role}_timeout"
    launch_code = f"{role}_launch_failed"
    try:
        result = _nvbit_process.execute_capture_process(
            command,
            environment=environment,
            working_directory=working_directory,
            inherited_descriptors=inherited_descriptors,
            timeout_seconds=_TRACE_CHILD_TIMEOUT_SECONDS,
            max_output_bytes=_TRACE_CHILD_MAX_STDOUT_STDERR_BYTES,
            timeout_code=timeout_code,
            launch_code=launch_code,
        )
    except _nvbit_process.NvbitCaptureProcessError as exc:
        if exc.process_cleanup_incomplete:
            reason = f"{role}_cleanup_failed"
        elif exc.code == _nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT:
            reason = f"{role}_output_limit"
        elif exc.code == timeout_code:
            reason = timeout_code
        elif _trace_capture_resource_failure(exc):
            reason = f"{role}_resource_unavailable"
        elif exc.code == launch_code:
            reason = launch_code
        else:
            reason = f"{role}_execution_failed"
        raise _TraceChildExecutionError(
            reason,
            exc.detail,
            stdout=exc.stdout,
            stderr=exc.stderr,
            process_id=exc.process_id,
            observed_output_bytes_lower_bound=(
                exc.observed_output_bytes_lower_bound
            ),
        ) from exc
    except (MemoryError, OSError) as exc:
        reason = (
            f"{role}_resource_unavailable"
            if _trace_capture_resource_failure(exc)
            else launch_code
        )
        raise _TraceChildExecutionError(reason, str(exc)) from exc
    return subprocess.CompletedProcess(
        list(command),
        result.returncode,
        result.stdout,
        result.stderr,
    )


def _run_retained_nvbit_capture(
    *,
    retained_executable: RetainedExecutable,
    retained_tracer: RetainedRegular,
    args: list[str],
    traces_folder: Path,
    traces_descriptor: int | None = None,
    runtime_stack: ExitStack | None = None,
) -> dict[str, Any]:
    """Run baseline and NVBit capture through retained source capabilities.

    Canonical callers provide the exclusive staging-directory descriptor and the
    source transaction's ``runtime_stack``. The subprocess receives
    ``/proc/self/fd/<n>`` rather than mutable lexical paths, and the executable,
    tracer, capture root, and runtime DSO descriptors remain retained through
    active publication. Standalone callers omit ``runtime_stack`` and retain the
    runtime only through child completion.
    """
    import time as _time

    if traces_descriptor is not None:
        if type(traces_descriptor) is not int or traces_descriptor < 0:
            raise ImmutableArtifactError(
                "capture staging descriptor must be a nonnegative integer"
            )
        stage_metadata = os.fstat(traces_descriptor)
        if not stat.S_ISDIR(stage_metadata.st_mode):
            raise ImmutableArtifactError(
                "capture staging descriptor is not a directory"
            )
        capture_folder = f"/proc/self/fd/{traces_descriptor}"
        capture_fds = (traces_descriptor,)
    else:
        capture_folder = str(traces_folder)
        capture_fds = ()

    retained_executable.verify_unchanged()
    retained_tracer.verify_unchanged()
    with nullcontext(retained_executable):
        with nullcontext(retained_tracer):
            executable_proc_path = (
                f"/proc/self/fd/{retained_executable.descriptor}"
            )
            capture_base_env = sanitized_capture_environment()
            baseline_env = dict(capture_base_env)
            baseline_env["TRACES_FOLDER"] = capture_folder
            baseline_t0 = _time.perf_counter()
            try:
                baseline_proc = _run_bounded_trace_child(
                    (executable_proc_path, *args),
                    environment=baseline_env,
                    working_directory=str(retained_executable.path.parent),
                    inherited_descriptors=(
                        retained_executable.descriptor,
                        *capture_fds,
                    ),
                    role="trace_baseline_process",
                )
            except _TraceChildExecutionError as exc:
                _bind_trace_child_failure_evidence(
                    exc,
                    directory=traces_folder,
                    role="trace_baseline_process",
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                )
                raise
            baseline_elapsed_s = _time.perf_counter() - baseline_t0
            if baseline_proc.returncode == 0:
                try:
                    baseline_result = json.loads(baseline_proc.stdout)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    baseline_result = None
            else:
                baseline_result = None

            trace_proc = None
            trace_elapsed_s = None
            retained_runtime = None
            if not (
                isinstance(baseline_result, dict)
                and baseline_result.get("result_kind") == "typed_unavailable"
            ):
                trace_proc = None
                owned_runtime_stack = (
                    ExitStack() if runtime_stack is None else None
                )
                selected_runtime_stack = (
                    owned_runtime_stack
                    if owned_runtime_stack is not None
                    else runtime_stack
                )
                assert selected_runtime_stack is not None
                try:
                    retained_runtime = selected_runtime_stack.enter_context(
                        retained_tracer_runtime_env(
                            retained_tracer.path,
                            base_env=capture_base_env,
                            runtime_manifest=default_runtime_manifest_path(
                                retained_tracer.path
                            ),
                            retained_descriptor=retained_tracer.descriptor,
                        )
                    )
                    env = dict(retained_runtime.environment)
                    env["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
                    env["USER_DEFINED_FOLDERS"] = "1"
                    env["TRACES_FOLDER"] = capture_folder
                    trace_t0 = _time.perf_counter()
                    trace_proc = _run_bounded_trace_child(
                        (executable_proc_path, *args),
                        environment=env,
                        working_directory=str(
                            retained_executable.path.parent
                        ),
                        inherited_descriptors=(
                            retained_executable.descriptor,
                            retained_tracer.descriptor,
                            *capture_fds,
                            *retained_runtime.pass_fds,
                        ),
                        role="trace_injected_process",
                    )
                    trace_elapsed_s = _time.perf_counter() - trace_t0
                    retained_runtime.verify_unchanged()
                except _TraceChildExecutionError as exc:
                    _bind_trace_child_failure_evidence(
                        exc,
                        directory=traces_folder,
                        role="trace_injected_process",
                        stdout=exc.stdout,
                        stderr=exc.stderr,
                    )
                    raise
                except TracerRuntimeError as exc:
                    if trace_proc is not None:
                        _bind_trace_child_failure_evidence(
                            exc,
                            directory=traces_folder,
                            role="trace_runtime_verification",
                            stdout=trace_proc.stdout,
                            stderr=trace_proc.stderr,
                        )
                    raise
                finally:
                    if owned_runtime_stack is not None:
                        owned_runtime_stack.close()
                        retained_runtime = None

            process_failure_evidence: tuple[dict[str, Any], ...] = ()
            if trace_proc is not None and trace_proc.returncode != 0:
                process_failure_evidence = _retain_trace_child_failure_streams(
                    traces_folder,
                    role="trace_injected_process",
                    stdout=trace_proc.stdout,
                    stderr=trace_proc.stderr,
                )

            result = {
                "baseline_proc": baseline_proc,
                "baseline_elapsed_s": baseline_elapsed_s,
                "baseline_result": baseline_result,
                "trace_proc": trace_proc,
                "trace_elapsed_s": trace_elapsed_s,
                "binary_sha256": retained_executable.sha256,
                "tracer_build_sha256": retained_tracer.sha256,
                "retained_runtime": retained_runtime,
                "runtime_dso_sha256": (
                    dict(retained_runtime.dso_sha256)
                    if retained_runtime is not None else None
                ),
                "process_failure_evidence": list(process_failure_evidence),
            }
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
    return result


def _capture_raw_trace_generation(
    *,
    artifact_dir: Path,
    shape_point: str,
    attempt_id: str,
    retained_executable: RetainedExecutable,
    retained_tracer: RetainedRegular,
    args: list[str],
    runtime_stack: ExitStack | None = None,
) -> dict[str, Any]:
    """Capture one attempt and commit its raw tree before any admission.

    The returned status is always typed.  Hidden stages and committed trees are
    never removed on failure; callers publish a non-active attempt event before
    returning or advancing to activation.
    """
    generation_root = _trace_generation_root(
        artifact_dir=artifact_dir,
        namespace=RAW_TRACE_GENERATION_NAMESPACE,
        shape_point=shape_point,
    )
    staging_path = generation_root / f".{attempt_id}.staging"
    try:
        stage = create_exclusive_directory_stage(
            generation_root, attempt_id
        )
    except ImmutableArtifactLimitError as exc:
        return {
            "status": "stage_creation_failed",
            "typed_reason": "trace_inventory_limit_exceeded",
            "detail": f"{type(exc).__name__}: {exc}",
            "execution": None,
            "generation": None,
            "staging_path": staging_path,
        }
    except ImmutableArtifactResourceError as exc:
        return {
            "status": "stage_creation_failed",
            "typed_reason": "trace_snapshot_resource_unavailable",
            "detail": f"{type(exc).__name__}: {exc}",
            "execution": None,
            "generation": None,
            "staging_path": staging_path,
        }
    except (ImmutableArtifactError, OSError) as exc:
        return {
            "status": "stage_creation_failed",
            "typed_reason": "trace_filesystem_invalid",
            "detail": f"{type(exc).__name__}: {exc}",
            "execution": None,
            "generation": None,
            "staging_path": staging_path,
        }

    selected_result: dict[str, Any] | None = None

    def select(result: dict[str, Any]) -> dict[str, Any]:
        nonlocal selected_result
        selected_result = result
        return result

    @contextmanager
    def preserve_selected_stage_result():
        entered = stage.__enter__()
        try:
            yield entered
        except BaseException as primary:
            try:
                stage.__exit__(
                    type(primary), primary, primary.__traceback__
                )
            except BaseException as cleanup:
                try:
                    primary.add_note(
                        "raw trace stage cleanup also failed: "
                        f"{type(cleanup).__name__}: {cleanup}"
                    )
                except (AttributeError, TypeError):
                    pass
            raise
        else:
            try:
                stage.__exit__(None, None, None)
            except BaseException as cleanup:
                if selected_result is None:
                    raise
                cleanup_detail = (
                    "raw trace stage cleanup failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
                if selected_result.get("status") == "raw_committed":
                    selected_result["status"] = "raw_stage_closure_failed"
                    selected_result["typed_reason"] = (
                        "trace_snapshot_resource_unavailable"
                        if isinstance(
                            cleanup, ImmutableArtifactResourceError
                        )
                        else "trace_filesystem_invalid"
                    )
                    selected_result["detail"] = cleanup_detail
                else:
                    prior_detail = selected_result.get("detail")
                    selected_result["detail"] = (
                        f"{prior_detail}; {cleanup_detail}"
                        if prior_detail
                        else cleanup_detail
                    )

    with preserve_selected_stage_result():
        try:
            execution = _run_retained_nvbit_capture(
                retained_executable=retained_executable,
                retained_tracer=retained_tracer,
                args=args,
                traces_folder=stage.staging_path,
                traces_descriptor=stage.descriptor,
                runtime_stack=runtime_stack,
            )
        except _TraceChildExecutionError as exc:
            return select({
                "status": "capture_failed",
                "typed_reason": exc.reason,
                "detail": exc.detail,
                "execution": None,
                "generation": None,
                "staging_path": stage.staging_path,
                "process_failure_evidence": list(exc.failure_evidence),
                "observed_output_bytes_lower_bound": (
                    exc.observed_output_bytes_lower_bound
                ),
            })
        except TracerRuntimeError as exc:
            return select({
                "status": "capture_failed",
                "typed_reason": exc.reason,
                "detail": str(exc),
                "execution": None,
                "generation": None,
                "staging_path": stage.staging_path,
                "process_failure_evidence": list(
                    getattr(exc, "failure_evidence", ())
                ),
            })
        except ImmutableArtifactLimitError as exc:
            return select({
                "status": "capture_failed",
                "typed_reason": "trace_inventory_limit_exceeded",
                "detail": f"{type(exc).__name__}: {exc}",
                "execution": None,
                "generation": None,
                "staging_path": stage.staging_path,
            })
        except ImmutableArtifactResourceError as exc:
            return select({
                "status": "capture_failed",
                "typed_reason": "trace_snapshot_resource_unavailable",
                "detail": f"{type(exc).__name__}: {exc}",
                "execution": None,
                "generation": None,
                "staging_path": stage.staging_path,
            })
        except (ImmutableArtifactError, OSError) as exc:
            return select({
                "status": "capture_failed",
                "typed_reason": "trace_input_identity_invalid",
                "detail": f"{type(exc).__name__}: {exc}",
                "execution": None,
                "generation": None,
                "staging_path": stage.staging_path,
            })

        baseline_result = execution["baseline_result"]
        if (
            isinstance(baseline_result, dict)
            and baseline_result.get("result_kind") == "typed_unavailable"
        ):
            return select({
                "status": "typed_unavailable",
                "typed_reason": str(
                    baseline_result.get("typed_reason")
                    or "typed_unavailable_without_reason"
                ),
                "detail": "baseline workload refused the trace launch",
                "execution": execution,
                "generation": None,
                "staging_path": stage.staging_path,
            })

        trace_process = execution["trace_proc"]
        if trace_process is None:
            return select({
                "status": "capture_failed",
                "typed_reason": "trace_execution_missing",
                "detail": "retained NVBit run did not execute",
                "execution": execution,
                "generation": None,
                "staging_path": stage.staging_path,
            })
        if trace_process.returncode != 0:
            return select({
                "status": "capture_failed",
                "typed_reason": "trace_process_failed",
                "detail": (
                    f"{retained_executable.path} under tracer "
                    f"rc={trace_process.returncode}: "
                    f"{_trace_child_presentation_text(trace_process.stderr)}"
                ),
                "execution": execution,
                "generation": None,
                "staging_path": stage.staging_path,
                "process_failure_evidence": execution.get(
                    "process_failure_evidence", []
                ),
            })

        generation: ImmutableDirectoryGeneration | None = None

        def raw_commit_failure(
            *,
            typed_reason: str,
            exc: BaseException,
        ) -> dict[str, Any]:
            nonlocal generation
            recovery_detail: str | None = None
            if generation is None and stage.committed:
                try:
                    generation = inspect_immutable_directory_generation(
                        stage.final_path,
                        require_nonempty=True,
                        **_trace_generation_commit_limits(),
                    )
                except (ImmutableArtifactError, OSError) as recovery_exc:
                    recovery_detail = (
                        f"; committed generation recovery failed: "
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
            return {
                "status": "raw_commit_failed",
                "typed_reason": typed_reason,
                "detail": (
                    f"{type(exc).__name__}: {exc}{recovery_detail or ''}"
                ),
                "execution": execution,
                "generation": generation,
                "staging_path": (
                    None if stage.committed else stage.staging_path
                ),
                "unresolved_committed_path": (
                    stage.final_path
                    if stage.committed and generation is None
                    else None
                ),
            }

        try:
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
            generation = stage.commit(
                require_nonempty=True,
                **_trace_generation_commit_limits(),
            )
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
        except ImmutableArtifactLimitError as exc:
            return select(raw_commit_failure(
                typed_reason="trace_inventory_limit_exceeded",
                exc=exc,
            ))
        except ImmutableArtifactResourceError as exc:
            return select(raw_commit_failure(
                typed_reason="trace_snapshot_resource_unavailable",
                exc=exc,
            ))
        except (ImmutableArtifactError, OSError) as exc:
            return select(raw_commit_failure(
                typed_reason="trace_filesystem_invalid",
                exc=exc,
            ))

        return select({
            "status": "raw_committed",
            "typed_reason": None,
            "detail": None,
            "execution": execution,
            "generation": generation,
            "staging_path": None,
        })


def _phase_evidence_identities(
    *,
    artifact_dir: Path,
    shape_point_name: str,
    expected_repeat_count: int,
    expected_launch_indices: tuple[int, ...],
    phase_requested: bool,
    generation: ImmutableDirectoryGeneration,
) -> list[dict[str, Any]]:
    """Validate and identify one exact committed phase-ledger generation."""
    if phase_requested is not True:
        raise ImmutableArtifactError(
            "optional phase evidence cannot acquire credit when requested is "
            "not exact true"
        )
    import phase_inject_orchestrator as _pinj  # type: ignore[import-not-found]

    try:
        _pinj._validated_repeat_and_launch_domain(
            expected_repeat_count,
            expected_launch_indices,
        )
    except (
        _pinj.PhaseInjectAbRepeatCountInvalid,
        _pinj.PhaseInjectLaunchAuthorityInvalid,
        _pinj.PhaseInjectResourceLimitExceeded,
    ) as exc:
        raise ImmutableArtifactError(
            f"optional phase expected repeat/launch authority is invalid: {exc}"
        ) from exc
    observed = inspect_immutable_directory_generation(
        generation.path,
        require_nonempty=True,
        max_files=generation.file_count,
        max_directories=generation.directory_count,
        max_total_file_bytes=generation.total_file_bytes,
        max_total_relative_path_bytes=4 * 1024 * 1024,
    )
    if observed != generation:
        raise ImmutableArtifactError(
            "optional phase ledger generation changed before activation"
        )
    artifact_root = Path(os.path.abspath(os.fspath(artifact_dir)))
    try:
        generation_relative = generation.path.relative_to(artifact_root)
    except ValueError as exc:
        raise ImmutableArtifactError(
            "optional phase ledger generation escapes artifact directory"
        ) from exc

    expected = {
        identity.relative_path: identity
        for identity in generation.files
    }
    identities: list[dict[str, Any]] = []
    observed_launches: dict[int, set[int]] = {}
    observed_keys: set[tuple[int, int]] = set()
    with PinnedDirectoryReader(generation.path) as reader:
        snapshot = reader.snapshot_tree(
            require_readonly_files=True,
            require_readonly_directories=True,
            max_files=generation.file_count,
            max_directories=generation.directory_count,
            max_total_file_bytes=generation.total_file_bytes,
            max_total_relative_path_bytes=4 * 1024 * 1024,
        )
        files = dict(snapshot.files)
        directories = dict(snapshot.directories)
        if set(directories) != {"."} or set(files) != set(expected):
            raise ImmutableArtifactError(
                "optional phase ledger generation inventory mismatch"
            )
        for filename, metadata in sorted(files.items()):
            match = re.fullmatch(
                r"phase_cycle_ledger_([0-9]+)_([0-9]+)\.json",
                filename,
            )
            if match is None:
                raise ImmutableArtifactError(
                    f"unexpected optional phase evidence file: {filename}"
                )
            repeat_text, launch_text = match.groups()
            try:
                repeat_index = int(repeat_text)
                launch_index = int(launch_text)
            except ValueError as exc:
                raise ImmutableArtifactError(
                    "optional phase evidence filename index is invalid: "
                    f"{filename}"
                ) from exc
            if (
                str(repeat_index) != repeat_text
                or str(launch_index) != launch_text
            ):
                raise ImmutableArtifactError(
                    "optional phase evidence filename indices are not canonical: "
                    f"{filename}"
                )
            identity_key = (repeat_index, launch_index)
            if identity_key in observed_keys:
                raise ImmutableArtifactError(
                    "duplicate normalized optional phase evidence identity: "
                    f"repeat={repeat_index} launch={launch_index}"
                )
            observed_keys.add(identity_key)
            if not stat.S_ISREG(metadata[2]) or metadata[3] != 1:
                raise ImmutableArtifactError(
                    "optional phase evidence is not a single-link regular file: "
                    f"{filename}"
                )
            payload = reader.read_snapshot(
                filename, expected_metadata=metadata
            ).payload
            identity = expected[filename]
            if (
                hashlib.sha256(payload).hexdigest() != identity.sha256
                or len(payload) != identity.size_bytes
            ):
                raise ImmutableArtifactError(
                    f"optional phase evidence seal mismatch: {filename}"
                )
            try:
                record = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ImmutableArtifactError(
                    f"optional phase evidence is invalid JSON: {filename}: {exc}"
                ) from exc
            if (
                type(record) is not dict
                or record.get("shape_point") != shape_point_name
                or type(record.get("repeat_index")) is not int
                or record.get("repeat_index") != repeat_index
                or type(record.get("launch_index")) is not int
                or record.get("launch_index") != launch_index
            ):
                raise ImmutableArtifactError(
                    "optional phase evidence identity disagrees with its trace "
                    f"or filename: {filename}"
                )
            observed_launches.setdefault(repeat_index, set()).add(launch_index)
            identities.append({
                "path": (generation_relative / filename).as_posix(),
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            })
        reader.verify_path()
    expected_repeats = set(range(expected_repeat_count))
    expected_launches = set(expected_launch_indices)
    if (
        set(observed_launches) != expected_repeats
        or any(
            launches != expected_launches
            for launches in observed_launches.values()
        )
    ):
        observed_coverage = {
            repeat_index: sorted(launches)
            for repeat_index, launches in sorted(observed_launches.items())
        }
        raise ImmutableArtifactError(
            "optional phase evidence coverage does not equal the admitted "
            "repeat-by-launch authority: "
            f"expected_repeats={sorted(expected_repeats)} "
            f"expected_launches={sorted(expected_launches)} "
            f"observed={observed_coverage}"
        )
    return identities


def handle_trace(row: dict[str, Any], props: DeviceProps, out_dir: Path,
                 shape_point_name: str | None = None,
                 phase_inject_ab: bool = False,
                 ab_repeat_count: int = 3,
                 tracer_source_path: Path | str | None = None,
                 tracer_sha256: str | None = None) -> int:
    if type(phase_inject_ab) is not bool:
        print(
            "phase_inject_request_invalid: phase_inject_ab must be an exact "
            f"boolean; got {phase_inject_ab!r}",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if phase_inject_ab:
        import phase_inject_orchestrator as _pinj  # type: ignore[import-not-found]

        if (
            type(ab_repeat_count) is not int
            or ab_repeat_count <= 0
            or ab_repeat_count > _pinj.PHASE_AB_MAX_REPEAT_COUNT
        ):
            print(
                "phase_inject_ab_repeat_count_invalid: "
                "ab_repeat_count must be an exact strictly positive integer "
                f"no greater than {_pinj.PHASE_AB_MAX_REPEAT_COUNT}; "
                f"got {ab_repeat_count!r}",
                file=sys.stderr,
            )
            return EXIT_USAGE
    try:
        tracer_authority = require_external_tracer_authority(
            tracer_source_path,
            tracer_sha256,
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        print(f"external_tracer_authority_invalid: {exc}", file=sys.stderr)
        return EXIT_USAGE
    source_stack = ExitStack()
    try:
        result = _handle_trace_with_sources(
            row,
            props,
            out_dir,
            shape_point_name=shape_point_name,
            phase_inject_ab=phase_inject_ab,
            ab_repeat_count=ab_repeat_count,
            tracer_authority=tracer_authority,
            source_stack=source_stack,
        )
    except BaseException as primary:
        try:
            source_stack.__exit__(
                type(primary), primary, primary.__traceback__
            )
        except BaseException as cleanup:
            try:
                primary.add_note(
                    "trace source cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    if result != EXIT_OK:
        # The typed failure has already won.  Retained capabilities still close
        # and attempt revalidation, but a cleanup-side inotify/resource error must
        # not replace the selected EXIT_* boundary result.
        failure = RuntimeError("trace source transaction already failed")
        try:
            source_stack.__exit__(type(failure), failure, None)
        except BaseException as cleanup_exc:
            print(
                "trace_source_cleanup_failed_after_typed_result: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
            )
        return result
    source_stack.close()
    return result


def _handle_trace_with_sources(
    row: dict[str, Any],
    props: DeviceProps,
    out_dir: Path,
    *,
    shape_point_name: str | None,
    phase_inject_ab: bool,
    ab_repeat_count: int,
    tracer_authority: ExternalTracerAuthority,
    source_stack: ExitStack,
) -> int:
    slug = row["slug"]
    shape_path = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not shape_path.exists():
        print(f"run_suite: trace for {slug} requires shapes/{slug}.toml (missing)", file=sys.stderr)
        return EXIT_UNSUPPORTED
    with open(shape_path, "rb") as f:
        shape = tomllib.load(f)
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
    artifact_path = artifact_dir / f"trace_{shape_point['name']}.json"
    # Task238 activation invariant: once a concrete per-shape rerun is selected,
    # retire its previous activation leaf before any preflight sidecar, capture
    # directory, subprocess, or feature output can mutate.  The no-follow
    # primitive preserves the exact old inode under a hidden history name.  A
    # crash after this point therefore leaves either no canonical activation or
    # a later, fully committed replacement; stale success can never survive.
    try:
        ensure_directory(artifact_dir)
        # Establish a persistent authority surface before retiring any legacy
        # activation. If this attempt exits before a replacement leaf is ready,
        # replay must refuse flat-summary fallback rather than reactivate stale V1.
        ensure_directory(artifact_dir / TRACE_CAPTURE_ATTEMPT_NAMESPACE)
        retire_regular_file(artifact_path)
    except ImmutableArtifactResourceError as exc:
        print(
            "trace_activation_retirement_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except (ImmutableArtifactError, OSError) as exc:
        print(
            "trace_activation_retirement_invalid: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED

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
    budget_bytes = row["trace_budget_mib"] * 1024 * 1024
    stale = [e for e in estimator_errors if "trace_prediction_stale" in e]
    overbudget = [e for e in estimator_errors if "trace_shape_overbudget" in e]
    if stale:
        for e in stale:
            print(e, file=sys.stderr)
        return EXIT_UNSUPPORTED
    if overbudget:
        sidecar = out_dir / "trace_volume_exceeded.json"
        # Pick the worst offender for the sidecar
        worst_point = None
        worst_bytes = 0
        for point in shape.get("trace_shapes", []):
            pred = int(point.get("predicted_bytes") or 0)
            if pred > worst_bytes:
                worst_bytes = pred
                worst_point = point.get("name", "<?>")
        _write_trace_volume_exceeded(
            sidecar=sidecar,
            slug=slug,
            shape_point=worst_point or "<?>",
            budget_bytes=budget_bytes,
            actual_trace_bytes=worst_bytes,
            actual_wall_time_s=0.0,
        )
        for e in overbudget:
            print(e, file=sys.stderr)
        print(f"trace_shape_overbudget: sidecar at {sidecar}", file=sys.stderr)
        return EXIT_SAMPLING_REQUIRED

    # Capture only through the explicit external tracer authority validated by
    # handle_trace before any output retirement or other mutation.
    tracer_so = tracer_authority.path
    executable = _find_kernel_executable(slug)
    if executable is None:
        print(f"executable_missing: {slug} has no {slug}_run executable (build the kernel first)", file=sys.stderr)
        return EXIT_UNSUPPORTED

    try:
        retained_executable = source_stack.enter_context(
            open_retained_executable(executable)
        )
        retained_tracer = source_stack.enter_context(
            open_retained_regular(
                tracer_so,
                max_bytes=EXTERNAL_TRACER_MAX_BYTES,
            )
        )
        verify_retained_tracer_authority(retained_tracer, tracer_authority)
    except ImmutableArtifactResourceError as exc:
        print(
            "trace_input_identity_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except (ImmutableArtifactError, OSError, ValueError) as exc:
        print(
            "trace_input_identity_invalid: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED

    attempt_id = _new_trace_attempt_id()
    args = _kernel_args_from_shape(shape_point, row)
    kernel_signature_sha256 = _kernel_signature_sha256(
        retained_executable.path, slug
    )
    capture_result = _capture_raw_trace_generation(
        artifact_dir=artifact_dir,
        shape_point=shape_point["name"],
        attempt_id=attempt_id,
        retained_executable=retained_executable,
        retained_tracer=retained_tracer,
        args=args,
        runtime_stack=source_stack,
    )
    capture_status = capture_result["status"]
    execution = capture_result["execution"]
    raw_generation = capture_result["generation"]
    raw_staging_path = capture_result["staging_path"]
    raw_unresolved_committed_path = capture_result.get(
        "unresolved_committed_path"
    )

    if capture_status not in {"raw_committed", "typed_unavailable"}:
        failed_raw_generation_record = None
        if isinstance(raw_generation, ImmutableDirectoryGeneration):
            failed_raw_generation_record = _directory_generation_record(
                generation=raw_generation,
                artifact_dir=artifact_dir,
                attempt_id=attempt_id,
            )
        try:
            _publish_trace_attempt_event(
                artifact_dir=artifact_dir,
                slug=slug,
                shape_point=shape_point["name"],
                attempt_id=attempt_id,
                event_name=(
                    "stage_creation_failed"
                    if capture_status == "stage_creation_failed"
                    else (
                        "raw_commit_failed"
                        if capture_status in {
                            "raw_commit_failed",
                            "raw_stage_closure_failed",
                        }
                        else "capture_failed"
                    )
                ),
                status=capture_status,
                typed_reason=capture_result["typed_reason"],
                raw_trace_generation=failed_raw_generation_record,
                raw_staging_path=(
                    None
                    if failed_raw_generation_record is not None
                    else raw_staging_path
                ),
                raw_unresolved_committed_path=(
                    None
                    if failed_raw_generation_record is not None
                    else raw_unresolved_committed_path
                ),
                process_failure_evidence=capture_result.get(
                    "process_failure_evidence"
                ),
                observed_output_bytes_lower_bound=capture_result.get(
                    "observed_output_bytes_lower_bound"
                ),
            )
        except (ImmutableArtifactError, OSError) as event_exc:
            print(
                "trace_attempt_event_publication_failed: "
                f"{type(event_exc).__name__}: {event_exc}",
                file=sys.stderr,
            )
        print(
            f"{capture_result['typed_reason']}: {capture_result['detail']}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED

    assert execution is not None

    baseline_proc = execution["baseline_proc"]
    baseline_result = execution["baseline_result"]
    if baseline_proc.returncode == 0:
        reported = _stdout_wall_time_s(baseline_proc.stdout)
        baseline_wall_time_s: float | None = (
            reported
            if reported is not None
            else execution["baseline_elapsed_s"]
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
    if capture_status == "typed_unavailable":
        assert isinstance(baseline_result, dict)
        typed_reason = capture_result["typed_reason"]
        tombstone_committed = False
        tombstone_payload: bytes | None = None

        def retire_failed_tombstone() -> None:
            nonlocal tombstone_committed
            if not tombstone_committed:
                return
            if tombstone_payload is None:
                raise ImmutableArtifactError(
                    "committed trace tombstone has no retained locator payload"
                )
            try:
                retire_regular_file(
                    artifact_path,
                    expected_payload=tombstone_payload,
                )
            except (ImmutableArtifactError, OSError) as retirement_error:
                print(
                    "trace_failed_tombstone_retirement_failed: "
                    f"{type(retirement_error).__name__}: {retirement_error}",
                    file=sys.stderr,
                )
            else:
                tombstone_committed = False

        manifest = build_measurement_manifest(
            slug=slug, mode="trace",
            warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
            props=props, binary_path=retained_executable.path,
            compiler_flags=_gencode_flags_for_arch_gate(row.get("arch_gate", [])),
            shape_point=shape_point["name"],
            binary_sha256=execution["binary_sha256"],
        )
        try:
            manifest_path = _emit_manifest(
                manifest,
                artifact_dir,
                filename=f"measurement_manifest_{shape_point['name']}.json",
                preserve_previous=True,
            )
            manifest_identity = _retained_artifact_identity(
                path=manifest_path, artifact_dir=artifact_dir
            )
            attempt_event = _publish_trace_attempt_event(
                artifact_dir=artifact_dir,
                slug=slug,
                shape_point=shape_point["name"],
                attempt_id=attempt_id,
                event_name="typed_unavailable",
                status="typed_unavailable",
                typed_reason=str(typed_reason),
                raw_staging_path=raw_staging_path,
                measurement_manifest=manifest_identity,
            )
            typed_trace_record = dict(baseline_result)
            typed_trace_record.update({
                "slug": slug,
                "shape_set": "trace_shapes",
                "shape_point": shape_point["name"],
                "activation_state": "non_active",
                "typed_tombstone": True,
                "trace_capture_attempt_id": attempt_id,
                "trace_capture_attempt_event": attempt_event,
                "raw_trace_staging_path": (
                    Path(os.path.abspath(os.fspath(raw_staging_path)))
                    .relative_to(
                        Path(os.path.abspath(os.fspath(artifact_dir)))
                    )
                    .as_posix()
                    if raw_staging_path is not None else None
                ),
                "binary_sha256": execution["binary_sha256"],
                "tracer_so_path": str(retained_tracer.path),
                "tracer_source_kind": tracer_authority.source_kind,
                "nvbit_tracer_build_hash": execution["tracer_build_sha256"],
                "kernel_signature_sha256": kernel_signature_sha256,
                "source_commit": _resolve_source_commit(),
                "measurement_manifest": manifest_identity,
                "nvbit_trace_summary_emit_status": "typed_unavailable",
                "nvbit_trace_summary_paths": [],
                "nvbit_trace_summary_failure_reason": str(typed_reason),
            })
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
            tombstone_payload = _trace_artifact_record_payload(
                typed_trace_record
            )
            _publish_trace_artifact_record(artifact_path, typed_trace_record)
            tombstone_committed = True
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
            source_stack.pop_all().close()
        except ImmutableArtifactResourceError as exc:
            retire_failed_tombstone()
            print(
                "trace_tombstone_resource_unavailable: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return EXIT_BUILD_FAILED
        except (ImmutableArtifactError, OSError) as exc:
            retire_failed_tombstone()
            print(
                "trace_tombstone_publication_failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return EXIT_BUILD_FAILED
        print(
            f"run_suite: trace for {slug}: typed_unavailable "
            f"({typed_reason}); non-active tombstone at {artifact_path}"
        )
        return EXIT_OK

    assert isinstance(raw_generation, ImmutableDirectoryGeneration)
    raw_generation_record = _directory_generation_record(
        generation=raw_generation,
        artifact_dir=artifact_dir,
        attempt_id=attempt_id,
    )

    def publish_committed_failure_event(
        *,
        event_name: str,
        status: str,
        typed_reason: str,
        derived_generation: dict[str, Any] | None = None,
        derived_staging_path: Path | None = None,
        phase_ledger_generation: dict[str, Any] | None = None,
        phase_ledger_staging_path: Path | None = None,
        summary_generation: dict[str, Any] | None = None,
        measurement_manifest: dict[str, Any] | None = None,
        observed_output_bytes_lower_bound: int | None = None,
    ) -> None:
        if typed_reason == "trace_snapshot_resource_unavailable":
            # The publication primitive depends on the same bounded no-follow
            # verifier resource that just failed.  Do not loop or misreport a
            # second event as durable; the committed raw tree / hidden derived
            # stage and the typed stderr result remain preserved.
            print(
                "trace_attempt_event_deferred: "
                "trace_snapshot_resource_unavailable",
                file=sys.stderr,
            )
            return
        try:
            _publish_trace_attempt_event(
                artifact_dir=artifact_dir,
                slug=slug,
                shape_point=shape_point["name"],
                attempt_id=attempt_id,
                event_name=event_name,
                status=status,
                typed_reason=typed_reason,
                raw_trace_generation=raw_generation_record,
                derived_generation=derived_generation,
                derived_staging_path=derived_staging_path,
                phase_ledger_generation=phase_ledger_generation,
                phase_ledger_staging_path=phase_ledger_staging_path,
                summary_generation=summary_generation,
                measurement_manifest=measurement_manifest,
                observed_output_bytes_lower_bound=(
                    observed_output_bytes_lower_bound
                ),
            )
        except (ImmutableArtifactError, OSError) as event_exc:
            print(
                "trace_attempt_event_publication_failed: "
                f"{type(event_exc).__name__}: {event_exc}",
                file=sys.stderr,
            )

    traces_folder = raw_generation.path
    proc = execution["trace_proc"]
    if proc is None:
        print(
            "trace_execution_missing: retained NVBit run did not execute",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    if proc.returncode != 0:
        print(
            f"build_failed: {executable} under tracer rc={proc.returncode}: "
            f"{_trace_child_presentation_text(proc.stderr)}",
            file=sys.stderr,
        )
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
        nvbit_reported
        if nvbit_reported is not None
        else execution["trace_elapsed_s"]
    )
    trace_pb = traces_folder / "dynamic_trace.pb"

    try:
        trace_admission = require_trace_admission(traces_folder)
    except TraceAdmissionError as exc:
        publish_committed_failure_event(
            event_name="admission_failed",
            status="admission_failed",
            typed_reason=exc.typed_reason,
        )
        print(
            f"trace_capture_not_admitted: {exc.typed_reason}: {exc.detail}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except ImmutableArtifactResourceError as exc:
        publish_committed_failure_event(
            event_name="admission_failed",
            status="admission_failed",
            typed_reason="trace_snapshot_resource_unavailable",
        )
        print(
            "trace_capture_not_admitted: trace_snapshot_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except (ImmutableArtifactError, OSError) as exc:
        publish_committed_failure_event(
            event_name="admission_failed",
            status="admission_failed",
            typed_reason="trace_filesystem_invalid",
        )
        print(
            "trace_capture_not_admitted: trace_filesystem_invalid: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    raw_generation_record.update({
        "trace_admission_sha256": trace_admission.trace_admission_sha256,
        "trace_filesystem_generation_sha256": (
            trace_admission.trace_filesystem_generation_sha256
        ),
        "dynamic_trace_sha256": trace_admission.dynamic_trace_sha256,
    })
    trace_bytes = trace_admission.dynamic_trace_size_bytes
    sidecar = artifact_dir / "trace_volume_exceeded.json"
    if _admitted_trace_exceeds_budget(
        trace_admission=trace_admission,
        budget_bytes=budget_bytes,
        sidecar=sidecar,
        slug=slug,
        shape_point=shape_point["name"],
        actual_wall_time_s=actual_wall_time_s,
    ):
        print(
            "trace_shape_overbudget: admitted snapshot "
            f"trace_bytes={trace_bytes} > budget_bytes={budget_bytes}",
            file=sys.stderr,
        )
        print(f"trace_shape_overbudget: sidecar at {sidecar}", file=sys.stderr)
        publish_committed_failure_event(
            event_name="budget_rejected",
            status="budget_rejected",
            typed_reason="trace_shape_overbudget",
        )
        return EXIT_SAMPLING_REQUIRED

    derived_generation_root = _trace_generation_root(
        artifact_dir=artifact_dir,
        namespace=TRACE_DERIVED_GENERATION_NAMESPACE,
        shape_point=shape_point["name"],
    )
    try:
        derived_stage = create_exclusive_directory_stage(
            derived_generation_root, attempt_id
        )
    except ImmutableArtifactLimitError as exc:
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status="derived_stage_creation_failed",
            typed_reason="trace_inventory_limit_exceeded",
        )
        print(
            "trace_derived_stage_limit_exceeded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except ImmutableArtifactResourceError as exc:
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status="derived_stage_creation_failed",
            typed_reason="trace_snapshot_resource_unavailable",
        )
        print(
            "trace_derived_stage_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except (ImmutableArtifactError, OSError) as exc:
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status="derived_stage_creation_failed",
            typed_reason="trace_filesystem_invalid",
        )
        print(
            "trace_derived_stage_invalid: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    feature_staging_folder = derived_stage.staging_path

    def close_derived_stage_after_typed_result() -> None:
        try:
            derived_stage.close()
        except BaseException as cleanup_exc:
            print(
                "trace_derived_stage_cleanup_failed_after_typed_result: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
            )

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
    try:
        budget_sidecars = _emit_nvbit_budget_sidecars(
            output_root=feature_staging_folder,
            artifact_dir=artifact_dir,
            slug=slug,
            shape_point=shape_point["name"],
            actual_slowdown_ratio=actual_slowdown_ratio,
            baseline_wall_time_s=baseline_wall_time_s,
            actual_wall_time_s=actual_wall_time_s,
            actual_drop_rate=actual_drop_rate,
            expected_record_count=expected_record_count,
            admitted_record_count=admitted_record_count,
        )
    except ImmutableArtifactLimitError as exc:
        close_derived_stage_after_typed_result()
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status="budget_sidecar_publication_failed",
            typed_reason="trace_inventory_limit_exceeded",
            derived_staging_path=feature_staging_folder,
        )
        print(
            "trace_budget_sidecar_limit_exceeded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except ImmutableArtifactResourceError as exc:
        close_derived_stage_after_typed_result()
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status="budget_sidecar_publication_failed",
            typed_reason="trace_snapshot_resource_unavailable",
            derived_staging_path=feature_staging_folder,
        )
        print(
            "trace_budget_sidecar_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except (ImmutableArtifactError, OSError) as exc:
        close_derived_stage_after_typed_result()
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status="budget_sidecar_publication_failed",
            typed_reason="trace_filesystem_invalid",
            derived_staging_path=feature_staging_folder,
        )
        print(
            "trace_budget_sidecar_publication_failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED

    tracer_so_serialized = str(retained_tracer.path)

    artifact_record: dict[str, Any] = {
        "schema_version": 1,
        "slug": slug,
        "shape_set": "trace_shapes",
        "shape_point": shape_point["name"],
        "trace_capture_attempt_id": attempt_id,
        "raw_trace_generation": raw_generation_record,
        "trace_pb_path": str(trace_pb.relative_to(artifact_dir)),
        "trace_bytes": trace_bytes,
        "wall_time_s": actual_wall_time_s,
        "tracer_so_path": tracer_so_serialized,
        "tracer_source_kind": tracer_authority.source_kind,
        "binary_sha256": execution["binary_sha256"],
        "nvbit_tracer_build_hash": execution["tracer_build_sha256"],
        "nvbit_runtime_dso_sha256": execution.get("runtime_dso_sha256"),
        "trace_contract_id": trace_admission.trace_contract_id,
        "memory_payload_classes": list(
            trace_admission.memory_payload_classes
        ),
        "kernel_signature_sha256": kernel_signature_sha256,
        "source_commit": _resolve_source_commit(),
        # The portable seal intentionally excludes inode/local-generation
        # metadata.  The second field scopes the immutable summary generation
        # on this host without contaminating the portable admission identity.
        "trace_admission_sha256": trace_admission.trace_admission_sha256,
        "trace_filesystem_generation_sha256": (
            trace_admission.trace_filesystem_generation_sha256
        ),
        "dynamic_trace_sha256": trace_admission.dynamic_trace_sha256,
        "stats_sha256": trace_admission.stats_sha256,
        "stats_size_bytes": trace_admission.stats_size_bytes,
        "threadblock_payload_manifest_sha256": (
            trace_admission.threadblock_payload_manifest_sha256
        ),
        "threadblock_payload_size_bytes": (
            trace_admission.threadblock_payload_size_bytes
        ),
        "allocations_sidecar_sha256": (
            trace_admission.allocations_sidecar_sha256
        ),
        "allocations_sidecar_size_bytes": (
            trace_admission.allocations_sidecar_size_bytes
        ),
    }
    for sidecar_kind, sidecar_path in budget_sidecars.items():
        field = (
            "nvbit_slowdown_budget_sidecar_path"
            if sidecar_kind == "slowdown"
            else "nvbit_drop_rate_sidecar_path"
        )
        artifact_record[field] = (
            Path(os.path.abspath(os.fspath(sidecar_path)))
            .relative_to(Path(os.path.abspath(os.fspath(artifact_dir))))
            .as_posix()
            if sidecar_path is not None else None
        )
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

    # Every producer, publication, and final source-generation gate runs before
    # the canonical trace leaf is created.  Keep Task235's admitted-source
    # capability open across the entire pre-activation pipeline so a mutation
    # during summary, manifest, or optional phase work is still observable at
    # the final gate.
    preactivation_stage = "trace_capability_open"
    derived_generation_record: dict[str, Any] | None = None
    phase_ledger_generation: ImmutableDirectoryGeneration | None = None
    phase_ledger_generation_record: dict[str, Any] | None = None
    phase_ledger_staging_path = (
        _trace_generation_root(
            artifact_dir=artifact_dir,
            namespace=PHASE_LEDGER_GENERATION_NAMESPACE,
            shape_point=shape_point["name"],
        )
        / f".{attempt_id}.staging"
        if phase_inject_ab
        else None
    )
    feature_evidence: list[dict[str, Any]] = []
    summary_generation: dict[str, Any] | None = None
    manifest_identity: dict[str, Any] | None = None
    retained_manifest: RetainedRegular | None = None
    expected_activation_roles = {
        "raw_generation",
        "derived_generation",
        "summary_generation",
        "measurement_manifest",
        "activation_ready_event",
        "executable",
        "tracer",
        "tracer_runtime",
        "admitted_trace_sources",
    }
    if phase_inject_ab:
        expected_activation_roles.add("phase_generation")
    activation_authorities = _TraceActivationAuthoritySet(
        expected_activation_roles
    )
    activation_rollback_armed = False
    activation_payload: bytes | None = None
    activation_authority_close_error: BaseException | None = None
    derived_stage_close_error: BaseException | None = None

    def retire_failed_activation(
        primary: BaseException | None = None,
    ) -> bool:
        nonlocal activation_rollback_armed
        if not activation_rollback_armed:
            return True
        if activation_payload is None:
            missing_payload_error = ImmutableArtifactError(
                "armed trace activation has no retained locator payload"
            )
            if primary is None:
                raise missing_payload_error
            try:
                primary.add_note(str(missing_payload_error))
            except (AttributeError, TypeError):
                pass
            return False
        try:
            retire_regular_file(
                artifact_path,
                expected_payload=activation_payload,
            )
        except BaseException as retirement_error:
            if primary is not None:
                try:
                    primary.add_note(
                        "failed activation retirement also failed: "
                        f"{type(retirement_error).__name__}: "
                        f"{retirement_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            print(
                "trace_failed_activation_retirement_failed: "
                f"{type(retirement_error).__name__}: {retirement_error}",
                file=sys.stderr,
            )
            return False
        activation_rollback_armed = False
        return True

    try:
        with open_admitted_trace_sources(
            traces_folder,
            expected_admission=trace_admission,
        ) as trace_sources:
            preactivation_stage = "features"
            _emit_admitted_trace_features(
                artifact_record=artifact_record,
                artifact_dir=artifact_dir,
                traces_folder=feature_staging_folder,
                trace_admission=trace_admission,
                slug=slug,
                shape_point=shape_point,
                binary_sha256=execution["binary_sha256"],
                kernel_signature_sha256=kernel_signature_sha256,
                props=props,
                trace_sources=trace_sources,
            )

            # A/B trace byproducts remain inside the hidden derived stage.  The
            # admitted phase ledgers commit as their own attempt-scoped immutable
            # generation and become authoritative only through the final active
            # trace record.
            phase_evidence: list[dict[str, Any]] = []
            if phase_inject_ab:
                preactivation_stage = "optional_phase"
                phase_result = _run_phase_inject_ab_for_handle_trace(
                    slug=slug,
                    shape_point_name=shape_point["name"],
                    attempt_id=attempt_id,
                    retained_executable=retained_executable,
                    args=args,
                    retained_tracer=retained_tracer,
                    traces_folder=feature_staging_folder,
                    artifact_dir=artifact_dir,
                    ab_repeat_count=ab_repeat_count,
                    expected_launch_indices=tuple(
                        launch.launch_index
                        for launch in trace_admission.launches
                    ),
                    row=row,
                    binary_sha256=execution["binary_sha256"],
                    kernel_signature_sha256=kernel_signature_sha256,
                    tracer_build_sha256=execution["tracer_build_sha256"],
                )
                rc_ab = int(phase_result["return_code"])
                if rc_ab == EXIT_OK:
                    phase_admission_error = _phase_summary_admission_error(
                        phase_result.get("summary"),
                        expected_ledger_count=(
                            ab_repeat_count * len(trace_admission.launches)
                        ),
                    )
                    if phase_admission_error is not None:
                        print(
                            "phase_cycle_ledger_inadmissible: "
                            + phase_admission_error,
                            file=sys.stderr,
                        )
                        phase_result = {
                            **phase_result,
                            "return_code": EXIT_BUILD_FAILED,
                            "typed_reason": "phase_cycle_ledger_inadmissible",
                        }
                        rc_ab = EXIT_BUILD_FAILED
                if rc_ab != EXIT_OK:
                    failed_phase_generation = phase_result.get("generation")
                    if isinstance(
                        failed_phase_generation, ImmutableDirectoryGeneration
                    ):
                        phase_ledger_generation = failed_phase_generation
                        phase_ledger_generation_record = (
                            _directory_generation_record(
                                generation=failed_phase_generation,
                                artifact_dir=artifact_dir,
                                attempt_id=attempt_id,
                            )
                        )
                    publish_committed_failure_event(
                        event_name="preactivation_failed",
                        status="optional_phase_failed",
                        typed_reason=str(
                            phase_result.get("typed_reason")
                            or "optional_phase_failed"
                        ),
                        derived_staging_path=feature_staging_folder,
                        phase_ledger_generation=(
                            phase_ledger_generation_record
                        ),
                        phase_ledger_staging_path=(
                            None
                            if phase_ledger_generation_record is not None
                            else phase_result.get("staging_path")
                        ),
                        observed_output_bytes_lower_bound=phase_result.get(
                            "observed_output_bytes_lower_bound"
                        ),
                    )
                    return rc_ab
                phase_ledger_generation = phase_result.get("generation")
                if not isinstance(
                    phase_ledger_generation, ImmutableDirectoryGeneration
                ):
                    raise ImmutableArtifactError(
                        "optional phase producer returned no committed generation"
                    )
                phase_ledger_generation_record = _directory_generation_record(
                    generation=phase_ledger_generation,
                    artifact_dir=artifact_dir,
                    attempt_id=attempt_id,
                )
                phase_evidence = _phase_evidence_identities(
                    artifact_dir=artifact_dir,
                    shape_point_name=shape_point["name"],
                    expected_repeat_count=ab_repeat_count,
                    expected_launch_indices=tuple(
                        launch.launch_index
                        for launch in trace_admission.launches
                    ),
                    phase_requested=phase_inject_ab,
                    generation=phase_ledger_generation,
                )
            artifact_record["phase_inject_ab"] = {
                "requested": phase_inject_ab,
                "repeat_count": ab_repeat_count if phase_inject_ab else 0,
                "generation": phase_ledger_generation_record,
                "evidence": phase_evidence,
            }

            preactivation_stage = "derived_commit"
            derived_generation = derived_stage.commit(
                require_nonempty=False,
                **_derived_generation_commit_limits(),
            )
            derived_generation_record = _directory_generation_record(
                generation=derived_generation,
                artifact_dir=artifact_dir,
                attempt_id=attempt_id,
            )
            _rebase_trace_feature_paths(
                artifact_record=artifact_record,
                artifact_dir=artifact_dir,
                staged_root=feature_staging_folder,
                committed_root=derived_generation.path,
            )
            artifact_record["trace_derived_generation"] = (
                derived_generation_record
            )
            feature_evidence = _collect_trace_feature_evidence_identities(
                artifact_record=artifact_record,
                artifact_dir=artifact_dir,
            )
            artifact_record["trace_feature_evidence"] = feature_evidence

            # Publish all launches together only after both raw and derived
            # generations are final.  No summary path can authorize a partial
            # feature tree.
            preactivation_stage = "summary_publish"
            nvbit_emit_result = _emit_nvbit_trace_summaries_for_trace_artifact(
                trace_admission=trace_admission,
                slug=slug,
                shape_point=shape_point["name"],
                binary_sha256=execution["binary_sha256"],
                tracer_build_sha256=execution["tracer_build_sha256"],
                artifact_dir=artifact_dir,
            )
            nvbit_summary_status = nvbit_emit_result["status"]
            nvbit_summary_paths = nvbit_emit_result["paths"]
            nvbit_summary_failure_reason = nvbit_emit_result["failure_reason"]
            if nvbit_emit_result["failed_hard"]:
                publish_committed_failure_event(
                    event_name="preactivation_failed",
                    status="summary_publish_failed",
                    typed_reason=str(nvbit_summary_status),
                    derived_generation=derived_generation_record,
                    phase_ledger_generation=phase_ledger_generation_record,
                )
                print(
                    "trace_nvbit_summary_emit_failed: "
                    f"status={nvbit_summary_status} "
                    f"reason={nvbit_summary_failure_reason}",
                    file=sys.stderr,
                )
                return EXIT_BUILD_FAILED

            preactivation_stage = "summary_generation_gate"
            summary_generation = (
                _require_one_committed_nvbit_summary_generation(
                    emit_result=nvbit_emit_result,
                    trace_admission=trace_admission,
                    slug=slug,
                    shape_point=shape_point["name"],
                    binary_sha256=execution["binary_sha256"],
                    tracer_build_sha256=execution["tracer_build_sha256"],
                    artifact_dir=artifact_dir,
                )
            )
            artifact_record["nvbit_trace_summary_emit_status"] = (
                nvbit_summary_status
            )
            artifact_record["nvbit_trace_summary_paths"] = nvbit_summary_paths
            artifact_record["nvbit_trace_summary_generation"] = (
                summary_generation
            )
            if nvbit_summary_failure_reason is not None:
                artifact_record["nvbit_trace_summary_failure_reason"] = (
                    nvbit_summary_failure_reason
                )

            preactivation_stage = "manifest"
            manifest = build_measurement_manifest(
                slug=slug, mode="trace",
                warm_up_iters=0, repeat_iters=1, cv_pct=0.0,
                props=props, binary_path=retained_executable.path,
                compiler_flags=_gencode_flags_for_arch_gate(
                    row.get("arch_gate", [])
                ),
                shape_point=shape_point["name"],
                binary_sha256=execution["binary_sha256"],
            )
            published_manifest = _emit_manifest(
                manifest,
                artifact_dir,
                filename=f"measurement_manifest_{shape_point['name']}.json",
                preserve_previous=True,
                retain_capability=True,
            )
            if not isinstance(published_manifest, RetainedRegular):
                raise _TraceActivationAuthorityError(
                    "trace_activation_authority_missing",
                    "measurement manifest publisher returned no retained capability",
                )
            retained_manifest = activation_authorities.stack.enter_context(
                published_manifest
            )
            manifest_path = retained_manifest.path
            manifest_identity = _retained_capability_identity(
                retained=retained_manifest, artifact_dir=artifact_dir
            )
            activation_authorities.add(
                "measurement_manifest", retained_manifest.verify_unchanged
            )
            artifact_record["measurement_manifest"] = manifest_identity

            # Arm descriptor-bound monitors before the final semantic gate. They
            # remain live through both active-leaf publication callbacks.
            preactivation_stage = "final_authority_acquisition"
            _retain_exact_directory_generation(
                authorities=activation_authorities,
                role="raw_generation",
                generation=raw_generation,
                require_nonempty=True,
                max_total_relative_path_bytes=int(
                    _trace_generation_commit_limits()[
                        "max_total_relative_path_bytes"
                    ]
                ),
            )
            _retain_exact_directory_generation(
                authorities=activation_authorities,
                role="derived_generation",
                generation=derived_generation,
                require_nonempty=False,
                max_total_relative_path_bytes=int(
                    _derived_generation_commit_limits()[
                        "max_total_relative_path_bytes"
                    ]
                ),
            )
            summary_directory_generation = _summary_generation_from_record(
                record=summary_generation,
                artifact_dir=artifact_dir,
            )
            _retain_exact_directory_generation(
                authorities=activation_authorities,
                role="summary_generation",
                generation=summary_directory_generation,
                require_nonempty=True,
                max_total_relative_path_bytes=max(
                    4096,
                    summary_directory_generation.file_count * 4096,
                ),
            )
            if phase_inject_ab:
                if phase_ledger_generation is None:
                    raise _TraceActivationAuthorityError(
                        "trace_activation_authority_missing",
                        "optional phase generation is absent at authority acquisition",
                    )
                _retain_exact_directory_generation(
                    authorities=activation_authorities,
                    role="phase_generation",
                    generation=phase_ledger_generation,
                    require_nonempty=True,
                    max_total_relative_path_bytes=4 * 1024 * 1024,
                )
            retained_runtime = execution.get("retained_runtime")
            if retained_runtime is None or not callable(
                getattr(retained_runtime, "verify_unchanged", None)
            ):
                raise _TraceActivationAuthorityError(
                    "trace_activation_authority_missing",
                    "retained tracer runtime authority is absent",
                )
            runtime_dso_sha256 = execution.get("runtime_dso_sha256")
            if (
                type(runtime_dso_sha256) is not dict
                or not runtime_dso_sha256
                or runtime_dso_sha256
                != dict(getattr(retained_runtime, "dso_sha256", {}))
            ):
                raise _TraceActivationAuthorityError(
                    "trace_activation_authority_invalid",
                    "retained tracer runtime DSO identities are absent or changed",
                )
            activation_authorities.add(
                "executable", retained_executable.verify_unchanged
            )
            activation_authorities.add(
                "tracer", retained_tracer.verify_unchanged
            )
            activation_authorities.add(
                "tracer_runtime", retained_runtime.verify_unchanged
            )
            activation_authorities.add(
                "admitted_trace_sources", trace_sources.verify_unchanged
            )

            preactivation_stage = "final_evidence_generation_gate"
            observed_summary_generation = (
                _require_one_committed_nvbit_summary_generation(
                    emit_result=nvbit_emit_result,
                    trace_admission=trace_admission,
                    slug=slug,
                    shape_point=shape_point["name"],
                    binary_sha256=execution["binary_sha256"],
                    tracer_build_sha256=execution["tracer_build_sha256"],
                    artifact_dir=artifact_dir,
                )
            )
            if observed_summary_generation != summary_generation:
                raise ImmutableArtifactError(
                    "NVBit summary generation changed before activation"
                )
            assert retained_manifest is not None
            retained_manifest.verify_unchanged()
            observed_manifest = _retained_capability_identity(
                retained=retained_manifest, artifact_dir=artifact_dir
            )
            if observed_manifest != manifest_identity:
                raise ImmutableArtifactError(
                    "measurement manifest generation changed before activation"
                )
            observed_feature_evidence = (
                _collect_trace_feature_evidence_identities(
                    artifact_record=artifact_record,
                    artifact_dir=artifact_dir,
                )
            )
            if observed_feature_evidence != feature_evidence:
                raise ImmutableArtifactError(
                    "trace feature evidence generation changed before activation"
                )
            observed_derived_generation = (
                inspect_immutable_directory_generation(
                    derived_generation.path,
                    require_nonempty=False,
                    **_derived_generation_commit_limits(),
                )
            )
            if observed_derived_generation != derived_generation:
                raise ImmutableArtifactError(
                    "trace derived generation changed before activation"
                )
            observed_raw_generation = inspect_immutable_directory_generation(
                raw_generation.path,
                require_nonempty=True,
                **_trace_generation_commit_limits(),
            )
            if observed_raw_generation != raw_generation:
                raise ImmutableArtifactError(
                    "complete raw trace generation changed before activation"
                )
            if phase_inject_ab:
                if phase_ledger_generation is None:
                    raise ImmutableArtifactError(
                        "optional phase generation is absent at final gate"
                    )
                observed_phase_evidence = _phase_evidence_identities(
                    artifact_dir=artifact_dir,
                    shape_point_name=shape_point["name"],
                    expected_repeat_count=ab_repeat_count,
                    expected_launch_indices=tuple(
                        launch.launch_index
                        for launch in trace_admission.launches
                    ),
                    phase_requested=phase_inject_ab,
                    generation=phase_ledger_generation,
                )
                if observed_phase_evidence != phase_evidence:
                    raise ImmutableArtifactError(
                        "optional phase evidence generation changed before activation"
                    )

            preactivation_stage = "final_trace_generation_gate"
            trace_sources.verify_unchanged()
            preactivation_stage = "attempt_ready"
            attempt_event = _publish_trace_attempt_event(
                artifact_dir=artifact_dir,
                slug=slug,
                shape_point=shape_point["name"],
                attempt_id=attempt_id,
                event_name="activation_ready",
                status="activation_ready",
                typed_reason=None,
                raw_trace_generation=raw_generation_record,
                derived_generation=derived_generation_record,
                phase_ledger_generation=phase_ledger_generation_record,
                summary_generation=summary_generation,
                measurement_manifest=manifest_identity,
                retained_authorities=activation_authorities,
            )
            artifact_record["trace_capture_attempt_event"] = attempt_event

            # The sole active leaf is committed while every executable, tracer,
            # and admitted raw-source capability remains live.  Closing the
            # executable/tracer stack here makes close-time revalidation part of
            # the transaction; any failure is caught below and retires this exact
            # active generation before control returns.
            preactivation_stage = "activation"
            artifact_record["activation_state"] = "active"
            artifact_record["typed_tombstone"] = False
            activation_payload = _trace_artifact_record_payload(
                artifact_record
            )
            # Arm exact-payload retirement before entering the publisher.  Its
            # no-replace rename may commit before a callback, wrapper, or later
            # authority closure raises; a precommit failure safely observes no
            # matching leaf, while a foreign replacement cannot authenticate.
            activation_rollback_armed = True
            _publish_trace_artifact_record(
                artifact_path,
                artifact_record,
                before_commit=activation_authorities.verify_unchanged,
                after_commit=activation_authorities.verify_unchanged,
            )
            preactivation_stage = "activation_authority_closure"
            activation_authorities.close()
            preactivation_stage = "activation_source_closure"
            trace_sources.verify_unchanged()
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
            retained_runtime.verify_unchanged()
            source_stack.pop_all().close()
    except _TraceActivationAuthorityError as exc:
        retire_failed_activation()
        activation_failure = preactivation_stage.startswith("activation")
        publish_committed_failure_event(
            event_name=(
                "activation_failed"
                if activation_failure else "preactivation_failed"
            ),
            status=(
                "activation_failed"
                if activation_failure
                else f"preactivation_{preactivation_stage}_failed"
            ),
            typed_reason=exc.typed_reason,
            derived_generation=derived_generation_record,
            derived_staging_path=(
                None if derived_stage.committed else feature_staging_folder
            ),
            phase_ledger_generation=phase_ledger_generation_record,
            phase_ledger_staging_path=(
                None
                if phase_ledger_generation is not None
                else phase_ledger_staging_path
            ),
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        print(
            f"trace_activation_authority_failed: {exc.typed_reason}: "
            f"{exc.detail}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except ImmutableArtifactLimitError as exc:
        retire_failed_activation()
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status=f"preactivation_{preactivation_stage}_failed",
            typed_reason="trace_inventory_limit_exceeded",
            derived_generation=derived_generation_record,
            derived_staging_path=(
                None if derived_stage.committed else feature_staging_folder
            ),
            phase_ledger_generation=phase_ledger_generation_record,
            phase_ledger_staging_path=(
                None
                if phase_ledger_generation is not None
                else phase_ledger_staging_path
            ),
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        print(
            "trace_preactivation_limit_exceeded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except TraceAdmissionError as exc:
        retire_failed_activation()
        if exc.typed_reason == "trace_snapshot_resource_unavailable":
            diagnostic = (
                "trace_feature_source_resource_unavailable"
                if preactivation_stage == "features"
                else "trace_preactivation_resource_unavailable"
            )
        elif preactivation_stage == "features":
            diagnostic = "trace_feature_source_changed"
        else:
            diagnostic = "trace_preactivation_snapshot_changed"
        publish_committed_failure_event(
            event_name="preactivation_failed",
            status=f"preactivation_{preactivation_stage}_failed",
            typed_reason=exc.typed_reason,
            derived_generation=derived_generation_record,
            derived_staging_path=(
                None if derived_stage.committed else feature_staging_folder
            ),
            phase_ledger_generation=phase_ledger_generation_record,
            phase_ledger_staging_path=(
                None
                if phase_ledger_generation is not None
                else phase_ledger_staging_path
            ),
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        print(
            f"{diagnostic}: {exc.typed_reason}: {exc.detail}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except TracerRuntimeError as exc:
        retire_failed_activation()
        activation_failure = preactivation_stage.startswith("activation")
        typed_reason = (
            "trace_snapshot_resource_unavailable"
            if exc.reason == "runtime_resource_unavailable"
            else exc.reason
        )
        publish_committed_failure_event(
            event_name=(
                "activation_failed"
                if activation_failure else "preactivation_failed"
            ),
            status=(
                "activation_failed"
                if activation_failure
                else f"preactivation_{preactivation_stage}_failed"
            ),
            typed_reason=typed_reason,
            derived_generation=derived_generation_record,
            derived_staging_path=(
                None if derived_stage.committed else feature_staging_folder
            ),
            phase_ledger_generation=phase_ledger_generation_record,
            phase_ledger_staging_path=(
                None
                if phase_ledger_generation is not None
                else phase_ledger_staging_path
            ),
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        print(
            f"trace_activation_runtime_failed: {typed_reason}: {exc.detail}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except ImmutableArtifactResourceError as exc:
        retire_failed_activation()
        activation_failure = preactivation_stage.startswith("activation")
        diagnostic = (
            "trace_activation_resource_unavailable"
            if activation_failure
            else (
                "trace_feature_source_resource_unavailable"
                if preactivation_stage == "features"
                else "trace_preactivation_resource_unavailable"
            )
        )
        publish_committed_failure_event(
            event_name=(
                "activation_failed"
                if activation_failure
                else "preactivation_failed"
            ),
            status=(
                "activation_failed"
                if activation_failure
                else f"preactivation_{preactivation_stage}_failed"
            ),
            typed_reason="trace_snapshot_resource_unavailable",
            derived_generation=derived_generation_record,
            derived_staging_path=(
                None if derived_stage.committed else feature_staging_folder
            ),
            phase_ledger_generation=phase_ledger_generation_record,
            phase_ledger_staging_path=(
                None
                if phase_ledger_generation is not None
                else phase_ledger_staging_path
            ),
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        print(
            f"{diagnostic}: trace_snapshot_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except (ImmutableArtifactError, OSError) as exc:
        retire_failed_activation()
        activation_failure = preactivation_stage.startswith("activation")
        if preactivation_stage == "features":
            diagnostic = "trace_feature_source_changed: trace_snapshot_changed"
        elif activation_failure:
            diagnostic = "trace_activation_failed"
        else:
            diagnostic = f"trace_preactivation_{preactivation_stage}_failed"
        publish_committed_failure_event(
            event_name=(
                "activation_failed"
                if activation_failure
                else "preactivation_failed"
            ),
            status=(
                "activation_failed"
                if activation_failure
                else f"preactivation_{preactivation_stage}_failed"
            ),
            typed_reason=(
                "trace_activation_failed"
                if activation_failure
                else "trace_filesystem_invalid"
            ),
            derived_generation=derived_generation_record,
            derived_staging_path=(
                None if derived_stage.committed else feature_staging_folder
            ),
            phase_ledger_generation=phase_ledger_generation_record,
            phase_ledger_staging_path=(
                None
                if phase_ledger_generation is not None
                else phase_ledger_staging_path
            ),
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        print(
            f"{diagnostic}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_BUILD_FAILED
    except BaseException as exc:
        # Ordinary Python exceptions after the active rename are recoverable:
        # retire only the armed exact payload, retain a typed attempt event when
        # the evidence publisher is usable, and then preserve the primary error.
        # Process/host termination remains outside this catchable boundary.
        retire_failed_activation(primary=exc)
        activation_failure = preactivation_stage.startswith("activation")
        typed_reason = (
            "trace_snapshot_resource_unavailable"
            if (
                isinstance(exc, ImmutableArtifactResourceError)
                or _trace_capture_resource_failure(exc)
            )
            else (
                exc.reason
                if isinstance(exc, TracerRuntimeError)
                else (
                    "trace_activation_failed"
                    if activation_failure
                    else "trace_filesystem_invalid"
                )
            )
        )
        try:
            publish_committed_failure_event(
                event_name=(
                    "activation_failed"
                    if activation_failure else "preactivation_failed"
                ),
                status=(
                    f"{preactivation_stage}_failed"
                    if activation_failure
                    else f"preactivation_{preactivation_stage}_failed"
                ),
                typed_reason=typed_reason,
                derived_generation=derived_generation_record,
                derived_staging_path=(
                    None if derived_stage.committed else feature_staging_folder
                ),
                phase_ledger_generation=phase_ledger_generation_record,
                phase_ledger_staging_path=(
                    None
                    if phase_ledger_generation is not None
                    else phase_ledger_staging_path
                ),
                summary_generation=summary_generation,
                measurement_manifest=manifest_identity,
            )
        except BaseException as evidence_error:
            try:
                exc.add_note(
                    "unexpected trace failure evidence preservation also failed: "
                    f"{type(evidence_error).__name__}: {evidence_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    finally:
        active_error = sys.exc_info()[1]
        try:
            activation_authorities.close()
        except BaseException as cleanup_exc:
            retire_failed_activation(primary=active_error or cleanup_exc)
            if active_error is not None:
                try:
                    active_error.add_note(
                        "trace activation authority cleanup also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                except (AttributeError, TypeError):
                    pass
            else:
                activation_authority_close_error = cleanup_exc
            print(
                "trace_activation_authority_cleanup_failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
            )
        try:
            derived_stage.close()
        except BaseException as cleanup_exc:
            derived_stage_close_error = cleanup_exc
            retire_failed_activation(primary=active_error or cleanup_exc)
            print(
                "trace_derived_stage_cleanup_failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
            )

    if activation_authority_close_error is not None:
        typed_reason = (
            "trace_snapshot_resource_unavailable"
            if isinstance(
                activation_authority_close_error,
                (ImmutableArtifactResourceError, MemoryError),
            )
            else (
                activation_authority_close_error.reason
                if isinstance(
                    activation_authority_close_error, TracerRuntimeError
                )
                else "trace_activation_failed"
            )
        )
        publish_committed_failure_event(
            event_name="activation_failed",
            status="activation_authority_closure_failed",
            typed_reason=typed_reason,
            derived_generation=derived_generation_record,
            phase_ledger_generation=phase_ledger_generation_record,
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        return EXIT_BUILD_FAILED

    if derived_stage_close_error is not None:
        typed_reason = (
            "trace_snapshot_resource_unavailable"
            if isinstance(
                derived_stage_close_error,
                ImmutableArtifactResourceError,
            )
            else "trace_filesystem_invalid"
        )
        publish_committed_failure_event(
            event_name="activation_failed",
            status="activation_derived_stage_closure_failed",
            typed_reason=typed_reason,
            derived_generation=derived_generation_record,
            phase_ledger_generation=phase_ledger_generation_record,
            summary_generation=summary_generation,
            measurement_manifest=manifest_identity,
        )
        return EXIT_BUILD_FAILED

    print(
        f"run_suite: trace for {slug}: {trace_bytes} bytes at {trace_pb} "
        f"(artifact at {artifact_path})"
    )
    return EXIT_OK


def _phase_summary_admission_error(
    summary: object,
    *,
    expected_ledger_count: int,
) -> str | None:
    """Require every produced repeat-by-launch ledger to be admitted."""

    if type(summary) is not dict:
        return "phase ledger producer summary is not an object"
    n_admitted = summary.get("n_admitted")
    n_rejected = summary.get("n_rejected")
    emitted_admitted = summary.get("emitted_admitted")
    emitted_rejected = summary.get("emitted_rejected")
    if (
        type(n_admitted) is not int
        or type(n_rejected) is not int
        or type(emitted_admitted) is not list
        or type(emitted_rejected) is not list
    ):
        return "phase ledger producer summary has malformed admission counts"
    if (
        n_admitted != len(emitted_admitted)
        or n_rejected != len(emitted_rejected)
        or n_admitted + n_rejected != expected_ledger_count
    ):
        return "phase ledger producer summary disagrees with repeat-by-launch authority"
    if n_rejected != 0:
        return f"phase ledger producer rejected {n_rejected} ledger(s)"
    if n_admitted != expected_ledger_count:
        return "phase ledger producer did not admit the complete repeat-by-launch set"
    return None


def _run_phase_inject_ab_for_handle_trace(
    *,
    slug: str,
    shape_point_name: str,
    attempt_id: str,
    retained_executable: RetainedExecutable,
    args: list[str],
    retained_tracer: RetainedRegular,
    traces_folder: Path,
    artifact_dir: Path,
    ab_repeat_count: int,
    expected_launch_indices: tuple[int, ...],
    row: dict[str, Any],
    binary_sha256: str,
    kernel_signature_sha256: str,
    tracer_build_sha256: str,
) -> dict[str, Any]:
    """Run the A/B pair and atomically publish its exact ledger generation."""
    import phase_inject_orchestrator as _pinj  # type: ignore[import-not-found]
    import phase_cycle_ledger_consumer as _pcons  # type: ignore[import-not-found]

    def failed(
        typed_reason: str,
        *,
        generation: ImmutableDirectoryGeneration | None = None,
        staging_path: Path | None = None,
        summary: dict[str, Any] | None = None,
        observed_output_bytes_lower_bound: int | None = None,
    ) -> dict[str, Any]:
        if (
            observed_output_bytes_lower_bound is not None
            and (
                type(observed_output_bytes_lower_bound) is not int
                or observed_output_bytes_lower_bound < 0
            )
        ):
            raise ImmutableArtifactError(
                "phase failure output-byte lower bound is invalid"
            )
        return {
            "return_code": EXIT_BUILD_FAILED,
            "typed_reason": typed_reason,
            "generation": generation,
            "staging_path": staging_path,
            "summary": summary,
            "observed_output_bytes_lower_bound": (
                observed_output_bytes_lower_bound
            ),
        }

    try:
        _pinj._validated_repeat_and_launch_domain(
            ab_repeat_count,
            expected_launch_indices,
        )
        _pinj._validated_arguments(args)
        retained_executable.verify_unchanged()
        retained_tracer.verify_unchanged()
    except (
        _pinj.PhaseInjectAbRepeatCountInvalid,
        _pinj.PhaseInjectLaunchAuthorityInvalid,
        _pinj.PhaseInjectResourceLimitExceeded,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return failed(exc.TOKEN)

    # A/B tracer bytes stay inside the hidden derived stage.  Pin the dedicated
    # child and pass its descriptor through /proc so no mutable lexical path is
    # authoritative for either subprocess or the ledger consumer.
    phase_traces_folder = traces_folder / "phase_inject_ab"
    admission = _pcons.load_admission_registry()
    try:
        admission_threshold_pct = _pinj._validated_admission_threshold(
            _pcons.admission_threshold_for_slug(admission, slug)
        )
    except _pinj.PhaseInjectAdmissionThresholdInvalid as exc:
        print(str(exc), file=sys.stderr)
        return failed(exc.TOKEN)
    ensure_directory(phase_traces_folder)
    generation: ImmutableDirectoryGeneration | None = None
    summary: dict[str, Any] | None = None
    postcommit_unwind_failure: _pinj.PhaseLedgerPostCommitFailure | None = None

    @contextmanager
    def preserve_phase_source_generation():
        nonlocal postcommit_unwind_failure
        try:
            with PinnedDirectoryReader(phase_traces_folder) as phase_sources:
                yield phase_sources
        except BaseException as exc:
            if generation is None:
                raise
            postcommit_unwind_failure = _pinj._postcommit_failure(
                generation,
                exc,
            )

    with preserve_phase_source_generation() as phase_sources:
        inputs = _pinj.AbOverheadInputs(
            executable_authority=retained_executable,
            args=list(args),
            tracer_so_path=retained_tracer.path,
            tracer_descriptor=retained_tracer.descriptor,
            traces_folder=phase_traces_folder,
            expected_launch_indices=expected_launch_indices,
            traces_descriptor=phase_sources.descriptor,
            traces_reader=phase_sources,
            repeat_count=ab_repeat_count,
        )
        try:
            ab_overhead_result = _pinj.run_ab_overhead_pair(
                inputs, admission_threshold_pct=admission_threshold_pct
            )
            phase_source_snapshot = ab_overhead_result.pop(
                "_phase_source_snapshot", None
            )
            if not isinstance(phase_source_snapshot, PinnedTreeSnapshot):
                raise ImmutableArtifactError(
                    "phase A/B producer returned no retained source snapshot"
                )
            ab_overhead_record = ab_overhead_result
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
        except _pinj.PhaseInjectAbRepeatCountInvalid as exc:
            print(str(exc), file=sys.stderr)
            return failed(exc.TOKEN)
        except _pinj.PhaseRecordSourceNotFresh as exc:
            print(str(exc), file=sys.stderr)
            return failed(exc.TOKEN)
        except _pinj.PhaseInjectAbRunFailed as exc:
            print(
                "phase_inject_ab_run_failed: "
                f"side={exc.side!r} rc={exc.returncode}: "
                f"{exc.stderr.strip()[:200]}",
                file=sys.stderr,
            )
            return failed(
                exc.TOKEN,
                observed_output_bytes_lower_bound=(
                    exc.observed_output_bytes_lower_bound
                ),
            )
        except _pinj.PhaseInjectAbRunTimeout as exc:
            print(
                "phase_inject_ab_run_timeout: "
                f"side={exc.side!r} timeout_s={exc.timeout_s}: "
                f"{exc.stderr.strip()[:200]}",
                file=sys.stderr,
            )
            return failed(
                exc.TOKEN,
                observed_output_bytes_lower_bound=(
                    exc.observed_output_bytes_lower_bound
                ),
            )
        except _pinj.PhaseInjectRuntimeUnavailable as exc:
            print(
                f"phase_inject_runtime_unavailable: {exc.detail}",
                file=sys.stderr,
            )
            return failed(exc.TOKEN)
        except (
            _pinj.PhaseInjectAbCleanupFailed,
            _pinj.PhaseInjectAbOutputLimitExceeded,
            _pinj.PhaseInjectAbResourceUnavailable,
            _pinj.PhaseInjectAbLaunchFailed,
        ) as exc:
            _bind_trace_child_failure_evidence(
                exc,
                directory=phase_traces_folder,
                role=exc.TOKEN,
                stdout=exc.stdout_bytes,
                stderr=exc.stderr_bytes,
            )
            print(str(exc), file=sys.stderr)
            return failed(
                exc.TOKEN,
                observed_output_bytes_lower_bound=(
                    exc.observed_output_bytes_lower_bound
                ),
            )
        except (
            _pinj.PhaseInjectLaunchAuthorityInvalid,
            _pinj.PhaseInjectAdmissionThresholdInvalid,
            _pinj.PhaseInjectTimeoutInvalid,
            _pinj.PhaseInjectResourceLimitExceeded,
            _pinj.PhaseInjectExecutableAuthorityInvalid,
            _pinj.PhaseRecordEvidenceInvalid,
            _pinj.PhaseRecordPayloadInvalid,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return failed(exc.TOKEN)

        ledger_root = _trace_generation_root(
            artifact_dir=artifact_dir,
            namespace=PHASE_LEDGER_GENERATION_NAMESPACE,
            shape_point=shape_point_name,
        )
        ledger_staging_path = ledger_root / f".{attempt_id}.staging"
        import phase_cycle_ledger_aggregator as _agg  # type: ignore[import-not-found]

        typed_ledger_exceptions = (
            _agg.PhaseRecordPairingFailed,
            _agg.PhaseRecordCycleDeltaNegative,
            _agg.PhaseLedgerSchemaInvalid,
            _agg.PhaseRecordDuplicateOpen,
            _agg.PhaseRecordDuplicateClose,
            _agg.PhaseRecordLaunchIndexMismatch,
            _agg.PhaseDefinitionInconsistent,
            _pinj.PhaseLedgerEvidenceMissing,
            _pinj.PhaseLedgerRepeatCoverageInvalid,
            _pinj.PhaseInjectLaunchAuthorityInvalid,
            _pinj.PhaseInjectAdmissionThresholdInvalid,
            _pinj.PhaseInjectResourceLimitExceeded,
            _pinj.PhaseRecordEvidenceInvalid,
            _pinj.PhaseRecordPayloadInvalid,
        )
        phase_sources.verify_path()
        try:
            summary = _pinj.emit_per_launch_phase_ledgers(
                traces_folder=phase_traces_folder,
                ledger_output_dir=ledger_root,
                attempt_id=attempt_id,
                expected_repeat_count=ab_repeat_count,
                expected_launch_indices=expected_launch_indices,
                slug=slug,
                shape_point=shape_point_name,
                kernel_signature_sha256=kernel_signature_sha256,
                binary_sha256=binary_sha256,
                nvbit_tracer_build_hash=tracer_build_sha256,
                claim_scope=row.get("claim_scope", "measured_local"),
                ab_overhead_record=ab_overhead_record,
                dual_run_manifest_ref=str(
                    artifact_dir / "dual_run_manifest.json"
                ),
                admission=admission,
                admission_threshold_pct=admission_threshold_pct,
                source_reader=phase_sources,
                expected_source_snapshot=phase_source_snapshot,
            )
        except _pinj.PhaseLedgerPostCommitFailure as exc:
            generation = exc.generation
            print(
                "phase_inject_ab_postcommit_source_invalid: "
                f"{exc}",
                file=sys.stderr,
            )
            return failed(
                exc.typed_reason,
                generation=generation,
            )
        except typed_ledger_exceptions as exc:
            print(
                "phase_inject_ab_ledger_typed_failure: "
                f"token={exc.TOKEN} slug={slug} "
                f"shape_point={shape_point_name}: "
                f"{str(exc).strip()[:400]}",
                file=sys.stderr,
            )
            return failed(
                exc.TOKEN,
                staging_path=ledger_staging_path,
            )
        generation = summary.get("generation")
        if not isinstance(generation, ImmutableDirectoryGeneration):
            raise ImmutableArtifactError(
                "phase ledger producer returned no immutable generation"
            )
        try:
            phase_sources.verify_path()
            retained_executable.verify_unchanged()
            retained_tracer.verify_unchanged()
        except ImmutableArtifactResourceError as exc:
            print(
                "phase_inject_ab_source_resource_unavailable: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return failed(
                "trace_snapshot_resource_unavailable",
                generation=generation,
                summary=summary,
            )
        except (ImmutableArtifactError, OSError) as exc:
            print(
                "phase_inject_ab_source_changed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return failed(
                "trace_filesystem_invalid",
                generation=generation,
                summary=summary,
            )

    if postcommit_unwind_failure is not None:
        print(
            "phase_inject_ab_postcommit_source_cleanup_invalid: "
            f"{postcommit_unwind_failure}",
            file=sys.stderr,
        )
        return failed(
            postcommit_unwind_failure.typed_reason,
            generation=postcommit_unwind_failure.generation,
            summary=summary,
        )
    assert generation is not None
    assert summary is not None
    phase_admission_error = _phase_summary_admission_error(
        summary,
        expected_ledger_count=(
            ab_repeat_count * len(expected_launch_indices)
        ),
    )
    if phase_admission_error is not None:
        print(
            "phase_cycle_ledger_inadmissible: " + phase_admission_error,
            file=sys.stderr,
        )
        return failed(
            "phase_cycle_ledger_inadmissible",
            generation=generation,
            summary=summary,
        )
    print(
        "run_suite: phase_inject_ab "
        f"slug={slug} shape_point={shape_point_name} "
        f"overhead_pct={ab_overhead_record['overhead_pct']:.3f} "
        f"admitted={ab_overhead_record['admitted']} "
        f"emitted_admitted={summary['n_admitted']} "
        f"emitted_rejected={summary['n_rejected']}"
    )
    return {
        "return_code": EXIT_OK,
        "typed_reason": None,
        "generation": generation,
        "summary": summary,
    }


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
    "the full-corpus replay driver",
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
    p.add_argument(
        "--nvbit-tracer",
        type=Path,
        default=None,
        help="Trace mode: absolute external non-Git split-v2 tracer path.",
    )
    p.add_argument(
        "--nvbit-tracer-sha256",
        default=None,
        help="Trace mode: exact lowercase SHA-256 for --nvbit-tracer.",
    )
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

    import phase_inject_orchestrator as _pinj  # type: ignore[import-not-found]

    if not (1 <= args.ab_repeat_count <= _pinj.PHASE_AB_MAX_REPEAT_COUNT):
        print(
            "phase_inject_ab_repeat_count_invalid: "
            "--ab-repeat-count must be in "
            f"[1, {_pinj.PHASE_AB_MAX_REPEAT_COUNT}]",
            file=sys.stderr,
        )
        return EXIT_USAGE
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
                tracer_source_path=args.nvbit_tracer,
                tracer_sha256=args.nvbit_tracer_sha256,
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
