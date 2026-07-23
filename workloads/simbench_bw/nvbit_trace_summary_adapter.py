"""NVBit-side adapter for the offline two-run join.

Provides two surfaces consumed by `merge_evidence.py join-cupti-nvbit`
and by `run_suite.py --mode trace`:

1. `compute_kernel_signature_sha256(inputs)` — production NVBit-side
   reference implementation of `workloads/simbench_bw/schemas/
   kernel_signature.spec.md`. Conformance-tested against every fixture
   under `golden/kernel_signature_fixtures/` by
   `tests/test_kernel_signature_conformance_nvbit_side.py`. The
   implementation mirrors the spec byte-for-byte: tuple-ordered field
   iteration, JSON canonical serialization with `ensure_ascii=False`
   and `separators=(",", ":")`, SHA-256 hex-lowercase 64 chars.

2. `emit_nvbit_trace_summary(...)` — opt-in helper invoked by
   `run_suite.py` after a successful `--mode trace` capture. Reads the
   `dynamic_trace.pb` produced by the NVBit tracer, walks the
   `traces/threadblocks/**/*.pb` subtree to sum instruction counts
   across warps × threadblocks, computes provenance hashes, and writes
   `nvbit_trace_summary_<repeat>_<launch>.json` next to the source
   `.pb`. Requires the `google.protobuf` Python runtime and the tracked
   `_pb_generated/*_pb2.py` authorities. A missing runtime is a host
   prerequisite failure; a missing or unusable tracked binding is repository
   corruption. Both conditions are hard failures and are never repaired by
   generating files in the checkout.

The two surfaces are intentionally split so the conformance test can
exercise the signature-spec implementation without depending on
protobuf being installed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUITE_ROOT.parent.parent
PB_GENERATED_DIR = SUITE_ROOT / "_pb_generated"


def _repo_relative_or_absolute(path: Path) -> str:
    """Serialize `path` as repo-relative when it is under REPO_ROOT
    (the normal case for trace artifacts under `results/`); fall back
    to absolute for paths outside the repo (test scaffolding,
    externally staged trace dirs). Mirrors the established pattern in
    `run_suite.py::handle_trace`'s `tracer_so_serialized` fallback so
    the schema's `nvbit_trace_pb_path` "repo-relative path" contract
    is honored for normal runs without surprising tests that stage
    artifacts in tmpdirs outside the repo."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Production NVBit-side reference implementation of
# `kernel_signature.spec.md`. Mirrors the spec's field order verbatim;
# any drift is caught by the conformance test parametrized against
# `golden/kernel_signature_fixtures/`.
# ---------------------------------------------------------------------------

KERNEL_SIGNATURE_FIELD_ORDER: tuple[str, ...] = (
    "kernel_name",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem_static_bytes",
    "shared_mem_dynamic_bytes",
    "regs_per_thread",
)


def _spec_strict_escape_kernel_name(s: str) -> str:
    """JSON-string-escape `s` per the spec at
    `workloads/simbench_bw/schemas/kernel_signature.spec.md:60-67`:

      - `"` → `\\"`
      - `\\` → `\\\\`
      - control characters U+0000–U+001F → `\\u00XX` (lowercase 4-digit hex)
      - non-ASCII (U+0080+) emitted verbatim as UTF-8 bytes

    Returns the encoded form INCLUDING the surrounding double-quotes
    (matches `json.dumps`'s convention so callers can drop the result
    directly into a JSON object). Mirrors the CUPTI-side C++
    implementation `cupti_activity_collector.cu::json_escape_kernel_name`
    byte-for-byte; the cross-implementation conformance suites
    enforce this equivalence against every fixture under
    `golden/kernel_signature_fixtures/`.

    Python's stdlib `json.dumps(s, ensure_ascii=False)` deviates from
    the spec by emitting RFC 8259's short escapes (`\\t`, `\\n`,
    `\\r`, `\\b`, `\\f`) for the corresponding control chars instead
    of `\\u00XX`. Both forms are RFC 8259-valid, but they produce
    different bytes — so the kernel_signature_sha256 cross-impl
    contract requires the spec-strict form to match the C++ side.
    """
    parts = ['"']
    for ch in s:
        if ch == '"':
            parts.append('\\"')
        elif ch == '\\':
            parts.append('\\\\')
        elif ord(ch) < 0x20:
            parts.append(f'\\u{ord(ch):04x}')
        else:
            parts.append(ch)
    parts.append('"')
    return ''.join(parts)


def canonical_serialize(inputs: dict[str, Any]) -> str:
    """Per the spec: single-line JSON object with the ten fields in
    fixed order, no whitespace inside the object, no trailing newline,
    `kernel_name` JSON-string-escaped per RFC 8259 §7 with non-ASCII
    emitted verbatim as UTF-8. Numeric fields are decimal integers
    (no leading zeros, `+`, decimal point, or exponent).

    Raises `KeyError` if a required field is missing or an unexpected
    field is present (the inputs MUST be exactly the 10-field set).
    Raises `TypeError` if a field has the wrong type. Raises
    `ValueError` if a numeric field is negative.
    """
    missing = [f for f in KERNEL_SIGNATURE_FIELD_ORDER if f not in inputs]
    if missing:
        raise KeyError(f"missing required signature field(s): {missing}")
    extra = sorted(set(inputs) - set(KERNEL_SIGNATURE_FIELD_ORDER))
    if extra:
        raise KeyError(f"unexpected field(s) in signature inputs: {extra}")
    parts: list[str] = []
    for field in KERNEL_SIGNATURE_FIELD_ORDER:
        value = inputs[field]
        if field == "kernel_name":
            if not isinstance(value, str):
                raise TypeError(
                    f"{field} must be a string, got {type(value).__name__}"
                )
            encoded = _spec_strict_escape_kernel_name(value)
        else:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field} must be a non-bool integer, got "
                    f"{type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{field} must be >= 0, got {value}")
            encoded = json.dumps(value)
        parts.append(f'"{field}":{encoded}')
    return "{" + ",".join(parts) + "}"


def compute_kernel_signature_sha256(inputs: dict[str, Any]) -> str:
    """Production reference: SHA-256 over the canonical serialization,
    hex-lowercase, 64 chars."""
    canonical = canonical_serialize(inputs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 of a file's contents, hex-lowercase, 64 chars.
    Streamed in 1 MB chunks for memory-friendly hashing of large
    artifacts (binaries, dynamic_trace.pb)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Protobuf-parsing path (requires the `google.protobuf` runtime and all nine
# tracked `_pb_generated/*_pb2.py` authorities).
# ---------------------------------------------------------------------------

class NvbitProtobufBindingsMissing(RuntimeError):
    """Compatibility base for hard protobuf dependency failures."""


class NvbitTrackedProtobufCorruption(NvbitProtobufBindingsMissing):
    """A tracked binding is absent, damaged, or resolves outside its owner."""


class NvbitProtobufRuntimePrerequisiteMissing(NvbitProtobufBindingsMissing):
    """The project-pinned Python protobuf runtime is absent or incompatible."""


_TRACKED_BINDING_NAMES: tuple[str, ...] = (
    "address_pb2.py",
    "cuda_stream_pb2.py",
    "dim3d_pb2.py",
    "gpu_device_pb2.py",
    "instruction_pb2.py",
    "kernel_pb2.py",
    "threadblock_pb2.py",
    "trace_pb2.py",
    "warp_pb2.py",
)


def _binding_restore_guidance(paths: list[Path]) -> str:
    relative = " ".join(
        str(path.relative_to(REPO_ROOT)) for path in sorted(paths)
    )
    return (
        "tracked protobuf authority is missing or unusable; restore its "
        "byte-exact committed content with `git restore --source=HEAD "
        f"--worktree -- {relative}`; do not regenerate bindings in the "
        "canonical checkout"
    )


def _import_pb_modules():
    """Import tracked dynamic-trace authorities and their pinned runtime.

    Dependency failures are deliberately split so callers can report
    repository corruption separately from a missing host prerequisite.
    """
    try:
        import google.protobuf  # noqa: F401  # pyright: ignore[reportMissingModuleSource, reportUnusedImport]
    except ImportError as exc:
        raise NvbitProtobufRuntimePrerequisiteMissing(
            "protobuf runtime prerequisite is missing; install the exact "
            "version pinned in requirements.txt before consuming the "
            "tracked dynamic-trace bindings"
        ) from exc
    required = [PB_GENERATED_DIR / name for name in _TRACKED_BINDING_NAMES]
    missing = [
        path for path in required if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise NvbitTrackedProtobufCorruption(_binding_restore_guidance(missing))
    if str(PB_GENERATED_DIR) not in sys.path:
        sys.path.insert(0, str(PB_GENERATED_DIR))
    try:
        import trace_pb2  # pyright: ignore[reportMissingImports]
        import threadblock_pb2  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        if exc.__class__.__module__.startswith("google.protobuf"):
            raise NvbitProtobufRuntimePrerequisiteMissing(
                "protobuf runtime prerequisite is incompatible with the "
                f"tracked bindings: {type(exc).__name__}: {exc}"
            ) from exc
        raise NvbitTrackedProtobufCorruption(
            f"{_binding_restore_guidance(required)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for module, path in (
        (trace_pb2, PB_GENERATED_DIR / "trace_pb2.py"),
        (threadblock_pb2, PB_GENERATED_DIR / "threadblock_pb2.py"),
    ):
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != path.resolve():
            raise NvbitTrackedProtobufCorruption(
                f"tracked protobuf authority resolved outside its owner: "
                f"{module.__name__} -> {getattr(module, '__file__', None)}; "
                f"{_binding_restore_guidance([path])}"
            )
    for filename in _TRACKED_BINDING_NAMES:
        module = sys.modules.get(filename.removesuffix(".py"))
        if module is None:
            continue
        expected = (PB_GENERATED_DIR / filename).resolve()
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != expected:
            raise NvbitTrackedProtobufCorruption(
                f"tracked protobuf authority {module.__name__} resolved "
                f"outside its owner: {getattr(module, '__file__', None)}; "
                f"{_binding_restore_guidance([PB_GENERATED_DIR / filename])}"
            )
    return trace_pb2, threadblock_pb2


def _enumerate_kernels(trace_pb2_mod, trace_pb_path: Path) -> list[Any]:
    """Parse `dynamic_trace.pb` and return all kernel records in
    source order. The trace .pb contains a single `Trace` message:
    `Trace -> gpu_device(map<int64,gpu_device>) ->
    cuda_stream(map<int64,cuda_stream>) -> kernel(repeated)`. The
    enumeration order is: device-id ascending, then stream-id
    ascending, then `kernel.id` ascending (the tracer's stable
    monotonic per-process kernel-launch index). This deterministic
    ordering MUST match the order `run_suite.py --mode trace`
    iterates over launches when emitting per-launch summaries, so
    `launch_index=N` in the summary always refers to the same kernel
    record across both producers.
    """
    trace = trace_pb2_mod.Trace()
    trace.ParseFromString(trace_pb_path.read_bytes())
    kernels: list[Any] = []
    for _device_id in sorted(trace.gpu_device.keys()):
        device = trace.gpu_device[_device_id]
        for _stream_id in sorted(device.streams.keys()):
            stream = device.streams[_stream_id]
            kernels.extend(sorted(stream.kernels, key=lambda k: int(k.id)))
    return kernels


def enumerate_launches(trace_pb_path: Path) -> int:
    """Public helper: count the discoverable kernel launches in a
    trace .pb. Returns 0 if the file is missing or contains no
    kernels (does NOT raise) so callers can decide how to handle the
    no-launches case. Dependency failures are hard and distinguish tracked
    repository corruption from a missing protobuf runtime prerequisite.
    """
    if not trace_pb_path.is_file():
        return 0
    trace_pb2_mod, _threadblock_pb2_mod = _import_pb_modules()
    return len(_enumerate_kernels(trace_pb2_mod, trace_pb_path))


def _extract_kernel_metadata_for_launch(
    trace_pb2_mod, trace_pb_path: Path, launch_index: int,
) -> tuple[dict[str, Any], int]:
    """Extract kernel metadata for the `launch_index`-th kernel
    discovered in the trace .pb (per `_enumerate_kernels()`'s order).
    Returns (kernel_metadata_dict, kernel_id) — the kernel_id is the
    tracer's stable per-launch identifier used to scope threadblock
    files in `_sum_inst_count_for_kernel()`.

    Raises `IndexError` if `launch_index` is out of range so the
    caller (and the orchestrator's producer hook) can fail loudly
    when an out-of-range launch index is requested.
    """
    kernels = _enumerate_kernels(trace_pb2_mod, trace_pb_path)
    if launch_index < 0 or launch_index >= len(kernels):
        raise IndexError(
            f"launch_index {launch_index} out of range; trace .pb at "
            f"{trace_pb_path} carries {len(kernels)} kernel record(s)"
        )
    kernel = kernels[launch_index]
    grid = kernel.grid_dim
    block = kernel.block_dim
    metadata = {
        "kernel_name": kernel.name,
        "grid_x": int(grid.x),
        "grid_y": int(grid.y),
        "grid_z": int(grid.z),
        "block_x": int(block.x),
        "block_y": int(block.y),
        "block_z": int(block.z),
        "shared_mem_static_bytes": int(kernel.size_shared_memory),
        # The .proto's kernel message does not carry dynamic shared
        # memory separately; default to 0 (the static field IS the
        # per-launch static allocation per the field name).
        "shared_mem_dynamic_bytes": 0,
        "regs_per_thread": int(kernel.number_of_registers),
    }
    return metadata, int(kernel.id)


def _sum_inst_count_for_kernel(
    threadblock_pb2_mod, threadblocks_root: Path, kernel_id: int,
) -> int:
    """Sum instruction counts ONLY across threadblock files for the
    specified `kernel_id`. The tracer writes one .pb per threadblock
    under `<traces>/threadblocks/device_<id>/stream_<id>/kernel_<id>/
    <tb_x>_<tb_y>_<tb_z>.pb`; each file holds one `threadblock`
    message with `map<int32,warp> warps`, each `warp` with `repeated
    instruction instructions`.

    Returns 0 if no matching kernel directory exists (e.g. a kernel
    that never wrote threadblock records — the launch is still
    counted in `enumerate_launches` because it appears in the Trace
    message, but its inst_count is 0).
    """
    if not threadblocks_root.is_dir():
        return 0
    # Match `**/kernel_<id>/**/*.pb`: the kernel-id segment is part
    # of the per-launch directory hierarchy the tracer writes.
    target_segment = f"kernel_{kernel_id}"
    total = 0
    for tb_path in sorted(threadblocks_root.rglob("*.pb")):
        if target_segment not in tb_path.parts:
            continue
        tb = threadblock_pb2_mod.threadblock()
        tb.ParseFromString(tb_path.read_bytes())
        for _warp_id, warp in tb.warps.items():
            total += len(warp.instructions)
    return total


def emit_nvbit_trace_summary(
    *,
    traces_dir: Path,
    slug: str,
    shape_point: str,
    repeat_index: int,
    launch_index: int,
    executable_path: Path,
    tracer_so_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Parse the NVBit trace artifacts under `traces_dir` and emit a
    `nvbit_trace_summary_<repeat>_<launch>.json` file at
    `output_path`. Returns the emitted record as a dict.

    Raises a hard protobuf dependency error if the runtime prerequisite or a
    tracked binding authority is unavailable.

    Raises `RuntimeError` if `dynamic_trace.pb` is missing or empty,
    or if the executable / tracer .so paths cannot be hashed.
    """
    trace_pb2_mod, threadblock_pb2_mod = _import_pb_modules()

    trace_pb_path = traces_dir / "dynamic_trace.pb"
    if not trace_pb_path.is_file():
        raise RuntimeError(
            f"dynamic_trace.pb not found at {trace_pb_path}; expected "
            "the NVBit trace .pb to be present in traces_dir"
        )
    threadblocks_root = traces_dir / "threadblocks"

    kernel_metadata, kernel_id = _extract_kernel_metadata_for_launch(
        trace_pb2_mod, trace_pb_path, launch_index,
    )
    inst_count = _sum_inst_count_for_kernel(
        threadblock_pb2_mod, threadblocks_root, kernel_id,
    )
    kernel_signature = compute_kernel_signature_sha256(kernel_metadata)
    binary_hash = sha256_of_file(executable_path)
    trace_pb_hash = sha256_of_file(trace_pb_path)
    tracer_build_hash = sha256_of_file(tracer_so_path)

    record: dict[str, Any] = {
        "schema_version": 1,
        "slug": slug,
        "shape_point": shape_point,
        "repeat_index": repeat_index,
        "launch_index": launch_index,
        "kernel_name": kernel_metadata["kernel_name"],
        "kernel_signature_sha256": kernel_signature,
        "inst_count": inst_count,
        "binary_sha256": binary_hash,
        "nvbit_trace_pb_path": _repo_relative_or_absolute(trace_pb_path),
        "nvbit_trace_pb_sha256": trace_pb_hash,
        "nvbit_tracer_build_hash": tracer_build_hash,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")
    return record


__all__ = [
    "KERNEL_SIGNATURE_FIELD_ORDER",
    "NvbitProtobufBindingsMissing",
    "NvbitProtobufRuntimePrerequisiteMissing",
    "NvbitTrackedProtobufCorruption",
    "canonical_serialize",
    "compute_kernel_signature_sha256",
    "emit_nvbit_trace_summary",
    "enumerate_launches",
    "sha256_of_file",
]
