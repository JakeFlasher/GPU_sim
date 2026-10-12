"""NVBit-side adapter for the offline two-run join.

Provides three surfaces consumed by `merge_evidence.py join-cupti-nvbit`,
`run_suite.py --mode trace`, and diagnostic callers:

1. `compute_kernel_signature_sha256(inputs)` — production NVBit-side
   reference implementation of `workloads/simbench_bw/schemas/
   kernel_signature.spec.md`. Applied to every captured workload
   under `the captured workload corpus` by
   captured workload corpus. The
   implementation mirrors the spec byte-for-byte: tuple-ordered field
   iteration, JSON canonical serialization with `ensure_ascii=False`
   and `separators=(",", ":")`, SHA-256 hex-lowercase 64 chars.

2. `emit_admitted_nvbit_trace_summary(...)` — evidence-producing helper
   invoked by `run_suite.py` after forward admission. It consumes only immutable
   launch metadata, instruction counts, root identity, payload-manifest identity,
   and complete admission seal retained by `TraceAdmission`; it never reopens the
   trace root or threadblock payloads.

3. `emit_nvbit_trace_summary(...)` — compatibility wrapper for callers that do
   not already hold an admitted snapshot. It performs the same complete forward
   trace admission before emitting; no direct parser may fabricate launch metadata.

Both parser surfaces require the `google.protobuf` Python runtime and tracked
`_pb_generated/*_pb2.py` authorities during admission or direct parsing. A
missing runtime is a host prerequisite failure; a missing or unusable tracked
binding is repository corruption. Neither condition is repaired by generating
files in the checkout.

The surfaces are intentionally split so the conformance test can
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ensure_directory,
    replace_regular_bytes,
    write_regular_bytes_at,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    EXPECTED_MEMORY_PAYLOAD_CLASSES,
    EXPECTED_TRACE_CONTRACT_ID,
    TRACE_ADMISSION_SCOPE,
    TraceAdmission,
    require_trace_admission,
)
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    ProtoBindingIntegrityError,
    ProtoBindingRuntimePrerequisiteError,
    load_dynamic_trace_bindings,
)


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
# `the captured workload corpus`.
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
    enforce this equivalence against every captured workload under
    `the captured workload corpus`.

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


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex")
    return value


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


def _import_pb_modules() -> tuple[Any, Any]:
    """Return centrally authenticated trace and threadblock authorities."""
    try:
        bindings = load_dynamic_trace_bindings()
    except ProtoBindingRuntimePrerequisiteError as exc:
        raise NvbitProtobufRuntimePrerequisiteMissing(str(exc)) from exc
    except ProtoBindingIntegrityError as exc:
        raise NvbitTrackedProtobufCorruption(str(exc)) from exc
    return bindings.trace_pb2, bindings.threadblock_pb2


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
    admission = require_trace_admission(trace_pb_path.parent)
    if launch_index < 0 or launch_index >= admission.launch_count:
        raise IndexError(
            f"launch_index {launch_index} out of range; admitted snapshot at "
            f"{trace_pb_path} carries {admission.launch_count} kernel record(s)"
        )
    launch = admission.launches[launch_index]
    return launch.kernel_signature_inputs(), launch.kernel_id


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


def _write_summary_record(
    *,
    kernel_metadata: dict[str, Any],
    inst_count: int,
    trace_pb_path: str,
    trace_pb_hash: str,
    slug: str,
    shape_point: str,
    repeat_index: int,
    launch_index: int,
    binary_sha256: str,
    tracer_build_sha256: str,
    output_path: Path,
    output_descriptor: int | None = None,
    threadblock_payload_size_bytes: int | None = None,
    threadblock_payload_manifest_sha256: str | None = None,
    trace_admission_sha256: str | None = None,
) -> dict[str, Any]:
    admission_fields = (
        threadblock_payload_size_bytes,
        threadblock_payload_manifest_sha256,
        trace_admission_sha256,
    )
    if all(value is None for value in admission_fields):
        schema_version = 1
    elif any(value is None for value in admission_fields):
        raise ValueError(
            "admitted summary identity fields must be supplied as one complete set"
        )
    else:
        if (
            type(threadblock_payload_size_bytes) is not int
            or threadblock_payload_size_bytes < 0
        ):
            raise ValueError(
                "threadblock_payload_size_bytes must be a nonnegative integer"
            )
        schema_version = 2

    record: dict[str, Any] = {
        "schema_version": schema_version,
        "slug": slug,
        "shape_point": shape_point,
        "repeat_index": repeat_index,
        "launch_index": launch_index,
        "kernel_name": kernel_metadata["kernel_name"],
        "kernel_signature_sha256": compute_kernel_signature_sha256(
            kernel_metadata
        ),
        "inst_count": inst_count,
        "binary_sha256": _require_sha256(
            binary_sha256, label="binary_sha256"
        ),
        "nvbit_trace_pb_path": trace_pb_path,
        "nvbit_trace_pb_sha256": _require_sha256(
            trace_pb_hash, label="trace_pb_hash"
        ),
        "nvbit_tracer_build_hash": _require_sha256(
            tracer_build_sha256, label="tracer_build_sha256"
        ),
    }
    if schema_version == 2:
        record.update(
            {
                "nvbit_threadblock_payload_size_bytes": (
                    threadblock_payload_size_bytes
                ),
                "nvbit_threadblock_payload_manifest_sha256": _require_sha256(
                    threadblock_payload_manifest_sha256,
                    label="threadblock_payload_manifest_sha256",
                ),
                "trace_admission_sha256": _require_sha256(
                    trace_admission_sha256,
                    label="trace_admission_sha256",
                ),
            }
        )
    payload = (
        json.dumps(record, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    if output_descriptor is None:
        ensure_directory(output_path.parent)
        replace_regular_bytes(output_path, payload)
    else:
        if output_path.parent != Path("."):
            raise ValueError(
                "descriptor-bound summary output_path must be one relative leaf"
            )
        write_regular_bytes_at(
            output_descriptor,
            output_path.name,
            payload,
        )
    return record


def emit_admitted_nvbit_trace_summary(
    *,
    admission: TraceAdmission,
    slug: str,
    shape_point: str,
    repeat_index: int,
    launch_index: int,
    binary_sha256: str,
    tracer_build_sha256: str,
    output_path: Path,
    output_descriptor: int | None = None,
) -> dict[str, Any]:
    """Emit one summary exclusively from immutable admission-derived inputs.

    The dynamic root and threadblock payload paths are never reopened. Version 2
    binds the instruction count to the admitted payload manifest and complete
    admission seal, alongside root identity and immutable launch metadata.
    """
    if not isinstance(admission, TraceAdmission):
        raise TypeError("admission must be a TraceAdmission value")
    if admission.admission_scope != TRACE_ADMISSION_SCOPE:
        raise RuntimeError(
            "historical-read-only trace admission cannot mint NVBit summary evidence"
        )
    if admission.trace_contract_id != EXPECTED_TRACE_CONTRACT_ID:
        raise RuntimeError(
            "NVBit summary evidence requires the exact split-v2 trace contract"
        )
    if admission.memory_payload_classes != EXPECTED_MEMORY_PAYLOAD_CLASSES:
        raise RuntimeError(
            "NVBit summary evidence requires the canonical capability vector"
        )
    if admission.launch_count != len(admission.launches):
        raise RuntimeError(
            "trace admission launch count disagrees with retained launch inputs"
        )
    if launch_index < 0 or launch_index >= admission.launch_count:
        raise IndexError(
            f"launch_index {launch_index} out of range; admitted snapshot carries "
            f"{admission.launch_count} launch record(s)"
        )
    launch = admission.launches[launch_index]
    if launch.launch_index != launch_index:
        raise RuntimeError(
            "trace admission launch ordering is internally inconsistent"
        )
    return _write_summary_record(
        kernel_metadata=launch.kernel_signature_inputs(),
        inst_count=launch.inst_count,
        trace_pb_path=admission.dynamic_trace_path,
        trace_pb_hash=admission.dynamic_trace_sha256,
        slug=slug,
        shape_point=shape_point,
        repeat_index=repeat_index,
        launch_index=launch_index,
        binary_sha256=binary_sha256,
        tracer_build_sha256=tracer_build_sha256,
        output_path=output_path,
        output_descriptor=output_descriptor,
        threadblock_payload_size_bytes=(
            admission.threadblock_payload_size_bytes
        ),
        threadblock_payload_manifest_sha256=(
            admission.threadblock_payload_manifest_sha256
        ),
        trace_admission_sha256=admission.trace_admission_sha256,
    )


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
    admission = require_trace_admission(traces_dir)
    return emit_admitted_nvbit_trace_summary(
        admission=admission,
        slug=slug,
        shape_point=shape_point,
        repeat_index=repeat_index,
        launch_index=launch_index,
        binary_sha256=sha256_of_file(executable_path),
        tracer_build_sha256=sha256_of_file(tracer_so_path),
        output_path=output_path,
    )


__all__ = [
    "KERNEL_SIGNATURE_FIELD_ORDER",
    "NvbitProtobufBindingsMissing",
    "NvbitProtobufRuntimePrerequisiteMissing",
    "NvbitTrackedProtobufCorruption",
    "canonical_serialize",
    "compute_kernel_signature_sha256",
    "emit_admitted_nvbit_trace_summary",
    "emit_nvbit_trace_summary",
    "enumerate_launches",
    "sha256_of_file",
]
