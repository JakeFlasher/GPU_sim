#!/usr/bin/env python3
"""Fail-closed enhanced protobuf -> legacy ``.traceg`` adapter probe.

The adapter is intentionally a very small representation check, not a timing
frontend.  It admits one SM86 launch containing one CTA/warp and only plain,
full-mask ``IADD3``/``FADD`` records.  Static opcode text and operands come
from the official control-bits JSON sidecar; the dynamic protobuf contributes
only execution identity, masks, PCs, and launch metadata.  A dynamic ``EXIT``
record may terminate the projection when it is actually present in the
protobuf.  Memory, branches, predicates, and every other instruction are
refused rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# These are resource-safety bounds, not a claim about the size of a native
# trace.  The adapter is deliberately limited to a tiny common endpoint.
MAX_RECORDS = 64
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_THREADBLOCK_BYTES = 64 * 1024 * 1024
MAX_SIDECAR_BYTES = 64 * 1024 * 1024
MAX_TRACEG_BYTES = 64 * 1024 * 1024
MAX_KERNEL_NAME_BYTES = 4096
WARP_MASK = 0xFFFF_FFFF
UINT32_MAX = 0xFFFF_FFFF
MAX_U64 = 0xFFFF_FFFF_FFFF_FFFF
_REGISTER_RE = re.compile(r"^R(?:Z|[0-9]+)$", re.IGNORECASE)

# Memory is intentionally not projected.  Keeping these names out of the
# admitted set prevents an enum value from being mistaken for a complete
# memory implementation.
_ALLOWED_ALU = frozenset({"IADD3", "FADD"})


class RefusalCode(str, Enum):
    """Stable machine-readable reasons for a rejected projection."""

    PROTOBUF_RUNTIME_MISSING = "protobuf_runtime_missing"
    INPUT_MISSING = "input_missing"
    INPUT_LIMIT_EXCEEDED = "input_limit_exceeded"
    ROOT_DECODE = "root_decode_failed"
    ROOT_CARDINALITY = "root_cardinality_not_one"
    ROOT_NOT_SM86 = "root_not_sm86"
    LAUNCH_SHAPE_UNSUPPORTED = "launch_shape_unsupported"
    LAUNCH_METADATA_MISSING = "launch_metadata_missing"
    THREADBLOCK_DECODE = "threadblock_decode_failed"
    THREADBLOCK_SHAPE_UNSUPPORTED = "threadblock_shape_unsupported"
    EMPTY_TRACE = "empty_trace"
    RECORD_LIMIT_EXCEEDED = "record_limit_exceeded"
    MASK_METADATA_MISSING = "mask_metadata_missing"
    MASK_NOT_FULL_WARP = "mask_not_full_warp"
    VPC_MISSING = "vpc_missing"
    VPC_PC_MISMATCH = "vpc_pc_mismatch"
    FUNCTION_JOIN_MISMATCH = "function_join_mismatch"
    SIDECAR_MISSING = "sidecar_missing"
    SIDECAR_DECODE = "sidecar_decode_failed"
    SIDECAR_SHAPE = "sidecar_shape_unsupported"
    SIDECAR_ARCHITECTURE = "sidecar_architecture_unsupported"
    SIDECAR_DUPLICATE_JOIN = "sidecar_duplicate_join"
    SIDECAR_PC_MISSING = "sidecar_pc_missing"
    SIDECAR_OPCODE_MISSING = "sidecar_opcode_missing"
    SIDECAR_STATUS_UNSUPPORTED = "sidecar_status_unsupported"
    UNSUPPORTED_OPCODE = "unsupported_opcode"
    PREDICATED_STATIC = "predicated_static_instruction"
    UNSUPPORTED_OPERAND = "unsupported_operand"
    REGISTER_ARITY = "register_arity_unsupported"
    REGISTER_TOKEN = "register_token_unsupported"
    MEMORY_UNSUPPORTED = "memory_projection_unsupported"
    MEMORY_METADATA = "memory_metadata_invalid"
    ADDRESS_ENCODING = "address_encoding_unsupported"
    ADDRESS_VALUE = "address_value_invalid"
    TERMINATION_INCOMPLETE = "termination_incomplete"
    TERMINATION_MALFORMED = "termination_malformed"
    OUTPUT_EXISTS = "output_exists"
    OUTPUT_WRITE = "output_write_failed"
    TRACEG_PARSE = "traceg_parse_failed"
    ROUNDTRIP_MISMATCH = "roundtrip_mismatch"


class AdapterRefusal(Exception):
    """Typed fail-closed adapter refusal.

    This is deliberately a normal mutable ``Exception``.  A frozen dataclass
    cannot accept the traceback assignments performed by ``BaseException`` and
    consequently hides the useful source location of a refusal.
    """

    def __init__(
        self,
        code: RefusalCode,
        detail: str,
        record_index: int | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.record_index = record_index
        super().__init__(detail)

    def __str__(self) -> str:
        where = "" if self.record_index is None else f" record={self.record_index}"
        return f"{self.code.value}{where}: {self.detail}"


@dataclass(frozen=True)
class CanonicalRecord:
    """Fields representable by the legacy trace parser."""

    block: tuple[int, int, int]
    warp: int
    pc: int
    mask: int
    opcode: str
    dest_regs: tuple[str, ...]
    src_regs: tuple[str, ...]
    mem_width: int
    addresses: tuple[int, ...]
    immediate: int = 0

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["block"] = list(self.block)
        value["dest_regs"] = list(self.dest_regs)
        value["src_regs"] = list(self.src_regs)
        value["addresses"] = list(self.addresses)
        return value


@dataclass(frozen=True)
class _StaticInstruction:
    pc: int
    opcode_text: str
    predicate: str | None
    status: str


@dataclass(frozen=True)
class _ControlSidecar:
    arch: str
    functions: dict[str, dict[int, _StaticInstruction]]
    payload: Mapping[str, Any]


def _bounded_read(path: Path, limit: int, code: RefusalCode) -> bytes:
    """Read at most ``limit + 1`` bytes, classifying every file failure."""

    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except FileNotFoundError as exc:
        raise AdapterRefusal(RefusalCode.INPUT_MISSING, f"not a file: {path}") from exc
    except OSError as exc:
        raise AdapterRefusal(code, f"cannot read {path}: {exc}") from exc
    if len(raw) > limit:
        raise AdapterRefusal(
            RefusalCode.INPUT_LIMIT_EXCEEDED,
            f"{path} exceeds {limit} byte input limit",
        )
    return raw


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    """Compatibility helper with the same bounded-input behavior as ``adapt``."""

    return _sha256_bytes(_bounded_read(path, MAX_TRACEG_BYTES, RefusalCode.INPUT_MISSING))


def _has(message: Any, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except (ValueError, AttributeError):
        return False


def _load_bindings() -> Any:
    try:
        from tools.nvbit_tracer.proto_binding_loader import (
            ProtoBindingRuntimePrerequisiteError,
            load_dynamic_trace_bindings,
        )

        return load_dynamic_trace_bindings()
    except Exception as exc:  # keep dependency failures typed for CLI users
        if exc.__class__.__name__ == "ProtoBindingRuntimePrerequisiteError":
            raise AdapterRefusal(
                RefusalCode.PROTOBUF_RUNTIME_MISSING,
                str(exc),
            ) from exc
        raise


def _int_field(value: Any, label: str, code: RefusalCode = RefusalCode.LAUNCH_METADATA_MISSING) -> int:
    try:
        # Do not coerce strings/floats (or booleans) into metadata integers.
        # JSON ``1.5`` -> ``int(1.5)`` and ``"1"`` -> ``int("1")`` would
        # otherwise turn malformed sidecars into a different valid join key.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("value is not an integer")
        return value
    except (TypeError, ValueError, OverflowError) as exc:
        raise AdapterRefusal(code, f"{label} is not an integer: {value!r}") from exc


def _dim3(value: Any, label: str) -> tuple[int, int, int]:
    if value is None:
        raise AdapterRefusal(RefusalCode.LAUNCH_METADATA_MISSING, f"{label} absent")
    return tuple(_int_field(getattr(value, axis, None), f"{label}.{axis}") for axis in ("x", "y", "z"))  # type: ignore[return-value]


def _load_root_and_kernel(trace_bytes: bytes, bindings: Any) -> tuple[Any, Any, int]:
    root = bindings.trace_pb2.Trace()
    try:
        root.ParseFromString(trace_bytes)
    except Exception as exc:
        raise AdapterRefusal(RefusalCode.ROOT_DECODE, f"protobuf root cannot be decoded: {exc}") from exc
    if len(root.gpu_device) != 1:
        raise AdapterRefusal(
            RefusalCode.ROOT_CARDINALITY,
            f"expected one device, got {len(root.gpu_device)}",
        )
    device_id, device = next(iter(root.gpu_device.items()))
    if _int_field(device_id, "device id") != 0:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            f"device id={device_id}; only device 0 is representable",
        )
    if _int_field(root.binary_version, "binary_version") != 86:
        raise AdapterRefusal(
            RefusalCode.ROOT_NOT_SM86,
            f"binary_version={int(root.binary_version)}; only SM86 is admitted",
        )
    if len(device.streams) != 1:
        raise AdapterRefusal(
            RefusalCode.ROOT_CARDINALITY,
            f"expected one CUDA stream, got {len(device.streams)}",
        )
    stream_id, stream = next(iter(device.streams.items()))
    if _int_field(stream_id, "stream id") < 0:
        raise AdapterRefusal(RefusalCode.LAUNCH_METADATA_MISSING, f"stream id={stream_id} is negative")
    if len(stream.kernels) != 1:
        raise AdapterRefusal(
            RefusalCode.ROOT_CARDINALITY,
            f"expected one kernel, got {len(stream.kernels)}",
        )
    kernel = stream.kernels[0]
    if _int_field(kernel.id, "kernel.id") <= 0 or _int_field(kernel.function_unique_id, "kernel.function_unique_id") <= 0:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "kernel id and function_unique_id must be positive",
        )
    kernel_name = str(kernel.name)
    if (
        not kernel_name.strip()
        or len(kernel_name.encode("utf-8")) > MAX_KERNEL_NAME_BYTES
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in kernel_name)
    ):
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "kernel name is empty, oversized, or contains control characters",
        )
    grid = _dim3(kernel.grid_dim, "grid_dim")
    block = _dim3(kernel.block_dim, "block_dim")
    if grid != (1, 1, 1) or block != (32, 1, 1):
        raise AdapterRefusal(
            RefusalCode.LAUNCH_SHAPE_UNSUPPORTED,
            f"expected grid=(1,1,1), block=(32,1,1), got {grid}/{block}",
        )
    cluster = _dim3(kernel.cluster_dim, "cluster_dim")
    if cluster not in {(0, 0, 0), (1, 1, 1)}:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_SHAPE_UNSUPPORTED,
            f"cluster_dim={cluster}; clustered launches are outside the one-CTA contract",
        )
    nregs = _int_field(kernel.number_of_registers, "number_of_registers")
    shared = _int_field(kernel.size_shared_memory, "size_shared_memory")
    if nregs <= 0 or shared < 0:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "nregs must be positive and shared-memory size nonnegative",
        )
    if not _has(kernel, "shared_memory_static_bytes") or not _has(kernel, "shared_memory_dynamic_bytes"):
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "explicit shared-memory static/dynamic components are required",
        )
    static = _int_field(kernel.shared_memory_static_bytes, "shared_memory_static_bytes")
    dynamic = _int_field(kernel.shared_memory_dynamic_bytes, "shared_memory_dynamic_bytes")
    if static < 0 or dynamic < 0 or static + dynamic != shared:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            f"shared-memory components {static}+{dynamic} do not equal {shared}",
        )
    for field in ("shared_memory_base_address", "local_memory_base_address"):
        address = _int_field(getattr(kernel, field), field)
        if not 0 <= address <= MAX_U64:
            raise AdapterRefusal(
                RefusalCode.LAUNCH_METADATA_MISSING,
                f"{field}={address} is outside the non-negative uint64 header range",
            )
    return root, kernel, _int_field(stream_id, "stream id")


def _strict_json(raw: bytes, path: Path) -> Mapping[str, Any]:
    def pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise AdapterRefusal(RefusalCode.SIDECAR_DECODE, f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out

    def constant(value: str) -> Any:
        raise AdapterRefusal(RefusalCode.SIDECAR_DECODE, f"invalid JSON constant {value!r} in {path}")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except AdapterRefusal:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterRefusal(RefusalCode.SIDECAR_DECODE, f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, f"sidecar root must be an object: {path}")
    return value


def _strip_instance_suffix(name: str) -> str:
    marker = name.rfind("___")
    if marker >= 0 and name[marker + 3 :].isdigit() and name[marker + 3 :]:
        return name[:marker]
    return name


def _load_control_sidecar(raw: bytes, path: Path) -> _ControlSidecar:
    payload = _strict_json(raw, path)
    arch_value = payload.get("arch", "")
    if not isinstance(arch_value, str):
        raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, "control sidecar arch must be text")
    arch = arch_value.strip().lower()
    if arch not in {"sm_86", "sm86", "86"}:
        raise AdapterRefusal(RefusalCode.SIDECAR_ARCHITECTURE, f"control sidecar arch={arch!r}; expected sm_86")
    functions_raw = payload.get("functions")
    if not isinstance(functions_raw, Mapping) or not functions_raw:
        # enhanced_execution_info.json has `kernels` and `op_code`; it is an
        # identity sidecar, not a source of authoritative opcode text.
        raise AdapterRefusal(
            RefusalCode.SIDECAR_SHAPE,
            "official control sidecar must contain non-empty functions mapping",
        )
    functions: dict[str, dict[int, _StaticInstruction]] = {}
    for function_name, records in functions_raw.items():
        if (
            not isinstance(function_name, str)
            or not function_name.strip()
            or len(function_name.encode("utf-8")) > MAX_KERNEL_NAME_BYTES
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in function_name)
        ):
            raise AdapterRefusal(
                RefusalCode.SIDECAR_SHAPE,
                "function name must be a bounded string without control characters",
            )
        # Official format maps name -> records.  Accept a small wrapper with
        # `instructions` only to make synthetic fixtures explicit, never a
        # legacy op_code-only list.
        if isinstance(records, Mapping):
            records = records.get("instructions")
        if not isinstance(records, list) or not records:
            raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, f"function {function_name!r} has no instruction list")
        joined: dict[int, _StaticInstruction] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, f"function {function_name!r} has malformed instruction")
            if "pc" not in record:
                raise AdapterRefusal(RefusalCode.SIDECAR_PC_MISSING, f"function {function_name!r} instruction has no pc")
            pc = _int_field(record.get("pc"), f"{function_name}.pc", RefusalCode.SIDECAR_PC_MISSING)
            if pc < 0 or pc > UINT32_MAX:
                raise AdapterRefusal(RefusalCode.SIDECAR_PC_MISSING, f"pc {pc} is outside uint32")
            if pc in joined:
                raise AdapterRefusal(RefusalCode.SIDECAR_DUPLICATE_JOIN, f"duplicate static key ({function_name!r}, {pc})")
            text = record.get("opcode_text")
            if not isinstance(text, str) or not text.strip():
                raise AdapterRefusal(RefusalCode.SIDECAR_OPCODE_MISSING, f"{function_name} pc={pc} has no opcode_text")
            status = str(record.get("status", "decoded"))
            if status != "decoded":
                raise AdapterRefusal(RefusalCode.SIDECAR_STATUS_UNSUPPORTED, f"{function_name} pc={pc} status={status!r}")
            predicate = record.get("predicate")
            if predicate is not None and not isinstance(predicate, str):
                raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, f"{function_name} pc={pc} predicate is not text/null")
            joined[pc] = _StaticInstruction(pc, text.strip(), predicate, status)
        functions[function_name] = joined
    return _ControlSidecar(arch, functions, payload)


def _load_exec_info(raw: bytes, path: Path, function_id: int) -> tuple[str, dict[int, Any]]:
    payload = _strict_json(raw, path)
    kernels = payload.get("kernels")
    if not isinstance(kernels, list):
        raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, "exec-info identity sidecar must contain kernels list")
    matches = [item for item in kernels if isinstance(item, Mapping) and _int_field(item.get("unique_function_id", -1), "unique_function_id", RefusalCode.FUNCTION_JOIN_MISMATCH) == function_id]
    if len(matches) != 1:
        raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"expected one exec-info kernel for function_unique_id={function_id}, got {len(matches)}")
    kernel = matches[0]
    name = kernel.get("kernel_name")
    if not isinstance(name, str) or not name.strip():
        raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, "exec-info kernel_name missing")
    if _int_field(kernel.get("architecture_version", -1), "architecture_version", RefusalCode.FUNCTION_JOIN_MISMATCH) != 86:
        raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, "exec-info architecture_version is not 86")
    static = kernel.get("instructions")
    if not isinstance(static, list) or not static:
        raise AdapterRefusal(RefusalCode.SIDECAR_SHAPE, "exec-info instructions list absent")
    pcs: dict[int, Any] = {}
    for item in static:
        if not isinstance(item, Mapping) or "pc_num_dec" not in item:
            raise AdapterRefusal(RefusalCode.SIDECAR_PC_MISSING, "exec-info instruction has no pc_num_dec")
        pc = _int_field(item.get("pc_num_dec"), "pc_num_dec", RefusalCode.SIDECAR_PC_MISSING)
        if pc < 0 or pc > UINT32_MAX or pc in pcs:
            raise AdapterRefusal(RefusalCode.SIDECAR_DUPLICATE_JOIN, f"duplicate/invalid exec-info pc {pc}")
        hex_text = item.get("pc_string_hex")
        if not isinstance(hex_text, str):
            raise AdapterRefusal(RefusalCode.SIDECAR_PC_MISSING, f"exec-info pc={pc} has no pc_string_hex")
        try:
            if int(hex_text, 16) != pc:
                raise ValueError
        except ValueError as exc:
            raise AdapterRefusal(RefusalCode.SIDECAR_PC_MISSING, f"exec-info pc transcription mismatch for {pc}") from exc
        pcs[pc] = item
    return _strip_instance_suffix(name), pcs


def _resolve_function(
    control: _ControlSidecar,
    kernel: Any,
    function_id: int,
    exec_info: tuple[str, dict[int, Any]] | None,
) -> tuple[str, dict[int, _StaticInstruction], dict[int, Any] | None]:
    kernel_name = str(kernel.name)
    unsuffixed = _strip_instance_suffix(kernel_name)
    candidate = unsuffixed
    if exec_info is not None:
        exec_name, exec_pcs = exec_info
        if exec_name != unsuffixed and exec_name != kernel_name:
            raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"kernel name {kernel_name!r} disagrees with exec-info {exec_name!r}")
        candidate = exec_name
    # The official sidecar normally stores the unsuffixed function name, but a
    # capture may retain the exact instance spelling.  Permit those two exact
    # keys only; never search by a fuzzy substring or arbitrary suffix.
    exact_candidates = tuple(
        key for key in dict.fromkeys((candidate, kernel_name, unsuffixed)) if key in control.functions
    )
    if len(exact_candidates) != 1:
        if not exact_candidates:
            detail = f"no exact control function join for {candidate!r}"
        else:
            detail = f"ambiguous exact control function joins {list(exact_candidates)!r}"
        raise AdapterRefusal(
            RefusalCode.FUNCTION_JOIN_MISMATCH,
            f"{detail} (function_unique_id={function_id})",
        )
    candidate = exact_candidates[0]
    static = control.functions[candidate]
    if exec_info is not None:
        missing = sorted(set(exec_info[1]) - set(static))
        extra = sorted(set(static) - set(exec_info[1]))
        if missing or extra:
            raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"control/exec-info PC sets differ missing={missing[:3]} extra={extra[:3]}")
    return candidate, static, None if exec_info is None else exec_info[1]


def _canonical_register(value: Any, record_index: int, role: str) -> str:
    token = str(value).strip().upper()
    if token == "RZ":
        return "R255"
    if not _REGISTER_RE.fullmatch(token):
        raise AdapterRefusal(RefusalCode.REGISTER_TOKEN, f"{role} token {token!r} is not an ordinary R register", record_index)
    try:
        number = 255 if token == "RZ" else int(token[1:])
    except ValueError as exc:
        raise AdapterRefusal(RefusalCode.REGISTER_TOKEN, f"register {token!r} is malformed", record_index) from exc
    if not 0 <= number <= 255:
        raise AdapterRefusal(RefusalCode.REGISTER_TOKEN, f"register {token!r} exceeds R255", record_index)
    return f"R{number}"


def _parse_immediate(token: str, record_index: int) -> int:
    try:
        # SASS constants in this admitted subset are integer literals only.
        if not re.fullmatch(r"[-+]?(?:0[xX][0-9a-fA-F]+|[0-9]+)", token):
            raise ValueError
        # ``int(text, 0)`` rejects decimal spellings with a leading zero
        # (e.g. ``08``), although that is an unambiguous decimal SASS literal.
        base = 16 if re.fullmatch(r"[-+]?0[xX][0-9a-fA-F]+", token) else 10
        value = int(token, base)
    except ValueError as exc:
        raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPERAND, f"operand {token!r} is not a supported integer literal", record_index) from exc
    if not -(1 << 63) <= value <= (1 << 63) - 1:
        raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPERAND, f"immediate {value} does not fit signed 64 bits", record_index)
    return value


def _static_operands(static: _StaticInstruction, record_index: int) -> tuple[str, tuple[str, ...], int]:
    text = static.opcode_text.strip()
    parts = text.split(None, 1)
    opcode = parts[0] if parts else ""
    separator = len(parts) == 2
    rest = parts[1] if separator else ""
    if opcode not in _ALLOWED_ALU:
        raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPCODE, f"opcode {opcode!r} not in common ALU subset", record_index)
    if not separator or not rest.strip() or static.predicate:
        raise AdapterRefusal(RefusalCode.PREDICATED_STATIC if static.predicate else RefusalCode.UNSUPPORTED_OPERAND, f"static instruction {text!r} is not an unpredicated register form", record_index)
    operands = [part.strip() for part in rest.split(",")]
    expected = 4 if opcode == "IADD3" else 3
    if len(operands) != expected:
        raise AdapterRefusal(RefusalCode.REGISTER_ARITY, f"{opcode} requires {expected} operands, got {len(operands)}", record_index)
    dest = _canonical_register(operands[0], record_index, "destination")
    src: list[str] = []
    immediate = 0
    immediate_count = 0
    for token in operands[1:]:
        if _REGISTER_RE.fullmatch(token):
            src.append(_canonical_register(token, record_index, "source"))
        elif opcode == "IADD3":
            immediate_count += 1
            if immediate_count > 1:
                raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPERAND, "IADD3 has more than one immediate", record_index)
            immediate = _parse_immediate(token, record_index)
        else:
            raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPERAND, f"FADD operand {token!r} is not a regular register", record_index)
    if len(src) > 4:
        raise AdapterRefusal(RefusalCode.REGISTER_ARITY, f"source count {len(src)} > 4", record_index)
    return opcode, tuple(src), immediate


def _record_from_messages(
    dynamic: Any,
    static: _StaticInstruction,
    index: int,
    function_id: int,
) -> CanonicalRecord | None:
    if _int_field(dynamic.function_unique_id, "instruction.function_unique_id", RefusalCode.FUNCTION_JOIN_MISMATCH) != function_id:
        raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"record function_unique_id={dynamic.function_unique_id} != kernel={function_id}", index)
    if not _has(dynamic, "active_mask") or not _has(dynamic, "predicate_mask"):
        raise AdapterRefusal(RefusalCode.MASK_METADATA_MISSING, "active/predicate mask presence required", index)
    active = _int_field(dynamic.active_mask, "active_mask", RefusalCode.MASK_METADATA_MISSING)
    predicate = _int_field(dynamic.predicate_mask, "predicate_mask", RefusalCode.MASK_METADATA_MISSING)
    if not 0 <= active <= WARP_MASK or not 0 <= predicate <= WARP_MASK:
        raise AdapterRefusal(RefusalCode.MASK_METADATA_MISSING, "mask does not fit uint32", index)
    mask = active & predicate
    if not _has(dynamic, "vpc"):
        raise AdapterRefusal(RefusalCode.VPC_MISSING, "vpc presence required", index)
    pc = _int_field(dynamic.pc, "pc", RefusalCode.VPC_PC_MISMATCH)
    vpc = _int_field(dynamic.vpc, "vpc", RefusalCode.VPC_PC_MISMATCH)
    if pc < 0 or pc > UINT32_MAX or vpc < 0 or vpc > MAX_U64 or vpc > UINT32_MAX or (vpc & UINT32_MAX) != pc:
        raise AdapterRefusal(RefusalCode.VPC_PC_MISMATCH, f"pc=0x{pc:x}, vpc=0x{vpc:x}", index)
    if not _has(dynamic, "warpid_tb") or _int_field(dynamic.warpid_tb, "warpid_tb", RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED) != 0:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED, "record is not warp 0", index)
    for field in ("warpid_sm", "sm_id"):
        if _has(dynamic, field) and _int_field(
            getattr(dynamic, field), field, RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED
        ) < 0:
            raise AdapterRefusal(
                RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED,
                f"{field} is negative",
                index,
            )
    static_opcode = static.opcode_text.split(None, 1)[0] if static.opcode_text else ""
    if static_opcode == "EXIT":
        if static.opcode_text.strip() != "EXIT":
            raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, f"EXIT has unexpected operands {static.opcode_text!r}", index)
        if mask not in (0, WARP_MASK):
            raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, f"EXIT effective mask=0x{mask:08x} is partial", index)
        if (
            len(dynamic.addresses) != 0
            or _has(dynamic, "memory_op_class")
            or _has(dynamic, "memory_op_access_kind")
            or _has(dynamic, "data_width")
        ):
            raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, "EXIT carries memory metadata", index)
        return None
    if static_opcode not in _ALLOWED_ALU:
        # This includes all memory classes: no enum value is interpreted as a
        # complete address projection.
        if (
            len(dynamic.addresses)
            or _has(dynamic, "memory_op_class")
            or _has(dynamic, "memory_op_access_kind")
            or _has(dynamic, "data_width")
        ):
            raise AdapterRefusal(RefusalCode.MEMORY_UNSUPPORTED, f"memory opcode {static_opcode!r} is outside the validated projection", index)
        raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPCODE, f"opcode {static_opcode!r} not in common ALU subset", index)
    if mask != WARP_MASK:
        raise AdapterRefusal(RefusalCode.MASK_NOT_FULL_WARP, f"effective mask=0x{mask:08x}", index)
    if (
        len(dynamic.addresses) != 0
        or _has(dynamic, "memory_op_class")
        or _has(dynamic, "memory_op_access_kind")
        or _has(dynamic, "data_width")
    ):
        raise AdapterRefusal(RefusalCode.MEMORY_METADATA, "ALU record carries memory metadata", index)
    opcode, src_regs, immediate = _static_operands(static, index)
    return CanonicalRecord((0, 0, 0), 0, pc, mask, opcode, ( _canonical_register(static.opcode_text.split(None, 1)[1].split(",", 1)[0], index, "destination"), ), src_regs, 0, (), immediate)


def _format_record(record: CanonicalRecord) -> str:
    parts: list[str] = [f"{record.pc:04x}", f"{record.mask:08x}", str(len(record.dest_regs))]
    parts.extend(record.dest_regs)
    parts += [record.opcode, str(len(record.src_regs))]
    parts.extend(record.src_regs)
    parts.extend([str(record.mem_width), str(record.immediate)])
    return " ".join(parts)


def _header(root: Any, kernel: Any, stream_id: int) -> list[str]:
    shared = _int_field(kernel.shared_memory_base_address, "shared_memory_base_address")
    local = _int_field(kernel.local_memory_base_address, "local_memory_base_address")
    return [
        f"-kernel name = {kernel.name}",
        f"-kernel id = {_int_field(kernel.id, 'kernel.id')}",
        "-grid dim = (1,1,1)",
        "-block dim = (32,1,1)",
        f"-shmem = {_int_field(kernel.size_shared_memory, 'size_shared_memory')}",
        f"-nregs = {_int_field(kernel.number_of_registers, 'number_of_registers')}",
        f"-binary version = {_int_field(root.binary_version, 'binary_version')}",
        f"-cuda stream id = {stream_id}",
        f"-shmem base_addr = 0x{shared & MAX_U64:016x}",
        f"-local mem base_addr = 0x{local & MAX_U64:016x}",
        f"-nvbit version = {root.nvbit_version}",
        f"-accelsim tracer version = {_int_field(root.accelsim_version, 'accelsim_version')}",
        "-enable lineinfo = 0",
        "",
        "#traces format = PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width immediate",
        "",
    ]


def _emit_traceg_bytes(root: Any, kernel: Any, stream_id: int, records: Sequence[CanonicalRecord], terminal_pc: int) -> bytes:
    if terminal_pc < 0 or terminal_pc > UINT32_MAX:
        raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, f"terminal PC {terminal_pc} is outside uint32")
    lines = _header(root, kernel, stream_id)
    lines += [
        "#BEGIN_TB",
        "thread block = 0,0,0",
        "cluster id = 0,0,0",
        "cluster cta = 0,0,0",
        "cluster rank = 0",
        "warp = 0",
        f"insts = {len(records) + 1}",
    ]
    lines.extend(_format_record(record) for record in records)
    # This line corresponds to an actual dynamic EXIT; no sidecar EOF marker
    # or inferred PC is ever used.
    lines.append(f"{terminal_pc:04x} 00000000 0 EXIT 0 0 0")
    lines.append("#END_TB")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_record_tokens(tokens: list[str], line_number: int) -> CanonicalRecord:
    try:
        cursor = 0
        pc = int(tokens[cursor], 16); cursor += 1
        mask = int(tokens[cursor], 16); cursor += 1
        dest_count = int(tokens[cursor]); cursor += 1
        if (
            not 0 <= pc <= UINT32_MAX
            or not 0 <= mask <= UINT32_MAX
            or mask != WARP_MASK
            or not 0 <= dest_count <= 1
        ):
            raise ValueError("PC/mask/destination count out of range")
        dest = tuple(_canonical_register(tokens[cursor + i], 0, "destination") for i in range(dest_count)); cursor += dest_count
        opcode = tokens[cursor]; cursor += 1
        src_count = int(tokens[cursor]); cursor += 1
        if not 0 <= src_count <= 4:
            raise ValueError("source count out of range")
        src = tuple(_canonical_register(tokens[cursor + i], 0, "source") for i in range(src_count)); cursor += src_count
        mem_width = int(tokens[cursor]); cursor += 1
        if mem_width != 0:
            raise ValueError("memory records are outside the adapter contract")
        immediate = _parse_immediate(tokens[cursor], line_number); cursor += 1
        if cursor != len(tokens):
            raise ValueError("trailing tokens")
        if opcode == "FADD" and (src_count != 2 or immediate != 0):
            raise ValueError("FADD requires two sources and no immediate")
        if opcode == "IADD3" and not 2 <= src_count <= 3:
            raise ValueError("IADD3 requires two or three register sources")
    except (IndexError, ValueError, AdapterRefusal) as exc:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: {exc}") from exc
    if opcode == "EXIT":
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: EXIT handled by terminal parser")
    if opcode not in _ALLOWED_ALU:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: unsupported opcode {opcode!r}")
    return CanonicalRecord((0, 0, 0), 0, pc, mask, opcode, dest, src, 0, (), immediate)


def parse_traceg(path: Path) -> tuple[list[CanonicalRecord], bool]:
    """Strictly parse the one-TB subset emitted by this module.

    Unlike the old parser, missing markers, counts, warp metadata, terminal
    records, empty bodies, and trailing non-comment content are refusals.
    """

    raw = _bounded_read(path, MAX_TRACEG_BYTES, RefusalCode.TRACEG_PARSE)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"traceg is not UTF-8: {exc}") from exc
    records: list[CanonicalRecord] = []
    expected_count: int | None = None
    terminal_pc: int | None = None
    in_tb = False
    begin_count = end_count = 0
    seen_fields: set[str] = set()
    seen_headers: set[str] = set()
    header_prefixes = (
        "-kernel name =",
        "-kernel id =",
        "-grid dim =",
        "-block dim =",
        "-shmem =",
        "-nregs =",
        "-binary version =",
        "-cuda stream id =",
        "-shmem base_addr =",
        "-local mem base_addr =",
        "-nvbit version =",
        "-accelsim tracer version =",
        "-enable lineinfo =",
    )

    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "#BEGIN_TB":
            begin_count += 1
            if in_tb or begin_count > 1:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: duplicate/nested #BEGIN_TB")
            in_tb = True
            continue
        if stripped == "#END_TB":
            end_count += 1
            if not in_tb:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: #END_TB without begin")
            in_tb = False
            continue
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith("insts ="):
            if "insts" in seen_fields or terminal_pc is not None or (end_count and not in_tb):
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: misplaced/duplicate insts")
            seen_fields.add("insts")
            try:
                expected_count = int(stripped.split("=", 1)[1].strip())
            except ValueError as exc:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: invalid insts") from exc
            if expected_count <= 0 or expected_count > MAX_RECORDS + 1:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: invalid insts={expected_count}")
            continue
        # A few retained legacy writers place the block/warp fields directly
        # before #BEGIN_TB, while the adapter emitter puts them inside the
        # marker pair.  They describe the same single TB and are accepted in
        # either location, but still require exact values and uniqueness.
        if stripped.startswith("thread block ="):
            if stripped != "thread block = 0,0,0" or "block" in seen_fields or end_count:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed thread block")
            seen_fields.add("block")
            continue
        if stripped.startswith("cluster id ="):
            if stripped != "cluster id = 0,0,0" or "cluster_id" in seen_fields or end_count:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed cluster id")
            seen_fields.add("cluster_id")
            continue
        if stripped.startswith("cluster cta ="):
            if stripped != "cluster cta = 0,0,0" or "cluster_cta" in seen_fields or end_count:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed cluster cta")
            seen_fields.add("cluster_cta")
            continue
        if stripped.startswith("cluster rank ="):
            if stripped != "cluster rank = 0" or "cluster_rank" in seen_fields or end_count:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed cluster rank")
            seen_fields.add("cluster_rank")
            continue
        if stripped.startswith("warp ="):
            if stripped != "warp = 0" or "warp" in seen_fields or end_count:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed warp")
            seen_fields.add("warp")
            continue
        if not in_tb:
            # Header metadata is accepted only before the TB; arbitrary data
            # is never ignored because that used to hide malformed input.
            if begin_count == 0:
                header_key = next(
                    (prefix for prefix in header_prefixes if stripped.startswith(prefix)),
                    None,
                )
                if header_key is not None:
                    if header_key in seen_headers:
                        raise AdapterRefusal(
                            RefusalCode.TRACEG_PARSE,
                            f"line {line_number}: duplicate header {header_key!r}",
                        )
                    seen_headers.add(header_key)
                    continue
            raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: data outside TB")
        if terminal_pc is not None:
            raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: data follows EXIT")
        tokens = stripped.split()
        if len(tokens) >= 4 and tokens[3] == "EXIT":
            if terminal_pc is not None or len(tokens) != 7:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed/duplicate EXIT")
            try:
                terminal_pc = int(tokens[0], 16)
                if (
                    not 0 <= terminal_pc <= UINT32_MAX
                    or int(tokens[1], 16) != 0
                    or int(tokens[2]) != 0
                    or tokens[4] != "0"
                    or tokens[5] != "0"
                    or tokens[6] != "0"
                ):
                    raise ValueError("EXIT fields are not canonical")
            except (ValueError, IndexError) as exc:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: malformed EXIT") from exc
            continue
        if terminal_pc is not None:
            raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: record follows EXIT")
        records.append(_parse_record_tokens(tokens, line_number))
        if len(records) > MAX_RECORDS:
            raise AdapterRefusal(RefusalCode.RECORD_LIMIT_EXCEEDED, f"more than {MAX_RECORDS} records")
    if in_tb or begin_count != 1 or end_count != 1:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, "exactly one closed #BEGIN_TB/#END_TB pair is required")
    required = {"insts", "block", "cluster_id", "cluster_cta", "cluster_rank", "warp"}
    if seen_fields != required:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"missing TB metadata: {sorted(required - seen_fields)}")
    if expected_count is None or terminal_pc is None:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, "insts and an explicit EXIT are required")
    if expected_count != len(records) + 1:
        raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"insts declares {expected_count}, parsed {len(records) + 1}")
    if not records:
        raise AdapterRefusal(RefusalCode.EMPTY_TRACE, "traceg contains no ALU records")
    return records, True


def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            written = handle.write(data)
            if written != len(data):
                raise OSError(f"short write ({written}/{len(data)} bytes)")
    except OSError as exc:
        raise AdapterRefusal(RefusalCode.OUTPUT_WRITE, f"cannot create {path}: {exc}") from exc


def adapt(
    trace_path: Path,
    threadblock_path: Path,
    sidecar_path: Path,
    out_dir: Path,
    *,
    allow_eof: bool = False,
    exec_info_path: Path | None = None,
) -> dict[str, Any]:
    """Adapt one bounded launch using a control sidecar and explicit EXIT.

    ``allow_eof`` remains as a source-compatible keyword but is intentionally
    ignored: EOF cannot establish the legacy terminal record and therefore
    always receives ``termination_incomplete``.
    """

    del allow_eof
    # Refuse an existing destination before decoding inputs.  Besides making
    # overwrite behavior explicit, this avoids doing work against a stale
    # destination whose prior contents must remain untouched.
    if out_dir.exists():
        raise AdapterRefusal(RefusalCode.OUTPUT_EXISTS, f"output directory already exists: {out_dir}")
    bindings = _load_bindings()
    trace_bytes = _bounded_read(trace_path, MAX_TRACE_BYTES, RefusalCode.ROOT_DECODE)
    tb_bytes = _bounded_read(threadblock_path, MAX_THREADBLOCK_BYTES, RefusalCode.THREADBLOCK_DECODE)
    control_bytes = _bounded_read(sidecar_path, MAX_SIDECAR_BYTES, RefusalCode.SIDECAR_MISSING)
    exec_bytes = None if exec_info_path is None else _bounded_read(exec_info_path, MAX_SIDECAR_BYTES, RefusalCode.SIDECAR_MISSING)
    root, kernel, stream_id = _load_root_and_kernel(trace_bytes, bindings)
    tb = bindings.threadblock_pb2.threadblock()
    try:
        tb.ParseFromString(tb_bytes)
    except Exception as exc:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_DECODE, f"threadblock cannot decode: {exc}") from exc
    if not _has(tb, "block_id"):
        raise AdapterRefusal(
            RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED,
            "threadblock.block_id presence is required",
        )
    block = _dim3(tb.block_id, "threadblock.block_id")
    if block != (0, 0, 0) or len(tb.warps) != 1 or 0 not in tb.warps:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED, f"expected block (0,0,0) and exactly warp 0, got block={block} warps={list(tb.warps)}")
    warp = tb.warps[0]
    if _int_field(warp.id, "warp.id", RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED) != 0:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED, f"warp id={warp.id}; expected 0")
    if not warp.instructions:
        raise AdapterRefusal(RefusalCode.EMPTY_TRACE, "warp has no dynamic instructions")
    if len(warp.instructions) > MAX_RECORDS:
        raise AdapterRefusal(RefusalCode.RECORD_LIMIT_EXCEEDED, f"{len(warp.instructions)} > {MAX_RECORDS}")
    control = _load_control_sidecar(control_bytes, sidecar_path)
    function_id = _int_field(kernel.function_unique_id, "kernel.function_unique_id", RefusalCode.FUNCTION_JOIN_MISMATCH)
    identity = None if exec_bytes is None else _load_exec_info(exec_bytes, exec_info_path or sidecar_path, function_id)
    _function_name, static_by_pc, exec_pcs = _resolve_function(control, kernel, function_id, identity)
    records: list[CanonicalRecord] = []
    terminal_pc: int | None = None
    for index, dynamic in enumerate(warp.instructions):
        if not _has(dynamic, "vpc"):
            raise AdapterRefusal(RefusalCode.VPC_MISSING, "vpc presence required", index)
        key_pc = _int_field(dynamic.vpc, "vpc", RefusalCode.VPC_PC_MISMATCH)
        if key_pc > UINT32_MAX:
            raise AdapterRefusal(RefusalCode.VPC_PC_MISMATCH, f"vpc {key_pc} does not fit function-relative uint32", index)
        if key_pc not in static_by_pc:
            raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"no unique control-sidecar join for function={function_id}, pc={key_pc}", index)
        if exec_pcs is not None and key_pc not in exec_pcs:
            raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"dynamic pc={key_pc} absent from exec-info identity table", index)
        static = static_by_pc[key_pc]
        record = _record_from_messages(dynamic, static, index, function_id)
        if record is None:
            if terminal_pc is not None or index != len(warp.instructions) - 1:
                raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, "EXIT must occur exactly once as the final dynamic record", index)
            terminal_pc = record_pc = _int_field(dynamic.pc, "EXIT.pc", RefusalCode.TERMINATION_MALFORMED)
            if record_pc < 0 or record_pc > UINT32_MAX:
                raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, "EXIT pc does not fit uint32", index)
        else:
            if terminal_pc is not None:
                raise AdapterRefusal(RefusalCode.TERMINATION_MALFORMED, "record follows EXIT", index)
            records.append(record)
    if terminal_pc is None:
        raise AdapterRefusal(RefusalCode.TERMINATION_INCOMPLETE, "source has no dynamic EXIT record; EOF cannot be projected")
    if not records:
        raise AdapterRefusal(RefusalCode.EMPTY_TRACE, "dynamic trace contains only EXIT")
    traceg_bytes = _emit_traceg_bytes(root, kernel, stream_id, records, terminal_pc)
    # Parse bytes before creating output so a failed validation leaves no
    # partially-created directory.  The parser itself uses the same bytes as
    # the evidence output, not a separately reread file.
    temp_path = out_dir.parent / f".{out_dir.name}.traceg-parse"
    try:
        _write_exclusive(temp_path, traceg_bytes)
        parsed, terminal = parse_traceg(temp_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    if parsed != records or not terminal:
        raise AdapterRefusal(RefusalCode.ROUNDTRIP_MISMATCH, f"canonical records differ after traceg parse: emitted={len(records)} parsed={len(parsed)}")
    try:
        out_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise AdapterRefusal(RefusalCode.OUTPUT_EXISTS, f"output directory already exists: {out_dir}") from exc
    except OSError as exc:
        raise AdapterRefusal(RefusalCode.OUTPUT_WRITE, f"cannot create output directory {out_dir}: {exc}") from exc
    traceg_path = out_dir / f"kernel-{_int_field(kernel.id, 'kernel.id')}.traceg"
    list_path = out_dir / "kernelslist.g"
    result_path = out_dir / "adapter_result.json"
    list_bytes = (traceg_path.name + "\n").encode("utf-8")
    result = {
        "schema": "sm86_traceg_adapter_probe_v2",
        "status": "ok",
        "scoring": False,
        "termination": "dynamic_exit",
        "record_count": len(records),
        "input": {
            "dynamic_trace_sha256": _sha256_bytes(trace_bytes),
            "threadblock_sha256": _sha256_bytes(tb_bytes),
            "control_sidecar_sha256": _sha256_bytes(control_bytes),
            **({"exec_info_sha256": _sha256_bytes(exec_bytes)} if exec_bytes is not None else {}),
        },
        "output": {
            "traceg": str(traceg_path),
            "traceg_sha256": _sha256_bytes(traceg_bytes),
            "kernelslist": str(list_path),
            "kernelslist_sha256": _sha256_bytes(list_bytes),
        },
        "canonical_records": [record.as_json() for record in records],
        "notes": [
            "legacy grammar round-trip only; no NCU or timing accuracy claim",
            "RZ is canonicalized to legacy R255 in the static operand projection",
            "termination is an actual dynamic EXIT record; EOF and synthetic EXIT are refused",
            "memory, predicated, and non-ALU instructions are outside this contract",
        ],
    }
    _write_exclusive(traceg_path, traceg_bytes)
    _write_exclusive(list_path, list_bytes)
    _write_exclusive(result_path, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return result


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--threadblock", type=Path, required=True)
    sidecar = parser.add_mutually_exclusive_group(required=True)
    sidecar.add_argument("--control", "--control-sidecar", dest="control_path", type=Path)
    # Kept as an alias for callers migrating from v1.  It now means control
    # sidecar, so passing enhanced_execution_info.json fails with a typed shape
    # refusal instead of silently using its untrusted op_code field.
    sidecar.add_argument("--sidecar", dest="control_path_legacy", type=Path)
    parser.add_argument("--exec-info", type=Path, help="optional enhanced_execution_info.json used only for fuid/name/PC identity")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-eof", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    control_path = args.control_path or args.control_path_legacy
    try:
        result = adapt(args.trace, args.threadblock, control_path, args.out_dir, allow_eof=args.allow_eof, exec_info_path=args.exec_info)
    except AdapterRefusal as exc:
        print(json.dumps({"status": "refused", "code": exc.code.value, "detail": exc.detail, "record_index": exc.record_index}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
