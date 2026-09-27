#!/usr/bin/env python3
"""Fail-closed enhanced protobuf -> legacy ``.traceg`` adapter probe.

This is an experiment tool, not a simulator input path.  It deliberately
admits a tiny common subset: one SM86 launch, one CTA, one warp, at most 64
records, full warp masks, ordinary regular-GPR instructions, and (optionally)
one ordinary global LDG/STG address operand.  Every other representation loss
is a typed refusal rather than an implicit default.

The emitted trace is parsed again by the independent parser in this file.  The
parser checks the legacy grammar and canonical record equality; it does not
claim that it is an Accel-Sim timing result.  An external Accel-Sim executable
may be run separately on the emitted files for the bounded smoke described in
the accompanying report.
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
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MAX_RECORDS = 64
WARP_MASK = 0xFFFF_FFFF
UINT32_MAX = 0xFFFF_FFFF
MAX_U64 = 0xFFFF_FFFF_FFFF_FFFF
_REGISTER_RE = re.compile(r"^R(?:Z|[0-9]+)$")
_ALLOWED_ALU = frozenset({"IADD3", "FADD"})
_ALLOWED_MEMORY_HEADS = frozenset({"LDG.E", "STG.E"})


class RefusalCode(str, Enum):
    """Stable machine-readable reasons for a rejected projection."""

    PROTOBUF_RUNTIME_MISSING = "protobuf_runtime_missing"
    ROOT_CARDINALITY = "root_cardinality_not_one"
    ROOT_NOT_SM86 = "root_not_sm86"
    LAUNCH_SHAPE_UNSUPPORTED = "launch_shape_unsupported"
    LAUNCH_METADATA_MISSING = "launch_metadata_missing"
    THREADBLOCK_SHAPE_UNSUPPORTED = "threadblock_shape_unsupported"
    RECORD_LIMIT_EXCEEDED = "record_limit_exceeded"
    MASK_METADATA_MISSING = "mask_metadata_missing"
    MASK_NOT_FULL_WARP = "mask_not_full_warp"
    VPC_MISSING = "vpc_missing"
    VPC_PC_MISMATCH = "vpc_pc_mismatch"
    FUNCTION_JOIN_MISMATCH = "function_join_mismatch"
    SIDECAR_MISSING = "sidecar_missing"
    SIDECAR_DUPLICATE_JOIN = "sidecar_duplicate_join"
    SIDECAR_PC_MISSING = "sidecar_pc_missing"
    SIDECAR_OPCODE_MISSING = "sidecar_opcode_missing"
    UNSUPPORTED_OPCODE = "unsupported_opcode"
    PREDICATED_STATIC = "predicated_static_instruction"
    UNSUPPORTED_OPERAND = "unsupported_operand"
    REGISTER_ARITY = "register_arity_unsupported"
    REGISTER_TOKEN = "register_token_unsupported"
    MEMORY_SHAPE = "memory_shape_unsupported"
    MEMORY_METADATA = "memory_metadata_invalid"
    ADDRESS_ENCODING = "address_encoding_unsupported"
    ADDRESS_VALUE = "address_value_invalid"
    TERMINATION_INCOMPLETE = "termination_incomplete"
    TRACEG_PARSE = "traceg_parse_failed"
    ROUNDTRIP_MISMATCH = "roundtrip_mismatch"


@dataclass(frozen=True)
class AdapterRefusal(Exception):
    """Typed fail-closed adapter refusal."""

    code: RefusalCode
    detail: str
    record_index: int | None = None

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _dim3(value: Any, label: str) -> tuple[int, int, int]:
    if value is None:
        raise AdapterRefusal(RefusalCode.LAUNCH_METADATA_MISSING, f"{label} absent")
    return int(value.x), int(value.y), int(value.z)


def _load_root_and_kernel(trace_path: Path, bindings: Any) -> tuple[Any, Any, int]:
    root = bindings.trace_pb2.Trace()
    try:
        root.ParseFromString(trace_path.read_bytes())
    except Exception as exc:
        raise AdapterRefusal(
            RefusalCode.ROOT_CARDINALITY,
            f"protobuf root cannot be decoded: {exc}",
        ) from exc
    if len(root.gpu_device) != 1:
        raise AdapterRefusal(
            RefusalCode.ROOT_CARDINALITY,
            f"expected one device, got {len(root.gpu_device)}",
        )
    device_id, device = next(iter(root.gpu_device.items()))
    if int(root.binary_version) != 86:
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
    if len(stream.kernels) != 1:
        raise AdapterRefusal(
            RefusalCode.ROOT_CARDINALITY,
            f"expected one kernel, got {len(stream.kernels)}",
        )
    kernel = stream.kernels[0]
    if int(kernel.id) <= 0 or int(kernel.function_unique_id) <= 0:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "kernel id and function_unique_id must be positive",
        )
    grid = _dim3(kernel.grid_dim, "grid_dim")
    block = _dim3(kernel.block_dim, "block_dim")
    if grid != (1, 1, 1) or block != (32, 1, 1):
        raise AdapterRefusal(
            RefusalCode.LAUNCH_SHAPE_UNSUPPORTED,
            f"expected grid=(1,1,1), block=(32,1,1), got {grid}/{block}",
        )
    if int(kernel.number_of_registers) <= 0 or int(kernel.size_shared_memory) < 0:
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "nregs must be positive and shared-memory size nonnegative",
        )
    if not _has(kernel, "shared_memory_static_bytes") or not _has(
        kernel, "shared_memory_dynamic_bytes"
    ):
        raise AdapterRefusal(
            RefusalCode.LAUNCH_METADATA_MISSING,
            "explicit shared-memory static/dynamic components are required",
        )
    return root, kernel, int(stream_id)


def _load_sidecar(path: Path, function_id: int) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise AdapterRefusal(RefusalCode.SIDECAR_MISSING, f"not a file: {path}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise AdapterRefusal(RefusalCode.SIDECAR_MISSING, f"invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("kernels"), list):
        raise AdapterRefusal(RefusalCode.SIDECAR_MISSING, "sidecar must contain a kernels list")
    candidates = [
        item
        for item in payload["kernels"]
        if isinstance(item, Mapping)
        and int(item.get("unique_function_id", -1)) == function_id
    ]
    if len(candidates) != 1:
        raise AdapterRefusal(
            RefusalCode.FUNCTION_JOIN_MISMATCH,
            f"expected one sidecar kernel for function_unique_id={function_id}, got {len(candidates)}",
        )
    static = candidates[0].get("instructions")
    if not isinstance(static, list):
        raise AdapterRefusal(RefusalCode.SIDECAR_MISSING, "kernel instructions list absent")
    joined: dict[tuple[int, int], dict[str, Any]] = {}
    for item in static:
        if not isinstance(item, Mapping) or "pc_num_dec" not in item:
            raise AdapterRefusal(RefusalCode.SIDECAR_PC_MISSING, "sidecar instruction has no pc_num_dec")
        pc = int(item["pc_num_dec"])
        key = (function_id, pc)
        if key in joined:
            raise AdapterRefusal(RefusalCode.SIDECAR_DUPLICATE_JOIN, f"duplicate static key {key}")
        joined[key] = dict(item)
    return joined, dict(payload)


def _canonical_register(value: Any, record_index: int, role: str) -> str:
    token = str(value).strip().upper()
    if token == "RZ":
        return "R255"
    if not _REGISTER_RE.fullmatch(token):
        raise AdapterRefusal(
            RefusalCode.REGISTER_TOKEN,
            f"{role} token {token!r} is not an ordinary R register",
            record_index,
        )
    number = 255 if token == "RZ" else int(token[1:])
    if number < 0 or number > 255:
        raise AdapterRefusal(RefusalCode.REGISTER_TOKEN, f"register {token!r} exceeds R255", record_index)
    return f"R{number}"


def _static_operands(static: Mapping[str, Any], record_index: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    operands = static.get("operands")
    if not isinstance(operands, list):
        raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPERAND, "sidecar operands list absent", record_index)
    regs: list[str] = []
    for operand in operands:
        if not isinstance(operand, Mapping):
            raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPERAND, "malformed operand", record_index)
        operand_type = str(operand.get("operand_type", ""))
        token = str(operand.get("operand_string", ""))
        if operand_type != "REG" or not bool(operand.get("has_reg", False)):
            raise AdapterRefusal(
                RefusalCode.UNSUPPORTED_OPERAND,
                f"{operand_type or 'unknown'} operand {token!r} cannot be represented in common subset",
                record_index,
            )
        regs.append(_canonical_register(token, record_index, "register"))
    try:
        destination_count = int(static.get("num_destination_registers", -1))
    except (TypeError, ValueError):
        destination_count = -1
    if destination_count < 0 or destination_count > 1 or destination_count > len(regs):
        raise AdapterRefusal(
            RefusalCode.REGISTER_ARITY,
            f"destination count {destination_count} with {len(regs)} registers",
            record_index,
        )
    source_count = len(regs) - destination_count
    if source_count > 4:
        raise AdapterRefusal(RefusalCode.REGISTER_ARITY, f"source count {source_count} > 4", record_index)
    return tuple(regs[:destination_count]), tuple(regs[destination_count:])


def _checked_address_vector(address: Any, mask: int, record_index: int) -> tuple[int, ...]:
    fmt = int(address.compression_format)
    if fmt == 0:
        if len(address.addrs) != 32:
            raise AdapterRefusal(
                RefusalCode.ADDRESS_ENCODING,
                f"list_all requires 32 addresses, got {len(address.addrs)}",
                record_index,
            )
        values = tuple(int(value) for value in address.addrs)
    elif fmt == 1:
        if len(address.addrs) != 0:
            raise AdapterRefusal(RefusalCode.ADDRESS_ENCODING, "base_stride must not carry deltas", record_index)
        base = int(address.base_address)
        stride = int(address.stride)
        values = tuple(base + stride * lane for lane in range(32))
    elif fmt == 2:
        expected = (mask.bit_count() - 1) if mask else 0
        if len(address.addrs) != expected:
            raise AdapterRefusal(
                RefusalCode.ADDRESS_ENCODING,
                f"base_delta expected {expected} deltas, got {len(address.addrs)}",
                record_index,
            )
        active_lanes = [lane for lane in range(32) if mask & (1 << lane)]
        values_list = [-1] * 32
        if active_lanes:
            running = int(address.base_address)
            values_list[active_lanes[0]] = running
            for lane, delta in zip(active_lanes[1:], address.addrs):
                running += int(delta)
                values_list[lane] = running
        values = tuple(values_list)
    else:
        raise AdapterRefusal(RefusalCode.ADDRESS_ENCODING, f"unsupported compression_format={fmt}", record_index)
    for lane, value in enumerate(values):
        active = bool(mask & (1 << lane))
        if active and not (0 <= value <= MAX_U64):
            raise AdapterRefusal(RefusalCode.ADDRESS_VALUE, f"lane {lane} address {value} invalid", record_index)
        if not active and value not in (-1, 0):
            raise AdapterRefusal(RefusalCode.ADDRESS_VALUE, f"inactive lane {lane} has {value}", record_index)
    return tuple(value for value in values if value >= 0)


def _record_from_messages(
    dynamic: Any,
    static: Mapping[str, Any],
    index: int,
    function_id: int,
) -> CanonicalRecord:
    if int(dynamic.function_unique_id) != function_id:
        raise AdapterRefusal(
            RefusalCode.FUNCTION_JOIN_MISMATCH,
            f"record function_unique_id={dynamic.function_unique_id} != kernel={function_id}",
            index,
        )
    if not _has(dynamic, "active_mask") or not _has(dynamic, "predicate_mask"):
        raise AdapterRefusal(RefusalCode.MASK_METADATA_MISSING, "active/predicate mask presence required", index)
    mask = int(dynamic.active_mask) & int(dynamic.predicate_mask)
    if mask != WARP_MASK:
        raise AdapterRefusal(RefusalCode.MASK_NOT_FULL_WARP, f"effective mask=0x{mask:08x}", index)
    if not _has(dynamic, "vpc"):
        raise AdapterRefusal(RefusalCode.VPC_MISSING, "vpc presence required", index)
    pc = int(dynamic.pc)
    vpc = int(dynamic.vpc)
    if vpc > UINT32_MAX or (vpc & UINT32_MAX) != pc:
        raise AdapterRefusal(RefusalCode.VPC_PC_MISMATCH, f"pc=0x{pc:x}, vpc=0x{vpc:x}", index)
    if not _has(dynamic, "warpid_tb") or int(dynamic.warpid_tb) != 0:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED, "record is not warp 0", index)
    op = str(static.get("op_code", "")).strip()
    if not op:
        raise AdapterRefusal(RefusalCode.SIDECAR_OPCODE_MISSING, "sidecar op_code absent", index)
    if bool(static.get("is_predicated", False)) or bool(static.get("is_uniform_predicate", False)):
        raise AdapterRefusal(RefusalCode.PREDICATED_STATIC, f"static opcode {op!r} is predicated", index)
    if op not in _ALLOWED_ALU and op not in _ALLOWED_MEMORY_HEADS:
        raise AdapterRefusal(RefusalCode.UNSUPPORTED_OPCODE, f"opcode {op!r} not in common subset", index)
    dest_regs, src_regs = _static_operands(static, index)
    addresses: tuple[int, ...] = ()
    mem_width = 0
    if op in _ALLOWED_MEMORY_HEADS:
        if len(dynamic.addresses) != 1:
            raise AdapterRefusal(RefusalCode.MEMORY_SHAPE, f"{op} requires one address operand", index)
        if not _has(dynamic, "memory_op_class") or int(dynamic.memory_op_class) != 1:
            raise AdapterRefusal(RefusalCode.MEMORY_METADATA, "only GLOBAL memory class is admitted", index)
        if not _has(dynamic, "data_width") or int(dynamic.data_width) <= 0:
            raise AdapterRefusal(RefusalCode.MEMORY_METADATA, "positive data_width required", index)
        address = dynamic.addresses[0]
        if _has(address, "lane_mask") and int(address.lane_mask) != mask:
            raise AdapterRefusal(RefusalCode.MEMORY_METADATA, "operand lane_mask differs from effective mask", index)
        if _has(address, "address_space") and int(address.address_space) not in (1, 4):
            raise AdapterRefusal(RefusalCode.MEMORY_METADATA, "address_space is not GLOBAL/GENERIC", index)
        mem_width = int(dynamic.data_width)
        if int(address.data_width) > 0 and int(address.data_width) != mem_width:
            raise AdapterRefusal(RefusalCode.MEMORY_METADATA, "instruction/address widths disagree", index)
        addresses = _checked_address_vector(address, mask, index)
        if len(addresses) != 32:
            raise AdapterRefusal(RefusalCode.ADDRESS_ENCODING, "full mask must reconstruct 32 addresses", index)
    elif len(dynamic.addresses) != 0 or _has(dynamic, "memory_op_class"):
        raise AdapterRefusal(RefusalCode.MEMORY_SHAPE, "non-memory opcode carries memory metadata", index)
    return CanonicalRecord(
        block=(0, 0, 0),
        warp=0,
        pc=pc,
        mask=mask,
        opcode=op,
        dest_regs=dest_regs,
        src_regs=src_regs,
        mem_width=mem_width,
        addresses=addresses,
    )


def _format_record(record: CanonicalRecord) -> str:
    parts: list[str] = [f"{record.pc:04x}", f"{record.mask:08x}", str(len(record.dest_regs))]
    parts.extend(record.dest_regs)
    parts += [record.opcode, str(len(record.src_regs))]
    parts.extend(record.src_regs)
    parts.append(str(record.mem_width))
    if record.mem_width:
        # The emitted common subset uses list_all so the parser sees the exact
        # per-lane vector.  This is intentionally verbose and bounded.
        parts.append("0")
        parts.extend(f"{address:x}" for address in record.addresses)
    parts.append(str(record.immediate))
    return " ".join(parts)


def _header(root: Any, kernel: Any, stream_id: int) -> list[str]:
    shared = int(kernel.shared_memory_base_address)
    local = int(kernel.local_memory_base_address)
    return [
        f"-kernel name = {kernel.name}",
        f"-kernel id = {int(kernel.id)}",
        "-grid dim = (1,1,1)",
        "-block dim = (32,1,1)",
        f"-shmem = {int(kernel.size_shared_memory)}",
        f"-nregs = {int(kernel.number_of_registers)}",
        f"-binary version = {int(root.binary_version)}",
        f"-cuda stream id = {stream_id}",
        f"-shmem base_addr = 0x{shared & MAX_U64:016x}",
        f"-local mem base_addr = 0x{local & MAX_U64:016x}",
        f"-nvbit version = {root.nvbit_version}",
        f"-accelsim tracer version = {int(root.accelsim_version)}",
        "-enable lineinfo = 0",
        "",
        "#traces format = [line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [addresscompress?] [mem_addresses] immediate",
    ]


def _emit_traceg(
    path: Path,
    root: Any,
    kernel: Any,
    stream_id: int,
    records: Sequence[CanonicalRecord],
    termination: Mapping[str, Any],
) -> tuple[str, str]:
    kind = str(termination.get("kind", "eof"))
    if kind not in {"explicit_exit", "eof"}:
        raise AdapterRefusal(RefusalCode.TERMINATION_INCOMPLETE, f"unsupported termination kind {kind!r}")
    if kind == "eof":
        raise AdapterRefusal(
            RefusalCode.TERMINATION_INCOMPLETE,
            "source has no explicit termination record; pass allow_eof only for parser evidence",
        )
    terminal_pc = int(termination.get("pc", (records[-1].pc + 16) if records else 0))
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
    lines.append(f"{terminal_pc:04x} 00000000 0 EXIT 0 0 0")
    lines.append("#END_TB")
    payload = "\n".join(lines) + "\n"
    path.write_text(payload)
    list_path = path.with_name("kernelslist.g")
    list_path.write_text(path.name + "\n")
    return str(path), str(list_path)


def parse_traceg(path: Path) -> tuple[list[CanonicalRecord], bool]:
    """Parse the subset emitted above using an independent token reader."""

    lines = path.read_text().splitlines()
    records: list[CanonicalRecord] = []
    expected_count: int | None = None
    terminal = False
    in_tb = False
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if stripped == "#BEGIN_TB":
                in_tb = True
            elif stripped == "#END_TB":
                in_tb = False
            continue
        if stripped.startswith("insts ="):
            expected_count = int(stripped.split("=", 1)[1].strip())
            continue
        if not in_tb or stripped.startswith(("thread block", "cluster ", "warp =")):
            continue
        tokens = stripped.split()
        try:
            cursor = 0
            pc = int(tokens[cursor], 16); cursor += 1
            mask = int(tokens[cursor], 16); cursor += 1
            dest_count = int(tokens[cursor]); cursor += 1
            dest = tuple(tokens[cursor + i] for i in range(dest_count)); cursor += dest_count
            opcode = tokens[cursor]; cursor += 1
            src_count = int(tokens[cursor]); cursor += 1
            src = tuple(tokens[cursor + i] for i in range(src_count)); cursor += src_count
            mem_width = int(tokens[cursor]); cursor += 1
            addresses: tuple[int, ...] = ()
            if mem_width:
                fmt = int(tokens[cursor]); cursor += 1
                if fmt != 0:
                    raise ValueError("probe parser expects list_all output")
                addresses = tuple(int(tokens[cursor + i], 16) for i in range(32)); cursor += 32
            immediate = int(tokens[cursor], 0); cursor += 1
            if cursor != len(tokens):
                raise ValueError("trailing tokens")
        except (IndexError, ValueError) as exc:
            raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"line {line_number}: {exc}") from exc
        if opcode == "EXIT":
            if mask != 0 or dest_count != 0 or src_count != 0 or mem_width != 0:
                raise AdapterRefusal(RefusalCode.TRACEG_PARSE, f"malformed EXIT line {line_number}")
            terminal = True
            continue
        records.append(
            CanonicalRecord(
                block=(0, 0, 0),
                warp=0,
                pc=pc,
                mask=mask,
                opcode=opcode,
                dest_regs=dest,
                src_regs=src,
                mem_width=mem_width,
                addresses=addresses,
                immediate=immediate,
            )
        )
    if expected_count is not None and expected_count != len(records) + int(terminal):
        raise AdapterRefusal(
            RefusalCode.TRACEG_PARSE,
            f"insts declares {expected_count}, parsed {len(records) + int(terminal)}",
        )
    return records, terminal


def adapt(
    trace_path: Path,
    threadblock_path: Path,
    sidecar_path: Path,
    out_dir: Path,
    *,
    allow_eof: bool = False,
) -> dict[str, Any]:
    """Adapt one bounded launch and return a content-bound result manifest."""

    bindings = _load_bindings()
    root, kernel, stream_id = _load_root_and_kernel(trace_path, bindings)
    tb = bindings.threadblock_pb2.threadblock()
    try:
        tb.ParseFromString(threadblock_path.read_bytes())
    except Exception as exc:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED, f"threadblock cannot decode: {exc}") from exc
    if _dim3(tb.block_id, "threadblock.block_id") != (0, 0, 0) or len(tb.warps) != 1 or 0 not in tb.warps:
        raise AdapterRefusal(
            RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED,
            f"expected block (0,0,0) and exactly warp 0, got block={tb.block_id} warps={list(tb.warps)}",
        )
    warp = tb.warps[0]
    if int(warp.id) != 0:
        raise AdapterRefusal(RefusalCode.THREADBLOCK_SHAPE_UNSUPPORTED, f"warp id={warp.id}; expected 0")
    if len(warp.instructions) > MAX_RECORDS:
        raise AdapterRefusal(RefusalCode.RECORD_LIMIT_EXCEEDED, f"{len(warp.instructions)} > {MAX_RECORDS}")
    joins, sidecar = _load_sidecar(sidecar_path, int(kernel.function_unique_id))
    records: list[CanonicalRecord] = []
    for index, dynamic in enumerate(warp.instructions):
        key = (int(dynamic.function_unique_id), int(dynamic.vpc) if _has(dynamic, "vpc") else int(dynamic.pc))
        static = joins.get(key)
        if static is None:
            raise AdapterRefusal(RefusalCode.FUNCTION_JOIN_MISMATCH, f"no unique sidecar join for {key}", index)
        records.append(_record_from_messages(dynamic, static, index, int(kernel.function_unique_id)))
    termination = sidecar.get("termination")
    if not isinstance(termination, Mapping):
        termination = {"kind": "eof"}
    if str(termination.get("kind", "eof")) == "eof" and not allow_eof:
        raise AdapterRefusal(
            RefusalCode.TERMINATION_INCOMPLETE,
            "no explicit terminal record; this is not admitted for baseline execution",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    traceg_path = out_dir / f"kernel-{int(kernel.id)}.traceg"
    list_path = out_dir / "kernelslist.g"
    if str(termination.get("kind", "eof")) == "explicit_exit":
        _emit_traceg(traceg_path, root, kernel, stream_id, records, termination)
    else:
        # A parser-only EOF artifact is still useful, but deliberately has no
        # synthetic EXIT and is marked non-scoring in the result manifest.
        lines = _header(root, kernel, stream_id) + [
            "#BEGIN_TB",
            "thread block = 0,0,0",
            "cluster id = 0,0,0",
            "cluster cta = 0,0,0",
            "cluster rank = 0",
            "warp = 0",
            f"insts = {len(records)}",
        ]
        lines.extend(_format_record(record) for record in records)
        lines.append("#END_TB")
        traceg_path.write_text("\n".join(lines) + "\n")
        list_path.write_text(traceg_path.name + "\n")
    parsed, terminal = parse_traceg(traceg_path)
    if parsed != records:
        raise AdapterRefusal(
            RefusalCode.ROUNDTRIP_MISMATCH,
            f"canonical records differ after traceg parse: emitted={len(records)} parsed={len(parsed)}",
        )
    result = {
        "schema": "sm86_traceg_adapter_probe_v1",
        "status": "ok",
        "scoring": False,
        "termination": "explicit_exit" if terminal else "eof_non_scoring",
        "record_count": len(records),
        "input": {
            "dynamic_trace_sha256": _sha256(trace_path),
            "threadblock_sha256": _sha256(threadblock_path),
            "sidecar_sha256": _sha256(sidecar_path),
        },
        "output": {
            "traceg": str(traceg_path),
            "traceg_sha256": _sha256(traceg_path),
            "kernelslist": str(list_path),
            "kernelslist_sha256": _sha256(list_path),
        },
        "canonical_records": [record.as_json() for record in records],
        "notes": [
            "legacy grammar round-trip only; no NCU or timing accuracy claim",
            "RZ is canonicalized to legacy R255 in the static operand projection",
        ],
    }
    (out_dir / "adapter_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--threadblock", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-eof",
        action="store_true",
        help="emit a parser-only EOF artifact; never scores as a baseline",
    )
    args = parser.parse_args(argv)
    try:
        result = adapt(
            args.trace,
            args.threadblock,
            args.sidecar,
            args.out_dir,
            allow_eof=args.allow_eof,
        )
    except AdapterRefusal as exc:
        print(json.dumps({"status": "refused", "code": exc.code.value, "detail": exc.detail, "record_index": exc.record_index}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
