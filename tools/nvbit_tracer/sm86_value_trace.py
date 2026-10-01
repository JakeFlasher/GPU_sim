#!/usr/bin/env python3
"""Finite execution-driven SM86 SASS trace regeneration.

This is deliberately a scalar, fail-closed interpreter for a tiny static SASS
subset.  It regenerates a *dynamic value trace* from a content-bound SM86
disassembly; it does not emit a cubin, replace NVIDIA's compiler, or claim
silicon timing.  Unsupported opcodes, branches, special registers, memory
forms, and malformed operands refuse rather than default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

MAX_SASS_BYTES = 32 * 1024 * 1024
MAX_STEPS = 100_000
LINE = re.compile(r"^\s*/\*(?P<pc>[0-9a-fA-F]+)\*/\s+(?P<text>[^;]+);")
REG = re.compile(r"^R(?P<num>[0-9]+)$", re.IGNORECASE)
CBANK = re.compile(r"^c\[(?P<bank>0x[0-9a-fA-F]+|[0-9]+)\]\[(?P<off>0x[0-9a-fA-F]+|[0-9]+)\]$")


class TraceRefusal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _read(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TraceRefusal("input_missing", f"cannot read {path}: {exc}") from exc
    if len(data) > MAX_SASS_BYTES:
        raise TraceRefusal("input_limit_exceeded", f"{path} exceeds {MAX_SASS_BYTES} bytes")
    return data


def _load_manifest(path: Path, sass: bytes) -> dict[str, Any]:
    try:
        raw = json.loads(_read(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceRefusal("manifest_invalid", str(exc)) from exc
    if not isinstance(raw, dict) or raw.get("schema") != "sm86_static_sass_artifact_v1":
        raise TraceRefusal("manifest_invalid", "expected sm86_static_sass_artifact_v1")
    if raw.get("target") != "sm_86" or raw.get("static_sass") is not True:
        raise TraceRefusal("manifest_target_mismatch", "manifest is not a static SM86 artifact")
    artifacts = raw.get("artifacts")
    sass_record = artifacts.get("sass") if isinstance(artifacts, dict) else None
    expected = sass_record.get("sha256") if isinstance(sass_record, dict) else None
    if not isinstance(expected, str):
        raise TraceRefusal("manifest_invalid", "manifest has no bound SASS digest")
    actual = hashlib.sha256(sass).hexdigest()
    if expected != actual:
        raise TraceRefusal("sass_digest_mismatch", f"manifest={expected!r}, input={actual!r}")
    return raw


def _number(token: str) -> int:
    try:
        return int(token, 0) if not re.fullmatch(r"0[0-9]+", token) else int(token, 10)
    except ValueError as exc:
        raise TraceRefusal("operand_invalid", f"not an integer literal: {token!r}") from exc


def _reg(token: str) -> int:
    match = REG.fullmatch(token.strip().split(".", 1)[0])
    if not match:
        raise TraceRefusal("register_unsupported", f"not an ordinary GPR: {token!r}")
    number = int(match.group("num"))
    if number > 255:
        raise TraceRefusal("register_unsupported", f"R{number} exceeds R255")
    return number


def parse_sass(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        match = LINE.match(raw)
        if not match:
            continue
        pc = int(match.group("pc"), 16)
        instruction = match.group("text").strip()
        if not instruction:
            raise TraceRefusal("sass_record_invalid", f"line {line_number} is empty")
        guard = None
        if instruction.startswith("@"):
            guard, _, instruction = instruction.partition(" ")
            guard = guard.upper()
        opcode, _, operands = instruction.partition(" ")
        records.append({"pc": pc, "opcode": opcode, "guard": guard, "operands": operands.strip(), "text": match.group("text").strip()})
    if not records:
        raise TraceRefusal("sass_empty", "no disassembly records found")
    if len({record["pc"] for record in records}) != len(records):
        raise TraceRefusal("sass_duplicate_pc", "duplicate static PC")
    return records


def _operand_value(token: str, regs: dict[int, int], constants: dict[str, int]) -> int:
    token = token.strip()
    if token.startswith("-"):
        return (-_operand_value(token[1:].strip(), regs, constants)) & 0xFFFF_FFFF
    if token.upper() == "RZ":
        return 0
    match = REG.fullmatch(token.split(".", 1)[0])
    if match:
        register = int(match.group("num"))
        if register not in regs:
            raise TraceRefusal("register_value_missing", f"no initialized value for R{register}")
        return regs[register]
    match = CBANK.fullmatch(token)
    if match:
        key = f"{int(match.group('bank'), 0)}:{int(match.group('off'), 0)}"
        if key not in constants:
            raise TraceRefusal("constant_missing", f"no supplied value for {key}")
        return constants[key] & 0xFFFF_FFFF
    return _number(token) & 0xFFFF_FFFF


def _lop3_source_value(token: str, regs: dict[int, int], constants: dict[str, int]) -> int:
    """Read the admitted SM86 data forms without discarding unknown modifiers."""
    if re.fullmatch(r"R(?:[0-9]+|Z)(?:\.reuse)?", token, re.IGNORECASE):
        register = token.split(".", 1)[0]
        if register.upper() != "RZ" and _reg(register) > 255:
            raise TraceRefusal("register_unsupported", f"unsupported LOP3 source {token!r}")
        return _operand_value(register, regs, constants)
    if CBANK.fullmatch(token):
        return _operand_value(token, regs, constants)
    if re.fullmatch(r"[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)", token):
        value = _number(token)
        if -(1 << 31) <= value <= 0xFFFF_FFFF:
            return value & 0xFFFF_FFFF
    raise TraceRefusal("lop3_operand_unsupported", f"unsupported LOP3 source {token!r}")


def _shf_source_value(token: str, regs: dict[int, int], constants: dict[str, int]) -> int:
    """Read the bounded SHF GPR/immediate/constant source forms."""
    try:
        return _lop3_source_value(token, regs, constants)
    except TraceRefusal as exc:
        raise TraceRefusal("shf_operand_unsupported", f"unsupported SHF source {token!r}") from exc


def execute(
    sass: bytes,
    *,
    lane_id: int,
    constants: dict[str, int] | None = None,
    initial_memory: dict[int, int] | None = None,
    initial_registers: dict[int, int] | None = None,
    max_steps: int = 100_000,
    bound: bool = False,
) -> dict[str, Any]:
    if not 0 <= lane_id < 32:
        raise TraceRefusal("lane_out_of_range", f"lane {lane_id} is outside 0..31")
    if max_steps <= 0 or max_steps > MAX_STEPS:
        raise TraceRefusal("step_limit_invalid", f"max_steps must be in 1..={MAX_STEPS}")
    constants = constants or {}
    records = parse_sass(sass.decode("utf-8"))
    regs: dict[int, int] = {
        int(register): int(value) & 0xFFFF_FFFF
        for register, value in (initial_registers or {}).items()
    }
    predicates: dict[int, bool] = {}
    memory: dict[int, int] = {
        int(address): int(value) & 0xFFFF_FFFF
        for address, value in (initial_memory or {}).items()
    }
    trace: list[dict[str, Any]] = []
    pc_index = {record["pc"]: index for index, record in enumerate(records)}
    index = 0
    for _ in range(max_steps):
        if index >= len(records):
            raise TraceRefusal("implicit_eof", "execution reached EOF without EXIT")
        record = records[index]
        opcode = record["opcode"].upper()
        operands = [part.strip() for part in record["operands"].split(",")] if record["operands"] else []
        before = {f"R{reg}": value for reg, value in sorted(regs.items())}
        after = dict(before)
        stores: list[dict[str, int]] = []
        loads: list[dict[str, int]] = []
        predicates_before = {f"P{pred}": value for pred, value in sorted(predicates.items())}
        guard = record["guard"]
        if guard is not None:
            negated = guard.startswith("@!")
            pred_text = guard[2:] if negated else guard[1:]
            if not pred_text.startswith("P") or not pred_text[1:].isdigit():
                raise TraceRefusal("predicate_unsupported", record["text"])
            predicate_index = int(pred_text[1:])
            if predicate_index > 7:
                raise TraceRefusal("predicate_unsupported", record["text"])
            if predicate_index not in predicates:
                raise TraceRefusal("predicate_value_missing", f"no initialized value for P{predicate_index}")
            pred_value = predicates[predicate_index]
            if pred_value == negated:
                trace.append({"sequence": len(trace), "pc": record["pc"], "opcode": record["text"], "guard": guard, "registers_before": before, "registers_after": after, "predicates_before": predicates_before, "predicates_after": dict(predicates_before), "loads": loads, "stores": stores})
                index += 1
                continue
        if opcode == "EXIT":
            if operands:
                raise TraceRefusal("exit_operands", "EXIT must not have operands")
            trace.append({"sequence": len(trace), "pc": record["pc"], "opcode": record["text"], "guard": guard, "registers_before": before, "registers_after": after, "predicates_before": predicates_before, "predicates_after": {f"P{pred}": value for pred, value in sorted(predicates.items())}, "loads": loads, "stores": stores})
            result = {
                "schema": "sm86_value_trace_v1",
                "status": "ok",
                "development_alpha": not bound,
                "input": {"size_bytes": len(sass)},
                "lane_id": lane_id,
                "records": trace,
                "memory": {str(address): value for address, value in sorted(memory.items())},
                "claims": {"dynamic_value_trace": True, "static_sass_regeneration": False, "timing": False},
            }
            if bound:
                result["static_sass_sha256"] = hashlib.sha256(sass).hexdigest()
            return result
        if opcode == "NOP":
            pass
        elif opcode == "S2R":
            if len(operands) != 2 or operands[1].upper() != "SR_TID.X":
                raise TraceRefusal("special_register_unsupported", record["text"])
            regs[_reg(operands[0])] = lane_id
        elif opcode == "MOV":
            if len(operands) != 2:
                raise TraceRefusal("operand_arity", record["text"])
            regs[_reg(operands[0])] = _operand_value(operands[1], regs, constants)
        elif opcode.startswith("ISETP."):
            pieces = opcode.split(".")
            if (
                len(operands) != 5
                or len(pieces) != 4
                or pieces[0] != "ISETP"
                or pieces[2] != "U32"
                or pieces[3] != "AND"
                or operands[1].upper() != "PT"
                or operands[4].upper() != "PT"
                or not re.fullmatch(r"P[0-9]+", operands[0].upper())
            ):
                raise TraceRefusal("predicate_form_unsupported", record["text"])
            relation = pieces[1].upper()
            left = _operand_value(operands[2], regs, constants)
            right = _operand_value(operands[3], regs, constants)
            relation_values = {
                "LT": left < right,
                "LE": left <= right,
                "EQ": left == right,
                "NE": left != right,
                "GE": left >= right,
                "GT": left > right,
            }
            if relation not in relation_values:
                raise TraceRefusal("predicate_relation_unsupported", record["text"])
            destination = int(operands[0].upper()[1:])
            if destination > 7:
                raise TraceRefusal("predicate_unsupported", record["text"])
            predicates[destination] = relation_values[relation]
        elif opcode in {"IMAD.MOV.U32", "IMAD.MOV"}:
            if len(operands) != 4 or operands[1].upper() != "RZ" or operands[2].upper() != "RZ":
                raise TraceRefusal("opcode_form_unsupported", record["text"])
            regs[_reg(operands[0])] = _operand_value(operands[3], regs, constants)
        elif opcode == "IMAD.SHL.U32":
            # Bounded Ampere form observed in SM86 address/index bodies:
            # D = (A * immediate_power_of_two + C) mod 2^32.  This is an
            # integer multiply/add datapath idiom, not a general IMAD or a
            # compiler-generation claim.
            if len(operands) != 4:
                raise TraceRefusal("operand_arity", record["text"])
            if not re.fullmatch(r"[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)", operands[2]):
                raise TraceRefusal("imad_shl_multiplier_unsupported", record["text"])
            multiplier = _number(operands[2])
            if (
                not 0 <= multiplier <= 0xFFFF_FFFF
                or multiplier == 0
                or multiplier & (multiplier - 1) != 0
            ):
                raise TraceRefusal("imad_shl_multiplier_unsupported", record["text"])
            left = _operand_value(operands[1], regs, constants)
            addend = _operand_value(operands[3], regs, constants)
            regs[_reg(operands[0])] = (left * multiplier + addend) & 0xFFFF_FFFF
        elif opcode == "IMAD.IADD":
            # Bounded Ampere idiom observed in address arithmetic:
            # D = A * 1 + C. Other IMAD.IADD encodings remain refused.
            if len(operands) != 4:
                raise TraceRefusal("operand_arity", record["text"])
            if not re.fullmatch(r"[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)", operands[2]):
                raise TraceRefusal("imad_iadd_multiplier_unsupported", record["text"])
            if _number(operands[2]) != 1:
                raise TraceRefusal("imad_iadd_multiplier_unsupported", record["text"])
            left = _operand_value(operands[1], regs, constants)
            addend = _operand_value(operands[3], regs, constants)
            regs[_reg(operands[0])] = (left + addend) & 0xFFFF_FFFF
        elif opcode.startswith("IMAD.WIDE.U32"):
            if len(operands) != 4:
                raise TraceRefusal("operand_arity", record["text"])
            product = _operand_value(operands[1], regs, constants) * _operand_value(operands[2], regs, constants)
            value = (product + _operand_value(operands[3], regs, constants)) & 0xFFFF_FFFF_FFFF_FFFF
            destination = _reg(operands[0])
            regs[destination] = value & 0xFFFF_FFFF
            if destination < 255:
                regs[destination + 1] = (value >> 32) & 0xFFFF_FFFF
        elif opcode == "ULDC.64":
            if len(operands) != 2:
                raise TraceRefusal("operand_arity", record["text"])
            # Descriptor setup is outside this scalar memory contract. The
            # instruction is admitted only when its source constant exists;
            # the descriptor itself is not dereferenced by LDG.E here.
            _operand_value(operands[1], regs, constants)
        elif opcode == "SEL":
            if len(operands) != 4 or not operands[3].startswith("!"):
                raise TraceRefusal("sel_form_unsupported", record["text"])
            predicate = operands[3][1:].strip().upper()
            if not predicate.startswith("P") or not predicate[1:].isdigit():
                raise TraceRefusal("predicate_unsupported", record["text"])
            predicate_index = int(predicate[1:])
            if predicate_index > 7:
                raise TraceRefusal("predicate_unsupported", record["text"])
            chosen = operands[2] if predicates.get(predicate_index, False) else operands[1]
            regs[_reg(operands[0])] = _operand_value(chosen, regs, constants)
        elif opcode == "IADD3":
            if len(operands) != 4:
                raise TraceRefusal("operand_arity", record["text"])
            regs[_reg(operands[0])] = sum(_operand_value(token, regs, constants) for token in operands[1:]) & 0xFFFF_FFFF
        elif opcode.startswith("SHF."):
            # The admitted SASS spelling is D, A, B, C, where B/A form the
            # 64-bit concatenation and C supplies the wrapped shift amount.
            # PTX defines SHF.L as the high 32 bits and SHF.R as the low 32
            # bits of [B,A] shifted by C. See the official SHF semantics:
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
            pieces = opcode.split(".")
            if (
                len(pieces) not in (3, 4)
                or pieces[0] != "SHF"
                or pieces[1] not in {"L", "R"}
                or pieces[2] != "U32"
                or (len(pieces) == 4 and pieces[3] != "HI")
                or len(operands) != 4
            ):
                raise TraceRefusal("shf_form_unsupported", record["text"])
            if not REG.fullmatch(operands[0]):
                raise TraceRefusal("register_unsupported", record["text"])
            left, high, shift = (
                _shf_source_value(token, regs, constants) for token in operands[1:]
            )
            amount = shift & 0x1F
            if pieces[1] == "L":
                result = (high << amount) & 0xFFFF_FFFF
                if amount:
                    result |= left >> (32 - amount)
            else:
                result = 0
                if amount:
                    result = (high << (32 - amount)) & 0xFFFF_FFFF
                result |= left >> amount
            regs[_reg(operands[0])] = result & 0xFFFF_FFFF
        elif opcode == "LOP3.LUT":
            # SM86's data-only disassembly has Rdst, A, B, C, LUT, !PT;
            # compiler output may additionally carry Pdst before Rdst.
            if len(operands) not in (6, 7):
                raise TraceRefusal("operand_arity", record["text"])
            if len(operands) == 7:
                predicate_destination = operands[0].upper()
                if not re.fullmatch(r"P[0-7]", predicate_destination):
                    raise TraceRefusal("predicate_destination_unsupported", record["text"])
                destination_token = operands[1]
                source_tokens = operands[2:5]
                lut_token = operands[5]
                predicate_input = operands[6].upper()
            else:
                predicate_destination = None
                destination_token = operands[0]
                source_tokens = operands[1:4]
                lut_token = operands[4]
                predicate_input = operands[5].upper()
            if predicate_input != "!PT":
                raise TraceRefusal("lop3_predicate_unsupported", record["text"])
            if not REG.fullmatch(destination_token):
                raise TraceRefusal("register_unsupported", record["text"])
            lut = _number(lut_token)
            if not 0 <= lut <= 0xFF:
                raise TraceRefusal("lut_unsupported", record["text"])
            left, middle, right = (
                _lop3_source_value(token, regs, constants)
                for token in source_tokens
            )
            result = 0
            for bit in range(32):
                # PTX's basis is A=0xf0, B=0xcc, C=0xaa: A is the
                # high input bit, not the low bit. Symmetric LUTs (XOR)
                # cannot distinguish this order, so tests include all LUTs.
                # https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-lop3
                lut_index = (
                    (((left >> bit) & 1) << 2)
                    | (((middle >> bit) & 1) << 1)
                    | ((right >> bit) & 1)
                )
                if (lut >> lut_index) & 1:
                    result |= 1 << bit
            regs[_reg(destination_token)] = result
            if predicate_destination is not None:
                predicates[int(predicate_destination[1:])] = result != 0
        elif opcode == "STG.E":
            if len(operands) != 2 or not operands[0].startswith("[") or not operands[0].endswith("]"):
                raise TraceRefusal("memory_form_unsupported", record["text"])
            address_token = operands[0][1:-1].split(".", 1)[0]
            address = _operand_value(address_token, regs, constants)
            value = _operand_value(operands[1], regs, constants)
            memory[address] = value
            stores.append({"address": address, "value_u32": value})
        elif opcode == "LDG.E":
            if len(operands) != 2 or not operands[1].startswith("[") or not operands[1].endswith("]"):
                raise TraceRefusal("memory_form_unsupported", record["text"])
            address_token = operands[1][1:-1].split(".", 1)[0]
            address = _operand_value(address_token, regs, constants)
            if address not in memory:
                raise TraceRefusal("memory_value_missing", f"no initialized value for address {address}")
            regs[_reg(operands[0])] = memory[address]
            loads.append({"address": address, "value_u32": memory[address]})
        elif opcode == "BRA":
            if len(operands) != 1:
                raise TraceRefusal("branch_form_unsupported", record["text"])
            target = operands[0].strip()
            if target.startswith("`") or target.startswith("("):
                raise TraceRefusal("branch_label_unresolved", record["text"])
            target_pc = _number(target)
            if target_pc not in pc_index:
                raise TraceRefusal("branch_target_missing", f"target PC {target_pc:#x} is absent")
            next_index = pc_index[target_pc]
            after = {f"R{reg}": value for reg, value in sorted(regs.items())}
            trace.append({"sequence": len(trace), "pc": record["pc"], "opcode": record["text"], "guard": guard, "registers_before": before, "registers_after": after, "predicates_before": predicates_before, "predicates_after": {f"P{pred}": value for pred, value in sorted(predicates.items())}, "loads": loads, "stores": stores})
            index = next_index
            continue
        else:
            raise TraceRefusal("opcode_unsupported", f"unsupported finite subset opcode {opcode!r}")
        after = {f"R{reg}": value for reg, value in sorted(regs.items())}
        trace.append({"sequence": len(trace), "pc": record["pc"], "opcode": record["text"], "guard": guard, "registers_before": before, "registers_after": after, "predicates_before": predicates_before, "predicates_after": {f"P{pred}": value for pred, value in sorted(predicates.items())}, "loads": loads, "stores": stores})
        index += 1
    raise TraceRefusal("step_limit_exceeded", f"execution exceeded {max_steps} steps")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", type=Path)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--constants", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--registers", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bound", action="store_true", help="opt into manifest/hash binding")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args(argv)
    try:
        sass = _read(args.sass)
        manifest = _load_manifest(args.manifest, sass) if args.manifest and args.bound else None
        constants = {}
        if args.constants:
            raw = json.loads(_read(args.constants).decode("utf-8"))
            if not isinstance(raw, dict) or not all(isinstance(key, str) and isinstance(value, int) for key, value in raw.items()):
                raise TraceRefusal("constants_invalid", "constants must be a string-to-integer object")
            constants = raw
        initial_memory = {}
        if args.memory:
            raw_memory = json.loads(_read(args.memory).decode("utf-8"))
            if not isinstance(raw_memory, dict) or not all(
                str(key).lstrip("-").isdigit() and isinstance(value, int)
                for key, value in raw_memory.items()
            ):
                raise TraceRefusal("memory_invalid", "memory must be an address-to-integer object")
            initial_memory = {int(key): value for key, value in raw_memory.items()}
        initial_registers = {}
        if args.registers:
            raw_registers = json.loads(_read(args.registers).decode("utf-8"))
            if not isinstance(raw_registers, dict) or not all(
                str(key).lstrip("-").isdigit() and isinstance(value, int)
                for key, value in raw_registers.items()
            ):
                raise TraceRefusal("registers_invalid", "registers must be a register-to-integer object")
            initial_registers = {int(key): value for key, value in raw_registers.items()}
        result = execute(
            sass,
            lane_id=args.lane,
            constants=constants,
            initial_memory=initial_memory,
            initial_registers=initial_registers,
            max_steps=args.max_steps,
            bound=args.bound,
        )
        if manifest is not None:
            result["artifact_manifest_sha256"] = hashlib.sha256(_read(args.manifest)).hexdigest()
    except (TraceRefusal, UnicodeDecodeError, json.JSONDecodeError) as exc:
        code = exc.code if isinstance(exc, TraceRefusal) else "input_invalid"
        detail = exc.detail if isinstance(exc, TraceRefusal) else str(exc)
        print(json.dumps({"schema": "sm86_value_trace_v1", "status": "refused", "code": code, "detail": detail}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
