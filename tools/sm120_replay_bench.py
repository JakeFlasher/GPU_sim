#!/usr/bin/env python3
"""Small real-corpus adapters shared by replay and NCU join tools.

The module name is retained for compatibility with historical SM120 reports,
but the adapter is architecture-neutral.  In particular, the SM86 capture
producer reused the ``native_static_sm120_v3`` exporter schema while recording
``architecture_version: 86``.  The target architecture is therefore inferred
from each kernel's explicit architecture field rather than from the stale
producer label.  NCU observations remain outside the replay input path.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
# Capture roots are supplied by the caller; no workstation-specific default is
# embedded in the adapter.
DEFAULT_WORK_ROOT = Path(".")
DEFAULT_SIM = REPO_ROOT / "sim/target/release/sim-run"
DEFAULT_CARD = REPO_ROOT / "arch_profiles/cards/sm_120-rtx-5090.json"


def architecture_token(value: Any) -> str:
    """Return the canonical deploy token for an exporter architecture value.

    NVBit metadata uses integer SM versions while cards/sidecars use the
    canonical ``sm_XX`` spelling.  Do not accept arbitrary strings here: a
    mixed or unknown architecture must fail closed during sidecar preparation
    instead of accidentally receiving another architecture's control model.
    """
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError("architecture_version must be an integer")
    try:
        version = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"unsupported architecture_version: {value!r}") from None
    if version == 80:
        return "sm_80"
    if version == 86:
        return "sm_86"
    if version == 120:
        return "sm_120"
    raise ValueError(f"unsupported architecture_version: {version}")


def metadata_architecture(metadata: dict[str, Any]) -> str:
    """Resolve one metadata document's architecture, rejecting mixed kernels."""
    kernels = metadata.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("native static metadata has no kernels")
    tokens: set[str] = set()
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError("native static kernel entry is not an object")
        tokens.add(architecture_token(kernel.get("architecture_version")))
    if len(tokens) != 1:
        raise ValueError(f"native static metadata mixes architectures: {sorted(tokens)}")
    return next(iter(tokens))


def metadata_architectures(metadata: dict[str, Any]) -> set[str]:
    """Return all source code architectures represented by one root."""
    kernels = metadata.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("native static metadata has no kernels")
    return {
        architecture_token(kernel.get("architecture_version"))
        for kernel in kernels
        if isinstance(kernel, dict)
    }


def card_architecture(card: Path) -> str:
    """Read and validate the canonical architecture token from an arch card."""
    try:
        value = json.loads(card.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read architecture card {card}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"architecture card {card} is not an object")
    token = value.get("arch_id")
    if not isinstance(token, str):
        raise ValueError(f"architecture card {card} has no arch_id")
    if token not in {"sm_86", "sm_120"}:
        raise ValueError(f"unsupported architecture card token: {token!r}")
    return token


def discover_captures(work_root: Path, selected: Iterable[str] | None) -> list[Path]:
    captures = work_root / "captures"
    if not captures.is_dir():
        return []
    wanted = set(selected or ())
    found: list[Path] = []
    for directory, subdirs, files in os.walk(captures):
        # `_diagnostics` contains failed/retired attempts and is not part of
        # the canonical SM86 acceptance corpus. Keep it available for an
        # explicitly selected diagnostic path, but never include it in an
        # unfiltered complete-corpus discovery.
        relative_dir = Path(directory).relative_to(captures)
        if relative_dir.parts and relative_dir.parts[0] == "_diagnostics" and not wanted:
            subdirs[:] = []
            continue
        if "dynamic_trace.pb" not in files:
            continue
        subdirs[:] = []
        path = Path(directory)
        relative = path.relative_to(captures).as_posix()
        if not wanted or relative in wanted or path.name in wanted:
            found.append(path)
    return sorted(found)


def load_static_metadata(capture: Path) -> tuple[dict[str, Any], Path] | None:
    """Load the one declared static sidecar and return its source path.

    The enhanced document is a compatibility fallback only when the native
    file is absent.  A present-but-corrupt preferred sidecar must fail closed;
    silently switching producers can change opcode/control provenance.
    """
    native = capture / "extra_info/native_static_metadata.json"
    enhanced = capture / "extra_info/enhanced_execution_info.json"
    path: Path | None = None
    for candidate in (native, enhanced):
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect static metadata {candidate}: {error}") from error
        try:
            mode = candidate.stat().st_mode
        except OSError as error:
            raise ValueError(f"cannot read static metadata {candidate}: {error}") from error
        if not stat.S_ISREG(mode):
            raise ValueError(f"static metadata path is not a regular file: {candidate}")
        path = candidate
        break
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"static metadata {path} is malformed: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"static metadata {path} is not an object")
    return value, path


def _wire_control(high_word: int) -> dict[str, int]:
    wire = (high_word >> 41) & ((1 << 23) - 1)
    if wire >> 21:
        raise ValueError("control word has nonzero reserved bits")
    return {
        "stall": wire & 0xF,
        "yield_flag": (wire >> 4) & 0x1,
        "write_barrier": (wire >> 5) & 0x7,
        "read_barrier": (wire >> 8) & 0x7,
        "wait_mask": (wire >> 11) & 0x3F,
        "reuse": (wire >> 17) & 0xF,
    }


def _parse_hex_word(value: Any, field: str, *, bits: int = 64) -> tuple[int, str]:
    """Parse an exporter word without silently manufacturing zero controls."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a hexadecimal string or integer")
    if isinstance(value, int):
        # JSON numeric words are already numeric.  Treating integer ``10`` as
        # the hexadecimal text ``0x10`` changes it to sixteen.
        number = value
    else:
        text = str(value).strip()
        if text.lower().startswith("0x"):
            digits = text[2:]
        else:
            digits = text
        if not digits or not re.fullmatch(r"[0-9a-fA-F]+", digits):
            raise ValueError(f"{field} is not valid hexadecimal: {value!r}")
        number = int(digits, 16)
    if number < 0 or number >= (1 << bits):
        raise ValueError(f"{field} exceeds {bits}-bit encoding: {value!r}")
    return number, f"0x{number:0{bits // 4}x}"


def synthesize_static_sidecar(
    capture: Path,
    metadata: dict[str, Any],
    output: Path,
    expected_arch: str | None = None,
    metadata_source: Path | None = None,
) -> dict[str, Any]:
    command_path = capture / "command.txt"
    command = command_path.read_text(encoding="utf-8").strip() if command_path.is_file() else ""
    binary = shlex.split(command)[0] if command else metadata.get("benchmark_name", "capture")
    metadata_kind = metadata.get("metadata_kind")
    enhanced_execution_info = not isinstance(metadata_kind, str) and all(
        isinstance(kernel, dict)
        and isinstance(kernel.get("instructions"), list)
        and all(
            isinstance(inst, dict) and isinstance(inst.get("control_bits"), dict)
            for inst in kernel.get("instructions", [])
        )
        for kernel in metadata.get("kernels", [])
    )
    # The SM86 capture lane intentionally retained the producer's old
    # ``native_static_sm120_v3`` label.  Version 3 describes the wire schema,
    # not the captured GPU.  Accept the known v3 labels and derive target arch
    # from every kernel's explicit architecture_version below.
    if enhanced_execution_info:
        metadata_kind = "enhanced_execution_info_sm86"
    elif not isinstance(metadata_kind, str) or not re.fullmatch(
        r"native_static_(?:sm86|sm120)_v3", metadata_kind
    ):
        raise ValueError(f"unsupported native static metadata kind: {metadata_kind!r}")
    if not enhanced_execution_info and metadata.get("static_exporter_version") != 3:
        raise ValueError("unsupported native static exporter version")
    source_arches = metadata_architectures(metadata)
    if expected_arch is not None:
        if source_arches == {expected_arch}:
            arch = expected_arch
        else:
            raise ValueError(
                f"capture metadata targets {sorted(source_arches)}, expected homogeneous {expected_arch}"
            )
    elif len(source_arches) == 1:
        arch = next(iter(source_arches))
    else:
        raise ValueError(f"capture metadata mixes architectures: {sorted(source_arches)}")
    functions: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, int]] = {}
    kernels = metadata["kernels"]
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError("native static kernel entry is not an object")
        raw_name = str(kernel.get("kernel_name", ""))
        if not raw_name:
            raise ValueError("native static kernel name is missing")
        code_arch = architecture_token(kernel.get("architecture_version"))
        if code_arch not in source_arches:
            raise ValueError(f"mixed kernel architecture for {raw_name}")
        name = re.sub(r"___[0-9]+$", "", raw_name)
        if not name:
            raise ValueError(f"kernel name normalizes to empty: {raw_name!r}")
        records: list[dict[str, Any]] = []
        instructions = kernel.get("instructions")
        if not isinstance(instructions, list):
            raise ValueError(f"instructions missing for kernel {raw_name}")
        for inst in instructions:
            if not isinstance(inst, dict):
                raise ValueError(f"instruction entry is not an object for {raw_name}")
            if enhanced_execution_info:
                bits = inst.get("control_bits")
                if not isinstance(bits, dict):
                    raise ValueError(f"control bits unavailable for {raw_name}")
                stall = int(bits.get("stall_count", 0))
                yield_flag = int(bool(bits.get("is_yield", False)))
                write_barrier = int(bits.get("id_new_write_barrier", 7))
                read_barrier = int(bits.get("id_new_read_barrier", 7))
                wait_mask = int(bits.get("wait_barrier_bits", 0))
                if not (0 <= stall <= 15 and 0 <= write_barrier <= 7
                        and 0 <= read_barrier <= 7 and 0 <= wait_mask <= 63):
                    raise ValueError(f"enhanced control bits out of range for {raw_name}")
                wire = (stall | (yield_flag << 4) | (write_barrier << 5)
                        | (read_barrier << 8) | (wait_mask << 11))
                high_word = wire << 41
                high_text = f"0x{high_word:016x}"
                low_text = "0x0000000000000000"
                decode_source = "enhanced_execution_info.control_bits"
            else:
                if inst.get("control_fields_available") is not True:
                    raise ValueError(f"control fields unavailable for {raw_name}")
                if inst.get("control_word_shift") != 41:
                    raise ValueError(f"unsupported control shift for {raw_name}")
                decode_source = inst.get("control_decode_source")
                if not isinstance(decode_source, str) or not decode_source:
                    raise ValueError(f"control decode provenance missing for {raw_name}")
                high_value = inst.get("control_word_hex")
                if high_value is None:
                    high_value = inst.get("word1_hex")
                high_word, high_text = _parse_hex_word(high_value, "control_word_hex")
                low_value = inst.get("word0_hex")
                _, low_text = _parse_hex_word(low_value, "word0_hex")
            pc_value = inst.get("pc_num_dec")
            if isinstance(pc_value, bool) or not isinstance(pc_value, int) or pc_value < 0:
                raise ValueError(f"invalid pc_num_dec for {raw_name}")
            pc = pc_value
            sass = str(
                inst.get("sass") or inst.get("opcode") or inst.get("op_code") or "UNKNOWN"
            ).strip().rstrip(";").strip()
            predicate = None
            if sass.startswith("@"):
                predicate, _, sass = sass.partition(" ")
                sass = sass.strip()
            records.append({
                "pc": pc,
                "predicate": predicate,
                "opcode_text": sass,
                "low_word": low_text,
                "high_word": high_text,
                "status": "decoded",
                "control": _wire_control(high_word),
                "control_decode_source": decode_source,
            })
        previous = functions.get(name)
        if previous is not None:
            # Real v3 exports can repeat the same normalized function for an
            # identical code object/reload.  Accept exact duplicates, but do
            # not let conflicting controls silently overwrite one another.
            if previous != records:
                raise ValueError(f"conflicting normalized kernel name: {name!r}")
            continue
        functions[name] = records
        summaries[name] = {
            "instructions": len(records),
            "decoded": len(records),
            "unknown_encoding": 0,
            "stall_cycles_static_sum": sum(int(x["control"]["stall"]) for x in records),
        }
    document = {
        "arch": arch,
        "binary": binary,
        "functions": functions,
            "parser_contract": {
            "supported_toolkit_versions": [
                "NVBit-native-static-v3",
                "NVBit-enhanced-execution-info-v1" if enhanced_execution_info else "NVBit-native-static-v3",
            ],
            "supported_arches": [arch],
            "control_layout": {
                "source": "high_word >> 41 (23 bits)",
                "stall": "bits 0:3",
                "yield_flag": "bit 4",
                "write_barrier": "bits 5:7 (7 = none)",
                "read_barrier": "bits 8:10 (7 = none)",
                "wait_mask": "bits 11:16",
                "reuse": "bits 17:20",
                "reserved_must_be_zero": "bits 21:22",
            },
        },
        "toolkit_version": (
            f"NVBit-enhanced-{arch.replace('_', '')}-execution-info-v1"
            if enhanced_execution_info
            else f"NVBit-native-{arch.replace('_', '')}-static-v3"
        ),
        "binary_source": (
            metadata_source.relative_to(capture).as_posix()
            if metadata_source is not None
            else "unknown"
        ),
        "metadata_provenance": {
            "metadata_kind": metadata.get("metadata_kind", metadata_kind),
            "static_exporter_version": metadata.get("static_exporter_version"),
            "static_source": metadata.get("static_source"),
            "control_latency_claims": metadata.get("control_latency_claims"),
            "runtime_architecture": arch,
            "source_architectures": sorted(source_arches),
            "function_source_architectures": {
                re.sub(r"___[0-9]+$", "", str(kernel.get("kernel_name"))):
                architecture_token(kernel.get("architecture_version"))
                for kernel in kernels
                if isinstance(kernel, dict)
            },
        },
        "summary": {
            "functions": summaries,
            "total_decoded": sum(item["decoded"] for item in summaries.values()),
            "total_unknown_encoding": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_ncu_raw(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header_index = next((i for i, row in enumerate(rows) if row and row[0] == "ID" and "Kernel Name" in row), None)
    if header_index is None or header_index + 2 >= len(rows):
        return []
    header = rows[header_index]
    units = rows[header_index + 1]
    index = {name: i for i, name in enumerate(header)}
    output: list[dict[str, Any]] = []
    identity = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"}
    for row in rows[header_index + 2 :]:
        if not row or not row[0].strip().isdigit():
            continue
        metrics: dict[str, dict[str, Any]] = {}
        for name, i in index.items():
            if name in identity or i >= len(row):
                continue
            numeric = _number(row[i])
            if numeric is not None:
                metrics[name] = {"value": numeric, "unit": units[i] if i < len(units) else ""}
        def field(name: str) -> str:
            i = index.get(name, -1)
            return row[i] if 0 <= i < len(row) else ""
        output.append({
            "ncu_launch_id": int(row[0]),
            "kernel_name": field("Kernel Name"),
            "grid_size": field("Grid Size"),
            "block_size": field("Block Size"),
            "context": field("Context"),
            "stream": field("Stream"),
            "metrics": metrics,
        })
    return output


def _shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)):
        if len(value) != 3 or any(isinstance(item, bool) for item in value):
            return None
        try:
            result = tuple(int(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if all(item >= 0 for item in result) else None
    if not isinstance(value, str):
        return None
    values = re.findall(r"-?\d+", value)
    if len(values) != 3:
        return None
    try:
        result = tuple(int(item) for item in values)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(item >= 0 for item in result) else None


def _normalized_kernel_name(value: str) -> str:
    return re.sub(r"___[0-9]+$", "", value or "")


def join_ncu_rows(sim_launches: list[dict[str, Any]], ncu_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def compatible(launch: dict[str, Any], row: dict[str, Any]) -> bool:
        return all(
            _shape(launch.get(left)) is None
            or _shape(row.get(right)) is None
            or _shape(launch.get(left)) == _shape(row.get(right))
            for left, right in (("grid", "grid_size"), ("block", "block_size"))
        )
    sim_ids = [launch.get("kernel_id") for launch in sim_launches]
    ncu_ids = [row.get("ncu_launch_id") for row in ncu_rows]
    ordinal: dict[int, int] = {}
    if (
        len(sim_launches) == len(ncu_rows)
        and all(isinstance(value, int) for value in sim_ids + ncu_ids)
        and len(set(sim_ids)) == len(sim_ids)
        and len(set(ncu_ids)) == len(ncu_ids)
    ):
        ordinal = dict(zip(
            sorted(range(len(sim_ids)), key=sim_ids.__getitem__),
            sorted(range(len(ncu_ids)), key=ncu_ids.__getitem__),
        ))
    names = [_normalized_kernel_name(str(row.get("kernel_name", ""))) for row in ncu_rows]
    unused = set(range(len(ncu_rows)))
    joined: list[dict[str, Any]] = []
    for index, launch in enumerate(sim_launches):
        sim_name = str(launch.get("kernel_name", ""))
        exact = [candidate for candidate in unused if names[candidate] == _normalized_kernel_name(sim_name) and compatible(launch, ncu_rows[candidate])]
        candidate = ordinal.get(index, index)
        if candidate in unused and compatible(launch, ncu_rows[candidate]) and (not exact or candidate in exact):
            status = "identity" if len(exact) == 1 else "ordered_fallback"
        elif len(exact) == 1:
            candidate, status = exact[0], "identity"
        else:
            candidate, status = None, "unmatched"
        if candidate is None:
            joined.append({"launch_index": index, "kernel_id": launch.get("kernel_id"), "kernel_name": sim_name, "join_status": status, "order_basis": "reported_launch_order"})
            continue
        unused.discard(candidate)
        target = ncu_rows[candidate]
        joined.append({
            "launch_index": index,
            "kernel_id": launch.get("kernel_id"),
            "kernel_name": sim_name,
            "join_status": status,
            "order_basis": "kernel_id_rank" if ordinal else "reported_launch_order",
            "ncu_row_index": candidate,
            "ncu_launch_id": target.get("ncu_launch_id"),
            "ncu_kernel_name": target.get("kernel_name"),
            "ncu_gpu_duration_us": (target.get("metrics", {}).get("gpu__time_duration.sum") or {}).get("value"),
        })
    return joined
