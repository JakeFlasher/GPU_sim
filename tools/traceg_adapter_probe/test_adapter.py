"""Adversarial tests for the bounded enhanced-to-``traceg`` probe.

The protobuf cases deliberately start from the retained repository canary and
copy it into a temporary directory before making the smallest fixture change.
They are representation/refusal tests only; no external simulator is invoked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from tools.traceg_adapter_probe.adapter import (
    AdapterRefusal,
    CanonicalRecord,
    RefusalCode,
    _load_control_sidecar,
    _canonical_register,
    _format_record,
    _StaticInstruction,
    _static_operands,
    parse_traceg,
)


REPO = Path(__file__).resolve().parents[2]
CANARY = REPO / "sim/run/tests/fixtures/addr_carry_canary_bundle"
CANARY_TRACE = CANARY / "trace/dynamic_trace.pb"
CANARY_THREADBLOCK = (
    CANARY
    / "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"
)
CANARY_CONTROL = CANARY / "control/canary_sm86.control_bits.json"
CANARY_EXEC_INFO = CANARY / "trace/extra_info/enhanced_execution_info.json"


def _record() -> CanonicalRecord:
    return CanonicalRecord(
        block=(0, 0, 0),
        warp=0,
        pc=0x10,
        mask=0xFFFF_FFFF,
        opcode="IADD3",
        dest_regs=("R0",),
        src_regs=("R1", "R2", "R3"),
        mem_width=0,
        addresses=(),
    )


def _traceg_text(*, body: str | None = None, count: int = 2) -> str:
    body = body or f"{_format_record(_record())}\n0030 00000000 0 EXIT 0 0 0"
    return (
        "-kernel name = fixture\n"
        "-kernel id = 1\n"
        "-grid dim = (1,1,1)\n"
        "-block dim = (32,1,1)\n"
        "-shmem = 0\n"
        "-nregs = 32\n"
        "-binary version = 86\n"
        "-cuda stream id = 0\n"
        "-shmem base_addr = 0x0000000000000000\n"
        "-local mem base_addr = 0x0000000000000000\n"
        "-nvbit version = 1.8\n"
        "-accelsim tracer version = 5\n"
        "-enable lineinfo = 0\n"
        "\n"
        "#traces format = PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width immediate\n"
        "#BEGIN_TB\n"
        "thread block = 0,0,0\n"
        "cluster id = 0,0,0\n"
        "cluster cta = 0,0,0\n"
        "cluster rank = 0\n"
        "warp = 0\n"
        f"insts = {count}\n"
        f"{body}\n"
        "#END_TB\n"
    )


def test_legacy_record_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "kernel.traceg"
    # ``insts`` before BEGIN_TB is accepted because retained legacy traces use
    # both placements; the adapter's own emission places it inside the TB.
    path.write_text(
        "insts = 2\n"
        "thread block = 0,0,0\n"
        "cluster id = 0,0,0\n"
        "cluster cta = 0,0,0\n"
        "cluster rank = 0\n"
        "warp = 0\n"
        "#BEGIN_TB\n"
        f"{_format_record(_record())}\n"
        "0030 00000000 0 EXIT 0 0 0\n"
        "#END_TB\n"
    )
    parsed, terminal = parse_traceg(path)
    assert parsed == [_record()]
    assert terminal


def test_zero_register_is_normalized_for_legacy_parser() -> None:
    assert _canonical_register("RZ", 0, "source") == "R255"


def test_unsupported_register_fails_typed() -> None:
    with pytest.raises(AdapterRefusal) as raised:
        _canonical_register("UR4", 2, "source")
    assert raised.value.code is RefusalCode.REGISTER_TOKEN


def test_common_alu_static_operands_include_fadd_and_decimal_immediate() -> None:
    opcode, sources, immediate = _static_operands(
        _StaticInstruction(0, "IADD3 R0, R1, R2, 08", None, "decoded"),
        0,
    )
    assert (opcode, sources, immediate) == ("IADD3", ("R1", "R2"), 8)
    opcode, sources, immediate = _static_operands(
        _StaticInstruction(0, "FADD R4, R5, R6", None, "decoded"),
        0,
    )
    assert (opcode, sources, immediate) == ("FADD", ("R5", "R6"), 0)


def test_control_json_duplicate_key_is_refused() -> None:
    with pytest.raises(AdapterRefusal) as raised:
        _load_control_sidecar(
            b'{"arch":"sm_86","functions":{},"functions":{}}',
            Path("control.json"),
        )
    assert raised.value.code is RefusalCode.SIDECAR_DECODE


def test_missing_terminal_is_typed_refusal(tmp_path: Path) -> None:
    path = tmp_path / "missing-exit.traceg"
    path.write_text(
        "#BEGIN_TB\n"
        "thread block = 0,0,0\n"
        "cluster id = 0,0,0\n"
        "cluster cta = 0,0,0\n"
        "cluster rank = 0\n"
        "warp = 0\n"
        "insts = 1\n"
        "0010 ffffffff 1 R0 IADD3 3 R1 R2 R3 0 0\n"
        "#END_TB\n"
    )
    with pytest.raises(AdapterRefusal) as raised:
        parse_traceg(path)
    assert raised.value.code is RefusalCode.TRACEG_PARSE


def test_trailing_record_after_exit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "trailing.traceg"
    path.write_text(
        "#BEGIN_TB\n"
        "thread block = 0,0,0\n"
        "cluster id = 0,0,0\n"
        "cluster cta = 0,0,0\n"
        "cluster rank = 0\n"
        "warp = 0\n"
        "insts = 2\n"
        "0030 00000000 0 EXIT 0 0 0\n"
        "0010 ffffffff 1 R0 IADD3 3 R1 R2 R3 0 0\n"
        "#END_TB\n"
    )
    with pytest.raises(AdapterRefusal) as raised:
        parse_traceg(path)
    assert raised.value.code is RefusalCode.TRACEG_PARSE


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    [
        (
            "unknown_prefix",
            lambda text: text.replace("#BEGIN_TB", "untrusted = 1\n#BEGIN_TB"),
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "duplicate_begin",
            lambda text: text.replace("#BEGIN_TB", "#BEGIN_TB\n#BEGIN_TB"),
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "duplicate_header",
            lambda text: text.replace("-kernel id = 1\n", "-kernel id = 1\n-kernel id = 1\n"),
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "trailing_data",
            lambda text: text + "trailing = 1\n",
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "wrong_count",
            lambda text: text.replace("insts = 2", "insts = 3"),
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "missing_exit",
            lambda text: text.replace("0030 00000000 0 EXIT 0 0 0\n", ""),
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "record_after_exit",
            lambda text: text.replace(
                "0030 00000000 0 EXIT 0 0 0\n",
                "0030 00000000 0 EXIT 0 0 0\n0040 ffffffff 1 R0 IADD3 3 R1 R2 R3 0 0\n",
            ).replace("insts = 2", "insts = 3"),
            RefusalCode.TRACEG_PARSE,
        ),
        (
            "partial_mask",
            lambda text: text.replace("ffffffff 1 R0", "0000ffff 1 R0"),
            RefusalCode.TRACEG_PARSE,
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_parser_refuses_malformed_legacy_input(
    tmp_path: Path,
    label: str,
    mutate: Callable[[str], str],
    expected: RefusalCode,
) -> None:
    del label
    path = tmp_path / "malformed.traceg"
    path.write_text(mutate(_traceg_text()))
    with pytest.raises(AdapterRefusal) as raised:
        parse_traceg(path)
    assert raised.value.code is expected


def _load_bindings_or_skip() -> Any:
    try:
        from tools.nvbit_tracer.proto_binding_loader import (
            ProtoBindingRuntimePrerequisiteError,
            load_dynamic_trace_bindings,
        )

        return load_dynamic_trace_bindings()
    except (ImportError, ProtoBindingRuntimePrerequisiteError) as exc:
        pytest.skip(f"pinned protobuf runtime unavailable: {exc}")


def _copy_canary_inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    result = {
        "trace": inputs / "dynamic_trace.pb",
        "threadblock": inputs / "threadblock.pb",
        "control": inputs / "control_bits.json",
        "exec_info": inputs / "enhanced_execution_info.json",
    }
    shutil.copyfile(CANARY_TRACE, result["trace"])
    shutil.copyfile(CANARY_THREADBLOCK, result["threadblock"])
    shutil.copyfile(CANARY_CONTROL, result["control"])
    shutil.copyfile(CANARY_EXEC_INFO, result["exec_info"])
    return result


def _set_required_launch_metadata(bindings: Any, path: Path) -> None:
    root = bindings.trace_pb2.Trace()
    root.ParseFromString(path.read_bytes())
    device = next(iter(root.gpu_device.values()))
    stream = next(iter(device.streams.values()))
    kernel = stream.kernels[0]
    kernel.number_of_registers = 32
    kernel.size_shared_memory = 0
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    path.write_bytes(root.SerializeToString())


def _small_explicit_exit_bundle(tmp_path: Path) -> dict[str, Path]:
    """Make a tiny real protobuf bundle from the retained canary schema."""

    bindings = _load_bindings_or_skip()
    paths = _copy_canary_inputs(tmp_path)
    _set_required_launch_metadata(bindings, paths["trace"])

    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.block_id.x = 0
    threadblock.block_id.y = 0
    threadblock.block_id.z = 0
    warp = threadblock.warps[0]
    warp.id = 0
    for pc in (0, 16, 32, 48):
        instruction = warp.instructions.add()
        instruction.pc = pc
        # EXIT is a legacy zero-mask terminal marker.  Keeping the source
        # dynamic mask zero makes the external and local instruction
        # endpoints observe the same nonterminal work rather than silently
        # projecting a full-mask EXIT to zero.
        instruction.active_mask = 0 if pc == 48 else 0xFFFF_FFFF
        instruction.predicate_mask = 0 if pc == 48 else 0xFFFF_FFFF
        instruction.function_unique_id = 1
        instruction.warpid_tb = 0
        instruction.vpc = pc
    paths["threadblock"].write_bytes(threadblock.SerializeToString())

    control = json.loads(paths["control"].read_text())
    function_name, instructions = next(iter(control["functions"].items()))
    for instruction in instructions:
        if instruction["pc"] == 16:
            instruction["opcode_text"] = "FADD R0, R1, R2"
        if instruction["pc"] == 48:
            instruction["opcode_text"] = "EXIT"
    # The official control fixture contains all 54 static PCs.  Keeping the
    # complete set lets the optional enhanced-execution-info identity sidecar
    # prove PC-set equality while the dynamic payload remains bounded.
    control["functions"][function_name] = instructions
    paths["control"].write_text(json.dumps(control, sort_keys=True) + "\n")
    return paths


def test_real_protobuf_round_trip_with_dynamic_exit(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    out = tmp_path / "out"
    result = adapt(
        paths["trace"],
        paths["threadblock"],
        paths["control"],
        out,
        exec_info_path=paths["exec_info"],
    )
    assert result["schema"] == "sm86_traceg_adapter_probe_v2"
    assert result["termination"] == "dynamic_exit"
    assert result["record_count"] == 3
    parsed, terminal = parse_traceg(out / "kernel-1.traceg")
    assert terminal and len(parsed) == 3
    assert parsed[0].src_regs[-1] == "R255"
    assert parsed[1].opcode == "FADD"


def test_full_mask_dynamic_exit_is_not_projected_to_external_marker(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    bindings = _load_bindings_or_skip()
    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(paths["threadblock"].read_bytes())
    terminal = threadblock.warps[0].instructions[-1]
    terminal.active_mask = 0xFFFF_FFFF
    terminal.predicate_mask = 0xFFFF_FFFF
    paths["threadblock"].write_bytes(threadblock.SerializeToString())
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.TERMINATION_MALFORMED


def test_retained_canary_refuses_memory_projection(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _copy_canary_inputs(tmp_path)
    bindings = _load_bindings_or_skip()
    _set_required_launch_metadata(bindings, paths["trace"])
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    # The retained canary's first non-ALU instruction is LDG.E at record 5;
    # no memory enum is treated as an address projection.
    assert raised.value.code is RefusalCode.MEMORY_UNSUPPORTED
    assert raised.value.record_index == 5


def test_retained_canary_missing_nregs_is_typed(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    _load_bindings_or_skip()
    paths = _copy_canary_inputs(tmp_path)
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.LAUNCH_METADATA_MISSING


def test_dynamic_eof_is_refused_without_synthetic_exit(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    bindings = _load_bindings_or_skip()
    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(paths["threadblock"].read_bytes())
    del threadblock.warps[0].instructions[-1]
    paths["threadblock"].write_bytes(threadblock.SerializeToString())
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.TERMINATION_INCOMPLETE


def test_dynamic_exit_must_be_final(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    bindings = _load_bindings_or_skip()
    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(paths["threadblock"].read_bytes())
    instruction = threadblock.warps[0].instructions.add()
    instruction.pc = 64
    instruction.active_mask = 0xFFFF_FFFF
    instruction.predicate_mask = 0xFFFF_FFFF
    instruction.function_unique_id = 1
    instruction.warpid_tb = 0
    instruction.vpc = 64
    paths["threadblock"].write_bytes(threadblock.SerializeToString())
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.TERMINATION_MALFORMED


def test_control_join_mismatch_is_typed(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    bindings = _load_bindings_or_skip()
    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(paths["threadblock"].read_bytes())
    threadblock.warps[0].instructions[1].vpc = 0x10000
    threadblock.warps[0].instructions[1].pc = 0x10000
    paths["threadblock"].write_bytes(threadblock.SerializeToString())
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.FUNCTION_JOIN_MISMATCH


def test_memory_metadata_on_alu_is_refused(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    bindings = _load_bindings_or_skip()
    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(paths["threadblock"].read_bytes())
    threadblock.warps[0].instructions[0].data_width = 4
    paths["threadblock"].write_bytes(threadblock.SerializeToString())
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.MEMORY_METADATA


def test_duplicate_control_pc_is_typed(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    control = json.loads(paths["control"].read_text())
    function = next(iter(control["functions"].values()))
    function.append(dict(function[0]))
    paths["control"].write_text(json.dumps(control))
    with pytest.raises(AdapterRefusal) as raised:
        adapt(
            paths["trace"],
            paths["threadblock"],
            paths["control"],
            tmp_path / "out",
            exec_info_path=paths["exec_info"],
        )
    assert raised.value.code is RefusalCode.SIDECAR_DUPLICATE_JOIN


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    from tools.traceg_adapter_probe.adapter import adapt

    paths = _small_explicit_exit_bundle(tmp_path)
    out = tmp_path / "out"
    adapt(paths["trace"], paths["threadblock"], paths["control"], out, exec_info_path=paths["exec_info"])
    sentinel = out / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(AdapterRefusal) as raised:
        adapt(paths["trace"], paths["threadblock"], paths["control"], out, exec_info_path=paths["exec_info"])
    assert raised.value.code is RefusalCode.OUTPUT_EXISTS
    assert sentinel.read_text() == "keep"
