"""Bounded parser/refusal tests for the enhanced-to-traceg probe."""

from pathlib import Path

import pytest

from tools.traceg_adapter_probe.adapter import (
    AdapterRefusal,
    CanonicalRecord,
    RefusalCode,
    _canonical_register,
    _format_record,
    parse_traceg,
)


def test_legacy_record_round_trip(tmp_path: Path) -> None:
    record = CanonicalRecord(
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
    path = tmp_path / "kernel.traceg"
    path.write_text(
        "insts = 2\n"
        "thread block = 0,0,0\n"
        "cluster id = 0,0,0\n"
        "cluster cta = 0,0,0\n"
        "cluster rank = 0\n"
        "warp = 0\n"
        "#BEGIN_TB\n"
        f"{_format_record(record)}\n"
        "0030 00000000 0 EXIT 0 0 0\n"
        "#END_TB\n"
    )
    parsed, terminal = parse_traceg(path)
    assert parsed == [record]
    assert terminal


def test_zero_register_is_normalized_for_legacy_parser() -> None:
    assert _canonical_register("RZ", 0, "source") == "R255"


def test_unsupported_register_fails_typed() -> None:
    with pytest.raises(AdapterRefusal) as raised:
        _canonical_register("UR4", 2, "source")
    assert raised.value.code is RefusalCode.REGISTER_TOKEN
