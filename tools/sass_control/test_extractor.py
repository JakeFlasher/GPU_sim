from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from tools.sass_control import extractor


def _align(value: int, boundary: int) -> int:
    return (value + boundary - 1) // boundary * boundary


def _synthetic_cuda_elf(function: str, raw_text: bytes) -> bytes:
    section_names = ["", ".shstrtab", ".strtab", ".symtab", f".text.{function}"]
    shstr = bytearray(b"\0")
    shname_offsets = {"": 0}
    for name in section_names[1:]:
        shname_offsets[name] = len(shstr)
        shstr.extend(name.encode("ascii") + b"\0")
    strings = b"\0" + function.encode("ascii") + b"\0"
    symbols = b"\0" * 24 + struct.pack(
        "<IBBHQQ", 1, 2, 0, 4, 0, len(raw_text),
    )

    shstr_offset = 64
    strings_offset = shstr_offset + len(shstr)
    symbols_offset = _align(strings_offset + len(strings), 8)
    text_offset = _align(symbols_offset + len(symbols), 128)
    section_table_offset = _align(text_offset + len(raw_text), 8)
    payload = bytearray(section_table_offset + 5 * 64)
    ident = b"\x7fELF" + bytes((2, 1, 1, 0x41, 8)) + b"\0" * 7
    struct.pack_into(
        "<16sHHIQQQIHHHHHH", payload, 0,
        ident, 2, 190, 1, 0, 0, section_table_offset, 0x6005604,
        64, 56, 0, 64, 5, 1,
    )
    payload[shstr_offset:shstr_offset + len(shstr)] = shstr
    payload[strings_offset:strings_offset + len(strings)] = strings
    payload[symbols_offset:symbols_offset + len(symbols)] = symbols
    payload[text_offset:text_offset + len(raw_text)] = raw_text
    headers = (
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (shname_offsets[".shstrtab"], 3, 0, 0, shstr_offset, len(shstr), 0, 0, 1, 0),
        (shname_offsets[".strtab"], 3, 0, 0, strings_offset, len(strings), 0, 0, 1, 0),
        (shname_offsets[".symtab"], 2, 0, 0, symbols_offset, len(symbols), 2, 1, 8, 24),
        (
            shname_offsets[f".text.{function}"], 1, 0x6, 0,
            text_offset, len(raw_text), 0, 0, 128, 0,
        ),
    )
    for index, header in enumerate(headers):
        struct.pack_into("<IIQQQQIIQQ", payload, section_table_offset + index * 64, *header)
    return bytes(payload)


def test_decode_rejects_barrier_index_six_like_rust_consumer() -> None:
    control = 6 << 5
    with pytest.raises(ValueError, match="barrier index 6"):
        extractor.decode_control(control << 41)


def test_exact_function_text_uses_closed_list_then_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "vendor.so"
    binary.write_bytes(b"vendor module")
    function = "ampere_test_kernel"
    section = f"x-{function}.sm_80.elf.bin"
    raw = b"".join((
        struct.pack("<QQ", 0x10, 0x000FE400078E00FF),
        struct.pack("<QQ", 0x20, 0x000FE80000000F00),
    ))
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[1] == "--list-text":
            return subprocess.CompletedProcess(
                command, 0,
                stdout=f"SASS text section 7 : {section}\n", stderr="",
            )
        assert command[1:3] == ["--extract-text", section]
        out = Path(kwargs["cwd"]) / section
        out.write_bytes(raw)
        return subprocess.CompletedProcess(
            command, 0,
            stdout=f"Extracting SASS text section 7 : {section}\n", stderr="",
        )

    monkeypatch.setattr(extractor, "assert_toolkit_supported", lambda _nvcc: "V13.2.51")
    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    result = extractor.extract_exact_function_text(
        binary, function_name=function, allowed_code_arches=("sm_80",),
        nvcc="/pinned/nvcc", cuobjdump="/pinned/cuobjdump",
    )
    assert calls == [
        ["/pinned/cuobjdump", "--list-text", str(binary)],
        ["/pinned/cuobjdump", "--extract-text", section, str(binary)],
    ]
    assert result["code_arch"] == "sm_80"
    assert result["section_index"] == 7
    assert result["section_name"] == section
    assert result["raw_text_bytes"] == raw
    assert result["raw_text_size_bytes"] == 32
    assert [row["pc"] for row in result["records"]] == [0, 16]
    assert all(row["status"] == "decoded" for row in result["records"])


def test_long_function_cross_binds_abbreviated_catalog_to_exact_cubin_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "cutlass.so"
    binary.write_bytes(b"host module")
    function = "_ZN7cutlass6KernelI" + "N" * 400 + "Ev"
    alias = "x-_ZN7cutl-1.sm_86.elf.bin"
    cubin_name = "cutlass.1.sm_86.cubin"
    raw = b"".join((
        struct.pack("<QQ", 0x10, 0x000FE400078E00FF),
        struct.pack("<QQ", 0x20, 0x000FE80000000F00),
    ))
    cubin = _synthetic_cuda_elf(function, raw)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[1] == "--list-text":
            return subprocess.CompletedProcess(
                command, 0,
                stdout=f"SASS text section 1 : {alias}\n", stderr="",
            )
        if command[1] == "--list-elf":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"ELF file    1: {cubin_name}\n", stderr="",
            )
        if command[1] == "--extract-elf":
            (Path(kwargs["cwd"]) / cubin_name).write_bytes(cubin)
            return subprocess.CompletedProcess(
                command, 0,
                stdout=f"Extracting ELF file    1: {cubin_name}\n", stderr="",
            )
        assert command[1:3] == ["--extract-text", alias]
        (Path(kwargs["cwd"]) / alias).write_bytes(raw)
        return subprocess.CompletedProcess(
            command, 0,
            stdout=f"Extracting SASS text section 1 : {alias}\n", stderr="",
        )

    monkeypatch.setattr(extractor, "assert_toolkit_supported", lambda _nvcc: "V13.2.51")
    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    result = extractor.extract_exact_function_text(
        binary, function_name=function, allowed_code_arches=("sm_86",),
        nvcc="/pinned/nvcc", cuobjdump="/pinned/cuobjdump",
    )
    assert calls == [
        ["/pinned/cuobjdump", "--list-text", str(binary)],
        ["/pinned/cuobjdump", "--list-elf", str(binary)],
        ["/pinned/cuobjdump", "--extract-elf", cubin_name, str(binary)],
        ["/pinned/cuobjdump", "--extract-text", alias, str(binary)],
    ]
    assert result["section_identity_method"] == "cubin_symbol_raw_match"
    assert result["section_name"] == alias
    assert result["raw_text_bytes"] == raw
    assert result["cubin_symbol_proof"] == {
        "symbol_index": 1,
        "symbol_table_index": 3,
        "text_section_index": 4,
        "text_section_name": f".text.{function}",
        "code_arch": "sm_86",
        "cubin_catalog_index": 1,
        "cubin_file_name": cubin_name,
        "cubin_sha256": extractor._sha256_bytes(cubin),
        "catalog_raw_match_count": 1,
    }


def test_long_function_rejects_nonunique_catalog_payload_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "cutlass.so"
    binary.write_bytes(b"host module")
    function = "_ZN7cutlass6KernelI" + "N" * 400 + "Ev"
    aliases = (
        "x-_ZN7cutl-1.sm_86.elf.bin",
        "x-_ZN7cutl-2.sm_86.elf.bin",
    )
    cubin_name = "cutlass.1.sm_86.cubin"
    raw = struct.pack("<QQ", 0x10, 0x000FE400078E00FF)
    cubin = _synthetic_cuda_elf(function, raw)

    def fake_run(command, **kwargs):
        if command[1] == "--list-text":
            return subprocess.CompletedProcess(
                command, 0,
                stdout="".join(
                    f"SASS text section {index} : {alias}\n"
                    for index, alias in enumerate(aliases, start=1)
                ),
                stderr="",
            )
        if command[1] == "--list-elf":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"ELF file    1: {cubin_name}\n", stderr="",
            )
        if command[1] == "--extract-elf":
            (Path(kwargs["cwd"]) / cubin_name).write_bytes(cubin)
            return subprocess.CompletedProcess(
                command, 0,
                stdout=f"Extracting ELF file    1: {cubin_name}\n", stderr="",
            )
        alias = command[2]
        (Path(kwargs["cwd"]) / alias).write_bytes(raw)
        index = aliases.index(alias) + 1
        return subprocess.CompletedProcess(
            command, 0,
            stdout=f"Extracting SASS text section {index} : {alias}\n", stderr="",
        )

    monkeypatch.setattr(extractor, "assert_toolkit_supported", lambda _nvcc: "V13.2.51")
    monkeypatch.setattr(extractor.subprocess, "run", fake_run)
    with pytest.raises(
        extractor.FunctionTextExtractionError, match="2 exact list-text payload matches",
    ):
        extractor.extract_exact_function_text(
            binary, function_name=function, allowed_code_arches=("sm_86",),
            nvcc="/pinned/nvcc", cuobjdump="/pinned/cuobjdump",
        )


def test_exact_function_text_rejects_ambiguous_or_unsafe_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "vendor.so"
    binary.write_bytes(b"vendor module")
    monkeypatch.setattr(extractor, "assert_toolkit_supported", lambda _nvcc: "V13.2.51")
    monkeypatch.setattr(
        extractor.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0,
            stdout=(
                "SASS text section 1 : x-k.sm_80.elf.bin\n"
                "SASS text section 2 : x-k.sm_86.elf.bin\n"
            ),
            stderr="",
        ),
    )
    with pytest.raises(extractor.FunctionTextExtractionError, match="2 exact"):
        extractor.extract_exact_function_text(
            binary, function_name="k", allowed_code_arches=("sm_80", "sm_86"),
            nvcc="nvcc", cuobjdump="cuobjdump",
        )
    with pytest.raises(extractor.FunctionTextExtractionError, match="unsafe"):
        extractor.extract_exact_function_text(
            binary, function_name="../k", allowed_code_arches=("sm_80",),
            nvcc="nvcc", cuobjdump="cuobjdump",
        )


def test_exact_function_text_types_pinned_nvcc_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "vendor.so"
    binary.write_bytes(b"vendor module")

    def fail_probe(_nvcc: str) -> str:
        raise OSError("nvcc unavailable")

    monkeypatch.setattr(extractor, "assert_toolkit_supported", fail_probe)
    with pytest.raises(
        extractor.FunctionTextExtractionError,
        match="pinned nvcc toolkit probe failed",
    ):
        extractor.extract_exact_function_text(
            binary, function_name="k", allowed_code_arches=("sm_80",),
            nvcc="/pinned/nvcc", cuobjdump="/pinned/cuobjdump",
        )
