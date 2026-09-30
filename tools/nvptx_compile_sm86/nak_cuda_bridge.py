#!/usr/bin/env python3
"""Insert a NAK SM86 shader body into an existing CUDA cubin template.

This is a bounded research bridge: it preserves NVIDIA-generated ELF/metadata,
replaces one text section, and optionally remaps constant-buffer offsets. It is
not a general CUDA linker or ABI implementation.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


NAK_INFO_SIZE = 728


def section(blob: bytes, name: str) -> tuple[int, int]:
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise ValueError("bridge requires a little-endian ELF64 cubin")
    # Read the ELF64 header fields explicitly (avoids platform ctypes layout).
    e_shoff = struct.unpack_from("<Q", blob, 40)[0]
    e_shentsize = struct.unpack_from("<H", blob, 58)[0]
    e_shnum = struct.unpack_from("<H", blob, 60)[0]
    e_shstrndx = struct.unpack_from("<H", blob, 62)[0]
    shstr_off = struct.unpack_from("<Q", blob, e_shoff + e_shstrndx * e_shentsize + 24)[0]
    shstr_size = struct.unpack_from("<Q", blob, e_shoff + e_shstrndx * e_shentsize + 32)[0]
    shstr = blob[shstr_off:shstr_off + shstr_size]
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from("<I", blob, off)[0]
        sh_offset = struct.unpack_from("<Q", blob, off + 24)[0]
        sh_size = struct.unpack_from("<Q", blob, off + 32)[0]
        end = shstr.find(b"\0", sh_name)
        if shstr[sh_name:end].decode() == name:
            return sh_offset, sh_size
    raise ValueError(f"ELF section not found: {name}")


def bridge(template: Path, nak_shader: Path, output: Path, text_name: str,
           maps: list[tuple[int, int, int]]) -> None:
    data = bytearray(template.read_bytes())
    raw = nak_shader.read_bytes()
    if len(raw) <= NAK_INFO_SIZE:
        raise ValueError("NAK shader has no code after its 728-byte info header")
    code = bytearray(raw[NAK_INFO_SIZE:])
    if len(code) % 16:
        raise ValueError(f"NAK code is not 16-byte aligned: {len(code)}")
    text_off, text_size = section(data, text_name)
    if len(code) > text_size:
        raise ValueError(f"NAK code {len(code)} exceeds template text {text_size}")
    for at, old, new in maps:
        # LEA constant offsets are encoded in bits 63:32 as offset/4 << 5.
        # The caller supplies byte offsets; patch only the expected field.
        if at >= len(code) // 16:
            raise ValueError(f"constant remap instruction index {at} is out of range")
        inst = at * 16
        word = struct.unpack_from("<Q", code, inst)[0]
        field = (word >> 32) & 0xFFFF
        expected = old << 6
        if field != expected:
            raise ValueError(f"instruction {at}: expected c0 offset 0x{old:x}, found field 0x{field:x}")
        word = (word & 0xFFFFFFFF) | (((new << 6) & 0xFFFF) << 32)
        struct.pack_into("<Q", code, inst, word)
    # Preserve the NAK terminal instruction as padding. CUDA metadata remains
    # the template's; this is intentionally a bounded body replacement.
    pad = bytes(code[-16:])
    padded = code + pad * ((text_size - len(code) + 15) // 16)
    data[text_off:text_off + text_size] = padded[:text_size]
    output.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_cubin", type=Path)
    parser.add_argument("nak_shader_bin", type=Path)
    parser.add_argument("output_cubin", type=Path)
    parser.add_argument("--text-section", default=".text.value_kernel")
    parser.add_argument("--map", dest="maps", action="append", default=[],
                        metavar="INDEX:OLD:NEW",
                        help="constant byte-offset remap at instruction index")
    args = parser.parse_args()
    maps = []
    for item in args.maps:
        try:
            index, old, new = (int(x, 0) for x in item.split(":", 2))
        except Exception as exc:
            raise SystemExit(f"invalid --map {item!r}: {exc}") from exc
        maps.append((index, old, new))
    bridge(args.template_cubin, args.nak_shader_bin, args.output_cubin,
           args.text_section, maps)
    print(f"bridged {args.nak_shader_bin} into {args.output_cubin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
