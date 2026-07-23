"""Static SASS control-bit extractor (consumer-side, no profiler).

Parses ``cuobjdump --dump-sass`` text and decodes, per instruction,
the scheduling control fields embedded in the high 64-bit word of each
128-bit sm_8x instruction:

    stall count, yield flag, write dependence-barrier index, read
    dependence-barrier index, wait-barrier mask, register reuse flags

The bit layout follows the community-documented Volta/Turing/Ampere
control encoding (control = high_word >> 41):

    bits [0:3]   stall count          (0..15)
    bit  [4]     yield flag
    bits [5:7]   write barrier index  (7 = none)
    bits [8:10]  read barrier index   (7 = none)
    bits [11:16] wait barrier mask    (6 bits, one per barrier)
    bits [17:20] reuse flags          (4 source-operand slots)

Fail-closed contract (the AC's negative arm: a parse may NEVER
silently drop unknown control fields):

- only arch sections in ``PARSER_CONTRACT['supported_arches']`` are
  decoded; other arches are skipped as whole sections (recorded);
- a toolkit outside ``supported_toolkit_versions`` raises typed
  ``UnsupportedToolkitError`` before any parse;
- an instruction whose control line is missing, malformed, or whose
  decoded fields leave their domains yields an ``unknown_encoding``
  record carrying the raw words — present in the output, never
  dropped;
- control bits BEYOND the documented layout (high control bits
  [21:22] of the 23-bit field) must be zero; a nonzero residue types
  the instruction ``unknown_encoding`` (an undocumented field is not
  silently ignored).
"""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# LOCKSTEP: sim/sm/src/control_bits.rs EXPECTED_LAYOUT mirrors control_layout
# string-for-string and its ControlBits::decode mirrors decode_control
# bit-for-bit; parity test: tools/sass_control/test_control_layout_rust_parity.py.
# Update all surfaces in the same commit.
PARSER_CONTRACT = {
    "supported_toolkit_versions": ["V13.2"],
    # sm_80 and sm_86 share the Ampere 128-bit scheduling-control layout.
    # The sm_80 entry is required for forward-compatible vendor-library code
    # selected by an sm_86 runtime.  Producers must still record the execution
    # arch and the code arch separately; accepting sm_80 here never relabels
    # those bytes as sm_86.
    "supported_arches": ["sm_80", "sm_86"],
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
}

_FUNC_RE = re.compile(r"\s*Function : (\S+)")
_ARCH_RE = re.compile(r"\s*code for (\S+)")
_INST_RE = re.compile(
    r"\s*/\*([0-9a-fA-F]{4,})\*/\s+(@!?\S+\s+)?([A-Z@][^;]*);\s*"
    r"/\*\s*0x([0-9a-fA-F]{16})\s*\*/")
_CTRL_RE = re.compile(r"\s*/\*\s*0x([0-9a-fA-F]{16})\s*\*/\s*$")
_VERSION_RE = re.compile(r"Cuda compilation tools, release [^,]+, "
                         r"(V[0-9]+\.[0-9]+)")


class UnsupportedToolkitError(RuntimeError):
    typed_reason = "unsupported_toolkit_version"


class SilentDropError(RuntimeError):
    """Internal invariant: every instruction line must produce exactly
    one output record (decoded or unknown_encoding)."""
    typed_reason = "control_field_silently_dropped"


class FunctionTextExtractionError(RuntimeError):
    """Fail-closed exact-function extraction from a CUDA fat binary."""

    typed_reason = "function_text_extraction_failed"


def assert_toolkit_supported(nvcc: str = "nvcc") -> str:
    proc = subprocess.run([nvcc, "--version"], capture_output=True,
                          text=True, timeout=60)
    m = _VERSION_RE.search(proc.stdout)
    version_prefix = None
    if m:
        version_prefix = m.group(1).split(".")[0] + "." + \
            m.group(1).split(".")[1].split(".")[0]
        version_prefix = m.group(1)
    for supported in PARSER_CONTRACT["supported_toolkit_versions"]:
        if version_prefix and version_prefix.startswith(supported):
            return version_prefix
    raise UnsupportedToolkitError(
        f"toolkit {version_prefix!r} is not in the pinned supported set "
        f"{PARSER_CONTRACT['supported_toolkit_versions']}; the control "
        "encoding is toolkit-verified per version (fail-closed)"
    )


def decode_control(high_word: int) -> dict:
    """Decode the 23-bit control field; raises ValueError when the
    reserved residue is nonzero (caller types unknown_encoding)."""
    c = (high_word >> 41) & 0x7FFFFF
    reserved = (c >> 21) & 0x3
    if reserved != 0:
        raise ValueError(
            f"reserved control bits nonzero (0x{reserved:x}): "
            "undocumented field present")
    decoded = {
        "stall": c & 0xF,
        "yield_flag": (c >> 4) & 0x1,
        "write_barrier": (c >> 5) & 0x7,
        "read_barrier": (c >> 8) & 0x7,
        "wait_mask": (c >> 11) & 0x3F,
        "reuse": (c >> 17) & 0xF,
    }
    # Ampere exposes six dependence barriers, SB0..SB5; wire value 7 means
    # "none".  Value 6 is invalid.  The Rust consumer has always rejected it,
    # so the producer must reject it too instead of emitting a sidecar the
    # consumer will later refuse.
    if decoded["write_barrier"] == 6 or decoded["read_barrier"] == 6:
        raise ValueError("dependence barrier index 6 is invalid (SB0..SB5, 7=none)")
    return decoded


_TEXT_SECTION_RE = re.compile(
    r"SASS text section ([1-9][0-9]*) : "
    r"(x-(.+)\.(sm_[0-9]+[a-z]?)\.elf\.bin)\Z"
)
_ELF_FILE_RE = re.compile(
    r"ELF file\s+([1-9][0-9]*): ([A-Za-z0-9_.-]+\.(sm_[0-9]+[a-z]?)\.cubin)\Z"
)


def _bounded_slice(payload: bytes, offset: int, size: int, role: str) -> bytes:
    if (
        type(offset) is not int
        or type(size) is not int
        or offset < 0
        or size < 0
        or offset > len(payload)
        or size > len(payload) - offset
    ):
        raise FunctionTextExtractionError(
            f"invalid {role} byte range offset={offset} size={size} "
            f"payload={len(payload)}"
        )
    return payload[offset:offset + size]


def _elf_string(table: bytes, offset: int, role: str) -> str:
    if type(offset) is not int or offset < 0 or offset >= len(table):
        raise FunctionTextExtractionError(
            f"invalid {role} string offset {offset} for table size {len(table)}"
        )
    end = table.find(b"\0", offset)
    if end < 0:
        raise FunctionTextExtractionError(f"unterminated {role} string at {offset}")
    try:
        value = table[offset:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise FunctionTextExtractionError(
            f"non-ASCII {role} string at {offset}"
        ) from exc
    if not value:
        raise FunctionTextExtractionError(f"empty {role} string at {offset}")
    return value


def _exact_cuda_elf_function(
    payload: bytes, function_name: str,
) -> dict[str, Any] | None:
    """Return one exact whole-section CUDA ELF function payload.

    CUDA 13.2 abbreviates very long ``--list-text`` names (for example a
    CUTLASS template symbol) even though the extracted cubin retains the full
    symbol and section names.  This parser intentionally supports only the
    narrow ELF64 little-endian CUDA shape needed to prove that one exact
    STT_FUNC owns one complete executable text section.  Anything else fails
    closed instead of becoming a heuristic name match.
    """
    if len(payload) < 64:
        raise FunctionTextExtractionError("extracted cubin is shorter than ELF64 header")
    ident = payload[:16]
    if ident[:4] != b"\x7fELF" or ident[4:7] != bytes((2, 1, 1)):
        raise FunctionTextExtractionError("extracted cubin is not ELF64 little-endian")
    try:
        (
            _ident, elf_type, machine, version, _entry, _phoff, section_offset,
            _flags, header_size, _phent_size, _phnum, section_entry_size,
            section_count, section_name_index,
        ) = struct.unpack_from("<16sHHIQQQIHHHHHH", payload, 0)
    except struct.error as exc:
        raise FunctionTextExtractionError("malformed extracted cubin ELF header") from exc
    if (
        elf_type != 2
        or machine != 190
        or version != 1
        or header_size != 64
        or section_entry_size != 64
        or section_count < 5
        or section_name_index <= 0
        or section_name_index >= section_count
    ):
        raise FunctionTextExtractionError(
            "unsupported extracted cubin ELF contract: "
            f"type={elf_type} machine={machine} version={version} "
            f"ehsize={header_size} shentsize={section_entry_size} "
            f"shnum={section_count} shstrndx={section_name_index}"
        )
    section_table = _bounded_slice(
        payload, section_offset, section_count * section_entry_size,
        "ELF section table",
    )
    sections: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for index in range(section_count):
        try:
            sections.append(struct.unpack_from("<IIQQQQIIQQ", section_table, index * 64))
        except struct.error as exc:
            raise FunctionTextExtractionError(
                f"malformed ELF section header {index}"
            ) from exc
    shstr = sections[section_name_index]
    if shstr[1] != 3:
        raise FunctionTextExtractionError("ELF section-name table is not STRTAB")
    section_names = _bounded_slice(payload, shstr[4], shstr[5], "section-name table")

    candidates: list[dict[str, Any]] = []
    for symtab_index, symtab in enumerate(sections):
        if symtab[1] != 2:
            continue
        if (
            symtab[6] <= 0
            or symtab[6] >= section_count
            or symtab[9] != 24
            or symtab[5] == 0
            or symtab[5] % symtab[9] != 0
        ):
            raise FunctionTextExtractionError(
                f"invalid CUDA ELF symbol table section {symtab_index}"
            )
        strings_header = sections[symtab[6]]
        if strings_header[1] != 3:
            raise FunctionTextExtractionError(
                f"symbol table {symtab_index} does not link to STRTAB"
            )
        strings = _bounded_slice(
            payload, strings_header[4], strings_header[5],
            f"symbol strings for section {symtab_index}",
        )
        symbols = _bounded_slice(
            payload, symtab[4], symtab[5], f"symbol table {symtab_index}",
        )
        for symbol_index in range(1, len(symbols) // 24):
            try:
                name_offset, info, _other, owner_index, value, size = struct.unpack_from(
                    "<IBBHQQ", symbols, symbol_index * 24,
                )
            except struct.error as exc:
                raise FunctionTextExtractionError(
                    f"malformed CUDA ELF symbol {symbol_index}"
                ) from exc
            if name_offset == 0:
                continue
            name = _elf_string(strings, name_offset, "symbol")
            if name != function_name:
                continue
            if (info & 0xF) != 2 or owner_index <= 0 or owner_index >= section_count:
                raise FunctionTextExtractionError(
                    f"exact symbol {function_name!r} is not a defined STT_FUNC"
                )
            owner = sections[owner_index]
            owner_name = _elf_string(section_names, owner[0], "section")
            if (
                owner[1] != 1
                or owner[2] & 0x4 == 0
                or owner_name != f".text.{function_name}"
                or value != owner[3]
                or size != owner[5]
                or size <= 0
            ):
                raise FunctionTextExtractionError(
                    f"exact symbol {function_name!r} does not own one complete "
                    "executable text section"
                )
            candidates.append({
                "symbol_index": symbol_index,
                "symbol_table_index": symtab_index,
                "text_section_index": owner_index,
                "text_section_name": owner_name,
                "raw_text_bytes": _bounded_slice(
                    payload, owner[4], owner[5], "exact CUDA text section",
                ),
            })
    if not candidates:
        return None
    if len(candidates) != 1:
        raise FunctionTextExtractionError(
            f"function {function_name!r} has {len(candidates)} exact cubin symbols"
        )
    return candidates[0]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _extract_catalog_text(
    binary: Path, *, section_index: int, section_name: str,
    cuobjdump: str, timeout_s: float,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="sass-text-") as temp_raw:
        temp = Path(temp_raw)
        temp.chmod(0o700)
        try:
            extracted = subprocess.run(
                [cuobjdump, "--extract-text", section_name, str(binary)],
                cwd=temp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FunctionTextExtractionError(
                f"cuobjdump extraction failed: {exc}"
            ) from exc
        expected_stdout = (
            f"Extracting SASS text section {section_index} : {section_name}\n"
        )
        if (
            extracted.returncode != 0
            or extracted.stderr
            or extracted.stdout != expected_stdout
        ):
            raise FunctionTextExtractionError(
                "cuobjdump extraction was not exact: "
                f"rc={extracted.returncode}, stdout={extracted.stdout[-300:]!r}, "
                f"stderr={extracted.stderr[-300:]!r}"
            )
        entries = list(temp.iterdir())
        if len(entries) != 1 or entries[0].name != section_name:
            raise FunctionTextExtractionError(
                f"unexpected extraction outputs {[entry.name for entry in entries]!r}"
            )
        raw_path = entries[0]
        if raw_path.is_symlink() or not raw_path.is_file():
            raise FunctionTextExtractionError(
                f"extracted section is not a regular file: {raw_path}"
            )
        return raw_path.read_bytes()


def _recover_abbreviated_catalog_text(
    binary: Path, *, function_name: str,
    catalog: list[tuple[int, str, str, str]],
    allowed_code_arches: tuple[str, ...], cuobjdump: str, timeout_s: float,
) -> tuple[tuple[int, str, str, str], bytes, dict[str, Any]]:
    """Cross-bind an abbreviated list-text row to one exact cubin symbol."""
    try:
        listed = subprocess.run(
            [cuobjdump, "--list-elf", str(binary)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FunctionTextExtractionError(f"cuobjdump ELF list failed: {exc}") from exc
    if listed.returncode != 0 or listed.stderr:
        raise FunctionTextExtractionError(
            "cuobjdump ELF list was not clean: "
            f"rc={listed.returncode}, stderr={listed.stderr[-300:]!r}"
        )
    elf_catalog: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(listed.stdout.splitlines(), start=1):
        match = _ELF_FILE_RE.fullmatch(line)
        if match is None:
            raise FunctionTextExtractionError(
                f"malformed ELF catalog line {line_no}: {line!r}"
            )
        elf_catalog.append((int(match.group(1)), match.group(2), match.group(3)))
    if (
        not elf_catalog
        or [row[0] for row in elf_catalog] != list(range(1, len(elf_catalog) + 1))
        or len({row[1] for row in elf_catalog}) != len(elf_catalog)
    ):
        raise FunctionTextExtractionError("ELF catalog is empty, duplicate, or unordered")

    symbol_candidates: list[dict[str, Any]] = []
    for elf_index, elf_name, code_arch in elf_catalog:
        if Path(elf_name).name != elf_name:
            raise FunctionTextExtractionError(f"unsafe cubin filename {elf_name!r}")
        with tempfile.TemporaryDirectory(prefix="sass-cubin-") as temp_raw:
            temp = Path(temp_raw)
            temp.chmod(0o700)
            try:
                extracted = subprocess.run(
                    [cuobjdump, "--extract-elf", elf_name, str(binary)],
                    cwd=temp,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise FunctionTextExtractionError(
                    f"cuobjdump ELF extraction failed: {exc}"
                ) from exc
            expected = re.compile(
                rf"Extracting ELF file\s+{elf_index}: {re.escape(elf_name)}\n\Z"
            )
            if (
                extracted.returncode != 0
                or extracted.stderr
                or expected.fullmatch(extracted.stdout) is None
            ):
                raise FunctionTextExtractionError(
                    "cuobjdump ELF extraction was not exact: "
                    f"rc={extracted.returncode}, stdout={extracted.stdout[-300:]!r}, "
                    f"stderr={extracted.stderr[-300:]!r}"
                )
            entries = list(temp.iterdir())
            if len(entries) != 1 or entries[0].name != elf_name:
                raise FunctionTextExtractionError(
                    "unexpected ELF extraction outputs "
                    f"{[entry.name for entry in entries]!r}"
                )
            cubin = entries[0]
            if cubin.is_symlink() or not cubin.is_file():
                raise FunctionTextExtractionError(
                    f"extracted cubin is not a regular file: {cubin}"
                )
            cubin_bytes = cubin.read_bytes()
        exact = _exact_cuda_elf_function(cubin_bytes, function_name)
        if exact is not None:
            symbol_candidates.append({
                **exact,
                "code_arch": code_arch,
                "cubin_catalog_index": elf_index,
                "cubin_file_name": elf_name,
                "cubin_sha256": _sha256_bytes(cubin_bytes),
            })
    if len(symbol_candidates) != 1:
        raise FunctionTextExtractionError(
            f"function {function_name!r} has {len(symbol_candidates)} "
            "exact cubin symbol candidates"
        )
    proof = symbol_candidates[0]
    if proof["code_arch"] not in allowed_code_arches:
        raise FunctionTextExtractionError(
            f"function {function_name!r} code arch {proof['code_arch']!r} "
            f"is not in {allowed_code_arches!r}"
        )
    exact_raw = proof.pop("raw_text_bytes")
    raw_matches: list[tuple[tuple[int, str, str, str], bytes]] = []
    for row in catalog:
        section_index, section_name, _catalog_name, code_arch = row
        if code_arch != proof["code_arch"]:
            continue
        raw = _extract_catalog_text(
            binary, section_index=section_index, section_name=section_name,
            cuobjdump=cuobjdump, timeout_s=timeout_s,
        )
        if raw == exact_raw:
            raw_matches.append((row, raw))
    if len(raw_matches) != 1:
        raise FunctionTextExtractionError(
            f"function {function_name!r} cubin section has {len(raw_matches)} "
            "exact list-text payload matches"
        )
    proof["catalog_raw_match_count"] = 1
    return raw_matches[0][0], raw_matches[0][1], proof


def extract_exact_function_text(
    binary: Path,
    *,
    function_name: str,
    allowed_code_arches: tuple[str, ...],
    nvcc: str,
    cuobjdump: str,
    timeout_s: float = 300,
) -> dict[str, Any]:
    """Extract and decode one exact raw SASS text section.

    ``cuobjdump --function --dump-sass`` is not a reliable selector for large
    vendor fatbins: it may return success while printing thousands of
    not-found warnings and unrelated sections.  This routine instead lists
    the complete text catalog, selects one exact filename in-process, then
    asks ``--extract-text`` for that full filename inside an empty private
    temporary directory.  Every intermediate surface is closed and checked.
    """
    if (
        not isinstance(function_name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name) is None
    ):
        raise FunctionTextExtractionError(
            f"unsafe or unsupported function name {function_name!r}"
        )
    if (
        not isinstance(allowed_code_arches, tuple)
        or not allowed_code_arches
        or len(set(allowed_code_arches)) != len(allowed_code_arches)
        or any(arch not in PARSER_CONTRACT["supported_arches"]
               for arch in allowed_code_arches)
    ):
        raise FunctionTextExtractionError(
            f"invalid allowed code arches {allowed_code_arches!r}"
        )
    if binary.is_symlink() or not binary.is_file():
        raise FunctionTextExtractionError(f"binary is not a regular file: {binary}")

    try:
        toolkit_version = assert_toolkit_supported(nvcc)
    except (OSError, subprocess.TimeoutExpired, UnsupportedToolkitError) as exc:
        raise FunctionTextExtractionError(
            f"pinned nvcc toolkit probe failed: {exc}"
        ) from exc
    try:
        listed = subprocess.run(
            [cuobjdump, "--list-text", str(binary)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FunctionTextExtractionError(f"cuobjdump list failed: {exc}") from exc
    if listed.returncode != 0 or listed.stderr:
        raise FunctionTextExtractionError(
            "cuobjdump list was not clean: "
            f"rc={listed.returncode}, stderr={listed.stderr[-300:]!r}"
        )

    catalog: list[tuple[int, str, str, str]] = []
    for line_no, line in enumerate(listed.stdout.splitlines(), start=1):
        match = _TEXT_SECTION_RE.fullmatch(line)
        if match is None:
            raise FunctionTextExtractionError(
                f"malformed text catalog line {line_no}: {line!r}"
            )
        catalog.append(
            (int(match.group(1)), match.group(2), match.group(3), match.group(4))
        )
    matches = [row for row in catalog if row[2] == function_name]
    if len(matches) > 1:
        raise FunctionTextExtractionError(
            f"function {function_name!r} has {len(matches)} exact text sections"
        )
    section_identity_method = "list_text_exact_name"
    cubin_symbol_proof: dict[str, Any] | None = None
    if matches:
        selected = matches[0]
        raw = _extract_catalog_text(
            binary, section_index=selected[0], section_name=selected[1],
            cuobjdump=cuobjdump, timeout_s=timeout_s,
        )
    else:
        selected, raw, cubin_symbol_proof = _recover_abbreviated_catalog_text(
            binary, function_name=function_name, catalog=catalog,
            allowed_code_arches=allowed_code_arches, cuobjdump=cuobjdump,
            timeout_s=timeout_s,
        )
        section_identity_method = "cubin_symbol_raw_match"
    section_index, section_name, _name, code_arch = selected
    if code_arch not in allowed_code_arches:
        raise FunctionTextExtractionError(
            f"function {function_name!r} code arch {code_arch!r} is not in "
            f"{allowed_code_arches!r}"
        )
    if Path(section_name).name != section_name:
        raise FunctionTextExtractionError(f"unsafe section name {section_name!r}")

    if not raw or len(raw) % 16 != 0:
        raise FunctionTextExtractionError(
            f"raw text size {len(raw)} is not a nonzero multiple of 16"
        )
    records: list[dict[str, Any]] = []
    for offset in range(0, len(raw), 16):
        low_word, high_word = struct.unpack_from("<QQ", raw, offset)
        try:
            control = decode_control(high_word)
        except ValueError as exc:
            raise FunctionTextExtractionError(
                f"{function_name} pc=0x{offset:x}: {exc}"
            ) from exc
        records.append({
            "pc": offset,
            "low_word": f"0x{low_word:016x}",
            "high_word": f"0x{high_word:016x}",
            "control": control,
            "status": "decoded",
        })
    return {
        "function_name": function_name,
        "code_arch": code_arch,
        "section_index": section_index,
        "section_name": section_name,
        "section_identity_method": section_identity_method,
        "cubin_symbol_proof": cubin_symbol_proof,
        "raw_text_size_bytes": len(raw),
        "raw_text_sha256": _sha256_bytes(raw),
        "raw_text_bytes": raw,
        "records": records,
        "toolkit_version": toolkit_version,
        "parser_contract": PARSER_CONTRACT,
    }


def parse_sass_text(text: str, arch: str = "sm_86") -> dict:
    """Parse cuobjdump SASS text into per-function instruction records
    for the requested arch ONLY. Every instruction line in a supported
    section produces exactly one record: ``status: decoded`` with the
    control fields, or ``status: unknown_encoding`` with the raw words
    and a reason. Unsupported arch sections are recorded (names only),
    never silently decoded."""
    if arch not in PARSER_CONTRACT["supported_arches"]:
        raise UnsupportedToolkitError(
            f"arch {arch!r} not in the pinned supported set")
    functions: dict[str, list[dict]] = {}
    skipped_arches: set[str] = set()
    current: list[dict] | None = None
    current_arch: str | None = None
    pending: dict | None = None

    def _flush_pending(reason: str) -> None:
        nonlocal pending
        if pending is not None:
            pending["status"] = "unknown_encoding"
            pending["reason"] = reason
            current.append(pending)  # type: ignore[union-attr]
            pending = None

    for line in text.splitlines():
        am = _ARCH_RE.match(line)
        if am:
            _flush_pending("control_line_missing_at_section_end")
            current_arch = am.group(1)
            if current_arch != arch:
                skipped_arches.add(current_arch)
            current = None
            continue
        fm = _FUNC_RE.search(line)
        if fm:
            _flush_pending("control_line_missing_at_function_end")
            if current_arch == arch:
                current = functions.setdefault(fm.group(1), [])
            else:
                current = None
            continue
        if current is None:
            continue
        im = _INST_RE.match(line)
        if im:
            _flush_pending("control_line_missing")
            pending = {
                "pc": int(im.group(1), 16),
                "predicate": (im.group(2) or "").strip() or None,
                "opcode_text": im.group(3).strip(),
                "low_word": f"0x{im.group(4)}",
            }
            continue
        cm = _CTRL_RE.match(line)
        if cm and pending is not None:
            high_word = int(cm.group(1), 16)
            pending["high_word"] = f"0x{cm.group(1)}"
            try:
                pending["control"] = decode_control(high_word)
                pending["status"] = "decoded"
            except ValueError as exc:
                pending["status"] = "unknown_encoding"
                pending["reason"] = str(exc)
            current.append(pending)
            pending = None
            continue
    _flush_pending("control_line_missing_at_eof")
    return {
        "arch": arch,
        "functions": functions,
        "skipped_arch_sections": sorted(skipped_arches),
        "parser_contract": PARSER_CONTRACT,
    }


def extract_for_binary(binary: Path, arch: str = "sm_86",
                       nvcc: str = "nvcc") -> dict:
    """cuobjdump + parse, with the toolkit gate enforced first."""
    toolkit = assert_toolkit_supported(nvcc)
    proc = subprocess.run(
        ["cuobjdump", "--dump-sass", str(binary)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {proc.stderr[-300:]}")
    out = parse_sass_text(proc.stdout, arch=arch)
    out["toolkit_version"] = toolkit
    out["binary"] = str(binary)
    # Completeness invariant: every record is decoded or typed.
    for name, records in out["functions"].items():
        for rec in records:
            if rec.get("status") not in ("decoded", "unknown_encoding"):
                raise SilentDropError(
                    f"{name} pc=0x{rec.get('pc', -1):x} has no status")
    return out
