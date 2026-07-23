"""Compile-path manifest inspector for the RTX PRO 6000 Blackwell
workstation target under the non-root measurement contract.

For every kernel that enters the active measurement corpus we need
to record `{kernel_name, compiler_version, sm_target, cubin_sha256,
tensor_instruction_family_observed, cluster_dim_observed}` and
refuse admission on any kernel whose SASS carries sm_100-only
opcodes (`tcgen05.*`, `UTC*MMA`, `LDT`, `STT`, `TMEM_*`) or whose
`sm_target` is outside the `sm_120*` family. This module produces
those records by shelling out once to `cuobjdump` per cubin and
parsing its text output.

Per-kernel scope. The `cuobjdump` output for a multi-kernel cubin
interleaves each kernel's SASS block behind a
`\\t\\tFunction : <mangled>` line and each kernel's PTX block
behind a `.visible .entry <mangled>(` / `.entry <mangled>(` line.
`inspect_kernel(..., kernel_name=...)` slices both dumps down to
the named kernel BEFORE running forbidden-mnemonic, tensor-family,
and cluster-dim scans, so a kernel whose SASS contains HMMA does
not contaminate a sibling kernel's record.

Tensor-family enum is frozen at the plan's declared set
`{hmma, qmma, omma, mixed, none}`. Observing a different tensor
mnemonic (e.g., `BMMA`, `IMMA`) raises
`TensorFamilyOutsideEnum`; silent widening of the contract is
forbidden. `write_compile_path_manifest` re-validates the enum at
serialization time so a drifting record cannot reach the
downstream admission gate.

Public surface:

    from compile_path_inspector import (
        KernelCompilePath,
        CompilePathErrorKind,
        CompilePathError,
        inspect_kernel,
        write_compile_path_manifest,
    )

The pure-text parser (`_parse_cuobjdump_output`) is factored out
so unit tests can drive it with fixture strings rather than
shelling out to `cuobjdump`. `inspect_kernel` wraps it with the
shell-out plus a cubin sha256.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


VALID_TENSOR_FAMILIES: frozenset[str] = frozenset(
    {"hmma", "qmma", "omma", "mixed", "none"}
)

# Opcodes that must NEVER appear in a sm_120 SASS dump. `tcgen05` is
# the sm_100 5th-gen tensor-core path; `UTCMMA`, `LDT`, `STT`, and
# `TMEM_*` are its data-movement + storage siblings. Any hit is a
# hard admission fail per the sm_120 compile-path contract.
_FORBIDDEN_MNEMONIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btcgen05\.[A-Za-z0-9_.]+", re.IGNORECASE),
    re.compile(r"\bUTC[A-Z]*MMA\b"),
    re.compile(r"\bLDT\b"),
    re.compile(r"\bSTT\b"),
    re.compile(r"\bTMEM_[A-Z0-9_]+\b"),
)

# SASS tensor-core instruction mnemonic detection. Matched as a
# whole word at the start of a decoded SASS line (after the
# `/* offset */` prefix). The three families below correspond
# directly to the plan's declared `tensor_instruction_family_observed`
# enum `{hmma, qmma, omma, mixed, none}`. Families that are valid
# sm_120 instructions but NOT in the plan's enum (currently BMMA
# and IMMA) are detected separately in `_UNMAPPED_TENSOR_PATTERNS`
# and raise `TensorFamilyOutsideEnum` so the contract cannot be
# silently widened.
_TENSOR_FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "hmma": re.compile(r"\bHMMA\.", re.IGNORECASE),
    "qmma": re.compile(r"\bQMMA\.", re.IGNORECASE),
    "omma": re.compile(r"\bOMMA\.", re.IGNORECASE),
}

# SASS tensor-core mnemonics that are valid sm_120 instructions
# but are outside the active tensor-family enum. Observing any of these
# is a hard admission fail: either the kernel needs re-analysis
# or the plan's enum needs widening.
_UNMAPPED_TENSOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "BMMA": re.compile(r"\bBMMA\.", re.IGNORECASE),
    "IMMA": re.compile(r"\bIMMA\.", re.IGNORECASE),
}

# `arch = sm_120a` / `arch = sm_120` in the ELF dump header.
_SM_TARGET_PATTERN = re.compile(r"^\s*arch\s*=\s*(sm_[A-Za-z0-9]+)\s*$", re.MULTILINE)
# `Tool Version: Cuda compilation tools, release 13.2, V13.2.78`.
_COMPILER_VERSION_PATTERN = re.compile(
    r"Tool Version:\s*Cuda compilation tools,\s*release\s*([0-9]+\.[0-9]+)"
    r"(?:,\s*(V[0-9.]+))?"
)
# `.target sm_120a` inside the SASS dump (alternative anchor).
_SASS_TARGET_PATTERN = re.compile(r"\.target\s+(sm_[A-Za-z0-9]+)")
# Static cluster-dim annotation in PTX or per-kernel metadata. The
# PTX directive form is `.reqnctapercluster.x 4` / `.y 2` / `.z 1`
# or the flattened `.reqnctapercluster 4,2,1`.
_CLUSTER_DIM_X = re.compile(r"\.reqnctapercluster\.x\s+(\d+)")
_CLUSTER_DIM_Y = re.compile(r"\.reqnctapercluster\.y\s+(\d+)")
_CLUSTER_DIM_Z = re.compile(r"\.reqnctapercluster\.z\s+(\d+)")
_CLUSTER_DIM_FLAT = re.compile(
    r"\.reqnctapercluster\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
)

# Per-kernel section delimiters. In `cuobjdump --dump-sass` each
# kernel's disassembly starts with `\t\tFunction : <mangled>` and
# runs until the next `Function :` line (or EOF). In
# `cuobjdump --dump-ptx` each kernel's PTX starts with
# `.visible .entry <mangled>(` or `.entry <mangled>(` and runs
# until the next `.entry` / `.func` header (or EOF).
_SASS_FUNCTION_HEADER = re.compile(r"^\s*Function\s*:\s*(\S+)", re.MULTILINE)
_PTX_ENTRY_HEADER = re.compile(
    r"^\s*(?:\.visible\s+)?\.(?:entry|func)\s+(\S+?)\s*\(", re.MULTILINE
)


class CompilePathErrorKind(enum.Enum):
    UnexpectedTcgen05OnSm120 = "UnexpectedTcgen05OnSm120"
    CompilePathSmTargetMismatch = "CompilePathSmTargetMismatch"
    TensorFamilyOutsideEnum = "TensorFamilyOutsideEnum"
    KernelNotFoundInCubin = "KernelNotFoundInCubin"
    KernelSelectorAmbiguous = "KernelSelectorAmbiguous"


class CompilePathError(Exception):
    def __init__(self, kind: CompilePathErrorKind, message: str) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.message = message


@dataclasses.dataclass(frozen=True)
class KernelCompilePath:
    """One row of `rtx_pro_6000_blackwell.json`'s `kernels` array."""

    kernel_name: str
    compiler_version: str
    sm_target: str
    cubin_sha256: str
    tensor_instruction_family_observed: str
    cluster_dim_observed: list[int] | str

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "compiler_version": self.compiler_version,
            "sm_target": self.sm_target,
            "cubin_sha256": self.cubin_sha256,
            "tensor_instruction_family_observed": (
                self.tensor_instruction_family_observed
            ),
            "cluster_dim_observed": self.cluster_dim_observed,
        }


def _extract_simple_name(mangled: str) -> str | None:
    """Best-effort extract the simple function name from an Itanium
    C++-mangled symbol. Returns `None` if the parse fails.

    Handles the two forms the CUDA toolchain emits:
      * `_Z<len><name><params>` — non-nested kernels (most common).
      * `_ZN<len1><scope1>...<lenN><nameN>E<params>` — nested
        kernels inside a namespace or class (returns the last
        component before the `E` terminator).
    """
    if not mangled.startswith("_Z"):
        return mangled
    rest = mangled[2:]
    if rest.startswith("N"):
        rest = rest[1:]
        last = None
        while rest and rest[0].isdigit():
            n_str = ""
            while rest and rest[0].isdigit():
                n_str += rest[0]
                rest = rest[1:]
            try:
                n = int(n_str)
            except ValueError:
                return None
            if len(rest) < n:
                return None
            last = rest[:n]
            rest = rest[n:]
            if rest.startswith("E"):
                return last
        return last
    n_str = ""
    while rest and rest[0].isdigit():
        n_str += rest[0]
        rest = rest[1:]
    if not n_str:
        return None
    try:
        n = int(n_str)
    except ValueError:
        return None
    if len(rest) < n:
        return None
    return rest[:n]


def _match_kernel_header(mangled: str, kernel_name: str) -> bool:
    """Return True iff `kernel_name` identifies this function.

    Exact mangled match wins. Otherwise compare against the simple
    (demangled) name.
    """
    if mangled == kernel_name:
        return True
    simple = _extract_simple_name(mangled)
    return simple is not None and simple == kernel_name


def _select_kernel_section(
    text: str, header_pattern: re.Pattern[str], kernel_name: str, *, dump_kind: str
) -> str:
    """Return the substring of `text` that belongs to `kernel_name`.

    If `text` contains no function / entry headers at all (common in
    synthetic fixture inputs), the whole text is treated as one
    kernel's section. Real `cuobjdump` output always carries at
    least one header, so the no-header fallback applies only to
    fixture paths.

    Raises `KernelNotFoundInCubin` if headers exist but none match.
    Raises `KernelSelectorAmbiguous` if multiple distinct headers
    match the same `kernel_name`.
    """
    matches = list(header_pattern.finditer(text))
    if not matches:
        return text

    hits = [m for m in matches if _match_kernel_header(m.group(1), kernel_name)]
    if not hits:
        observed = sorted({m.group(1) for m in matches})
        raise CompilePathError(
            CompilePathErrorKind.KernelNotFoundInCubin,
            f"kernel_name {kernel_name!r} matched no {dump_kind} header "
            f"(observed: {observed[:8]}"
            f"{' ...' if len(observed) > 8 else ''})",
        )
    distinct = sorted({m.group(1) for m in hits})
    if len(distinct) > 1:
        raise CompilePathError(
            CompilePathErrorKind.KernelSelectorAmbiguous,
            f"kernel_name {kernel_name!r} matched multiple {dump_kind} "
            f"headers: {distinct}",
        )

    target = hits[0]
    start = target.start()
    for m in matches:
        if m.start() > start:
            return text[start:m.start()]
    return text[start:]


def _select_kernel_ptx_section(ptx_text: str, kernel_name: str) -> str:
    """Slice PTX to the named kernel's `.entry` block.

    PTX is best-effort: if the cubin carries only SASS,
    `inspect_kernel` passes `ptx_text=""` and we return `""`. If
    PTX exists but does not declare the kernel (rare — typically
    means the kernel was device-linked elsewhere), we also return
    `""` rather than raising, so a valid SASS-driven record can
    still be built. Ambiguous matches still raise.
    """
    if not ptx_text.strip():
        return ""
    matches = list(_PTX_ENTRY_HEADER.finditer(ptx_text))
    if not matches:
        return ptx_text
    hits = [m for m in matches if _match_kernel_header(m.group(1), kernel_name)]
    if not hits:
        return ""
    distinct = sorted({m.group(1) for m in hits})
    if len(distinct) > 1:
        raise CompilePathError(
            CompilePathErrorKind.KernelSelectorAmbiguous,
            f"kernel_name {kernel_name!r} matched multiple PTX .entry "
            f"headers: {distinct}",
        )
    target = hits[0]
    start = target.start()
    for m in matches:
        if m.start() > start:
            return ptx_text[start:m.start()]
    return ptx_text[start:]


def _parse_sm_target(elf_text: str, sass_text: str) -> str | None:
    # Prefer the ELF-header `arch = sm_XXX` line. Fall back to the
    # SASS `.target sm_XXX` anchor when the ELF dump is absent
    # (fixture fallback — real `cuobjdump` output always carries
    # both).
    match = _SM_TARGET_PATTERN.search(elf_text)
    if match:
        return match.group(1)
    match = _SASS_TARGET_PATTERN.search(sass_text)
    if match:
        return match.group(1)
    return None


def _parse_compiler_version(elf_text: str) -> str:
    match = _COMPILER_VERSION_PATTERN.search(elf_text)
    if not match:
        return "unknown"
    release, build = match.group(1), match.group(2)
    return f"{release}/{build}" if build else release


def _detect_forbidden_mnemonics(sass_text: str) -> list[str]:
    hits: list[str] = []
    for pattern in _FORBIDDEN_MNEMONIC_PATTERNS:
        for m in pattern.finditer(sass_text):
            hits.append(m.group(0))
    seen: set[str] = set()
    unique: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def _detect_unmapped_tensor_mnemonics(sass_text: str) -> list[str]:
    """Return the sorted unique list of tensor-family mnemonics that
    appear in `sass_text` but are outside the active tensor-family enum
    `{hmma, qmma, omma, mixed, none}`. Currently `BMMA` and
    `IMMA`.
    """
    hits: set[str] = set()
    for label, pattern in _UNMAPPED_TENSOR_PATTERNS.items():
        if pattern.search(sass_text):
            hits.add(label)
    return sorted(hits)


def _parse_tensor_family(sass_text: str) -> str:
    families_seen: list[str] = []
    for family, pattern in _TENSOR_FAMILY_PATTERNS.items():
        if pattern.search(sass_text):
            families_seen.append(family)
    if not families_seen:
        return "none"
    if len(families_seen) == 1:
        return families_seen[0]
    return "mixed"


def _parse_cluster_dim(sass_text: str, ptx_text: str) -> list[int] | str:
    combined = f"{ptx_text}\n{sass_text}"
    flat = _CLUSTER_DIM_FLAT.search(combined)
    if flat:
        return [int(flat.group(1)), int(flat.group(2)), int(flat.group(3))]
    mx = _CLUSTER_DIM_X.search(combined)
    my = _CLUSTER_DIM_Y.search(combined)
    mz = _CLUSTER_DIM_Z.search(combined)
    if mx or my or mz:
        return [
            int(mx.group(1)) if mx else 1,
            int(my.group(1)) if my else 1,
            int(mz.group(1)) if mz else 1,
        ]
    return "dynamic"


def _parse_cuobjdump_output(
    kernel_name: str,
    sass_text: str,
    elf_text: str,
    *,
    ptx_text: str = "",
    cubin_sha256: str = "",
    expected_arch: str = "sm_120",
) -> KernelCompilePath:
    """Pure-text parser used by `inspect_kernel` and by the unit
    tests. Does not shell out.

    Slices `sass_text` and `ptx_text` to the named kernel's section
    before running the forbidden-mnemonic, tensor-family, and
    cluster-dim scans so a multi-kernel cubin yields distinct
    records per kernel.

    Raises `CompilePathError` with:
      * `KernelNotFoundInCubin` / `KernelSelectorAmbiguous` when
        the kernel section cannot be uniquely resolved.
      * `UnexpectedTcgen05OnSm120` when the per-kernel SASS
        contains `tcgen05.*` / `UTC*MMA` / `LDT` / `STT` /
        `TMEM_*` (all sm_100-only).
      * `CompilePathSmTargetMismatch` when the resolved cubin
        `sm_target` is not in the `{expected_arch}*` family
        (default `sm_120*`).
      * `TensorFamilyOutsideEnum` when the per-kernel SASS
        contains a tensor-core mnemonic outside the plan's
        `{hmma, qmma, omma, mixed, none}` enum.

    Callers pass `cubin_sha256` separately so this function is
    purely text-in / record-out and deterministic for fixtures.
    """
    kernel_sass = _select_kernel_section(
        sass_text, _SASS_FUNCTION_HEADER, kernel_name, dump_kind="SASS"
    )
    kernel_ptx = _select_kernel_ptx_section(ptx_text, kernel_name)

    forbidden = _detect_forbidden_mnemonics(kernel_sass)
    if forbidden:
        raise CompilePathError(
            CompilePathErrorKind.UnexpectedTcgen05OnSm120,
            f"kernel {kernel_name!r} SASS contains sm_100-only mnemonics "
            f"that must never appear on sm_120: {forbidden[:5]}"
            f"{' (more follow)' if len(forbidden) > 5 else ''}",
        )

    sm_target = _parse_sm_target(elf_text, sass_text)
    if sm_target is None:
        raise CompilePathError(
            CompilePathErrorKind.CompilePathSmTargetMismatch,
            f"kernel {kernel_name!r}: no sm_target resolvable from cuobjdump "
            f"output (expected sm_target in the {expected_arch}* family)",
        )
    if not sm_target.startswith(expected_arch):
        raise CompilePathError(
            CompilePathErrorKind.CompilePathSmTargetMismatch,
            f"kernel {kernel_name!r}: sm_target {sm_target!r} is outside "
            f"the {expected_arch}* family",
        )

    unmapped = _detect_unmapped_tensor_mnemonics(kernel_sass)
    if unmapped:
        raise CompilePathError(
            CompilePathErrorKind.TensorFamilyOutsideEnum,
            f"kernel {kernel_name!r} SASS contains tensor mnemonics outside "
            f"the active tensor-family enum {sorted(VALID_TENSOR_FAMILIES)}: "
            f"{unmapped}. Either re-analyze the kernel or widen the plan's "
            f"enum before admitting it.",
        )

    compiler_version = _parse_compiler_version(elf_text)
    tensor_family = _parse_tensor_family(kernel_sass)
    cluster_dim = _parse_cluster_dim(kernel_sass, kernel_ptx)

    return KernelCompilePath(
        kernel_name=kernel_name,
        compiler_version=compiler_version,
        sm_target=sm_target,
        cubin_sha256=cubin_sha256,
        tensor_instruction_family_observed=tensor_family,
        cluster_dim_observed=cluster_dim,
    )


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_kernel(
    cubin_path: str | Path,
    kernel_name: str,
    *,
    cuobjdump: str = "cuobjdump",
    expected_arch: str = "sm_120",
) -> KernelCompilePath:
    """Run `cuobjdump` against `cubin_path` and return the
    per-kernel compile-path record for `kernel_name`.

    The SASS and PTX dumps are sliced down to the named kernel's
    section before any scan runs, so a multi-kernel cubin yields
    distinct records per kernel.

    Parameters
    ----------
    cubin_path
        Absolute or repo-relative path to a cubin / fatbin / ELF
        object file that `cuobjdump` understands.
    kernel_name
        Either the fully-mangled Itanium C++ symbol (e.g.
        `_Z22cluster_rmsnorm_kernelPKfPfi`) or the demangled
        simple name (e.g. `cluster_rmsnorm_kernel`). The selector
        accepts both forms.
    cuobjdump
        Override for the `cuobjdump` executable (tests point this
        at a stub shell script).
    expected_arch
        sm-family prefix required for success. Defaults to
        `sm_120` so workstation cubins built for `sm_120a` still
        pass.

    Raises
    ------
    CompilePathError
        On `KernelNotFoundInCubin`, `KernelSelectorAmbiguous`,
        `UnexpectedTcgen05OnSm120`, `CompilePathSmTargetMismatch`,
        or `TensorFamilyOutsideEnum`.
    FileNotFoundError
        If `cubin_path` does not exist.
    subprocess.CalledProcessError
        If `cuobjdump` exits non-zero.
    """
    cubin = Path(cubin_path)
    if not cubin.is_file():
        raise FileNotFoundError(f"cubin not found: {cubin}")

    sass = subprocess.run(
        [cuobjdump, "--dump-sass", str(cubin)],
        check=True, capture_output=True, text=True,
    ).stdout
    elf = subprocess.run(
        [cuobjdump, "--dump-elf", str(cubin)],
        check=True, capture_output=True, text=True,
    ).stdout
    # PTX dump is best-effort: cuobjdump rejects `--dump-ptx` on
    # fatbins that only carry SASS. A non-zero exit here is not
    # fatal for compile-path inspection.
    ptx_proc = subprocess.run(
        [cuobjdump, "--dump-ptx", str(cubin)],
        capture_output=True, text=True,
    )
    ptx = ptx_proc.stdout if ptx_proc.returncode == 0 else ""

    return _parse_cuobjdump_output(
        kernel_name,
        sass_text=sass,
        elf_text=elf,
        ptx_text=ptx,
        cubin_sha256=_sha256_file(cubin),
        expected_arch=expected_arch,
    )


def write_compile_path_manifest(
    kernels: Iterable[KernelCompilePath],
    out_path: str | Path,
    *,
    target_arch: str = "sm_120_gb202",
    compiler_version_default: str = "",
    notes: str = "",
) -> None:
    """Serialize a sequence of kernel records to a compile-path
    manifest such as `results/compile_paths/rtx_pro_6000_blackwell.json`.

    Every record's `tensor_instruction_family_observed` is
    validated against `VALID_TENSOR_FAMILIES`
    (`{hmma, qmma, omma, mixed, none}`).
    A drifted value raises `TensorFamilyOutsideEnum` instead of
    silently widening the downstream admission contract.

    The file carries a stable header (`manifest_version`,
    `target_arch`, `compiler_version_default`, `notes`) plus the
    `kernels` array. When `kernels` is empty the manifest is
    written as a contract anchor for downstream admission
    tooling — `task16` / `task17` / `task18` append to it once
    those kernels land.
    """
    records: list[dict] = []
    for k in kernels:
        family = k.tensor_instruction_family_observed
        if family not in VALID_TENSOR_FAMILIES:
            raise CompilePathError(
                CompilePathErrorKind.TensorFamilyOutsideEnum,
                f"kernel {k.kernel_name!r} has "
                f"tensor_instruction_family_observed={family!r}, which is "
                f"outside the active tensor-family enum "
                f"{sorted(VALID_TENSOR_FAMILIES)}",
            )
        records.append(k.to_dict())
    payload = {
        "manifest_version": 1,
        "target_arch": target_arch,
        "compiler_version_default": compiler_version_default,
        "notes": notes,
        "kernels": records,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


__all__ = [
    "CompilePathError",
    "CompilePathErrorKind",
    "KernelCompilePath",
    "VALID_TENSOR_FAMILIES",
    "inspect_kernel",
    "write_compile_path_manifest",
]
