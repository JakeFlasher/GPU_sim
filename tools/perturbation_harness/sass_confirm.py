"""SASS modifier confirmation for perturbation binaries.

Implements Section 3 of the task14 analysis artifact: invokes
`cuobjdump --dump-sass <binary>`, scopes parsing to the target kernel's
SASS section delimited by the `Function : <mangled>` marker
cuobjdump emits, applies the per-variant LDG regex, normalizes
matching lines (strip whitespace, strip leading
`/* ... */` instruction-address comments, collapse internal
whitespace, preserve order), and hashes with sha256.

The kernel-scoped extraction is the R29 patch per Codex's R28 review:
helper-function `LDG`s and host-side disassembly do NOT contribute to
the count. The two-target-modifier rule is the R29 patch per the
R28 false-admit boundary closure: cache-modifier admission requires
`sass_match_count >= sass_expected_match_count = 2`. The synthetic
one-missing-modifier negative-control fixture (a SASS file with
exactly one matching `LDG...CA`/`LDG...CG` line in the target kernel
section) MUST produce `sass_status="fail"`.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

LDG_ANY_RE = re.compile(r"\bLDG(?:\.[A-Z0-9_]+)*\b")
# `LDG_CA_RE` / `LDG_CG_RE` / `LDG_CS_RE` accept BOTH the legacy
# Ampere-style mnemonic (`LDG.E.CA` / `LDG.E.CG` / `LDG.E.CS`) AND
# the Blackwell-style mnemonic the CUDA 13.x toolchain emits for
# `ld.global.{ca,cg,cs}.f32` PTX. The Blackwell encoding folds the
# cache-modifier semantic into a different qualifier surface
# (verified against actual sm_120 cuobjdump output for
# dram_stream_stride_mix perturbation variants in R45):
#   `.ca` → `LDG.E.STRONG.SM`     (strong-scope SM)
#   `.cg` → `LDG.E.STRONG.GPU`    (strong-scope GPU)
#   `.cs` → `LDG.E.EF`            (eviction-first streaming hint)
# All three forms confirm the same "intended cache modifier was
# applied" semantic (per CLAUDE.md "Two-Device Local Development
# Model": producer-side wire format and offline validators must be
# architecture-independent — the SASS confirmation regex tolerates
# per-architecture variance instead of pinning Ampere mnemonics)
# per BL-20260505-cross-arch-sass-mnemonic-divergence.
LDG_CA_RE = re.compile(
    r"\bLDG(?:\.[A-Z0-9_]+)*\.(?:CA|STRONG\.SM)(?:\.[A-Z0-9_]+)*\b"
)
LDG_CG_RE = re.compile(
    r"\bLDG(?:\.[A-Z0-9_]+)*\.(?:CG|STRONG\.GPU)(?:\.[A-Z0-9_]+)*\b"
)
LDG_CS_RE = re.compile(
    r"\bLDG(?:\.[A-Z0-9_]+)*\.(?:CS|EF)(?:\.[A-Z0-9_]+)*\b"
)

# Strip leading `/* 0xN... */` instruction-address comments cuobjdump emits.
# Example: `/*0010*/   LDG.E.CA R5, [R4]`. The `0xN...` form may also
# appear; both are stripped so the normalized line starts at the SASS
# mnemonic.
_LEADING_ADDR_COMMENT_RE = re.compile(r"^/\*[0-9A-Fa-fx_ ]+\*/\s*")
_INTERNAL_WS_RE = re.compile(r"\s+")


class SassConfirmResult(NamedTuple):
    """Per-binary SASS confirmation outcome."""
    status: str  # "pass" | "fail"
    expected_match_count: int
    match_count: int
    confirmation_hash: str  # sha256 of normalized matching lines
    artifact_path: str  # relative path of the on-disk .sass dump


def regex_for_family(family: str) -> re.Pattern[str]:
    """Return the per-family LDG regex.

    `cache_modifier_ca` -> `LDG_CA_RE`; `cache_modifier_cg` ->
    `LDG_CG_RE`; `cache_modifier_cs` -> `LDG_CS_RE`;
    `working_set_sweep` -> `LDG_ANY_RE` (records any `LDG` line
    because that family does not perturb the cache operator).
    """
    if family == "cache_modifier_ca":
        return LDG_CA_RE
    if family == "cache_modifier_cg":
        return LDG_CG_RE
    if family == "cache_modifier_cs":
        return LDG_CS_RE
    if family == "working_set_sweep":
        return LDG_ANY_RE
    raise ValueError(f"unknown perturbation family: {family!r}")


def extract_kernel_sass_section(
    sass_text: str, kernel_mangled_name: str,
) -> str:
    """Return only the lines inside the target kernel's SASS section.

    cuobjdump emits one section per function with the header
    `Function : <mangled name>` followed by SASS lines until the next
    `Function :` header (or EOF). Helper functions, host disassembly,
    and other kernels have their own sections; only the target kernel's
    body counts toward the modifier match.
    """
    lines = sass_text.splitlines()
    in_section = False
    out: list[str] = []
    function_marker = f"Function : {kernel_mangled_name}"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Function :"):
            in_section = stripped == function_marker
            continue
        if in_section:
            out.append(line)
    return "\n".join(out)


def find_kernel_mangled_name(
    sass_text: str, kernel_unmangled_name: str,
) -> str | None:
    """Find the cuobjdump-emitted mangled name for the target kernel.

    The dram_stream_stride_mix kernel is declared `extern "C"` in the
    source, so the mangled name equals the unmangled name. For
    completeness this helper falls back to a `Function : <unmangled>`
    match and otherwise scans `Function :` lines for an entry
    containing the unmangled name as a substring.
    """
    for line in sass_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function :"):
            name = stripped[len("Function :"):].strip()
            if name == kernel_unmangled_name:
                return name
            if kernel_unmangled_name in name:
                return name
    return None


def normalize_lines(matching_lines: list[str]) -> list[str]:
    """Apply the Section 3 normalization to matching SASS lines.

    1. Keep only matching lines.
    2. Strip leading/trailing whitespace.
    3. Remove leading instruction-address comments
       (`/* 0x... */`).
    4. Collapse internal whitespace to one ASCII space.
    5. Preserve file order.
    """
    out: list[str] = []
    for line in matching_lines:
        stripped = line.strip()
        stripped = _LEADING_ADDR_COMMENT_RE.sub("", stripped)
        stripped = _INTERNAL_WS_RE.sub(" ", stripped)
        if stripped:
            out.append(stripped)
    return out


def confirmation_hash(normalized: list[str]) -> str:
    """sha256 over `\\n`-joined normalized lines plus a final newline."""
    payload = "\n".join(normalized) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_cuobjdump(binary_path: Path) -> str:
    """Invoke `cuobjdump --dump-sass <binary>` and return stdout."""
    proc = subprocess.run(
        ["cuobjdump", "--dump-sass", str(binary_path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cuobjdump --dump-sass {binary_path} failed: "
            f"rc={proc.returncode}; stderr_tail="
            f"{proc.stderr[-1024:]!r}"
        )
    return proc.stdout


def confirm_from_sass_text(
    sass_text: str, *, family: str,
    kernel_unmangled_name: str,
    expected_match_count: int,
    sass_artifact_path: str,
) -> SassConfirmResult:
    """Run the confirmation pipeline against an in-memory SASS dump.

    Used by both the live producer (which dumps from cuobjdump first
    and passes the text in) and the negative-control test fixtures
    (which inject synthetic SASS to exercise the typed-unavailable
    arms).
    """
    mangled = find_kernel_mangled_name(sass_text, kernel_unmangled_name)
    if mangled is None:
        return SassConfirmResult(
            status="fail",
            expected_match_count=expected_match_count,
            match_count=0,
            confirmation_hash=confirmation_hash([]),
            artifact_path=sass_artifact_path,
        )
    section_text = extract_kernel_sass_section(sass_text, mangled)
    pattern = regex_for_family(family)
    matching = [
        line for line in section_text.splitlines() if pattern.search(line)
    ]
    normalized = normalize_lines(matching)
    match_count = len(normalized)
    hash_hex = confirmation_hash(normalized)
    status = "pass" if match_count >= expected_match_count else "fail"
    return SassConfirmResult(
        status=status,
        expected_match_count=expected_match_count,
        match_count=match_count,
        confirmation_hash=hash_hex,
        artifact_path=sass_artifact_path,
    )


def confirm_binary(
    binary_path: Path, *, family: str,
    kernel_unmangled_name: str,
    expected_match_count: int,
    sass_dump_dir: Path,
) -> SassConfirmResult:
    """End-to-end live confirmation: dump SASS, persist to disk, run
    `confirm_from_sass_text`, and return the result.

    The persisted dump is written to
    `<sass_dump_dir>/<binary_name>.sass`; that path is the
    `sass_artifact_path` recorded in the ledger record.
    """
    sass_text = run_cuobjdump(binary_path)
    sass_dump_dir.mkdir(parents=True, exist_ok=True)
    sass_path = sass_dump_dir / f"{binary_path.name}.sass"
    sass_path.write_text(sass_text, encoding="utf-8")
    return confirm_from_sass_text(
        sass_text,
        family=family,
        kernel_unmangled_name=kernel_unmangled_name,
        expected_match_count=expected_match_count,
        sass_artifact_path=str(sass_path),
    )


__all__ = [
    "LDG_ANY_RE",
    "LDG_CA_RE",
    "LDG_CG_RE",
    "LDG_CS_RE",
    "SassConfirmResult",
    "regex_for_family",
    "extract_kernel_sass_section",
    "find_kernel_mangled_name",
    "normalize_lines",
    "confirmation_hash",
    "run_cuobjdump",
    "confirm_from_sass_text",
    "confirm_binary",
]
