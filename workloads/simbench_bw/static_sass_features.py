"""Static SASS structural-feature extractor.

Runs the read-only CUDA binary toolchain (``cuobjdump --dump-sass`` +
``cuobjdump -res-usage``) over a BUILT kernel binary and emits a schema-valid
``static_sass_features`` record (the closed schema at
``workloads/simbench_bw/schemas/static_sass_features.schema.json`` is consumed
unchanged) plus a SEPARATE schema-valid ``static_work`` record carrying a
structural FLOP/byte estimate.

The feature record is a per-kernel mnemonic-family histogram (the closed
9-family taxonomy: floating_point, integer, load_store, tensor_core, control,
atomics, shared_memory_ops, predication, other) plus the static register-usage
record. Every disassembled instruction is classified; a mnemonic outside the
known Ampere map lands in the ``other`` family AND is recorded in that bucket's
``exemplar_mnemonics`` (a lossless fallback per CLAUDE.md's unknown-opcode
discipline, never silently dropped). Pass ``--strict-unknown`` to raise instead.

The ``kernel_signature_sha256`` is NOT derivable from static SASS alone — the
canonical signature (``kernel_signature.spec.md``) hashes launch geometry
(grid/block dims + shared-mem) that only exists at launch time. The caller
therefore supplies the full ten-field launch-config record (via
``--launch-config <json>``), and this extractor computes the signature through
the single production implementation
(``nvbit_trace_summary_adapter.compute_kernel_signature_sha256``) so the value
matches the CUPTI / NVBit join key byte-for-byte. The disassembly-derived
``regs_per_thread`` is cross-checked against the launch config's value and a
mismatch fails closed.

The FLOP/byte estimate is a SANITY-CHECK covariate only: its ``claim_class`` is
fixed to ``static_sanity_check_only`` and the closed feature schema deliberately
does not carry a work estimate, so the estimate rides the separate
``static_work`` artifact.

Toolchain is read-only and permitted under the non-root contract; no NCU, no
driver state, no kernel execution.

CLI:
  python workloads/simbench_bw/static_sass_features.py \
    --binary workloads/portable_lcd/<slug>/<slug>_run --slug <slug> \
    --launch-config <launch.json> [--kernel <mangled-or-demangled-name>] \
    [--arch sm_86] [--features-out <path.json>] [--work-out <path.json>] \
    [--strict-unknown]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The single production Python implementation of the canonical kernel signature
# (byte-identical to the CUPTI-side C++ encoder). Reused here so the static
# feature record's join key matches the timing / cache evidence join keys.
from workloads.simbench_bw.nvbit_trace_summary_adapter import (  # noqa: E402
    KERNEL_SIGNATURE_FIELD_ORDER as CANONICAL_FIELD_ORDER,
    compute_kernel_signature_sha256,
)

CUOBJDUMP = "cuobjdump"
CUFILT = "cu++filt"
DEFAULT_ARCH = "sm_86"

# The closed static_sass_features schema's 9-family taxonomy. The order here is
# the canonical emission order; every family is always present in the record.
FAMILIES = (
    "floating_point",
    "integer",
    "load_store",
    "tensor_core",
    "control",
    "atomics",
    "shared_memory_ops",
    "predication",
    "other",
)

# Mnemonic -> family map for the Ampere (sm_86 / GA102) SASS surface. Derived
# from the distilled SASS reference card and confirmed against the head
# mnemonics emitted by every built local binary. Classification rules:
#   - arithmetic ops are bucketed by pipe (fp / integer / tensor);
#   - predicate-producing / predicate-moving ops (the *SETP family, P2R/R2P,
#     PLOP3, VOTE) go to ``predication``;
#   - generic + async global/local/constant loads and stores go to
#     ``load_store``; shared loads/stores go to ``shared_memory_ops``;
#   - branch / call / return / exit control flow goes to ``control``;
#   - movement, conversion, special-register reads, barriers/fences, texture,
#     and the uniform datapath go to ``other`` (the closed schema has no
#     dedicated bucket for these; they are structural context, not a scored
#     axis).
_MNEMONIC_FAMILY: dict[str, str] = {
    # floating-point arithmetic (fp16 / fp32 / fp64 pipes)
    "FADD": "floating_point", "FADD32I": "floating_point",
    "FFMA": "floating_point", "FFMA32I": "floating_point",
    "FMUL": "floating_point", "FMUL32I": "floating_point",
    "FMNMX": "floating_point", "FMNMX3": "floating_point",
    "FSEL": "floating_point", "MUFU": "floating_point",
    "FCHK": "floating_point", "FRND": "floating_point",
    "FADD2": "floating_point", "FFMA2": "floating_point",
    "FMUL2": "floating_point",
    "HADD2": "floating_point", "HFMA2": "floating_point",
    "HMUL2": "floating_point", "HMNMX2": "floating_point",
    "DADD": "floating_point", "DFMA": "floating_point",
    "DMUL": "floating_point", "DMNMX": "floating_point",
    # integer arithmetic / logic / shift
    "IADD": "integer", "IADD3": "integer", "IADD32I": "integer",
    "IMAD": "integer", "IMAD32I": "integer", "IMUL": "integer",
    "IMNMX": "integer", "IMNMX3": "integer", "IMNMX2": "integer",
    "LEA": "integer", "LOP": "integer", "LOP3": "integer",
    "LOP32I": "integer", "POPC": "integer", "SHF": "integer",
    "SHL": "integer", "SHR": "integer", "BMSK": "integer",
    "FLO": "integer", "IABS": "integer", "BFE": "integer",
    "BFI": "integer", "VABSDIFF": "integer", "VABSDIFF4": "integer",
    "IDP": "integer", "IDP4A": "integer",
    # generic / async load-store (global, local, constant)
    "LD": "load_store", "ST": "load_store",
    "LDG": "load_store", "STG": "load_store",
    "LDL": "load_store", "STL": "load_store",
    "LDC": "load_store", "LDCU": "load_store",
    # LDGSTS (cp.async global->shared) is BOTH a global load and a shared store;
    # it is bucketed as load_store because the schema has no async-copy bucket.
    "LDGSTS": "load_store",
    "LDGMC": "load_store",
    # tensor-core MMA + matrix feeders. NOTE: LDSM/STSM/MOVM are matrix-tile
    # MOVEMENT feeders, not MMA arithmetic; they live in the tensor_core
    # histogram family for structural context but are EXCLUDED from the MMA FLOP
    # estimate (see _MMA_FLOP_MNEMONICS).
    "HMMA": "tensor_core", "IMMA": "tensor_core", "BMMA": "tensor_core",
    "DMMA": "tensor_core", "OMMA": "tensor_core", "QMMA": "tensor_core",
    "LDSM": "tensor_core", "STSM": "tensor_core", "MOVM": "tensor_core",
    # control flow
    "BRA": "control", "BRX": "control", "BRXU": "control",
    "BSSY": "control", "BSYNC": "control", "BREAK": "control",
    "CALL": "control", "JMP": "control", "JMX": "control",
    "RET": "control", "EXIT": "control", "KILL": "control",
    "WARPSYNC": "control", "YIELD": "control", "NANOSLEEP": "control",
    "SYNC": "control", "SSY": "control", "BPT": "control", "RTT": "control",
    # atomics (incl. RED: a reducing store, atomic-class, not a plain load/store)
    "ATOM": "atomics", "ATOMG": "atomics", "ATOMS": "atomics",
    "RED": "atomics", "REDG": "atomics", "MATCH": "atomics", "QSPC": "atomics",
    # shared-memory load/store
    "LDS": "shared_memory_ops", "STS": "shared_memory_ops",
    # predicate-producing / predicate-moving
    "ISETP": "predication", "FSETP": "predication", "DSETP": "predication",
    "HSETP2": "predication", "PSETP": "predication", "PLOP3": "predication",
    "P2R": "predication", "R2P": "predication", "VOTE": "predication",
    # MUFU (reciprocal / rsqrt / transcendental special-function unit) is an FP
    # pipe op but NOT FFMA-class throughput; counted in floating_point for the
    # histogram, but it carries NO weight in the FLOP estimate (only fp arith +
    # tensor do — see _FLOP_WEIGHTS), so it does not inflate estimated_flops.
    # ----------------------------------------------------------------------
    # `other` is a deliberate heterogeneous catch-all (movement + conversion +
    # special-register reads + barriers/fences + texture + the entire uniform
    # datapath). The closed static_sass_features schema has no dedicated bucket
    # for these, and none of them is a scored axis: the static_work FLOP/byte
    # estimate reads ONLY floating_point / tensor_core / load_store /
    # shared_memory_ops, never `other`. So routing this mix into `other` is a
    # structural-context bucket, not a measured quantity.
    "MOV": "other", "MOV32I": "other", "PRMT": "other", "SEL": "other",
    "SHFL": "other", "SGXT": "other",
    "F2F": "other", "F2I": "other", "I2F": "other", "I2I": "other",
    "I2IP": "other", "I2FP": "other", "F2IP": "other", "F2FP": "other",
    "S2R": "other", "CS2R": "other", "B2R": "other", "R2B": "other",
    "BAR": "other", "DEPBAR": "other", "MEMBAR": "other",
    "ERRBAR": "other", "FENCE": "other",
    # LDGDEPBAR is a Global Load Dependency Barrier (load ordering), NOT memory
    # traffic — classed with the other barriers, never charged bytes.
    "LDGDEPBAR": "other",
    "NOP": "other", "CCTL": "other", "CCTLL": "other", "CCTLT": "other",
    "TEX": "other", "TLD": "other", "TLD4": "other", "TXQ": "other",
    "TMML": "other", "TXD": "other",
    "ULDC": "other", "UMOV": "other", "UIADD3": "other", "UIMAD": "other",
    "ULOP3": "other", "USHF": "other", "USEL": "other", "UFLO": "other",
    "UPOPC": "other", "R2UR": "other", "S2UR": "other", "UISETP": "other",
    "UPSETP": "other", "UFFMA": "other", "UPRMT": "other", "VOTEU": "other",
    "ULDCU": "other",
}

# The FLOP/byte sanity-check estimate is computed from EXPLICIT per-mnemonic
# sets (not the closed histogram families), so matrix-movement feeders
# (LDSM/STSM/MOVM) and load-ordering barriers (LDGDEPBAR) are NOT miscounted as
# MMA FLOPs or memory bytes. Convention: ORDER-OF-MAGNITUDE, PER-WARP-INSTRUCTION
# (the static SASS dump is the per-warp instruction stream).
#
# Genuine MMA arithmetic ops (each charged the MMA FLOP weight). LDSM/STSM/MOVM
# are deliberately ABSENT — they move tiles, they do not multiply-accumulate.
_MMA_FLOP_MNEMONICS = frozenset({"HMMA", "IMMA", "BMMA", "DMMA", "OMMA", "QMMA"})
# Scalar/vector FP-arithmetic ops charged 2 FLOP/instr (FFMA-class). MUFU
# (special-function unit) is included as a coarse 2-FLOP proxy.
_FP_ARITH_FLOP_MNEMONICS = frozenset({
    "FADD", "FADD32I", "FFMA", "FFMA32I", "FMUL", "FMUL32I", "FMNMX", "FMNMX3",
    "FSEL", "MUFU", "FCHK", "FRND", "FADD2", "FFMA2", "FMUL2",
    "HADD2", "HFMA2", "HMUL2", "HMNMX2", "DADD", "DFMA", "DMUL", "DMNMX",
})
# Genuine global-memory traffic ops (charged the global byte width). Barriers
# (LDGDEPBAR) are ABSENT — they carry no traffic.
_GLOBAL_BYTE_MNEMONICS = frozenset({
    "LD", "ST", "LDG", "STG", "LDL", "STL", "LDC", "LDCU", "LDGSTS", "LDGMC",
})
# Genuine shared-memory traffic ops (charged the shared byte width). LDSM/STSM
# (matrix-tile shared loads/stores) DO move shared bytes, so they are counted
# here for the byte estimate even though they sit in the tensor_core histogram.
_SHARED_BYTE_MNEMONICS = frozenset({"LDS", "STS", "LDSM", "STSM"})

_FP_ARITH_FLOP_PER_INSTR = 2.0
# 4096 = 2*16*8*16, the FLOP of one 16x8x16 BF16/FP16 HMMA warp-instruction
# (the common Ampere shape, e.g. HMMA.16816). Coarse sanity-check magnitude,
# NOT a calibrated per-shape count — a 16x8x8 TF32 HMMA has a different count
# this single weight does not distinguish.
_MMA_FLOP_PER_INSTR = 4096.0
_GLOBAL_BYTES_PER_INSTR = 16.0
_SHARED_BYTES_PER_INSTR = 8.0

# Human-readable convention string emitted in every static_work record so a
# downstream reader cannot mistake the estimate for a calibrated count.
_ESTIMATE_CONVENTION = (
    "order_of_magnitude_per_warp_instruction_sanity_check_only; "
    "FFMA=2 FLOP/instr; HMMA assumed 16x8x16 (2*16*8*16=4096 FLOP/instr); "
    "global op=16 B/instr; shared op=8 B/instr; "
    "NOT a calibrated per-shape FLOP/byte count"
)


class StaticSassExtractionError(RuntimeError):
    """Raised when the SASS toolchain fails, when a launch-config field is
    missing / inconsistent, or (in strict mode) when an unmapped mnemonic is
    encountered."""


class StaticSassArchitectureUnavailable(StaticSassExtractionError):
    """Typed fail-closed result for an unsupported architecture surface."""

    TOKEN = "static_sass_architecture_research_gap"


_INSTR_RE = re.compile(
    r"/\*[0-9a-f]+\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)"
)
# cuobjdump emits a per-kernel header line of the form
# ``\t\tFunction : <mangled_name>`` in --dump-sass output and
# ``\t Function <mangled_name>:`` in -res-usage output. Match both shapes
# (name after a colon, or name before a trailing colon).
_FUNC_SASS_RE = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
_FUNC_RES_RE = re.compile(r"^\s*Function\s+(\S+?):\s*$")
_REG_RE = re.compile(r"REG:(\d+)")
_STACK_RE = re.compile(r"STACK:(\d+)")


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise StaticSassExtractionError(f"tool not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise StaticSassExtractionError(
            f"{cmd[0]} failed (rc={exc.returncode}): {exc.stderr.strip()[:200]}"
        ) from exc
    return out.stdout


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _demangle(name: str) -> str:
    if not name.startswith("_Z"):
        return name
    try:
        out = _run([CUFILT, name]).strip()
        return out or name
    except StaticSassExtractionError:
        return name


def classify_mnemonic(mnemonic: str, *, strict: bool = False) -> str:
    """Return the closed-schema family for a SASS head mnemonic.

    Unknown mnemonics map to ``other`` (lossless fallback) unless ``strict`` is
    set, in which case a StaticSassExtractionError is raised.
    """
    fam = _MNEMONIC_FAMILY.get(mnemonic)
    if fam is not None:
        return fam
    if strict:
        raise StaticSassExtractionError(
            f"unmapped SASS mnemonic {mnemonic!r} (strict mode)"
        )
    return "other"


def _parse_sass_by_function(sass_text: str) -> dict[str, list[str]]:
    """Return {mangled_function_name: [head_mnemonic, ...]} for the dump."""
    per_func: dict[str, list[str]] = {}
    current: str | None = None
    for line in sass_text.splitlines():
        fm = _FUNC_SASS_RE.match(line)
        if fm:
            current = fm.group(1).strip()
            per_func.setdefault(current, [])
            continue
        if current is None:
            continue
        im = _INSTR_RE.search(line)
        if im:
            per_func[current].append(im.group(1))
    return per_func


def _parse_res_usage(res_text: str) -> dict[str, dict[str, int]]:
    """Return {mangled_function_name: {regs_per_thread, stack_frame_bytes}}."""
    per_func: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in res_text.splitlines():
        fm = _FUNC_RES_RE.match(line)
        if fm:
            current = fm.group(1).strip()
            continue
        if current is None:
            continue
        reg = _REG_RE.search(line)
        stack = _STACK_RE.search(line)
        if reg or stack:
            per_func[current] = {
                "regs_per_thread": int(reg.group(1)) if reg else 0,
                "stack_frame_bytes": int(stack.group(1)) if stack else 0,
            }
            current = None
    return per_func


def _build_histogram(mnemonics: list[str], *, strict: bool) -> dict:
    counts = {fam: 0 for fam in FAMILIES}
    exemplars: dict[str, list[str]] = {fam: [] for fam in FAMILIES}
    for m in mnemonics:
        fam = classify_mnemonic(m, strict=strict)
        counts[fam] += 1
        if m not in exemplars[fam] and len(exemplars[fam]) < 8:
            exemplars[fam].append(m)
    families = {
        fam: {
            "count": counts[fam],
            "exemplar_mnemonics": sorted(exemplars[fam]),
        }
        for fam in FAMILIES
    }
    return {
        "schema_version": 1,
        "total_instruction_count": len(mnemonics),
        "families": families,
    }


def _validate_launch_config(launch_config: dict) -> dict:
    """Validate the caller-supplied ten-field launch config and return a copy
    restricted to the canonical fields (so the signature inputs are exactly the
    spec's fields, in order)."""
    missing = [f for f in CANONICAL_FIELD_ORDER if f not in launch_config]
    if missing:
        raise StaticSassExtractionError(
            f"launch_config missing canonical signature field(s): {missing}"
        )
    return {f: launch_config[f] for f in CANONICAL_FIELD_ORDER}


def _extract_feature_and_mnemonics(
    *,
    binary: Path,
    slug: str,
    launch_config: dict,
    kernel: str | None = None,
    arch: str = DEFAULT_ARCH,
    strict_unknown: bool = False,
) -> tuple[dict, list[str]]:
    """Core extraction: return (static_sass_features record, raw mnemonic list).

    ``launch_config`` is the caller-supplied ten-field canonical launch record
    (kernel_name + grid/block dims + shared-mem + regs_per_thread); the kernel
    signature is computed from it through the production implementation. The
    selected function MUST be the one the launch config (or ``--kernel``) names —
    there is no silent dominant-function fallback — and its demangled name and
    static REG count must agree with the signature inputs.
    """
    binary = Path(binary)
    if arch != "sm_86":
        raise StaticSassArchitectureUnavailable(
            f"static_sass_architecture_research_gap: no classified static "
            f"SASS feature map for {arch!r}; numeric extraction is disabled"
        )
    if not binary.is_file():
        raise StaticSassExtractionError(f"binary not found: {binary}")
    sig_inputs = _validate_launch_config(launch_config)
    binary_sha = _sha256_file(binary)
    sass = _run([CUOBJDUMP, "--dump-sass", "-arch", arch, str(binary)])
    res = _run([CUOBJDUMP, "-res-usage", "-arch", arch, str(binary)])
    per_func = _parse_sass_by_function(sass)
    if not per_func:
        raise StaticSassExtractionError(
            f"no {arch} functions disassembled from {binary}"
        )
    res_usage = _parse_res_usage(res)

    # Kernel selection is EXPLICIT, never a silent fallback: the chosen
    # function must be the one the launch config / --kernel names, matched by
    # mangled symbol OR demangled name. A silent "dominant function" fallback
    # was the defect Codex flagged — it let the record describe one kernel while
    # kernel_signature_sha256 hashed a launch config for another. ``--kernel``
    # overrides the match target; otherwise the launch config's ``kernel_name``
    # (the canonical-signature demangled name) is the target.
    target_name = kernel if kernel is not None else sig_inputs["kernel_name"]
    matches = [
        name for name in per_func
        if name == target_name or _demangle(name) == target_name
    ]
    if not matches:
        available = sorted(_demangle(n) for n in per_func)
        raise StaticSassExtractionError(
            f"kernel {target_name!r} not found in {binary} ({arch}); the launch "
            f"config's kernel_name (or --kernel) must name a disassembled "
            f"function so the emitted record and kernel_signature_sha256 "
            f"describe the SAME kernel. Disassembled: {available}"
        )
    chosen = matches[0]

    # The launch config's kernel_name is hashed into kernel_signature_sha256, so
    # the emitted record's kernel_name MUST equal it (the spec's canonical name
    # is the demangled symbol). Fail closed if they disagree.
    chosen_demangled = _demangle(chosen)
    if chosen_demangled != sig_inputs["kernel_name"]:
        raise StaticSassExtractionError(
            f"kernel_name mismatch: launch config hashes "
            f"{sig_inputs['kernel_name']!r} into kernel_signature_sha256, but the "
            f"selected SASS function demangles to {chosen_demangled!r}. The "
            f"signature and the record must describe the same kernel; set the "
            f"launch config's kernel_name to {chosen_demangled!r}."
        )

    mnemonics = per_func[chosen]
    histogram = _build_histogram(mnemonics, strict=strict_unknown)
    spill_loads = sum(1 for m in mnemonics if m == "LDL")
    spill_stores = sum(1 for m in mnemonics if m == "STL")
    ru = res_usage.get(chosen, {"regs_per_thread": 0, "stack_frame_bytes": 0})

    # The signature's regs_per_thread (the launch config's
    # cuFuncGetAttribute(NUM_REGS) value) and the disassembly's static cubin
    # REG count must agree for the SAME build, or the emitted
    # register_usage.regs_per_thread would contradict the value baked into
    # kernel_signature_sha256. Fail closed on a mismatch rather than emit an
    # internally inconsistent record.
    static_regs = ru["regs_per_thread"]
    sig_regs = sig_inputs["regs_per_thread"]
    if static_regs > 0 and sig_regs != static_regs:
        raise StaticSassExtractionError(
            f"regs_per_thread mismatch for {chosen!r}: launch config carries "
            f"{sig_regs} (the value hashed into kernel_signature_sha256) but the "
            f"cuobjdump -res-usage static cubin REG count is {static_regs}; for "
            f"one build these must match. Correct the launch config's "
            f"regs_per_thread to the static value {static_regs}."
        )

    sig = compute_kernel_signature_sha256(sig_inputs)
    record = {
        "schema_version": 1,
        "slug": slug,
        "binary_sha256": binary_sha,
        "kernel_name": chosen_demangled,
        "kernel_signature_sha256": sig,
        "claim_scope": "measured_local",
        "claim_class": "static_sanity_check_only",
        "mnemonic_histogram": histogram,
        "register_usage": {
            "regs_per_thread": ru["regs_per_thread"],
            "stack_frame_bytes": ru["stack_frame_bytes"],
            "spill_loads_static": spill_loads,
            "spill_stores_static": spill_stores,
        },
        "tool_provenance": {
            "extractor_name": "cuobjdump_dump_sass",
            "extractor_version": _cuobjdump_version(),
            "extracted_dated_iso8601": _now_iso8601(),
        },
    }
    return record, mnemonics


def extract_features(
    *,
    binary: Path,
    slug: str,
    launch_config: dict,
    kernel: str | None = None,
    arch: str = DEFAULT_ARCH,
    strict_unknown: bool = False,
) -> dict:
    """Extract a single schema-valid static_sass_features record (the feature
    record only; use ``extract_records`` to also build the static_work record)."""
    record, _ = _extract_feature_and_mnemonics(
        binary=binary, slug=slug, launch_config=launch_config,
        kernel=kernel, arch=arch, strict_unknown=strict_unknown,
    )
    return record


def unknown_mnemonics(mnemonics: list[str]) -> list[str]:
    """Return the sorted list of DISTINCT mnemonics not in the known Ampere map.

    These are the tokens that fall into the ``other`` histogram family via the
    fail-closed fallback. The histogram's per-family ``exemplar_mnemonics`` is
    capped at 8, so this complete list — emitted on the static_work artifact — is
    the lossless record of every unrecognized opcode.
    """
    return sorted({m for m in mnemonics if m not in _MNEMONIC_FAMILY})


def compute_work_estimate(mnemonics: list[str]) -> tuple[float, float]:
    """Return (estimated_flops, estimated_bytes) from EXPLICIT per-mnemonic sets.

    Computed from the raw mnemonic counts, NOT the closed histogram families, so
    matrix-movement feeders (LDSM/STSM/MOVM) are not charged MMA FLOPs and the
    load-ordering barrier (LDGDEPBAR) is not charged bytes. Order-of-magnitude
    per-warp-instruction sanity-check only.
    """
    fp_arith = sum(1 for m in mnemonics if m in _FP_ARITH_FLOP_MNEMONICS)
    mma = sum(1 for m in mnemonics if m in _MMA_FLOP_MNEMONICS)
    global_mem = sum(1 for m in mnemonics if m in _GLOBAL_BYTE_MNEMONICS)
    shared_mem = sum(1 for m in mnemonics if m in _SHARED_BYTE_MNEMONICS)
    estimated_flops = (
        fp_arith * _FP_ARITH_FLOP_PER_INSTR + mma * _MMA_FLOP_PER_INSTR
    )
    estimated_bytes = (
        global_mem * _GLOBAL_BYTES_PER_INSTR
        + shared_mem * _SHARED_BYTES_PER_INSTR
    )
    return estimated_flops, estimated_bytes


def build_static_work(
    feature_record: dict,
    *,
    mnemonics: list[str],
    source_commit: str,
    arch: str = DEFAULT_ARCH,
) -> dict:
    """Build the SEPARATE static_work artifact (FLOP/byte sanity-check
    covariate) from a static_sass_features record + its raw mnemonic list.

    The estimate is computed from explicit per-mnemonic sets (see
    ``compute_work_estimate``), and the full unknown-opcode list is recorded so
    the artifact is a lossless record of every unrecognized mnemonic.
    """
    estimated_flops, estimated_bytes = compute_work_estimate(mnemonics)
    return {
        "schema_version": 1,
        "artifact_kind": "static_work",
        "claim_class": "static_sanity_check_only",
        "arch": arch,
        "kernel_signature_sha256": feature_record["kernel_signature_sha256"],
        "binary_sha256": feature_record["binary_sha256"],
        "source_commit": source_commit,
        "estimate_convention": _ESTIMATE_CONVENTION,
        "estimated_flops": estimated_flops,
        "estimated_bytes": estimated_bytes,
        "flop_weights": {
            "fp32_arith": _FP_ARITH_FLOP_PER_INSTR,
            "fp64_arith": _FP_ARITH_FLOP_PER_INSTR,
            "fp16_arith": _FP_ARITH_FLOP_PER_INSTR,
            "tensor_mma": _MMA_FLOP_PER_INSTR,
        },
        "byte_widths": {
            "memory_global": _GLOBAL_BYTES_PER_INSTR,
            "memory_shared": _SHARED_BYTES_PER_INSTR,
            "memory_local": _GLOBAL_BYTES_PER_INSTR,
            "memory_texture": _GLOBAL_BYTES_PER_INSTR,
        },
        "unknown_mnemonics": unknown_mnemonics(mnemonics),
    }


def extract_records(
    *,
    binary: Path,
    slug: str,
    launch_config: dict,
    source_commit: str,
    kernel: str | None = None,
    arch: str = DEFAULT_ARCH,
    strict_unknown: bool = False,
) -> tuple[dict, dict]:
    """Disassemble once and return (static_sass_features, static_work) records."""
    record, mnemonics = _extract_feature_and_mnemonics(
        binary=binary, slug=slug, launch_config=launch_config,
        kernel=kernel, arch=arch, strict_unknown=strict_unknown,
    )
    work = build_static_work(
        record, mnemonics=mnemonics, source_commit=source_commit, arch=arch
    )
    return record, work


def _cuobjdump_version() -> str:
    try:
        out = _run([CUOBJDUMP, "--version"])
    except StaticSassExtractionError:
        return "unknown"
    for line in out.splitlines():
        if "release" in line.lower():
            return line.strip()
    stripped = out.strip().splitlines()
    return stripped[-1] if stripped else "unknown"


def _now_iso8601() -> str:
    return (
        _dt.datetime.now(tz=_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _write_canonical(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument(
        "--launch-config", type=Path, required=True,
        help="JSON file with the ten canonical launch-signature fields.",
    )
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--features-out", type=Path, default=None)
    parser.add_argument("--work-out", type=Path, default=None)
    parser.add_argument("--strict-unknown", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    launch_config = json.loads(args.launch_config.read_text(encoding="utf-8"))
    record, work = extract_records(
        binary=args.binary,
        slug=args.slug,
        launch_config=launch_config,
        source_commit=_git_head_sha(),
        kernel=args.kernel,
        arch=args.arch,
        strict_unknown=args.strict_unknown,
    )
    features_out = args.features_out or (
        REPO_ROOT / "data" / "local_labels" / "static_sass_features"
        / f"{args.slug}_{args.arch}.json"
    )
    work_out = args.work_out or (
        REPO_ROOT / "data" / "local_labels" / "static_work"
        / f"{args.slug}_{args.arch}.json"
    )
    _write_canonical(features_out, record)
    _write_canonical(work_out, work)
    print(
        f"wrote {features_out.relative_to(REPO_ROOT)} "
        f"(total_instr={record['mnemonic_histogram']['total_instruction_count']}) "
        f"and {work_out.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
