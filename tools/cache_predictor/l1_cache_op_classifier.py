"""L1 cache-op classifier.

Consumes cuobjdump SASS output text for one kernel and emits a
per-PC mapping from program counter to one of 12
underscore-vocabulary instruction-cache-op tokens. The mapping is
the input the L1 predictor head consults to decide which dynamic
trace records contribute L1 line state and which bypass the L1.

Wire vocabulary (closed set):

    LDG_E_CA, LDG_E_CG, LDG_E_CS, LDG_E_EF,
    LDG_E_STRONG_SM, LDG_E_STRONG_GPU,
    LDL, STL, LDGSTS, ATOM, TEX, OTHER

Per CLAUDE.md "Two-Device Local Development Model" and
``BL-20260505-cross-arch-sass-mnemonic-divergence``, the classifier
records the actual SASS opcode the toolchain emitted, NOT the
semantic equivalent class. The Blackwell ``LDG.E.STRONG.SM``
encoding for the PTX ``.ca`` modifier is recorded as
``LDG_E_STRONG_SM`` (L1-bypass on Blackwell), not collapsed into
``LDG_E_CA`` (the sm_86 mnemonic for the same PTX intent), because
the cache behavior is per-architecture distinct: ``LDG.E.CA``
stages into L1; ``LDG.E.STRONG.SM`` enforces SM-scope coherence
which on Blackwell bypasses L1 in favor of an L2-shared
coherence point.

The classifier is cuobjdump-backed and host-only: no NVBit, no
nvidia-smi, no privileged CUPTI. Live sm_86 coverage exercises 10
Ampere-recognizable tokens; the two Blackwell-only tokens
(``LDG_E_STRONG_SM``, ``LDG_E_STRONG_GPU``) are exercised via
canned cuobjdump TEXT FIXTURES on the sm_86-only development host;
live sm_120 coverage defers to the Tier-2-cross-arch follow-up
loop.

The classifier's output validates against
``tools/cache_predictor/cache_op_map.schema.json``. The downstream
L1 predictor head emits typed_unavailable with reason
``predictor_l1_sass_provenance_missing`` when the manifest is
absent, the cuobjdump invocation failed, or the kernel section is
empty.

KNOWN DEFECT (2026-07-06): ``_extract_kernel_section``
concatenates the kernel's SASS section across EVERY fatbin arch
slice (and matches function names by substring), so manifests
produced from multi-arch fatbins are cross-arch contaminated.
Fix pending in the label-pipeline backlog. The Rust sim does not
consume these manifests — it classifies from the control-bits
sidecar's opcode_text instead (``sim/mem/src/cache_op.rs``).
"""
from __future__ import annotations

import enum
import re
import subprocess
from pathlib import Path
from typing import Any, NamedTuple


# Typed-reason literal the L1 predictor head emits when the
# classifier cannot supply a per-PC manifest. Exported here so
# callers do not have to import the cache_predictor module's
# private frozenset. The literal is enforced in 3-surface lockstep
# with ``workloads/simbench_bw/cache_predictor.py::_TYPED_REASONS``
# and ``workloads/simbench_bw/schemas/cache_predictor.schema.json``
# (see
# ``workloads/simbench_bw/tests/test_predictor_l1_sass_provenance_missing_lockstep.py``).
PREDICTOR_L1_SASS_PROVENANCE_MISSING: str = (
    "predictor_l1_sass_provenance_missing"
)


# The 12-token closed enum the schema and the classifier mirror in
# lockstep. Adding a token requires widening BOTH this tuple AND
# the schema enum in
# ``tools/cache_predictor/cache_op_map.schema.json``; a structural
# lockstep test in this module's test suite asserts the two stay
# in sync.
INSTRUCTION_CACHE_OP_TOKENS: tuple[str, ...] = (
    "LDG_E_CA",
    "LDG_E_CG",
    "LDG_E_CS",
    "LDG_E_EF",
    "LDG_E_STRONG_SM",
    "LDG_E_STRONG_GPU",
    "LDL",
    "STL",
    "LDGSTS",
    "ATOM",
    "TEX",
    "OTHER",
)


# Per-token L1-eligibility derivation. The L1 predictor head's
# per-SM walker only fills line state on l1_eligible=True
# accesses; l1_eligible=False accesses bypass the L1 line state
# entirely. The split is documented in the schema field's
# description (``cache_op_map.schema.json``: see
# ``per_pc_instruction_cache_op.items.properties.l1_eligible``).
L1_ELIGIBLE_TOKENS: frozenset[str] = frozenset({
    # Cache at all levels (`.ca`): stages into L1 on sm_86.
    "LDG_E_CA",
    # Eviction-first hint: still passes through L1 transiently.
    "LDG_E_EF",
    # Local-memory loads: L1 is the first cache backing local
    # memory on sm_86 / sm_120.
    "LDL",
    # Local-memory stores: trigger L1 line allocation under
    # write-allocate.
    "STL",
})


L1_BYPASS_TOKENS: frozenset[str] = frozenset({
    # Cache global (`.cg`): caches at L2 only; bypasses L1.
    "LDG_E_CG",
    # Cache streaming (`.cs`): bypass-L1 streaming.
    "LDG_E_CS",
    # Blackwell strong-SM scope: enforces SM-scope coherence which
    # bypasses L1 in favor of an L2-shared coherence point.
    "LDG_E_STRONG_SM",
    # Blackwell strong-GPU scope: bypasses both L1 and the
    # SM-scope L2 coherence point in favor of a GPU-scope
    # coherence point.
    "LDG_E_STRONG_GPU",
    # Async-copy: stages into shared memory without an implicit
    # L1 transit. The plan defers any "explicit-staging-proven"
    # widening to a future round.
    "LDGSTS",
    # Atomics: go to L2 or memory directly; bypass the modeled
    # L1.
    "ATOM",
    # Texture: routes through the texture cache, not the L1.
    "TEX",
    # Unrecognized LDG / load opcodes default to bypass under the
    # conservative interpretation. The plan's typed_unavailable
    # arm covers truly-unparsed cases.
    "OTHER",
})


def derive_l1_eligible(token: str) -> bool:
    """Return ``True`` iff the given instruction_cache_op token is
    L1-eligible per the documented Ampere/Blackwell semantics. A
    token outside the closed enum raises ``ValueError`` because
    the schema closure rejects it before this helper is called."""
    if token not in INSTRUCTION_CACHE_OP_TOKENS:
        raise ValueError(
            f"unknown instruction_cache_op token: {token!r}; "
            f"expected one of {INSTRUCTION_CACHE_OP_TOKENS!r}"
        )
    return token in L1_ELIGIBLE_TOKENS


class SassProvenanceStatus(enum.Enum):
    """Outcome of the cuobjdump-and-kernel-section pipeline.

    ``MATCHED`` admits a non-empty per-PC mapping with at least one
    load-class instruction. The four fail-closed statuses route to
    the L1 predictor head's typed-unavailable arm; the head
    distinguishes them so the operator can audit the underlying
    cause:

      - ``BINARY_NOT_PARSEABLE``: cuobjdump exited non-zero or
        the binary file was missing.
      - ``NO_KERNEL_SECTION``: cuobjdump output had no
        ``Function : <kernel>`` marker for the requested kernel.
      - ``KERNEL_SECTION_EMPTY``: the kernel section was found but
        contained no SASS lines at all (e.g., a degenerate or
        stripped binary).
      - ``MATCHED_NO_LOAD_CLASS_INSTRUCTIONS``: the section parsed
        cleanly but contained no LDG / LDGSTS / LDL / STL / ATOM /
        TEX class instructions (e.g., a shared-only kernel whose
        only loads are LDS / LDSM, or a pure ALU kernel). The L1
        head routes this status to
        ``predictor_l1_no_modeled_l1_eligible_accesses`` because
        SASS provenance is present and parseable; there is just
        nothing for the L1 model to walk.
    """

    MATCHED = "matched"
    BINARY_NOT_PARSEABLE = "binary_not_parseable"
    NO_KERNEL_SECTION = "no_kernel_section"
    KERNEL_SECTION_EMPTY = "kernel_section_empty"
    MATCHED_NO_LOAD_CLASS_INSTRUCTIONS = "matched_no_load_class_instructions"


class OpMapEntry(NamedTuple):
    """One per-PC mapping entry."""

    pc: str
    instruction_cache_op: str
    l1_eligible: bool
    raw_mnemonic: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "pc": self.pc,
            "instruction_cache_op": self.instruction_cache_op,
            "l1_eligible": self.l1_eligible,
        }
        if self.raw_mnemonic:
            out["raw_mnemonic"] = self.raw_mnemonic
        return out


# Pre-compiled patterns for opcode mnemonic matching. Order matters
# for the LDG variants: the more specific
# ``LDG.E.STRONG.<scope>`` patterns must be tested before the
# generic ``LDG.E.<modifier>`` ones because both share the
# ``LDG.E.`` prefix.
#
# Each pattern matches the OPCODE token at the start of the
# normalized SASS line (after the leading PC comment). The
# patterns are word-bounded to avoid false-matching opcodes
# embedded in operand syntax (e.g., a hypothetical ``[R5+LDG]``
# operand suffix).
# Note: the prior regex `0x?(...)` required the PC literal to start
# with `0` followed by an optional `x`, which silently dropped any
# PC whose first hex character was not `0` (e.g., `/* 0x1234 */`).
# The corrected pattern `(?:0x)?(...)` treats the entire `0x` prefix
# as optional, admitting all PC literals regardless of first hex
# character. See R14 dual-root-cause diagnostic at
# `data/sm_86_predictor_accuracy_recovery_v2/carveout_l1_head_diagnostic.json`
# (root_cause_classification.class_a_classifier_manifest_filter_gap).
# IMPORTANT (R15 finding): re-running
# `tools/sm_86_predictor_accuracy_recovery_v2/generate_admitted_cell_sass_manifests.py`
# with this fix admits more PCs into each manifest (carveout
# 25 -> 111). When fed into
# `compute_predicted_l1_per_admitted_cell.py` the larger PC set
# produces walker predictions that REGRESS against the calibration
# cohort (pre-R15 walker outputs are baked into the L1 subfamily
# calibration parity tests + task12_substitution_preflight
# verifiers). A holistic re-calibration round (re-derive
# `_L1_SHARED_REDUCTION_CORRECTION_FACTOR` etc. against the
# larger-PC walker outputs + refresh the task12 preflight
# diagnostic) is required before the committed downstream
# artifacts are regenerated with the corrected classifier. R15
# lands the source-code fix so the classifier is correct going
# forward; the committed downstream artifacts remain on the
# broken-classifier-derived snapshot until that re-calibration
# round.
_PC_RE = re.compile(r"^/\*\s*(?:0x)?([0-9a-fA-F]+)\s*\*/\s*(.*)$")

# Token-recognizer rules, in priority order. Each rule maps a
# regex match on the opcode portion of the line to a wire token.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Blackwell strong-scope variants (must test before generic
    # LDG.E.CA-like patterns).
    (re.compile(r"\bLDG\.E\.STRONG\.SM\b"), "LDG_E_STRONG_SM"),
    (re.compile(r"\bLDG\.E\.STRONG\.GPU\b"), "LDG_E_STRONG_GPU"),
    # Cache modifiers (.ca / .cg / .cs / .ef).
    (re.compile(r"\bLDG\.E\.CA\b"), "LDG_E_CA"),
    (re.compile(r"\bLDG\.E\.CG\b"), "LDG_E_CG"),
    (re.compile(r"\bLDG\.E\.CS\b"), "LDG_E_CS"),
    (re.compile(r"\bLDG\.E\.EF\b"), "LDG_E_EF"),
    # Async copy (matches LDGSTS and LDGSTS.E etc.).
    (re.compile(r"\bLDGSTS\b"), "LDGSTS"),
    # Local-memory loads / stores.
    (re.compile(r"\bLDL\b"), "LDL"),
    (re.compile(r"\bSTL\b"), "STL"),
    # Atomics: ATOM, ATOMS, ATOMG variants all map to ATOM. Use a
    # word-bounded prefix so e.g. ``ATOM.E.ADD`` matches.
    (re.compile(r"\bATOM\b|\bATOMS\b|\bATOMG\b"), "ATOM"),
    # Texture (TEX, TLD, TLD4 are texture-fetch family). The
    # plan's vocabulary collapses them to TEX.
    (re.compile(r"\bTEX(?:\.[A-Z0-9]+)*\b|\bTLD(?:4|S)?\b"), "TEX"),
    # Generic LDG (the default cache modifier when no .CA/.CG/.CS
    # suffix is present). Falls under OTHER.
    (re.compile(r"\bLDG\.E\b|\bLDG\b"), "OTHER"),
)


# Opcodes the classifier considers candidates for the per-PC
# manifest. Lines without a load-class opcode (e.g., MOV, EXIT,
# FADD) are silently skipped because they do not contribute to
# the L1 line state model.
_OPCODE_LOAD_LIKE_RE = re.compile(
    r"\b(?:LDG|LDGSTS|LDL|STL|ATOM|ATOMS|ATOMG|TEX|TLD)\b"
)


def _strip_leading_pc_comment(line: str) -> tuple[str | None, str]:
    """Return ``(pc, opcode_part)`` if the line begins with a
    cuobjdump PC comment, else ``(None, line)``.

    cuobjdump emits one of two shapes:
      - ``/*0x10*/  LDG.E.CA R5, [R4] ;``
      - ``/*0010*/  LDG.E.CA R5, [R4] ;``
    Both are normalized to a lowercase hex ``0x``-prefixed PC.
    """
    m = _PC_RE.match(line.strip())
    if m is None:
        return None, line
    pc_hex_raw = m.group(1).lower()
    # cuobjdump prints PCs as 4+ hex digits without a leading 0x.
    # Normalize to ``0x<lowercase-hex>`` with leading zeros
    # stripped (but keep a single 0 for PC zero).
    pc_clean = pc_hex_raw.lstrip("0") or "0"
    return f"0x{pc_clean}", m.group(2).strip()


_BARE_LDG_E_RE = re.compile(r"\bLDG\.E\b")
_BARE_LDG_RE = re.compile(r"\bLDG\b")

# Arch tokens for which a bare ``LDG.E`` (no explicit cache
# modifier suffix in the cuobjdump output) is resolved to
# ``LDG_E_CA``. On Ampere (sm_86) the PTX-to-SASS lowering emits
# bare ``LDG.E`` for the default cache operator, which IS
# ``.CA`` per the NVIDIA PTX ISA (Cache-All; routes through L1).
# nvcc's default ``-Xptxas -dlcm`` is ``ca`` on this arch.
# Treating bare LDG.E as ``LDG_E_CA`` is the architecturally
# honest mapping for sm_86. Other arches stay conservative
# (bare LDG.E -> ``OTHER``) until their default modifier is
# explicitly documented and tested per CLAUDE.md's
# architecture-coupling discipline.
_BARE_LDG_E_DEFAULTS_TO_CA: frozenset[str] = frozenset({"sm_86"})


def _match_opcode_token(
    opcode_part: str, *, arch_token: str = "",
) -> tuple[str, str] | None:
    """Match the opcode portion of a SASS line to one of the 12
    wire tokens. Returns ``(token, raw_mnemonic_head)`` on a hit,
    else ``None``.

    The ``raw_mnemonic_head`` is the matched substring (audit
    field; recorded in the manifest for downstream inspection).

    The ``arch_token`` argument resolves the bare-``LDG.E``
    ambiguity: cuobjdump on Ampere (sm_86) emits the default
    cache modifier as bare ``LDG.E`` rather than ``LDG.E.CA``,
    even though the PTX ISA default IS ``.CA``. When
    ``arch_token`` is in ``_BARE_LDG_E_DEFAULTS_TO_CA``, the
    classifier resolves bare ``LDG.E`` to ``LDG_E_CA`` (the
    architecturally honest mapping). Other arches stay
    conservative (bare ``LDG.E`` -> ``OTHER``).
    """
    if not _OPCODE_LOAD_LIKE_RE.search(opcode_part):
        return None
    for pattern, token in _RULES:
        m = pattern.search(opcode_part)
        if m is not None:
            # Arch-conditional resolution of the bare-LDG.E case.
            # The bare-LDG.E rule in _RULES matches ``LDG.E`` (no
            # cache modifier suffix) AND ``LDG`` (no .E either)
            # and maps to ``OTHER``. On sm_86 the ISA-documented
            # default modifier IS ``.CA``, so we upgrade the
            # bare-LDG.E match to ``LDG_E_CA``.
            if (
                token == "OTHER"
                and arch_token in _BARE_LDG_E_DEFAULTS_TO_CA
                and _BARE_LDG_E_RE.search(opcode_part) is not None
            ):
                # Bare ``LDG.E`` (not bare ``LDG``) on an arch
                # where ``.CA`` is the documented default modifier.
                return "LDG_E_CA", m.group(0)
            return token, m.group(0)
    # Fallback: a load-like opcode that does not match any of the
    # specific rules falls under OTHER.
    return "OTHER", ""


def _extract_kernel_section(
    sass_text: str, kernel_unmangled_name: str,
) -> tuple[str | None, str]:
    """Return ``(matched_name, section_text)``. ``matched_name``
    is the cuobjdump-emitted name whose section was extracted; it
    is ``None`` when no ``Function : <kernel>`` marker matched
    the requested kernel. ``section_text`` is the joined SASS
    lines of the matched section (empty when ``matched_name`` is
    ``None``).

    KNOWN DEFECT (2026-07-06): the loop re-arms at EVERY matching
    ``Function :`` header, so on a multi-arch fatbin the returned
    section is the concatenation of the kernel's section from
    every arch slice (cross-arch contamination); and the
    substring match (``kernel_unmangled_name in name``) can merge
    distinct kernels whose names contain the requested one. Fix
    pending in the label-pipeline backlog (see module docstring).
    """
    matched_name: str | None = None
    in_section = False
    out: list[str] = []
    for line in sass_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function :"):
            name = stripped[len("Function :"):].strip()
            in_section = (
                name == kernel_unmangled_name
                or kernel_unmangled_name in name
            )
            if in_section:
                matched_name = name
            continue
        if in_section:
            out.append(line)
    return matched_name, "\n".join(out)


def classify_sass_text(
    sass_text: str,
    *,
    kernel_unmangled_name: str,
    arch_token: str,
    binary_source: str,
    binary_path: str = "",
) -> dict[str, Any]:
    """Classify the SASS lines for a single kernel and emit a
    schema-valid manifest dict.

    Arguments:
      sass_text: the cuobjdump --dump-sass stdout for the binary.
      kernel_unmangled_name: the kernel symbol the classifier
        should restrict its parsing to. The classifier extracts
        the section between ``Function : <unmangled>`` and the
        next ``Function :`` header.
      arch_token: ``sm_86`` / ``sm_120`` / etc. Recorded for
        per-arch tolerance audits.
      binary_source: ``live_cuobjdump`` or ``canned_fixture``.
      binary_path: optional repo-relative path of the binary or
        canned-fixture identifier; recorded for provenance.

    Returns:
      A dict validating against
      ``tools/cache_predictor/cache_op_map.schema.json``.
    """
    matched_name, section_text = _extract_kernel_section(
        sass_text, kernel_unmangled_name,
    )
    if matched_name is None:
        return _typed_unavailable_manifest(
            arch_token=arch_token,
            binary_source=binary_source,
            binary_path=binary_path,
            status=SassProvenanceStatus.NO_KERNEL_SECTION,
            diagnostic=(
                f"no 'Function : {kernel_unmangled_name}' marker "
                f"in SASS text"
            ),
        )

    entries: list[OpMapEntry] = []
    saw_any_line = False
    for line in section_text.splitlines():
        if line.strip() == "":
            continue
        # Even non-instruction lines (e.g., `.headerflags`,
        # `.unified_kernel`) count toward "we saw at least one
        # line inside the section" so the empty-section
        # diagnostic does not fire incorrectly on a kernel that
        # only has metadata lines.
        saw_any_line = True
        pc, opcode_part = _strip_leading_pc_comment(line)
        if pc is None:
            # Not an instruction line (could be a directive like
            # `.headerflags` or a blank section header). Skip.
            continue
        match = _match_opcode_token(opcode_part, arch_token=arch_token)
        if match is None:
            # Non-load instruction (MOV, EXIT, FADD, etc.) — does
            # not contribute to the L1 line state model.
            continue
        token, raw_mnemonic = match
        entries.append(OpMapEntry(
            pc=pc,
            instruction_cache_op=token,
            l1_eligible=derive_l1_eligible(token),
            raw_mnemonic=raw_mnemonic,
        ))

    if not saw_any_line:
        return _typed_unavailable_manifest(
            arch_token=arch_token,
            binary_source=binary_source,
            binary_path=binary_path,
            status=SassProvenanceStatus.KERNEL_SECTION_EMPTY,
            diagnostic=(
                f"kernel section for '{matched_name}' contained no "
                f"SASS lines"
            ),
        )

    if not entries:
        # The section parsed cleanly but contained no load-class
        # opcodes. This is the AC-spec'd "shared-only kernel" or
        # "pure ALU kernel" case — SASS provenance is present and
        # parseable; there is just nothing for the L1 model to walk.
        # Distinct from KERNEL_SECTION_EMPTY (which means the
        # section had zero lines).
        out: dict[str, Any] = {
            "schema_version": 1,
            "binary_source": binary_source,
            "arch_token": arch_token,
            "kernel_mangled_name": matched_name,
            "sass_provenance_status": (
                SassProvenanceStatus.MATCHED_NO_LOAD_CLASS_INSTRUCTIONS.value
            ),
            "per_pc_instruction_cache_op": [],
            "diagnostic": (
                f"kernel section for '{matched_name}' contained no "
                f"LDG / LDGSTS / LDL / STL / ATOM / TEX class "
                f"instructions; the L1 predictor head will emit "
                f"predictor_l1_no_modeled_l1_eligible_accesses"
            ),
        }
        if binary_path:
            out["binary_path"] = binary_path
        return out

    return {
        "schema_version": 1,
        "binary_source": binary_source,
        "arch_token": arch_token,
        "kernel_mangled_name": matched_name,
        "binary_path": binary_path,
        "sass_provenance_status": SassProvenanceStatus.MATCHED.value,
        "per_pc_instruction_cache_op": [e.to_dict() for e in entries],
    }


def _typed_unavailable_manifest(
    *,
    arch_token: str,
    binary_source: str,
    binary_path: str,
    status: SassProvenanceStatus,
    diagnostic: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": 1,
        "binary_source": binary_source,
        "arch_token": arch_token,
        "sass_provenance_status": status.value,
        "per_pc_instruction_cache_op": [],
        "diagnostic": diagnostic,
    }
    if binary_path:
        out["binary_path"] = binary_path
    return out


def classify_binary(
    binary_path: Path,
    *,
    kernel_unmangled_name: str,
    arch_token: str,
) -> dict[str, Any]:
    """End-to-end: invoke ``cuobjdump --dump-sass <binary>``, parse
    the kernel's section, return a manifest. Fail-closes with
    ``sass_provenance_status=binary_not_parseable`` when cuobjdump
    cannot read the file. On multi-arch fatbins the manifest is
    cross-arch contaminated — see the module-docstring KNOWN
    DEFECT (2026-07-06)."""
    try:
        proc = subprocess.run(
            ["cuobjdump", "--dump-sass", str(binary_path)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        return _typed_unavailable_manifest(
            arch_token=arch_token,
            binary_source="live_cuobjdump",
            binary_path=str(binary_path),
            status=SassProvenanceStatus.BINARY_NOT_PARSEABLE,
            diagnostic=f"cuobjdump not on PATH: {exc}",
        )
    if proc.returncode != 0:
        return _typed_unavailable_manifest(
            arch_token=arch_token,
            binary_source="live_cuobjdump",
            binary_path=str(binary_path),
            status=SassProvenanceStatus.BINARY_NOT_PARSEABLE,
            diagnostic=(
                f"cuobjdump exited rc={proc.returncode}; "
                f"stderr_tail={proc.stderr[-512:]!r}"
            ),
        )
    return classify_sass_text(
        proc.stdout,
        kernel_unmangled_name=kernel_unmangled_name,
        arch_token=arch_token,
        binary_source="live_cuobjdump",
        binary_path=str(binary_path),
    )


# memory_op_class membership groups (per the existing
# ``MemoryOpClass`` enum in
# ``tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto``).
# Used by ``is_orthogonality_defect`` to enforce the
# memory-space-vs-cache-operator orthogonality discipline
# documented in the plan's per-PC manifest contract.
_GLOBAL_SPACE_TOKENS: frozenset[str] = frozenset({
    "LDG_E_CA", "LDG_E_CG", "LDG_E_CS", "LDG_E_EF",
    "LDG_E_STRONG_SM", "LDG_E_STRONG_GPU",
    "LDGSTS",  # async-copy SOURCE is global; staged into shared
})
_LOCAL_SPACE_TOKENS: frozenset[str] = frozenset({"LDL", "STL"})
_ATOMIC_SPACE_TOKENS: frozenset[str] = frozenset({"ATOM"})
_TEXTURE_SPACE_TOKENS: frozenset[str] = frozenset({"TEX"})


def is_orthogonality_defect(
    *, memory_op_class: str, instruction_cache_op: str,
) -> bool:
    """Return ``True`` iff the (memory_op_class, instruction_cache_op)
    pair is impossible by construction (the cache-op cannot
    legitimately appear with the named memory space).

    The two fields answer different questions: memory_op_class
    tags the memory SPACE (where the access lives); the
    instruction_cache_op tags the cache OPERATOR (how the cache
    line state is touched). They are orthogonal in principle but
    not every combination is realizable in real SASS:

      - SHARED / CLUSTER_DSMEM memory cannot pair with any
        LDG_E_* token (shared loads use LDS / LDSM, not LDG).
      - SHARED / CLUSTER_DSMEM cannot pair with LDL / STL
        (those are local-memory mnemonics).
      - SHARED / CLUSTER_DSMEM cannot pair with LDGSTS-as-LOAD
        (LDGSTS staged INTO shared, but its source is global).
      - GLOBAL cannot pair with LDL / STL (those are local
        memory mnemonics).
      - LOCAL cannot pair with LDG_E_* tokens.
      - ATOMIC and TEX accesses each have their own designated
        cache-op token.
      - OTHER is the catch-all and is admissible with any
        memory_op_class (because the classifier hasn't resolved
        the cache-op yet).

    The defect surface is the BL-20260507 lockstep-mirror
    enforcement at the producer-side wire boundary; the L1
    predictor head fails-closed when a defect is detected at
    sidecar ingress (downstream rounds wire that admission
    check).
    """
    if instruction_cache_op == "OTHER":
        # OTHER admits any memory_op_class.
        return False

    if memory_op_class in ("SHARED", "CLUSTER_DSMEM"):
        # SHARED / CLUSTER_DSMEM cannot pair with any classified
        # cache-op token in the 12-token vocabulary (shared/cluster
        # loads use LDS / LDSM mnemonics that the classifier never
        # tags with any of LDG_E_* / LDL / STL / LDGSTS / ATOM /
        # TEX).
        return True

    if memory_op_class == "GLOBAL":
        # GLOBAL is the canonical space for LDG_E_* and the
        # source side of LDGSTS; not for LDL / STL / ATOM / TEX.
        return instruction_cache_op in (
            _LOCAL_SPACE_TOKENS | _ATOMIC_SPACE_TOKENS
            | _TEXTURE_SPACE_TOKENS
        )

    if memory_op_class == "LOCAL":
        # LOCAL is LDL / STL only.
        return instruction_cache_op not in _LOCAL_SPACE_TOKENS

    if memory_op_class == "ATOMIC":
        return instruction_cache_op not in _ATOMIC_SPACE_TOKENS

    if memory_op_class == "TEXTURE":
        return instruction_cache_op not in _TEXTURE_SPACE_TOKENS

    # Unknown memory_op_class: do not fire a defect (the upstream
    # producer's memory_op_class enum is the canonical surface;
    # an unknown value is upstream's problem, not the
    # classifier's).
    return False


__all__ = [
    "PREDICTOR_L1_SASS_PROVENANCE_MISSING",
    "INSTRUCTION_CACHE_OP_TOKENS",
    "L1_ELIGIBLE_TOKENS",
    "L1_BYPASS_TOKENS",
    "SassProvenanceStatus",
    "OpMapEntry",
    "derive_l1_eligible",
    "classify_sass_text",
    "classify_binary",
    "is_orthogonality_defect",
]
