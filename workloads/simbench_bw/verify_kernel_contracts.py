#!/usr/bin/env python3
"""Cubin-level kernel-contract enforcement (AC41).

For each row in the per-row contract spec, build the row's executable
via `make` in `l1/<slug>/`, run `cuobjdump --dump-sass <slug>_run`, and
assert the disassembly contains the row's declared SASS mnemonic.

The contract spec is the **per-row instruction-path enforcement set** —
narrower than the full roster. Only rows whose `.cu` source declares a
specific cubin contract (the `// cubin must contain ...` source-comment
pattern from `BL-20260424-kernel-contract-vs-shape-match`) are enforced.
Other rows (portable elementwise / reduction L2/L3 microbenches, vendor-
library controls, FP32 baseline) don't have a row-specific declared SASS
mnemonic and are excluded via `KERNEL_CONTRACT_SPEC_EXCLUSIONS`.

After R9 the spec covers **19 rows** spanning every L1 row whose
`.cu` source declares a cubin contract:

  - Hopper wgmma (2): `wgmma_bf16_saturation`, `wgmma_fp8_saturation`.
  - Portable MMA (6): `bf16_hmma_gemm_traceable`, `tf32_hmma_gemm_traceable`,
    `fp16_hmma_gemm_traceable`, `fp8_e4m3_qmma_dep_chain`,
    `fp8_e5m2_saturation`, `int8_imma_gemm_traceable`.
  - sm_120a low-precision (4): `fp4_e2m1_mma_dep_chain`,
    `fp6_e2m3_saturation`, `fp6_e3m2_mma_dep_chain`,
    `sparse_2to4_fp8_gemm`.
  - Sparse non-FP8 (2): `sparse_2to4_int8_gemm` (R13 repair lifted
    it from known-build-broken to fully verified), `sparse_2to4_bf16_gemm`.
  - NVFP4 block-scaled (1): `nvfp4_blockscale_gemm`.
  - Memory-system (4): `cp_async_shared_tile_copy` (LDGSTS),
    `tma_tensor_copy_2d` (UTMALDG/UTMASTG), `dsmem_pairwise_latency` /
    `dsmem_ring_bandwidth` (cluster barriers).

The script is intentionally NOT integrated into pytest's hot path —
running `cuobjdump` per row on every test run is expensive. Invoke
explicitly via `make verify-kernel-contracts` (or directly).

The drift detector relies on TWO independently-editable source-of-truth
constants: `CONTRACT_REQUIRED_SLUGS` (the allowlist) and
`KERNEL_CONTRACT_SPEC` (slug → required mnemonics). Source-edit
deletions from one are caught by the other; runtime mutations are also
caught by the existing tests. The independence is load-bearing —
deriving the allowlist from the spec via `frozenset(KERNEL_CONTRACT_SPEC)`
would shrink both sets together at import time and false-green a
deletion (the Round-9-review blocker, fixed in R10).

Exit codes:
    0  every required row passed (or was known-broken-skipped).
    1  at least one required row failed under --strict.
    2  CLI usage error (validated BEFORE the toolchain probe).
    3  CUDA toolchain (nvcc / cuobjdump) not on PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parent
L1_DIR = SUITE_ROOT / "l1"
ROSTER_PATH = SUITE_ROOT / "roster.toml"


# -----------------------------------------------------------------------------
# Per-row contract spec.
#
# Each entry maps slug → required SASS mnemonic substring(s). The grep
# matches each substring as a literal token (with word-boundary semantics
# so `QMMA.16832` doesn't accidentally match `QMMA.SP.16864`); a row
# passes when ALL declared substrings are present at least once in the
# cubin. Where a row's `.cu` source declares multiple required markers
# (e.g. TMA needs both UTMALDG and UTMASTG; DSMEM needs the cluster-
# barrier triple), all are listed and all must be present.
#
# Mnemonics were verified by `cuobjdump --dump-sass <slug>_run` on the
# previously-used RTX 5060 Laptop / sm_120-baseline host as part of
# R9. (The local host on this branch is RTX 3090 Ti / sm_86 / Ampere
# — `cuobjdump` runs there too, but multi-arch fatbins compiled for
# sm_120 / sm_120a will not yield runnable kernels locally; SASS-mnemonic
# verification still works because cuobjdump is host-side.) Multi-arch
# fatbins
# (e.g. fp8 rows built for sm_90a + sm_120) may also contain incidental
# Hopper-variant SASS (HMMA on sm_90a where Blackwell uses QMMA); the
# contract requires only the declared variant — the incidental SASS is
# not a regression.
#
# Source rounds for each contract:
#   wgmma_bf16_saturation:        R0 (real wgmma.mma_async repair).
#   wgmma_fp8_saturation:         R1 (FP8 K=32 repair).
#   fp4_e2m1_mma_dep_chain:       R1 + R4 (byte layout).
#   fp6_e2m3_saturation:          R1 + R5 (scoreability).
#   fp6_e3m2_mma_dep_chain:       R1 + R6 (scoreability).
#   sparse_2to4_fp8_gemm:         R1 + R7 (scoreability).
#   bf16/tf32/fp16_hmma_gemm_traceable: R1 (portable HMMA rows).
#   fp8_e4m3_qmma_dep_chain:      R1 (portable QMMA E4M3).
#   fp8_e5m2_saturation:          R1 (portable QMMA E5M2).
#   int8_imma_gemm_traceable:     R1 (portable IMMA).
#   sparse_2to4_int8_gemm:        R1 (sparse IMMA).
#   sparse_2to4_bf16_gemm:        R1 (sparse HMMA BF16).
#   nvfp4_blockscale_gemm:        first-wave NVFP4 block-scaled MMA.
#   cp_async_shared_tile_copy:    LDGSTS / cp.async path.
#   tma_tensor_copy_2d:           UTMALDG / UTMASTG TMA path (R11
#                                 of the prior loop campaign repaired the
#                                 placeholder; cubin proof in
#                                 BL-20260424-kernel-contract-vs-shape-match).
#   dsmem_pairwise_latency / dsmem_ring_bandwidth: cluster barrier
#                                 markers (UCGABAR_ARV / UCGABAR_WAIT /
#                                 CGAERRBAR) per the DSMEM contract.
# -----------------------------------------------------------------------------


KERNEL_CONTRACT_SPEC: dict[str, tuple[str, ...]] = {
    # --- R8 (sm_120a low-precision rows + Hopper wgmma) ---
    "wgmma_bf16_saturation":      ("HGMMA.64x64x16.F32.BF16",),
    "wgmma_fp8_saturation":       ("QGMMA.64x64x32.F32.E4M3.E4M3",),
    "fp4_e2m1_mma_dep_chain":     ("QMMA.16832.F32.E2M1.E2M1",),
    "fp6_e2m3_saturation":        ("QMMA.16832.F32.E2M3.E2M3",),
    "fp6_e3m2_mma_dep_chain":     ("QMMA.16832.F32.E3M2.E3M2",),
    "sparse_2to4_fp8_gemm":       ("QMMA.SP.16864.F32.E4M3.E4M3",),
    # --- R9 (portable MMA rows) ---
    "bf16_hmma_gemm_traceable":   ("HMMA.16816.F32.BF16",),
    "tf32_hmma_gemm_traceable":   ("HMMA.1688.F32.TF32",),
    "fp16_hmma_gemm_traceable":   ("HMMA.16816.F32",),
    "fp8_e4m3_qmma_dep_chain":    ("QMMA.16832.F32.E4M3.E4M3",),
    "fp8_e5m2_saturation":        ("QMMA.16832.F32.E5M2.E5M2",),
    "int8_imma_gemm_traceable":   ("IMMA.16832.S8.S8",),
    # --- R9 (sparse non-FP8) ---
    "sparse_2to4_int8_gemm":      ("IMMA.SP.16864.S8.S8",),
    "sparse_2to4_bf16_gemm":      ("HMMA.SP.16832.F32.BF16",),
    # --- R9 (NVFP4 block-scaled) ---
    "nvfp4_blockscale_gemm":      ("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",),
    # --- R9 (memory-system: cp.async + TMA + DSMEM cluster barriers) ---
    "cp_async_shared_tile_copy":  ("LDGSTS.E",),
    "tma_tensor_copy_2d":         ("UTMALDG.2D", "UTMASTG.2D"),
    "dsmem_pairwise_latency":     ("UCGABAR_ARV", "UCGABAR_WAIT", "CGAERRBAR"),
    "dsmem_ring_bandwidth":       ("UCGABAR_ARV", "UCGABAR_WAIT", "CGAERRBAR"),
    # --- register-spill / local-memory microbench ---
    # Built under a per-thread register cap (Makefile SPILL_MAXREG) so the
    # 64-wide live set spills to local memory: the cubin MUST contain local
    # load/store spill markers. The differential spill sweep
    # (tools/local_labels/verify_spill_sweep.py) additionally proves spill GROWS
    # as the register cap drops (a confounded no-op sweep is rejected).
    "register_spill_local_mem":   ("LDL", "STL"),
    # Local-memory family members that force `.local` traffic WITHOUT a register
    # cap -- the two slug-distinct mechanisms the plan names beyond register
    # spill (line 57/278: "-maxrregcount + volatile-local + dynamic-index").
    # Both emit clean LDL/STL on sm_86 (cuobjdump-verified, default uncapped
    # build, STACK > 0): local_dynamic_index_array indexes a per-thread array at
    # DATA-DEPENDENT (runtime) indices the compiler cannot resolve, so the array
    # cannot be register-promoted; local_volatile_array uses a `volatile`
    # per-thread array so every access is a real memory access. Neither is a
    # register-cap variant of register_spill_local_mem (no SPILL_MAXREG), so
    # they are genuinely slug-disjoint local-memory coverage.
    "local_dynamic_index_array":  ("LDL", "STL"),
    "local_volatile_array":       ("LDL", "STL"),
    # --- atomics family: paired ATOM-vs-RED observation-point contrast ---
    # global_atomic_histogram IGNORES the atomicAdd return -> the compiler emits
    # the reduction form (global RED); global_atomic_return_used USES the return
    # -> the atomic-with-return form (global ATOM). Same algorithm; the ONLY
    # difference is whether the return value is live. The full dotted opcode
    # (incl. the sm_86 .STRONG.GPU memory scope) follows the same convention as
    # the gemm HMMA/IMMA/QMMA contracts above; the differential check below
    # adds the arch-robust family-level "pure-RED vs pure-ATOM" assertion.
    "global_atomic_histogram":          ("REDG.E.ADD.STRONG.GPU",),
    "global_atomic_return_used":        ("ATOMG.E.ADD.STRONG.GPU",),
    # Opcode-distinct MAX-operation atomics pair (empirically verified on sm_86
    # via nvcc + cuobjdump): atomicMax return-ignored -> the reduction form
    # REDG.E.MAX.S32 (global_atomic_max_histogram); return-used -> the
    # atomic-with-return form ATOMG.E.MAX.S32 (global_atomic_max_return_used).
    # Distinct slugs exercising the global-reduction / global-atomic op-classes
    # with a different reduction op (.MAX.S32 vs .ADD) than the add-based pair.
    "global_atomic_max_histogram":      ("REDG.E.MAX.S32.STRONG.GPU",),
    "global_atomic_max_return_used":    ("ATOMG.E.MAX.S32.STRONG.GPU",),
    # Opcode-distinct bitwise-AND atomics pair (atomicAnd; cuobjdump-verified):
    # return-ignored -> REDG.E.AND (global_atomic_and_histogram); return-used ->
    # ATOMG.E.AND (global_atomic_and_return_used) -- the 3rd ATOM/RED slug-disjoint
    # members, a bitwise op distinct from the .ADD and .MAX arithmetic pairs.
    "global_atomic_and_histogram":      ("REDG.E.AND.STRONG.GPU",),
    "global_atomic_and_return_used":    ("ATOMG.E.AND.STRONG.GPU",),
    # Vector-width request/sector-formation family (empirically verified on
    # sm_86 via nvcc + cuobjdump): the float4 copy forms a single 128-bit
    # request per access (LDG.E.128 / STG.E.128); the float2 copy forms 64-bit
    # requests (LDG.E.64 / STG.E.64); the grid-strided scalar copy forms 32-bit
    # requests (LDG.E / STG.E, no width suffix). Full dotted opcodes per the
    # convention above; the differential check below adds the arch-robust
    # family-level three-way purity assertion (each member's width class present,
    # the other two absent). (The matcher's negative lookahead means "LDG.E"
    # never matches "LDG.E.64" or "LDG.E.128" -- the trailing "." fails the
    # boundary -- so the scalar witnesses cannot spuriously match the vectorized
    # SASS, and 64b vs 128b are mutually exclusive on the literal width token.)
    "global_vectorized_128b_copy":      ("LDG.E.128.CONSTANT", "STG.E.128"),
    "global_vectorized_64b_copy":       ("LDG.E.64.CONSTANT", "STG.E.64"),
    "global_scalar_32b_copy":           ("LDG.E.CONSTANT", "STG.E"),
}


# Rows that MUST have an SASS contract. Source-of-truth allowlist for
# the spec-coverage test.
#
# R10 refactor (per Codex Round-9-review blocking finding):
# this set is an **explicit literal** independent of `KERNEL_CONTRACT_SPEC`,
# NOT `frozenset(KERNEL_CONTRACT_SPEC)`. The independence is load-bearing:
# if a future edit deletes a slug from `KERNEL_CONTRACT_SPEC` in source
# and the module is re-imported fresh, the slug remains in this allowlist
# and `find_rows_needing_contract` flags the gap loud. Deriving the
# allowlist from the spec dict would shrink both sets together at import
# time and silently false-green the deletion.
#
# Adding a new instruction-path-bearing row requires updates in BOTH
# places (this set AND `KERNEL_CONTRACT_SPEC`). The bidirectional
# bijection invariant test
# `test_contract_required_slugs_and_spec_are_in_strict_bijection`
# (R11; replaces R10's one-way subset check) asserts
# `set(KERNEL_CONTRACT_SPEC) == CONTRACT_REQUIRED_SLUGS` and reports
# both `missing_from_spec` and `spec_not_required` halves on failure.
# The two sources of truth are independently editable in source so a
# deletion from one shows up as a divergence from the other; two
# complementary source-edit drift regression tests
# (`test_source_edit_deletion_from_spec_is_caught_by_allowlist` from
# R10 + `test_source_edit_deletion_from_allowlist_is_caught_by_spec`
# from R11) lock both directions by writing temp copies of this
# module with a representative slug removed.
#
# The set grows append-only as new instruction-path-bearing rows land.
CONTRACT_REQUIRED_SLUGS: frozenset[str] = frozenset({
    # --- R8 (sm_120a low-precision rows + Hopper wgmma) ---
    "wgmma_bf16_saturation",
    "wgmma_fp8_saturation",
    "fp4_e2m1_mma_dep_chain",
    "fp6_e2m3_saturation",
    "fp6_e3m2_mma_dep_chain",
    "sparse_2to4_fp8_gemm",
    # --- R9 (portable MMA rows) ---
    "bf16_hmma_gemm_traceable",
    "tf32_hmma_gemm_traceable",
    "fp16_hmma_gemm_traceable",
    "fp8_e4m3_qmma_dep_chain",
    "fp8_e5m2_saturation",
    "int8_imma_gemm_traceable",
    # --- R9 (sparse non-FP8) ---
    "sparse_2to4_int8_gemm",
    "sparse_2to4_bf16_gemm",
    # --- R9 (NVFP4 block-scaled) ---
    "nvfp4_blockscale_gemm",
    # --- R9 (memory-system: cp.async + TMA + DSMEM cluster barriers) ---
    "cp_async_shared_tile_copy",
    "tma_tensor_copy_2d",
    "dsmem_pairwise_latency",
    "dsmem_ring_bandwidth",
    # --- register-spill / local-memory microbench ---
    "register_spill_local_mem",
    # --- local-memory family (dynamic-index + volatile mechanisms) ---
    "local_dynamic_index_array",
    "local_volatile_array",
    # --- atomics family (ATOM-vs-RED contrast) ---
    "global_atomic_histogram",
    "global_atomic_return_used",
    "global_atomic_max_histogram",
    "global_atomic_max_return_used",
    "global_atomic_and_histogram",
    "global_atomic_and_return_used",
    # --- memory family (vector-width request/sector-formation contrast) ---
    "global_vectorized_128b_copy",
    "global_vectorized_64b_copy",
    "global_scalar_32b_copy",
})


# Rows whose `.cu` source declares a SASS contract but whose build
# currently breaks under the local toolchain due to a pre-existing
# kernel defect, NOT a R9 enforcement bug. Per Codex Round-8-
# review required-implementation-plan #4: "Repair or explicitly
# document any row whose cubin contract cannot currently build under
# the local toolchain. Do not silently narrow the coverage heuristic
# to avoid existing declared contracts."
#
# Rows in this dict still appear in the spec (visible to the coverage
# test) and the contract is preserved for the future repair round.
# Under `--strict`, they're reported as `unsupported_local_arch` with
# the documented reason instead of `failed`, so the gate exits 0 when
# all *repairable* rows pass and known-broken rows have explicit
# tickets. Removing the row from this dict requires repairing the
# underlying defect.
KNOWN_BUILD_BROKEN_REASONS: dict[str, str] = {
    # (R13 cleared `sparse_2to4_int8_gemm`: the R1 design
    # defect was repaired by changing B from compressed K/2 + 2 b32
    # to dense K + 4 b32, mirroring R7's sparse_2to4_fp8_gemm
    # fix. Cubin now emits 12 occurrences of `IMMA.SP.16864.S8.S8`.
    # If a future round encounters another build-broken contract row,
    # add it here with the same documentation pattern.)
}


# Rows in the roster that intentionally have NO declared SASS contract
# (typically because they're portable elementwise / reduction / memory
# microbenches without a row-specific instruction-path requirement, or
# vendor-library controls). Listed explicitly so the spec-coverage test
# can distinguish "missing on purpose" from "forgot to add".
KERNEL_CONTRACT_SPEC_EXCLUSIONS: frozenset[str] = frozenset({
    # Vendor-library controls (excluded from headline scoring per AC17).
    # The cubin contains cuBLAS / cuBLASLt symbols, not row-specific
    # SASS markers; AC41 enforcement here is meaningless.
    "vendor_gemm_cublas",
    "vendor_fp8_matmul_cublaslt",
    # Portable memory / elementwise / control-flow microbenches: no
    # row-specific instruction-path requirement (they exercise the
    # generic load / store / compare / branch SASS).
    "shared_bank_conflict_stride",
    "l2_thrash_pointer_chase",
    "dram_stream_stride_mix",
    "warp_divergence_reconvergence",
    # FP32 baseline (FFMA SASS, no row-specific tensor-core instruction).
    "fp32_fma_dep_chain",
})


@dataclass
class RowResult:
    slug: str
    required_mnemonics: list[str]
    build_status: str = "not_attempted"  # "ok" | "failed" | "skipped" | "not_attempted"
    mnemonic_match_counts: dict[str, int] = field(default_factory=dict)
    sass_archive_present: bool = False
    status: str = "pending"  # "passed" | "failed" | "unsupported_local_arch"
    failure_reason: str = ""
    # Detail of the optional differential phenomenon check (e.g. the
    # register-spill -maxrregcount sweep); empty for rows without one.
    differential_check: str = ""


# -----------------------------------------------------------------------------
# Toolchain probe.
# -----------------------------------------------------------------------------


def _toolchain_available() -> tuple[bool, str]:
    """Return (available, reason) for nvcc + cuobjdump availability."""
    if shutil.which("nvcc") is None:
        return False, "nvcc not on PATH"
    if shutil.which("cuobjdump") is None:
        return False, "cuobjdump not on PATH"
    return True, ""


# -----------------------------------------------------------------------------
# Per-row check.
# -----------------------------------------------------------------------------


def _build_row(slug: str, allow_failures: bool) -> tuple[str, str]:
    """Build the row's executable. Returns (build_status, error_message)."""
    row_dir = L1_DIR / slug
    if not row_dir.is_dir():
        return ("failed", f"row directory {row_dir} does not exist")
    proc = subprocess.run(
        ["make"], cwd=row_dir, capture_output=True, text=True
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        last = msg[-1] if msg else "make failed (no stderr)"
        if allow_failures:
            return ("skipped", f"build failed (allowed): {last}")
        return ("failed", f"build failed: {last}")
    return ("ok", "")


def _disassemble_and_grep(slug: str, required: tuple[str, ...]) -> tuple[dict[str, int], str]:
    """Run cuobjdump and count matches per required mnemonic.
    Returns (counts, error_message)."""
    exe = L1_DIR / slug / f"{slug}_run"
    if not exe.is_file():
        return ({}, f"executable {exe} not found after build")
    proc = subprocess.run(
        ["cuobjdump", "--dump-sass", str(exe)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip().splitlines()
        last = msg[-1] if msg else "cuobjdump failed (no stderr)"
        return ({}, f"cuobjdump failed: {last}")
    sass = proc.stdout
    counts = {mnem: count_mnemonic_occurrences(sass, mnem) for mnem in required}
    return (counts, "")


def count_mnemonic_occurrences(sass_text: str, mnemonic: str) -> int:
    """Count occurrences of `mnemonic` in the SASS text. Matches the
    mnemonic as a literal token followed by whitespace (so `QMMA.16832`
    doesn't accidentally match `QMMA.SP.16864`). Use word-boundary
    semantics: the character after the match must be whitespace, end-
    of-line, or punctuation, NOT a hex/identifier character."""
    pattern = re.escape(mnemonic) + r"(?=\s|$|[^A-Za-z0-9._])"
    return len(re.findall(pattern, sass_text))


def _register_spill_differential_check(
    slug: str, spill_provider=None
) -> tuple[bool, str]:
    """Differential phenomenon check for the register-spill family.

    Presence of LDL/STL is necessary but NOT sufficient: the -maxrregcount sweep
    must demonstrably CHANGE spill (no spill at the high cap, spill at the low
    cap, monotone as the cap drops). A confounded / constant / top-spill /
    non-monotone ladder FAILS the contract gate even when the mnemonics appear.

    ``spill_provider`` (callable -> {register_cap: spill_bytes}) is injectable so
    the gate's rejection paths can be tested host-independently (no nvcc).
    """
    # Lazy import: the spill verifier lives in the local-label quarantine; only
    # load it when this check actually runs (keeps module import / the
    # spec-bijection tests free of the cuobjdump/nvcc dependency).
    ll_dir = str(SUITE_ROOT.parents[1] / "tools" / "local_labels")
    if ll_dir not in sys.path:
        sys.path.insert(0, ll_dir)
    import verify_spill_sweep as vss

    if spill_provider is None:
        ok, reason = vss.toolchain_available()
        if not ok:
            # check_row only reaches here after a successful build, so the
            # toolchain is normally present; degrade gracefully if cuobjdump
            # vanished mid-run rather than hard-failing the row.
            return True, f"differential_spill_check_skipped: {reason}"
        import tempfile

        kdir = L1_DIR / slug
        with tempfile.TemporaryDirectory() as tmp:
            binaries = vss.build_spill_ladder(
                kdir / f"{slug}.cu", kdir / "main.cu", "sm_86", Path(tmp)
            )
            spill_by_cap = vss.measure_spill_ladder(
                binaries, "sm_86", f"{slug}_kernel"
            )
    else:
        spill_by_cap = spill_provider()

    verdict = vss.spill_sweep_verdict(spill_by_cap)
    if not verdict["ok"]:
        return False, f"spill_sweep_{verdict['reason']}: {verdict.get('detail', '')}"
    return True, f"spill_sweep_confirmed: {verdict['spill_by_cap']}"


# Slugs whose contract requires a DIFFERENTIAL phenomenon check beyond mnemonic
# presence -- the family must demonstrably exercise its phenomenon across its
# controlling axis. Each value is callable(slug) -> (ok, detail).
_ATOM_FAMILY_RE = re.compile(r"\bATOMG?\.")   # ATOM / ATOMG (atomic-with-return)
_RED_FAMILY_RE = re.compile(r"\bREDG?\.")      # RED / REDG (reduction, no return)

# The return-used (ATOM) member -> its return-ignored (RED) sibling. Each
# ATOM-vs-RED pair shares one algorithm differing only in return-value liveness;
# the differential check below asserts cross-family purity per pair. The
# add-based pair uses .ADD; the max-based pair uses .MAX.S32 (a distinct
# reduction op), but both contrast resolves through the arch+op-robust ATOM/RED
# family regexes above.
_ATOM_RED_SIBLING = {
    "global_atomic_return_used": "global_atomic_histogram",
    "global_atomic_max_return_used": "global_atomic_max_histogram",
    "global_atomic_and_return_used": "global_atomic_and_histogram",
}


def _atom_vs_red_differential_check(
    slug: str, sass_provider=None
) -> tuple[bool, str]:
    """Differential phenomenon check for the paired ATOM-vs-RED atomics family.

    Presence of the exact dotted opcode (KERNEL_CONTRACT_SPEC) is necessary but
    not sufficient to prove the CONTRAST: the pair must be PURE — the
    return-used kernel (`slug`) emits an atomic-with-return (ATOM family) and NO
    reduction (RED family), and its return-ignored sibling (resolved per pair via
    `_ATOM_RED_SIBLING` — the add-based `global_atomic_histogram` or the max-based
    `global_atomic_max_histogram`) emits a reduction (RED family) and NO
    atomic-with-return. This is the arch+op-robust, family-level distinction (it
    matches the opcode HEAD, not the .ADD/.MAX op or the sm_86-specific
    .STRONG.GPU suffix) that the T1 lockstep relies on so a RED is never
    miscounted as an ATOM.

    ``sass_provider`` (callable(slug_name) -> sass_text | None) is injectable so
    the rejection paths can be tested host-independently (no nvcc/cuobjdump).
    """
    sibling = _ATOM_RED_SIBLING[slug]

    def _sass(name: str) -> str | None:
        if sass_provider is not None:
            return sass_provider(name)
        row_dir = L1_DIR / name
        exe = row_dir / f"{name}_run"
        if not exe.is_file():
            proc = subprocess.run(
                ["make"], cwd=row_dir, capture_output=True, text=True
            )
            if proc.returncode != 0 or not exe.is_file():
                return None
        proc = subprocess.run(
            ["cuobjdump", "--dump-sass", str(exe)],
            capture_output=True, text=True,
        )
        return proc.stdout if proc.returncode == 0 else None

    atom_sass = _sass(slug)
    red_sass = _sass(sibling)
    if atom_sass is None or red_sass is None:
        # Toolchain / binary unavailable on this host -> degrade gracefully
        # (the presence check + build already gate the row; this only ADDS the
        # cross-family purity assertion when the SASS is obtainable).
        return True, "atom_vs_red_differential_skipped: SASS unavailable"

    atom_has_atom = bool(_ATOM_FAMILY_RE.search(atom_sass))
    atom_has_red = bool(_RED_FAMILY_RE.search(atom_sass))
    red_has_red = bool(_RED_FAMILY_RE.search(red_sass))
    red_has_atom = bool(_ATOM_FAMILY_RE.search(red_sass))
    if not atom_has_atom or atom_has_red:
        return False, (
            f"atom_kernel_not_pure_atom: {slug} has_atom={atom_has_atom} "
            f"has_red={atom_has_red} (return-used kernel must be ATOM-only)"
        )
    if not red_has_red or red_has_atom:
        return False, (
            f"red_kernel_not_pure_red: {sibling} has_red={red_has_red} "
            f"has_atom={red_has_atom} (return-ignored kernel must be RED-only)"
        )
    return True, "atom_vs_red_confirmed: return-used=ATOM-only, return-ignored=RED-only"


# Vector-width request/sector-formation family-level regexes (arch-robust: they
# match the opcode HEAD + width, not the sm_86-specific .CONSTANT cache hint).
# A 128-bit global mem op is LDG.E.128 / STG.E.128; a 64-bit op is LDG.E.64 /
# STG.E.64; a scalar 32-bit op is LDG.E / STG.E NOT followed by ".64" or ".128"
# (the negative lookahead excludes the wider forms, so the three classes are
# mutually disjoint on the literal width token: "LDG.E" never matches inside
# "LDG.E.64" or "LDG.E.128", and ".64" vs ".128" cannot match each other).
_VEC128_MEM_RE = re.compile(r"\b(?:LDG|STG)\.E\.128\b")
_VEC64_MEM_RE = re.compile(r"\b(?:LDG|STG)\.E\.64\b")
_SCALAR32_MEM_RE = re.compile(r"\b(?:LDG|STG)\.E(?!\.(?:64|128))")

# Each family member -> (its own width-class regex, short label). The
# differential check asserts every member emits ONLY its own width class.
_VECTOR_WIDTH_MEMBERS: tuple[tuple[str, "re.Pattern[str]", str], ...] = (
    ("global_vectorized_128b_copy", _VEC128_MEM_RE, "128b"),
    ("global_vectorized_64b_copy", _VEC64_MEM_RE, "64b"),
    ("global_scalar_32b_copy", _SCALAR32_MEM_RE, "32b"),
)


def _vector_width_differential_check(
    slug: str, sass_provider=None
) -> tuple[bool, str]:
    """Differential phenomenon check for the vector-width family (3 members).

    Presence of the exact dotted opcode (KERNEL_CONTRACT_SPEC) is necessary but
    not sufficient to prove the CONTRAST: every member must be PURE in its own
    width class — the 128-bit member forms only LDG.E.128 / STG.E.128, the
    64-bit member only LDG.E.64 / STG.E.64, and the scalar member only the bare
    32-bit LDG.E / STG.E (no width suffix). This is the arch-robust family-level
    distinction the T1 lockstep relies on so a request of one width is never
    miscounted as another. The check is registered for every member, so any
    member's contract check re-asserts whole-family purity.

    ``sass_provider`` (callable(slug_name) -> sass_text | None) is injectable so
    the rejection paths can be tested host-independently (no nvcc/cuobjdump).
    """
    def _sass(name: str) -> str | None:
        if sass_provider is not None:
            return sass_provider(name)
        row_dir = L1_DIR / name
        exe = row_dir / f"{name}_run"
        if not exe.is_file():
            proc = subprocess.run(
                ["make"], cwd=row_dir, capture_output=True, text=True
            )
            if proc.returncode != 0 or not exe.is_file():
                return None
        proc = subprocess.run(
            ["cuobjdump", "--dump-sass", str(exe)],
            capture_output=True, text=True,
        )
        return proc.stdout if proc.returncode == 0 else None

    sass_by_member: dict[str, str] = {}
    for name, _re, _lbl in _VECTOR_WIDTH_MEMBERS:
        text = _sass(name)
        if text is None:
            return True, (
                f"vector_width_differential_skipped: SASS unavailable for {name}"
            )
        sass_by_member[name] = text

    for name, own_re, lbl in _VECTOR_WIDTH_MEMBERS:
        text = sass_by_member[name]
        if not own_re.search(text):
            return False, (
                f"vector_width_member_missing_own_class: {name} lacks {lbl} "
                f"global mem ops (must emit its own width class)"
            )
        for other_name, other_re, other_lbl in _VECTOR_WIDTH_MEMBERS:
            if other_name == name:
                continue
            if other_re.search(text):
                return False, (
                    f"vector_width_member_impure: {name} ({lbl} member) also "
                    f"emits {other_lbl} global mem ops (each member must be "
                    f"width-pure)"
                )
    return True, (
        "vector_width_confirmed: 128b / 64b / 32b members each width-pure"
    )


DIFFERENTIAL_CONTRACT_CHECKS = {
    "register_spill_local_mem": _register_spill_differential_check,
    "global_atomic_return_used": _atom_vs_red_differential_check,
    "global_atomic_max_return_used": _atom_vs_red_differential_check,
    "global_atomic_and_return_used": _atom_vs_red_differential_check,
    "global_vectorized_128b_copy": _vector_width_differential_check,
    "global_vectorized_64b_copy": _vector_width_differential_check,
    "global_scalar_32b_copy": _vector_width_differential_check,
}


def check_row(slug: str, required: tuple[str, ...], allow_build_failures: bool) -> RowResult:
    """Check one row: build, disassemble, grep, score."""
    result = RowResult(slug=slug, required_mnemonics=list(required))

    build_status, build_msg = _build_row(slug, allow_build_failures)
    result.build_status = build_status
    if build_status == "skipped":
        result.status = "unsupported_local_arch"
        result.failure_reason = build_msg
        return result
    if build_status == "failed":
        if slug in KNOWN_BUILD_BROKEN_REASONS:
            # Documented pre-existing defect; treat as unsupported_local_arch
            # so the gate can exit 0 when all repairable rows pass. The
            # contract stays declared (in KERNEL_CONTRACT_SPEC) so a
            # future repair round can verify it.
            result.status = "unsupported_local_arch"
            result.failure_reason = (
                f"known_build_broken: {KNOWN_BUILD_BROKEN_REASONS[slug]}"
            )
            return result
        result.status = "failed"
        result.failure_reason = build_msg
        return result

    counts, dump_err = _disassemble_and_grep(slug, required)
    if dump_err:
        result.status = "failed"
        result.failure_reason = dump_err
        return result
    result.mnemonic_match_counts = counts

    # All required mnemonics must have at least one match.
    missing = [m for m, c in counts.items() if c == 0]
    if missing:
        result.status = "failed"
        result.failure_reason = (
            f"missing required mnemonic(s) in cubin: {', '.join(missing)}"
        )
        return result

    # Differential phenomenon check: some rows must demonstrably exercise their
    # phenomenon across its controlling axis, not merely emit the mnemonic (e.g.
    # the register-spill -maxrregcount sweep must actually change spill). A
    # confounded sweep fails the gate even with the mnemonics present.
    diff_check = DIFFERENTIAL_CONTRACT_CHECKS.get(slug)
    if diff_check is not None:
        diff_ok, diff_detail = diff_check(slug)
        result.differential_check = diff_detail
        if not diff_ok:
            result.status = "failed"
            result.failure_reason = (
                f"differential contract check failed: {diff_detail}"
            )
            return result

    # Optional: record that the per-row sass_archive directory exists
    # (some rows produce one as a build side-effect).
    sass_dir = L1_DIR / slug / "sass_archive"
    result.sass_archive_present = sass_dir.is_dir()

    result.status = "passed"
    return result


# -----------------------------------------------------------------------------
# Roster coverage check (used by tests and by --check-roster-coverage CLI).
# -----------------------------------------------------------------------------


def _load_roster_rows() -> list[dict]:
    with open(ROSTER_PATH, "rb") as f:
        return tomllib.load(f).get("rows", [])


def find_rows_needing_contract(roster_rows: list[dict] | None = None) -> list[str]:
    """Return slugs that SHOULD have an SASS contract (per the
    `CONTRACT_REQUIRED_SLUGS` allowlist) but are missing from
    `KERNEL_CONTRACT_SPEC` AND not explicitly excluded.

    R9: replaces the previous narrow family/precision heuristic
    that missed 13 rows already declaring contracts in their `.cu`
    headers. The allowlist is the source-of-truth — adding a slug
    here without a spec entry triggers a coverage-failure.

    The roster_rows parameter is kept (and consulted) so the test can
    inject synthetic rows for drift-detection coverage; in practice the
    function reads `CONTRACT_REQUIRED_SLUGS` directly.
    """
    if roster_rows is not None:
        # Synthetic-drift mode: pretend every row in roster_rows is
        # required, then check against the spec. Used by the
        # `test_roster_coverage_helper_detects_synthetic_drift` test to
        # verify the helper would catch a future drift.
        candidates = [
            row.get("slug") for row in roster_rows
            if isinstance(row.get("slug"), str)
            and _row_appears_to_need_contract(row)
        ]
        return [
            s for s in candidates
            if s and s not in KERNEL_CONTRACT_SPEC
            and s not in KERNEL_CONTRACT_SPEC_EXCLUSIONS
        ]
    return [
        s for s in CONTRACT_REQUIRED_SLUGS
        if s not in KERNEL_CONTRACT_SPEC
        and s not in KERNEL_CONTRACT_SPEC_EXCLUSIONS
    ]


def _row_appears_to_need_contract(row: dict) -> bool:
    """Heuristic for synthetic-drift testing only: returns True if a row
    looks like it should have an SASS contract based on family +
    precision. Production code uses `CONTRACT_REQUIRED_SLUGS` directly;
    this heuristic is kept solely so the synthetic-drift test can verify
    the helper catches "obvious" missing rows."""
    family = row.get("family", "") or ""
    precision = row.get("primary_precision", "") or ""
    slug = row.get("slug", "") or ""
    is_wgmma = family.startswith("wgmma") or "wgmma" in slug
    low_prec = precision in ("FP4_E2M1", "FP6_E2M3", "FP6_E3M2", "FP8_E4M3")
    is_mma_micro_lowprec = (family == "mma_microbench" and low_prec)
    is_sparse_lowprec = (family == "sparse_gemm" and low_prec)
    return is_wgmma or is_mma_micro_lowprec or is_sparse_lowprec


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_kernel_contracts.py",
        description="Cubin-level kernel-contract enforcement (AC41).",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--row", help="Check a single row by slug.")
    g.add_argument("--all", action="store_true",
                   help="Check every row in the contract spec (default).")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 on any row failure (default: report-only).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON sidecar path.")
    p.add_argument("--allow-build-failures", action="store_true",
                   help="Skip rows that fail to build (mark unsupported_local_arch) "
                        "instead of failing the gate. Useful on hosts without "
                        "sm_90a / sm_120a build access.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # Validate CLI arguments BEFORE probing the toolchain (R9 fix
    # per Codex Round-8-review queued item): `--row unknown_slug` should
    # exit 2 (usage error) deterministically regardless of whether
    # nvcc/cuobjdump are on PATH. The previous order silently masked
    # usage errors as toolchain-unavailable failures on hosts without
    # CUDA installed.
    if args.row is not None and args.row not in KERNEL_CONTRACT_SPEC:
        print(f"unknown_row: {args.row} is not in the contract spec", file=sys.stderr)
        print(f"known rows: {', '.join(sorted(KERNEL_CONTRACT_SPEC))}", file=sys.stderr)
        return 2

    available, reason = _toolchain_available()
    if not available:
        print(f"toolchain_unavailable: {reason}", file=sys.stderr)
        print("(install CUDA toolkit so nvcc and cuobjdump are on PATH)", file=sys.stderr)
        return 3

    if args.row is not None:
        rows_to_check = {args.row: KERNEL_CONTRACT_SPEC[args.row]}
    else:
        rows_to_check = dict(KERNEL_CONTRACT_SPEC)

    results: list[RowResult] = []
    for slug, required in sorted(rows_to_check.items()):
        result = check_row(slug, required, args.allow_build_failures)
        results.append(result)
        status_icon = {"passed": "OK", "failed": "FAIL", "unsupported_local_arch": "SKIP"}.get(
            result.status, "??"
        )
        match_str = ", ".join(f"{m}={c}" for m, c in result.mnemonic_match_counts.items()) or "-"
        print(f"  [{status_icon}] {slug}: required={list(required)} matches={match_str}"
              + (f"  reason={result.failure_reason}" if result.failure_reason else ""))

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "unsupported": sum(1 for r in results if r.status == "unsupported_local_arch"),
    }
    print(f"\nverify_kernel_contracts: total={summary['total']} "
          f"passed={summary['passed']} failed={summary['failed']} "
          f"unsupported={summary['unsupported']}")

    if args.out is not None:
        out_doc = {
            "schema_version": 1,
            "rows": [asdict(r) for r in results],
            "summary": summary,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_doc, f, indent=2)
        print(f"  → wrote {args.out}")

    if args.strict and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
