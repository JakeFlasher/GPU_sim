#!/usr/bin/env bash
#
# verify_lcd_arch_slices.sh — fatbin-slice + forbidden-opcode
# verifier for the portable_lcd suite.
#
# Asserts that a produced kernel binary's fatbin embeds the LCD
# real-arch cubin slices (sm_86 + sm_90 + sm_120) and a compute_80
# PTX fallback, that no architecture-conditional slices (sm_90a /
# sm_120a / sm_120f) leaked in, AND that the sm_120 cubin slice
# does not contain Blackwell-architecture-conditional opcodes from
# the tcgen05/TMEM family (UTC*MMA, LDT, STT) which would indicate
# an LCD-discipline violation that escaped the source-level
# allowlist.
#
# Usage:
#   verify_lcd_arch_slices.sh <path-to-binary> [<path-to-binary> ...]
#
# Exit codes:
#   0 — all binaries pass.
#   1 — at least one binary failed; typed diagnostic written to
#       stderr.
#   2 — usage error (missing argument, missing cuobjdump on PATH).
#
# Typed diagnostics:
#   lcd_arch_slice_missing: <arch>
#       The binary does NOT embed a cubin slice for one of the
#       required base archs sm_86, sm_90, sm_120.
#   lcd_arch_conditional_slice_forbidden: <arch>
#       The binary embeds a forbidden architecture-conditional cubin
#       slice (sm_90a, sm_120a, or sm_120f).
#   lcd_ptx_fallback_missing
#       The binary does NOT embed a compute_80 PTX section.
#   lcd_arch_unknown_slice: <arch>
#       The binary embeds a cubin slice for an architecture not in
#       the LCD allowlist or the documented forbidden set.
#   lcd_forbidden_sm120_sass_opcode: <mnemonic>
#       The sm_120 cubin slice contains a tcgen05/TMEM-family
#       opcode (UTC*MMA, LDT, STT) that the LCD discipline forbids.
#
# Required: cuobjdump on PATH (CUDA toolkit, version 13.x).

set -u -o pipefail

REQUIRED_ARCHS=(sm_86 sm_90 sm_120)
FORBIDDEN_ARCHS=(sm_90a sm_120a sm_120f sm_100 sm_100a sm_100f)
ALLOWED_ARCHS=(sm_86 sm_90 sm_120)
REQUIRED_PTX_ARCH=sm_80

# Forbidden Blackwell-architecture-conditional opcode regex. Each
# alternative is a complete mnemonic; instruction-line anchoring
# (via SASS_INSTR_PREFIX below) keeps these from matching against
# unrelated mnemonics that contain the same letters as a substring.
#
#   UTC[A-Z0-9]*MMA — tcgen05 tensor-core MMA family
#   LDT             — Tensor Memory load
#   STT             — Tensor Memory store
SM120_FORBIDDEN_OPCODE_RE='(UTC[A-Z0-9]*MMA|LDT|STT)'
SASS_INSTR_PREFIX='^[[:space:]]+/\*[^*]+\*/[[:space:]]+'

if [[ $# -lt 1 ]]; then
    echo "verify_lcd_arch_slices.sh: missing argument" >&2
    echo "usage: verify_lcd_arch_slices.sh <binary> [<binary> ...]" >&2
    exit 2
fi

if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "verify_lcd_arch_slices.sh: cuobjdump not on PATH" >&2
    exit 2
fi

failures=0

for bin in "$@"; do
    if [[ ! -f "$bin" ]]; then
        echo "verify_lcd_arch_slices.sh: $bin: no such file" >&2
        failures=$((failures + 1))
        continue
    fi

    elf_listing=$(cuobjdump --list-elf "$bin" 2>&1) || {
        echo "verify_lcd_arch_slices.sh: $bin: cuobjdump --list-elf failed" >&2
        echo "$elf_listing" >&2
        failures=$((failures + 1))
        continue
    }

    ptx_listing=$(cuobjdump --list-ptx "$bin" 2>&1) || {
        echo "verify_lcd_arch_slices.sh: $bin: cuobjdump --list-ptx failed" >&2
        echo "$ptx_listing" >&2
        failures=$((failures + 1))
        continue
    }

    # Extract architecture tokens from the ELF listing. cuobjdump emits lines
    # of the form: `ELF file    N: <stem>.<seq>.<arch>.cubin` — take the
    # second-to-last dot-separated field.
    archs=$(echo "$elf_listing" \
        | awk '/^ELF file/ {n=split($NF, parts, "."); if (n >= 3) print parts[n-1]}' \
        | sort -u)

    binary_failed=0

    for required in "${REQUIRED_ARCHS[@]}"; do
        if ! grep -qx "$required" <<< "$archs"; then
            echo "verify_lcd_arch_slices.sh: $bin: lcd_arch_slice_missing: $required" >&2
            binary_failed=1
        fi
    done

    for forbidden in "${FORBIDDEN_ARCHS[@]}"; do
        if grep -qx "$forbidden" <<< "$archs"; then
            echo "verify_lcd_arch_slices.sh: $bin: lcd_arch_conditional_slice_forbidden: $forbidden" >&2
            binary_failed=1
        fi
    done

    while IFS= read -r arch; do
        [[ -z "$arch" ]] && continue
        is_allowed=0
        for ok in "${ALLOWED_ARCHS[@]}"; do
            [[ "$arch" == "$ok" ]] && { is_allowed=1; break; }
        done
        is_known_forbidden=0
        for forbidden in "${FORBIDDEN_ARCHS[@]}"; do
            [[ "$arch" == "$forbidden" ]] && { is_known_forbidden=1; break; }
        done
        if [[ $is_allowed -eq 0 && $is_known_forbidden -eq 0 ]]; then
            echo "verify_lcd_arch_slices.sh: $bin: lcd_arch_unknown_slice: $arch" >&2
            binary_failed=1
        fi
    done <<< "$archs"

    # PTX section check. cuobjdump labels the compute_80 PTX as `sm_80.ptx`.
    if ! echo "$ptx_listing" | awk '/^PTX file/ {n=split($NF, parts, "."); if (n >= 3) print parts[n-1]}' | grep -qx "$REQUIRED_PTX_ARCH"; then
        echo "verify_lcd_arch_slices.sh: $bin: lcd_ptx_fallback_missing" >&2
        binary_failed=1
    fi

    # sm_120 SASS-opcode scan: dump just the sm_120 cubin slice and
    # reject any tcgen05/TMEM-family opcode. The scan is gated on the
    # sm_120 slice being present (else there's nothing to dump).
    if grep -qx sm_120 <<< "$archs"; then
        sm120_sass=$(cuobjdump --dump-sass --gpu-architecture sm_120 "$bin" 2>&1) || {
            echo "verify_lcd_arch_slices.sh: $bin: cuobjdump --dump-sass --gpu-architecture sm_120 failed" >&2
            echo "$sm120_sass" >&2
            failures=$((failures + 1))
            continue
        }
        forbidden_hits=$(echo "$sm120_sass" \
            | grep -E "${SASS_INSTR_PREFIX}${SM120_FORBIDDEN_OPCODE_RE}([[:space:].]|$)" \
            | sed -E "s|.*\\*/[[:space:]]+([A-Z0-9_]+([.][A-Z0-9_]+)*).*|\1|" \
            | sed -E 's/[.].*$//' \
            | sort -u)
        if [[ -n "$forbidden_hits" ]]; then
            while IFS= read -r mnemonic; do
                [[ -z "$mnemonic" ]] && continue
                echo "verify_lcd_arch_slices.sh: $bin: lcd_forbidden_sm120_sass_opcode: $mnemonic" >&2
            done <<< "$forbidden_hits"
            binary_failed=1
        fi
    fi

    if [[ $binary_failed -eq 1 ]]; then
        failures=$((failures + 1))
    else
        echo "verify_lcd_arch_slices.sh: $bin: OK (sm_86 sm_90 sm_120 + compute_80 PTX; sm_120 SASS clean)"
    fi
done

if [[ $failures -gt 0 ]]; then
    exit 1
fi
exit 0
