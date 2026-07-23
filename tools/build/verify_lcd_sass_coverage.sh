#!/usr/bin/env bash
#
# verify_lcd_sass_coverage.sh — SASS-mnemonic-coverage verifier for
# the portable_lcd suite.
#
# Asserts that the SASS emitted across one or more kernel binaries
# COLLECTIVELY exercises:
#
#   CTA-barrier vs memory-fence distinction:
#     * At least one binary emits a TRUE BAR.SYNC mnemonic
#       (the __syncthreads() lowering — CTA-level synchronization
#       barrier).
#     * At least one binary emits a memory fence from the family
#       MEMBAR / ERRBAR / DEPBAR / LDGDEPBAR (NOT a sync barrier).
#     * The split serves as the regression workload for the
#       phase-injection's `is_bar` substring-match bug, which would
#       conflate MEMBAR with BAR.SYNC under naive substring matching.
#
#   Branch / return SASS coverage:
#     * Each of BSSY, BSYNC, EXIT, RET appears in at least one
#       binary's SASS.
#     * At least one of WARPSYNC, CALL, JMX appears in at least one
#       binary's SASS.
#     * Serves as the regression workload for the phase-injection's
#       `is_branch` and `is_ret` substring-match bugs.
#
# Both expectations are SUITE-level, not per-binary. The verifier
# aggregates SASS across all input binaries and reports the union of
# mnemonics observed.
#
# Usage:
#   verify_lcd_sass_coverage.sh <path-to-binary> [<path-to-binary> ...]
#
# Exit codes:
#   0 — collective SASS satisfies both expectations.
#   1 — collective SASS misses at least one mnemonic; typed diagnostic
#       written to stderr.
#   2 — usage error (missing argument, missing cuobjdump).
#
# Typed diagnostics:
#   bar_sync_coverage_missing — no binary emits a true BAR.SYNC.
#   membar_family_coverage_missing — no binary emits a memory fence
#       (MEMBAR / ERRBAR / DEPBAR / LDGDEPBAR).
#   bssy_coverage_missing, bsync_coverage_missing,
#   exit_coverage_missing, ret_coverage_missing — the named mnemonic
#       is absent across the suite.
#   warp_branch_coverage_missing — none of WARPSYNC / CALL / JMX
#       present.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
    echo "verify_lcd_sass_coverage.sh: missing argument" >&2
    echo "usage: verify_lcd_sass_coverage.sh <binary> [<binary> ...]" >&2
    exit 2
fi

if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "verify_lcd_sass_coverage.sh: cuobjdump not on PATH" >&2
    exit 2
fi

# The pattern matches a SASS instruction line of the form:
#   /* 0x... */    OPCODE ...
# where OPCODE is the first whitespace-delimited token after the
# closing comment marker. The leading `^\s+/\*[^*]+\*/\s+` anchor
# ensures we're matching an instruction line and not a string
# literal that happens to contain the mnemonic.
SASS_INSTR_PREFIX='^\s+/\*[^*]+\*/\s+'

BAR_SYNC_RE="${SASS_INSTR_PREFIX}BAR\.SYNC"
MEMBAR_FAM_RE="${SASS_INSTR_PREFIX}(MEMBAR|ERRBAR|DEPBAR|LDGDEPBAR)"
BSSY_RE="${SASS_INSTR_PREFIX}BSSY"
BSYNC_RE="${SASS_INSTR_PREFIX}BSYNC"
EXIT_RE="${SASS_INSTR_PREFIX}EXIT"
RET_RE="${SASS_INSTR_PREFIX}RET"
WARP_BR_RE="${SASS_INSTR_PREFIX}(WARPSYNC|CALL|JMX)"

bar_sync_seen=0
membar_seen=0
bssy_seen=0
bsync_seen=0
exit_seen=0
ret_seen=0
warp_branch_seen=0

inspect_one() {
    local bin="$1"
    if [[ ! -f "$bin" ]]; then
        echo "verify_lcd_sass_coverage.sh: $bin: no such file" >&2
        return 1
    fi
    local sass
    sass=$(cuobjdump --dump-sass "$bin" 2>&1) || {
        echo "verify_lcd_sass_coverage.sh: $bin: cuobjdump --dump-sass failed" >&2
        echo "$sass" >&2
        return 1
    }
    grep -Eq "$BAR_SYNC_RE"   <<< "$sass" && bar_sync_seen=1
    grep -Eq "$MEMBAR_FAM_RE" <<< "$sass" && membar_seen=1
    grep -Eq "$BSSY_RE"       <<< "$sass" && bssy_seen=1
    grep -Eq "$BSYNC_RE"      <<< "$sass" && bsync_seen=1
    grep -Eq "$EXIT_RE"       <<< "$sass" && exit_seen=1
    grep -Eq "$RET_RE"        <<< "$sass" && ret_seen=1
    grep -Eq "$WARP_BR_RE"    <<< "$sass" && warp_branch_seen=1
    return 0
}

cuobjdump_failed=0
for bin in "$@"; do
    inspect_one "$bin" || cuobjdump_failed=1
done

failures=0
[[ $bar_sync_seen     -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: bar_sync_coverage_missing" >&2; failures=$((failures + 1)); }
[[ $membar_seen       -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: membar_family_coverage_missing" >&2; failures=$((failures + 1)); }
[[ $bssy_seen         -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: bssy_coverage_missing" >&2; failures=$((failures + 1)); }
[[ $bsync_seen        -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: bsync_coverage_missing" >&2; failures=$((failures + 1)); }
[[ $exit_seen         -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: exit_coverage_missing" >&2; failures=$((failures + 1)); }
[[ $ret_seen          -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: ret_coverage_missing" >&2; failures=$((failures + 1)); }
[[ $warp_branch_seen  -eq 0 ]] && { echo "verify_lcd_sass_coverage.sh: warp_branch_coverage_missing" >&2; failures=$((failures + 1)); }

if [[ $cuobjdump_failed -ne 0 || $failures -gt 0 ]]; then
    exit 1
fi

cat <<EOF
verify_lcd_sass_coverage.sh: OK
  Barrier / fence: BAR.SYNC + (MEMBAR|ERRBAR|DEPBAR|LDGDEPBAR) both observed.
  Branch / return: BSSY, BSYNC, EXIT, RET, and (WARPSYNC|CALL|JMX) all observed.
EOF
exit 0
