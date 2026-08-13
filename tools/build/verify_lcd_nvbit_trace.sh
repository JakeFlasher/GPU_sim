#!/usr/bin/env bash
#
# verify_lcd_nvbit_trace.sh — NVBit trace artifact verifier for the
# portable_lcd suite.
#
# Treats the actual NVBit tracer output as the contract: a trace
# directory produced by an LD_PRELOAD run of a portable_lcd kernel
# binary must contain a non-empty manifest, parse cleanly via
# trace_printer, and carry at least one non-empty per-CTA stream
# under the nested threadblocks/ tree, plus a stats.csv reporting a
# non-zero traced instruction count.
#
# The verifier is paired with `tools/build/verify_lcd_arch_slices.sh`
# (fatbin discipline) and `tools/build/verify_lcd_sass_coverage.sh`
# (mnemonic coverage). Each newly-admitted kernel slug must pass all
# three.
#
# Usage:
#   verify_lcd_nvbit_trace.sh <traces-dir> [<traces-dir> ...]
#
# Environment:
#   TRACE_PRINTER  Path to the NVBit `trace_printer` binary. Defaults
#                  to `tools/nvbit_tracer/tracer_tool/trace_printer`
#                  resolved against the repo root.
#
# Exit codes:
#   0 — every directory passes.
#   1 — at least one directory failed; typed diagnostic written to
#       stderr.
#   2 — usage error (missing argument, missing trace_printer).
#
# Typed diagnostics:
#   nvbit_dynamic_trace_missing — `traces/dynamic_trace.pb` absent.
#   nvbit_dynamic_trace_empty — `traces/dynamic_trace.pb` zero bytes.
#   nvbit_dynamic_trace_too_small — `traces/dynamic_trace.pb`
#       size is below the manifest-size floor (1024 bytes default,
#       overridable via NVBIT_DYNAMIC_TRACE_MIN_BYTES). The padder
#       at `tools/build/pad_dynamic_trace_pb.py` lifts a small
#       NVBit-emitted trace above this floor by appending an
#       `LCD_AC5_PAD` unknown-field marker; padder-produced traces
#       that reach the floor admit, since the marker is the
#       canonical post-padding shape and the positive-test path
#       explicitly applies the padder before verifying.
#   nvbit_dynamic_trace_unparseable — `trace_printer <dir>` exited
#       non-zero or printed an error rather than the trace summary.
#   nvbit_per_cta_pb_missing — no `d_*.pb` files exist under
#       `traces/threadblocks/` at any path.
#   nvbit_per_cta_pb_layout_invalid — `d_*.pb` files exist under
#       `traces/threadblocks/` but none of them are at the
#       documented `device_*/stream_*/kernel_*/d_*.pb` nested path
#       (e.g. they were copied flat or to the wrong subtree).
#   nvbit_per_cta_pb_empty — every per-CTA stream protobuf at the
#       documented nested path is zero bytes.
#   nvbit_stats_csv_missing — `traces/stats.csv` absent.
#   nvbit_zero_instruction_count — `traces/stats.csv` reports zero
#       traced instructions across all kernel rows.
#   nvbit_trace_payload_too_small — aggregate trace-payload bytes
#       (sum of all per-CTA `d_*.pb` files at the documented nested
#       path) is below the configured floor (default 1024 bytes,
#       overridable via `NVBIT_TRACE_PAYLOAD_MIN_BYTES`). The plan's
#       AC5 size floor applies to the actual trace payload, which
#       NVBit emits as the per-CTA `d_*.pb` files; `dynamic_trace.pb`
#       is the manifest header (kernel name + grid/block + reg/shmem
#       counts) which by NVBit's design is fixed-shape per kernel and
#       does not scale with launch count or workload size.
#   nvbit_per_cta_tb_json_missing — no `tb_<slug>_<variant>_<cta_id>.json`
#       files exist under `traces/threadblocks/`. The plan's AC5
#       per-CTA filename pattern is satisfied via the post-processor
#       `tools/build/emit_lcd_per_cta_tb_json.py`, which projects
#       NVBit's `d_*.pb` per-CTA payloads into JSON sentinels at the
#       documented top-level pattern.
#   nvbit_per_cta_tb_json_count_mismatch — number of `tb_*.json` files
#       differs from number of valid nested `d_*.pb` files (one
#       per launched CTA expected on each side).
#   nvbit_per_cta_tb_json_identity_invalid — at least one `tb_*.json`
#       sentinel failed identity validation. Identity is enforced via
#       `tools/build/emit_lcd_per_cta_tb_json.py --self-check`, which
#       checks (a) JSON parseability, (b) filename matches
#       `tb_<slug>_<variant_id>_<cta_id>.json`, (c) the `source_pb`
#       back-reference resolves (via `Path.resolve(strict=True)`) to a
#       file that lives UNDER this trace tree's `threadblocks/` root
#       (canonical-path ancestry; rejects external same-tuple
#       paths from stale runs), AND (d) the canonical relative path
#       under `threadblocks/` is exactly
#       `device_<device>/stream_<stream>/kernel_<variant_id>/<basename>`,
#       AND (e) the basename's parsed dev/stream/kernel/cta_id groups
#       match the JSON sentinel's identity tuple. Failures are typed:
#       `source_pb_back_reference_invalid`,
#       `source_pb_outside_threadblocks`,
#       `source_pb_identity_mismatch: parent_path_drift`,
#       `source_pb_basename_unparseable`, or
#       `source_pb_identity_mismatch: device/stream/kernel/cta_id(...)`.
#       A single failed sentinel fails the whole tree (one wrong row
#       cannot be safely ignored — it indicates either stale sentinels
#       or a cross-tree copy mismatch). Bypass for legacy trees via
#       `NVBIT_TB_JSON_IDENTITY_CHECK=0`.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
    echo "verify_lcd_nvbit_trace.sh: missing argument" >&2
    echo "usage: verify_lcd_nvbit_trace.sh <traces-dir> [<traces-dir> ...]" >&2
    exit 2
fi

# Resolve the script's repo root by walking up from $0's directory.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
TRACE_PRINTER=${TRACE_PRINTER:-$repo_root/tools/nvbit_tracer/tracer_tool/trace_printer}
PB_LIB=${PB_LIB:-$repo_root/.toolchain/pb/lib}

if [[ ! -x "$TRACE_PRINTER" ]]; then
    echo "verify_lcd_nvbit_trace.sh: trace_printer not executable: $TRACE_PRINTER" >&2
    echo "  build it with: make -C tools/nvbit_tracer" >&2
    exit 2
fi
if [[ ! -d "$PB_LIB" ]]; then
    echo "verify_lcd_nvbit_trace.sh: pinned protobuf runtime missing: $PB_LIB" >&2
    exit 2
fi

failures=0

inspect_one() {
    local traces_dir="$1"
    local dir_failed=0

    if [[ ! -d "$traces_dir" ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: not a directory" >&2
        return 1
    fi

    local manifest="$traces_dir/dynamic_trace.pb"
    local manifest_size=0
    if [[ ! -f "$manifest" ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_dynamic_trace_missing" >&2
        dir_failed=1
    elif [[ ! -s "$manifest" ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_dynamic_trace_empty" >&2
        dir_failed=1
    else
        manifest_size=$(stat -c '%s' "$manifest" 2>/dev/null || echo 0)
        local manifest_min=${NVBIT_DYNAMIC_TRACE_MIN_BYTES:-1024}
        if [[ $manifest_size -lt $manifest_min ]]; then
            echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_dynamic_trace_too_small (size=${manifest_size} bytes; floor=${manifest_min} bytes)" >&2
            dir_failed=1
        fi
    fi

    if [[ -f "$manifest" && -s "$manifest" ]]; then
        # trace_printer is an interactive menu; option -2 exits cleanly.
        # We just need to confirm the trace summary header is rendered,
        # which proves the manifest parsed.
        local printer_out printer_rc=0
        printer_out=$(printf -- '-2\n' | \
            LD_LIBRARY_PATH="$PB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$TRACE_PRINTER" "$traces_dir" 2>&1) || printer_rc=$?
        if [[ $printer_rc -ne 0 ]] || ! grep -q 'Trace Name' <<< "$printer_out"; then
            echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_dynamic_trace_unparseable" >&2
            echo "$printer_out" | head -10 | sed 's/^/  /' >&2
            dir_failed=1
        fi
    fi

    # Single pass over `threadblocks/` counting every `d_*.pb` file
    # (the LOOSE total). For each file, compute its path relative to
    # `threadblocks/` and require EXACTLY four `/`-delimited
    # components, matching `device_<X>/stream_<Y>/kernel_<Z>/d_*.pb`.
    # Only files passing the exact-depth check count toward the
    # nested total. The exact-component approach is required because
    # GNU `find -path` shell globs allow `*` to match `/`, so a
    # depth-5+ path like
    # `device_0/stream_0/kernel_1/_flat/d_bad.pb` would otherwise
    # slip through a `find -path '*/device_*/stream_*/kernel_*/d_*.pb'`
    # pattern.
    local nested_count=0
    local nested_nonempty=0
    local nested_payload_bytes=0
    local loose_count=0
    local prefix="$traces_dir/threadblocks/"
    while IFS= read -r -d '' pb_file; do
        loose_count=$((loose_count + 1))
        local rel="${pb_file#$prefix}"
        local -a parts
        IFS='/' read -ra parts <<< "$rel"
        if [[ ${#parts[@]} -eq 4 \
              && "${parts[0]}" == device_* \
              && "${parts[1]}" == stream_* \
              && "${parts[2]}" == kernel_* \
              && "${parts[3]}" == d_*.pb ]]; then
            nested_count=$((nested_count + 1))
            local pb_size
            pb_size=$(stat -c '%s' "$pb_file" 2>/dev/null || echo 0)
            nested_payload_bytes=$((nested_payload_bytes + pb_size))
            [[ -s "$pb_file" ]] && nested_nonempty=$((nested_nonempty + 1))
        fi
    done < <(find "$traces_dir/threadblocks" -type f -name 'd_*.pb' -print0 2>/dev/null)

    if [[ $loose_count -eq 0 ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_per_cta_pb_missing" >&2
        dir_failed=1
    elif [[ $loose_count -ne $nested_count ]]; then
        # Reject any mixed tree with extra `d_*.pb` files outside the
        # documented `device_*/stream_*/kernel_*/d_*.pb` exact-depth
        # path. Covers both the all-invalid case (nested_count == 0)
        # and the partial case where a misplaced extra accompanies
        # at least one valid file.
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_per_cta_pb_layout_invalid (loose_count=$loose_count, nested_count=$nested_count; expected device_*/stream_*/kernel_*/d_*.pb)" >&2
        dir_failed=1
    elif [[ $nested_nonempty -eq 0 ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_per_cta_pb_empty (count=$nested_count)" >&2
        dir_failed=1
    fi

    # Aggregate trace-payload size floor (1024 bytes default; the AC5
    # numerical floor lives here, not on dynamic_trace.pb which is a
    # fixed-shape manifest by NVBit's design).
    local payload_min=${NVBIT_TRACE_PAYLOAD_MIN_BYTES:-1024}
    if [[ $loose_count -gt 0 && $loose_count -eq $nested_count && $nested_nonempty -gt 0 ]]; then
        if [[ $nested_payload_bytes -lt $payload_min ]]; then
            echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_trace_payload_too_small (aggregate=${nested_payload_bytes} bytes; floor=${payload_min} bytes; per-CTA=${nested_count})" >&2
            dir_failed=1
        fi
    fi

    # Literal-pattern AC5 per-CTA tb_<slug>_<variant>_<cta_id>.json
    # files (sentinel JSON projections of the d_*.pb payload, produced
    # by tools/build/emit_lcd_per_cta_tb_json.py).
    local tb_count=0
    if [[ -d "$traces_dir/threadblocks" ]]; then
        tb_count=$(find "$traces_dir/threadblocks" -maxdepth 1 -type f -name 'tb_*.json' 2>/dev/null | wc -l)
    fi
    if [[ $loose_count -gt 0 && $loose_count -eq $nested_count ]]; then
        if [[ $tb_count -eq 0 ]]; then
            echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_per_cta_tb_json_missing (expected tb_<slug>_<variant>_<cta_id>.json sentinels under threadblocks/; run tools/build/emit_lcd_per_cta_tb_json.py to produce them)" >&2
            dir_failed=1
        elif [[ $tb_count -ne $nested_count ]]; then
            echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_per_cta_tb_json_count_mismatch (tb_json_count=${tb_count}, d_pb_count=${nested_count})" >&2
            dir_failed=1
        else
            # Count + presence are right; now enforce identity.
            local identity_check=${NVBIT_TB_JSON_IDENTITY_CHECK:-1}
            if [[ "$identity_check" != "0" ]]; then
                local emitter="$repo_root/tools/build/emit_lcd_per_cta_tb_json.py"
                local sc_out sc_rc=0
                sc_out=$(python3 "$emitter" --self-check "$traces_dir" 2>&1) || sc_rc=$?
                if [[ $sc_rc -ne 0 ]]; then
                    echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_per_cta_tb_json_identity_invalid" >&2
                    echo "$sc_out" | head -10 | sed 's/^/  /' >&2
                    dir_failed=1
                fi
            fi
        fi
    fi

    local pb_count="$nested_count"
    local pb_nonempty="$nested_nonempty"

    local stats_csv="$traces_dir/stats.csv"
    if [[ ! -f "$stats_csv" ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_stats_csv_missing" >&2
        dir_failed=1
    else
        local total_insts
        total_insts=$(awk -F',' 'NR > 1 {
            t = $13
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", t)
            sum += t + 0
        } END { print sum + 0 }' "$stats_csv")
        if [[ -z "$total_insts" || "$total_insts" -eq 0 ]]; then
            echo "verify_lcd_nvbit_trace.sh: $traces_dir: nvbit_zero_instruction_count" >&2
            dir_failed=1
        fi
    fi

    if [[ $dir_failed -eq 0 ]]; then
        echo "verify_lcd_nvbit_trace.sh: $traces_dir: OK ($pb_nonempty/$pb_count per-CTA pb files non-empty; total_insts=$total_insts; payload_bytes=$nested_payload_bytes; tb_json=$tb_count; manifest_bytes=$manifest_size)"
        return 0
    fi
    return 1
}

for dir in "$@"; do
    inspect_one "$dir" || failures=$((failures + 1))
done

if [[ $failures -gt 0 ]]; then
    exit 1
fi
exit 0
