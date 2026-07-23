#!/usr/bin/env bash
# Local-label driver: invoke Nsight Compute on a kernel binary on a local
# GPU device (RTX 3090 Ti sm_86 or RTX 5060 laptop sm_120) and emit a
# per-launch JSON sidecar to
#
#   data/local_labels/<device>/<slug>/<shape_point>/label_<repeat>_<launch>.json
#
# The cloud/server contract still forbids ncu and nv-nsight-cu-cli on
# every non-local target. This script lives under tools/local_labels/
# and is the only path in the repository that may invoke ncu.
#
# Usage:
#   tools/local_labels/label_driver.sh \
#       --device sm_86 \
#       --slug gemm_hmma_bf16_sm80 \
#       --shape-point row_0 \
#       --repeat 0 \
#       --launch 0 \
#       --binary path/to/kernel.elf \
#       [--shape-binding application-args|logical-workload] \
#       --warmup 1 \
#       [--ncu /usr/local/cuda/bin/ncu] \
#       [--out-dir data/local_labels] \
#       [--kernel-name <regex>] \
#       -- <application args ...>
#
# Application arguments after `--` are forwarded verbatim to the profiled
# binary. Direct workloads bind the requested shape through those arguments.
# Adapter-backed workloads can instead use `--shape-binding logical-workload`:
# the shape remains required label metadata, while shape-like tokens are
# forbidden in the target argv because the adapter consumes its own pinned
# workload contract.
#
# Every NCU collection invocation passes:
#   --launch-skip <launch> --launch-count 1   (R4 launch filter)
#   --clock-control none                      (Rule 2: no clock locking)
# and uses the safe Python parser at parse_ncu_metrics.py to extract
# percentage metrics from the documented "Metric Value" column of the
# raw-page CSV — never the first numeric token of an identifier-bearing
# row.

set -euo pipefail

DEVICE=""
SLUG=""
SHAPE_POINT=""
SHAPE_BINDING="application-args"
REPEAT_INDEX=""
LAUNCH_INDEX=""
BINARY_PATH=""
WARMUP_COUNT="1"
WARMUP_COUNT_OVERRIDDEN=0
NCU_BIN="ncu"
OUT_DIR="data/local_labels"
KERNEL_NAME=""
CACHE_CONTROL=""
EXPECTED_CUDA_RELEASE=""
EXPECTED_NCU_VERSION_PREFIX=""
PUBLICATION_METRIC_SCOPE=""
# v2 protocol (2026-07-06 re-founding, CLAUDE.md Rule 2): NCU clock control is
# EXPLICIT on every capture, never defaulted (the NCU default changed
# base->boost in 2026.1). "none" preserves the historical behavior and is the
# right mode when the host clocks are externally locked (nvidia-smi -lgc).
NCU_CLOCK_CONTROL="none"
MEASUREMENT_MANIFEST_PATH=""
EXPECTED_VARIANT=""
APP_ARGS=()
APP_ARGS_PROVIDED=0
# Non-percentage (count/ns/cycles) or non-flat-field registry metrics to
# collect alongside the two mandatory hit-rate percentages and route through
# parse_ncu_metrics.py's registry-driven unit-aware path into the sidecar's
# optional unit_tagged_metrics block.
UNIT_METRICS=()

usage() {
    cat <<'EOF'
local-label driver

Required arguments:
  --device <sm_*>           Compute capability tag (e.g. sm_86, sm_120).
  --slug <name>             Kernel slug from the workload roster.
  --shape-point <id>        Shape-file row identifier.
  --shape-binding <mode>    How the target invocation binds the shape.
                            application-args (default) requires the existing
                            workload flag after `--`; logical-workload keeps
                            the shape as label metadata and forbids forwarding
                            shape-like flags to the adapter target.
  --repeat <int>            Zero-based repeat index.
  --launch <int>            Zero-based launch index inside the repeat.
  --binary <path>           Kernel executable to profile.

Optional arguments:
  --warmup <int>            Warmup launches before measurement (default: 1).
  --ncu <path>              Path to the Nsight Compute CLI (default: 'ncu' on PATH).
  --out-dir <path>          Quarantine root (default: data/local_labels).
  --kernel-name <regex>     Forward as NCU --kernel-name filter.
  --cache-control <mode>    Forward as NCU --cache-control flag
                            (mode in {all, none}). Optional;
                            omit to defer to NCU's default behavior.
  --clock-control <mode>    Forward as NCU --clock-control flag
                            (mode in {base, boost, none}; default: none).
                            Always passed explicitly (Rule 2). Use none
                            when host clocks are locked via nvidia-smi
                            -lgc; base/boost delegate locking to NCU.
  --unit-metric <ncu_metric>
                            Canonical NCU metric id (e.g. gpc__cycles_elapsed.max,
                            gpu__time_duration.sum, dram__sectors_read.sum) to
                            collect alongside the two mandatory hit-rate
                            percentages and route through the registry-driven
                            unit-aware parser into the sidecar's optional
                            unit_tagged_metrics block. Repeatable. Each id must
                            resolve to a tools/local_labels/metric_registry.json
                            entry; count/ns/cycles values are never divided by 100.
  --expected-cuda-release <X.Y>
                            Require nvcc on the supplied environment's PATH to
                            report this exact release before NCU starts.
  --expected-ncu-version-prefix <X.Y[.Z]>
                            Require the NCU executable to report this release
                            prefix on a numeric component boundary.
  --publication-metric-scope cycles_only_binding
                            Bind publication to gpc__cycles_elapsed.max while
                            retaining the requested registry-superset metrics
                            as non-binding diagnostic evidence.
  --measurement-manifest <path>
                            Path to a measurement_manifest.json carrying
                            the operator-supplied kernel_signature_sha256
                            (and optionally binary_sha256). When set, the
                            driver copies the manifest's
                            kernel_signature_sha256 into the emitted
                            label_run_manifest_*_*.json (instead of the
                            locally-computed fallback) and fails closed
                            BEFORE NCU collection if the manifest is
                            absent, malformed, missing
                            kernel_signature_sha256, or carries a
                            binary_sha256 that disagrees with the
                            profiled binary. The manifest is the
                            operator-pinned source-of-truth for the
                            kernel-signature value.
  --expected-variant <token>
                            Optional canonical register-cap build-variant
                            token (register_cap_<cap>) the captured cell is
                            expected to be. Requires --measurement-manifest.
                            When set, the driver requires the manifest's
                            register_cap/variant_id to normalize to this exact
                            token and fails closed BEFORE NCU on a mismatch,
                            noncanonical alias, or negative cap. The validated
                            canonical pair is copied into the run manifest and
                            sidecar; base captures omit both fields.
  --help, -h                Show this help and exit.
  --                        Marker; everything after this token is
                            forwarded verbatim to the profiled binary.
                            The forwarded args must encode the same
                            shape_point as --shape-point above.

Output paths:
  <out-dir>/<device>/<slug>/<shape_point>/label_<repeat>_<launch>.json
  <out-dir>/<device>/<slug>/<shape_point>/run_<repeat>_<launch>.ncu-rep
EOF
}

while (("$#")); do
    case "$1" in
        --help | -h)
            usage
            exit 0
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --slug)
            SLUG="$2"
            shift 2
            ;;
        --shape-point)
            SHAPE_POINT="$2"
            shift 2
            ;;
        --shape-binding)
            SHAPE_BINDING="$2"
            case "$SHAPE_BINDING" in
                application-args|logical-workload)
                    ;;
                *)
                    echo "label_driver: --shape-binding expects 'application-args' or 'logical-workload', got: $SHAPE_BINDING" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --repeat)
            REPEAT_INDEX="$2"
            shift 2
            ;;
        --launch)
            LAUNCH_INDEX="$2"
            shift 2
            ;;
        --binary)
            BINARY_PATH="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_COUNT="$2"
            WARMUP_COUNT_OVERRIDDEN=1
            shift 2
            ;;
        --ncu)
            NCU_BIN="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --kernel-name)
            KERNEL_NAME="$2"
            shift 2
            ;;
        --cache-control)
            CACHE_CONTROL="$2"
            case "$CACHE_CONTROL" in
                all|none)
                    ;;
                *)
                    echo "label_driver: --cache-control expects 'all' or 'none', got: $CACHE_CONTROL" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --clock-control)
            NCU_CLOCK_CONTROL="$2"
            case "$NCU_CLOCK_CONTROL" in
                base|boost|none)
                    ;;
                *)
                    echo "label_driver: --clock-control expects 'base', 'boost' or 'none', got: $NCU_CLOCK_CONTROL" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --measurement-manifest)
            MEASUREMENT_MANIFEST_PATH="$2"
            shift 2
            ;;
        --expected-variant)
            EXPECTED_VARIANT="$2"
            shift 2
            ;;
        --unit-metric)
            UNIT_METRICS+=("$2")
            shift 2
            ;;
        --expected-cuda-release)
            EXPECTED_CUDA_RELEASE="$2"
            if ! [[ "$EXPECTED_CUDA_RELEASE" =~ ^(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$ ]]; then
                echo "label_driver: --expected-cuda-release expects canonical X.Y, got: $EXPECTED_CUDA_RELEASE" >&2
                exit 2
            fi
            shift 2
            ;;
        --expected-ncu-version-prefix)
            EXPECTED_NCU_VERSION_PREFIX="$2"
            if ! [[ "$EXPECTED_NCU_VERSION_PREFIX" =~ ^(0|[1-9][0-9]*)([.](0|[1-9][0-9]*)){1,3}$ ]]; then
                echo "label_driver: --expected-ncu-version-prefix expects canonical dotted numeric components, got: $EXPECTED_NCU_VERSION_PREFIX" >&2
                exit 2
            fi
            shift 2
            ;;
        --publication-metric-scope)
            PUBLICATION_METRIC_SCOPE="$2"
            if [[ "$PUBLICATION_METRIC_SCOPE" != "cycles_only_binding" ]]; then
                echo "label_driver: --publication-metric-scope expects cycles_only_binding, got: $PUBLICATION_METRIC_SCOPE" >&2
                exit 2
            fi
            shift 2
            ;;
        --)
            shift
            APP_ARGS_PROVIDED=1
            while (("$#")); do
                APP_ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "label_driver: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "label_driver: missing required argument $name" >&2
        usage >&2
        exit 2
    fi
}

require "--device" "$DEVICE"
require "--slug" "$SLUG"
require "--shape-point" "$SHAPE_POINT"
require "--repeat" "$REPEAT_INDEX"
require "--launch" "$LAUNCH_INDEX"
require "--binary" "$BINARY_PATH"

# Architecture capability gate (C-AC-E). Translate the legacy local
# label `sm_120` to the manifest's `sm_120-laptop` (the bare
# `sm_120` value is the B200 server SKU and forbidden during Loop
# 2). All other tags pass through unchanged so server arches
# (`sm_80`, `sm_89`, `sm_90a`) reach the helper as-is and reject
# with typed `arch_unsupported`. The helper exits non-zero with
# JSON on stderr; we propagate the exit and stderr verbatim before
# any binary / NCU work begins. The driver opts out of the helper
# only when `LABEL_DRIVER_SKIP_ARCH_CHECK` is set to `1` (kept as
# an emergency override; CI / runbook flows MUST NOT set it).
if [[ "${LABEL_DRIVER_SKIP_ARCH_CHECK:-0}" != "1" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    HELPER_PY="$REPO_ROOT/tools/arch_capabilities.py"
    case "$DEVICE" in
        sm_120)
            CAP_ARCH="sm_120-laptop"
            ;;
        *)
            CAP_ARCH="$DEVICE"
            ;;
    esac
    # Suppress the helper's stdout admit message (the "ok: ..."
    # line on success) but let stderr flow through so the
    # operator sees the typed-reason JSON when the helper rejects
    # the arch (`tools/arch_capabilities.py` writes the typed-
    # reason JSON to stderr on rc=1; admit messages on stdout).
    # A single invocation suffices — re-running was the prior
    # implementation's attempt to surface the JSON, but it is
    # already on stderr from the first call.
    if ! python3 "$HELPER_PY" \
            --path "tools/local_labels/label_driver.sh" \
            --arch "$CAP_ARCH" >/dev/null; then
        exit 2
    fi
fi

if [[ ! -x "$BINARY_PATH" ]]; then
    echo "label_driver: --binary path is not executable: $BINARY_PATH" >&2
    exit 2
fi

# AC2 (R4-A): bind the profiled invocation to the requested shape_point.
# Without this, the sidecar can claim shape_point=row_0 while NCU measured
# the binary's default invocation. The operator must pass the workload
# shape flag after the `--` separator. Special-cased shape_points map to
# legacy workload flags (correctness_shape, timing_shape); every other
# value must be present as `--shape-point <id>`.
#
# The check is fail-closed against duplicate or conflicting shape flags.
# `--shape-point row_42 --shape-point row_99` is rejected even if the
# requested shape_point matches one of the two — last-one-wins parsing
# in the workload binary would otherwise let one shape's metadata
# describe a different shape's measurement.
correctness_count=0
timing_count=0
shape_point_count=0
shape_point_matches=0
shape_point_values=()
prev=""
for arg in "${APP_ARGS[@]+"${APP_ARGS[@]}"}"; do
    if [[ "$arg" == "--correctness_shape" ]]; then
        correctness_count=$((correctness_count + 1))
    elif [[ "$arg" == "--timing_shape" ]]; then
        timing_count=$((timing_count + 1))
    elif [[ "$arg" == "--shape-point" ]]; then
        shape_point_count=$((shape_point_count + 1))
    fi
    if [[ "$prev" == "--shape-point" ]]; then
        shape_point_values+=("$arg")
        if [[ "$arg" == "$SHAPE_POINT" ]]; then
            shape_point_matches=$((shape_point_matches + 1))
        fi
    fi
    prev="$arg"
done

shape_args_ok=0
if [[ "$SHAPE_BINDING" == "logical-workload" ]]; then
    # The adapter owns definition/workload selection.  Passing any of the
    # direct-workload shape flags would either be rejected by that adapter or,
    # worse, let metadata describe a different invocation.
    if [[ "$APP_ARGS_PROVIDED" -eq 1 && ${#APP_ARGS[@]} -gt 0 \
        && "$correctness_count" -eq 0 && "$timing_count" -eq 0 \
        && "$shape_point_count" -eq 0 ]]; then
        shape_args_ok=1
    fi
elif [[ "$SHAPE_POINT" == "correctness_shape" ]]; then
    # Reject any cross-form mixing: special-case shape_points must NOT
    # carry --shape-point in app args, otherwise last-one-wins parsing
    # in the workload binary could let a generic shape override the
    # special-case one while the sidecar still claims correctness_shape.
    if [[ "$correctness_count" -eq 1 && "$timing_count" -eq 0 && "$shape_point_count" -eq 0 ]]; then
        shape_args_ok=1
    fi
elif [[ "$SHAPE_POINT" == "timing_shape" ]]; then
    if [[ "$timing_count" -eq 1 && "$correctness_count" -eq 0 && "$shape_point_count" -eq 0 ]]; then
        shape_args_ok=1
    fi
else
    # Generic shape_points must NOT carry --correctness_shape /
    # --timing_shape either: a generic shape running with
    # --correctness_shape would silently switch the workload to
    # correctness mode while the sidecar claims a generic shape.
    if [[ "$shape_point_count" -eq 1 && "$shape_point_matches" -eq 1 \
        && "$correctness_count" -eq 0 && "$timing_count" -eq 0 ]]; then
        shape_args_ok=1
    fi
fi

if [[ "$shape_args_ok" -ne 1 ]]; then
    echo "label_driver: forwarded application args do not unambiguously bind shape_point=$SHAPE_POINT with shape_binding=$SHAPE_BINDING" >&2
    if [[ "$APP_ARGS_PROVIDED" -ne 1 ]]; then
        echo "label_driver: did you forget the '--' separator before app args?" >&2
    fi
    if [[ "$SHAPE_BINDING" == "logical-workload" ]]; then
        echo "label_driver: logical-workload binding requires nonempty target argv with no --shape-point, --correctness_shape, or --timing_shape token" >&2
    elif [[ "$correctness_count" -gt 0 && "$timing_count" -gt 0 ]]; then
        echo "label_driver: forwarded args contain both --correctness_shape and --timing_shape; pick exactly one" >&2
    elif [[ ( "$SHAPE_POINT" == "correctness_shape" || "$SHAPE_POINT" == "timing_shape" ) \
        && "$shape_point_count" -gt 0 ]]; then
        echo "label_driver: shape_point=$SHAPE_POINT is a special-case workload flag and must not be combined with --shape-point in app args" >&2
    elif [[ "$SHAPE_POINT" != "correctness_shape" && "$SHAPE_POINT" != "timing_shape" \
        && "$correctness_count" -gt 0 ]]; then
        echo "label_driver: shape_point=$SHAPE_POINT is a generic shape and must not be combined with --correctness_shape in app args" >&2
    elif [[ "$SHAPE_POINT" != "correctness_shape" && "$SHAPE_POINT" != "timing_shape" \
        && "$timing_count" -gt 0 ]]; then
        echo "label_driver: shape_point=$SHAPE_POINT is a generic shape and must not be combined with --timing_shape in app args" >&2
    elif [[ "$shape_point_count" -gt 1 ]]; then
        echo "label_driver: forwarded args contain $shape_point_count '--shape-point' tokens; pass exactly one" >&2
    elif [[ "$shape_point_count" -eq 1 && "$shape_point_matches" -eq 0 ]]; then
        echo "label_driver: forwarded --shape-point ${shape_point_values[0]:-?} does not match --shape-point $SHAPE_POINT" >&2
    elif [[ "$correctness_count" -gt 1 ]]; then
        echo "label_driver: forwarded args contain --correctness_shape $correctness_count times; pass exactly one" >&2
    elif [[ "$timing_count" -gt 1 ]]; then
        echo "label_driver: forwarded args contain --timing_shape $timing_count times; pass exactly one" >&2
    else
        if [[ "$SHAPE_POINT" == "correctness_shape" ]]; then
            echo "label_driver: append '--correctness_shape' after the '--' separator" >&2
        elif [[ "$SHAPE_POINT" == "timing_shape" ]]; then
            echo "label_driver: append '--timing_shape' after the '--' separator" >&2
        else
            echo "label_driver: append '--shape-point $SHAPE_POINT' after the '--' separator" >&2
        fi
    fi
    exit 2
fi

# Workload-aware warmup forwarding (R5).
#
# SimBench runners (e.g. workloads/simbench_bw/l1/.../main.cu) reject
# unknown args, so unconditionally appending `--warmup <count>` to a
# generic `--shape-point <id>` invocation breaks the workload. The
# portable-LCD runner accepts `--warmup`. The driver therefore:
#
#   * For shape_point ∈ {correctness_shape, timing_shape}: append
#     `--warmup <count>` only when no warmup flag is already
#     forwarded, and write the actual forwarded value (or driver
#     default) into the sidecar's warmup_count.
#   * For generic `--shape-point <id>` invocations: do NOT append
#     anything. If the forwarded args already carry a warmup flag,
#     parse its value into warmup_count; otherwise warmup_count = 0.
#   * If the driver-level --warmup disagrees with a forwarded warmup
#     value, fail-fast with a typed error.
#
# WARMUP_DEFAULT_USED tracks whether the operator overrode the driver
# default. The default is "1"; when WARMUP_COUNT is at the default we
# still allow forwarded warmup values to be written, but we only
# trigger the disagreement check when the operator explicitly
# overrode --warmup.
forwarded_warmup_value=""
warmup_token_count=0
prev=""
for arg in "${APP_ARGS[@]+"${APP_ARGS[@]}"}"; do
    if [[ "$arg" == "--warmup" ]]; then
        warmup_token_count=$((warmup_token_count + 1))
    elif [[ "$arg" == --warmup=* ]]; then
        warmup_token_count=$((warmup_token_count + 1))
        forwarded_warmup_value="${arg#--warmup=}"
    fi
    if [[ "$prev" == "--warmup" ]]; then
        forwarded_warmup_value="$arg"
    fi
    prev="$arg"
done

if [[ "$warmup_token_count" -gt 1 ]]; then
    echo "label_driver: forwarded args contain $warmup_token_count '--warmup' tokens; pass at most one" >&2
    exit 2
fi

if [[ -n "$forwarded_warmup_value" ]]; then
    if ! [[ "$forwarded_warmup_value" =~ ^[0-9]+$ ]]; then
        echo "label_driver: forwarded --warmup '$forwarded_warmup_value' is not a non-negative integer" >&2
        exit 2
    fi
    if [[ "$WARMUP_COUNT_OVERRIDDEN" -eq 1 \
        && "$forwarded_warmup_value" != "$WARMUP_COUNT" ]]; then
        echo "label_driver: warmup_provenance_mismatch: driver --warmup $WARMUP_COUNT disagrees with forwarded --warmup $forwarded_warmup_value" >&2
        exit 2
    fi
fi

case "$SHAPE_POINT" in
    correctness_shape | timing_shape)
        # Special-case shape: workload accepts --warmup. Append the
        # driver default only if the operator did not forward one.
        if [[ "$warmup_token_count" -eq 0 ]]; then
            APP_ARGS+=("--warmup" "$WARMUP_COUNT")
            EFFECTIVE_WARMUP_COUNT="$WARMUP_COUNT"
        else
            EFFECTIVE_WARMUP_COUNT="$forwarded_warmup_value"
        fi
        ;;
    *)
        # Generic shape: do NOT append --warmup; SimBench runners
        # reject unknown args. Record only what is actually forwarded.
        if [[ "$warmup_token_count" -eq 0 ]]; then
            EFFECTIVE_WARMUP_COUNT="0"
        else
            EFFECTIVE_WARMUP_COUNT="$forwarded_warmup_value"
        fi
        ;;
esac

# Path helpers.
CELL_DIR="${OUT_DIR}/${DEVICE}/${SLUG}/${SHAPE_POINT}"
SIDECAR_PATH="${CELL_DIR}/label_${REPEAT_INDEX}_${LAUNCH_INDEX}.json"
REPORT_PATH="${CELL_DIR}/run_${REPEAT_INDEX}_${LAUNCH_INDEX}.ncu-rep"

# Provenance gathering. Every value below feeds the JSON sidecar that
# the importer (tools/local_labels/import_label.py) validates.
#
# safe_capture runs a command, returns its first stdout line ONLY
# when the command exited 0 with non-empty output, otherwise the
# literal string "unknown". The returned value is then exported
# through an environment variable; the sidecar Python program reads
# it via os.environ and json.dump-encodes it, so even shell output
# containing double quotes, backslashes, or other Python
# metacharacters cannot break the JSON emitter.
#
# R6: the helper used to be `cmd 2>/dev/null | head -n 1` followed
# by `|| true`, which discarded the underlying command's exit status
# whenever stdout was non-empty. A failed `ncu --version` that wrote
# any output then leaked into the sidecar as canonical
# `profiler_version`. The corrected helper captures stdout into a
# variable and reads `$?` immediately, so a non-zero exit normalizes
# to `unknown` regardless of stdout.
safe_capture() {
    local out status
    out="$("$@" 2>/dev/null)"
    status=$?
    if [[ "$status" -ne 0 || -z "$out" ]]; then
        printf 'unknown'
        return
    fi
    # First line only, no subshell pipeline (so we don't reintroduce
    # the same exit-status problem).
    printf '%s' "${out%%$'\n'*}"
}

# NCU --version output is a multi-line banner: line 1 is the product
# name ("NVIDIA (R) Nsight Compute Command Line Profiler"), line 2 is
# the copyright, line 3 is the actual release line ("Version
# 2026.1.1.0 (build 37634170) (public-release)"). The plain
# safe_capture helper returned only line 1 — R10's sidecar
# carried the product banner instead of a real version string,
# which Codex's R10 review correctly rejected. R11 walks the
# multi-line stdout and returns the FIRST line matching
# `^Version[[:space:]]+`. Returns the literal "unknown" if the
# command fails or no matching line is present.
safe_ncu_version() {
    local out status
    out="$("$@" --version 2>/dev/null)"
    status=$?
    if [[ "$status" -ne 0 || -z "$out" ]]; then
        printf 'unknown'
        return
    fi
    while IFS= read -r line; do
        if [[ "$line" =~ ^Version[[:space:]]+ ]]; then
            printf '%s' "$line"
            return
        fi
    done <<< "$out"
    printf 'unknown'
}

PROFILER_VERSION="$(safe_ncu_version "$NCU_BIN")"
# nvcc --version output is a multiline banner — we want only the
# canonical "release X.Y" token. The R5 implementation chained
# grep|head|awk under `set -euo pipefail`, which aborted the entire
# driver whenever nvcc was missing or exited non-zero (the pipeline's
# rightmost failing command propagates via pipefail). R7 wraps
# the capture in an `if !` guard so the non-zero exit normalizes to
# the literal `unknown` and the driver still reaches sidecar
# emission.
safe_cuda_version() {
    local out
    if ! out="$(nvcc --version 2>/dev/null)"; then
        printf 'unknown'
        return
    fi
    if [[ "$out" =~ release[[:space:]]+([0-9]+[.][0-9]+) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        printf 'unknown'
    fi
}

CUDA_VERSION="$(safe_cuda_version)"
DRIVER_VERSION="$(safe_capture nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
GPU_UUID_RAW="$(safe_capture nvidia-smi --query-gpu=uuid --format=csv,noheader)"
GPU_UUID_SHA256="$(printf '%s' "$GPU_UUID_RAW" | sha256sum | awk '{print $1}')"
BINARY_SHA256="$(sha256sum "$BINARY_PATH" | awk '{print $1}')"

# The campaign passes these explicit expectations from its sealed toolchain
# pins.  Check them before creating the label directory and before NCU sees the
# target binary.  Historical callers may omit the flags and retain their
# existing behavior; live-v2 callers are required to supply both.
if [[ -n "$EXPECTED_CUDA_RELEASE" \
    && "$CUDA_VERSION" != "$EXPECTED_CUDA_RELEASE" ]]; then
    echo "label_driver: CUDA_RELEASE_MISMATCH: observed=$CUDA_VERSION expected=$EXPECTED_CUDA_RELEASE" >&2
    exit 2
fi
if [[ -n "$EXPECTED_NCU_VERSION_PREFIX" ]]; then
    observed_ncu_release="${PROFILER_VERSION#Version }"
    observed_ncu_release="${observed_ncu_release%% *}"
    if ! [[ "$observed_ncu_release" =~ ^(0|[1-9][0-9]*)([.](0|[1-9][0-9]*)){1,3}$ ]] \
        || { [[ "$observed_ncu_release" != "$EXPECTED_NCU_VERSION_PREFIX" ]] \
            && [[ "$observed_ncu_release" != "$EXPECTED_NCU_VERSION_PREFIX".* ]]; }; then
        echo "label_driver: NCU_VERSION_MISMATCH: observed=$PROFILER_VERSION expected_prefix=$EXPECTED_NCU_VERSION_PREFIX" >&2
        exit 2
    fi
fi
if [[ "$PUBLICATION_METRIC_SCOPE" == "cycles_only_binding" ]]; then
    cycles_binding_count=0
    for _um in "${UNIT_METRICS[@]+"${UNIT_METRICS[@]}"}"; do
        if [[ "$_um" == "gpc__cycles_elapsed.max" ]]; then
            cycles_binding_count=$((cycles_binding_count + 1))
        fi
    done
    if [[ "$cycles_binding_count" -ne 1 ]]; then
        echo "label_driver: NCU_PUBLICATION_METRIC_INVALID: cycles_only_binding requires gpc__cycles_elapsed.max exactly once; observed=$cycles_binding_count" >&2
        exit 2
    fi
fi

mkdir -p "$CELL_DIR"

# Kernel-signature provenance (operator-pinned source-of-truth).
#
# When the operator passes --measurement-manifest <path>, that file is
# the binding source-of-truth for kernel_signature_sha256: the driver
# reads it BEFORE NCU collection and fails closed before any sidecar
# is written if the file is absent, the JSON does not parse, the
# kernel_signature_sha256 field is missing/empty, or the manifest's
# binary_sha256 (when present) disagrees with the actual binary's
# sha256. The validated manifest value is copied into the emitted
# label_run_manifest_*_*.json.
#
# When --measurement-manifest is absent, the driver falls back to the
# locally-computed sha256(slug|canonical_binary_path) form mirroring
# workloads/simbench_bw/run_suite.py::_kernel_signature_sha256 (lines
# 1623-1651), and records the kernel_signature_source="fallback_legacy"
# marker so the aggregator (and downstream callers) can decide whether
# to admit the fallback. The canonical_binary_path is the repo-relative
# form when $BINARY_PATH resolves under REPO_ROOT, else the bare
# basename — matching the run_suite.py-side rule byte-for-byte so the
# fallback hash joins the predictor-side hash for the same binary
# regardless of whether the operator passed an absolute, ./-prefixed,
# repo-relative, or out-of-tree spelling. Live operator workflows must
# drive the harness with --measurement-manifest; the fallback exists
# for unit tests and offline harness rehearsal.
if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
canonicalize_kernel_sig_binary_path() {
    local raw="$1"
    local abs
    abs="$(realpath -- "$raw" 2>/dev/null || true)"
    if [[ -z "$abs" ]]; then
        abs="$raw"
    fi
    if [[ "$abs" == "$REPO_ROOT"/* ]]; then
        printf '%s' "${abs#"$REPO_ROOT"/}"
    else
        printf '%s' "$(basename -- "$abs")"
    fi
}
KERNEL_SIG_BINARY_PATH="$(canonicalize_kernel_sig_binary_path "$BINARY_PATH")"
KERNEL_SIGNATURE_SOURCE="fallback_legacy"
KERNEL_SIGNATURE_SHA256="$(printf '%s|%s' "$SLUG" "$KERNEL_SIG_BINARY_PATH" | sha256sum | awk '{print $1}')"
# Optional register-cap build-variant identity, validated from the
# measurement manifest below and propagated into the run manifest + sidecar.
# Empty for a base capture (byte-identical to the pre-variant path).
LD_REGISTER_CAP=""
LD_VARIANT_ID=""
# --expected-variant has nothing to validate against without a manifest.
if [[ -n "$EXPECTED_VARIANT" && -z "$MEASUREMENT_MANIFEST_PATH" ]]; then
    echo "label_driver: --expected-variant requires --measurement-manifest" >&2
    exit 2
fi
if [[ -n "$MEASUREMENT_MANIFEST_PATH" ]]; then
    if [[ ! -f "$MEASUREMENT_MANIFEST_PATH" ]]; then
        echo "label_driver: --measurement-manifest path does not exist: $MEASUREMENT_MANIFEST_PATH" >&2
        exit 2
    fi
    MANIFEST_VALIDATION="$(
        LD_MANIFEST_PATH="$MEASUREMENT_MANIFEST_PATH" \
        LD_BINARY_SHA256="$BINARY_SHA256" \
        LD_CLI_SLUG="$SLUG" \
        LD_CLI_SHAPE_POINT="$SHAPE_POINT" \
        LD_EXPECTED_VARIANT="$EXPECTED_VARIANT" \
        LD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
        python3 - <<'PY'
import json
import os
import sys

path = os.environ["LD_MANIFEST_PATH"]
expected_binary_sha = os.environ["LD_BINARY_SHA256"]
expected_slug = os.environ.get("LD_CLI_SLUG", "")
expected_shape_point = os.environ.get("LD_CLI_SHAPE_POINT", "")
try:
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
except json.JSONDecodeError as exc:
    sys.stderr.write(
        f"label_driver: --measurement-manifest JSON did not parse: {exc}\n"
    )
    sys.exit(2)
except OSError as exc:
    sys.stderr.write(
        f"label_driver: --measurement-manifest could not be read: {exc}\n"
    )
    sys.exit(2)
if not isinstance(manifest, dict):
    sys.stderr.write(
        "label_driver: --measurement-manifest top-level value is not a JSON object\n"
    )
    sys.exit(2)
# Slug + shape_point cell-identity binding: a manifest captured
# for one cell MUST NOT be reused for another cell. Without
# this check, passing the wrong manifest (e.g. one whose binary
# happens to match by SHA but whose kernel_signature_sha256
# pins a DIFFERENT cell's kernel) would silently emit a label
# whose run-time CLI slug/shape_point pair carries the WRONG
# kernel signature. Downstream predictor joins would then read
# the manifest's kernel_signature for the CLI cell, poisoning
# the 7-tuple cell identity. Fail closed when the manifest's
# slug/shape_point do not match the CLI's cell.
manifest_slug = manifest.get("slug")
if expected_slug and manifest_slug != expected_slug:
    sys.stderr.write(
        f"label_driver: --measurement-manifest slug={manifest_slug!r} "
        f"does not match CLI cell slug={expected_slug!r}\n"
    )
    sys.exit(2)
manifest_shape = manifest.get("shape_point")
# Legacy portable-LCD aliases: the runners emit `timing_00` /
# `correctness_00` in their manifests, but sweep_repeatability_matrix
# passes the legacy `timing_shape` / `correctness_shape` aliases on
# the CLI. Normalize both sides to the runner-emission form before
# comparing so the documented timing/correctness-shape labeling
# path validates correctly.
_SHAPE_POINT_ALIASES = {
    "timing_shape": "timing_00",
    "correctness_shape": "correctness_00",
}
def _normalize_shape_point(name):
    if not isinstance(name, str):
        return name
    return _SHAPE_POINT_ALIASES.get(name, name)
if expected_shape_point and _normalize_shape_point(manifest_shape) != _normalize_shape_point(expected_shape_point):
    sys.stderr.write(
        f"label_driver: --measurement-manifest shape_point="
        f"{manifest_shape!r} does not match CLI cell shape_point="
        f"{expected_shape_point!r}\n"
    )
    sys.exit(2)
kernel_sig = manifest.get("kernel_signature_sha256")
if not isinstance(kernel_sig, str) or not kernel_sig:
    sys.stderr.write(
        "label_driver: --measurement-manifest is missing required field "
        "kernel_signature_sha256 (or value is empty / not a string)\n"
    )
    sys.exit(2)
manifest_binary_sha = manifest.get("binary_sha256")
if manifest_binary_sha is not None:
    if not isinstance(manifest_binary_sha, str) or not manifest_binary_sha:
        sys.stderr.write(
            "label_driver: --measurement-manifest binary_sha256 must be a "
            "non-empty string when present\n"
        )
        sys.exit(2)
    if manifest_binary_sha != expected_binary_sha:
        sys.stderr.write(
            "label_driver: --measurement-manifest binary_sha256 disagrees "
            f"with profiled binary: manifest={manifest_binary_sha!r} "
            f"actual={expected_binary_sha!r}\n"
        )
        sys.exit(2)
# Optional register-cap build-variant identity. Normalize the manifest's
# register_cap/variant_id through the SINGLE canonical normalizer (the
# dependency-free variant_identity module -- importable under this bare
# python3 because it pulls no jsonschema), and fail closed BEFORE NCU (this
# heredoc runs before the run manifest / .ncu-rep / sidecar are written) on
# an inconsistent / noncanonical-alias / negative-cap pair, or on an
# expected-vs-manifest mismatch. Base manifests (no variant fields) yield an
# empty pair and the run manifest + sidecar stay byte-identical.
script_dir = os.environ.get("LD_SCRIPT_DIR", "")
if script_dir and script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from variant_identity import variant_identity  # noqa: E402
# Strict type pre-check at the operator-input boundary: the shared normalizer
# TOLERATES absence (a missing field => base cell), so a PRESENT-but-malformed
# value (register_cap as a JSON string / bool, or an empty/non-string
# variant_id) would otherwise be silently downgraded to a base capture. A
# manifest that explicitly declares a (malformed) variant identity must fail
# closed, not run as base.
if "register_cap" in manifest and (
    not isinstance(manifest["register_cap"], int)
    or isinstance(manifest["register_cap"], bool)
):
    sys.stderr.write(
        "label_driver: --measurement-manifest register_cap is present but is "
        f"not an integer: {manifest['register_cap']!r}\n"
    )
    sys.exit(2)
if "variant_id" in manifest and (
    not isinstance(manifest["variant_id"], str) or not manifest["variant_id"]
):
    sys.stderr.write(
        "label_driver: --measurement-manifest variant_id is present but is not "
        f"a non-empty string: {manifest['variant_id']!r}\n"
    )
    sys.exit(2)
variant_record = {}
if "register_cap" in manifest:
    variant_record["register_cap"] = manifest["register_cap"]
if "variant_id" in manifest:
    variant_record["variant_id"] = manifest["variant_id"]
variant_id, register_cap, variant_error = variant_identity(variant_record)
if variant_error is not None:
    sys.stderr.write(
        "label_driver: --measurement-manifest build-variant identity is "
        f"inconsistent: {variant_error}\n"
    )
    sys.exit(2)
expected_variant = os.environ.get("LD_EXPECTED_VARIANT", "")
if expected_variant:
    exp_variant_id, _exp_cap, exp_error = variant_identity(
        {"variant_id": expected_variant}
    )
    if exp_error is not None:
        sys.stderr.write(
            f"label_driver: --expected-variant {expected_variant!r} is not a "
            f"canonical register_cap_<int> token: {exp_error}\n"
        )
        sys.exit(2)
    if variant_id != exp_variant_id:
        sys.stderr.write(
            "label_driver: --measurement-manifest variant_id="
            f"{variant_id!r} does not match --expected-variant "
            f"{exp_variant_id!r}\n"
        )
        sys.exit(2)
register_cap_out = "" if register_cap is None else str(register_cap)
# Pipe-delimited so the bash side parses three fields with `IFS='|' read`.
# kernel_signature_sha256 is hex, register_cap is an int, variant_id is
# register_cap_<int> -- none contain a pipe.
sys.stdout.write(f"{kernel_sig}|{register_cap_out}|{variant_id}")
PY
    )" || exit $?
    # The heredoc emits kernel_sig|register_cap|variant_id (the latter two
    # empty for a base manifest). IFS='|' (non-whitespace) preserves the empty
    # trailing fields so a base capture leaves LD_REGISTER_CAP/LD_VARIANT_ID
    # empty.
    IFS='|' read -r KERNEL_SIGNATURE_SHA256 LD_REGISTER_CAP LD_VARIANT_ID <<< "$MANIFEST_VALIDATION"
    KERNEL_SIGNATURE_SOURCE="measurement_manifest"
fi

# Source-commit provenance: capture the project's HEAD at the time of
# the NCU collection invocation. The sweep harness's verdict-table
# joins admitted cells against the comparison runner's contract on the
# 7-field key, of which source_commit is one. Use the safe-capture
# pattern (non-zero exit normalizes to the literal "unknown") so a
# missing-git environment does not abort the driver.
safe_git_head() {
    local out
    # `|| true` keeps the assignment's exit status at 0 under
    # `set -e` when the git invocation fails (non-git
    # environment, exported tarball, missing git binary).
    # Without this guard, bash's `set -e` semantics around
    # assignments-with-command-substitution vary by version
    # and can abort the driver before the empty-output branch
    # below has a chance to normalize the value to "unknown".
    out="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
    if [[ -z "$out" ]]; then
        printf 'unknown'
        return
    fi
    printf '%s' "${out%%$'\n'*}"
}

if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
SOURCE_COMMIT="$(safe_git_head)"

# Emit the per-launch run manifest at the start of the NCU collection
# block. The manifest carries the join-key fields the comparison
# runner expects: source_commit, binary_sha256, kernel_signature_sha256
# plus the cell-identity fields (arch, slug, shape_point, repeat_index,
# launch_index). The aggregator reads every manifest in a cell
# directory, validates them against the per-launch sidecars, and
# enriches the admitted-cell record with the runner-joinable 7-field
# key. Use the same safe Python-heredoc emitter the sidecar uses so
# shell-captured strings cannot break the JSON.
RUN_MANIFEST_PATH="${CELL_DIR}/label_run_manifest_${REPEAT_INDEX}_${LAUNCH_INDEX}.json"
# Orphan-manifest cleanup discipline: under `set -euo pipefail`
# any failure between this manifest emission and the sidecar
# emission below (NCU collection failure, metric-extraction
# failure, missing L2/L1TEX values) would leave the manifest
# on disk without a matching label sidecar. The aggregator's
# `_validate_provenance` pass would then treat the orphan
# manifest as a stale duplicate against later valid repeats,
# blocking provenance admission for the cell. Install an EXIT
# trap that removes the manifest if the script exits before
# the sidecar emission clears the trap (the cleared trap is a
# no-op on legitimately-completed runs). The trap path uses
# the EXIT pseudo-signal which fires on every script exit
# regardless of cause (signal, error, or normal exit) — the
# `_RUN_MANIFEST_NEEDS_CLEANUP` flag below toggles the
# behavior so successful runs preserve the manifest.
_RUN_MANIFEST_NEEDS_CLEANUP=1
_label_driver_probe_output_storage() {
    # The failing command may be an external profiler/writer whose errno is
    # otherwise unavailable to the Python parent.  Probe only CELL_DIR (the
    # output domain), and emit an exact marker only when that explicit output
    # operation reports a closed storage errno.  Input failures therefore do
    # not become storage receipts merely because they carry an OSError cause.
    set +e
    LABEL_DRIVER_STORAGE_PROBE_DIR="$CELL_DIR" python3 - <<'PY'
import errno
import json
import os
from pathlib import Path

root = Path(os.environ["LABEL_DRIVER_STORAGE_PROBE_DIR"])
probe = root / f".label_driver_storage_probe_{os.getpid()}"
fd = None
closed = {
    errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS,
    *(
        value for value in (
            getattr(errno, "ENXIO", None),
            getattr(errno, "ESTALE", None),
        ) if value is not None
    ),
}
failure = None
try:
    fd = os.open(
        probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
except OSError as exc:
    if exc.errno in closed:
        failure = exc
if failure is None:
    try:
        if os.write(fd, b"probe\n") != 6:
            raise OSError(errno.EIO, "short label-driver output probe")
    except OSError as exc:
        if exc.errno in closed:
            failure = exc
        else:
            raise
if failure is None:
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in closed:
            failure = exc
        else:
            raise
if failure is None:
    try:
        os.close(fd)
        fd = None
    except OSError as exc:
        if exc.errno in closed:
            failure = exc
        else:
            raise
if failure is None:
    try:
        probe.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        if exc.errno in closed:
            failure = exc
        else:
            raise
if failure is not None:
    # First closed output fault wins.  Do not close, unlink, or otherwise
    # mutate the failed medium after it is observed; process exit drops fd.
    print(
        "LABEL_DRIVER_STORAGE_FAILURE:"
        + json.dumps({
            "schema_version": 1,
            "errno": failure.errno,
            "operation": "label_driver_output_probe",
        }, sort_keys=True, separators=(",", ":")),
        file=os.sys.stderr,
    )
PY
    set -e
    return 0
}
trap '_label_driver_probe_output_storage' ERR
trap '
    # Canonical chunk capture sets LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE=1
    # for immutable attempt directories.  It is a conservative, robust
    # storage-stop hand-off: never issue rm against a medium that may have
    # reported ENOSPC/EDQUOT/EIO/EROFS.  The attempt journal retains and types
    # semantic failures too, so preserving those bytes cannot admit them.
    if [[ "${LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE:-0}" != "1" && "${_RUN_MANIFEST_NEEDS_CLEANUP:-0}" == "1" && -f "$RUN_MANIFEST_PATH" ]]; then
        rm -f "$RUN_MANIFEST_PATH"
    fi
    if [[ "${LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE:-0}" != "1" && "${_RUN_MANIFEST_NEEDS_CLEANUP:-0}" == "1" && -n "${PARSER_DIAGNOSTICS_PATH:-}" && -f "${PARSER_DIAGNOSTICS_PATH:-}" ]]; then
        rm -f "$PARSER_DIAGNOSTICS_PATH"
    fi
    if [[ "${LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE:-0}" != "1" && "${_RUN_MANIFEST_NEEDS_CLEANUP:-0}" == "1" && -n "${LAUNCH_OBSERVATION_PATH:-}" && -f "${LAUNCH_OBSERVATION_PATH:-}" ]]; then
        rm -f "$LAUNCH_OBSERVATION_PATH"
    fi
' EXIT
LD_RUN_MANIFEST_PATH="$RUN_MANIFEST_PATH" \
LD_SOURCE_COMMIT="$SOURCE_COMMIT" \
LD_BINARY_SHA256="$BINARY_SHA256" \
LD_KERNEL_SIG_SHA256="$KERNEL_SIGNATURE_SHA256" \
LD_KERNEL_SIG_SOURCE="$KERNEL_SIGNATURE_SOURCE" \
LD_DEVICE="$DEVICE" \
LD_SLUG="$SLUG" \
LD_SHAPE_POINT="$SHAPE_POINT" \
LD_REPEAT_INDEX="$REPEAT_INDEX" \
LD_LAUNCH_INDEX="$LAUNCH_INDEX" \
LD_REGISTER_CAP="$LD_REGISTER_CAP" \
LD_VARIANT_ID="$LD_VARIANT_ID" \
python3 - <<'PY'
import json
import os
manifest = {
    "schema_version": 1,
    "source_commit": os.environ["LD_SOURCE_COMMIT"],
    "binary_sha256": os.environ["LD_BINARY_SHA256"],
    "kernel_signature_sha256": os.environ["LD_KERNEL_SIG_SHA256"],
    "kernel_signature_source": os.environ["LD_KERNEL_SIG_SOURCE"],
    "arch": os.environ["LD_DEVICE"],
    "slug": os.environ["LD_SLUG"],
    "shape_point": os.environ["LD_SHAPE_POINT"],
    "repeat_index": int(os.environ["LD_REPEAT_INDEX"]),
    "launch_index": int(os.environ["LD_LAUNCH_INDEX"]),
}
# Optional build-variant identity (validated upstream against the canonical
# domain). Present ONLY for a variant cell; a base run manifest is unchanged.
_variant_id = os.environ.get("LD_VARIANT_ID", "")
_register_cap = os.environ.get("LD_REGISTER_CAP", "")
if _variant_id:
    manifest["variant_id"] = _variant_id
    if _register_cap:
        manifest["register_cap"] = int(_register_cap)
with open(os.environ["LD_RUN_MANIFEST_PATH"], "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

# AC2 (R4-B, R4-C) + v2 protocol: launch filtering + EXPLICIT clock control.
# Launch filtering locks the profiled launch to exactly the index recorded in
# the sidecar's launch_index field. Clock control is always passed explicitly
# (new Rule 2 — the NCU default changed base->boost in 2026.1): "none" when the
# host clocks are externally locked, base/boost to delegate locking to NCU.
# The raw lookup-hit / lookup-miss counter pairs for each .pct ratio
# enable the parser's raw-counter ratio fallback when NCU emits an
# out-of-range .pct value (observed on sm_120 / Blackwell consumer
# for some kernels). The counters are mathematically bounded; the
# parser uses them only when the .pct overshoot exceeds the
# [100, 105] saturation-rounding window, and emits a typed
# diagnostic (`ncu_lts_pct_overcount_raw_counter_ratio_used`) for
# the audit trail.
# The two mandatory hit-rate percentages + their raw-counter fallback pairs.
# Any operator-requested --unit-metric ids are appended so NCU collects them in
# the same single replay; the parser then routes each through its registry unit.
NCU_METRICS="lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,l1tex__t_sectors_lookup_hit.sum,l1tex__t_sectors_lookup_miss.sum"
if ((${#UNIT_METRICS[@]} > 0)); then
    UNIT_METRICS_JOINED="$(IFS=,; echo "${UNIT_METRICS[*]}")"
    NCU_METRICS="${NCU_METRICS},${UNIT_METRICS_JOINED}"
fi
NCU_COLLECT_ARGS=(
    --target-processes all
    --replay-mode kernel
    --metrics "$NCU_METRICS"
    --section MemoryWorkloadAnalysis
    --launch-skip "$LAUNCH_INDEX"
    --launch-count 1
    --clock-control "$NCU_CLOCK_CONTROL"
    --export "$REPORT_PATH"
)
if [[ "$SHAPE_BINDING" == "logical-workload" ]]; then
    # Adapter-backed programs perform setup/reference work before the selected
    # candidate.  Their capture driver opens one explicit cudaProfiler range;
    # profiling from process start would attribute launch 0 to setup instead.
    NCU_COLLECT_ARGS+=(--profile-from-start off)
fi
if [[ -n "$KERNEL_NAME" ]]; then
    NCU_COLLECT_ARGS+=(--kernel-name "$KERNEL_NAME")
fi
if [[ -n "$CACHE_CONTROL" ]]; then
    NCU_COLLECT_ARGS+=(--cache-control "$CACHE_CONTROL")
fi

"$NCU_BIN" \
    "${NCU_COLLECT_ARGS[@]}" \
    "$BINARY_PATH" \
    "${APP_ARGS[@]}" \
    >/dev/null

# AC2 (R4-D): safe metric extraction. The Python helper imports the
# .ncu-rep* report and parses the raw-page CSV via Python's csv module,
# selecting rows by exact metric name from the documented "Metric Name"
# column. The old shell-awk parser silently returned the first numeric
# token of a row, which could be a row id, PID, context id, stream id,
# or compute capability — a class of bug the helper makes unreachable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Persist the parser's non-fatal clamp diagnostics into a sibling JSON
# file so the sidecar emitter can merge them into the per-launch
# sidecar's parser_diagnostics field. The file is co-located with the
# .ncu-rep report and removed after successful sidecar emission; the
# orphan-cleanup trap below also removes it on any failure between
# manifest emission and sidecar emission.
PARSER_DIAGNOSTICS_PATH="${REPORT_PATH%.ncu-rep}.parser_diagnostics.json"
LAUNCH_OBSERVATION_PATH="${REPORT_PATH%.ncu-rep}.launch_observation.json"
# Unit-tagged metrics output path; empty when no --unit-metric was requested
# (an empty SIDECAR_UNIT_TAGGED_METRICS_PATH means "not requested" to the
# emitter, which then omits the optional unit_tagged_metrics field).
UNIT_TAGGED_PATH=""
PARSER_ARGS=(
    --ncu "$NCU_BIN"
    --report "$REPORT_PATH"
    --metric "lts__t_sector_hit_rate.pct"
    --metric "l1tex__t_sector_hit_rate.pct"
    --diagnostics-json "$PARSER_DIAGNOSTICS_PATH"
    --launch-observation-json "$LAUNCH_OBSERVATION_PATH"
    --requested-launch-skip "$LAUNCH_INDEX"
    --requested-launch-count 1
)
if ((${#UNIT_METRICS[@]} > 0)); then
    UNIT_TAGGED_PATH="${REPORT_PATH%.ncu-rep}.unit_tagged.json"
    for _um in "${UNIT_METRICS[@]}"; do
        PARSER_ARGS+=(--unit-metric "$_um")
    done
    PARSER_ARGS+=(--unit-tagged-json "$UNIT_TAGGED_PATH")
fi
METRICS_OUTPUT="$(python3 "$SCRIPT_DIR/parse_ncu_metrics.py" "${PARSER_ARGS[@]}")"

CACHE_HIT_RATE_L2=""
CACHE_HIT_RATE_L1TEX=""
while IFS=$'\t' read -r metric value; do
    case "$metric" in
        lts__t_sector_hit_rate.pct)
            CACHE_HIT_RATE_L2="$value"
            ;;
        l1tex__t_sector_hit_rate.pct)
            CACHE_HIT_RATE_L1TEX="$value"
            ;;
    esac
done <<< "$METRICS_OUTPUT"

if [[ -z "$CACHE_HIT_RATE_L2" ]]; then
    echo "label_driver: failed to extract L2 sector hit rate from $REPORT_PATH" >&2
    exit 3
fi

# AC3 (R4-E/F): L1/TEX is required for label admission, so the parser
# is also required to return the L1/TEX value. parse_ncu_metrics.py
# already errors out if any requested metric is missing — but be
# explicit at the driver layer too, so a failure shows up in stderr
# rather than as a silently L1-missing sidecar.
if [[ -z "$CACHE_HIT_RATE_L1TEX" ]]; then
    echo "label_driver: failed to extract L1/TEX sector hit rate from $REPORT_PATH" >&2
    exit 3
fi

CONFIDENCE_LEVEL="0.95"
UNITS_TAG="sector_hit_rate_fraction"
AGGREGATION_TAG="kernel"

# Sidecar emission (R5). The JSON construction now factors out of the
# inline heredoc into `tools/local_labels/sidecar_emitter.py` so the
# sidecar shape (including the optional `parser_diagnostics` field) is
# unit-tested at `tools/local_labels/test_sidecar_emitter.py`. Every
# dynamic value is exported via the environment so the bash shell does
# NOT interpolate values into the Python source; the emitter reads them
# through os.environ and json.dump encodes each string with proper
# escaping. This makes it impossible for shell-captured profiler/driver
# text containing double quotes, backslashes, or other Python
# metacharacters to break the emitter.
#
# `SIDECAR_PARSER_DIAGNOSTICS_PATH` points the emitter at the parser
# diagnostics JSON written above. When the file is non-empty (the
# parser clamped one or more `lts__t_sector_hit_rate.pct` overshoots),
# its contents merge into the sidecar's `parser_diagnostics` field.
# v2.1 protocol: record the observed host clock/persistence state at emission
# time alongside the explicit NCU clock-control mode (CLAUDE.md Rule 2 — the
# clock-lock state is part of every capture's provenance). Read-only queries.
# Semantics caveat (recorded in the schema description): this is ONE sample
# taken AFTER the capture — a mid-capture excursion that clears by emission
# time is not visible here; the NCU metrics themselves are the in-capture
# evidence. LABEL_DRIVER_GPU_INDEX selects the device on multi-GPU hosts
# (default 0). The throttle field name uses the current
# clocks_event_reasons spelling with fallback to the deprecated
# clocks_throttle_reasons alias (dropped in newer drivers).
_HOST_GPU_INDEX="${LABEL_DRIVER_GPU_INDEX:-0}"
_HOST_CLOCK_STATE="$(nvidia-smi -i "$_HOST_GPU_INDEX" --query-gpu=persistence_mode,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader 2>/dev/null | head -1)"
if [[ -z "$_HOST_CLOCK_STATE" ]]; then
    _HOST_CLOCK_STATE="$(nvidia-smi -i "$_HOST_GPU_INDEX" --query-gpu=persistence_mode,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1)"
fi
HOST_PERSISTENCE_MODE="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f1 | xargs)"
HOST_SM_CLOCK_MHZ="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f2 | xargs | grep -oE '^[0-9]+' || echo '')"
HOST_MAX_SM_CLOCK_MHZ="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f3 | xargs | grep -oE '^[0-9]+' || echo '')"
HOST_MEM_CLOCK_MHZ="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f4 | xargs | grep -oE '^[0-9]+' || echo '')"
HOST_POWER_W="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f5 | xargs | grep -oE '^[0-9]+\.?[0-9]*' || echo '')"
HOST_TEMP_C="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f6 | xargs | grep -oE '^[0-9]+' || echo '')"
HOST_THROTTLE_REASONS="$(echo "$_HOST_CLOCK_STATE" | cut -d, -f7 | xargs)"

SIDECAR_NCU_CLOCK_CONTROL="$NCU_CLOCK_CONTROL" \
SIDECAR_NCU_CACHE_CONTROL="$CACHE_CONTROL" \
SIDECAR_NCU_REPLAY_MODE="kernel" \
SIDECAR_HOST_GPU_INDEX="$_HOST_GPU_INDEX" \
SIDECAR_HOST_MEM_CLOCK_MHZ="$HOST_MEM_CLOCK_MHZ" \
SIDECAR_HOST_POWER_W="$HOST_POWER_W" \
SIDECAR_HOST_TEMP_C="$HOST_TEMP_C" \
SIDECAR_HOST_PERSISTENCE_MODE="$HOST_PERSISTENCE_MODE" \
SIDECAR_HOST_SM_CLOCK_MHZ="$HOST_SM_CLOCK_MHZ" \
SIDECAR_HOST_MAX_SM_CLOCK_MHZ="$HOST_MAX_SM_CLOCK_MHZ" \
SIDECAR_HOST_THROTTLE_REASONS="$HOST_THROTTLE_REASONS" \
SIDECAR_DEVICE="$DEVICE" \
SIDECAR_SLUG="$SLUG" \
SIDECAR_SHAPE_POINT="$SHAPE_POINT" \
SIDECAR_REPEAT_INDEX="$REPEAT_INDEX" \
SIDECAR_LAUNCH_INDEX="$LAUNCH_INDEX" \
SIDECAR_BINARY_SHA256="$BINARY_SHA256" \
SIDECAR_PROFILER_VERSION="$PROFILER_VERSION" \
SIDECAR_CUDA_VERSION="$CUDA_VERSION" \
SIDECAR_DRIVER_VERSION="$DRIVER_VERSION" \
SIDECAR_GPU_UUID_SHA256="$GPU_UUID_SHA256" \
SIDECAR_AGGREGATION="$AGGREGATION_TAG" \
SIDECAR_UNITS="$UNITS_TAG" \
SIDECAR_CONFIDENCE_LEVEL="$CONFIDENCE_LEVEL" \
SIDECAR_WARMUP_COUNT="$EFFECTIVE_WARMUP_COUNT" \
SIDECAR_HIT_RATE_L2="$CACHE_HIT_RATE_L2" \
SIDECAR_HIT_RATE_L1TEX="$CACHE_HIT_RATE_L1TEX" \
SIDECAR_PARSER_DIAGNOSTICS_PATH="$PARSER_DIAGNOSTICS_PATH" \
SIDECAR_UNIT_TAGGED_METRICS_PATH="$UNIT_TAGGED_PATH" \
SIDECAR_REGISTER_CAP="$LD_REGISTER_CAP" \
SIDECAR_VARIANT_ID="$LD_VARIANT_ID" \
python3 "$SCRIPT_DIR/sidecar_emitter.py" > "$SIDECAR_PATH"

# Sidecar emission succeeded: the run manifest is no longer an
# orphan. Clear the cleanup flag so the EXIT trap installed
# before the manifest emission becomes a no-op on this run.
_RUN_MANIFEST_NEEDS_CLEANUP=0

# The parser diagnostics file's contents are now persisted into the
# sidecar; remove the sibling temp file so the sweep dir stays clean.
rm -f "$PARSER_DIAGNOSTICS_PATH"
# Same for the unit-tagged metrics temp file (when one was produced).
if [[ -n "$UNIT_TAGGED_PATH" ]]; then
    rm -f "$UNIT_TAGGED_PATH"
fi

echo "label_driver: wrote $SIDECAR_PATH"
