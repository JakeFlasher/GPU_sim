#!/usr/bin/env bash
# Cloud preflight: emit a reproducible toolchain + GPU provenance
# manifest BEFORE any measurement run on the sm_120 cloud target or
# the local-development sm_86 / sm_120-laptop hosts (per CLAUDE.md
# "Two-Device Local Development Model" — both arches admitted at
# preflight; per-arch nvcc -arch token chosen from the local CC).
#
# Non-root contract: the cloud GPU host denies hardware-counter access
# and denies driver-level operations (no clock control, no kernel-module
# parameter toggle). Container apt-sudo is fine; GPU-driver root is not.
# preflight therefore probes only the things that actually work in this
# regime: arch identity, toolchain versions, NVBit injection,
# cluster-attribute API, CUPTI Activity API availability.
#
# Usage:
#   tools/cloud/preflight.sh [--out PATH] [--mock-input DIR] [--help]
#
# Flags:
#   --out PATH         Write the manifest JSON to PATH (default: stdout).
#   --mock-input DIR   Read toolchain outputs from canned files under DIR
#                      instead of calling `nvidia-smi` / `nvcc`.
#                      Expected filenames: nvidia_smi.csv, nvcc_version.txt,
#                      nvbit_build_hash.txt, permissions.json,
#                      cluster_attr_probe.txt, cupti_activity_probe.txt,
#                      nvbit_injection_probe.txt, cutlass_version.txt,
#                      flashinfer_version.txt. Missing files default to
#                      "unknown" / empty / absent-object. Mock mode also
#                      downgrades the hard-exit assertions to warnings so
#                      offline tests can construct malformed manifests.
#   --help             Print this header and exit.
#
# Output: structured JSON manifest with the toolchain + GPU fields the
# downstream preflight_manifest_validator consumes. A 'timestamp' field
# is included but is excluded from the canonical SHA-256 digest used as
# `preflight_manifest_sha256` on collection artifacts.
#
# Hard exit categories (each exits non-zero BEFORE emitting the manifest;
# mock mode downgrades all of these to stderr warnings):
#   NvbitUnavailable      — NVBit injection smoke probe failed. The
#                           probe compiles a minimal CUDA stub, runs it
#                           under `CUDA_INJECTION64_PATH=<tracer.so>` with the
#                           pinned protobuf runtime on `LD_LIBRARY_PATH`, then
#                           requires pinned DSO resolution, a parseable root
#                           with a launch, instruction-bearing threadblock
#                           protobufs, a passing per-instruction contract walk,
#                           and positive stats.csv counters. The tracer's sha256
#                           is only recorded after that admission succeeds.
#   TargetArchMismatch    — detected compute capability is outside both
#                           admitted arch families: sm_86 (CC 8.6) and
#                           sm_120 (CC 12.x). Server arches (sm_80 /
#                           sm_89 / sm_90a / server sm_120) and other
#                           arches are rejected here.
#
# Additional capability fields emitted into the manifest (checked by
# the downstream validator rather than hard-exited here):
#   capabilities.cluster_attribute_nonportable_cluster_size_allowed
#     — real `nvcc -arch=${GPU_NVCC_ARCH}` compile + optional run of a tiny
#       CUDA stub that references
#       `cudaFuncAttributeNonPortableClusterSizeAllowed`. Values:
#       `available`, `compile_failed`, `runtime_failed`,
#       `probe_failed`, `mock_input_missing`.
#   capabilities.cupti_activity_api_available
#     — header / shared-library presence check for the CUPTI Activity
#       API (`cupti.h` + libcupti.so). Preserved for backward
#       compatibility with the v2 schema. Values: `available`,
#       `header_missing`, `library_missing`, `probe_failed`,
#       `mock_input_missing`.
#   capabilities.cupti_activity_api_runtime  (schema v3)
#     — full compile + link + run of a CUPTI Activity stub that
#       calls `cuptiActivityRegisterCallbacks` and
#       `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)`. CUPTI
#       Activity is the wall-clock telemetry surface used by
#       counter-free measurement (kernel timestamps, memcpy events,
#       runtime API tracing); it does NOT require profiling
#       permissions. Values, in status precedence (earliest-
#       reachable failure wins):
#         mock_input_missing < compile_failed < link_failed
#         < runtime_failed < insufficient_privilege < cc_unsupported
#         < available
#       `insufficient_privilege` is surfaced when CUPTI returns
#       `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` (admin-gated host
#       under the non-root contract — Activity should be allowed
#       but a misconfigured host can still deny it).
#       `cc_unsupported` is surfaced when CUPTI rejects the call
#       under Confidential Computing mode without DevTools.
#   capabilities.cc_mode  (schema v3)
#     — Confidential Computing mode of the GPU. Values:
#       `off` | `devtools` | `on` | `unsupported_query` | `unknown`.
#       The validator hard-rejects collection evidence when
#       `cc_mode == "on"` (DevTools off) via
#       `ConfidentialComputingNotSupportedForCollection`.
#   capabilities.nvml_capability_matrix  (schema v3)
#     — degraded-NVML probe summary. Values:
#       `full` | `partial_power_only` | `partial_temp_only` | `denied`.
#       Records what NVML telemetry the cloud container exposes;
#       referenced by the perturbation-sweep covariate-regression
#       analysis. The validator does NOT hard-reject on `denied`
#       (the run can proceed with degraded-accuracy regression per
#       cloud_protocol.md §9), but downstream consumers must
#       declare the regime explicitly.
#   capabilities.mps_active  (schema v4)
#     — boolean: True iff an NVIDIA MPS (Multi-Process Service)
#       daemon or server process is detectable. Record-only
#       deployment-mode signal: the validator emits this field on
#       the manifest and applies a schema-correctness check, but
#       does NOT hard-reject on True. Downstream artifacts read this
#       field to reason about CUPTI per-client attribution semantics
#       under MPS (per-client kernel attribution is the only
#       attribution that survives an active MPS server).
#   capabilities.vgpu_active  (schema v4)
#     — boolean: True iff the GPU is exposed via NVIDIA vGPU
#       (probed via `nvidia-smi vgpu -q`). Record-only deployment-
#       mode signal alongside `mps_active`: the validator emits this
#       field on the manifest and applies a schema-correctness
#       check, but does NOT hard-reject on True. Downstream
#       artifacts read this field to choose `_KERNEL` activity-kind
#       fallback when HES source-correlation is unavailable under
#       vGPU.
#
# MOCK_DIR inputs (used when --mock-input is set):
#   nvidia_smi.csv, nvcc_version.txt, cluster_attr_probe.txt,
#   cupti_activity_probe.txt, cupti_runtime_probe.txt,
#   cc_mode_probe.txt, nvml_capability_matrix_probe.txt,
#   mps_active_probe.txt (`true` / `false`),
#   vgpu_active_probe.txt (`true` / `false`),
#   nvbit_injection_probe.txt, nvbit_build_hash.txt,
#   permissions.json.
#
# Environment variables:
#   CLOUD_PREFLIGHT_REPO_ROOT   Override for the repo root (default: script
#                               location's great-grandparent dir).
#   CLOUD_PREFLIGHT_CUPTI_SEARCH_ROOTS
#                               Semicolon-separated absolute paths to
#                               prepend to the hard-coded CUPTI search
#                               roots. Used by tests / behavioral smokes
#                               to stage a fake CUDA/CUPTI install
#                               without root privilege.
#   CLOUD_PREFLIGHT_PYTHON      Python interpreter carrying the matching
#                               google.protobuf runtime. Defaults to the
#                               repository .venv, then python3. Real NVBit
#                               readiness also requires all nine tracked
#                               `_pb_generated/*_pb2.py` authorities; missing
#                               bytes are repository corruption, not a build.
#   CUDA_VISIBLE_DEVICES        Forwarded into nvidia-smi queries; unset
#                               values default to device 0.
#   CLOUD_PREFLIGHT_MPS_DEFAULT_PIPE_DIR
#                               Override for the default MPS pipe
#                               directory used as a fallback signal when
#                               CUDA_MPS_PIPE_DIRECTORY is unset
#                               (default: /tmp/nvidia-mps). Used by
#                               host-independent behavior smokes to
#                               stage stale-or-absent default-pipe-dir
#                               cases without depending on the host
#                               filesystem.
#   CLOUD_PREFLIGHT_PROBE_ONLY  Restricts the script to a named probe
#                               subset and exits 0 after emitting the
#                               probe results (no manifest emitted).
#                               Recognized values:
#                                 mps_vgpu — emit two lines
#                                            "mps_active=<bool>" +
#                                            "vgpu_active=<bool>" then
#                                            exit 0. Invokes the real
#                                            (non-mock) MPS / vGPU
#                                            detection paths only;
#                                            ignores --mock-input.
#                               Used by host-independent behavior
#                               smokes that fake `pgrep` and
#                               `nvidia-smi` on PATH.
#
# Determinism: calling preflight twice on the same host produces byte-
# identical JSON modulo the `timestamp` string. The companion validator
# strips `timestamp` before hashing for cross-artifact referencing.

set -euo pipefail

usage() {
    sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" \
        | sed -e 's/^# \{0,1\}//' \
        | sed '/^set -euo pipefail$/d'
}

OUT_PATH=""
MOCK_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)                     OUT_PATH="$2"; shift 2 ;;
        --mock-input)              MOCK_DIR="$2"; shift 2 ;;
        --help|-h)                 usage; exit 0 ;;
        *)                         echo "cloud_preflight: unknown argument: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CLOUD_PREFLIGHT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NVBIT_SO_PATH="${REPO_ROOT}/tools/nvbit_tracer/tracer_tool/tracer_tool.so"
NVBIT_PB_LIB="${REPO_ROOT}/.toolchain/pb/lib"
NVBIT_RUNTIME_CHECK="${SCRIPT_DIR}/../nvbit_tracer/runtime_env.py"
NVBIT_TRACE_ADMISSION="${SCRIPT_DIR}/../nvbit_tracer/trace_admission.py"
if [[ -n "${CLOUD_PREFLIGHT_PYTHON:-}" ]]; then
    NVBIT_PYTHON="${CLOUD_PREFLIGHT_PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    NVBIT_PYTHON="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${SCRIPT_DIR}/../../.venv/bin/python" ]]; then
    # Behavioral smokes override REPO_ROOT with a staged fake tracer tree;
    # the validator itself still belongs to this checkout.
    NVBIT_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
else
    NVBIT_PYTHON="python3"
fi

# sudo's `secure_path` (common on Ubuntu cloud images) strips the user's
# PATH and does not include /usr/local/cuda/bin by default, so `nvcc`
# can vanish inside the elevated shell even though it works as the
# user. Probe well-known CUDA install locations and prepend the first
# one that carries the binary. No-op if the binary is already
# reachable.
ensure_cuda_bin_on_path() {
    local bin="$1"
    if command -v "${bin}" >/dev/null 2>&1; then
        return 0
    fi
    local candidate_roots=(
        /usr/local/cuda
        /usr/local/cuda-13 /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13.0
        /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4
        /opt/cuda
        /opt/cuda-13 /opt/cuda-13.2 /opt/cuda-13.1 /opt/cuda-13.0
        /opt/cuda-12 /opt/cuda-12.8 /opt/cuda-12.6 /opt/cuda-12.4
    )
    for root in "${candidate_roots[@]}"; do
        if [[ -x "${root}/bin/${bin}" ]]; then
            PATH="${root}/bin:${PATH}"
            export PATH
            return 0
        fi
    done
    return 1
}

# Opportunistic PATH healing — don't fail here; the per-tool checks
# below decide whether a specific binary's absence is fatal.
if [[ -z "${MOCK_DIR}" ]]; then
    ensure_cuda_bin_on_path nvcc >/dev/null 2>&1 || true
    ensure_cuda_bin_on_path nvidia-smi >/dev/null 2>&1 || true
fi

trim() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

fatal_or_warn() {
    local category="$1"; shift
    local message="$*"
    if [[ -n "${MOCK_DIR}" ]]; then
        printf 'cloud_preflight: WARNING %s: %s\n' "${category}" "${message}" >&2
    else
        printf 'cloud_preflight: ERROR %s: %s\n' "${category}" "${message}" >&2
        exit 1
    fi
}

read_mock_or_empty() {
    local basename="$1"
    if [[ -n "${MOCK_DIR}" ]] && [[ -f "${MOCK_DIR}/${basename}" ]]; then
        cat "${MOCK_DIR}/${basename}"
    else
        printf ''
    fi
}

# --- Real (non-mock) MPS / vGPU probe functions ----------------------
# These are factored into named functions so the host-independent
# behavior smokes can call them in isolation via the
# CLOUD_PREFLIGHT_PROBE_ONLY=mps_vgpu test hook with `pgrep` and
# `nvidia-smi` faked on PATH. Both functions print `true` or `false`
# to stdout and have no shared state with the rest of the script.

# probe_mps_active_real: returns `true` iff an NVIDIA MPS daemon /
# server / pipe-directory signal is detectable. The probe requires
# direct daemon/server evidence — the GPU compute-mode setting alone
# is NOT proof of an MPS server (the recommended-with-MPS mode can be
# set on a GPU that never has the MPS daemon launched against it), so
# compute-mode is intentionally not a positive signal.
#
# Daemon/server signals (any one present flips the field):
#   - `nvidia-cuda-mps-control` : the canonical MPS control daemon
#     (started by `nvidia-cuda-mps-control -d`).
#   - `nvidia-cuda-mps-server`  : the per-user MPS server process
#     (spawned by the control daemon when a client connects).
#   - `CUDA_MPS_PIPE_DIRECTORY` : the env var documented by NVIDIA as
#     the MPS client/daemon handoff path; if it is set AND the
#     directory exists, the workload was configured for MPS.
#   - `${CLOUD_PREFLIGHT_MPS_DEFAULT_PIPE_DIR:-/tmp/nvidia-mps}` :
#     the documented default pipe directory used when
#     CUDA_MPS_PIPE_DIRECTORY is unset.
probe_mps_active_real() {
    if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
        printf 'true\n'; return 0
    fi
    if pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
        printf 'true\n'; return 0
    fi
    if [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]] && [[ -d "${CUDA_MPS_PIPE_DIRECTORY}" ]]; then
        printf 'true\n'; return 0
    fi
    local default_pipe_dir="${CLOUD_PREFLIGHT_MPS_DEFAULT_PIPE_DIR:-/tmp/nvidia-mps}"
    if [[ -d "${default_pipe_dir}" ]]; then
        printf 'true\n'; return 0
    fi
    printf 'false\n'
}

# probe_vgpu_active_real: returns `true` iff `nvidia-smi vgpu -q`
# enumerates at least one active vGPU instance. The query is the
# documented verbose query for active vGPU instances on the host or
# in the guest:
#   - On bare-metal hosts (no vGPU support compiled into the driver)
#     the command exits non-zero -> `false`.
#   - On vGPU-capable drivers without an active vGPU partition the
#     output contains `No supported devices in vGPU mode` -> `false`.
#   - On a host or guest with active vGPU instances the output
#     enumerates them with `Active vGPUs : <count>` and per-instance
#     `vGPU ID :` lines -> `true`.
# The query is cheap (single nvidia-smi call) and side-effect-free.
probe_vgpu_active_real() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf 'false\n'; return 0
    fi
    local vgpu_query_out
    vgpu_query_out="$(nvidia-smi vgpu -q 2>/dev/null)" || { printf 'false\n'; return 0; }
    if [[ -z "${vgpu_query_out}" ]]; then
        printf 'false\n'; return 0
    fi
    if grep -q "No supported devices in vGPU mode" <<<"${vgpu_query_out}"; then
        printf 'false\n'; return 0
    fi
    if grep -qE '^[[:space:]]*(Active vGPUs[[:space:]]*:[[:space:]]*[1-9][0-9]*|vGPU ID[[:space:]]*:)' <<<"${vgpu_query_out}"; then
        printf 'true\n'; return 0
    fi
    printf 'false\n'
}

# --- CLOUD_PREFLIGHT_PROBE_ONLY test hook ----------------------------
# Lets host-independent behavior smokes invoke just the real MPS /
# vGPU probes against fake `pgrep` / `nvidia-smi` on PATH without
# triggering any other probe (GPU query, nvcc, CUPTI smoke, etc.).
# Recognized values:
#   mps_vgpu — emit two lines `mps_active=<bool>` + `vgpu_active=<bool>`
#              and exit 0. Ignores --mock-input.
case "${CLOUD_PREFLIGHT_PROBE_ONLY:-}" in
    "")
        ;;
    mps_vgpu)
        printf 'mps_active=%s\n' "$(probe_mps_active_real)"
        printf 'vgpu_active=%s\n' "$(probe_vgpu_active_real)"
        exit 0
        ;;
    *)
        printf 'cloud_preflight: unknown CLOUD_PREFLIGHT_PROBE_ONLY value: %s\n' \
            "${CLOUD_PREFLIGHT_PROBE_ONLY}" >&2
        exit 2
        ;;
esac

# --- GPU query ---------------------------------------------------------
GPU_QUERY_CSV=""
if [[ -n "${MOCK_DIR}" ]]; then
    GPU_QUERY_CSV="$(read_mock_or_empty nvidia_smi.csv)"
else
    GPU_QUERY_CSV="$(nvidia-smi \
        --query-gpu=name,uuid,driver_version,compute_cap,clocks.max.sm,clocks.max.mem,clocks.current.sm,mig.mode.current,compute_mode \
        --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
fi
GPU_QUERY_CSV="${GPU_QUERY_CSV%%$'\n'*}"
IFS=',' read -r GPU_NAME GPU_UUID GPU_DRIVER GPU_CAP GPU_SM_MAX GPU_MEM_MAX GPU_SM_CUR GPU_MIG GPU_COMPUTE_MODE <<<"${GPU_QUERY_CSV}"
GPU_NAME="$(trim "${GPU_NAME:-unknown}")"
GPU_UUID="$(trim "${GPU_UUID:-unknown}")"
GPU_DRIVER="$(trim "${GPU_DRIVER:-unknown}")"
GPU_CAP="$(trim "${GPU_CAP:-unknown}")"
GPU_SM_MAX="$(trim "${GPU_SM_MAX:-unknown}")"
GPU_MEM_MAX="$(trim "${GPU_MEM_MAX:-unknown}")"
GPU_SM_CUR="$(trim "${GPU_SM_CUR:-unknown}")"
GPU_MIG="$(trim "${GPU_MIG:-unknown}")"
GPU_COMPUTE_MODE="$(trim "${GPU_COMPUTE_MODE:-unknown}")"

# --- compute-capability arch gate -------------------------------------
# Per CLAUDE.md "Two-Device Local Development Model" the preflight admits
# both local-development arches: sm_86 (RTX 3090 Ti, CC 8.6, Ampere
# desktop) and sm_120 family (RTX 5060 laptop, CC 12.x, Blackwell consumer
# laptop). Server arches (sm_80 / sm_89 / sm_90a / server sm_120) and other
# arches are rejected here.
case "${GPU_CAP}" in
    8.6) ;;
    12.*) ;;
    *) fatal_or_warn TargetArchMismatch "compute capability '${GPU_CAP}' is not in the sm_86 or sm_120 family" ;;
esac

# When the cap is sm_120 family, additionally reject server /
# workstation Blackwell SKUs whose device name matches
# `*pro 6000*` (case-insensitive). The canonical workstation marker
# is "RTX PRO 6000 Blackwell Workstation"; the consumer laptop SKU
# (e.g. "GeForce RTX 5060 Laptop GPU") does not contain this
# substring and continues to be admitted. This mirrors the
# REJECTED_SM120_NAME_SUBSTRINGS narrowing in
# tools/cloud/preflight_manifest.py so the runtime gate and the
# downstream validator agree on the local-development-consumer-only
# policy.
case "${GPU_CAP}" in
    12.*)
        gpu_name_lower="${GPU_NAME,,}"
        if [[ "${gpu_name_lower}" == *"pro 6000"* ]]; then
            fatal_or_warn TargetArchMismatch "sm_120 device name '${GPU_NAME}' matches the server/workstation Blackwell marker 'pro 6000'; the local-development contract admits only consumer Blackwell laptop SKUs"
        fi
        ;;
esac

# Map the local arch family to an nvcc -arch token for compile probes
# below. The cluster-attribute / CUPTI Activity / NVBit injection probes
# compile a tiny C++ source for the LOCAL arch so the produced cubin
# actually runs on the local device.
case "${GPU_CAP}" in
    8.6)   GPU_NVCC_ARCH="sm_86" ;;
    12.*)  GPU_NVCC_ARCH="sm_120" ;;
    *)     GPU_NVCC_ARCH="sm_120" ;;  # fall-through (unreachable; gated above)
esac

# --- CUDA / nvcc -------------------------------------------------------
if [[ -n "${MOCK_DIR}" ]]; then
    CUDA_VER="$(trim "$(read_mock_or_empty nvcc_version.txt)")"
else
    CUDA_VER="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -1)"
    CUDA_VER="$(trim "${CUDA_VER:-unknown}")"
fi
[[ -z "${CUDA_VER}" ]] && CUDA_VER="unknown"
NVCC_VER="${CUDA_VER}"

# --- Cluster-attribute probe ------------------------------------------
if [[ -n "${MOCK_DIR}" ]]; then
    probe_txt="$(trim "$(read_mock_or_empty cluster_attr_probe.txt)")"
    if [[ -z "${probe_txt}" ]]; then
        CLUSTER_ATTR="mock_input_missing"
    else
        CLUSTER_ATTR="${probe_txt}"
    fi
else
    CLUSTER_ATTR="probe_failed"
    if command -v nvcc >/dev/null 2>&1; then
        probe_tmpdir="$(mktemp -d)"
        trap 'rm -rf "${probe_tmpdir}"' EXIT
        cat > "${probe_tmpdir}/cluster_attr_probe.cu" <<'PROBE_EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    cudaFuncAttribute attr = cudaFuncAttributeNonPortableClusterSizeAllowed;
    (void)attr;
    int dev = 0;
    cudaError_t rc = cudaGetDevice(&dev);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %d\n", (int)rc);
        return 2;
    }
    return 0;
}
PROBE_EOF
        if nvcc -arch=${GPU_NVCC_ARCH} "${probe_tmpdir}/cluster_attr_probe.cu" \
                -o "${probe_tmpdir}/cluster_attr_probe" >"${probe_tmpdir}/nvcc.log" 2>&1; then
            if "${probe_tmpdir}/cluster_attr_probe" >/dev/null 2>&1; then
                CLUSTER_ATTR="available"
            else
                CLUSTER_ATTR="runtime_failed"
            fi
        else
            CLUSTER_ATTR="compile_failed"
        fi
    fi
fi

# --- CUPTI search root + include/lib resolution (shared) --------------
CUPTI_SEARCH_ROOTS=(
    /usr/local/cuda
    /usr/local/cuda-13 /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13.0
    /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.6
    /opt/cuda
)

# Tests / behavioral smokes can prepend additional CUPTI search roots
# (e.g. a staged fake CUPTI installation under /tmp) via the
# CLOUD_PREFLIGHT_CUPTI_SEARCH_ROOTS environment variable. Format:
# semicolon-separated absolute paths. Prepended to the hard-coded
# list so that test fixtures take precedence over real installs.
if [[ -n "${CLOUD_PREFLIGHT_CUPTI_SEARCH_ROOTS:-}" ]]; then
    IFS=';' read -ra _cupti_root_override <<<"${CLOUD_PREFLIGHT_CUPTI_SEARCH_ROOTS}"
    CUPTI_SEARCH_ROOTS=("${_cupti_root_override[@]}" "${CUPTI_SEARCH_ROOTS[@]}")
fi

# Resolved lazily by the legacy `cupti_activity_api_available` probe
# below; consumed (when set) by the new `cupti_activity_api_runtime`
# probe to skip its own filesystem walk. Empty strings indicate
# unresolved or unavailable paths; the runtime probe re-tries with
# its own walk if needed.
CUPTI_INC_DIR=""
CUPTI_LIB_DIR=""

# --- CUPTI Activity API probe (legacy: header/library presence) ------
if [[ -n "${MOCK_DIR}" ]]; then
    cupti_txt="$(trim "$(read_mock_or_empty cupti_activity_probe.txt)")"
    if [[ -z "${cupti_txt}" ]]; then
        CUPTI_ACTIVITY="mock_input_missing"
    else
        CUPTI_ACTIVITY="${cupti_txt}"
    fi
else
    CUPTI_ACTIVITY="probe_failed"
    cupti_header_found=0
    cupti_library_found=0
    for root in "${CUPTI_SEARCH_ROOTS[@]}"; do
        if [[ -z "${CUPTI_INC_DIR}" ]]; then
            for inc_candidate in "${root}/extras/CUPTI/include" "${root}/include"; do
                if [[ -f "${inc_candidate}/cupti.h" ]]; then
                    CUPTI_INC_DIR="${inc_candidate}"
                    cupti_header_found=1
                    break
                fi
            done
        fi
        if [[ -z "${CUPTI_LIB_DIR}" ]]; then
            for lib_candidate in "${root}/extras/CUPTI/lib64" "${root}/lib64" "${root}/targets/x86_64-linux/lib"; do
                if compgen -G "${lib_candidate}/libcupti.so*" >/dev/null; then
                    CUPTI_LIB_DIR="${lib_candidate}"
                    cupti_library_found=1
                    break
                fi
            done
        fi
    done
    if (( cupti_header_found == 0 )); then
        CUPTI_ACTIVITY="header_missing"
    elif (( cupti_library_found == 0 )); then
        CUPTI_ACTIVITY="library_missing"
    else
        CUPTI_ACTIVITY="available"
    fi
fi

# --- CUPTI Activity API runtime smoke probe ---------------------------
# Status precedence (earliest-reachable failure wins; documented in
# the script header):
#   mock_input_missing < compile_failed < link_failed
#   < runtime_failed < insufficient_privilege < cc_unsupported
#   < available
if [[ -n "${MOCK_DIR}" ]]; then
    cupti_runtime_txt="$(trim "$(read_mock_or_empty cupti_runtime_probe.txt)")"
    if [[ -z "${cupti_runtime_txt}" ]]; then
        CUPTI_RUNTIME="mock_input_missing"
    else
        CUPTI_RUNTIME="${cupti_runtime_txt}"
    fi
else
    CUPTI_RUNTIME="runtime_failed"
    if ! command -v nvcc >/dev/null 2>&1; then
        CUPTI_RUNTIME="compile_failed"
    elif [[ -z "${CUPTI_INC_DIR}" ]]; then
        CUPTI_RUNTIME="compile_failed"
    elif [[ -z "${CUPTI_LIB_DIR}" ]]; then
        CUPTI_RUNTIME="link_failed"
    else
        cupti_runtime_tmpdir="$(mktemp -d)"
        # Reset trap to clean up all probe tmpdirs at exit.
        trap 'rm -rf "${probe_tmpdir:-}" "${nvbit_tmpdir:-}" "${cupti_runtime_tmpdir:-}"' EXIT
        cat > "${cupti_runtime_tmpdir}/cupti_runtime_probe.cu" <<'CUPTI_RUNTIME_EOF'
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 8 * 1024;
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
    (void)ctx; (void)streamId; (void)size; (void)validSize;
    free(buffer);
}

int main(void) {
    CUptiResult r1 = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    if (r1 == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {
        fprintf(stderr, "PROBE_INSUFFICIENT_PRIVILEGES\n");
        return 30;
    }
    if (r1 != CUPTI_SUCCESS) {
        fprintf(stderr, "PROBE_REGISTER_CALLBACKS_FAILED:%d\n", (int)r1);
        return 31;
    }
    CUptiResult r2 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    if (r2 == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {
        fprintf(stderr, "PROBE_INSUFFICIENT_PRIVILEGES\n");
        return 30;
    }
    if (r2 != CUPTI_SUCCESS) {
        fprintf(stderr, "PROBE_ENABLE_KIND_FAILED:%d\n", (int)r2);
        return 32;
    }
    cudaError_t cr = cudaSetDevice(0);
    if (cr != cudaSuccess) {
        fprintf(stderr, "PROBE_CUDA_SET_DEVICE_FAILED:%d\n", (int)cr);
        return 33;
    }
    fprintf(stdout, "PROBE_OK\n");
    return 0;
}
CUPTI_RUNTIME_EOF
        # Stage 1: compile-only.
        if ! nvcc -arch=${GPU_NVCC_ARCH} "-I${CUPTI_INC_DIR}" -c \
                "${cupti_runtime_tmpdir}/cupti_runtime_probe.cu" \
                -o "${cupti_runtime_tmpdir}/cupti_runtime_probe.o" \
                >"${cupti_runtime_tmpdir}/compile.log" 2>&1; then
            CUPTI_RUNTIME="compile_failed"
        # Stage 2: link.
        elif ! nvcc "-L${CUPTI_LIB_DIR}" \
                "${cupti_runtime_tmpdir}/cupti_runtime_probe.o" \
                -lcupti -o "${cupti_runtime_tmpdir}/cupti_runtime_probe" \
                >"${cupti_runtime_tmpdir}/link.log" 2>&1; then
            CUPTI_RUNTIME="link_failed"
        else
            # Stage 3: run.
            cupti_run_exit=0
            LD_LIBRARY_PATH="${CUPTI_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
                "${cupti_runtime_tmpdir}/cupti_runtime_probe" \
                >"${cupti_runtime_tmpdir}/run.stdout" \
                2>"${cupti_runtime_tmpdir}/run.stderr" || cupti_run_exit=$?
            if [[ ${cupti_run_exit} -eq 0 ]]; then
                CUPTI_RUNTIME="available"
            elif grep -q 'PROBE_INSUFFICIENT_PRIVILEGES' "${cupti_runtime_tmpdir}/run.stderr"; then
                CUPTI_RUNTIME="insufficient_privilege"
            elif grep -qE '(CONFIDENTIAL_COMPUTING|cc_unsupported)' "${cupti_runtime_tmpdir}/run.stderr"; then
                CUPTI_RUNTIME="cc_unsupported"
            else
                CUPTI_RUNTIME="runtime_failed"
            fi
        fi
    fi
fi

# --- Confidential Computing mode probe -------------------------------
if [[ -n "${MOCK_DIR}" ]]; then
    cc_mode_txt="$(trim "$(read_mock_or_empty cc_mode_probe.txt)")"
    if [[ -z "${cc_mode_txt}" ]]; then
        CC_MODE="unknown"
    else
        CC_MODE="${cc_mode_txt}"
    fi
else
    CC_MODE="unsupported_query"
    if command -v nvidia-smi >/dev/null 2>&1; then
        cc_query_out="$(nvidia-smi conf-compute -f 2>/dev/null || true)"
        if [[ -n "${cc_query_out}" ]]; then
            cc_mode_lower="$(printf '%s' "${cc_query_out}" | tr '[:upper:]' '[:lower:]')"
            case "${cc_mode_lower}" in
                *"cc protected"*on*|*"cc state: on"*|*"cc-on"*|*"on"*on-devtools*) CC_MODE="on" ;;
                *"devtools"*|*"cc-devtools"*) CC_MODE="devtools" ;;
                *"off"*|*"disabled"*) CC_MODE="off" ;;
                *) CC_MODE="off" ;;
            esac
        else
            # Confidential Computing query unsupported on this driver.
            CC_MODE="unsupported_query"
        fi
    fi
fi

# --- NVML capability matrix probe ------------------------------------
if [[ -n "${MOCK_DIR}" ]]; then
    nvml_matrix_txt="$(trim "$(read_mock_or_empty nvml_capability_matrix_probe.txt)")"
    if [[ -z "${nvml_matrix_txt}" ]]; then
        NVML_MATRIX="denied"
    else
        NVML_MATRIX="${nvml_matrix_txt}"
    fi
else
    NVML_MATRIX="denied"
    if command -v python3 >/dev/null 2>&1; then
        nvml_probe_out="$(python3 - <<'NVML_PROBE_EOF' 2>/dev/null || true
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
        NVML_TEMPERATURE_GPU,
    )
except Exception:
    print("denied")
    raise SystemExit(0)
try:
    nvmlInit()
except Exception:
    print("denied")
    raise SystemExit(0)
try:
    h = nvmlDeviceGetHandleByIndex(0)
except Exception:
    nvmlShutdown(); print("denied"); raise SystemExit(0)
power_ok = False
temp_ok = False
try:
    nvmlDeviceGetPowerUsage(h); power_ok = True
except Exception:
    pass
try:
    nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU); temp_ok = True
except Exception:
    pass
nvmlShutdown()
if power_ok and temp_ok:
    print("full")
elif power_ok:
    print("partial_power_only")
elif temp_ok:
    print("partial_temp_only")
else:
    print("denied")
NVML_PROBE_EOF
)"
        nvml_probe_out="$(trim "${nvml_probe_out}")"
        case "${nvml_probe_out}" in
            full|partial_power_only|partial_temp_only|denied) NVML_MATRIX="${nvml_probe_out}" ;;
            *) NVML_MATRIX="denied" ;;
        esac
    fi
fi

# --- MPS active probe ------------------------------------------------
# `capabilities.mps_active` is a record-only deployment-mode signal
# (schema v4). The validator emits the field on the manifest and
# applies a schema-correctness check (present + boolean-typed) but
# does NOT hard-reject on `true`; downstream consumers read it to
# choose CUPTI attribution semantics (per-client kernel attribution
# is the only attribution that survives an active MPS server). The
# real (non-mock) detection logic is `probe_mps_active_real` defined
# above; the MOCK_DIR branch reads `mps_active_probe.txt` directly.
# Schema_version 4 emits this field unconditionally; v3 manifests
# that omit it remain admissible per backward-compat.
if [[ -n "${MOCK_DIR}" ]]; then
    mps_active_txt="$(trim "$(read_mock_or_empty mps_active_probe.txt)")"
    case "${mps_active_txt}" in
        true)  MPS_ACTIVE="true" ;;
        false) MPS_ACTIVE="false" ;;
        *)     MPS_ACTIVE="false" ;;  # default false on a bare container
    esac
else
    MPS_ACTIVE="$(probe_mps_active_real)"
fi

# --- vGPU active probe -----------------------------------------------
# `capabilities.vgpu_active` is a record-only deployment-mode signal
# (schema v4). The validator emits the field on the manifest and
# applies a schema-correctness check but does NOT hard-reject on
# `true`; downstream consumers read it to choose activity-kind
# fallback (`_KERNEL` instead of source-correlated activity kinds
# when HES source-correlation is unavailable under vGPU). The real
# (non-mock) detection logic is `probe_vgpu_active_real` defined
# above; the MOCK_DIR branch reads `vgpu_active_probe.txt` directly.
if [[ -n "${MOCK_DIR}" ]]; then
    vgpu_active_txt="$(trim "$(read_mock_or_empty vgpu_active_probe.txt)")"
    case "${vgpu_active_txt}" in
        true)  VGPU_ACTIVE="true" ;;
        false) VGPU_ACTIVE="false" ;;
        *)     VGPU_ACTIVE="false" ;;  # default false on a bare container
    esac
else
    VGPU_ACTIVE="$(probe_vgpu_active_real)"
fi

# --- NVBit injection smoke probe + tracer hash ------------------------
NVBIT_HASH="unknown"
if [[ -n "${MOCK_DIR}" ]]; then
    injection_verdict="$(trim "$(read_mock_or_empty nvbit_injection_probe.txt)")"
    [[ -z "${injection_verdict}" ]] && injection_verdict="mock_input_missing"
    if [[ "${injection_verdict}" == "ok" ]]; then
        NVBIT_HASH="$(trim "$(read_mock_or_empty nvbit_build_hash.txt)")"
        [[ -z "${NVBIT_HASH}" ]] && fatal_or_warn NvbitUnavailable "mock nvbit_injection_probe.txt=ok but nvbit_build_hash.txt is missing"
    else
        fatal_or_warn NvbitUnavailable "mock injection probe returned '${injection_verdict}' (expected 'ok')"
    fi
else
    if [[ ! -f "${NVBIT_SO_PATH}" ]]; then
        fatal_or_warn NvbitUnavailable "NVBit tracer shared object not found at ${NVBIT_SO_PATH}"
    fi
    if [[ ! -d "${NVBIT_PB_LIB}" ]]; then
        fatal_or_warn NvbitUnavailable "NVBit pinned protobuf runtime not found at ${NVBIT_PB_LIB}"
    fi
    injection_verdict="injection_failed"
    if command -v nvcc >/dev/null 2>&1 && [[ -f "${NVBIT_SO_PATH}" && -d "${NVBIT_PB_LIB}" ]]; then
        nvbit_tmpdir="$(mktemp -d)"
        trap 'rm -rf "${probe_tmpdir:-}" "${nvbit_tmpdir}"' EXIT
        cat > "${nvbit_tmpdir}/nvbit_injection_probe.cu" <<'NVBIT_EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void nop_kernel(void) {
}

int main(void) {
    nop_kernel<<<1, 1>>>();
    cudaError_t rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %d\n", (int)rc);
        return 2;
    }
    return 0;
}
NVBIT_EOF
        if ! command -v "${NVBIT_PYTHON}" >/dev/null 2>&1; then
            injection_verdict="python_missing"
        elif ! "${NVBIT_PYTHON}" "${NVBIT_RUNTIME_CHECK}" \
                --pb-lib "${NVBIT_PB_LIB}" "${NVBIT_SO_PATH}" \
                >"${nvbit_tmpdir}/runtime_check.log" 2>&1; then
            injection_verdict="dependency_check_failed"
        elif nvcc -arch=${GPU_NVCC_ARCH} "${nvbit_tmpdir}/nvbit_injection_probe.cu" \
                -o "${nvbit_tmpdir}/nvbit_injection_probe" >"${nvbit_tmpdir}/nvcc.log" 2>&1; then
            pushd "${nvbit_tmpdir}" >/dev/null
            if LD_LIBRARY_PATH="${NVBIT_PB_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
               CUDA_INJECTION64_PATH="${NVBIT_SO_PATH}" \
               ./nvbit_injection_probe >"${nvbit_tmpdir}/inj.log" 2>&1; then
                if "${NVBIT_PYTHON}" "${NVBIT_TRACE_ADMISSION}" \
                        "${nvbit_tmpdir}/traces" \
                        >"${nvbit_tmpdir}/trace_admission.log" 2>&1; then
                    injection_verdict="ok"
                else
                    injection_verdict="trace_contract_failed"
                fi
            fi
            popd >/dev/null
        else
            injection_verdict="compile_failed"
        fi
    fi
    if [[ "${injection_verdict}" == "ok" ]]; then
        NVBIT_HASH="$(sha256sum "${NVBIT_SO_PATH}" | awk '{print $1}')"
    else
        fatal_or_warn NvbitUnavailable "NVBit injection smoke probe failed (verdict=${injection_verdict})"
    fi
fi
[[ -z "${NVBIT_HASH}" ]] && NVBIT_HASH="unknown"

# --- Optional library versions ----------------------------------------
CUTLASS_VER="$(trim "$(read_mock_or_empty cutlass_version.txt)")"
FLASHINFER_VER="$(trim "$(read_mock_or_empty flashinfer_version.txt)")"

# --- Permissions probe ------------------------------------------------
if [[ -n "${MOCK_DIR}" ]] && [[ -f "${MOCK_DIR}/permissions.json" ]]; then
    PERMS_JSON="$(cat "${MOCK_DIR}/permissions.json")"
else
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -x "$(command -v nvidia-smi)" ]]; then
        P_SMI="true"; else P_SMI="false"; fi
    nvidia_dev_readable="false"
    if compgen -G "/dev/nvidia*" >/dev/null; then
        first_dev="$(compgen -G "/dev/nvidia*" | head -1)"
        [[ -r "${first_dev}" ]] && nvidia_dev_readable="true"
    fi
    PERMS_JSON=$(printf '{"nvidia_smi_executable":%s,"nvidia_dev_readable":%s}' \
        "${P_SMI}" "${nvidia_dev_readable}")
fi

# --- Emit the manifest -------------------------------------------------
TIMESTAMP="$(date -Iseconds)"
HOSTNAME_VAL="$(hostname 2>/dev/null || echo unknown)"

MANIFEST=$(cat <<EOF
{
  "schema_version": 4,
  "timestamp": "$(json_escape "${TIMESTAMP}")",
  "hostname": "$(json_escape "${HOSTNAME_VAL}")",
  "gpu": {
    "name": "$(json_escape "${GPU_NAME}")",
    "uuid": "$(json_escape "${GPU_UUID}")",
    "compute_capability": "$(json_escape "${GPU_CAP}")",
    "mig_mode": "$(json_escape "${GPU_MIG}")",
    "compute_mode": "$(json_escape "${GPU_COMPUTE_MODE}")",
    "clocks_mhz": {
      "sm_max": "$(json_escape "${GPU_SM_MAX}")",
      "sm_current": "$(json_escape "${GPU_SM_CUR}")",
      "mem_max": "$(json_escape "${GPU_MEM_MAX}")"
    }
  },
  "toolchain": {
    "driver_version": "$(json_escape "${GPU_DRIVER}")",
    "cuda_version": "$(json_escape "${CUDA_VER}")",
    "nvcc_version": "$(json_escape "${NVCC_VER}")",
    "nvbit_build_hash": "$(json_escape "${NVBIT_HASH}")",
    "cutlass_version": "$(json_escape "${CUTLASS_VER}")",
    "flashinfer_version": "$(json_escape "${FLASHINFER_VER}")"
  },
  "capabilities": {
    "cluster_attribute_nonportable_cluster_size_allowed": "$(json_escape "${CLUSTER_ATTR}")",
    "cupti_activity_api_available": "$(json_escape "${CUPTI_ACTIVITY}")",
    "cupti_activity_api_runtime": "$(json_escape "${CUPTI_RUNTIME}")",
    "cc_mode": "$(json_escape "${CC_MODE}")",
    "nvml_capability_matrix": "$(json_escape "${NVML_MATRIX}")",
    "mps_active": ${MPS_ACTIVE},
    "vgpu_active": ${VGPU_ACTIVE}
  },
  "permissions": ${PERMS_JSON}
}
EOF
)

if [[ -n "${OUT_PATH}" ]]; then
    printf '%s\n' "${MANIFEST}" > "${OUT_PATH}"
else
    printf '%s\n' "${MANIFEST}"
fi
