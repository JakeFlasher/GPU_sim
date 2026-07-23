#!/usr/bin/env bash
# Focused regression smoke for the NVBit tracer's
# `traced_execution::add_no_binary_kernel` idempotency fix.
#
# Reproduces the smallest input that exercises the duplicate-kernel
# registration path: a one-kernel CUDA program that does
# `nop_kernel<<<1, 1>>>()` + `cudaDeviceSynchronize()` and exits.
# Pre-fix, the tracer's enhanced-tracer atexit hook aborted with
# `Assertion 'm_kernels.find(kernel_name) == m_kernels.end()' failed`
# because the same kernel was registered through more than one ELF
# extraction path. Post-fix, idempotent registration short-circuits
# the second call and the program exits 0.
#
# Pass conditions:
#   1. Probe binary exits 0 (no SIGABRT from the assertion).
#   2. Tracer protobuf/Abseil DSOs resolve from the pinned checkout prefix.
#   3. The trace passes the non-vacuous admission gate (root launch,
#      instruction-bearing threadblock, contract walk, positive stats).
#
# Skip conditions (advisory host mode only):
#   * NOOP_SMOKE_CUDA_ROOT is unset and `nvcc` cannot be found.
#   * The compiled probe cannot reach a CUDA runtime (no GPU device,
#     no driver, etc.).
#
# Usage:
#   bash tools/nvbit_tracer/smoketest/no_op_kernel_idempotency.sh
#   NVBIT_SMOKE_ARCH=sm_120 \
#     bash tools/nvbit_tracer/smoketest/no_op_kernel_idempotency.sh
#
# Exit codes:
#   0  test passed (or skipped — written to stderr).
#   1  test failed (present-tool validation/compile failure, assertion fired,
#      or output missing).
#   2  authoritative prerequisite missing or invalid invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVBIT_DIR="${SCRIPT_DIR}/.."
NVBIT_SO="${NVBIT_DIR}/tracer_tool/tracer_tool.so"
NVBIT_SO_SHA256="de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d"
PB_LIB="$(cd "${NVBIT_DIR}/../.." && pwd)/.toolchain/pb/lib"
REPO_ROOT="$(cd "${NVBIT_DIR}/../.." && pwd)"
PYTHON="${NVBIT_SMOKE_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -f "${NVBIT_SO}" || -L "${NVBIT_SO}" ]]; then
    echo "no_op_kernel_idempotency: FAIL (authoritative preserved tracer missing or not regular at ${NVBIT_SO}; recover its byte-preserved pinned content; canonical rebuild is forbidden while H3 holds)" >&2
    exit 2
fi
actual_nvbit_sha256="$(sha256sum "${NVBIT_SO}" | awk '{print $1}')"
if [[ "${actual_nvbit_sha256}" != "${NVBIT_SO_SHA256}" ]]; then
    echo "no_op_kernel_idempotency: FAIL (preserved tracer SHA-256 mismatch: expected ${NVBIT_SO_SHA256}, got ${actual_nvbit_sha256})" >&2
    exit 1
fi

if [[ ${NOOP_SMOKE_CUDA_ROOT+x} ]]; then
    if [[ -z "${NOOP_SMOKE_CUDA_ROOT}" ]]; then
        echo "no_op_kernel_idempotency: FAIL (NOOP_SMOKE_CUDA_ROOT is set but empty; authoritative nvcc path is missing)" >&2
        exit 2
    fi
    NVCC="${NOOP_SMOKE_CUDA_ROOT}/bin/nvcc"
    if [[ ! -x "${NVCC}" ]]; then
        echo "no_op_kernel_idempotency: FAIL (authoritative nvcc missing at ${NVCC})" >&2
        exit 2
    fi
else
    NVCC="$(command -v nvcc 2>/dev/null || true)"
    if [[ -z "${NVCC}" ]]; then
        for root in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.2 /opt/cuda; do
            if [[ -x "${root}/bin/nvcc" ]]; then
                NVCC="${root}/bin/nvcc"
                break
            fi
        done
    fi
    if [[ -z "${NVCC}" ]]; then
        echo "no_op_kernel_idempotency: SKIP (advisory nvcc prerequisite unavailable; set NOOP_SMOKE_CUDA_ROOT for an authoritative run)" >&2
        exit 0
    fi
fi
if [[ ! -d "${PB_LIB}" ]]; then
    echo "no_op_kernel_idempotency: FAIL (authoritative pinned protobuf runtime missing at ${PB_LIB}; provision only through pinned_build_env.sh --ensure)" >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "no_op_kernel_idempotency: FAIL (authoritative project Python missing at ${PYTHON})" >&2
    exit 2
fi
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT
if ! "${PYTHON}" "${NVBIT_DIR}/runtime_env.py" \
        --pb-lib "${PB_LIB}" "${NVBIT_SO}" >"${TMPDIR}/nvbit_runtime_check.log" 2>&1; then
    echo "no_op_kernel_idempotency: FAIL (tracer runtime dependency check failed)" >&2
    exit 1
fi

SMOKE_ARCH="${NVBIT_SMOKE_ARCH:-}"
if [[ -z "${SMOKE_ARCH}" ]]; then
    COMPUTE_CAP=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
            2>/dev/null | head -n 1 | tr -d '[:space:]')"
    fi
    if [[ "${COMPUTE_CAP}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        SMOKE_ARCH="sm_${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    else
        echo "no_op_kernel_idempotency: SKIP (could not detect GPU compute capability; set NVBIT_SMOKE_ARCH=sm_<major><minor> to run explicitly)" >&2
        exit 0
    fi
fi
if [[ ! "${SMOKE_ARCH}" =~ ^sm_[0-9]+a?$ ]]; then
    echo "no_op_kernel_idempotency: invalid NVBIT_SMOKE_ARCH=${SMOKE_ARCH@Q}" >&2
    exit 2
fi

cd "${TMPDIR}"

cat > no_op_kernel.cu <<'CUDA_EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void nop_kernel(void) {}

int main(void) {
    nop_kernel<<<1, 1>>>();
    cudaError_t rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %d\n", (int)rc);
        return 2;
    }
    return 0;
}
CUDA_EOF

if ! "${NVCC}" -arch="${SMOKE_ARCH}" no_op_kernel.cu -o no_op_kernel >"${TMPDIR}/build.log" 2>&1; then
    echo "no_op_kernel_idempotency: FAIL (present nvcc compile for ${SMOKE_ARCH} failed; tail of build.log:)" >&2
    tail -10 "${TMPDIR}/build.log" >&2
    exit 1
fi

set +e
LD_LIBRARY_PATH="${PB_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
CUDA_INJECTION64_PATH="${NVBIT_SO}" \
    ./no_op_kernel >"${TMPDIR}/run.log" 2>&1
PROBE_EXIT=$?
set -e

if [[ ${PROBE_EXIT} -ne 0 ]]; then
    # No-driver / no-device skip path: the C stub returns 2 when
    # cudaDeviceSynchronize fails. CUDA_ERROR_NO_DEVICE (100) and
    # CUDA_ERROR_NOT_INITIALIZED (101) are the canonical "no GPU
    # available" classes; treating those as a failure would
    # contradict the docstring's host-independence promise.
    # Other cudaDeviceSynchronize errors (e.g. invalid value,
    # launch failure) and the NVBit assertion still hard-fail.
    if [[ ${PROBE_EXIT} -eq 2 ]] && \
       grep -qE 'cudaDeviceSynchronize failed: (100|101)' "${TMPDIR}/run.log"; then
        echo "no_op_kernel_idempotency: SKIP (probe could not reach a CUDA device — cudaDeviceSynchronize returned the no-device class. The idempotency contract cannot be exercised without a working driver/device on this host)" >&2
        echo "Probe log tail (skipped):" >&2
        tail -10 "${TMPDIR}/run.log" >&2
        exit 0
    fi
    echo "no_op_kernel_idempotency: FAIL — probe exited ${PROBE_EXIT} (expected 0)" >&2
    if grep -q 'add_no_binary_kernel' "${TMPDIR}/run.log"; then
        echo "no_op_kernel_idempotency: ROOT CAUSE = NVBit tracer assertion in traces_enhanced/src/traced_execution.cc::add_no_binary_kernel — the preserved tracer failed its runtime contract. Do not rebuild in the canonical worktree while H3 holds." >&2
    fi
    echo "Probe log tail:" >&2
    tail -20 "${TMPDIR}/run.log" >&2
    exit 1
fi

if ! "${PYTHON}" "${NVBIT_DIR}/trace_admission.py" "${TMPDIR}/traces" \
        >"${TMPDIR}/trace_admission.log" 2>&1; then
    echo "no_op_kernel_idempotency: FAIL — trace output is vacuous or contract-invalid" >&2
    tail -20 "${TMPDIR}/trace_admission.log" >&2
    exit 1
fi

if [[ ! -d "${TMPDIR}/traces/threadblocks" ]]; then
    echo "no_op_kernel_idempotency: FAIL — traces/threadblocks/ directory missing after probe ran (NVBit injection did not run as expected)" >&2
    exit 1
fi

if [[ ! -s "${TMPDIR}/traces/dynamic_trace.pb" ]]; then
    echo "no_op_kernel_idempotency: FAIL — traces/dynamic_trace.pb missing or empty after probe ran" >&2
    exit 1
fi

PB_BYTES="$(stat -c%s "${TMPDIR}/traces/dynamic_trace.pb")"
echo "no_op_kernel_idempotency: PASS — probe exited 0; traces/threadblocks/ + traces/dynamic_trace.pb (${PB_BYTES} bytes) both produced. NVBit add_no_binary_kernel is idempotent for the no-op kernel path." >&2
exit 0
