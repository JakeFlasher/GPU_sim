#!/usr/bin/env bash
# Optional campaign-compatibility host smoke for tools/cloud/preflight.sh.
#
# Checks the current NVBit tracer and protobuf environment, runs
# full real-mode `tools/cloud/preflight.sh`, and records the emitted manifest's
# `capabilities.cupti_activity_api_runtime` value informationally.
# The capability is globally record-only; no particular status is asserted.
# Designed to run as a one-shot host invocation from the repo root:
#
#   bash tools/cloud/host_smoke_local_preflight.sh
#
# This is NOT part of the offline-reproducible `test_preflight_probes.sh`
# smoke suite (which uses bash shims + mock fixtures). This script is a
# host smoke that requires:
#
#   * an NVIDIA driver and a Blackwell-or-later GPU,
#   * CUDA 13.2 at /usr/local/cuda-13.2,
#   * g++-14 for the NVBit build,
#   * compatible protobuf 34.1 under the checkout's .toolchain/pb,
#   * the project `.venv` with matching google.protobuf runtime,
#   * the nine generated `_pb_generated/*_pb2.py` reader files,
#   * `cupti.h` + `libcupti.so` discoverable under one of preflight's
#     CUDA search roots.
#
# Exit codes:
#   0  cupti_activity_api_runtime is any recognized status; the value was
#      recorded and reported.
#   1  cupti_activity_api_runtime is missing or unrecognized; manifest is
#      dumped to stderr.
#   2  preflight failed entirely (NvbitUnavailable, TargetArchMismatch,
#      etc.); preflight log is dumped to stderr.
#   3  Toolchain or generated-reader prerequisite missing.
#   5  NVBit tracer ran but the project-side `traced_execution.cc`
#      enhanced-trace assertion fires for a no-op kernel; the assertion
#      output is dumped. Resolution: fix and rebuild the tracer.
#
# Logs and artifacts are kept under a temp directory printed to stderr
# so the operator can inspect them if needed.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREFLIGHT="${SCRIPT_DIR}/preflight.sh"
NVBIT_DIR="${REPO_ROOT}/tools/nvbit_tracer"
NVBIT_SO="${NVBIT_DIR}/tracer_tool/tracer_tool.so"
TMPDIR="$(mktemp -d)"
echo "host_smoke_local_preflight: working dir=${TMPDIR}" >&2

if [[ ! -f "${NVBIT_SO}" || -L "${NVBIT_SO}" ]]; then
  echo "host_smoke_local_preflight: NVBit tracer is missing or not a regular file: ${NVBIT_SO}" >&2
  echo "host_smoke_local_preflight: build it with make -C tools/nvbit_tracer" >&2
  exit 3
fi
echo "host_smoke_local_preflight: using current tracer ${NVBIT_SO}" >&2

if ! source "${NVBIT_DIR}/pinned_build_env.sh"; then
  echo "host_smoke_local_preflight: compatible NVBit environment is unavailable or mismatched" >&2
  exit 3
fi

for binding in address cuda_stream dim3d gpu_device instruction kernel threadblock trace warp; do
  binding_path="${REPO_ROOT}/workloads/simbench_bw/_pb_generated/${binding}_pb2.py"
  if [[ ! -f "${binding_path}" || -L "${binding_path}" ]]; then
    echo "host_smoke_local_preflight: protobuf reader is missing: ${binding_path}" >&2
    exit 3
  fi
done

OUT_JSON="${TMPDIR}/preflight.json"
echo "host_smoke_local_preflight: running real-mode preflight.sh..." >&2
if ! bash "${PREFLIGHT}" --out "${OUT_JSON}" >"${TMPDIR}/preflight.log" 2>&1; then
  echo "host_smoke_local_preflight: preflight FAILED (exit non-zero). Preflight log tail:" >&2
  tail -40 "${TMPDIR}/preflight.log" >&2
  if grep -qE 'Aborted.*nvbit_injection_probe|add_no_binary_kernel' "${TMPDIR}/preflight.log"; then
    echo "" >&2
    echo "host_smoke_local_preflight: LIKELY ROOT CAUSE = NVBit tracer assertion in traces_enhanced/src/traced_execution.cc::add_no_binary_kernel firing during the no-op nop_kernel atexit hook. The traces ARE produced but the assertion fires in atexit, blocking the CUPTI runtime probe from being exercised." >&2
    echo "host_smoke_local_preflight: Resolution: fix and rebuild the tracer, then rerun." >&2
    exit 5
  fi
  exit 2
fi

if [[ ! -f "${OUT_JSON}" ]]; then
  echo "host_smoke_local_preflight: preflight exited 0 but no manifest written" >&2
  exit 2
fi

RUNTIME_VALUE="$(python3 -c "
import json
with open('${OUT_JSON}') as f:
    print(json.load(f)['capabilities']['cupti_activity_api_runtime'])
")"
echo "host_smoke_local_preflight: cupti_activity_api_runtime=${RUNTIME_VALUE}" >&2

RECOGNIZED="available compile_failed link_failed runtime_failed insufficient_privilege cc_unsupported mock_input_missing"
if [[ " ${RECOGNIZED} " != *" ${RUNTIME_VALUE} "* ]]; then
  echo "host_smoke_local_preflight: FAIL — cupti_activity_api_runtime='${RUNTIME_VALUE}' is not a recognized recorded value" >&2
  echo "Manifest dump:" >&2
  cat "${OUT_JSON}" >&2
  exit 1
fi

echo "host_smoke_local_preflight: PASS — cupti_activity_api_runtime recorded ('${RUNTIME_VALUE}', informational, non-gating per D8)" >&2
echo "host_smoke_local_preflight: manifest preserved at ${OUT_JSON} (working dir ${TMPDIR})" >&2
exit 0
