#!/usr/bin/env bash
# Local recorded-capability host smoke for tools/cloud/preflight.sh.
#
# Verifies the preserved NVBit tracer and pinned protobuf environment, runs
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
#   * pinned protobuf 34.1 under the checkout's .toolchain/pb,
#   * the project `.venv` with matching google.protobuf runtime,
#   * the nine tracked `_pb_generated/*_pb2.py` authority files,
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
#   3  Toolchain or tracked-authority prerequisite missing.
#   5  NVBit tracer ran but the project-side `traced_execution.cc`
#      enhanced-trace assertion fires for a no-op kernel; the assertion
#      output is dumped. Resolution: restore the preserved tracer bytes from
#      their pinned artifact, never rebuild in the canonical worktree.
#
# Logs and artifacts are kept under a temp directory printed to stderr
# so the operator can inspect them if needed.
#
# Reviewer note: this wrapper requires GPU access. Reviewers without a
# local NVIDIA driver / Blackwell-or-later GPU can audit acceptance
# instead by reading the committed evidence under
# `tools/cloud/host_smoke_evidence/<host>/`. Each host directory there
# preserves the wrapper's full transcript, the preflight manifest the
# wrapper produced, and structured provenance (driver version, NVBit
# tracer SHA-256, GPU identity, git commit). See
# `tools/cloud/host_smoke_evidence/README.md` for the layout and the
# step-by-step verification checklist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREFLIGHT="${SCRIPT_DIR}/preflight.sh"
NVBIT_DIR="${REPO_ROOT}/tools/nvbit_tracer"
NVBIT_SO="${NVBIT_DIR}/tracer_tool/tracer_tool.so"
EXPECTED_NVBIT_SHA256="de47b3af6a18ae69287f61ff149dbdb03b4baf6ca8e33f03d2a2221a0482ee5d"
TMPDIR="$(mktemp -d)"
echo "host_smoke_local_preflight: working dir=${TMPDIR}" >&2

if [[ ! -f "${NVBIT_SO}" || -L "${NVBIT_SO}" ]]; then
  echo "host_smoke_local_preflight: preserved NVBit tracer is missing or not a regular file: ${NVBIT_SO}" >&2
  echo "host_smoke_local_preflight: restore the byte-exact tracked/pinned artifact; canonical rebuild is forbidden while H3 holds" >&2
  exit 3
fi
actual_nvbit_sha256="$(sha256sum "${NVBIT_SO}" | awk '{print $1}')"
if [[ "${actual_nvbit_sha256}" != "${EXPECTED_NVBIT_SHA256}" ]]; then
  echo "host_smoke_local_preflight: preserved NVBit tracer SHA-256 mismatch" >&2
  echo "host_smoke_local_preflight: expected=${EXPECTED_NVBIT_SHA256} actual=${actual_nvbit_sha256}" >&2
  echo "host_smoke_local_preflight: restore the byte-exact pinned artifact; canonical rebuild is forbidden while H3 holds" >&2
  exit 1
fi
echo "host_smoke_local_preflight: preserved tracer verified (sha256=${actual_nvbit_sha256})" >&2

if ! source "${NVBIT_DIR}/pinned_build_env.sh"; then
  echo "host_smoke_local_preflight: pinned NVBit environment is unavailable or mismatched" >&2
  exit 3
fi

for binding in address cuda_stream dim3d gpu_device instruction kernel threadblock trace warp; do
  binding_path="${REPO_ROOT}/workloads/simbench_bw/_pb_generated/${binding}_pb2.py"
  if [[ ! -f "${binding_path}" || -L "${binding_path}" ]]; then
    echo "host_smoke_local_preflight: tracked protobuf authority is missing: ${binding_path}" >&2
    echo "host_smoke_local_preflight: restore the byte-exact tracked file; ambient generation is forbidden" >&2
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
    echo "host_smoke_local_preflight: Resolution: restore the byte-exact preserved tracer and tracked bindings, then rerun; do not rebuild in this worktree." >&2
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
