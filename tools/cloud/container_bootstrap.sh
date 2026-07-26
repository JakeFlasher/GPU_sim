#!/usr/bin/env bash
# container_bootstrap.sh -- cloud GPU container bootstrap for SimBench-BW.
#
# Goal:
#   Prepare the CUDA base of a fresh
#   `nvidia/cuda:13.2.1-devel-ubuntu24.04` container (run with `--gpus all`
#   via NVIDIA Container Toolkit) for deferred cloud build, NVBit trace/phase
#   collection, and static provenance validation. Repository-local pinned
#   prerequisites remain an explicit handoff below; this script does not
#   silently install an ambient protobuf ABI or compiler.
#   New CUPTI sidecar production and admission are retired (Stage 4, C1):
#   this script installs no collector and defines no CUPTI workflow.
#
# This cloud-container stage has no hardware-counter or clock-lock authority.
# New GB202 NCU capture is deferred to its separately authorized host
# runbook; this bootstrap therefore installs neither NCU nor a counter workflow.
#
# Companion artifacts (same workspace):
#   workloads/simbench_bw/run_suite.py -- the per-kernel orchestrator that
#       this script unblocks for NVBit trace collection. Its
#       `--mode emit-dual-run-manifest` remains available only as historical
#       replay over immutable pre-existing CUPTI/NVBit artifact directories;
#       it is not part of this bootstrap.
#
# Usage:
#   bash tools/cloud/container_bootstrap.sh [--with-venv]
#
# Flags:
#   --with-venv       Also create /opt/simbench-bw-venv with pip deps
#                     (pytest, tomli, tomli-w, jsonschema). Equivalent to
#                     setting the VENV=1 environment variable. Default: off.
#
# Exit codes:
#   0  CUDA-base bootstrap completed; follow the printed pinned-prefix handoff
#      before invoking run_suite.py.
#   2  host-driver too old for the CUDA 13.2.1 image baseline.
#   3  nvidia-smi not available (missing --gpus all or wrong env).
#   4  required toolchain (nvcc / cuobjdump) missing after apt install.
#
# What this script DOES NOT DO (and never should):
#   - Install or upgrade NVIDIA host drivers (the host driver is pre-baked
#     by the cloud provider via NVIDIA Container Toolkit).
#   - Reboot, shutdown, or modify the kernel command line.
#   - Lock GPC/memory clocks (no CAP_SYS_ADMIN in containers).
#   - Detect Secure Boot, manage MOK enrollment, or run DKMS.
#   - Leave NCU counter probing to the separately authorized host capture
#     workflow; this container bootstrap does not install or invoke NCU.
#   - Build NVBit (NVBit 1.8 driver ceiling 575.xx; cloud hosts run 580+/595+).
#
# Cloud-container constraints:
#   - Image: nvidia/cuda:13.2.1-devel-ubuntu24.04 (latest CUDA 13.2 Update 1, April 2026).
#   - Required host driver: >= 595.58.03 (the CUDA 13.2.1 image baseline).
#   - Blackwell GB202 mandates the open kernel module on host (proprietary
#     driver does not work; this is a host-side concern, not container).

set -euo pipefail

readonly REQUIRED_DRIVER="595.58.03"
readonly CUDA_IMAGE_TAG="nvidia/cuda:13.2.1-devel-ubuntu24.04"
readonly OLDER_IMAGE_FALLBACK="nvidia/cuda:13.0.3-devel-ubuntu24.04"
readonly VENV_DIR="/opt/simbench-bw-venv"

readonly APT_PACKAGES=(
    python3-venv
    build-essential
    git
    make
    pkg-config
)

# Pip deps installed only when --with-venv (or VENV=1) is set.
readonly VENV_PIP_DEPS=(
    pytest
    tomli
    tomli-w
    jsonschema
)

WITH_VENV="${VENV:-0}"
for arg in "$@"; do
    case "$arg" in
        --with-venv) WITH_VENV=1 ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//' >&2
            exit 0
            ;;
        *)
            echo "container_bootstrap: unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

# --------------------------------------------------------------------------
# Compatibility gate: host driver must be >= 595.58.03 for the CUDA 13.2.1
# devel image. The container's libcudart links against the host driver
# at runtime; an older host driver produces ABI errors at first cudaInit.
# --------------------------------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<EOF
container_bootstrap: nvidia_smi_unavailable_check_gpu_passthrough
  nvidia-smi is not on PATH inside this container. The CUDA 13.2.1 devel
  image bundles nvidia-smi via NVIDIA Container Toolkit; if it is missing
  the container was launched WITHOUT GPU passthrough.

  Re-run the container with:
    docker run --rm -it --gpus all \\
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\
        -v "\$PWD:/work" -w /work \\
        ${CUDA_IMAGE_TAG} bash

  If you used podman / nerdctl / a managed runtime, the equivalent flags
  are runtime-specific; consult your vendor's documentation.
EOF
    exit 3
fi

OBSERVED_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
if [[ -z "$OBSERVED_DRIVER" ]]; then
    cat >&2 <<EOF
container_bootstrap: nvidia_smi_unavailable_check_gpu_passthrough
  nvidia-smi exited successfully but did not return a driver_version.
  Confirm --gpus all + NVIDIA_DRIVER_CAPABILITIES=compute,utility are set.
EOF
    exit 3
fi

# Compare driver versions as dotted ints (major.minor.patch). sort -V handles it.
LOWEST_OK="$(printf '%s\n%s\n' "$OBSERVED_DRIVER" "$REQUIRED_DRIVER" | sort -V | head -n 1)"
if [[ "$LOWEST_OK" != "$REQUIRED_DRIVER" ]]; then
    cat >&2 <<EOF
container_bootstrap: host_driver_too_old_for_cuda_image
  Observed host driver: ${OBSERVED_DRIVER}
  Required minimum:     ${REQUIRED_DRIVER}  (for ${CUDA_IMAGE_TAG})

  Two recovery options:
    (a) Switch to a newer-host VM whose NVIDIA host driver is at least
        ${REQUIRED_DRIVER} (most cloud providers offer a "latest CUDA"
        instance type).
    (b) Re-run the container with the older CUDA devel image:
            docker run --rm -it --gpus all ${OLDER_IMAGE_FALLBACK} bash
        (this image's runtime is compatible with driver 580.xx and below;
        you will not get CUDA 13.2.1 features.)
EOF
    exit 2
fi

echo "container_bootstrap: host driver ${OBSERVED_DRIVER} >= ${REQUIRED_DRIVER}, proceeding."

# --------------------------------------------------------------------------
# Install build dependencies. apt-get is the script-stable interface; we
# never touch /etc/apt/sources.list.d/ (that's bare-metal-bootstrap territory).
# --------------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"

# --------------------------------------------------------------------------
# Verify the toolchain is functional.
# --------------------------------------------------------------------------

for tool in nvcc cuobjdump; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "container_bootstrap: ${tool}_missing_after_apt_install" >&2
        echo "  ${tool} not on PATH after apt install. The CUDA devel image" >&2
        echo "  should ship it in /usr/local/cuda/bin; check the image tag." >&2
        exit 4
    fi
done

nvcc --version
cuobjdump --version
nvidia-smi -L

# System protobuf packages are intentionally not installed: they are not the
# repository's pinned C++ ABI. The exact seven-row explicit lock is the sole
# provisioning authority. A mounted checkout may provision an absent prefix
# with `bash tools/nvbit_tracer/pinned_build_env.sh --ensure`; ordinary sourcing
# is verify/export-only and never repairs a mismatched prefix.
# This base-image bootstrap deliberately does not claim to install a Conda-
# compatible provisioner or g++-14. The pinned environment always requires a
# verified g++-14; an absent prefix additionally requires micromamba, mamba, or
# conda for the sole authorized --ensure path.

# --------------------------------------------------------------------------
# Optional: create a Python virtualenv for the SimBench-BW pip deps.
# Skipped by default because operators frequently mount their own venvs
# from the host volume. Opt in with --with-venv or VENV=1.
# --------------------------------------------------------------------------

if [[ "$WITH_VENV" == "1" ]]; then
    echo "container_bootstrap: creating venv at ${VENV_DIR}"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install "${VENV_PIP_DEPS[@]}"
    echo "container_bootstrap: venv ready. Activate via: source ${VENV_DIR}/bin/activate"
fi

# --------------------------------------------------------------------------
# Operator handoff: tell the user what to do next.
# --------------------------------------------------------------------------

cat <<EOF

container_bootstrap: CUDA base ok.

Next steps (NVBit trace collection; new CUPTI production is retired):
  # Supply a verified g++-14. If the exact prefix is absent, additionally
  # supply micromamba/mamba/conda; --ensure is its only authorized writer.
  bash tools/nvbit_tracer/pinned_build_env.sh --ensure
  source tools/nvbit_tracer/pinned_build_env.sh
  cd workloads/simbench_bw
  python3 run_suite.py --mode trace --kernel rmsnorm_fwd_bf16 \\
      --shape-point joined_ac8_evidence --phase-inject-ab \\
      --out ../../results/nvbit_capture_staging/simbench_bw

New GB202 NCU capture is deferred to its separately authorized host runbook
and does not run from this container.

Before collecting data, validate the static tree:
  python3 validate.py --all
  python3 verify_coverage.py --strict
EOF
