#!/usr/bin/env bash
set -euo pipefail

# The campaign admits operator intent before constructing its compiler
# environment.  Remove the root profile's ambient override rather than
# replacing it, so the Python preflight and every chunk gate observe absence.
unset CUDA_HOME
unset CUDA_PATH
unset LD_LIBRARY_PATH
unset NVCC_CCBIN
unset NVCC_PREPEND_FLAGS
unset NVCC_APPEND_FLAGS
unset CC
unset CXX
unset CUDAHOSTCXX
unset CUTLASS_DIR
unset PYTHONPATH
unset PYTHONHOME
unset FLASHINFER_TRACE_DIR

# The producer builds its exact compiler/runtime paths from the sealed
# supplemental environment authority.  Give that construction a minimal,
# deterministic lookup base and prevent user-site imports from entering before
# the authority has been checked.
export PATH="/usr/local/cuda-13.2/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1

if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 64
fi

exec "$@"
