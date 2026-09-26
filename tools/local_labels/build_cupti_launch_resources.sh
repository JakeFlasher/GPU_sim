#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-13.2}"
CXX="${CXX:-/usr/bin/g++-14}"
OUT="${1:-$PWD/tools/local_labels/cupti_launch_resources.so}"

"$CXX" -std=c++17 -O2 -fPIC -shared \
  tools/local_labels/cupti_launch_resources.cpp \
  -I"$CUDA_ROOT/include" \
  -I"$CUDA_ROOT/extras/CUPTI/include" \
  -L"$CUDA_ROOT/extras/CUPTI/lib64" \
  -Wl,-rpath,"$CUDA_ROOT/extras/CUPTI/lib64" \
  -lcupti -pthread -o "$OUT"
printf 'built %s\n' "$OUT"
