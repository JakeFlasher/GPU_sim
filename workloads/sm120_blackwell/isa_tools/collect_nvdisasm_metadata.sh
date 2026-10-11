#!/usr/bin/env bash
# CPU-only CUDA 13.2 SM120 metadata extraction using the retained DocumentSASS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${DOCUMENTSASS_ROOT:-/root/DocumentSASS}"
OUT="${1:?provide a fresh output directory}"
CUDA=/usr/local/cuda-13.2
mkdir -p "$OUT"
[[ ! -e "$OUT/intercept.bin" ]] || {
  echo "Raw capture already exists; choose a fresh output directory." >&2
  exit 2
}
/usr/bin/gcc-14 -fPIC -shared -fno-builtin -o "$OUT/intercept.so" "$SOURCE/intercept.c" -ldl
"$CUDA/bin/nvcc" -ccbin /usr/bin/g++-14 -arch=sm_120 -cubin \
  "$SOURCE/example.cu" -o "$OUT/example.cubin"
OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LD_PRELOAD="$(realpath "$OUT/intercept.so")" \
  "$CUDA/bin/nvdisasm" "$OUT/example.cubin" > "$OUT/intercept.bin" 2> "$OUT/nvdisasm.stderr"
python3 "$HERE/extract_nvdisasm_metadata.py" "$OUT/intercept.bin" "$OUT/scheduling_metadata.txt"
cp "$SOURCE/intercept.c" "$SOURCE/LICENSE" "$OUT/"
"$CUDA/bin/nvdisasm" --version > "$OUT/nvdisasm_version.txt"
