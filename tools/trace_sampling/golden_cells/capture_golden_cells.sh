#!/usr/bin/env bash
# Regenerate the real golden traces into guarded active staging.  Historical
# campaign evidence under data/local_labels/campaign is read-only.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SO="$REPO/tools/nvbit_tracer/tracer_tool/tracer_tool.so"
NVCC=/opt/cuda/bin/nvcc
PYTHON="${NVBIT_CAPTURE_PYTHON:-$REPO/.venv/bin/python}"
OUTPUT_ROOT="${NVBIT_CAPTURE_OUTPUT_ROOT:-$REPO/results/nvbit_capture_staging/golden_cells}"
PLAN_ONLY=0
if [[ "${1:-}" == "--plan-only" ]]; then
  PLAN_ONLY=1
  shift
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "capture_golden_cells: Python environment missing: $PYTHON" >&2
  exit 2
fi

clear_path() {
  "$PYTHON" - "$REPO" "$1" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
candidate = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from tools.local_labels.protected_paths import assert_write_allowed

resolved = assert_write_allowed(
    candidate, repo_root=repo, operation="capture golden-cell output",
)
staging = (repo / "results" / "nvbit_capture_staging").resolve(strict=False)
try:
    resolved.relative_to(staging)
except ValueError as exc:
    raise SystemExit(
        f"capture_golden_cells: refusing output outside {staging}: {resolved}"
    ) from exc
print(resolved)
PY
}

if (( $# )); then
  CELLS=("$@")
else
  CELLS=(gs_stream_copy gs_stream_reuse gs_var_stride ac4_global_atomic_histogram)
fi

# Phase 1: derive and clear the entire batch before runtime probes or mutation.
GOLDEN="$(clear_path "$OUTPUT_ROOT")"
declare -A CU BIN OUT STAGED NCU_CSV NCU_STDERR NCU_FINAL NCU_STDERR_FINAL NEEDS_NCU
for slug in "${CELLS[@]}"; do
  case "$slug" in
    gs_stream_copy|gs_stream_reuse|gs_var_stride)
      cu="$(dirname "$0")/${slug}.cu"; needs_ncu=1 ;;
    ac4_global_atomic_histogram)
      cu="$REPO/tools/local_labels/canary/ac4_global_atomic_histogram.cu"; needs_ncu=0 ;;
    *) echo "unknown golden cell: $slug" >&2; exit 2 ;;
  esac
  cell="$GOLDEN/$slug"
  bin="$(clear_path "$cell/${slug}.bin")"
  out="$(clear_path "$cell/trace")"
  staged_candidate="$(clear_path "$cell/trace.staged.$$")"
  ncu_csv_candidate="$(clear_path "$cell/ncu_l2.csv.staged.$$")"
  ncu_stderr_candidate="$(clear_path "$cell/ncu_l2.stderr.staged.$$")"
  ncu_final="$(clear_path "$cell/ncu_l2.csv")"
  ncu_stderr_final="$(clear_path "$cell/ncu_l2.stderr")"
  CU["$slug"]="$cu"; BIN["$slug"]="$bin"; OUT["$slug"]="$out"
  STAGED["$slug"]="$staged_candidate"; NCU_CSV["$slug"]="$ncu_csv_candidate"
  NCU_STDERR["$slug"]="$ncu_stderr_candidate"; NCU_FINAL["$slug"]="$ncu_final"
  NCU_STDERR_FINAL["$slug"]="$ncu_stderr_final"; NEEDS_NCU["$slug"]="$needs_ncu"
done

if (( PLAN_ONLY )); then
  printf 'capture_golden_cells: plan ok (%d cells) root=%s\n' "${#CELLS[@]}" "$GOLDEN"
  exit 0
fi

# Trap-visible deletion targets remain empty until the complete plan passed.
staged=""; ncu_csv=""; ncu_stderr=""
cleanup() {
  local cleared
  if [[ -n "$staged" ]]; then
    if cleared="$(clear_path "$staged")"; then rm -rf "$cleared" || true; fi
  fi
  if [[ -n "$ncu_csv" ]]; then
    if cleared="$(clear_path "$ncu_csv")"; then rm -f "$cleared" || true; fi
  fi
  if [[ -n "$ncu_stderr" ]]; then
    if cleared="$(clear_path "$ncu_stderr")"; then rm -f "$cleared" || true; fi
  fi
}
trap cleanup EXIT

"$PYTHON" "$REPO/tools/nvbit_tracer/runtime_env.py" \
  --pb-lib "$REPO/.toolchain/pb/lib" "$SO" >/dev/null

for slug in "${CELLS[@]}"; do
  cu="${CU[$slug]}"; bin="${BIN[$slug]}"; out="${OUT[$slug]}"
  staged="${STAGED[$slug]}"; needs_ncu="${NEEDS_NCU[$slug]}"
  ncu_csv="${NCU_CSV[$slug]}"; ncu_stderr="${NCU_STDERR[$slug]}"
  # Re-clear the exact compiler destination immediately before nvcc.  The
  # phase-1 batch plan is still mandatory, while this second check closes a
  # symlink-substitution interval at the final output component.
  bin="$(clear_path "$bin")"
  bin_parent="$(clear_path "$(dirname "$bin")")"
  mkdir -p "$bin_parent"
  "$NVCC" -ccbin=g++-14 -O3 -std=c++17 \
    -gencode arch=compute_86,code=sm_86 "$cu" -o "$bin"
  staged="$(clear_path "$staged")"
  rm -rf "$staged"; mkdir -p "$staged"
  LD_LIBRARY_PATH="$REPO/.toolchain/pb/lib" CUDA_INJECTION64_PATH="$SO" \
    USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$staged" "$bin"
  "$PYTHON" "$REPO/tools/nvbit_tracer/trace_admission.py" "$staged"
  if (( needs_ncu )); then
    ncu_csv="$(clear_path "$ncu_csv")"
    ncu_stderr="$(clear_path "$ncu_stderr")"
    ncu --metrics lts__t_sector_hit_rate.pct --clock-control none \
      --cache-control all --replay-mode kernel --launch-count 1 --csv "$bin" \
      > "$ncu_csv" 2> "$ncu_stderr"
  fi
  out="$(clear_path "$out")"; staged="$(clear_path "$staged")"
  rm -rf "$out"; mv "$staged" "$out"; staged=""
  echo "$slug: $(find "$out/threadblocks" -name '*.pb' | wc -l) blocks"
  if (( needs_ncu )); then
    ncu_final="$(clear_path "${NCU_FINAL[$slug]}")"
    ncu_csv="$(clear_path "$ncu_csv")"
    mv "$ncu_csv" "$ncu_final"; ncu_csv=""
    ncu_stderr_final="$(clear_path "${NCU_STDERR_FINAL[$slug]}")"
    ncu_stderr="$(clear_path "$ncu_stderr")"
    mv "$ncu_stderr" "$ncu_stderr_final"; ncu_stderr=""
    tail -1 "$ncu_final"
  fi
done
