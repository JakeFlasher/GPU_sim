#!/usr/bin/env bash
# SM86 live smoke: same-name modules, context-less libraries, handle reuse,
# and anonymous cubins. Output is retained so failures can be inspected.
# Usage (repository root):
#   source tools/nvbit_tracer/pinned_build_env.sh
#   bash tools/nvbit_tracer/smoketest/module_origin.sh [NEW_OUTPUT_DIRECTORY]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CUDA_ROOT="${CUDA_INSTALL_PATH:-/usr/local/cuda-13.2}"
TRACER="$REPO/tools/nvbit_tracer/tracer_tool/tracer_tool.so"
if [[ $# -gt 1 ]]; then echo 'expected at most one output directory' >&2; exit 2; fi
if [[ $# == 1 ]]; then
  mkdir "$1"
  OUT="$(cd "$1" && pwd)"
else
  OUT="$(mktemp -d "$REPO/results/captures/sm86_module_origin_20260926_XXXXXX")"
fi
echo "module_origin output: $OUT"
export LD_LIBRARY_PATH="$REPO/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi --query-gpu=index,name,uuid,compute_cap,driver_version \
  --format=csv >"$OUT/hardware.csv"
sha256sum "$TRACER" "$HERE/module_origin_kernel.cu" "$HERE/module_origin_probe.cpp" \
  >"$OUT/inputs.sha256"
for spec in 'a 11' 'b 29'; do
  read -r name value <<<"$spec"
  "$CUDA_ROOT/bin/nvcc" -ccbin "${NVCC_CCBIN:-/usr/bin/g++-14}" \
    -arch=sm_86 --cubin -DORIGIN_VALUE="$value" \
    "$HERE/module_origin_kernel.cu" -o "$OUT/$name.cubin" >"$OUT/$name.build.log" 2>&1
done
"${NVCC_CCBIN:-/usr/bin/g++-14}" -std=c++17 -I"$CUDA_ROOT/include" \
  "$HERE/module_origin_probe.cpp" -lcuda -o "$OUT/probe"
for mode in module library library-function library-precontext; do
  mkdir "$OUT/$mode"
  timeout 60s env CUDA_VISIBLE_DEVICES=0 CUDA_INJECTION64_PATH="$TRACER" \
    USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$OUT/$mode/trace" \
    SKIP_ENHANCED_TRACER=1 ACTIVE_FROM_START=1 EXCLUDE_PRED_OFF=0 \
    DYNAMIC_KERNEL_LIMIT_START=0 DYNAMIC_KERNEL_LIMIT_END=0 \
    "$OUT/probe" "$mode" "$OUT/a.cubin" "$OUT/b.cubin" \
    >"$OUT/$mode/run.log" 2>&1 || { tail -35 "$OUT/$mode/run.log"; exit 1; }
done
python3 - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
digests = {n: hashlib.sha256((root / f'{n}.cubin').read_bytes()).hexdigest()
           for n in ('a', 'b')}
for mode in ('module', 'library', 'library-function', 'library-precontext'):
    trace = root / mode / 'trace'
    data = json.loads((trace / 'extra_info/traced_kernel_modules.json').read_text())
    expected = ['a', 'b', 'a', 'b', None] if mode == 'module' else ['a', 'b', 'a', 'a']
    if mode == 'library-precontext':
        expected = [None] * 4
    assert len(data['launches']) == len(expected), (mode, data['launches'])
    assert (trace / 'dynamic_trace.pb').stat().st_size > 0
    assert any(p.stat().st_size for p in (trace / 'threadblocks').rglob('*.pb'))
    for row, owner in zip(data['launches'], expected):
        module = row['module']
        assert row['kernel_name'] == 'origin_collision', row
        if owner is None:
            assert module['status'] == 'no_stable_static_module', row
            reason = 'origin_unresolved' if mode == 'library-precontext' else 'jit_or_anonymous_mapping'
            assert module['reason'] == reason, row
            assert not module['sha256'], row
        else:
            assert module['status'] == 'stable_static_module', (mode, row)
            assert module['sha256'] == digests[owner], (mode, row, owner)
            assert module['path_at_capture'] == str(root / f'{owner}.cubin'), row
    print(f'{mode}: {len(expected)} exact-owner/typed-unavailable launches PASS')
PY
