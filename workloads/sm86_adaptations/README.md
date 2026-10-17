# SM86 Python workload adaptations

`runner.py` is derived from the 5090 FlashInfer/SOL reference runner but is a
local Ampere control. It preserves the input/axis/reference contract while:

* using FP16 in place of native FP8 inputs unless explicitly overridden;
* labeling vectorized paths `sm86_vectorized_aten_control`;
* recording `Ampere_FP16_BF16_control` rather than an SM120 native backend;
* enforcing configurable input-memory and free-VRAM limits;
* retaining an explicit `cudaProfilerStart()`/`cudaProfilerStop()` region; and
* writing the exact runner, reference, definition, axes, dtypes, and estimated
  input bytes into each capture's `workload_source/` directory.

These traces are semantic controls, not FlashInfer-native or Blackwell-native
instruction evidence.

Example:

```bash
PYTHONPATH="$SM86_PYTHON_SITE" "$PYTHON_BIN" \
  workloads/sm86_adaptations/runner.py \
  --problem-dir "$REPO_ROOT/workloads/sm120_blackwell/sol_adaptations/L1_035_flux_ada_layer_norm_zero_sm120" \
  --reference-file "$REPO_ROOT/workloads/sm120_blackwell/sol_adaptations/L1_035_flux_ada_layer_norm_zero_sm120/reference.py" \
  --workload-index 0
```
