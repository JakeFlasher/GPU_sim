# Tracer Runtime Signal Demo

Persisted captures of the default `LD_PRELOAD` tracer behavior on the
local 3090 Ti host across two representative slugs. The captures
demonstrate the runtime signal contract that the typed
`enhanced_tracer_cubin_parse_timeout` diagnostic fires exactly once
on the CUTLASS-vendored sm_86 cubin (whose generated `.sass`/`.rfu`
parser inputs exceed the 64KB threshold), and that no typed
diagnostic fires on the 14 inline-PTX rows.

## Files

- `cutlass_sm86_diagnostic_fires.log` — captured stdout from the
  default tracer command applied to
  `conv2d_implicit_gemm_bf16_sm80` (CUTLASS Conv2dFprop sm_80
  vendoring). The diagnostic line
  `enhanced_tracer_cubin_parse_timeout: conv2d_implicit_gemm_bf16_sm80.sm_86.cubin (sass=252KB rfu=3736KB threshold=64KB; producer=exited); skipping auxiliary extra_info for this cubin (primary trace artifacts unaffected).`
  fires exactly once. The producer exits 0 cleanly under hard
  timeout.

- `inline_ptx_diagnostic_quiet.log` — captured stdout from the
  default tracer command applied to `silu_fwd_bf16_sm80`
  (representative inline-PTX row). No typed diagnostic line is
  emitted. The producer exits 0 cleanly under hard timeout.

The same default-tracer command (`LD_PRELOAD=tracer_tool.so
./<slug>_run --correctness_shape --reps 1 --output_dir <out>`)
applies uniformly across all 15 active rows; the two captures are
the canonical CUTLASS-vendoring case and the canonical inline-PTX
case.

## Reproduce

```bash
mkdir -p /tmp/runtime_signal_cutlass /tmp/runtime_signal_inline_ptx

LD_PRELOAD=/abs/path/to/tools/nvbit_tracer/tracer_tool/tracer_tool.so \
  ./workloads/portable_lcd/conv2d_implicit_gemm_bf16_sm80/conv2d_implicit_gemm_bf16_sm80_run \
  --correctness_shape --reps 1 \
  --output_dir /tmp/runtime_signal_cutlass \
  > /tmp/runtime_signal_cutlass/stdout.log 2>&1

LD_PRELOAD=/abs/path/to/tools/nvbit_tracer/tracer_tool/tracer_tool.so \
  ./workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80_run \
  --correctness_shape --reps 1 \
  --output_dir /tmp/runtime_signal_inline_ptx \
  > /tmp/runtime_signal_inline_ptx/stdout.log 2>&1

grep -cE '^enhanced_tracer_cubin_parse_timeout:' \
  /tmp/runtime_signal_cutlass/stdout.log
# Expected: 1

grep -cE '^enhanced_tracer_cubin_parse_timeout:' \
  /tmp/runtime_signal_inline_ptx/stdout.log
# Expected: 0
```

## Provenance

Captured live on the local 3090 Ti host (Ampere consumer, sm_86,
24 GB GDDR6X) using the host-built tracer at
`tools/nvbit_tracer/tracer_tool/tracer_tool.so`. The captures are
runtime evidence for the typed-diagnostic-fires-once invariant.
The stored trace artifacts under
`<slug>/dual_source_evidence/nvbit_traces/repeat_0/traces/` are
verified independently by `tools/build/verify_lcd_nvbit_trace.sh`.
