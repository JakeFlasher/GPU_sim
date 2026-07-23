# problem_012_gqa_paged_decode_h32_kv4_d128_ps1 (sm_86 port)

Batched Grouped Query Attention decode with a paged KV
cache. Captured from Qwen3-30B-A3B. Reference spec at
`references/NVIDIA_Manuals/solbench_problems/problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md`.

## Cross-arch role

Port admitted to fill the predictor-hostile validation
minima for the sm_86 expansion ledger. Authorized by
operator (R30, 2026-05-29) as the local-host hostile-cohort
resolution path after Codex R28 named it as the most
concrete fourth-hostile candidate.

`predictor_design_coverage.py` already declares
`problem_012_gqa_paged_decode_h32_kv4_d128_ps1` as
`predictor_hostile`; the port closes the corpus side
of the cohort.

## Variant

- `cuda_cpp/` — CUDA C++ with online softmax (Flash-
  Attention style); FP32 accumulators; BF16 inputs/
  outputs. One CTA per `(batch, query_head)`; 128
  threads per CTA (one per `head_dim` element);
  block-wide warp-shuffle reduction for the dot
  product.

## Evidence

| Field            | Value                  |
|------------------|------------------------|
| Evidence         | `Profiled-local-sm_86` |
| Evidence (sm_86) | `Profiled-local-sm_86` |

Full `sol-execbench` correctness passes on all 48
workloads against the FlashInfer trace data once
`FLASHINFER_TRACE_DIR` resolves to the repo-root
`data/flashinfer-trace/` tree (48/48 PASSED). The
`__sm86` profiling bundle emitted its five
counter-free analytic surfaces cleanly.

## SOL Score

Phase 2 (sm_100a) only. Local-host evidence is
calibration-only.

## Architecture-coupling notes

- sm_86 native: `cutlass::arch::Sm80` MMA shape NOT
  used (this kernel doesn't use tensor cores; it is a
  pure scalar attention decode kernel suited to the
  decode regime where M=1).
- No NVFP4 / TCGen05 / wgmma paths (M4 stage A
  compliant).
- Phase 2 B200 (sm_100a) divergence:
  - DSMEM and cluster-cooperative QK staging
    (sm_100a-only) are out of scope; the per-CTA
    online-softmax loop stays scalar.
  - TCGen05 fp16/bf16 mma path would accelerate the
    QK dot product on B200 but is NOT exercised here.
  - Larger HBM3e L2 on B200 would absorb the KV-cache
    gather more cheaply; sm_86's 6 MiB L2 is the
    constraint that makes this cell predictor_hostile.

## Files

This port was relocated from `baselines/L1/` to
`baselines/FIB/` (R27) once classified as a trace-driven
FlashInfer-bench cell; the canonical layout is:

```
baselines/FIB/problem_012_gqa_paged_decode_h32_kv4_d128_ps1/
  README.md                       (this file)
  cuda_cpp/
    solution.json                 (target_hardware=[LOCAL]; no blocked cflags)
    main.cpp                      (run() entry point; dest-passing-style)
    kernel.cu                     (online-softmax kernel; 128-thread CTAs)
    kernel.h                      (launcher declaration)
```

Smoke-correctness gate:
`scripts/baselines_harness/tests/test_problem_012_gqa_paged_decode_smoke_v1.py`
(2 tests; compiles the kernel + matches the reference
within `max_atol=0.01` / `matched_ratio>=0.99` on
workload-1 axes + the empty-batch boundary).

## R30+ port chain

| Round | Deliverable | Status |
|---|---|---|
| R30 | File skeleton + correct kernel + smoke correctness against reference (workload-1 axes) | DONE |
| R31 | Full 48-workload sol-execbench correctness (needs operator-side safetensors trace data) | pending |
| R32 | Re-emit the calibration workstream brief admitting `problem_012` as the 4th hostile; flip `plan_compliance_status` to `READY_FOR_TASK17` | pending |
