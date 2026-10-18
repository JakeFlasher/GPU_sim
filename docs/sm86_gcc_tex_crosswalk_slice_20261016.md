# SM86 GCC/TEX source-client crosswalk — H73 — 2026-10-16

## Same-binary probe evidence

The existing CUDA 13.2 `gcc_demand_h65` binary and NCU exports provide a
same-binary static/source-client observation on the locked RTX 3090 Ti. The
arithmetic control reports GCC requests 1,764, GCC L2 sectors 952, GCC misses
125, and DRAM reads 3,100; the constant-load arm reports GCC requests 562,919,
GCC L2 sectors 1,960, GCC misses 269, and DRAM reads 276. These are independent
NCU observations, not replay inputs.

Two fresh NVBit attempts were made against the same probe, once with the
existing CUDA 13.4 binary and once rebuilt with CUDA 13.2/g++-14 to match the
tracer. Both tracer launches segfaulted during context/tool initialization
before a dynamic protobuf was produced. Logs are retained under
`/tmp/h73_gcc_capture_v1/` and `/tmp/h73_gcc_capture_v2/`; this is a tracer
compatibility blocker, not evidence that GCC traffic is absent.

## Observer-only crosswalk diagnostic

The post-replay reporter now compares the existing trace-derived GCC
instruction/constant **sector** candidates with the independent NCU GCC-sector
endpoint as warnings only. It never changes `DramStats.read_sectors`, L2 state,
service timing, or metric selection. On the full H72 artifact, the diagnostic
found 286 gross instruction-candidate mismatches and 264 gross
constant-candidate mismatches; request-domain GCC counters remain a separate
dimension. This falsifies the current candidates as direct GCC sector
predictors and prevents an unsafe global multiplier.

## Replay and decision

The reused full-first H72 replay remains 356 roots (286 `ok`, 62 controls, 8
replay errors), 12,392 full launches, 12 workers, compressed at
`../sm86_replay_results/full_corpus_h72_ldgsts_scope_20261016_v1/`. The H73
post-replay report is:

`../sm86_replay_results/metric_report_h73_gcc_crosswalk_20261016_v2/`.

H73 is a negative crosswalk for timing promotion: GCC/TEX source identity is
not available in the dynamic trace and the existing candidates are not
endpoint-equivalent. Keep H66 observer rows and aggregate carded DRAM unchanged.
Future progress requires a repaired NVBit/tracer build that emits dynamic
source-client identity for the same binary, followed by held-out completion
ownership evidence before any multi-client DRAM model is enabled.
