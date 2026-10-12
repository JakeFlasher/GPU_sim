# Continuation memory

**Updated:** 2026-10-12

- Branch: `GPUSim_main` in `/root/GPU_simulator_rebase`.
- Active target: GeForce RTX 5090, `sm_120`.
- Active plan: `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
- Full corpus: `/root/sm120_rtx_5090_capture`.
- Replay outputs: `/dev/shm/sm120_rtx_5090_work`.
- Cleanup commit: `14b40f26`.
- Latest cleanup/replay commit: `0330bedc`.
- Workspace test-surface cleanup is complete: synthetic inline `#[test]`
  modules and their test-only dependencies are removed. Only the three
  environment-driven real-trace/real-sidecar Cargo probes remain.
- Full replay is primary; fast replay is complementary.
- NCU/CUPTI/CUDA-event measurements are independent observations joined after
  replay, never replay inputs.

## Current evidence

Canonical recursive replay completes 327/327 captures and 11,529/11,529
launches in both modes.  The trace-scoped sidecar index joins 1,932 TMA rows
and 402,208 synchronization rows with zero missing joins or payload-byte
fields.  The corrected unit-aware comparator joins every launch pair with no
unmatched rows. The post-cleanup corpus driver replayed 339/339 full and
339/339 fast roots (11,558 launches per mode; 327 canonical rows plus 12
controls) with no failures. Outputs are under
`/dev/shm/sm120_rtx_5090_work/cleanup_20261012/replay_all/`.
The post-test-surface cleanup rerun is under
`/dev/shm/sm120_rtx_5090_work/cleanup_20261012/replay_after_test_cleanup/`:
339/339 roots and 11,558 launches per mode, with 789.156 s full and 638.017 s
fast driver totals. The canonical subset remains 327/327 and 11,529 launches.

Residual native limitations remain explicit: proprietary issue-port and cache
policy details are not inferred, and native TMA/mbarrier completion timestamps
are not present in the captured evidence.  Do not introduce per-workload
corrections or NCU-derived replay inputs.

## Working rules

Use complete real workloads and real sidecars.  Do not add, restore, invoke, or
maintain fixture/synthetic simulator tests or procedural hash/check/validation/
verification/admission/schema gates.  Production checked arithmetic and
fail-closed input handling remain runtime behavior, not acceptance tests.

Keep this file short.  Detailed rows and joins belong beside the replay output,
not in continuation memory.
