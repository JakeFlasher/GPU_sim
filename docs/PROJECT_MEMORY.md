# Continuation memory

**Updated:** 2026-10-13

- Target: GeForce RTX 5090, `sm_120`.
- Active plan: `/root/gpu_cache_memory_modeling_proposal/PLAN.md`.
- Canonical corpus: `/root/sm120_rtx_5090_capture` (339 direct captures and
  339 matching profile roots).
- Detailed full replay is primary; fast frontier replay is complementary.
- The saved corpus run covers 11,558 launches in each mode. Driver walls were
  827.656 s full and 666.395 s fast.
- NCU and CUDA-event observations are independent timing surfaces, never replay
  inputs. The saved comparison has 98 identity and 11,460 ordered joins per
  mode.
- The source carries raw async destination identity, the measured strict
  SM120 cache-operation route, and explicit unavailable/unique/ambiguous async
  timing evidence. Native issue topology, cache geometry/service, and completion
  clocks remain open because the 339-root corpus exports no device clocks.
