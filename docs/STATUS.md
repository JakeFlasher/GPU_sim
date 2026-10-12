# Current status

**Updated:** 2026-10-12
**Branch:** `GPUSim_main`
**Active plan:** `/root/gpu_cache_memory_modeling_proposal/PLAN.md`
**Corpus:** `/root/sm120_rtx_5090_capture`
**Replay workspace:** `/dev/shm/sm120_rtx_5090_work`  
**Cleanup commit:** `14b40f26`

## Alpha evidence policy

The live lane keeps only real-trace/real-sidecar functionality probes.  Small
fixtures and synthetic simulator runs are not acceptance evidence, and no
hashing, checking, validation, verification, admission, or schema test gate is
used.  Acceptance is full-suite generalization and performance against
independent NCU ground truth; full replay precedes fast replay.

## Replay state

The SM120/RTX 5090 path is executable.  The canonical recursive corpus has
327/327 captures and 11,529/11,529 launches in both full and fast modes.  The
trace-scoped typed-sidecar join covers 1,932 TMA observations and 402,208 sync
observations with no missing TMA/sync joins or missing payload-byte fields.

The corrected unit-aware comparator joins all 11,529 grid/block pairs (97
identity joins and 11,432 ordered fallbacks) with no unmatched rows.  NCU
values remain comparison targets only; native async completion timestamps and
proprietary physical topology remain explicit gaps.

## Active implementation

- `sim-run --sm120-alpha --mode flat` is the primary full replay.
- `sim-run --sm120-alpha --fast-frontier-only` is complementary diagnostics.
- `tools/sm120_replay_bench.py` drives complete-corpus replay and joins.
- `tools/sm120_trace_features.py`, `sm120_async_state_model.py`,
  `sm120_dependency_features.py`, and `sm120_memory_service_model.py` provide
  conservative trace-derived features and typed service envelopes.

No endpoint coefficient or scored model change is admitted without independent
whole-corpus evidence.  Keep residual gaps and unit/timing domains visible in
reports.

## Repository hygiene

The checkout now contains the active Rust simulator, RTX 5090 architecture
inputs, NVBit producer, SOLBench sources, and the complete-corpus replay tools.
Superseded local-label/calibration lanes, fixture bundles, small-fixture test
harnesses, and stale SM86 campaign outputs are not part of the live branch.
