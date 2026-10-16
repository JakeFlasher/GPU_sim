# Current SM120 state

This file is a compact current-state contract, not a development diary.

## Scope

- Target: RTX 5090, compute capability `sm_120`.
- Source tree: `/root/GPU_simulator_rebase`.
- Real captures/profiles: `/root/sm120_rtx_5090_capture`.
- Complete corpus: 339 roots and 11,558 launches per replay lane.
- Full replay is primary and uses integrated L1+L2; the fast lane is a flat
  frontier. Complete runs use twelve worker processes and full-before-fast order.
- NCU and CUDA observations are post-replay comparisons, never replay inputs.

## Implemented contract

The native pipeline is:

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
NCU/CUDA observations ----------------------------------------------> report
```

`SYNCS.PHASECHK` is a protocol-specific mbarrier operation. Async-copy-group
waits, TMA submissions, deferred arrivals, LDGSTS, barrier generations, and
signed/unknown accounting remain separate. Unsupported forms fail closed.

Full SM120 replay resolves the effective L1/L2 scenarios when omitted and
rejects flat or L1-only full runs. The fast frontier remains flat. Replay and
comparison JSONL are written as `.jsonl.zst` through `tools/jsonl_io.py`; the
reader accepts a legacy uncompressed sibling during migration.

TMA provenance, sector/line reuse, and same-domain clock-sandwich summaries are
read-only diagnostics. They do not add cache demand, scheduling events, DRAM
service, or completion timestamps. `completion_clock` is unavailable because
no supported interface exposes a generation-keyed engine event and timestamp.

## Evidence summary

The canonical result is exposed as:

`/root/sm120_rtx_5090_capture/replay_results/authoritative_full_replay_20261016/`
The latest post-cleanup verification is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016/`; it
completed the same full-before-fast 339-root run with twelve workers.
The current shared-memory refinement result is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_shared_memory/`.
The current shared-memory and atomic-observer result is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_shared_memory_atomic_alias/`.
The current effective-observation result is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_effective_observation/`.
The current partial-store RMW result is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_partial_store_rmw/`.
The current partial global/local-store RMW result is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_partial_store_rmw_local/`.

Current full-lane mean APEs include:

- model-relative active IPC: 9.5537%;
- source-TEX operation-read lookup misses: 0.3474%;
- source-TEX mixed lookup misses: 2.3031%;
- unified L1 lookup: 60.7210%;
- LSU global-load L1 lookup: 44.5013%;
- unqualified DRAM read sectors: 50.5194%.

The shared-memory projection now partitions ordinary LDS/STS opcode classes
from LDSM/STSM and accumulates bank-conflict excess per instruction. The
corresponding full-lane means are 0.1049% (shared-load SASS instructions),
16.3564% (shared-load wavefronts), 3.7771% (shared-store wavefronts),
49.2552% (shared-load bank conflicts), and 30.8590% (shared-store bank
conflicts), with no regression in other scored metrics.

The scoped global-atomic observer arm reduces shared-atomic wavefront mean APE
from 64.4022% to 1.6116%; global atomic opcode classification remains separate.

The launch-boundary elapsed-IPC observation arm reduces mean APE from 20.3993%
to 13.6860% and P90 from 38.5692% to 27.8167%; maximum and other metric
families are unchanged.

The partial-sector global-store RMW arm reduces DRAM read mean APE from 50.5194%
to 50.3123% and P90 from 90.9404% to 90.7480%, with no regression in other
scored metrics.

Including decoded local-store byte gaps further reduces DRAM read mean APE to
50.3119% without changing other scored metrics.

Aggregate L1 scope, unqualified DRAM/service and physical topology, and the
internal TMA completion clock remain typed residuals. A boundary-clipped TMA
read-hit deficit is preserved as an additive lookup population absent from the
valid address trace; the simulator does not manufacture hits from descriptor or
out-of-bounds padding.

## Working rules

Read `docs/STATUS.md`, `docs/ARCHITECTURE.md`, and the external proposal
`/root/gpu_cache_memory_modeling_proposal/PLAN.md` before changing the model.
Use real traces and sidecars for evidence. Keep temporary files in
`/dev/shm/sm120_alpha_tmp_20261013`, retain only compact results under the
canonical corpus, and do not add fixture, approval, hashing, or admission
workflows.
