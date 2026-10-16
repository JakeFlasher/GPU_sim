# Simulator status — SM120

Updated 2026-10-16. The simulator and replay tools build from the `sim/`
workspace. The complete real corpus has 339 roots and 11,558 launches per
lane; full integrated L1+L2 replay precedes the fast flat frontier and uses
exactly twelve workers.

## Runtime

- Protocol-specific mbarrier phase queries are distinct from async-copy-group
  waits and require no routing switch.
- TMA, LDGSTS, deferred arrivals, barrier generations, and signed/unknown
  accounting are represented independently. Unsupported directions and
  malformed input fail closed.
- Full SM120 replay rejects flat and L1-only memory modes, resolves effective
  L1/L2 scenarios when omitted, and keeps fast replay flat.
- Replay and post-replay JSONL use `.jsonl.zst` through `tools/jsonl_io.py`.
  Raw trace sidecars and workload manifests are inputs and remain unchanged.
- TMA provenance, reuse decomposition, and clock-sandwich fields are
  observation-only. `completion_clock` is unavailable.

## Evidence

The authoritative complete-corpus result is available through
`/root/sm120_rtx_5090_capture/replay_results/authoritative_full_replay_20261016/`.
The latest post-cleanup verification is retained at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016/`; it
contains successful full and fast streams for every root and launch.
The current shared-memory refinement result is at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_shared_memory/`.
The current shared-memory and atomic-observer result is at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_shared_memory_atomic_alias/`.
The current effective-observation result is at
`/root/sm120_rtx_5090_capture/replay_results/current_full_replay_20261016_effective_observation/`.
Representative full-lane mean APEs are:

| Metric | Mean APE |
| --- | ---: |
| Model-relative active IPC | 9.5537% |
| Source-TEX read lookup miss | 0.3474% |
| Source-TEX mixed lookup miss | 2.3031% |
| Unified L1 lookup | 60.7210% |
| LSU global-load L1 lookup | 44.5013% |
| DRAM read sectors | 50.5194% |

The shared-memory projection now partitions ordinary LDS/STS opcode classes
from LDSM/STSM and accumulates bank-conflict excess per instruction. Its
complete-corpus means are **0.1049%** for shared-load SASS instructions,
**16.3564%** for shared-load wavefronts, **3.7771%** for shared-store
wavefronts, **49.2552%** for shared-load bank conflicts, and **30.8590%** for
shared-store bank conflicts; no other scored metric changed.

The scoped global-atomic observer arm reduces shared-atomic wavefront mean APE
from **64.4022%** to **1.6116%** without changing other scored metrics.

The elapsed-IPC observation boundary reduces mean APE from **20.3993%** to
**13.6860%** and P90 from **38.5692%** to **27.8167%**; maximum and other
metric families are unchanged.

These scores are effective metric projections. Aggregate L1 client scope,
DRAM/service and topology, and engine completion remain explicit residuals.
Metrics absent from both lanes are reported by family with counts and reasons;
they are never scored as zero.

## Development

The authoritative model contract is in the external proposal `PLAN.md`.
Component responsibilities are in `ARCHITECTURE.md` and `COMPONENTS.md`; the
complete-corpus command is in `runbook_host.md`. Keep temporary work under
`/dev/shm/sm120_alpha_tmp_20261013` and retain compact evidence only. Do not
add synthetic fixtures or procedural approval machinery.
