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
It contains successful full and fast streams for every root and launch.
Representative full-lane mean APEs are:

| Metric | Mean APE |
| --- | ---: |
| Model-relative active IPC | 9.5537% |
| Source-TEX read lookup miss | 0.3474% |
| Source-TEX mixed lookup miss | 2.3031% |
| Unified L1 lookup | 60.7210% |
| LSU global-load L1 lookup | 44.5013% |
| DRAM read sectors | 50.5194% |

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
