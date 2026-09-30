> **Superseded context (2026-09-30).** This file is retained for historical source/data ancestry, not as an active plan or task queue. The compact capability boundary is [`docs/STATUS.md`](../docs/STATUS.md); root `AGENTS.md` controls execution.

> Dated context; current capabilities and open work are in [STATUS](../docs/STATUS.md).
> Historical completion claims and next-step prose are not active instructions.

# SM86 simulator roadmap

This remains the SM86 workstream within the
[general GPU simulator roadmap](cross_architecture_gpu_simulator_research_plan_20260923.md).
Keep its silicon evidence separate from the new hypothetical architecture path.

## Goal

Make the trace driven SM86 simulator useful for workload comparison while
keeping issue, memory service, completion, and launch observables separate.

## Current implementation

- NVBit protobuf traces and SASS control sidecars are decoded by `sim-trace`.
- `sim-sm` models warp dependencies, barriers, async copies, control flow,
  issue ports, and typed completion state.
- `sim-mem` models L1/TEX and shared L2 request/response state.
- `sim-run` emits machine readable results and errors.
- The latest 15-workload SM86 replay is in
  `results/captures/sm86_adversarial_current_campaign_20260904_v11`.

## Next slices

1. Replay a representative current trace after scheduler or memory changes.
2. Measure queue/event cost only when a replay identifies a bottleneck.
3. Preserve exact trace traffic while reducing avoidable allocation and queue
   overhead.
4. Extend architecture capabilities from SM120 traces and cards, not Ampere
   defaults.

## Out of scope

A new approval workflow, per-workload timing constants, host command coupling,
and a replacement for NCU measurement are outside the simulator.
