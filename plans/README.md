# Plans

The canonical continuation roadmap is the
[general GPU simulator research plan](cross_architecture_gpu_simulator_research_plan_20260923.md).
It preserves SM86 validation while adding a separately scoped user-defined
architecture path. Supporting roadmaps and historical context:

- [SM86 simulator roadmap](micro2027_ncu_sim_execution_plan.md) — simulator
  fidelity and performance work.
- [Live capture plan](sm86_comprehensive_live_capture_campaign_plan.md) —
  direct NVBit and NCU workload capture.
- [Current closure slice](sm86_simulator_live_baseline_core_closure.md) — the
  active SM86 issue/memory seam.
- [Heterogeneous novelty plan](heterogeneous_novelty_implementation_plan.md) —
  earlier cross-vendor semantic IR and CDNA3/MI300X proposal; AMD timing is
  deferred, and the new roadmap starts with an open/custom execution target.

All implementation decisions should be expressed in code, architecture cards,
small workload probes, or current result roots. Keep a plan short enough to
read before running the next replay.
