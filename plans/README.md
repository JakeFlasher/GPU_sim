# Plans

The active direction is the software-only roadmap
[cross_architecture_gpu_simulator_research_plan_20260923.md](cross_architecture_gpu_simulator_research_plan_20260923.md),
with the evidence and prior-art review in
[docs/gpu_simulator_research_positioning_20260923.md](../docs/gpu_simulator_research_positioning_20260923.md).
It keeps the SM86 NVBit/NCU path and the bounded custom-SIMT feasibility seam;
its completion bar is final NCU accuracy plus simulator wall time.

The [adversarial roadmap review](../docs/roadmap_adversarial_review_20260923.md)
expands the current milestones into executable steps, exit criteria, and
fallback decisions.

The following files are retained context, not active gates or instructions:

- [SM86 simulator roadmap](micro2027_ncu_sim_execution_plan.md)
- [Live capture plan](sm86_comprehensive_live_capture_campaign_plan.md)
- [Current closure slice](sm86_simulator_live_baseline_core_closure.md)
- [Heterogeneous novelty plan](heterogeneous_novelty_implementation_plan.md)
- [Exact-cycle acceleration plan](exact_cycle_preserving_acceleration_carry_on_20260912.md)
- [Typed L2/power plan](typed_l2_ownership_power_certification_plan_20260912.md)

Older plans may mention RTL, silicon, hashes, receipts, audits, or broad test
campaigns. Those are historical data; do not revive them for this alpha branch.
Do not add process machinery around a replay. Use the current roadmap, code,
architecture cards, and final machine-grounded NCU/speed runs.
