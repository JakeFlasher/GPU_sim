# Plans

The only active plan is the [software-only GPU simulator roadmap](cross_architecture_gpu_simulator_research_plan_20260923.md).
It covers the native SM86 trace/NCU path and the bounded custom-SIMT resource
frontend. Its current evidence bar is a declared NCU observable plus honest
simulator wall-time/design evidence; fair Accel-Sim/MGPUSim numerical comparison
is deferred.

Use [current status](../docs/STATUS.md), [project memory](../docs/PROJECT_MEMORY.md),
and the [adversarial synthesis](../docs/adversarial_review_and_research_20260924.md)
before opening a dated plan. The following are retained context or operational
notes, not active gates:

- [SM86 workstream](micro2027_ncu_sim_execution_plan.md)
- [Live capture commands](sm86_comprehensive_live_capture_campaign_plan.md)
- [Baseline closure slice](sm86_simulator_live_baseline_core_closure.md)
- [Heterogeneous proposal](heterogeneous_novelty_implementation_plan.md)
- [Exact-cycle acceleration study](exact_cycle_preserving_acceleration_carry_on_20260912.md)
- [Typed L2/power study](typed_l2_ownership_power_certification_plan_20260912.md)

Historical plans may contain RTL, AMD, silicon, hashes, receipts, or audit
language. Treat those as dated research context. Root `AGENTS.md` governs
execution, and no plan creates workflow machinery or approval gates.
