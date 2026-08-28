# Documentation

These are the current project documents:

- [Architecture](ARCHITECTURE.md): simulator components and data flow.
- [Status](STATUS.md): implemented features and immediate research work.
- [Host runbook](runbook_host.md): direct build, capture, and test commands.
- [Testing](testing.md): proportionate correctness checks.
- [NVBit tracing](nvbit-tracing.md): tracer output and usage.
- [Metric registry](metric-registry.md): NCU metric names and units.
- [Architecture profiles](archprofile.md): device-card inputs.
- [Predictors](predictors.md): experimental comparison models.
- [GPU simulator notes synthesis](gpu_sims_notes_research_synthesis_20260817.md):
  corpus-wide findings, current-code gap map, and milestone decision.
- [sm_86 causal readiness milestone](sm86_causal_readiness_milestone.md):
  completed implementation, controlled GPU probe, and falsified fixed-latency
  result.
- [sm_86 barrier pipeline and cache-handoff milestone](sm86_barrier_pipeline_cache_handoff_milestone.md):
  completed traced warm L2 state, barrier throughput, coupled route, and
  multi-CTA validation.
- [sm_86 LDGSTS front-end decision](../results/captures/sm86_ldgsts_front_end_transfer_20260828_v1/README.md):
  56 CPU replays reject the simplified composite on transfer, guard, route,
  interaction-sign, and all-replay accounting gates; its active code is retired.
- [sm_86 execution-port and control-pipeline milestone](sm86_issue_port_control_pipeline_milestone.md):
  completed dependent/independent pipe study, exact forward-branch service,
  and rejected scalar/dual-issue candidates.
- [sm_86 scheduler and math-pipe contention milestone](sm86_scheduler_math_pipe_contention_milestone.md):
  completed 99-cell width/phase/order campaign, identified half-warp service
  saturation, and rejected non-transferring scheduler/queue candidates.
- [sm_86 Control/Allocate and register-file milestone](sm86_control_allocate_register_file_milestone.md):
  completed operand projection, exact-bank and completion-distance campaigns,
  and rejected non-transferring timing candidates.
- [sm_86 reuse cache and result/writeback queue milestone](sm86_reuse_result_queue_milestone.md):
  completed exact RFC identity campaign and rejected active timing placements.
- [sm_86 result, bypass, and writeback queue milestone](sm86_result_writeback_queue_milestone.md):
  completed pacing, order, same-bank, capacity-feasibility, and fixed/L1TEX
  campaigns plus the direct consumer frontier; identified two-result and
  distance-4 fixed bypass, rejected capacity-8/pop-1, and promoted narrow
  bank-local fixed-over-load writeback priority. A dense W4–W16 follow-up
  localizes the remaining shift to a per-SMSP front-end phase and rejects ten
  non-transferring IBuffer/fetch models. Its U1/U2/U4/U8 footprint successor
  identifies three uniform backward-target resolution cycles from direct NCU
  counters, promotes that typed readiness fact, and leaves IBuffer capacity
  inactive. A branch-free CLOCK successor leaves capacity two as the sole
  candidate but rejects the full front-end state because cache parameters and
  exact issue order do not transfer.
- [sm_86 atomic completion and sector-serialization milestone](sm86_atomic_completion_serialization_milestone.md):
  completed ATOMG/RED return and address-layout campaign; fixes alias-aware
  atomic sector projection, rejects an extra same-address penalty, and retains
  a low-error sector-service queue only as an adversarial successor candidate.
- [sm_86 adversarial atomic topology and cross-SM service milestone](sm86_atomic_adversarial_milestone.md):
  supersedes that timing candidate with 244 runtime-address/multi-SM cells;
  promotes exact L2 request topology and CLOCK intervals, then rejects
  operation-only, per-SM, spacing-threshold, and ordinary-cache-hash timing. A
  544-cell pairwise closure promotes the exact 256-byte contention region.
- [sm_86 launch-cycle, completion, and range-semantics milestone](sm86_launch_cycle_semantics_milestone.md):
  separates issue, modeled events, incomplete retirement tails, per-kernel NCU
  envelopes, and untraced multi-launch range gaps; promotes a typed completion
  boundary while rejecting numerical GPC-cycle emission.
- [sm_86 memory-retirement service milestone](sm86_memory_retirement_service_milestone.md):
  completes the depth/lane/layout/reuse/fence adversarial campaign, promotes
  readiness schema 6 ideal global-memory work, and rejects fixed store/RED
  tail delays while retaining incomplete retirement status.
- [sm_86 L1TEX arrival/output-replay service milestone](sm86_l1tex_arrival_replay_service_milestone.md):
  completes fixed-work concurrency, CTA, skew, and topology transfer; promotes
  exact unique-line/atomic-request work and rejects fixed-rate and output-replay
  timing models.
- [bounded sm_86 streaming memory-service candidate milestone](sm86_streaming_memory_service_candidates_milestone.md):
  adds fixed-capacity schema-7 evaluation at actual issue timestamps, rejects
  SMSP-local scope, and retains strong SM-shared STORE/RED rates only as
  prospective finite-queue hypotheses.
- [sm_86 L1TEX admission/backpressure and reuse milestone](sm86_l1tex_admission_backpressure_milestone.md):
  completes a prospective CLOCK/NCU/NVBit campaign, identifies compiler
  operand-read admission barriers, promotes schema 8 and direct L2 source
  provenance, rejects frozen line rates on unseen reuse, and completes the
  schema-9 finite hit/miss-aware admission-candidate successor without
  promoting a capacity or timing rule.
- [sm_86 finite L1TEX admission-candidate milestone](sm86_l1tex_finite_admission_candidates_milestone.md):
  evaluates finite SM/per-SMSP queues from exact per-instruction L2 outcomes,
  promotes schema 9, rejects open-loop backpressure, and configures one
  feedback-coupled replay lane.
- [sm_86 feedback-coupled L1TEX admission milestone](sm86_l1tex_feedback_admission_milestone.md):
  promotes an opt-in schema-10 queue/readiness feedback mechanism, executes
  downstream cache-order changes, corrects target-operation attribution, and
  withholds the frozen capacities and rates after strict held-out transfer
  failure.
- [sm_86 feedback miss-service rate/width milestone](sm86_l1tex_feedback_service_rate_width_milestone.md):
  adds a backward-compatible fixed miss-sector quantum, rejects every tested
  rate/width default on strict transfer, and isolates write-class completion as
  the next independent frontier.
- [sm_86 write-class completion feedback milestone](sm86_write_class_completion_feedback_milestone.md):
  promotes an opt-in schema-11 per-warp completion/MEMBAR relation, sharply
  improves STORE fence transfer, and retains RED RMW and STORE visibility tails
  as typed incomplete capabilities.
- [sm_86 write-completion latency milestone](sm86_write_completion_latency_milestone.md):
  adds a non-occupying schema-12 visibility/RMW tail, rejects every nonzero
  latency on joint fence/GPC transfer, and isolates work-dependent RED RMW
  completion as the next component.
- [sm_86 RED RMW completion-work milestone](sm86_red_rmw_completion_work_milestone.md):
  adds schema-13 exact request/256-byte-region work with bounded completion
  lanes, rejects a uniform rate or width after strict transfer and repeat
  sensitivity, and advances simulator event-throughput profiling.
- [sm_86 many-region atomic/LTS scheduling milestone](sm86_atomic_many_region_lts_milestone.md):
  fixes allocation geometry, identifies maximum LTS-slice RED occupancy as the
  exact local triplet discriminator, and withholds timing because every tested
  address hash fails collision transfer.
- [sm_86 GA102 address-to-LTS mapping milestone](sm86_atomic_lts_address_mapping_milestone.md):
  adds exact six-way weighted and one-launch 48-class collision oracles,
  classifies 98,303 dense/held-out addresses, and rejects every address-only
  hash after the high-address holdout exposes non-transferable local structure.
- [sm_86 CUDA-VMM alias discriminator](sm86_atomic_vmm_alias_milestone.md):
  maps one physical allocation through seven virtual proxies, rejects read
  maxima as a slice oracle, and proves with self-validating sequential RED that
  absolute atomic LTS identity follows physical placement while the complete
  2 MiB collision partition transfers exactly; extended/cross-allocation tests
  then reject relative offset as a universal key because granule phase varies.
- [simulator event-throughput milestone](simulator_event_throughput_milestone.md):
  measures pinned-CPU wall time, memory, exact event composition, and phase
  cost; removes static phase reparsing and feedback-state cloning for up to a
  1.42× matched speedup without changing modeled bytes.
- [deterministic same-cycle event-batch milestone](simulator_same_cycle_event_batch_milestone.md):
  separates logical from physical events, cuts deep STORE/RED queue depth 96.8%,
  and improves time and memory without changing replay bytes.
- [compact direct trace-lifetime milestone](simulator_compact_trace_lifetime_milestone.md):
  releases raw payload bytes after unchanged validation, cutting peak RSS up to
  29.5% while preserving raw/campaign loaders and replay bytes.
- [one-time simulator adversarial hardening](simulator_adversarial_hardening_20260820.md):
  independently verifies the two optimizations, repairs feature/case binding,
  paired and phase reductions, replay/analyzer validation, and surplus-output
  detection, then regenerates the complete evidence and replay corpus.
- [compact validated-instruction identity](simulator_compact_identity_milestone.md):
  falsifies streaming payload reads as the dominant-case optimization, hoists
  validation-proven identity from every record to its warp, and reduces the
  retained instruction from 32 to 20 bytes and cohort RSS by 7.8–22.4%.
- [SOL-ExecBench](sol-execbench.md): optional retained benchmark/campaign
  capability; invoke through `.claude/skills/run-sol-campaign/SKILL.md`.

The active roadmaps are:

- [Simulator roadmap](../plans/micro2027_ncu_sim_execution_plan.md)
- [sm_86 capture roadmap](../plans/sm86_comprehensive_live_capture_campaign_plan.md)
- [Simulator event-throughput milestone](../plans/simulator_event_throughput_milestone.md)
- [Deterministic same-cycle event-batch milestone](../plans/same_cycle_event_batch_milestone.md)
- [Direct-replay trace snapshot lifetime milestone](../plans/trace_snapshot_lifetime_milestone.md)
- [Compact validated-instruction identity milestone](../plans/compact_validated_instruction_identity_milestone.md)
- [sm_86 many-region atomic/LTS scheduling milestone](../plans/sm86_atomic_many_region_lts_milestone.md)
- [sm_86 GA102 address-to-LTS mapping milestone](../plans/sm86_atomic_lts_address_mapping_milestone.md)
- [sm_86 CUDA-VMM alias discriminator](../plans/sm86_atomic_vmm_alias_milestone.md)

Documentation describes technical behavior and current evidence. It does not
create permissions, locks, approval states, or mandatory review workflows.
