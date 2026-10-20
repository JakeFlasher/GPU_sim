//! Cohesive child mechanism extracted from the parent module.

use super::*;

impl<'a> Sm86MetricState<'a> {
    pub(crate) fn add_sm86_readiness(&mut self) -> Result<(), RunError> {
        let result = self.result;
        let external = self.external;
        let card = self.card;
        let trace = result.trace_metrics();
        let sm_count = self.sm_count;
        let active_sm_count = self.active_sm_count;
        let clock = self.clock;
        let mut m = &mut self.metrics;
        let instructions = &trace.instructions;
        let inst = u128::from(instructions.warp_instructions);
        let active: u128 = result
            .per_sm()
            .values()
            .map(|s| u128::from(s.activity.sm_active_cycles))
            .sum();
        let warp_cycles: u128 = result
            .per_sm()
            .values()
            .map(|s| s.activity.warp_active_cycles)
            .sum();
        let smsp_cycles: u128 = result
            .per_sm()
            .values()
            .flat_map(|s| s.activity.smsp_active_cycles)
            .map(u128::from)
            .sum();
        let issue_cycles: u128 = result
            .per_sm()
            .values()
            .flat_map(|s| s.activity.issue_active_cycles_per_smsp)
            .map(u128::from)
            .sum();
        let max_warps = result
            .per_sm()
            .values()
            .next()
            .map(|s| s.activity.max_resident_warps_per_sm)
            .filter(|v| *v != 0)
            .unwrap_or(1);
        let readiness = aggregate_readiness(result)?;
        // Keep the causal terminal/issue interval explicit for IPC work. The
        // width is a mechanism-local uncertainty projection; it is not silently
        // folded into any NCU endpoint or used to select a scored prediction.
        let issue_frontier = result.modeled_issue_lower_bound_cycles();
        let terminal_frontier = result
            .sm_active_terminal_event_time()
            .map(|time| time.max(issue_frontier));
        add_count_row(
            &mut m,
            "ipc_critical_path_issue_frontier_cycles",
            None,
            "cycle",
            issue_frontier,
            "causal issue lower frontier; mechanism-local IPC boundary, not an NCU endpoint",
        );
        if let Some(terminal) = terminal_frontier {
            add_count_row(
                &mut m,
                "ipc_critical_path_terminal_frontier_cycles",
                None,
                "cycle",
                terminal,
                "causal SM terminal frontier; unmodeled retirement tails remain separate",
            );
            add_count_row(
                &mut m,
                "ipc_critical_path_interval_width_cycles",
                None,
                "cycle",
                terminal.saturating_sub(issue_frontier),
                "width between issue and terminal causal frontiers; not a statistical interval",
            );
        }
        add_cause_rows(
            &mut m,
            "stall_limiting",
            &readiness.limiting_cycles,
            "exclusive model-causal readiness bucket; overlapping NCU warp-stall cycles are not reconstructed",
        );
        add_cause_rows(
            &mut m,
            "stall_exposure",
            &readiness.constraint_exposure_cycles,
            "overlapping model-causal readiness exposure; never sum as elapsed time",
        );
        let limiting = &readiness.limiting_cycles;
        let limiting_values = [
            limiting.instruction_front_end,
            limiting.control_stall,
            limiting.register_raw,
            limiting.register_predicate,
            limiting.register_waw,
            limiting.register_read_port,
            limiting.dependence_counter_read,
            limiting.dependence_counter_global_load_operand_read,
            limiting.dependence_counter_global_store_operand_read,
            limiting.dependence_counter_global_atomic_operand_read,
            limiting.dependence_counter_global_reduction_operand_read,
            limiting.dependence_counter_non_memory_write,
            limiting.dependence_counter_shared_write,
            limiting.dependence_counter_l1tex,
            limiting.dependence_counter_unknown_origin,
            limiting.async_source,
            limiting.async_destination_local,
            limiting.async_destination_sm_service,
            limiting.cta_barrier_rendezvous,
            limiting.pipeline_barrier,
            limiting.pipeline_branch,
            limiting.pipeline_tensor,
            limiting.pipeline_sfu,
            limiting.pipeline_alu,
            limiting.memory_admission_queue,
            limiting.memory_issue_queue,
            limiting.memory_write_completion,
            limiting.eligible_not_selected,
        ];
        let limiting_total = limiting_values
            .iter()
            .copied()
            .fold(0_u64, u64::saturating_add);
        let limiting_max = limiting_values.iter().copied().max().unwrap_or(0);
        let limiting_max_fraction_ppm = if limiting_total == 0 {
            0
        } else {
            u64::try_from(
                u128::from(limiting_max)
                    .saturating_mul(1_000_000)
                    .checked_div(u128::from(limiting_total))
                    .unwrap_or(0),
            )
            .unwrap_or(u64::MAX)
        };
        add_count_row(
            &mut m,
            "ipc_limiting_cause_max_fraction_ppm",
            None,
            "ppm",
            limiting_max_fraction_ppm,
            "observer-only concentration of the dominant exclusive readiness cause; overlapping exposure is separate and no frontier is selected",
        );
        add_count_row(
            &mut m,
            "ipc_limiting_cause_total_cycles",
            None,
            "cycle",
            limiting_total,
            "sum of exclusive model-causal limiting buckets; not an elapsed-cycle denominator",
        );
        add_scheduler_proxy_rows(&mut m, &readiness);
        add_sm86_register_read_rows(&mut m, &readiness);
        add_count_row(
            &mut m,
            "stall_delayed_ready_issues",
            None,
            "issue",
            readiness.delayed_ready_issues,
            "model-causal issue opportunities delayed by at least one readiness relation",
        );
        add_count_row(
            &mut m,
            "stall_tied_limiting_causes",
            None,
            "issue",
            readiness.tied_limiting_causes,
            "model-causal issue opportunities with multiple equal limiting relations",
        );
        if readiness.global_memory_issue != GlobalMemoryIssueDiagnostics::default() {
            add_global_issue_rows(&mut m, &readiness.global_memory_issue);
        }
        if let Some(front_end) = readiness.instruction_front_end.as_deref() {
            for (name, value, unit) in [
                (
                    "instructions_fetched",
                    front_end.instructions_fetched,
                    "inst",
                ),
                ("demand_l0_hits", front_end.demand_l0_hits, "line"),
                (
                    "demand_stream_buffer_hits",
                    front_end.demand_stream_buffer_hits,
                    "line",
                ),
                ("demand_l0_misses", front_end.demand_l0_misses, "line"),
                ("demand_l1_hits", front_end.demand_l1_hits, "line"),
                ("demand_l1_misses", front_end.demand_l1_misses, "line"),
                ("prefetch_requests", front_end.prefetch_requests, "line"),
                ("prefetch_l1_hits", front_end.prefetch_l1_hits, "line"),
                ("prefetch_l1_misses", front_end.prefetch_l1_misses, "line"),
                ("stream_restarts", front_end.stream_restarts, "event"),
                (
                    "maximum_fetch_latency_cycles",
                    front_end.maximum_fetch_latency_cycles,
                    "cycle",
                ),
                (
                    "fetch_last_issued_continuations",
                    front_end.fetch_last_issued_continuations,
                    "fetch",
                ),
                (
                    "yield_hold_separations",
                    front_end.yield_hold_separations,
                    "fetch",
                ),
            ] {
                add_count_row(
                    &mut m,
                    format!("instruction_front_end_{name}"),
                    None,
                    unit,
                    value,
                    "model-relative instruction-front-end observation",
                );
            }
        }
        if let Some(memory_pipeline) = readiness.memory_issue_pipeline.as_deref() {
            for (name, value, unit) in [
                (
                    "instructions_admitted",
                    memory_pipeline.instructions_admitted,
                    "inst",
                ),
                (
                    "operand_pipeline_delay_cycles",
                    memory_pipeline.operand_pipeline_delay_cycles,
                    "cycle",
                ),
                (
                    "shared_queue_delay_cycles",
                    memory_pipeline.shared_queue_delay_cycles,
                    "cycle",
                ),
                (
                    "maximum_operand_pipeline_delay_cycles",
                    memory_pipeline.maximum_operand_pipeline_delay_cycles,
                    "cycle",
                ),
                (
                    "maximum_shared_queue_delay_cycles",
                    memory_pipeline.maximum_shared_queue_delay_cycles,
                    "cycle",
                ),
                (
                    "maximum_queue_occupancy",
                    memory_pipeline.maximum_queue_occupancy,
                    "entry",
                ),
                (
                    "terminal_shared_accept_cycle",
                    memory_pipeline.terminal_shared_accept_cycle,
                    "cycle",
                ),
            ] {
                add_count_row(
                    &mut m,
                    format!("memory_issue_pipeline_{name}"),
                    None,
                    unit,
                    value,
                    "model-relative memory-issue queue observation",
                );
            }
        }
        for (name, value, unit) in [
            (
                "async_groups_committed",
                readiness.async_groups.groups_committed,
                "group",
            ),
            (
                "async_groups_waited",
                readiness.async_groups.groups_waited,
                "group",
            ),
            (
                "async_source_cycles",
                readiness.async_groups.source_cycles,
                "cycle",
            ),
            (
                "async_destination_local_cycles",
                readiness.async_groups.destination_local_cycles,
                "cycle",
            ),
            (
                "async_destination_sm_service_cycles",
                readiness.async_groups.destination_sm_service_cycles,
                "cycle",
            ),
            (
                "async_completion_hidden_before_wait_cycles",
                readiness.async_groups.completion_hidden_before_wait_cycles,
                "cycle",
            ),
            (
                "async_destination_groups_observed",
                readiness.async_groups.destination_groups_observed,
                "group",
            ),
            (
                "async_destination_wavefronts_observed",
                readiness.async_groups.destination_wavefronts_observed,
                "wavefront",
            ),
        ] {
            add_count_row(
                &mut m,
                name.to_owned(),
                None,
                unit,
                value,
                "model-relative asynchronous-copy protocol observation",
            );
        }
        m.insert(
            "warp_instructions".into(),
            Metric::count(
                Some("smsp__inst_executed.sum"),
                "inst",
                inst,
                "one issue per trace record; includes all-predicated-off records",
            ),
        );
        m.insert(
            "active_thread_instructions".into(),
            Metric::count(
                None,
                "thread_inst",
                u128::from(instructions.active_thread_instructions),
                "sum popcount(active_mask); trace population, not a pipe counter",
            ),
        );
        m.insert(
            "predicate_on_thread_instructions".into(),
            Metric::count(
                None,
                "thread_inst",
                u128::from(instructions.predicate_on_thread_instructions),
                "sum popcount(active_mask & predicate_mask); includes uniform op lane populations",
            ),
        );
        m.insert(
        "active_mask_missing_instructions".into(),
        Metric::count(
            None,
            "inst",
            u128::from(instructions.active_mask_missing_instructions),
            "trace records whose producer omitted active_mask; active-lane and CFG populations are intentionally not fabricated",
        ),
    );
        m.insert(
        "predicate_mask_missing_instructions".into(),
        Metric::count(
            None,
            "inst",
            u128::from(instructions.predicate_mask_missing_instructions),
            "trace records whose producer omitted predicate_mask; predicate-on populations are intentionally not fabricated",
        ),
    );
        m.insert(
            "warp_execution_efficiency_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(instructions.active_thread_instructions) * 100,
                inst * 32,
                false,
                "active lanes divided by 32 times warp instructions",
            ),
        );
        m.insert(
            "predicate_on_lane_efficiency_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(instructions.predicate_on_thread_instructions) * 100,
                inst * 32,
                false,
                "predicate-on lanes divided by 32 times warp instructions",
            ),
        );
        m.insert("uniform_branch_predicate_pct".into(), Metric::ratio(None, "%", u128::from(trace.branch_warp_instructions-trace.branch_mixed_predicate_warp_instructions)*100, u128::from(trace.branch_warp_instructions), false, "BRA/JMP predicate uniformity only; indirect target divergence is not captured by lane populations"));
        add_branch_rows(&mut m, trace);
        for (key, unit, value, semantics) in [
            (
                "cfg_observed_edges",
                "edge",
                trace.control_flow.observed_edges,
                "lane-consistent consecutive trace transitions; disjoint serialized paths are excluded",
            ),
            (
                "cfg_unique_edges",
                "edge",
                trace.control_flow.unique_edges,
                "unique lane-consistent dynamic control-flow edges observed in the trace",
            ),
            (
                "cfg_active_lane_transitions",
                "thread",
                trace.control_flow.active_lane_transitions,
                "active-lane transition memberships across observed dynamic edges",
            ),
            (
                "cfg_disjoint_path_switches",
                "event",
                trace.control_flow.disjoint_path_switches,
                "serialized divergent-path switches excluded from edge construction",
            ),
        ] {
            add_count_row(&mut m, key, None, unit, value, semantics);
        }
        // Static cross-SM line sharing is a useful prerequisite for future
        // coherence modeling, but it is not a hardware coherence transaction
        // counter.  Keep the rows unavailable when the replay intentionally did
        // not decode addresses (flat mode) rather than turning an absent address
        // domain into a false zero.
        for (key, unit, value, semantics) in [
            (
                "coherence_decoded_global_lines",
                "line",
                trace.coherence.decoded_global_lines,
                "unique decoded global line-domain keys touched by this launch; trace sharing domain, not physical-page identity",
            ),
            (
                "coherence_cross_sm_lines",
                "line",
                trace.coherence.cross_sm_lines,
                "decoded lines touched by more than one traced SM; static sharing domain, not an invalidation count",
            ),
            (
                "coherence_cross_sm_sectors",
                "sector",
                trace.coherence.cross_sm_sectors,
                "unique sectors on decoded cross-SM lines; not completed coherence or DRAM traffic",
            ),
            (
                "coherence_cross_sm_sector_overlap",
                "sector",
                trace.coherence.cross_sm_sector_overlap,
                "decoded 32-byte sectors touched by more than one traced SM; tighter contention domain than line sharing, not a coherence transaction count",
            ),
            (
                "coherence_cross_sm_read_write_lines",
                "line",
                trace.coherence.cross_sm_read_write_lines,
                "cross-SM lines containing both read and write access classes; temporal ownership is not inferred",
            ),
            (
                "coherence_cross_sm_write_write_lines",
                "line",
                trace.coherence.cross_sm_write_write_lines,
                "cross-SM lines containing writes from multiple SMs; atomicity and ordering remain unmodeled",
            ),
            (
                "coherence_cross_sm_read_write_sectors",
                "sector",
                trace.coherence.cross_sm_read_write_sectors,
                "decoded sectors touched by both read and write classes from multiple SMs; protocol ordering remains unmodeled",
            ),
            (
                "coherence_cross_sm_write_write_sectors",
                "sector",
                trace.coherence.cross_sm_write_write_sectors,
                "decoded sectors written by multiple traced SMs; atomicity and ordering remain unmodeled",
            ),
            (
                "coherence_unknown_sm_or_line_records",
                "record",
                trace.coherence.unknown_sm_or_line_records,
                "global access records excluded from the static sharing domain because SM or sector identity was unavailable",
            ),
        ] {
            m.insert(
                key.into(),
                if trace.sectors_decoded {
                    if key == "coherence_decoded_global_lines" {
                        Metric::count(None, unit, u128::from(value), semantics)
                    } else {
                        Metric::approximate_count(None, unit, u128::from(value), semantics)
                    }
                } else {
                    Metric::unavailable(
                        None,
                        unit,
                        "flat replay does not decode global sector addresses",
                    )
                },
            );
        }
        // A deliberately abstract coherence service envelope for decoded traces:
        // one model service cycle per cross-SM overlapping 32-byte sector. This
        // is a mechanism-local design knob, not a measured protocol latency; the
        // native completion tail remains asserted and no NCU coherence counter is
        // synthesized.
        let coherence_lower_bound = result.modeled_terminal_lower_bound_cycles();
        let coherence_width = u128::from(trace.coherence.cross_sm_sector_overlap);
        let coherence_topology_complete = trace.coherence.unknown_sm_or_line_records == 0;
        let coherence_scenario_end = coherence_lower_bound.and_then(|lower| {
            u64::try_from(coherence_width)
                .ok()
                .and_then(|width| lower.checked_add(width))
        });
        for (key, value, semantics) in [
            (
                "coherence_completion_frontier_cycles",
                coherence_lower_bound.map(u128::from),
                "last modeled event before the abstract cross-SM coherence service envelope; lower bound only",
            ),
            (
                "coherence_completion_scenario_end_cycles",
                coherence_scenario_end.map(u128::from),
                "modeled terminal frontier plus one abstract service cycle per overlapping cross-SM sector under the declared scenario policy; not a protocol timing claim",
            ),
            (
                "coherence_completion_service_width_cycles",
                Some(coherence_width),
                "abstract service-width component of the cross-SM coherence envelope; not a hardware counter",
            ),
        ] {
            m.insert(
                key.into(),
                if trace.sectors_decoded && coherence_topology_complete {
                    match value {
                        Some(value) => Metric::approximate_count(None, "cycle", value, semantics),
                        None => Metric::unavailable(
                            None,
                            "cycle",
                            "requires a modeled terminal event frontier",
                        ),
                    }
                } else {
                    Metric::unavailable(
                        None,
                        "cycle",
                        if !trace.sectors_decoded {
                            "flat replay does not decode cross-SM sector topology"
                        } else {
                            "cross-SM topology contains unknown SM/sector records"
                        },
                    )
                },
            );
        }
        // Source requests are warp instructions with at least one effective lane.
        // Sector counts are separate; validated ATOM/RED spans preserve the
        // alias-expanded transaction surface used by the SM86 write observer.

        Ok(())
    }
}
