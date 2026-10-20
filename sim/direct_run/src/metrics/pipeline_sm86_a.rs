//! Cohesive child mechanism extracted from the parent module.

use super::*;

impl<'a> Sm86MetricState<'a> {
    pub(crate) fn add_sm86_core(&mut self) -> Result<(), RunError> {
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
        let trace = result.trace_metrics();
        // SM86's NCU global-load opcode endpoint includes LDG.CONSTANT, while
        // LDGSTS remains a separate async-copy endpoint.  Publish the
        // form-qualified LDG union as the target row; retain LTC128B and LDGSTS
        // as separate scope observers because mixed LTC128B captures demonstrate
        // a client-scope mismatch rather than a universal inclusion rule.
        let global_load_opcode_count = ["global_load_ldg", "global_load_constant"]
            .iter()
            .filter_map(|scope| trace.memory_scope.get(*scope))
            .map(|row| row.requests)
            .fold(0_u64, u64::saturating_add);
        add_count_row(
            &mut m,
            "global_load_opcode_count",
            Some("smsp__sass_inst_executed_op_global_ld.sum"),
            "inst",
            global_load_opcode_count,
            "form-qualified LDG SASS scope candidate; includes CONSTANT forms, excludes LTC128B/LDGSTS until client crosswalk evidence exists",
        );
        // Keep the static-SASS scope partitions so downstream reports can compare
        // plain LDG, CONSTANT, LTC128B, and LDGSTS independently without moving
        // any L1/L2 sectors or selecting a timing route.
        for (key, ncu_metric, scope, semantics) in [(
            "ldgsts_opcode_count",
            "smsp__inst_executed_op_ldgsts.sum",
            "ldgsts_async_copy",
            "trace-derived LDGSTS opcode scope candidate; source-cache route remains separate",
        )] {
            let count = trace.memory_scope.get(scope).map_or(0, |row| row.requests);
            add_count_row(&mut m, key, Some(ncu_metric), "inst", count, semantics);
        }
        let instructions = &trace.instructions;
        let inst = u128::from(instructions.warp_instructions);
        // NCU's `.avg` roll-up is across the fixed SM86 SMSP population, not just
        // the SMSPs that happened to receive a trace record. Keep this derived
        // row separate from the active-lifetime ratios below.
        m.insert(
        "warp_instructions_avg".into(),
        Metric::ratio(
            Some("smsp__inst_executed.avg"),
            "inst",
            inst,
            u128::from(sm_count) * 4,
            true,
            "trace instruction total divided by the carded SM86 SMSP population; mirrors the NCU average roll-up without consuming NCU values",
        ),
    );
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
        let peak_live_warps_sum: u128 = result
            .per_sm()
            .values()
            .map(|s| u128::from(s.activity.peak_live_warps))
            .sum();
        let peak_live_warps_max = result
            .per_sm()
            .values()
            .map(|s| s.activity.peak_live_warps)
            .max()
            .unwrap_or(0);
        let max_warps = result
            .per_sm()
            .values()
            .next()
            .map(|s| s.activity.max_resident_warps_per_sm)
            .filter(|v| *v != 0)
            .ok_or_else(|| RunError::new("metric occupancy", "missing resident-warp capacity"))?;
        pub(super) const LIFE: &str = "modeled admission through last instruction inclusive; excludes physical EXIT drain; ratio of summed counters, not mean per-SM percentages";
        m.insert(
            "sm_active_cycles_avg".into(),
            Metric::ratio(
                Some("sm__cycles_active.avg"),
                "cycle",
                active,
                u128::from(sm_count),
                true,
                LIFE,
            ),
        );
        m.insert(
            "sm_cycles_active".into(),
            Metric::ratio(
                Some("sm__cycles_active.avg"),
                "cycle",
                active,
                u128::from(sm_count),
                true,
                LIFE,
            ),
        );
        m.insert(
            "smsp_active_cycles_avg".into(),
            Metric::ratio(
                Some("smsp__cycles_active.avg"),
                "cycle",
                smsp_cycles,
                u128::from(sm_count) * 4,
                true,
                LIFE,
            ),
        );
        m.insert(
            "warp_active_cycles".into(),
            Metric::ratio(
                Some("sm__warps_active.sum"),
                "warp_cycle",
                warp_cycles,
                1,
                true,
                LIFE,
            ),
        );
        m.insert(
            "active_warps_per_sm".into(),
            Metric::ratio(
                Some("sm__warps_active.avg.per_cycle_active"),
                "warp",
                warp_cycles,
                active,
                true,
                LIFE,
            ),
        );
        m.insert(
            "active_warps_per_smsp".into(),
            Metric::ratio(
                Some("smsp__warps_active.avg.per_cycle_active"),
                "warp",
                warp_cycles,
                smsp_cycles,
                true,
                LIFE,
            ),
        );
        m.insert(
            "smsp_warps_active_per_cycle".into(),
            Metric::ratio(
                Some("smsp__warps_active.avg.per_cycle_active"),
                "warp",
                warp_cycles,
                smsp_cycles,
                true,
                LIFE,
            ),
        );
        m.insert(
            "achieved_occupancy_pct".into(),
            Metric::ratio(
                Some("sm__warps_active.avg.pct_of_peak_sustained_active"),
                "%",
                warp_cycles * 100,
                active * u128::from(max_warps),
                true,
                LIFE,
            ),
        );
        m.insert(
            "ipc_per_sm_active".into(),
            Metric::ratio(
                Some("sm__inst_executed.avg.per_cycle_active"),
                "inst/cycle",
                inst,
                active,
                true,
                LIFE,
            ),
        );
        if let (issue_cycles, Some(terminal_cycles)) = (
            result.modeled_issue_lower_bound_cycles(),
            result.sm_active_terminal_event_time(),
        ) {
            let issue_cycles = u128::from(issue_cycles);
            let terminal_cycles = u128::from(terminal_cycles.max(issue_cycles as u64));
            let population = u128::from(active_sm_count.max(1));
            let lower = Metric::ratio(
                None,
                "inst/cycle",
                inst,
                terminal_cycles.saturating_mul(population),
                true,
                "SM-active IPC lower edge using the modeled terminal lower bound; unmodeled tails remain explicit",
            );
            let upper = Metric::ratio(
                None,
                "inst/cycle",
                inst,
                issue_cycles.saturating_mul(population),
                true,
                "SM-active IPC upper edge using the modeled issue lower bound; not an achieved hardware IPC claim",
            );
            m.insert("ipc_per_sm_active_lower_edge".into(), lower);
            m.insert("ipc_per_sm_active_upper_edge".into(), upper);
            m.insert(
            "ipc_per_sm_active_interval_width".into(),
            Metric::ratio(
                None,
                "inst/cycle",
                inst.saturating_mul(terminal_cycles.saturating_sub(issue_cycles)),
                issue_cycles
                    .saturating_mul(terminal_cycles)
                    .saturating_mul(population),
                true,
                "width of the modeled SM-active IPC uncertainty envelope; not a statistical confidence interval",
            ),
        );
        }
        m.insert(
            "ipc_per_smsp_active".into(),
            Metric::ratio(
                Some("smsp__inst_executed.avg.per_cycle_active"),
                "inst/cycle",
                inst,
                smsp_cycles,
                true,
                LIFE,
            ),
        );
        m.insert(
            "issue_active_pct".into(),
            Metric::ratio(
                Some("smsp__issue_active.avg.pct_of_peak_sustained_active"),
                "%",
                issue_cycles * 100,
                smsp_cycles,
                true,
                LIFE,
            ),
        );
        m.insert(
            "peak_live_warps_per_sm_avg".into(),
            Metric::ratio(
                Some("sm__warps_active.max"),
                "warp",
                peak_live_warps_sum,
                u128::from(sm_count),
                true,
                "maximum modeled resident warp population per traced SM; not an achieved average",
            ),
        );
        add_count_row(
            &mut m,
            "peak_live_warps_per_sm_max",
            Some("sm__warps_active.max"),
            "warp",
            u64::from(peak_live_warps_max),
            "maximum modeled resident warp population across traced SMs",
        );
        // Keep the pipeline mix as a flat, queryable surface in addition to the
        // nested trace_metrics object. Zero-valued rows are intentional: they
        // make a workload matrix schema-stable without inventing traffic.
        for (pipeline, activity, ncu_metric) in [
            (
                "dispatch_only",
                trace.by_pipeline.get("dispatch_only"),
                None,
            ),
            (
                "tensor",
                trace.by_pipeline.get("tensor"),
                Some("smsp__inst_executed_pipe_tensor.sum"),
            ),
            ("scalar_alu", trace.by_pipeline.get("scalar_alu"), None),
            ("uniform_sfu", trace.by_pipeline.get("uniform_sfu"), None),
            ("barrier", trace.by_pipeline.get("barrier"), None),
            ("branch", trace.by_pipeline.get("branch"), None),
        ] {
            let activity = activity.cloned().unwrap_or_default();
            add_count_row(
                &mut m,
                format!("pipeline_{pipeline}_warp_instructions"),
                ncu_metric,
                "inst",
                activity.warp_instructions,
                "trace-derived dynamic instruction mix by joined scheduler pipeline",
            );
            add_count_row(
                &mut m,
                format!("pipeline_{pipeline}_active_thread_instructions"),
                None,
                "thread_inst",
                activity.active_thread_instructions,
                "trace-derived active lane population by joined scheduler pipeline",
            );
            add_count_row(
                &mut m,
                format!("pipeline_{pipeline}_predicate_on_thread_instructions"),
                None,
                "thread_inst",
                activity.predicate_on_thread_instructions,
                "trace-derived predicate-on lane population by joined scheduler pipeline",
            );
        }
        for (pipeline, ncu_metric, semantics) in [
            (
                "fma",
                "smsp__inst_executed_pipe_fma.sum",
                "trace opcode projection for NVIDIA's logical FMA pipeline; IMAD/IMUL and FP arithmetic families are included",
            ),
            (
                "alu",
                "smsp__inst_executed_pipe_alu.sum",
                "trace opcode projection for NVIDIA's logical ALU pipeline; integer/logic/address/move families are included",
            ),
            (
                "tensor",
                "smsp__inst_executed_pipe_tensor.sum",
                "trace opcode projection for NVIDIA's Tensor pipeline",
            ),
            (
                "lsu",
                "smsp__inst_executed_pipe_lsu.sum",
                "trace opcode projection for NVIDIA's LSU logical pipeline; includes modeled memory, S2R, SHF, and BAR families",
            ),
        ] {
            let activity = trace
                .by_ncu_pipeline
                .get(pipeline)
                .cloned()
                .unwrap_or_default();
            m.insert(
                format!("ncu_pipeline_{pipeline}_instructions"),
                if matches!(pipeline, "alu" | "lsu") {
                    Metric::approximate_count(
                        Some(ncu_metric),
                        "inst",
                        u128::from(activity.warp_instructions),
                        semantics,
                    )
                } else {
                    Metric::count(
                        Some(ncu_metric),
                        "inst",
                        u128::from(activity.warp_instructions),
                        semantics,
                    )
                },
            );
            add_count_row(
                &mut m,
                format!("ncu_pipeline_{pipeline}_active_thread_instructions"),
                None,
                "thread_inst",
                activity.active_thread_instructions,
                semantics,
            );
        }
        let classified_instructions: u128 = ["fma", "alu", "tensor", "lsu"]
            .iter()
            .filter_map(|pipeline| trace.by_ncu_pipeline.get(*pipeline))
            .map(|activity| u128::from(activity.warp_instructions))
            .sum();
        let unclassified_instructions = trace
            .by_ncu_pipeline
            .get("unclassified")
            .map_or(0, |activity| activity.warp_instructions);
        m.insert(
        "ncu_pipeline_unclassified_instructions".into(),
        Metric::approximate_count(
            None,
            "inst",
            u128::from(unclassified_instructions),
            "trace opcode heads without a unique SM86 logical-pipeline projection; dynamic pipe assignment is unavailable",
        ),
    );
        m.insert(
        "ncu_pipeline_classified_pct".into(),
        Metric::ratio(
            None,
            "%",
            classified_instructions * 100,
            inst,
            true,
            "fraction of trace warp instructions assigned to a unique logical FMA/ALU/LSU/Tensor projection",
        ),
    );
        let physical = &trace.physical_pipeline;
        m.insert(
            "physical_pipeline_unique_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(physical.unique_instructions) * 100,
                u128::from(physical.total_warp_instructions),
                true,
                "trace opcode/domain instructions with one conservative physical-pipe candidate",
            ),
        );
        m.insert(
            "physical_pipeline_ambiguous_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(physical.ambiguous_instructions) * 100,
                u128::from(physical.total_warp_instructions),
                true,
                "trace instructions with multiple conservative physical-pipe candidates",
            ),
        );
        m.insert(
            "physical_pipeline_unknown_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(physical.unknown_instructions) * 100,
                u128::from(physical.total_warp_instructions),
                true,
                "trace instructions without a conservative physical-pipe candidate domain",
            ),
        );
        for (pipe, count) in &physical.candidate_instructions {
            m.insert(
            format!("physical_pipeline_candidate_{pipe}_instructions"),
            Metric::approximate_count(
                None,
                "inst",
                u128::from(*count),
                "conservative candidate membership upper bound; candidates may overlap and are not NCU pipe counters",
            ),
        );
        }
        for (pipe, count) in &physical.selected_instructions {
            m.insert(
            format!("physical_pipeline_selected_{pipe}_instructions"),
            Metric::approximate_count(
                None,
                "inst",
                u128::from(*count),
                "deterministic trace-order pressure-balancing selection from the conservative candidate domain; not an NCU physical-pipe counter",
            ),
        );
        }
        m.insert(
            "physical_pipeline_selected_unknown_instructions".into(),
            Metric::approximate_count(
                None,
                "inst",
                u128::from(physical.selected_unknown_instructions),
                "trace records with no physical-pipe candidate; no selection is fabricated",
            ),
        );
        let selected = physical
            .selected_instructions
            .values()
            .copied()
            .map(u128::from)
            .sum::<u128>();
        m.insert(
        "physical_pipeline_selected_assignment_pct".into(),
        Metric::ratio(
            None,
            "%",
            selected * 100,
            u128::from(physical.total_warp_instructions),
            true,
            "share of trace records assigned by the deterministic candidate-selection policy; hardware dynamic selection remains unavailable",
        ),
    );
        for (opcode, activity) in &trace.by_opcode {
            let key = opcode.to_ascii_lowercase();
            add_count_row(
                &mut m,
                format!("opcode_{key}_warp_instructions"),
                None,
                "inst",
                activity.warp_instructions,
                "trace-derived dynamic opcode-head population",
            );
            add_count_row(
                &mut m,
                format!("opcode_{key}_active_thread_instructions"),
                None,
                "thread_inst",
                activity.active_thread_instructions,
                "trace-derived active lane population by opcode head",
            );
            add_count_row(
                &mut m,
                format!("opcode_{key}_predicate_on_thread_instructions"),
                None,
                "thread_inst",
                activity.predicate_on_thread_instructions,
                "trace-derived predicate-on lane population by opcode head",
            );
        }

        Ok(())
    }
}
