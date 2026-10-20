// L2/DRAM and branch/stall attribution helpers; report-only post-replay projections.

fn ensure_unavailable(
    metrics: &mut BTreeMap<String, Metric>,
    key: &'static str,
    ncu_metric: Option<&str>,
    unit: &'static str,
    reason: &'static str,
) {
    metrics
        .entry(key.to_owned())
        .or_insert_with(|| Metric::unavailable(ncu_metric, unit, reason));
}

fn add_branch_rows(metrics: &mut BTreeMap<String, Metric>, trace: &sim_sm::TraceMetrics) {
    // NCU's SM86 metric description: CBU instructions except BMOV/BSSY.
    // These are the CBU heads covered by this projection. Keep it separate
    // from the older BRA/JMP-only count and from target-bearing events.
    let instructions: u128 = [
        "BRA", "BRX", "BRXU", "JMP", "JMX", "JMXU", "CALL", "RET", "EXIT", "BSYNC", "BREAK",
    ]
    .iter()
    .filter_map(|head| trace.by_opcode.get(*head))
    .map(|activity| u128::from(activity.warp_instructions))
    .sum();
    metrics.insert("branch_control_instructions".into(), Metric::approximate_count(
        Some("smsp__inst_executed_op_branch.sum"), "inst", instructions,
        "SM86 CBU opcode projection including calls, returns, EXIT, BSYNC and BREAK; excludes BMOV/BSSY; not the BRA/JMP-only count or a target-event denominator",
    ));
    let targets = &trace.branch_targets;
    for (key, ncu, unit, count, meaning) in [
        (
            "branch_target_divergent_branches",
            Some("smsp__branch_targets_threads_divergent"),
            "branches",
            targets.divergent_events,
            "BRA/JMP/CALL/RET events with distinct first subsequent (function, PC) identities across active lanes, including fallthrough; not predicate disagreement",
        ),
        (
            "branch_target_uniform_branches",
            Some("smsp__sass_branch_targets_threads_uniform.sum"),
            "branches",
            targets.uniform_events,
            "target-bearing events whose active lanes all have the same first subsequent (function, PC); includes all-predicated-off fallthrough",
        ),
        (
            "branch_target_divergent_active_threads",
            None,
            "thread",
            targets.divergent_threads,
            "all active threads participating in resolved divergent target events; not an NCU branch count or minority-path size",
        ),
    ] {
        metrics.insert(key.into(), if targets.unresolved_events == 0 {
            Metric::approximate_count(ncu, unit, u128::from(count), meaning)
        } else {
            Metric::unavailable(ncu, unit, "some branch lanes have no successor in this trace; resolved subset retained in trace_metrics.branch_targets")
        });
    }
    for (key, unit, count, meaning) in [
        (
            "branch_target_successor_active_threads",
            "thread",
            targets.successor_active_threads,
            "sum of active lanes assigned a first successor across resolved target events",
        ),
        (
            "branch_target_divergent_minority_threads",
            "thread",
            targets.divergent_minority_threads,
            "sum of the smallest successor-path population across resolved divergent events",
        ),
        (
            "branch_target_uniform_active_threads",
            "thread",
            targets.uniform_active_threads,
            "active lanes in resolved uniform successor events",
        ),
    ] {
        metrics.insert(
            key.into(),
            if targets.unresolved_events == 0 {
                Metric::approximate_count(None, unit, u128::from(count), meaning)
            } else {
                Metric::unavailable(
                    None,
                    unit,
                    "some branch lanes have no successor in this trace; resolved subset remains in trace_metrics.branch_targets",
                )
            },
        );
    }
    metrics.insert(
        "branch_target_unresolved_events".into(),
        Metric::count(
            None,
            "branches",
            u128::from(targets.unresolved_events),
            "branch events with unobserved successors at warp trace end",
        ),
    );
    metrics.insert(
        "branch_target_unresolved_threads".into(),
        Metric::count(
            None,
            "thread",
            u128::from(targets.unresolved_threads),
            "active lanes whose first successor was not observed before trace end",
        ),
    );
    metrics.insert("branch_target_uniform_pct".into(), if targets.unresolved_events == 0 {
        Metric::ratio(Some("smsp__sass_average_branch_targets_threads_uniform.pct"), "%",
            u128::from(targets.uniform_events) * 100, u128::from(targets.events), true,
            "uniform successor events / all target-bearing events; does not use the wider CBU-instruction denominator")
    } else {
        Metric::unavailable(Some("smsp__sass_average_branch_targets_threads_uniform.pct"), "%", "branch successor coverage is incomplete")
    });
}

fn add_cause_rows(
    metrics: &mut BTreeMap<String, Metric>,
    prefix: &str,
    causes: &ReadinessCauseCycles,
    semantics: &'static str,
) {
    let rows = [
        ("instruction_front_end", causes.instruction_front_end),
        ("control", causes.control_stall),
        ("dependence_read", causes.dependence_counter_read),
        (
            "dependence_global_load_operand_read",
            causes.dependence_counter_global_load_operand_read,
        ),
        (
            "dependence_global_store_operand_read",
            causes.dependence_counter_global_store_operand_read,
        ),
        (
            "dependence_global_atomic_operand_read",
            causes.dependence_counter_global_atomic_operand_read,
        ),
        (
            "dependence_global_reduction_operand_read",
            causes.dependence_counter_global_reduction_operand_read,
        ),
        (
            "dependence_non_memory_write",
            causes.dependence_counter_non_memory_write,
        ),
        (
            "dependence_shared_write",
            causes.dependence_counter_shared_write,
        ),
        ("dependence_l1tex", causes.dependence_counter_l1tex),
        (
            "dependence_unknown_origin",
            causes.dependence_counter_unknown_origin,
        ),
        ("async_source", causes.async_source),
        ("async_destination_local", causes.async_destination_local),
        (
            "async_destination_sm_service",
            causes.async_destination_sm_service,
        ),
        ("cta_barrier_rendezvous", causes.cta_barrier_rendezvous),
        ("pipeline_barrier", causes.pipeline_barrier),
        ("pipeline_branch", causes.pipeline_branch),
        ("pipeline_tensor", causes.pipeline_tensor),
        ("pipeline_sfu", causes.pipeline_sfu),
        ("pipeline_alu", causes.pipeline_alu),
        ("memory_admission_queue", causes.memory_admission_queue),
        ("memory_issue_queue", causes.memory_issue_queue),
        ("memory_write_completion", causes.memory_write_completion),
        ("eligible_not_selected", causes.eligible_not_selected),
    ];
    for (name, value) in rows {
        add_count_row(
            metrics,
            format!("{prefix}_{name}_cycles"),
            None,
            "cycle",
            value,
            semantics,
        );
    }
}

fn add_global_issue_rows(
    metrics: &mut BTreeMap<String, Metric>,
    issue: &GlobalMemoryIssueDiagnostics,
) {
    let rows = [
        ("load_instructions", issue.load_instructions),
        ("store_instructions", issue.store_instructions),
        ("atomic_instructions", issue.atomic_instructions),
        ("reduction_instructions", issue.reduction_instructions),
        (
            "load_operand_read_barrier_instructions",
            issue.load_operand_read_barrier_instructions,
        ),
        (
            "store_operand_read_barrier_instructions",
            issue.store_operand_read_barrier_instructions,
        ),
        (
            "atomic_operand_read_barrier_instructions",
            issue.atomic_operand_read_barrier_instructions,
        ),
        (
            "reduction_operand_read_barrier_instructions",
            issue.reduction_operand_read_barrier_instructions,
        ),
        ("load_sectors", issue.load_sectors),
        ("store_sectors", issue.store_sectors),
        ("atomic_sectors", issue.atomic_sectors),
        ("reduction_sectors", issue.reduction_sectors),
        ("load_unique_lines", issue.load_unique_lines),
        ("store_unique_lines", issue.store_unique_lines),
        ("atomic_unique_lines", issue.atomic_unique_lines),
        ("reduction_unique_lines", issue.reduction_unique_lines),
        ("atomic_l2_requests", issue.atomic_l2_requests),
        ("reduction_l2_requests", issue.reduction_l2_requests),
        ("load_ideal_wavefronts", issue.load_ideal_wavefronts),
        ("store_ideal_wavefronts", issue.store_ideal_wavefronts),
        ("atomic_ideal_wavefronts", issue.atomic_ideal_wavefronts),
        (
            "reduction_ideal_wavefronts",
            issue.reduction_ideal_wavefronts,
        ),
        ("total_issue_span_cycles", issue.total_issue_span_cycles),
        ("max_issue_span_cycles", issue.max_issue_span_cycles),
    ];
    for (name, value) in rows {
        add_count_row(
            metrics,
            format!("global_issue_{name}"),
            None,
            if name.ends_with("cycles") {
                "cycle"
            } else if name.ends_with("instructions") {
                "inst"
            } else if name.ends_with("requests") {
                "request"
            } else if name.ends_with("wavefronts") {
                "wavefront"
            } else if name.ends_with("lines") {
                "line"
            } else {
                "sector"
            },
            value,
            "trace-derived global-memory issue topology; ideal work is not a hardware replay counter",
        );
    }
}

