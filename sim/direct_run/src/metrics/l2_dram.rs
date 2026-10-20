//! L2/DRAM and readiness attribution helpers. Report-only.

use super::*;

pub(crate) fn add_cause_rows(
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

pub(crate) fn add_global_issue_rows(
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

/// Project the simulator's exclusive causal buckets into the vocabulary used
/// by NCU's warp-sampling section. These are intentionally *proxy* rows: NCU
/// samples a randomly selected active warp at hardware-defined intervals and
/// can overlap states, while this model assigns one limiting cause to each
/// delayed issue. Keeping the rows separate gives consumers a complete schema
/// without pretending that sampled hardware counters were reconstructed.
pub(crate) fn add_scheduler_proxy_rows(
    metrics: &mut BTreeMap<String, Metric>,
    readiness: &ReadinessDiagnostics,
) {
    let cause = &readiness.limiting_cycles;
    let imc_miss = readiness
        .immediate_constant_cache
        .as_deref()
        .map_or(0, |cache| cache.miss_wait_cycles);
    let rows = [
        (
            "barrier",
            cause
                .cta_barrier_rendezvous
                .saturating_add(cause.pipeline_barrier),
            "CTA rendezvous plus barrier-pipeline limiting cycles; not sampled warp state",
        ),
        (
            "branch_resolving",
            cause.pipeline_branch,
            "modeled branch-resolution limiting cycles; NCU samples PC-update waits",
        ),
        (
            "dispatch_stall",
            cause.instruction_front_end,
            "modeled instruction-front-end limiting cycles; dispatch conflicts are not separately identified",
        ),
        (
            "drain",
            cause.memory_write_completion,
            "modeled write-completion limiting cycles; physical EXIT drain is not modeled",
        ),
        (
            "imc_miss",
            imc_miss,
            "modeled immediate-constant miss wait cycles",
        ),
        (
            "lg_throttle",
            cause
                .memory_admission_queue
                .saturating_add(cause.memory_issue_queue),
            "modeled memory queue limiting cycles; not an NCU LG-throttle sample",
        ),
        (
            "long_scoreboard",
            cause
                .dependence_counter_l1tex
                .saturating_add(cause.dependence_counter_global_load_operand_read)
                .saturating_add(cause.dependence_counter_global_store_operand_read)
                .saturating_add(cause.dependence_counter_global_atomic_operand_read)
                .saturating_add(cause.dependence_counter_global_reduction_operand_read),
            "modeled L1TEX/global operand dependency cycles; NCU long-scoreboard samples overlap",
        ),
        (
            "math_pipe_throttle",
            cause
                .pipeline_alu
                .saturating_add(cause.pipeline_sfu)
                .saturating_add(cause.pipeline_tensor),
            "modeled compute-pipeline limiting cycles; not a sampled math-throttle state",
        ),
        ("membar", 0, "no independent modeled membar-sample state"),
        (
            "mio_throttle",
            cause.dependence_counter_shared_write,
            "modeled shared-write dependency cycles; MIO sampling includes additional causes",
        ),
        (
            "misc",
            cause.dependence_counter_unknown_origin,
            "modeled unknown-origin dependency cycles",
        ),
        (
            "no_instructions",
            0,
            "no independent sampled no-instructions state",
        ),
        (
            "not_selected",
            cause.eligible_not_selected,
            "modeled eligible-but-not-selected cycles; NCU state samples are not identical",
        ),
        (
            "selected",
            0,
            "scheduler selection is not reconstructed as a sampled warp state",
        ),
        ("sleeping", 0, "no independent sampled sleeping state"),
        (
            "tex_throttle",
            cause.dependence_counter_l1tex,
            "modeled L1TEX dependency cycles; TEX-throttle samples are distinct",
        ),
        (
            "wait",
            cause.control_stall,
            "modeled control-stall cycles; NCU wait samples are broader",
        ),
    ];
    for (suffix, value, semantics) in rows {
        metrics.insert(
            format!("scheduler_proxy_{suffix}_cycles"),
            Metric::approximate_count(
                Some(&format!("smsp__warps_issue_stalled_{suffix}.sum")),
                "cycle",
                u128::from(value),
                semantics,
            ),
        );
    }
}

pub(crate) fn add_sm86_register_read_rows(
    metrics: &mut BTreeMap<String, Metric>,
    readiness: &ReadinessDiagnostics,
) {
    let Some(rf) = readiness.sm86_register_read.as_deref() else {
        return;
    };
    for (name, value, unit, semantics) in [
        (
            "reservations",
            rf.reservations,
            "inst",
            "SM86 conditional full-mode register-read reservations; not a physical RF event count",
        ),
        (
            "source_reads_bank0",
            rf.source_reads[0],
            "read",
            "projected regular-register source reads assigned to effective bank 0",
        ),
        (
            "source_reads_bank1",
            rf.source_reads[1],
            "read",
            "projected regular-register source reads assigned to effective bank 1",
        ),
        (
            "applied_wait_cycles",
            rf.applied_wait_cycles,
            "cycle",
            "SM86 conditional register-read readiness cycles actually charged before issue",
        ),
        (
            "port_conflict_cycles",
            rf.port_conflict_cycles,
            "cycle",
            "conditional one-port-per-bank reservation conflict cycles; not NCU stall cycles",
        ),
        (
            "rfc_hits",
            rf.rfc_hits,
            "hit",
            "compiler-keyed reuse-cache hits in the bounded effective model",
        ),
        (
            "rfc_misses",
            rf.rfc_misses,
            "miss",
            "compiler-keyed reuse-cache misses in the bounded effective model",
        ),
        (
            "unsupported_shapes",
            rf.unsupported_shapes,
            "inst",
            "projected fixed-latency records whose vector/multi-cell shape was not modeled",
        ),
    ] {
        add_count_row(
            metrics,
            format!("sm86_register_read_{name}"),
            None,
            unit,
            value,
            semantics,
        );
    }
}
