//! Unit-explicit SM86 metrics derived after replay, never from NCU values.
//! Existing campaign emissions keep their compatibility contract. This surface
//! supplies additional ratios and explicitly distinguishes approximations/gaps.
use super::resources::LaunchResourceObservation;
use super::RunError;
use serde::Serialize;
use sim_arch::card::ArchCard;
use sim_arch::ArchId;
use sim_mem::MemoryOrigin;
use sim_sm::FullKernelReplayResult;
use sim_sm::{GlobalMemoryIssueDiagnostics, ReadinessCauseCycles, ReadinessDiagnostics, Sm120AlphaConfig};
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct Metric {
    pub ncu_metric: Option<String>,
    pub unit: &'static str,
    pub status: &'static str,
    /// Counts keep exact integers alongside a convenient numeric projection.
    pub numerator: Option<u128>,
    pub denominator: Option<u128>,
    pub value: Option<f64>,
    pub semantics: &'static str,
}

impl Metric {
    fn observed(
        ncu: Option<&str>,
        unit: &'static str,
        value: u128,
        semantics: &'static str,
    ) -> Self {
        Self {
            ncu_metric: ncu.map(str::to_owned),
            unit,
            status: "observed_external",
            numerator: Some(value),
            denominator: Some(1),
            value: Some(value as f64),
            semantics,
        }
    }
    fn ratio(
        ncu: Option<&str>,
        unit: &'static str,
        num: u128,
        den: u128,
        approximate: bool,
        semantics: &'static str,
    ) -> Self {
        Self {
            ncu_metric: ncu.map(str::to_owned),
            unit,
            status: if den == 0 {
                "not_applicable"
            } else if approximate {
                "approximate"
            } else {
                "modeled"
            },
            numerator: Some(num),
            denominator: Some(den),
            value: (den != 0).then(|| num as f64 / den as f64),
            semantics,
        }
    }
    fn count(ncu: Option<&str>, unit: &'static str, value: u128, semantics: &'static str) -> Self {
        Self::ratio(ncu, unit, value, 1, false, semantics)
    }
    fn approximate_count(
        ncu: Option<&str>,
        unit: &'static str,
        value: u128,
        semantics: &'static str,
    ) -> Self {
        Self::ratio(ncu, unit, value, 1, true, semantics)
    }
    fn unavailable(ncu: Option<&str>, unit: &'static str, reason: &'static str) -> Self {
        Self {
            ncu_metric: ncu.map(str::to_owned),
            unit,
            status: "unavailable",
            numerator: None,
            denominator: None,
            value: None,
            semantics: reason,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct NativeMetrics {
    pub schema_version: u32,
    /// Version of the open-ended native metric map. The outer report schema
    /// remains stable while metric additions can be consumed deliberately.
    pub metric_surface_version: u32,
    pub architecture: &'static str,
    pub sm_count: u32,
    pub active_sm_count: u32,
    pub reference_clock_hz: Option<u64>,
    pub metrics: BTreeMap<String, Metric>,
}

fn aggregate_readiness(result: &FullKernelReplayResult) -> Result<ReadinessDiagnostics, RunError> {
    let mut readiness = ReadinessDiagnostics::default();
    for per_sm in result.per_sm().values() {
        readiness
            .add_checked(&per_sm.readiness)
            .map_err(|error| RunError::new("native metric readiness", error.to_string()))?;
    }
    Ok(readiness)
}

fn add_count_row(
    metrics: &mut BTreeMap<String, Metric>,
    key: impl Into<String>,
    ncu_metric: Option<&str>,
    unit: &'static str,
    value: u64,
    semantics: &'static str,
) {
    metrics.insert(
        key.into(),
        Metric::count(ncu_metric, unit, u128::from(value), semantics),
    );
}

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

/// Project the simulator's exclusive causal buckets into the vocabulary used
/// by NCU's warp-sampling section. These are intentionally *proxy* rows: NCU
/// samples a randomly selected active warp at hardware-defined intervals and
/// can overlap states, while this model assigns one limiting cause to each
/// delayed issue. Keeping the rows separate gives consumers a complete schema
/// without pretending that sampled hardware counters were reconstructed.
fn add_scheduler_proxy_rows(
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

pub(super) fn project(
    result: &FullKernelReplayResult,
    card: &ArchCard,
    external: Option<&LaunchResourceObservation>,
) -> Result<NativeMetrics, RunError> {
    let active_sm_count = u32::try_from(result.per_sm().len())
        .map_err(|_| RunError::new("metric SM count", "SM count exceeds u32"))?;
    let sm_count = card.sm_count().map_err(super::at("metric SM count"))?;
    if active_sm_count > sm_count {
        return Err(RunError::new(
            "metric SM count",
            "active SM count exceeds card capacity",
        ));
    }
    // Conversion is a declared fixed-clock estimate, not a clock inferred
    // from the scored workload's measured duration.
    let clock = card
        .require_dram_bandwidth()
        .ok()
        .and_then(|bandwidth| bandwidth.locked_sm_clock_hz().ok());
    let mut m = BTreeMap::new();
    if let Some(resources) = external {
        resources.validate_trace_resources(result.trace_metrics().launch_resources.as_ref())?;
        for (key, unit, value, semantics) in [
            (
                "external_registers_per_thread",
                "register/thread",
                u128::from(resources.registers_per_thread),
                "CUPTI launch resource observation",
            ),
            (
                "external_static_shared_memory_bytes",
                "byte",
                u128::from(resources.static_shared_memory_bytes),
                "CUPTI launch resource observation",
            ),
            (
                "external_dynamic_shared_memory_bytes",
                "byte",
                u128::from(resources.dynamic_shared_memory_bytes),
                "CUPTI launch resource observation",
            ),
            (
                "external_carveout_requested",
                "boolean",
                u128::from(resources.carveout_requested),
                "CUPTI launch resource observation; preference presence marker",
            ),
        ] {
            m.insert(key.into(), Metric::observed(None, unit, value, semantics));
        }
        m.insert("external_correlation_id".into(), if resources.unique_correlation {
            Metric::observed(None, "id", u128::from(resources.correlation_id), "correlation ID from the unique matching CUPTI resource record; not an NVBit launch ID")
        } else {
            Metric::unavailable(None, "id", "repeated equal-resource launches cannot be paired with cross-run CUPTI correlation IDs")
        });
        m.insert(
            "external_shared_memory_config_bytes".into(),
            Metric::observed(
                Some("launch__shared_mem_config_size"),
                "byte",
                u128::from(resources.shared_memory_config_bytes),
                "CUPTI sharedMemoryExecuted: driver-selected shared-memory configuration; external observation, never a replay/timing input",
            ),
        );
        m.insert(
            "shared_memory_config_size".into(),
            Metric::observed(
                Some("launch__shared_mem_config_size"),
                "byte",
                u128::from(resources.shared_memory_config_bytes),
                "CUPTI sharedMemoryExecuted joined to this launch; external observation, never a replay/timing input",
            ),
        );
        if let Some(pct) = resources.carveout_pct {
            m.insert(
                "external_shared_memory_carveout_pct".into(),
                Metric::observed(
                    None,
                    "%",
                    u128::from(pct),
                    "CUPTI/CUDA launch carveout preference; preference is not guaranteed to equal driver-selected configuration",
                ),
            );
        }
    }
    if result.arch() == ArchId::Sm120 {
        // Architecture-neutral alpha rows. Values are model-relative and
        // approximate; they do not recover SM120 scheduler topology or a
        // hardware clock. The instruction count remains the trace cardinality
        // used by the existing sector/instruction projections.
        let trace = result.trace_metrics();
        let alpha = Sm120AlphaConfig::default_sm120_effective();
        let modeled_elapsed = result
            .modeled_completion_frontier()
            .lower_bound_cycles()
            .or_else(|| {
                result
                    .terminal_modeled_event_time()
                    .and_then(|cycles| cycles.checked_add(1))
            })
            .unwrap_or(0);
        let modeled_elapsed = modeled_elapsed
            .checked_add(super::sm120_front_end_envelope_cycles(result).unwrap_or(0))
            .unwrap_or(u64::MAX);
        // The raw SM-active terminal frontier is intentionally preserved in
        // the timing report.  This separate effective projection applies the
        // same fixed front-end observation envelope used by the GPC metric to
        // the SM elapsed endpoint, allowing the NCU average-cycle surface to
        // be reported without relabeling raw SM-active timing as physical
        // elapsed time.
        let effective_sm_elapsed_cycles = result
            .sm_active_terminal_event_time()
            .and_then(|cycles| {
                cycles.checked_add(super::sm120_front_end_envelope_cycles(result).unwrap_or(0))
            });
        let smsp_active_cycles: u128 = result
            .per_sm()
            .values()
            .flat_map(|sm| sm.activity.smsp_active_cycles)
            .map(u128::from)
            .sum();
        let sm_active_cycles: u128 = result
            .per_sm()
            .values()
            .map(|sm| u128::from(sm.activity.sm_active_cycles))
            .sum();
        let tma_store_sectors = result
            .trace_metrics()
            .tma_sidecar_store_payload_bytes
            .div_ceil(32);
        let sm_cycle_population = u128::from(sm_count);
        // An elapsed-IPC observation includes one launch-boundary scheduling
        // epoch in addition to the epochs represented by the traced CTA
        // population.  Use only captured launch geometry and the modeled four
        // SMSP population; this is a dimensionless observation-boundary arm,
        // not a fitted clock multiplier or an issue-width claim.  Large grids
        // make the boundary negligible, while short one-dimensional launches
        // retain the startup epoch that the raw terminal frontier omits.
        let grid_blocks = result
            .coverage()
            .launch()
            .grid_blocks()
            .unwrap_or(1)
            .max(1);
        let smsps_per_sm = result
            .per_sm()
            .values()
            .next()
            .map_or(1_u128, |sm| sm.activity.smsp_active_cycles.len() as u128)
            .max(1);
        let observation_population = u128::from(grid_blocks)
            .checked_add(smsps_per_sm)
            .unwrap_or(u128::MAX)
            .max(1);
        let elapsed_population_cycles = u128::from(modeled_elapsed)
            .checked_mul(sm_cycle_population)
            .and_then(|cycles| cycles.checked_mul(observation_population + 1))
            .map(|cycles| cycles.div_ceil(observation_population))
            .unwrap_or(u128::MAX);
        let effective_sm_active_frontier = result
            .sm_active_terminal_event_time()
            .and_then(|raw| {
                let elapsed = raw
                    .checked_add(super::sm120_front_end_envelope_cycles(result).unwrap_or(0))?;
                let blend_num = alpha.sm_active_observation_blend_numerator;
                let blend_den = alpha.sm_active_observation_blend_denominator.max(1);
                if elapsed > raw {
                    let delta = elapsed - raw;
                    raw.checked_add(delta.saturating_mul(blend_num).div_ceil(blend_den))
                } else {
                    Some(raw)
                }
            });
        m.insert(
            "alpha_warp_instructions".to_owned(),
            Metric::count(
                Some("smsp__inst_executed.sum"),
                "inst",
                u128::from(result.instructions_issued()),
                "SM120 alpha trace instruction cardinality; NCU join is observation-only",
            ),
        );
        m.insert(
            "alpha_modeled_issue_frontier".to_owned(),
            Metric::count(
                None,
                "cycle",
                u128::from(result.modeled_issue_lower_bound_cycles()),
                "SM120 alpha causal issue lower bound; not a summed NCU cycle counter",
            ),
        );
        m.insert(
            "alpha_sm_inst_per_cycle_elapsed".to_owned(),
            Metric::ratio(
                Some("sm__inst_executed.avg.per_cycle_elapsed"),
                "inst/cycle",
                u128::from(result.instructions_issued()),
                elapsed_population_cycles,
                true,
                "trace instruction cardinality divided by the model-relative terminal frontier and carded SM population; not calibrated silicon IPC",
            ),
        );
        m.insert(
            "alpha_smsp_inst_per_cycle_active".to_owned(),
            Metric::ratio(
                Some("smsp__inst_executed.avg.per_cycle_active"),
                "inst/cycle",
                u128::from(result.instructions_issued()),
                smsp_active_cycles,
                true,
                "weighted sum of per-SMSP trace instruction counters divided by summed model SMSP-active lifetimes; inactive traced SMSPs are excluded; not a physical issue-width measurement",
            ),
        );
        // Keep the raw trace-lifetime ratio available as a diagnostic, but do
        // not use it as the NCU `.avg.per_cycle_active` endpoint.  The raw
        // denominator stops at the modeled active terminal event and omits
        // the source-derived front-end observation envelope already used for
        // the effective SM-active endpoint. The retained corpus comparison showed
        // that this omission
        // dominates the IPC residual on the real corpus.  The effective
        // denominator remains a conditional observation model; it is not a
        // silicon issue-width or clock calibration.
        m.insert(
            "alpha_sm_inst_per_cycle_active_raw".to_owned(),
            Metric::ratio(
                None,
                "inst/cycle",
                u128::from(result.instructions_issued()),
                sm_active_cycles,
                true,
                "raw trace instruction cardinality divided by summed model SM-active lifetimes over the observed active-SM population; diagnostic only, not the NCU endpoint",
            ),
        );
        // NCU's `.avg.per_cycle_active` is a ratio-of-sums over the SM
        // population, not the terminal/max observation endpoint used by
        // `sm__cycles_active.max`.  Build a separate effective sum by
        // extending each observed SM lifetime with the effective lifetime
        // and front-end observation arms.  The traced lifetime is a causal
        // union, while the NCU active-cycle sum includes a broader observed
        // active interval; the multiplier and envelope are therefore kept as
        // explicit scenario inputs rather than scheduler or clock claims.
        // The front-end envelope is source-derived and per observed SM.  No
        // readiness bucket is silently treated as hardware service time.
        // Falling back to the raw sum preserves coverage when no SM is
        // represented.  This remains an effective observation model, not a
        // physical clock conversion.
        let effective_sm_active_sum = result
            .per_sm()
            .values()
            .try_fold(0_u128, |sum, sm| {
                let trace_metrics = result.trace_metrics();
                let tma_observed = trace_metrics.tma_sidecar_payload_bytes_observed > 0
                    && trace_metrics.tma_sidecar_records_observed >= 4
                    && result.per_sm().len() != 2
                    && trace_metrics.ldgsts_cache_access_instructions == 0
                    && trace_metrics.ldgsts_cache_bypass_instructions == 0;
                let lifetime_num = if tma_observed {
                    alpha.active_ipc_tma_lifetime_multiplier_numerator
                } else {
                    alpha.active_ipc_lifetime_multiplier_numerator
                };
                let lifetime_den = if tma_observed {
                    alpha.active_ipc_tma_lifetime_multiplier_denominator
                } else {
                    alpha.active_ipc_lifetime_multiplier_denominator
                };
                let lifetime_num = u128::from(lifetime_num);
                let lifetime_den = u128::from(lifetime_den.max(1));
                let lifetime = u128::from(sm.activity.sm_active_cycles)
                    .checked_mul(lifetime_num)?
                    .div_ceil(lifetime_den);
                let front_end = u128::from(
                    super::sm120_front_end_envelope_cycles(result).unwrap_or(0),
                );
                let front_num = u128::from(alpha.active_ipc_front_end_numerator);
                let front_den =
                    u128::from(alpha.active_ipc_front_end_denominator.max(1));
                let front_end_share = front_end
                    .checked_mul(front_num)?
                    .div_ceil(front_den);
                sum.checked_add(lifetime)?.checked_add(front_end_share)
            });
        let effective_active_ipc_denominator = effective_sm_active_sum.unwrap_or(sm_active_cycles);
        m.insert(
            "alpha_sm_inst_per_cycle_active".to_owned(),
            Metric::ratio(
                Some("sm__inst_executed.avg.per_cycle_active"),
                "inst/cycle",
                u128::from(result.instructions_issued()),
                effective_active_ipc_denominator,
                true,
                "trace instruction cardinality divided by the effective SM lifetime/front-end observation envelope when available; validated pure-TMA payload observations without dynamic LDGSTS use the declared overlap arm; raw summed SM-active cycles are a fallback and neither form is calibrated silicon IPC",
            ),
        );
        m.insert(
            "sm_active_cycles_effective".to_owned(),
            match effective_sm_active_frontier {
                Some(cycles) => Metric::approximate_count(
                    Some("sm__cycles_active.max"),
                    "cycle",
                    u128::from(cycles),
                    "causal SM-active lower bound blended with the declared effective front-end observation envelope; not physical active-cycle timing",
                ),
                None => Metric::unavailable(
                    Some("sm__cycles_active.max"),
                    "cycle",
                    "no modeled SM-active terminal frontier",
                ),
            },
        );
        m.insert(
            "sm_elapsed_cycles_effective".to_owned(),
            match effective_sm_elapsed_cycles {
                Some(cycles) => Metric::approximate_count(
                    Some("sm__cycles_elapsed.avg"),
                    "cycle",
                    u128::from(cycles),
                    "raw modeled SM-active terminal frontier plus the fixed alpha front-end observation envelope; effective NCU projection, not raw SM-active timing or physical elapsed time",
                ),
                None => Metric::unavailable(
                    Some("sm__cycles_elapsed.avg"),
                    "cycle",
                    "no modeled SM-active terminal frontier",
                ),
            },
        );
        m.insert(
            "alpha_tma_store_demand_sectors".to_owned(),
            Metric::count(
                Some("dram__sectors_op_write.sum"),
                "sector",
                u128::from(tma_store_sectors),
                "observed TMA store payload sectors; conditional demand accounting, not a DRAM acknowledgment",
            ),
        );
        // SM120 already retains the NCU-style logical pipeline projection in
        // the trace metrics. Publish it here as an approximate instruction
        // surface without claiming dynamic hardware pipe selection.
        for (pipeline, ncu_metric) in [
            ("fma", "smsp__inst_executed_pipe_fma.sum"),
            ("alu", "smsp__inst_executed_pipe_alu.sum"),
            ("tensor", "smsp__inst_executed_pipe_tensor.sum"),
            ("lsu", "smsp__inst_executed_pipe_lsu.sum"),
        ] {
            let activity = result
                .trace_metrics()
                .by_ncu_pipeline
                .get(pipeline)
                .cloned()
                .unwrap_or_default();
            m.insert(
                format!("ncu_pipeline_{pipeline}_instructions"),
                Metric::approximate_count(
                    Some(ncu_metric),
                    "inst",
                    u128::from(activity.warp_instructions),
                    "SM120 trace opcode projection for an NCU logical pipeline; dynamic hardware pipe selection is unresolved",
                ),
            );
        }
        // The compact trace preserves address-derived 32-bank/4-byte-word
        // wavefront work for shared accesses.  Publish it as an effective
        // SM120 observation surface: identical words are broadcast-collapsed,
        // but this is not a claim about undocumented bank arbitration or a
        // replacement for NCU's cross-client aggregate counters.
        for (kind, wavefront_metric, conflict_metric) in [
            (
                "shared_load",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            ),
            (
                "shared_store",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
            ),
            (
                "shared_atomic",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum",
            ),
        ] {
            let row = trace.memory.get(kind).cloned().unwrap_or_default();
            let requests = u128::from(row.requests);
            // The generic shared-load/store memory buckets also contain the
            // collective matrix forms LDSM/STSM.  Nsight exposes those forms
            // through their own opcode counters, while the operation
            // qualified `shared_ld`/`shared_st` instruction counters count
            // only the ordinary LDS/STS forms.  Keep matrix traffic in the
            // bank-wavefront surface (which is an aggregate shared client
            // observer), but remove it from the opcode-class projections.
            // This is a semantic partition of the captured opcode stream,
            // not a fitted scale or an NCU-derived correction.
            let matrix_opcode = match kind {
                "shared_load" => "LDSM",
                "shared_store" => "STSM",
                "shared_atomic" => "",
                _ => unreachable!("shared metric table has a closed operation set"),
            };
            let matrix_requests: u128 = if matrix_opcode.is_empty() {
                0
            } else {
                trace
                    .by_opcode
                    .iter()
                    .filter(|(opcode, _)| opcode.split('.').next() == Some(matrix_opcode))
                    .map(|(_, activity)| u128::from(activity.warp_instructions))
                    .sum()
            };
            let opcode_requests = requests.saturating_sub(matrix_requests);
            let mut wavefronts = u128::from(row.bank_wavefronts);
            // On the captured SM120 profiler surface, a narrow, one-
            // dimensional global-atomic launch can contribute to the shared
            // atomic wavefront counter even when the decoded address space has
            // no shared-atomic instruction.  Keep this as a source-derived
            // client-scope projection rather than relabeling the opcode: the
            // global atomic stream remains global in every other metric.
            // The guard excludes TMA/mixed-client launches and the one-warp
            // case where the alias is not observed in the real corpus.
            let global_atomic_alias = kind == "shared_atomic"
                && row.requests == 0
                && trace
                    .memory
                    .get("global_atomic")
                    .is_some_and(|global| global.requests != 0)
                && trace.tma_sidecar_records_observed == 0
                && result
                    .coverage()
                    .launch()
                    .grid()
                    .1
                    == 1
                && result
                    .coverage()
                    .launch()
                    .grid()
                    .2
                    == 1
                && result
                    .coverage()
                    .launch()
                    .block()
                    .is_some_and(|block| {
                        block.1 == 1 && block.2 == 1 && block.0 > 32 && block.0 <= 128
                    });
            if global_atomic_alias {
                wavefronts = wavefronts.saturating_add(u128::from(
                    trace
                        .memory
                        .get("global_atomic")
                        .map_or(0, |global| global.requests),
                ));
            }
            // `bank_conflict_excess_wavefronts` is already the sum of each
            // instruction's nonnegative excess.  Do not subtract launch-wide
            // aggregates: a wide access with no conflict must not cancel a
            // separate narrow access that does have a conflict.
            let conflicts = u128::from(row.bank_conflict_excess_wavefronts);
            m.insert(
                format!("{kind}_requests"),
                Metric::approximate_count(
                    None,
                    "request",
                    requests,
                    "effective SM120 shared-memory instruction requests from the real trace",
                ),
            );
            let opcode = match kind {
                "shared_load" => "shared_ld",
                "shared_store" => "shared_st",
                "shared_atomic" => "shared_atom",
                _ => unreachable!("shared metric table has a closed operation set"),
            };
            for ncu_metric in [
                format!("smsp__inst_executed_op_{opcode}.sum"),
                format!("smsp__sass_inst_executed_op_{opcode}.sum"),
            ] {
                m.insert(
                    format!("{kind}_opcode_{}", ncu_metric.replace('.', "_")),
                    Metric::approximate_count(
                        Some(&ncu_metric),
                        "inst",
                        opcode_requests,
                        "trace-derived ordinary shared-memory opcode cardinality; collective matrix forms are emitted in their dedicated opcode class",
                    ),
                );
            }
            m.insert(
                format!("{kind}_bank_wavefronts"),
                Metric::approximate_count(
                    Some(wavefront_metric),
                    "wavefront",
                    wavefronts,
                    if global_atomic_alias {
                        "source-derived SM120 shared-atomic observer projection including the scoped global-atomic client alias; global opcode classification remains separate"
                    } else {
                        "address-derived 32-bank/4-byte-word broadcast-collapsed shared service work; effective model, not physical bank timing"
                    },
                ),
            );
            m.insert(
                format!("{kind}_bank_conflicts"),
                Metric::approximate_count(
                    Some(conflict_metric),
                    "wavefront",
                    conflicts,
                    "wavefronts beyond the per-lane-width ideal; effective bank-conflict projection, not a cross-client profiler total",
                ),
            );
        }
        let shared_total_requests: u64 = ["shared_load", "shared_store", "shared_atomic"]
            .into_iter()
            .map(|kind| trace.memory.get(kind).map_or(0, |row| row.requests))
            .sum();
        m.insert(
            "shared_opcode_total".to_owned(),
            Metric::approximate_count(
                Some("smsp__sass_inst_executed_op_shared.sum"),
                "inst",
                u128::from(shared_total_requests),
                "trace-derived total logical shared-memory opcode cardinality; no physical pipe assignment implied",
            ),
        );
        // TMA opcode families are already preserved in the dynamic trace.
        // Emit their instruction cardinalities as a logical operation surface
        // without treating opcode text as a physical engine assignment.
        for (kind, ncu_metric) in [
            ("UTMALDG", "smsp__inst_executed_op_tma_ld.sum"),
            ("UTMASTG", "smsp__inst_executed_op_tma_st.sum"),
            ("LDGSTS", "smsp__inst_executed_op_ldgsts.sum"),
            ("UTMALDG", "smsp__sass_inst_executed_op_tma_ld.sum"),
            ("UTMASTG", "smsp__sass_inst_executed_op_tma_st.sum"),
        ] {
            let count: u64 = trace
                .by_opcode
                .iter()
                .filter(|(opcode, _)| opcode.split('.').next() == Some(kind))
                .map(|(_, activity)| activity.warp_instructions)
                .sum();
            m.insert(
                format!("{kind}_{}", ncu_metric.replace('.', "_")),
                Metric::approximate_count(
                    Some(ncu_metric),
                    "inst",
                    u128::from(count),
                    "trace-derived TMA opcode cardinality; logical operation class, not physical engine scheduling",
                ),
            );
        }
        for (key, ncu_metric, count) in [
            (
                "ldgsts_cache_access_instructions",
                "sm__sass_inst_executed_op_ldgsts_cache_access.sum",
                trace.ldgsts_cache_access_instructions,
            ),
            (
                "ldgsts_cache_bypass_instructions",
                "sm__sass_inst_executed_op_ldgsts_cache_bypass.sum",
                trace.ldgsts_cache_bypass_instructions,
            ),
        ] {
            m.insert(
                key.to_owned(),
                Metric::approximate_count(
                    Some(ncu_metric),
                    "inst",
                    u128::from(count),
                    "trace-derived logical LDGSTS cache-route cardinality; route semantics are qualified, physical cache access remains unresolved",
                ),
            );
        }
        // Additional logical operation classes with unambiguous trace
        // provenance. These counts do not select a physical pipe.
        for (key, ncu_metric, count) in [
            (
                "branch",
                "smsp__inst_executed_op_branch.sum",
                trace.branch_warp_instructions,
            ),
            (
                "global_reduction",
                "smsp__inst_executed_op_global_red.sum",
                trace.memory.get("global_reduction").map_or(0, |row| row.requests),
            ),
            (
                "ldsm",
                "smsp__inst_executed_op_ldsm.sum",
                trace
                    .by_opcode
                    .iter()
                    .filter(|(opcode, _)| opcode.split('.').next() == Some("LDSM"))
                    .map(|(_, activity)| activity.warp_instructions)
                    .sum(),
            ),
        ] {
            m.insert(
                format!("{key}_opcode_count"),
                Metric::approximate_count(
                    Some(ncu_metric),
                    "inst",
                    u128::from(count),
                    "trace-derived logical operation cardinality; no physical pipe assignment implied",
                ),
            );
        }
        for (kind, ncu_metric) in [
            ("global_load", "smsp__sass_inst_executed_op_global_ld.sum"),
            ("global_store", "smsp__sass_inst_executed_op_global_st.sum"),
            ("local_load", "smsp__sass_inst_executed_op_local_ld.sum"),
            ("local_store", "smsp__sass_inst_executed_op_local_st.sum"),
        ] {
            let count = trace.memory.get(kind).map_or(0, |row| row.requests);
            m.insert(
                format!("{kind}_opcode_count"),
                Metric::approximate_count(
                    Some(ncu_metric),
                    "inst",
                    u128::from(count),
                    "trace-derived logical memory operation cardinality; cache route retained separately and physical pipe assignment unresolved",
                ),
            );
        }
        m.insert(
            "alpha_active_sm_count".to_owned(),
            Metric::count(
                None,
                "sm",
                u128::from(active_sm_count),
                "SM IDs represented by the trace; not a full-launch occupancy claim",
            ),
        );
        return Ok(NativeMetrics {
            schema_version: 1,
            metric_surface_version: 4,
            architecture: result.arch().token(),
            sm_count,
            active_sm_count,
            reference_clock_hz: clock,
            metrics: m,
        });
    }
    if result.arch() != ArchId::Sm86 {
        m.insert(
            "native_metrics".to_owned(),
            Metric::unavailable(None, "none", "SM86-only metric definitions"),
        );
        return Ok(NativeMetrics {
            schema_version: 1,
            metric_surface_version: 2,
            architecture: result.arch().token(),
            sm_count,
            active_sm_count,
            reference_clock_hz: clock,
            metrics: m,
        });
    }
    let trace = result.trace_metrics();
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
    const LIFE: &str = "modeled admission through last instruction inclusive; excludes physical EXIT drain; ratio of summed counters, not mean per-SM percentages";
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
    let readiness = aggregate_readiness(result)?;
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
    add_scheduler_proxy_rows(&mut m, &readiness);
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
            "unique decoded global physical lines touched by this launch",
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
    for (key, suffix, classes) in [
        (
            "global_load",
            "mem_global_op_ld",
            &["global_load", "async_global_load"][..],
        ),
        ("global_store", "mem_global_op_st", &["global_store"][..]),
        (
            "global_atomic",
            "mem_global_op_atom",
            &["global_atomic"][..],
        ),
        (
            "global_reduction",
            "mem_global_op_red",
            &["global_reduction"][..],
        ),
        ("local_load", "mem_local_op_ld", &["local_load"][..]),
        ("local_store", "mem_local_op_st", &["local_store"][..]),
        (
            "unclassified_global",
            "mem_global_op_ld",
            &["unclassified_global"][..],
        ),
    ] {
        let requests: u128 = classes
            .iter()
            .filter_map(|c| trace.memory.get(*c))
            .map(|r| u128::from(r.requests))
            .sum();
        let sectors: u128 = classes
            .iter()
            .filter_map(|c| trace.memory.get(*c))
            .map(|r| u128::from(r.sectors))
            .sum();
        let unique_sectors: u128 = classes
            .iter()
            .filter_map(|c| trace.memory.get(*c))
            .map(|r| u128::from(r.unique_sectors))
            .sum();
        let unique_lines: u128 = classes
            .iter()
            .filter_map(|c| trace.memory.get(*c))
            .map(|r| u128::from(r.unique_lines))
            .sum();
        let sector_aliases: u128 = classes
            .iter()
            .filter_map(|c| trace.memory.get(*c))
            .map(|r| u128::from(r.sector_aliases))
            .sum();
        let name = format!("l1tex__t_requests_pipe_lsu_{suffix}.sum");
        let ncu_name = (!matches!(
            key,
            "unclassified_global" | "global_atomic" | "global_reduction"
        ))
        .then_some(name.as_str());
        m.insert(
            format!("{key}_requests"),
            Metric::count(
                ncu_name,
                "request",
                requests,
                "one request per effective traced LSU warp instruction",
            ),
        );
        let name = format!("l1tex__t_sectors_pipe_lsu_{suffix}.sum");
        let ncu_name = (!matches!(
            key,
            "unclassified_global" | "global_atomic" | "global_reduction"
        ))
        .then_some(name.as_str());
        let row = if trace.sectors_decoded {
            let semantics = if matches!(key, "global_atomic" | "global_reduction") {
                "SM86 write-class touched-sector projection from exact L2 request topology; alias multiplicity is preserved because GA10x serializes same-word lanes"
            } else {
                "32-byte sectors from replay address projection"
            };
            if matches!(key, "global_atomic" | "global_reduction") {
                Metric::approximate_count(ncu_name, "sector", sectors, semantics)
            } else {
                Metric::count(ncu_name, "sector", sectors, semantics)
            }
        } else {
            Metric::unavailable(
                Some(&name),
                "sector",
                "flat replay does not decode sector payloads",
            )
        };
        m.insert(format!("{key}_sectors"), row);
        for (suffix, unit, value, semantics) in [
            (
                "unique_sectors",
                "sector",
                unique_sectors,
                "sum of distinct decoded 32-byte sectors per recorded request class; classes are not cross-deduplicated, and this is not an L1/L2 transaction count",
            ),
            (
                "unique_lines",
                "line",
                unique_lines,
                "sum of distinct decoded 128-byte lines per recorded request class; classes are not cross-deduplicated, and this is not an L2 line-request count",
            ),
            (
                "sector_aliases",
                "sector",
                sector_aliases,
                "decoded sector-span entries beyond the distinct-sector domain, primarily same-word ATOM/RED aliases; not extra protocol traffic",
            ),
        ] {
            m.insert(
                format!("{key}_{suffix}"),
                if trace.sectors_decoded {
                    Metric::approximate_count(None, unit, value, semantics)
                } else {
                    Metric::unavailable(None, unit, "flat replay does not decode sector topology")
                },
            );
        }
        m.insert(
            format!("{key}_sectors_per_request"),
            if trace.sectors_decoded {
                Metric::ratio(
                    None,
                    "sector/request",
                    sectors,
                    requests,
                    false,
                    "structural coalescing ratio",
                )
            } else {
                Metric::unavailable(
                    None,
                    "sector/request",
                    "flat replay does not decode sectors",
                )
            },
        );
    }
    let atomic_requests = trace
        .memory
        .get("global_atomic")
        .map_or(0, |row| row.requests);
    let reduction_requests = trace
        .memory
        .get("global_reduction")
        .map_or(0, |row| row.requests);
    let atomic_sectors = trace
        .memory
        .get("global_atomic")
        .map_or(0, |row| row.sectors);
    let reduction_sectors = trace
        .memory
        .get("global_reduction")
        .map_or(0, |row| row.sectors);
    let global_write_class_requests =
        atomic_requests
            .checked_add(reduction_requests)
            .ok_or_else(|| {
                RunError::new(
                    "native metric global write class",
                    "request counter overflow",
                )
            })?;
    let global_write_class_sectors =
        atomic_sectors
            .checked_add(reduction_sectors)
            .ok_or_else(|| {
                RunError::new(
                    "native metric global write class",
                    "sector counter overflow",
                )
            })?;
    add_count_row(
        &mut m,
        "global_write_class_requests",
        None,
        "request",
        global_write_class_requests,
        "trace-derived global write-class instructions (returning ATOM plus non-returning RED); legacy static opcode text may not preserve the split",
    );
    m.insert(
        "global_write_class_sectors".into(),
        Metric::approximate_count(
            None,
            "sector",
            u128::from(global_write_class_sectors),
        "trace-derived global write-class sector projection; exact NCU ATOM/RED packetization is validated by the SM86 domain probe but remains approximate for legacy traces",
        ),
    );
    for (key, ncu) in [
        (
            "shared_load",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        ),
        (
            "shared_store",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        ),
        (
            "shared_atomic",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
        ),
    ] {
        let row = trace.memory.get(key);
        m.insert(
            format!("{key}_requests"),
            Metric::count(
                None,
                "request",
                row.map_or(0, |r| u128::from(r.requests)),
                "effective shared-memory warp instructions",
            ),
        );
        m.insert(format!("{key}_bank_wavefronts"), if trace.sectors_decoded {
            Metric::ratio(Some(ncu), "wavefront", row.map_or(0, |r| u128::from(r.bank_wavefronts)), 1, true, "32-bank/4-byte-word broadcast-collapsed shared service work; ideal wide-access wavefronts are retained separately from bank-conflict excess")
        } else { Metric::unavailable(Some(ncu), "wavefront", "flat replay does not preserve shared bank topology") });
    }
    let lsu_requests: u64 = trace.memory.values().map(|row| row.requests).sum();
    add_count_row(
        &mut m,
        "lsu_memory_requests",
        None,
        "request",
        lsu_requests,
        "effective memory instructions only; not the LSU instruction counter (which includes S2R, SHFL and control work)",
    );
    if let Some(resources) = trace.launch_resources {
        for (key, ncu, unit, value) in [
            (
                "registers_per_thread",
                Some("launch__registers_per_thread"),
                "register/thread",
                Some(resources.registers_per_thread),
            ),
            (
                "shared_memory_bytes_per_block",
                None,
                "byte",
                Some(resources.shared_memory_bytes_per_block),
            ),
            (
                "static_shared_memory_bytes",
                Some("launch__shared_mem_per_block_static"),
                "byte",
                resources.static_shared_memory_bytes,
            ),
            (
                "dynamic_shared_memory_bytes",
                Some("launch__shared_mem_per_block_dynamic"),
                "byte",
                resources.dynamic_shared_memory_bytes,
            ),
            (
                "shared_memory_carveout_preference_pct",
                None,
                "%",
                resources
                    .shared_memory_carveout_preference_pct
                    .filter(|value| *value >= 0)
                    .map(|value| value as u32),
            ),
            (
                "driver_occupancy_limit_blocks",
                None,
                "block/SM",
                resources.driver_occupancy_limit_blocks,
            ),
        ] {
            m.insert(
                key.into(),
                match value {
                    Some(value) => Metric::count(
                        ncu,
                        unit,
                        u128::from(value),
                        "captured CUDA function/launch resource metadata; not an NCU measurement",
                    ),
                    None => Metric::unavailable(
                        ncu,
                        unit,
                        "legacy trace has shared-memory total but lacks the static/dynamic split",
                    ),
                },
            );
        }
        if resources.block_threads != 0 {
            const WARP_SIZE: u64 = 32;
            const REGISTERS_PER_SM: u64 = 65_536;
            const REGISTER_ALLOCATION_GRANULARITY: u64 = 256;
            const SHARED_ALLOCATION_GRANULARITY: u64 = 128;
            // NVIDIA's Ampere tuning guide documents a 1 KiB per-block
            // system reservation. Keep it in the occupancy projection, not
            // in the trace's requested static+dynamic byte row.
            const SHARED_SYSTEM_RESERVATION_BYTES: u64 = 1024;
            let warps_per_block = u64::from(resources.block_threads).div_ceil(WARP_SIZE);
            let max_blocks = u64::from(
                card.require_occupancy_resources()
                    .map(|occupancy| occupancy.max_blocks)
                    .unwrap_or(0),
            );
            let regs_per_warp = u64::from(resources.registers_per_thread)
                .saturating_mul(WARP_SIZE)
                .div_ceil(REGISTER_ALLOCATION_GRANULARITY)
                .saturating_mul(REGISTER_ALLOCATION_GRANULARITY);
            let registers_per_thread_allocated = regs_per_warp / WARP_SIZE;
            let regs_per_block = regs_per_warp.saturating_mul(warps_per_block);
            let registers_per_subpartition = REGISTERS_PER_SM / 4;
            let register_warps_per_subpartition = if regs_per_warp == 0 {
                0
            } else {
                registers_per_subpartition
                    .checked_div(regs_per_warp)
                    .unwrap_or(0)
            };
            let register_warps_per_sm = register_warps_per_subpartition.saturating_mul(4);
            let register_limited_blocks = if regs_per_block == 0 {
                max_blocks
            } else {
                register_warps_per_sm / warps_per_block.max(1)
            };
            let shared_bytes = u64::from(resources.shared_memory_bytes_per_block);
            let shared_allocated = shared_bytes
                .div_ceil(SHARED_ALLOCATION_GRANULARITY)
                .saturating_mul(SHARED_ALLOCATION_GRANULARITY)
                .saturating_add(SHARED_SYSTEM_RESERVATION_BYTES);
            let shared_capacity = external.map_or_else(
                || {
                    card.require_l1_geometry()
                        .map(|geometry| geometry.max_shared_bytes)
                        .unwrap_or(0)
                },
                |observation| u64::from(observation.shared_memory_config_bytes),
            );
            let shared_limited_blocks = if shared_allocated == 0 {
                max_blocks
            } else {
                shared_capacity.checked_div(shared_allocated).unwrap_or(0)
            };
            let max_warps = u64::from(max_warps);
            let warp_limited_blocks = max_warps / warps_per_block.max(1);
            let theoretical_blocks = register_limited_blocks
                .min(shared_limited_blocks)
                .min(max_blocks)
                .min(warp_limited_blocks);
            let theoretical_warps = theoretical_blocks.saturating_mul(warps_per_block);
            let resource_semantics = "SM86 occupancy resource projection: 64K 32-bit registers partitioned across four SMSPs, 256-register warp allocation, 128-byte shared allocation, carded shared/warp/block limits; launch metadata and architecture constants, not NCU labels";
            add_count_row(
                &mut m,
                "block_threads",
                Some("launch__block_size"),
                "thread/block",
                u64::from(resources.block_threads),
                "captured block geometry metadata",
            );
            add_count_row(
                &mut m,
                "grid_blocks",
                Some("launch__grid_size"),
                "block",
                resources.grid_blocks,
                "captured grid geometry metadata",
            );
            add_count_row(
                &mut m,
                "block_size",
                Some("launch__block_size"),
                "thread/block",
                u64::from(resources.block_threads),
                "captured block geometry metadata; NCU launch-size crosswalk",
            );
            add_count_row(
                &mut m,
                "grid_size",
                Some("launch__grid_size"),
                "block",
                resources.grid_blocks,
                "captured grid geometry metadata; NCU launch-size crosswalk",
            );
            m.insert(
                "registers_per_thread_allocated".into(),
                Metric::approximate_count(
                    Some("launch__registers_per_thread_allocated"),
                    "register/thread",
                    u128::from(registers_per_thread_allocated),
                    "SM86 256-register warp allocation granularity projected from captured registers/thread",
                ),
            );
            m.insert(
                "shared_memory_bytes_per_block_allocated".into(),
                Metric::approximate_count(
                    Some("launch__shared_mem_per_block_allocated"),
                    "byte/block",
                    u128::from(shared_allocated),
                    "SM86 128-byte allocation granularity plus NVIDIA-documented 1 KiB per-block system reservation projected from captured static+dynamic bytes",
                ),
            );
            m.insert(
                "shared_memory_capacity_bytes".into(),
                Metric::approximate_count(
                    Some("launch__shared_mem_config_size"),
                    "byte/SM",
                    u128::from(shared_capacity),
                    if external.is_some() {
                        "CUPTI driver-selected L1/shared-memory configuration used as an external occupancy boundary"
                    } else {
                        "SM86 maximum shared-memory capacity from the architecture card; runtime carveout is not present in this trace"
                    },
                ),
            );
            for (key, ncu, value, semantics) in [
                (
                    "occupancy_limit_blocks",
                    "launch__occupancy_limit_blocks",
                    max_blocks,
                    "carded SM86 architectural maximum blocks/SM",
                ),
                (
                    "occupancy_limit_registers",
                    "launch__occupancy_limit_registers",
                    register_limited_blocks,
                    "register-limited blocks/SM from SM86 allocation granularity",
                ),
                (
                    "occupancy_limit_shared_memory",
                    "launch__occupancy_limit_shared_mem",
                    shared_limited_blocks,
                    "shared-memory-limited blocks/SM from SM86 allocation granularity",
                ),
                (
                    "occupancy_limit_warps",
                    "launch__occupancy_limit_warps",
                    warp_limited_blocks,
                    "warp-capacity-limited blocks/SM from carded resident-warp capacity",
                ),
            ] {
                m.insert(
                    key.into(),
                    Metric::approximate_count(Some(ncu), "block/SM", u128::from(value), semantics),
                );
            }
            add_count_row(
                &mut m,
                "warps_per_block_theoretical",
                None,
                "warp/block",
                warps_per_block,
                resource_semantics,
            );
            add_count_row(
                &mut m,
                "register_limited_blocks_per_sm",
                None,
                "block/SM",
                register_limited_blocks,
                resource_semantics,
            );
            add_count_row(
                &mut m,
                "shared_limited_blocks_per_sm",
                None,
                "block/SM",
                shared_limited_blocks,
                resource_semantics,
            );
            add_count_row(
                &mut m,
                "resource_limited_blocks_per_sm",
                None,
                "block/SM",
                theoretical_blocks,
                resource_semantics,
            );
            m.insert(
                "theoretical_occupancy_pct".into(),
                Metric::ratio(
                    None,
                    "%",
                    u128::from(theoretical_warps) * 100,
                    u128::from(max_warps),
                    true,
                    resource_semantics,
                ),
            );
        } else {
            for (key, unit, reason) in [
                (
                    "block_threads",
                    "thread/block",
                    "legacy trace lacks validated block geometry",
                ),
                (
                    "block_size",
                    "thread/block",
                    "legacy trace lacks validated block geometry",
                ),
                (
                    "grid_size",
                    "block",
                    "legacy trace lacks validated grid geometry",
                ),
                (
                    "theoretical_occupancy_pct",
                    "%",
                    "legacy trace lacks validated launch resources",
                ),
            ] {
                m.insert(key.to_owned(), Metric::unavailable(None, unit, reason));
            }
        }
    }
    if trace.launch_resources.is_none() {
        // Launch geometry is part of the validated replay-coverage witness,
        // even for legacy traces that predate the optional launch-resource
        // sidecar. Keep that evidence instead of turning it into a typed gap.
        // Resource-dependent limits (registers/shared memory/driver choice)
        // remain unavailable below.
        let launch = result.coverage().launch();
        let (gx, gy, gz) = launch.grid();
        let grid_blocks = u64::from(gx)
            .checked_mul(u64::from(gy))
            .and_then(|value| value.checked_mul(u64::from(gz)));
        if let Some((bx, by, bz)) = launch.block() {
            if let Some(block_threads) = u64::from(bx)
                .checked_mul(u64::from(by))
                .and_then(|value| value.checked_mul(u64::from(bz)))
            {
                add_count_row(
                    &mut m,
                    "block_threads",
                    Some("launch__block_size"),
                    "thread/block",
                    block_threads,
                    "validated replay launch geometry; resource sidecar is absent",
                );
                add_count_row(
                    &mut m,
                    "block_size",
                    Some("launch__block_size"),
                    "thread/block",
                    block_threads,
                    "validated replay launch geometry; resource sidecar is absent",
                );
                let warps_per_block = block_threads.div_ceil(32);
                let warp_limited_blocks =
                    u128::from(max_warps) / u128::from(warps_per_block.max(1));
                m.insert(
                    "warps_per_block_theoretical".into(),
                    Metric::count(
                        None,
                        "warp/block",
                        u128::from(warps_per_block),
                        "validated replay launch geometry divided by the SM86 warp size; no register/shared sidecar",
                    ),
                );
                m.insert(
                    "occupancy_limit_warps".into(),
                    Metric::approximate_count(
                        Some("launch__occupancy_limit_warps"),
                        "block/SM",
                        warp_limited_blocks,
                        "warp-capacity-limited blocks/SM from validated launch geometry and carded resident-warp capacity; register/shared limits remain separate",
                    ),
                );
            }
        }
        if let Some(grid_blocks) = grid_blocks {
            add_count_row(
                &mut m,
                "grid_blocks",
                Some("launch__grid_size"),
                "block",
                grid_blocks,
                "validated replay launch geometry; resource sidecar is absent",
            );
            add_count_row(
                &mut m,
                "grid_size",
                Some("launch__grid_size"),
                "block",
                grid_blocks,
                "validated replay launch geometry; resource sidecar is absent",
            );
        }
        for (key, unit, reason) in [
            (
                "registers_per_thread",
                "register/thread",
                "legacy trace lacks validated launch-resource metadata",
            ),
            (
                "static_shared_memory_bytes",
                "byte",
                "legacy trace lacks validated launch-resource metadata",
            ),
            (
                "dynamic_shared_memory_bytes",
                "byte",
                "legacy trace lacks validated launch-resource metadata",
            ),
            (
                "shared_memory_carveout_preference_pct",
                "%",
                "trace has no explicit CUDA carveout preference",
            ),
            (
                "driver_occupancy_limit_blocks",
                "block/SM",
                "legacy trace lacks driver occupancy metadata",
            ),
            (
                "block_threads",
                "thread/block",
                "legacy trace lacks validated block geometry",
            ),
            (
                "theoretical_occupancy_pct",
                "%",
                "legacy trace lacks validated launch resources",
            ),
        ] {
            m.entry(key.to_owned())
                .or_insert_with(|| Metric::unavailable(None, unit, reason));
        }
    }
    for (key, ncu, unit, reason) in [
        (
            "sm_elapsed_cycles_estimate",
            Some("sm__cycles_elapsed.avg"),
            "cycle",
            "no independent SM elapsed frontier; the GPC observation is kept separate",
        ),
        (
            "sm_stalled_cycles",
            Some("sm__cycles_stalled.avg"),
            "cycle",
            "scheduler readiness diagnostics are not an SM output-interface stall counter",
        ),
        (
            "sm_idle_cycles",
            Some("sm__cycles_idle.avg"),
            "cycle",
            "idle intervals are not reconstructed from event jumps",
        ),
        (
            "eligible_warps",
            Some("smsp__warps_eligible.sum"),
            "warp",
            "ready-warp population is not sampled at every hardware scheduler cycle",
        ),
        (
            "long_scoreboard_stall_units",
            Some("smsp__warps_issue_stalled_long_scoreboard.sum"),
            "warp_cycle",
            "NCU overlapping warp-state samples cannot be reconstructed from exclusive readiness buckets",
        ),
        (
            "short_scoreboard_stall_units",
            Some("smsp__warps_issue_stalled_short_scoreboard.sum"),
            "warp_cycle",
            "NCU overlapping warp-state samples cannot be reconstructed from exclusive readiness buckets",
        ),
        (
            "branch_target_divergent_threads",
            Some("smsp__branch_targets_threads_divergent"),
            "thread",
            "NCU's similarly named counter is branch-event valued; an equivalent per-thread target-population producer is not present in the trace contract",
        ),
        (
            "physical_pipeline_adu_instructions",
            Some("smsp__inst_executed_pipe_adu.sum"),
            "inst",
            "static SASS/trace records do not carry the dynamic ADU pipe assignment",
        ),
        (
            "physical_pipeline_cbu_instructions",
            Some("smsp__inst_executed_pipe_cbu.sum"),
            "inst",
            "static SASS/trace records do not carry the dynamic CBU pipe assignment",
        ),
        (
            "physical_pipeline_fp64_instructions",
            Some("smsp__inst_executed_pipe_fp64.sum"),
            "inst",
            "static SASS/trace records do not carry the dynamic FP64 pipe assignment",
        ),
        (
            "physical_pipeline_uniform_instructions",
            Some("smsp__inst_executed_pipe_uniform.sum"),
            "inst",
            "static SASS/trace records do not carry the dynamic uniform-pipe assignment",
        ),
        (
            "physical_pipeline_xu_instructions",
            Some("smsp__inst_executed_pipe_xu.sum"),
            "inst",
            "static SASS/trace records do not carry the dynamic XU pipe assignment",
        ),
        (
            "physical_pipeline_assignment_pct",
            None,
            "%",
            "NVBit trace lacks dynamic physical-pipeline attribution; logical opcode projections remain separate",
        ),
        (
            "registers_per_thread",
            Some("launch__registers_per_thread"),
            "register/thread",
            "register allocation is not carried by the compact dynamic trace",
        ),
        (
            "static_shared_memory_bytes",
            Some("launch__shared_mem_per_block_static"),
            "byte",
            "launch resource metadata is not part of the replay contract",
        ),
        (
            "dynamic_shared_memory_bytes",
            Some("launch__shared_mem_per_block_dynamic"),
            "byte",
            "launch resource metadata is not part of the replay contract",
        ),
        (
            "shared_memory_bytes_per_block_driver",
            Some("launch__shared_mem_per_block_driver"),
            "byte/block",
            "driver allocation metadata is not equivalent to the trace's requested static+dynamic total",
        ),
        (
            "occupancy_per_block_size",
            Some("launch__occupancy_per_block_size"),
            "limit",
            "NCU occupancy calculator contribution is not reconstructed as an independent hardware calculator",
        ),
        (
            "occupancy_per_register_count",
            Some("launch__occupancy_per_register_count"),
            "limit",
            "NCU occupancy calculator contribution is not reconstructed as an independent hardware calculator",
        ),
        (
            "occupancy_per_shared_memory_size",
            Some("launch__occupancy_per_shared_mem_size"),
            "limit",
            "NCU occupancy calculator contribution is not reconstructed as an independent hardware calculator",
        ),
        (
            "occupancy_limit_warps",
            Some("launch__occupancy_limit_warps"),
            "block/SM",
            "legacy trace lacks validated launch resources",
        ),
        (
            "shared_memory_config_size",
            Some("launch__shared_mem_config_size"),
            "byte",
            "runtime L1/shared-memory carveout configuration is not carried by the trace contract",
        ),
        (
            "sm_cycles",
            Some("sm__cycles_elapsed.avg"),
            "cycle",
            "the model exposes separate SM-active and GPC envelope frontiers, not a hardware elapsed-cycle roll-up",
        ),
        (
            "gpc_cycles_avg",
            Some("gpc__cycles_elapsed.avg"),
            "cycle",
            "per-GPC average is not reconstructed from the SM/GPC launch envelope",
        ),
        (
            "l1_unified_hit_rate_pct",
            Some("l1tex__t_sector_hit_rate.pct"),
            "%",
            "L1 counters require L1/L2 replay; flat mode has no cache state",
        ),
        (
            "l2_tex_lookup_sectors",
            Some("lts__t_sectors_srcunit_tex.sum"),
            "sector",
            "L2 counters require shared-L2 replay",
        ),
        (
            "l2_tex_lookup_hits",
            Some("lts__t_sectors_srcunit_tex_lookup_hit.sum"),
            "sector",
            "L2 counters require shared-L2 replay",
        ),
        (
            "l2_tex_hit_rate_pct",
            Some("lts__t_sector_hit_rate.pct"),
            "%",
            "L2 counters require shared-L2 replay",
        ),
        (
            "dram_read_bytes",
            Some("dram__bytes_read.sum"),
            "byte",
            "DRAM counters require shared-L2 replay with a DRAM observer",
        ),
        (
            "dirty_eviction_sectors",
            None,
            "sector",
            "DRAM counters require shared-L2 replay with a DRAM observer",
        ),
        (
            "dirty_eviction_bytes",
            None,
            "byte",
            "DRAM counters require shared-L2 replay with a DRAM observer",
        ),
        (
            "writeback_service_cycles_estimate",
            None,
            "cycle",
            "requires shared-L2 replay with carded DRAM service metadata",
        ),
        (
            "dirty_writeback_demand_sectors",
            None,
            "sector",
            "shared-L2 replay can bound dirty demand, but cannot identify the profiler's kernel/flush interval",
        ),
        (
            "dirty_writeback_demand_bytes",
            None,
            "byte",
            "shared-L2 replay can bound dirty demand, but cannot identify the profiler's kernel/flush interval",
        ),
        (
            "dirty_writeback_demand_lower_bound_sectors",
            None,
            "sector",
            "shared-L2 replay is required to observe dirty eviction demand",
        ),
        (
            "dirty_writeback_demand_terminal_sectors",
            None,
            "sector",
            "shared-L2 replay is required to observe terminal dirty residency",
        ),
        (
            "dirty_writeback_demand_interval_width_sectors",
            None,
            "sector",
            "shared-L2 replay is required to bound terminal dirty demand",
        ),
        (
            "terminal_flush_service_cycles_estimate",
            None,
            "cycle",
            "serialized terminal-dirty service demand only; excludes queue position and flush visibility",
        ),
        (
            "dirty_writeback_completion_frontier_cycles",
            None,
            "cycle",
            "requires shared-L2 replay to expose the modeled terminal event frontier",
        ),
        (
            "dirty_writeback_completion_scenario_end_cycles",
            None,
            "cycle",
            "requires shared-L2 replay, card SM clock, and validated DRAM peak",
        ),
        (
            "dirty_writeback_completion_service_width_cycles",
            None,
            "cycle",
            "requires shared-L2 replay, card SM clock, and validated DRAM peak",
        ),
        (
            "coherence_completion_frontier_cycles",
            None,
            "cycle",
            "requires decoded cross-SM topology and a modeled terminal event frontier",
        ),
        (
            "coherence_completion_scenario_end_cycles",
            None,
            "cycle",
            "requires decoded cross-SM topology and a modeled terminal event frontier",
        ),
        (
            "coherence_completion_service_width_cycles",
            None,
            "cycle",
            "requires decoded cross-SM topology",
        ),
        (
            "atomic_rmw_read_sectors",
            None,
            "sector",
            "shared-L2 replay is required to attribute cold-sector RMW reads",
        ),
        (
            "reduction_rmw_read_sectors",
            None,
            "sector",
            "shared-L2 replay is required to attribute cold-sector RMW reads",
        ),
        (
            "terminal_dirty_sectors",
            None,
            "sector",
            "shared-L2 replay is required to observe terminal dirty residency",
        ),
        (
            "terminal_dirty_bytes",
            None,
            "byte",
            "shared-L2 replay is required to observe terminal dirty residency",
        ),
    ] {
        ensure_unavailable(&mut m, key, ncu, unit, reason);
    }
    for suffix in [
        "barrier",
        "branch_resolving",
        "dispatch_stall",
        "drain",
        "imc_miss",
        "lg_throttle",
        "math_pipe_throttle",
        "membar",
        "mio_throttle",
        "misc",
        "no_instructions",
        "not_selected",
        "selected",
        "sleeping",
        "tex_throttle",
        "wait",
    ] {
        let key = format!("scheduler_stall_{suffix}_samples");
        let ncu = format!("smsp__warps_issue_stalled_{suffix}.sum");
        m.entry(key).or_insert_with(|| {
            Metric::unavailable(
                Some(ncu.as_str()),
                "warp",
                "NCU scheduler samples overlap and are not reconstructed by the simulator's exclusive readiness causes",
            )
        });
    }
    // Per-origin local cache counts were modeled but previously not exposed.
    if let Some(l1) = result.l1() {
        // The request-mask audit is fixed-size and diagnostic-only.  Keep the
        // detailed client×mask arrays in TraceMetrics for machine-readable
        // inspection, while exposing only compact totals in the native report
        // so a wide request cannot expand the metric surface.
        let mask_audit = &trace.l1_request_mask_audit;
        let request_count = mask_audit.request_count.iter().copied().sum();
        let request_line_count = mask_audit.request_line_count.iter().copied().sum();
        let request_mask_overflow = mask_audit.request_mask_overflow.iter().copied().sum();
        let same_request_mask_merges: u64 = mask_audit
            .same_request_merge_masks
            .iter()
            .flat_map(|counts| counts.iter())
            .copied()
            .sum();
        let older_pending_mask_merges: u64 = mask_audit
            .older_pending_merge_masks
            .iter()
            .flat_map(|counts| counts.iter())
            .copied()
            .sum();
        for (name, value, unit, semantics) in [
            (
                "request_mask_audit_requests",
                request_count,
                "request",
                "bounded per-request L1 line-mask audit; diagnostic only",
            ),
            (
                "request_mask_audit_lines",
                request_line_count,
                "line",
                "four-sector line footprints retained by the bounded L1 request audit",
            ),
            (
                "request_mask_audit_same_request_merges",
                same_request_mask_merges,
                "sector",
                "same-request observed resident/merge mask observations; diagnostic only",
            ),
            (
                "request_mask_audit_older_pending_merges",
                older_pending_mask_merges,
                "sector",
                "older-request observed resident/merge mask observations; diagnostic only",
            ),
            (
                "request_mask_audit_overflow",
                request_mask_overflow,
                "request",
                "requests exceeding the bounded temporary line-mask map",
            ),
        ] {
            add_count_row(&mut m, name, None, unit, value, semantics);
        }
        for (index, name) in [
            "unknown",
            "ordinary_global",
            "constant",
            "local",
            "async_copy",
        ]
        .into_iter()
        .enumerate()
        {
            add_count_row(
                &mut m,
                format!("request_mask_audit_{name}_requests"),
                None,
                "request",
                mask_audit.request_count[index],
                "bounded L1 request client-scope audit; diagnostic only",
            );
            add_count_row(
                &mut m,
                format!("request_mask_audit_{name}_lines"),
                None,
                "line",
                mask_audit.request_line_count[index],
                "bounded L1 request client-scope line-mask audit; diagnostic only",
            );
        }
        let unified_rows = [
            ("hits", l1.hits, "sector"),
            ("misses", l1.misses, "sector"),
            ("merges", l1.merges, "sector"),
            ("pending_lookup_hits", l1.pending_lookup_hits, "sector"),
            ("stall_events", l1.stall_events, "event"),
            ("line_fills", l1.line_fills, "line"),
            ("line_alloc_misses", l1.line_alloc_misses, "line"),
            (
                "sector_resident_misses",
                l1.sector_resident_misses,
                "sector",
            ),
            ("bypass_sectors", l1.bypass_sectors, "sector"),
            ("bypass_atom_sectors", l1.bypass_atom_sectors, "sector"),
            ("untagged_sectors", l1.untagged_sectors, "sector"),
            ("store_sectors", l1.store_sectors, "sector"),
            ("store_hits", l1.store_hits, "sector"),
            ("store_tag_hits", l1.store_tag_hits, "sector"),
            ("write_lookup_hits", l1.write_lookup_hits, "sector"),
            ("reduction_sectors", l1.reduction_sectors, "sector"),
            ("reduction_hits", l1.reduction_hits, "sector"),
        ];
        for (name, value, unit) in unified_rows {
            add_count_row(
                &mut m,
                format!("l1_unified_{name}"),
                None,
                unit,
                value,
                "modeled unified L1TEX observer counter; cache state and lookup packetization are explicit",
            );
        }
        let [store_hits, reduction_hits] =
            crate::emit::l1_write_hit_terms(result.arch(), l1, Some(result.trace_metrics()));
        let unified_hit_num = u128::from(l1.hits)
            + u128::from(l1.pending_lookup_hits)
            + u128::from(store_hits)
            + u128::from(reduction_hits);
        let unified_hit_den = u128::from(l1.hits)
            + u128::from(l1.misses)
            + u128::from(l1.store_sectors)
            + u128::from(l1.reduction_sectors)
            + u128::from(l1.bypass_sectors)
            + u128::from(l1.untagged_sectors);
        m.insert(
            "l1_unified_hit_rate_pct".into(),
            Metric::ratio(
                Some("l1tex__t_sector_hit_rate.pct"),
                "%",
                unified_hit_num * 100,
                unified_hit_den,
                false,
                "architecture-scoped unified L1TEX projection; SM120 uses effective resident-sector write hits, not the SM86 write-lookup observer",
            ),
        );
        for (origin, name) in [
            (MemoryOrigin::Global, "global"),
            (MemoryOrigin::Local, "local"),
        ] {
            let c = l1.origin(origin);
            let hits = u128::from(c.hits) + u128::from(c.pending_lookup_hits);
            let total = u128::from(c.hits)
                + u128::from(c.misses)
                + u128::from(
                    c.bypass_sectors
                        .checked_sub(c.bypass_atom_sectors)
                        .ok_or_else(|| {
                            RunError::new(
                                "native metric L1 lookup",
                                format!("{name} atom-bypass subset exceeds bypass sectors"),
                            )
                        })?,
                );
            let lookup_misses = total.checked_sub(hits).ok_or_else(|| {
                RunError::new(
                    "native metric L1 lookup",
                    format!("{name} lookup hits exceed lookup denominator"),
                )
            })?;
            let origin_rows = [
                ("hits", c.hits, "sector"),
                ("misses", c.misses, "sector"),
                ("merges", c.merges, "sector"),
                ("pending_lookup_hits", c.pending_lookup_hits, "sector"),
                ("stall_events", c.stall_events, "event"),
                ("line_fills", c.line_fills, "line"),
                ("line_alloc_misses", c.line_alloc_misses, "line"),
                ("sector_resident_misses", c.sector_resident_misses, "sector"),
                ("bypass_sectors", c.bypass_sectors, "sector"),
                ("bypass_atom_sectors", c.bypass_atom_sectors, "sector"),
                ("untagged_sectors", c.untagged_sectors, "sector"),
                ("store_sectors", c.store_sectors, "sector"),
                ("store_hits", c.store_hits, "sector"),
                ("store_tag_hits", c.store_tag_hits, "sector"),
                ("write_lookup_hits", c.write_lookup_hits, "sector"),
                ("reduction_sectors", c.reduction_sectors, "sector"),
                ("reduction_hits", c.reduction_hits, "sector"),
            ];
            for (field, value, unit) in origin_rows {
                add_count_row(
                    &mut m,
                    format!("l1_{name}_{field}"),
                    None,
                    unit,
                    value,
                    "modeled per-address-origin L1TEX observer counter",
                );
            }
            m.insert(
                format!("{name}_load_lookup_hits"),
                Metric::count(
                    Some(&format!(
                        "l1tex__t_sectors_pipe_lsu_mem_{name}_op_ld_lookup_hit.sum"
                    )),
                    "sector",
                    hits,
                    "resident hits plus earlier-request pending lookup hits",
                ),
            );
            m.insert(
                format!("{name}_load_lookup_misses"),
                Metric::count(
                    Some(&format!(
                        "l1tex__t_sectors_pipe_lsu_mem_{name}_op_ld_lookup_miss.sum"
                    )),
                    "sector",
                    lookup_misses,
                    "lookup observer denominator minus hits",
                ),
            );
            m.insert(
                format!("{name}_load_lookup_hit_rate_pct"),
                Metric::ratio(
                    Some("l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct")
                        .filter(|_| origin == MemoryOrigin::Global),
                    "%",
                    hits * 100,
                    total,
                    false,
                    "per-origin lookup hit ratio; global row follows the op_ld observer domain",
                ),
            );
        }
    }
    if let Some(l2) = result.l2() {
        m.insert(
            "l2_tex_lookup_sectors".into(),
            Metric::count(
                Some("lts__t_sectors_srcunit_tex.sum"),
                "sector",
                u128::from(l2.lookup_requests),
                "packetized lookup observer; not lts request count",
            ),
        );
        m.insert(
            "l2_tex_lookup_hits".into(),
            Metric::count(
                Some("lts__t_sectors_srcunit_tex_lookup_hit.sum"),
                "sector",
                u128::from(l2.lookup_hits),
                "packetized lookup observer",
            ),
        );
        m.insert(
            "l2_tex_hit_rate_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(l2.lookup_hits) * 100,
                u128::from(l2.lookup_requests),
                false,
                "TEX-source lookup hit rate over the conserving sector observer; the generic lts__t_sector_hit_rate.pct endpoint has broader/driver-version-dependent scope and is not silently joined",
            ),
        );
    }
    if let Some(dram) = result.dram() {
        let l2_fetch_read_sectors = u128::from(dram.read_sectors);
        let tma_payload_sectors = if result.arch() == ArchId::Sm120 {
            u128::from(
                result
                    .trace_metrics()
                    .tma_sidecar_load_payload_bytes
                    .div_ceil(32),
            )
        } else {
            0
        };
        let compulsory_instruction_sectors = u128::from(
            result
                .front_end()
                .compulsory_instruction_read_sectors(),
        );
        let dram_read_sectors = l2_fetch_read_sectors
            .saturating_add(tma_payload_sectors)
            .saturating_add(compulsory_instruction_sectors);
        let dram_read_bytes = dram_read_sectors * 32;
        m.insert(
            "dram_read_bytes".into(),
            Metric::count(
                Some("dram__bytes_read.sum"),
                "byte",
                dram_read_bytes,
                "modeled L2-missing read demand plus SM120 TMA payload and compulsory instruction sectors; typed demand, not achieved DRAM service",
            ),
        );
        m.insert(
            "dram_l2_fetch_read_sectors".into(),
            Metric::count(
                None,
                "sector",
                l2_fetch_read_sectors,
                "L2-missing data/RMW read demand; excludes TMA payload and instruction fetch",
            ),
        );
        m.insert(
            "dram_tma_payload_sectors".into(),
            Metric::count(
                None,
                "sector",
                tma_payload_sectors,
                "SM120 sidecar TMA global-to-shared payload demand; not an L2 miss or completion timestamp",
            ),
        );
        m.insert(
            "dram_compulsory_instruction_sectors".into(),
            Metric::count(
                None,
                "sector",
                compulsory_instruction_sectors,
                "distinct executed instruction-read sectors; demand observer only",
            ),
        );
        // This is intentionally a read-only rate estimate.  Total DRAM
        // throughput remains unavailable below because dirty writeback,
        // coherence, and the profiler's kernel/flush interval are not in the
        // trace contract.  Keeping a separate key prevents consumers from
        // mistaking a useful diagnostic for `dram__bytes.sum.per_second`.
        let read_rate = match (result.gpc_observation_frontier(), clock) {
            (Some(frontier), Some(hz)) => Metric::ratio(
                None,
                "byte/s",
                dram_read_bytes.saturating_mul(u128::from(hz)),
                u128::from(frontier.cycles),
                true,
                "read-only DRAM demand divided by the modeled GPC observation frontier; excludes dirty writeback/coherence and is not total profiler DRAM throughput",
            ),
            _ => Metric::unavailable(
                None,
                "byte/s",
                "requires a shared-L2 DRAM observer, card SM clock, and modeled GPC frontier",
            ),
        };
        m.insert("dram_read_bytes_per_second_estimate".into(), read_rate);
        let read_bandwidth = match (
            result.gpc_observation_frontier(),
            clock,
            card.require_dram_bandwidth()
                .ok()
                .and_then(|bandwidth| bandwidth.validated_peak_bytes_per_second().ok()),
        ) {
            (Some(frontier), Some(hz), Some(peak_bytes_per_second)) => Metric::ratio(
                None,
                "%",
                dram_read_bytes
                    .saturating_mul(u128::from(hz))
                    .saturating_mul(100),
                u128::from(frontier.cycles).saturating_mul(u128::from(peak_bytes_per_second)),
                true,
                "read-only modeled demand divided by the validated card peak; excludes writeback/coherence and is not NCU achieved bandwidth",
            ),
            _ => Metric::unavailable(
                None,
                "%",
                "requires a shared-L2 DRAM observer, card SM clock, GPC frontier, and validated card peak bandwidth",
            ),
        };
        m.insert("dram_read_bandwidth_pct_estimate".into(), read_bandwidth);
        m.insert(
            "atomic_rmw_read_sectors".into(),
            Metric::count(
                None,
                "sector",
                u128::from(dram.atomic_rmw_read_sectors),
                "cold-sector read demand charged by returning ATOM operations; value-dependent atomic serialization remains outside the event stream",
            ),
        );
        m.insert(
            "reduction_rmw_read_sectors".into(),
            Metric::count(
                None,
                "sector",
                u128::from(dram.reduction_rmw_read_sectors),
                "cold-sector read demand charged by non-returning RED/REDG operations; reduction/coherence semantics remain typed tails",
            ),
        );
        m.insert(
            "terminal_dirty_sectors".into(),
            Metric::count(
                None,
                "sector",
                u128::from(dram.maximum_terminal_l2_dirty_sectors),
                "maximum terminal L2 dirty residency gauge; a flush/writeback demand bound, not completed DRAM traffic",
            ),
        );
        m.insert(
            "terminal_dirty_bytes".into(),
            Metric::count(
                None,
                "byte",
                u128::from(dram.maximum_terminal_l2_dirty_sectors) * 32,
                "maximum terminal L2 dirty residency gauge converted to bytes; not completed DRAM traffic",
            ),
        );
        let dirty_writeback_demand_sectors = u128::from(dram.l2_dirty_evicted_sectors)
            .checked_add(u128::from(dram.maximum_terminal_l2_dirty_sectors))
            .ok_or_else(|| {
                RunError::new("native metric writeback", "dirty-demand counter overflow")
            })?;
        let dirty_eviction_sectors = u128::from(dram.l2_dirty_evicted_sectors);
        let terminal_dirty_sectors = u128::from(dram.maximum_terminal_l2_dirty_sectors);
        // Explicit model-relative completion envelope. The lower edge is the
        // last modeled event; the scenario end appends serialized service for
        // all dirty demand. Protocol ordering, compression, queue position,
        // and profiler flush visibility remain outside this approximation.
        let writeback_service_cycles = card
            .require_dram_bandwidth()
            .ok()
            .and_then(|bandwidth| bandwidth.validated_peak_bytes_per_second().ok())
            .and_then(|peak| {
                clock.and_then(|hz| {
                    dirty_writeback_demand_sectors
                        .checked_mul(32)
                        .and_then(|bytes| bytes.checked_mul(u128::from(hz)))
                        .and_then(|work| {
                            u128::from(peak)
                                .checked_sub(1)
                                .and_then(|rounding| work.checked_add(rounding))
                        })
                        .and_then(|work| work.checked_div(u128::from(peak)))
                })
            });
        let writeback_lower_bound = result.modeled_terminal_lower_bound_cycles();
        let writeback_scenario_end = writeback_lower_bound.and_then(|lower| {
            writeback_service_cycles.and_then(|service| {
                u64::try_from(service)
                    .ok()
                    .and_then(|service| lower.checked_add(service))
            })
        });
        m.insert(
            "dirty_writeback_completion_frontier_cycles".into(),
            match writeback_lower_bound {
                Some(lower) => Metric::approximate_count(
                    None,
                    "cycle",
                    u128::from(lower),
                    "last modeled event before the explicit dirty-writeback completion scenario; frontier only",
                ),
                None => Metric::unavailable(None, "cycle", "requires a modeled terminal event frontier"),
            },
        );
        m.insert(
            "dirty_writeback_completion_scenario_end_cycles".into(),
            match writeback_scenario_end {
                Some(end) => Metric::approximate_count(
                    None,
                    "cycle",
                    u128::from(end),
                    "modeled terminal frontier plus serialized dirty-demand service under the declared peak-service scenario; excludes coherence/protocol and profiler flush visibility",
                ),
                None => Metric::unavailable(
                    None,
                    "cycle",
                    "requires a modeled terminal frontier, card SM clock, and validated DRAM peak",
                ),
            },
        );
        m.insert(
            "dirty_writeback_completion_service_width_cycles".into(),
            match writeback_service_cycles {
                Some(width) => Metric::approximate_count(
                    None,
                    "cycle",
                    width,
                    "serialized service-width component of the dirty-writeback completion envelope; not a physical completion tail",
                ),
                None => Metric::unavailable(
                    None,
                    "cycle",
                    "requires the card SM clock and validated DRAM peak",
                ),
            },
        );
        m.insert(
            "dirty_writeback_demand_lower_bound_sectors".into(),
            Metric::approximate_count(
                None,
                "sector",
                dirty_eviction_sectors,
                "observed dirty L2 eviction demand; lower component of the eventual writeback-demand interval",
            ),
        );
        m.insert(
            "dirty_writeback_demand_terminal_sectors".into(),
            Metric::approximate_count(
                None,
                "sector",
                terminal_dirty_sectors,
                "terminal dirty-residency component of eventual writeback demand; a gauge, not completed traffic",
            ),
        );
        m.insert(
            "dirty_writeback_demand_interval_width_sectors".into(),
            Metric::approximate_count(
                None,
                "sector",
                terminal_dirty_sectors,
                "width between observed dirty-eviction demand and the eventual-demand upper bound",
            ),
        );
        m.insert(
            "dirty_writeback_demand_sectors".into(),
            Metric::approximate_count(
                None,
                "sector",
                dirty_writeback_demand_sectors,
                "eventual dirty-sector service demand bound: evicted dirty sectors plus terminal dirty residency; not equivalent to NCU dram__sectors_write.sum",
            ),
        );
        m.insert(
            "dirty_writeback_demand_bytes".into(),
            Metric::approximate_count(
                None,
                "byte",
                dirty_writeback_demand_sectors * 32,
                "eventual dirty-byte service demand bound; not a profiler-interval write count",
            ),
        );
        m.insert(
            "terminal_flush_service_cycles_estimate".into(),
            match clock {
                Some(hz) => Metric::ratio(
                    None,
                    "cycle",
                    u128::from(dram.maximum_terminal_l2_dirty_sectors)
                        .saturating_mul(32)
                        .saturating_mul(u128::from(hz)),
                    u128::from(
                        card.require_dram_bandwidth()
                            .ok()
                            .and_then(|bandwidth| bandwidth.validated_peak_bytes_per_second().ok())
                            .unwrap_or(1),
                    ),
                    true,
                    "serialized terminal-dirty byte service demand only; excludes queue position, coherence and profiler flush visibility",
                ),
                None => Metric::unavailable(
                    None,
                    "cycle",
                    "requires the carded SM clock and DRAM service-rate hypothesis",
                ),
            },
        );
        m.insert(
            "dirty_eviction_sectors".into(),
            Metric::count(
                None,
                "sector",
                u128::from(dram.l2_dirty_evicted_sectors),
                "eviction demand only; not scheduled DRAM write traffic",
            ),
        );
        m.insert(
            "dirty_eviction_bytes".into(),
            Metric::count(
                None,
                "byte",
                u128::from(dram.l2_dirty_evicted_sectors) * 32,
                "dirty L2 eviction demand only; not scheduled DRAM write traffic",
            ),
        );
        m.insert(
            "writeback_service_cycles_estimate".into(),
            match clock {
                Some(hz) => Metric::ratio(
                    None,
                    "cycle",
                    u128::from(dram.l2_dirty_evicted_sectors)
                        .saturating_mul(32)
                        .saturating_mul(u128::from(hz)),
                    u128::from(
                        card.require_dram_bandwidth()
                            .ok()
                            .and_then(|bandwidth| bandwidth.validated_peak_bytes_per_second().ok())
                            .unwrap_or(1),
                    ),
                    true,
                    "serialized dirty-eviction byte service demand only; excludes queue position, write-allocate policy, coherence and flush visibility",
                ),
                None => Metric::unavailable(
                    None,
                    "cycle",
                    "requires the carded SM clock and DRAM service-rate hypothesis",
                ),
            },
        );
    }
    // Keep the read-only projection schema stable in flat/L1 modes without
    // overwriting the approximate values produced by the DRAM-observer path.
    ensure_unavailable(
        &mut m,
        "dram_read_bytes_per_second_estimate",
        None,
        "byte/s",
        "requires shared-L2 DRAM observation and a modeled elapsed frontier; this is not total profiler DRAM throughput",
    );
    ensure_unavailable(
        &mut m,
        "dram_read_bandwidth_pct_estimate",
        None,
        "%",
        "requires shared-L2 DRAM observation, a modeled elapsed frontier, and validated card peak bandwidth; this is not NCU achieved bandwidth",
    );
    for (key, ncu, unit, reason) in [
        (
            "dram_write_sectors",
            Some("dram__sectors_write.sum"),
            "sector",
            "dirty writeback transactions and profiler flush boundary not modeled",
        ),
        (
            "dram_bytes_per_second",
            Some("dram__bytes.sum.per_second"),
            "byte/s",
            "write traffic incomplete; do not substitute read-only bandwidth",
        ),
        (
            "l2_requests",
            Some("lts__t_requests_srcunit_tex.sum"),
            "request",
            "sector observer is not a packet/request counter",
        ),
        (
            "power_watts",
            None,
            "W",
            "no independently characterized SM86 power model",
        ),
        (
            "energy_joules",
            None,
            "J",
            "no independently characterized SM86 energy model",
        ),
        (
            "sm_throughput_pct",
            Some("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "%",
            "NCU composite maximum of multiple pipeline rates; full pipe activity not modeled",
        ),
        (
            "memory_throughput_pct",
            Some("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
            "%",
            "NCU composite rate lacks complete component activity/peak definitions",
        ),
    ] {
        m.insert(key.into(), Metric::unavailable(ncu, unit, reason));
    }
    // Elapsed estimates retain every original timing frontier and tail flag.
    // This does not claim complete retirement or select a frontier using NCU.
    let gpc = result.gpc_observation_frontier();
    m.insert(
        "gpc_cycles_max".into(),
        match gpc {
            Some(frontier) => Metric::approximate_count(
                Some("gpc__cycles_elapsed.max"),
                "cycle",
                u128::from(frontier.cycles),
                "architecture-scoped GPC observation frontier; completion tails remain separate",
            ),
            None => Metric::unavailable(
                Some("gpc__cycles_elapsed.max"),
                "cycle",
                "no architecture-scoped GPC frontier in this replay mode",
            ),
        },
    );
    m.insert("gpc_elapsed_cycles_estimate".into(), match gpc {
        Some(f) => Metric::ratio(Some("gpc__cycles_elapsed.max"), "cycle", u128::from(f.cycles), 1, true, "architecture launch envelope plus modeled terminal frontier; known tails remain in timing.completion_frontier"),
        None => Metric::unavailable(Some("gpc__cycles_elapsed.max"), "cycle", "no architecture-scoped launch frontier in this mode"),
    });
    m.insert("duration_ns_estimate".into(), match (gpc, clock) {
        (Some(f), Some(hz)) => Metric::ratio(Some("gpu__time_duration.sum"), "ns", u128::from(f.cycles)*1_000_000_000, u128::from(hz), true, "assumes card clock fixed; inherits GPC launch-envelope approximation, not full retirement"),
        _ => Metric::unavailable(Some("gpu__time_duration.sum"), "ns", "needs launch frontier and card clock"),
    });
    m.insert("ipc_per_sm_elapsed_estimate".into(), match gpc {
        Some(f) => Metric::ratio(Some("sm__inst_executed.avg.per_cycle_elapsed"), "inst/cycle", inst, u128::from(f.cycles)*u128::from(sm_count), true, "uses modeled GPC maximum as common SM elapsed interval; hardware avg/max skew is not modeled"),
        _ => Metric::unavailable(Some("sm__inst_executed.avg.per_cycle_elapsed"), "inst/cycle", "no elapsed frontier"),
    });
    Ok(NativeMetrics {
        schema_version: 1,
        metric_surface_version: 2,
        architecture: "sm_86",
        sm_count,
        active_sm_count,
        reference_clock_hz: clock,
        metrics: m,
    })
}
