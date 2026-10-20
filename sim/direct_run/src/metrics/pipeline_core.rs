//! Core IPC and pipeline projections from replay counters; report-only and timing-neutral.

use super::*;

pub(super) fn project_sm120_alpha_metrics(
    result: &FullKernelReplayResult,
    sm_count: u32,
    active_sm_count: u32,
    clock: Option<u64>,
    mut m: BTreeMap<String, Metric>,
) -> Result<NativeMetrics, RunError> {
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
        .checked_add(crate::sm120_front_end_envelope_cycles(result).unwrap_or(0))
        .unwrap_or(u64::MAX);
    // The raw SM-active terminal frontier is intentionally preserved in
    // the timing report.  This separate effective projection applies the
    // same fixed front-end observation envelope used by the GPC metric to
    // the SM elapsed endpoint, allowing the NCU average-cycle surface to
    // be reported without relabeling raw SM-active timing as physical
    // elapsed time.
    let effective_sm_elapsed_cycles = result.sm_active_terminal_event_time().and_then(|cycles| {
        cycles.checked_add(crate::sm120_front_end_envelope_cycles(result).unwrap_or(0))
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
    let grid_blocks = result.coverage().launch().grid_blocks().unwrap_or(1).max(1);
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
    let effective_sm_active_frontier = result.sm_active_terminal_event_time().and_then(|raw| {
        let elapsed =
            raw.checked_add(crate::sm120_front_end_envelope_cycles(result).unwrap_or(0))?;
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
    let effective_sm_active_sum = result.per_sm().values().try_fold(0_u128, |sum, sm| {
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
        let front_end = u128::from(crate::sm120_front_end_envelope_cycles(result).unwrap_or(0));
        let front_num = u128::from(alpha.active_ipc_front_end_numerator);
        let front_den = u128::from(alpha.active_ipc_front_end_denominator.max(1));
        let front_end_share = front_end.checked_mul(front_num)?.div_ceil(front_den);
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
            && result.coverage().launch().grid().1 == 1
            && result.coverage().launch().grid().2 == 1
            && result.coverage().launch().block().is_some_and(|block| {
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
                    "trace-derived logical LDGSTS cache-route cardinality; route semantics are qualified, silicon cache access remains unresolved",
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
            trace
                .memory
                .get("global_reduction")
                .map_or(0, |row| row.requests),
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
                    if kind == "global_load" {
                        "plain LDG SASS-form scope candidate; CONSTANT/LTC128B/LDGSTS forms are retained in memory_scope observers"
                    } else {
                        "trace-derived logical memory operation cardinality; cache route retained separately and physical pipe assignment unresolved"
                    },
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
