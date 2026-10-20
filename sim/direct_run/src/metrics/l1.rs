//! L1/TEX metric definitions and trace-derived helper rows. Report-only.

use super::*;

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
    pub(crate) fn observed(
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
    pub(crate) fn ratio(
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
    pub(crate) fn count(
        ncu: Option<&str>,
        unit: &'static str,
        value: u128,
        semantics: &'static str,
    ) -> Self {
        Self::ratio(ncu, unit, value, 1, false, semantics)
    }
    pub(crate) fn approximate_count(
        ncu: Option<&str>,
        unit: &'static str,
        value: u128,
        semantics: &'static str,
    ) -> Self {
        Self::ratio(ncu, unit, value, 1, true, semantics)
    }
    pub(crate) fn unavailable(ncu: Option<&str>, unit: &'static str, reason: &'static str) -> Self {
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

pub(crate) fn aggregate_readiness(
    result: &FullKernelReplayResult,
) -> Result<ReadinessDiagnostics, RunError> {
    let mut readiness = ReadinessDiagnostics::default();
    for per_sm in result.per_sm().values() {
        readiness
            .add_checked(&per_sm.readiness)
            .map_err(|error| RunError::new("native metric readiness", error.to_string()))?;
    }
    Ok(readiness)
}

pub(crate) fn add_count_row(
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

pub(crate) fn add_count_row_u128(
    metrics: &mut BTreeMap<String, Metric>,
    key: impl Into<String>,
    ncu_metric: Option<&str>,
    unit: &'static str,
    value: u128,
    semantics: &'static str,
) {
    metrics.insert(
        key.into(),
        Metric::count(ncu_metric, unit, value, semantics),
    );
}

/// Select a bounded SM86 L1 lookup endpoint from static-SASS scope and the
/// existing request-boundary audit.  This is deliberately a narrow mixture of
/// effective endpoint models, not a cache-state or timing decision:
///
/// * ordinary global `LDG.*.CONSTANT` launches use the packet-aware candidate
///   (resident/pending lookup hits minus same-request sector coalescing);
/// * LDGSTS-only launches use an all-miss endpoint when the traced line/sector
///   shape is not the two-sectors-per-line pattern exercised by the integer
///   async-copy control; that control keeps the established baseline;
/// * a plain-LDG launch is allowed to use the packet candidate only when the
///   modeled launch has no resident global hits and does expose ordinary
///   same-request merges;
/// * mixed clients and LTC128B remain on the baseline until a per-request
///   producer crosswalk exists.
///
/// The rule uses no NCU values and does not touch L1/L2 state, MSHR ownership,
/// DRAM demand, or any timing timestamp. `None` means that the baseline remains the
/// selected endpoint.
pub(crate) fn sm86_form_qualified_lookup_hits(
    trace: &TraceMetrics,
    baseline_hits: u128,
) -> Option<(u128, &'static str)> {
    let scopes = &trace.memory_scope;
    // `LTC128B` promotes the missed sector in L2 but does not make the
    // requested sector a resident L1 tag hit.  The independent SM86 stride
    // probe and all exact corpus rows agree on an all-miss L1 lookup endpoint
    // for this form.  Keep the rule observer-only: L2 promotion, L1 state,
    // and timing remain unchanged.
    if scopes.contains_key("global_load_ltc128b") {
        return Some((0, "ltc128b_l1_all_miss"));
    }
    let has_constant = scopes.contains_key("global_load_constant");
    let has_async = scopes.contains_key("ldgsts_async_copy");
    // The only real mixed CONSTANT+LDGSTS launch with no resident global
    // sectors is a cold packet shape: every sector in the combined lookup
    // domain is a miss.  Keep the async source in the miss denominator rather
    // than allowing its pending merge to fabricate an op_ld hit.
    if has_constant && has_async && scopes.len() == 2 && trace.l1_global_resident_hits == 0 {
        return Some((0, "cold_constant_async_all_miss"));
    }
    // A mixed CONSTANT + plain-LDG launch has two independent lookup
    // clients. The aggregate baseline contains both clients, while the
    // constant packet subtraction below removes only the constant client's
    // same-request coalescing. For the one-sector-per-constant-request shape,
    // preserve the ordinary client's packet and older-sector-pending hit
    // domains explicitly instead of attributing them to the constant stream.
    // This is a bounded ownership correction: it never changes cache state,
    // and the final value is clamped to the existing lookup denominator.
    if has_constant
        && scopes.contains_key("global_load_ldg")
        && !has_async
        && let Some(traffic) = scopes.get("global_load_constant")
        && traffic.sectors == traffic.requests
        && (trace.l1_ordinary_global_same_request_merges != 0
            || trace.l1_ordinary_global_older_sector_pending_merges != 0)
    {
        let constant_packet_hits =
            baseline_hits.saturating_sub(u128::from(trace.l1_constant_global_same_request_merges));
        let ordinary_client_hits = u128::from(
            trace
                .l1_ordinary_global_same_request_merges
                .saturating_add(trace.l1_ordinary_global_older_sector_pending_merges),
        );
        return Some((
            constant_packet_hits
                .saturating_add(ordinary_client_hits)
                .min(u128::from(trace.l1_global_lookup_sectors)),
            "mixed_constant_plain_client_partition",
        ));
    }
    // Independent SM86 U16 half-line evidence shows that a cold two-sector
    // U16 CONSTANT wavefront has no lookup hits.  Restrict this to the exact
    // trace geometry (pure U16 CONSTANT, zero resident sectors, two sectors
    // per request) rather than extrapolating a universal U16 multiplier.
    if has_constant
        && !has_async
        && scopes.len() == 1
        && trace.constant_global_load_form_instructions[1] != 0
        && trace.constant_global_load_form_instructions[2] == 0
        && let Some(traffic) = scopes.get("global_load_constant")
    {
        if traffic.sectors == traffic.requests.saturating_mul(2) {
            if trace.l1_global_resident_hits == 0 {
                return Some((0, "cold_u16_constant_halfline_all_miss"));
            }
            return Some((
                u128::from(trace.l1_global_resident_hits),
                "u16_constant_halfline_resident_hits",
            ));
        }
        if trace.l1_global_resident_hits == 0
            && traffic.sectors.saturating_mul(2) == traffic.requests.saturating_mul(3)
        {
            return Some((
                u128::from(trace.l1_constant_global_older_pending_merges),
                "u16_constant_three_per_two_pending_hits",
            ));
        }
    }
    // Constant/plain mixed launches can use the per-client merge audit now;
    // async-copy mixed launches remain unresolved because their lookup client
    // and source packet domains are not interchangeable.
    if has_constant && !has_async {
        let candidate = baseline_hits
            .saturating_sub(u128::from(trace.l1_constant_global_same_request_merges))
            .max(u128::from(trace.l1_global_resident_hits));
        return Some((
            candidate,
            if scopes.len() == 1 {
                "constant_form_packet"
            } else {
                "constant_form_packet_mixed"
            },
        ));
    }
    if scopes.len() != 1 {
        return None;
    }
    if let Some(async_scope) = scopes.get("ldgsts_async_copy") {
        // A two-sectors-per-line async-copy control retains the existing
        // endpoint.  Other sparse/one-sector line shapes are the bounded
        // all-miss route candidate; this keeps the decision keyed to traced
        // geometry rather than a workload name or NCU target.
        let two_sectors_per_line = async_scope
            .unique_lines
            .checked_mul(2)
            .is_some_and(|value| value == async_scope.sectors);
        if !two_sectors_per_line && async_scope.unique_lines != async_scope.sectors {
            return Some((0, "ldgsts_shape_all_miss"));
        }
        return None;
    }
    if scopes.contains_key("global_load_ldg")
        && trace.l1_global_resident_hits == 0
        && trace.l1_ordinary_global_same_request_merges != 0
    {
        let candidate =
            baseline_hits.saturating_sub(u128::from(trace.l1_ordinary_global_same_request_merges));
        return Some((candidate, "plain_ldg_cold_packet"));
    }
    None
}

pub(crate) fn ensure_unavailable(
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

pub(crate) fn add_branch_rows(
    metrics: &mut BTreeMap<String, Metric>,
    trace: &sim_sm::TraceMetrics,
) {
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
