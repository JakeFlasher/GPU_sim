// L1/L1TEX counter attribution helpers; report-only, no timing/cache mutation.

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

fn add_count_row_u128(
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
fn sm86_form_qualified_lookup_hits(
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

