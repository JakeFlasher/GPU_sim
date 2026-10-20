//! SM86 completion-ledger projections; report-only and timing-neutral.

use super::*;

impl<'a> Sm86MetricState<'a> {
    pub(crate) fn add_sm86_memory(&mut self) -> Result<(), RunError> {
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
            let mut sectors: u128 = classes
                .iter()
                .filter_map(|c| trace.memory.get(*c))
                .map(|r| u128::from(r.sectors))
                .sum();
            // For the two real dynamic-index local surfaces, the trace carries
            // stack-relative offsets that collapse lane-private slots.  Compose
            // the captured 256-byte frame expansion with the transfer-width
            // expansion as a bounded coalescing correction:
            //   raw + (frame256 - width_stride)
            // The raw term preserves observed coalescing, while the difference is
            // only the additional lane-private address domain.  Scalar volatile
            // and register-spill records have one sector/request and therefore
            // remain on their exact raw/payload endpoint.
            if key == "local_load" {
                let raw_requests = trace
                    .memory
                    .get("local_load")
                    .map_or(0_u128, |r| u128::from(r.requests));
                if raw_requests != 0
                    && sectors > raw_requests
                    && trace.local_frame256_load_sectors != 0
                {
                    sectors = u128::from(trace.local_frame256_load_sectors)
                        .saturating_sub(u128::from(trace.local_canonical_load_sectors))
                        .saturating_add(sectors);
                } else if trace.local_payload_load_sectors != 0 {
                    sectors = u128::from(trace.local_payload_load_sectors);
                }
            } else if key == "local_store" {
                let raw_requests = trace
                    .memory
                    .get("local_store")
                    .map_or(0_u128, |r| u128::from(r.requests));
                if raw_requests != 0
                    && sectors > raw_requests
                    && trace.local_frame256_store_sectors != 0
                {
                    sectors = u128::from(trace.local_frame256_store_sectors)
                        .saturating_sub(u128::from(trace.local_canonical_store_sectors))
                        .saturating_add(sectors);
                } else if trace.local_payload_store_sectors != 0 {
                    // Scalar/volatile and register-spill local records retain
                    // the validated payload candidate; unlike dynamic-index
                    // records they do not need frame expansion.
                    sectors = u128::from(trace.local_payload_store_sectors);
                }
            }
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
                        Metric::unavailable(
                            None,
                            unit,
                            "flat replay does not decode sector topology",
                        )
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
        let global_write_class_requests = atomic_requests
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

        Ok(())
    }
}
