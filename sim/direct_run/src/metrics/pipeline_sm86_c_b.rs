//! SM86 residual pipeline projections; report-only and timing-neutral.

use super::*;

impl<'a> Sm86MetricState<'a> {
    pub(crate) fn add_sm86_tail(&mut self) -> Result<(), RunError> {
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
            add_count_row(
                &mut m,
                "l1_mshr_stall_occupancy_samples",
                None,
                "sample",
                l1.mshr_stall_occupancy_samples,
                "unified samples taken when a miss found the finite L1 MSHR set full; observer only",
            );
            add_count_row(
                &mut m,
                "l1_mshr_stall_occupancy_sum",
                None,
                "entry-sample",
                l1.mshr_stall_occupancy_sum,
                "sum of L1 MSHR entries in use at full-set stall samples; not an MSHR capacity claim",
            );
            m.insert(
                "l1_mshr_stall_occupancy_mean".into(),
                Metric::ratio(
                    None,
                    "entry",
                    u128::from(l1.mshr_stall_occupancy_sum),
                    u128::from(l1.mshr_stall_occupancy_samples),
                    false,
                    "mean L1 MSHR occupancy at full-set stall samples; observer only",
                ),
            );
            let [store_hits, reduction_hits] =
                crate::emit::l1_write_hit_terms(result.arch(), l1, Some(result.trace_metrics()));
            // Keep the SM86 unified-L1 native percentage on the same
            // architecture-scoped lookup basis as the already-approved global
            // lookup report. SM120 exits this native-metrics path earlier; this
            // correction therefore preserves its existing output while bringing
            // SM86 form-qualified LDG.CONSTANT/U16/LTC128B accounting into the
            // scored unified endpoint. Replay state and timing are untouched.
            let global = l1.origin(MemoryOrigin::Global);
            let local = l1.origin(MemoryOrigin::Local);
            let global_baseline = u128::from(global.effective_lookup_hits_for_arch(result.arch()));
            let global_lookup_hits = if result.arch() == ArchId::Sm86 {
                sm86_form_qualified_lookup_hits(trace, global_baseline)
                    .map_or(global_baseline, |(hits, _)| hits)
            } else {
                global_baseline.saturating_sub(u128::from(async_global_l1_hit_exclusion(
                    result.arch(),
                    l1,
                    Some(trace),
                )))
            };
            let local_lookup_hits = u128::from(local.effective_lookup_hits_for_arch(result.arch()));
            let unified_hit_num = global_lookup_hits
                .checked_add(local_lookup_hits)
                .and_then(|value| value.checked_add(u128::from(store_hits)))
                .and_then(|value| value.checked_add(u128::from(reduction_hits)))
                .ok_or_else(|| {
                    RunError::new("native metric L1 unified", "hit numerator overflow")
                })?;
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
                // NCU's lookup rows observe the L1 tag result, not only the
                // timing cache's resident-data result.  Include bounded pending
                // merge provenance through the L1 observer projection; cache
                // state, MSHR ownership, and response timing remain unchanged.
                let baseline_hits = u128::from(c.effective_lookup_hits_for_arch(result.arch()));
                let baseline_total = u128::from(c.hits)
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
                let baseline_lookup_misses =
                    baseline_total.checked_sub(baseline_hits).ok_or_else(|| {
                        RunError::new(
                            "native metric L1 lookup",
                            format!("{name} lookup hits exceed lookup denominator"),
                        )
                    })?;
                let local_dynamic_candidate = if origin == MemoryOrigin::Local {
                    let raw = trace.memory.get("local_load");
                    let requests = raw.map_or(0, |r| r.requests);
                    let sectors = raw.map_or(0, |r| r.sectors);
                    if requests != 0 && sectors > requests && trace.local_frame256_load_sectors != 0
                    {
                        Some(
                            u128::from(trace.local_frame256_load_sectors)
                                .saturating_sub(u128::from(trace.local_canonical_load_sectors))
                                .saturating_add(u128::from(sectors)),
                        )
                    } else if trace.local_payload_load_sectors != 0
                        && requests != 0
                        && sectors <= requests
                    {
                        Some(u128::from(trace.local_payload_load_sectors))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let (hits, selected_reason) = if origin == MemoryOrigin::Global {
                    sm86_form_qualified_lookup_hits(trace, baseline_hits)
                        .unwrap_or((baseline_hits, "baseline"))
                } else if let Some(candidate) = local_dynamic_candidate {
                    (candidate, "local_frame_coalesced")
                } else {
                    (baseline_hits, "baseline")
                };
                let total = local_dynamic_candidate.unwrap_or(baseline_total);
                let lookup_misses = total.checked_sub(hits).ok_or_else(|| {
                    RunError::new(
                        "native metric L1 lookup",
                        format!("{name} selected lookup hits exceed lookup denominator"),
                    )
                })?;
                if origin == MemoryOrigin::Global {
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_hits_baseline",
                        None,
                        "sector",
                        baseline_hits,
                        "SM86 baseline L1 lookup-hit projection retained beside the form-qualified selected endpoint",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_misses_baseline",
                        None,
                        "sector",
                        baseline_lookup_misses,
                        "SM86 baseline L1 lookup-miss projection retained beside the form-qualified selected endpoint",
                    );
                    add_count_row(
                        &mut m,
                        "l1_global_load_lookup_selection_form_qualified",
                        None,
                        "selection",
                        u64::from(selected_reason != "baseline"),
                        "trace-only SM86 form/client endpoint selection; zero means the baseline remains selected",
                    );
                    // Candidate only: ordinary same-request merges are the
                    // packet/coalescing domain that the baseline projection
                    // cannot distinguish from older pending-line merges.  Keep
                    // the baseline NCU-shaped row above unchanged; expose this
                    // bounded subtraction for held-out comparison and interval
                    // reporting without touching cache state or timing.
                    let same_request = trace.l1_ordinary_same_request_merges_global;
                    let packet_hits = baseline_hits.saturating_sub(u128::from(same_request));
                    let packet_misses = total.checked_sub(packet_hits).ok_or_else(|| {
                        RunError::new(
                            "native metric L1 packet-aware lookup",
                            "packet-aware lookup hits exceed lookup denominator",
                        )
                    })?;
                    add_count_row(
                        &mut m,
                        "l1_global_load_same_request_merge_sectors",
                        None,
                        "sector",
                        same_request,
                        "ordinary global same-request sector coalescing observer; not an NCU lookup prediction",
                    );
                    add_count_row(
                        &mut m,
                        "l1_global_unrequested_subline_miss_sectors",
                        None,
                        "sector",
                        trace.l1_unrequested_subline_miss_sectors,
                        "observer-only disjoint-sector misses on a 128B line; no endpoint or timing selection",
                    );
                    add_count_row(
                        &mut m,
                        "l1_global_wavefront_mask_lookup_hits_candidate",
                        None,
                        "sector",
                        trace.l1_wavefront_mask_lookup_hits,
                        "observer-only resident-sector wavefront lookup candidate; not selected for the H108 endpoint",
                    );
                    add_count_row(
                        &mut m,
                        "l1_global_wavefront_mask_lookup_misses_candidate",
                        None,
                        "sector",
                        trace.l1_wavefront_mask_lookup_misses,
                        "observer-only resident-sector wavefront lookup candidate; not selected for the H108 endpoint",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_hits_packet_aware_candidate",
                        None,
                        "sector",
                        packet_hits,
                        "bounded packet-aware candidate: baseline lookup hits minus ordinary same-request sector merges; observer-only",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_misses_packet_aware_candidate",
                        None,
                        "sector",
                        packet_misses,
                        "bounded packet-aware candidate denominator minus candidate hits; observer-only",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_hits_packet_aware_interval_low",
                        None,
                        "sector",
                        packet_hits,
                        "lower edge of packet-aware lookup-hit interval; same-request merges treated as misses",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_hits_packet_aware_interval_high",
                        None,
                        "sector",
                        baseline_hits,
                        "upper edge of packet-aware lookup-hit interval; baseline merge projection",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_misses_packet_aware_interval_low",
                        None,
                        "sector",
                        baseline_lookup_misses,
                        "lower edge of packet-aware lookup-miss interval; baseline merge projection",
                    );
                    add_count_row_u128(
                        &mut m,
                        "l1_global_load_lookup_misses_packet_aware_interval_high",
                        None,
                        "sector",
                        packet_misses,
                        "upper edge of packet-aware lookup-miss interval; same-request merges treated as misses",
                    );
                }
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
                    "selected SM86 form-qualified L1 tag-lookup projection; baseline is retained when per-request client/form identity is unresolved",
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
                    "selected SM86 form-qualified L1 lookup-miss projection; denominator conserved with the selected hit endpoint",
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
            // SM86's L2 source split retains operation-class packetization for
            // L1-fill, bypass, untagged and store traffic.  The aggregate L2
            // observer counters predate that split and can omit write-class
            // lookup hits (for example cp.async's shared-destination stores).
            // Prefer the conserving source split for the NCU-shaped TEX-sector
            // surface while retaining the raw cache counters elsewhere.
            let packetized_store = true;
            let source_lookup_requests = result
                .l2_by_source()
                .map(|split| {
                    // H80/H81 already select an effective packet-sector and
                    // packet-hit endpoint for operation-qualified writes. Reuse
                    // that same bounded candidate in the aggregate TEX rows so
                    // additive operation-class conservation holds. The raw
                    // one-sector-per-request store totals remain diagnostic.
                    let store_requests = if packetized_store {
                        split.store_packet_candidate_sectors
                    } else {
                        split.store_lookup_requests
                    };
                    split
                        .read_lookup_requests
                        .checked_add(store_requests)
                        .and_then(|value| value.checked_add(split.atomic_lookup_requests))
                        .and_then(|value| value.checked_add(split.reduction_lookup_requests))
                        .ok_or_else(|| {
                            RunError::new("metric L2 lookup", "request counter overflow")
                        })
                })
                .transpose()?;
            let source_lookup_hits = result
                .l2_by_source()
                .map(|split| {
                    let store_hits = if packetized_store {
                        split
                            .store_lookup_hits
                            .min(split.store_packet_candidate_sectors)
                    } else {
                        split.store_lookup_hits
                    };
                    split
                        .read_lookup_hits
                        .checked_add(store_hits)
                        .and_then(|value| value.checked_add(split.atomic_lookup_hits))
                        .and_then(|value| value.checked_add(split.reduction_lookup_hits))
                        .ok_or_else(|| RunError::new("metric L2 lookup", "hit counter overflow"))
                })
                .transpose()?;
            let lookup_requests = source_lookup_requests.unwrap_or(l2.lookup_requests);
            let lookup_hits = source_lookup_hits.unwrap_or(l2.lookup_hits);
            if lookup_hits > lookup_requests {
                return Err(RunError::new(
                    "metric L2 lookup",
                    "source lookup hits exceed source lookup requests",
                ));
            }
            if let Some(split) = result.l2_by_source() {
                add_count_row(
                    &mut m,
                    "l2_tex_lookup_sectors_request_baseline",
                    None,
                    "sector",
                    split
                        .read_lookup_requests
                        .checked_add(split.store_lookup_requests)
                        .and_then(|value| value.checked_add(split.atomic_lookup_requests))
                        .and_then(|value| value.checked_add(split.reduction_lookup_requests))
                        .ok_or_else(|| {
                            RunError::new("metric L2 lookup", "baseline request overflow")
                        })?,
                    "SM86 aggregate TEX lookup request baseline before H80/H81 write-packet conservation",
                );
                add_count_row(
                    &mut m,
                    "l2_tex_lookup_hits_request_baseline",
                    None,
                    "sector",
                    split
                        .read_lookup_hits
                        .checked_add(split.store_lookup_hits)
                        .and_then(|value| value.checked_add(split.atomic_lookup_hits))
                        .and_then(|value| value.checked_add(split.reduction_lookup_hits))
                        .ok_or_else(|| {
                            RunError::new("metric L2 lookup", "baseline hit overflow")
                        })?,
                    "SM86 aggregate TEX lookup-hit baseline before H80/H81 write-packet conservation",
                );
            }
            m.insert(
            "l2_tex_lookup_sectors".into(),
            Metric::count(
                Some("lts__t_sectors_srcunit_tex.sum"),
                "sector",
                u128::from(lookup_requests),
                "conserving SM86 source-sector lookup observer; includes write-class source lanes and is not an LTS request count",
            ),
        );
            m.insert(
                "l2_tex_lookup_hits".into(),
                Metric::count(
                    Some("lts__t_sectors_srcunit_tex_lookup_hit.sum"),
                    "sector",
                    u128::from(lookup_hits),
                    "conserving SM86 source-sector lookup-hit observer",
                ),
            );
            m.insert(
            "l2_tex_lookup_misses".into(),
            Metric::count(
                Some("lts__t_sectors_srcunit_tex_lookup_miss.sum"),
                "sector",
                lookup_requests.checked_sub(lookup_hits).ok_or_else(|| {
                    RunError::new("metric L2 lookup", "lookup hit/miss subtraction underflow")
                })?.into(),
                "conserving SM86 source-sector lookup-miss observer; ordinary write sectors use the conditional all-write-hit policy",
            ),
        );
            m.insert(
            "l2_tex_hit_rate_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(lookup_hits) * 100,
                u128::from(lookup_requests),
                false,
                "TEX-source lookup hit rate over the conserving SM86 source-sector observer; the generic lts__t_sector_hit_rate.pct endpoint has broader/driver-version-dependent scope and is not silently joined",
            ),
        );
            if let Some(split) = result.l2_by_source() {
                m.insert(
                "l2_op_write_packet_coalesced_candidate_sectors".into(),
                Metric::count(
                    None,
                    "sector",
                    u128::from(split.store_packet_candidate_sectors),
                    "observer-only per-SM line/sector write packet candidate within an 8-cycle admission window; baseline operation sectors and timing are unchanged",
                ),
            );
                // Operation-split L2 source observers. These are sourced directly
                // from the event request construction seam; no write-lookup or
                // atomic policy is inferred from the aggregate counter.
                for (name, ncu_metric, requests, hits) in [
                    (
                        "read",
                        "lts__t_sectors_srcunit_tex_op_read.sum",
                        split.read_lookup_requests,
                        split.read_lookup_hits,
                    ),
                    (
                        "write",
                        "lts__t_sectors_srcunit_tex_op_write.sum",
                        split.store_lookup_requests,
                        split.store_lookup_hits,
                    ),
                    (
                        "atom_dot_alu",
                        "lts__t_sectors_srcunit_tex_op_atom_dot_alu.sum",
                        split.atomic_lookup_requests,
                        split.atomic_lookup_hits,
                    ),
                    (
                        "red",
                        "lts__t_sectors_srcunit_tex_op_red.sum",
                        split.reduction_lookup_requests,
                        split.reduction_lookup_hits,
                    ),
                ] {
                    let sector_value = if name == "write" {
                        split.store_packet_candidate_sectors
                    } else {
                        requests
                    };
                    let reported_hits = if name == "write" {
                        // The H80 packet candidate is the only independently
                        // supported write lookup endpoint. H75 showed that a
                        // physical lookup-hit count need not equal packet count,
                        // so this remains an effective source-operation model,
                        // not a physical tag claim.
                        m.insert(
                            "l2_op_write_lookup_hits_request_baseline".into(),
                            Metric::count(
                                None,
                                "sector",
                                u128::from(hits),
                                "baseline one-sector-per-request write lookup-hit observer",
                            ),
                        );
                        hits.min(sector_value)
                    } else {
                        hits
                    };
                    let misses = sector_value.checked_sub(reported_hits).ok_or_else(|| {
                        RunError::new("metric L2 operation split", "lookup hits exceed requests")
                    })?;
                    if name == "write" {
                        // The operation-sector endpoint is packetized at the
                        // L1TEX→XBAR boundary. Promote only the independently
                        // observed write-packet candidate; preserve the old
                        // one-sector-per-request value under a diagnostic key.
                        m.insert(
                        "l2_op_write_sectors_request_baseline".into(),
                        Metric::count(
                            None,
                            "sector",
                            u128::from(requests),
                            "baseline one-sector-per-request write observer retained for differential comparison",
                        ),
                    );
                        m.insert(
                            "l2_op_write_lookup_misses_request_baseline".into(),
                            Metric::count(
                                None,
                                "sector",
                                u128::from(requests.checked_sub(hits).ok_or_else(|| {
                                    RunError::new(
                                        "metric L2 operation split",
                                        "baseline write lookup hits exceed requests",
                                    )
                                })?),
                                "baseline one-sector-per-request write lookup-miss observer",
                            ),
                        );
                    }
                    let sector_semantics = if name == "write" {
                        "bounded per-SM line/sector write-packet candidate within an 8-cycle admission window; L2 state, lookup-hit/miss rows, DRAM demand, and timing remain unchanged"
                    } else {
                        "observer-only operation-split L2 TEX source sectors"
                    };
                    m.insert(
                        format!("l2_op_{name}_sectors"),
                        Metric::count(
                            Some(ncu_metric),
                            "sector",
                            u128::from(sector_value),
                            sector_semantics,
                        ),
                    );
                    m.insert(
                        format!("l2_op_{name}_lookup_hits"),
                        Metric::count(
                            Some(&format!(
                                "lts__t_sectors_srcunit_tex_op_{name}_lookup_hit.sum"
                            )),
                            "sector",
                            u128::from(reported_hits),
                            "observer-only operation-split L2 lookup hits",
                        ),
                    );
                    m.insert(
                        format!("l2_op_{name}_lookup_misses"),
                        Metric::count(
                            Some(&format!(
                                "lts__t_sectors_srcunit_tex_op_{name}_lookup_miss.sum"
                            )),
                            "sector",
                            u128::from(misses),
                            "observer-only operation-split L2 lookup misses",
                        ),
                    );
                }
                for (source, name) in [
                    (sim_mem::L2Source::L1Fill, "l1_fill"),
                    (sim_mem::L2Source::Bypass, "bypass"),
                    (sim_mem::L2Source::Untagged, "untagged"),
                    (sim_mem::L2Source::Store, "store"),
                ] {
                    let index = source as usize;
                    let samples = split.pending_pressure_samples[index];
                    let sum = split.pending_pressure_sum[index];
                    m.insert(
                    format!("l2_{name}_pending_pressure_samples"),
                    Metric::count(
                        None,
                        "sample",
                        u128::from(samples),
                        "source-qualified pending-L2-sector pressure samples before event requests; observer only",
                    ),
                );
                    m.insert(
                    format!("l2_{name}_pending_pressure_sum"),
                    Metric::count(
                        None,
                        "sector-sample",
                        u128::from(sum),
                        "source-qualified sum of pending L2 sectors observed before event requests; not a capacity claim",
                    ),
                );
                    m.insert(
                    format!("l2_{name}_pending_pressure_mean"),
                    Metric::ratio(
                        None,
                        "sector",
                        u128::from(sum),
                        u128::from(samples),
                        false,
                        "source-qualified mean pending L2 sectors per event-request sample; observer only",
                    ),
                );
                }
            }
            let fill_ownership_total = l2
                .pending_merge_sectors
                .checked_add(l2.new_fill_sectors)
                .ok_or_else(|| {
                    RunError::new(
                        "metric L2 fill ownership",
                        "pending-merge/new-fill observer total overflows",
                    )
                })?;
            m.insert(
                "l2_pending_merge_sectors".into(),
                Metric::count(
                    None,
                    "sector",
                    u128::from(l2.pending_merge_sectors),
                    "event-mode sectors merged onto an already-pending L2 fill; observer only",
                ),
            );
            m.insert(
                "l2_new_fill_sectors".into(),
                Metric::count(
                    None,
                    "sector",
                    u128::from(l2.new_fill_sectors),
                    "event-mode sectors owning a new L2 fill capability; observer only",
                ),
            );
            m.insert(
            "l2_pending_merge_fraction_pct".into(),
            Metric::ratio(
                None,
                "%",
                u128::from(l2.pending_merge_sectors) * 100,
                u128::from(fill_ownership_total),
                false,
                "pending-fill sectors divided by new-fill plus pending-merge sectors; bounded coalescing observer, not a timing fit",
            ),
        );
            // Effective slice-routing observer. The active card exposes six
            // slices, while the physical GA102 hash remains unknown; these rows
            // report only the selected effective hash's request distribution and
            // never feed routing or timing decisions back into replay.
            for (slice, count) in l2.slice_accesses.iter().enumerate() {
                if *count != 0 {
                    m.insert(
                    format!("l2_slice_{slice}_lookup_sectors"),
                    Metric::count(
                        None,
                        "sector",
                        u128::from(*count),
                        "effective L2 slice-routing request observer; not a proprietary GA102 hash claim",
                    ),
                );
                }
            }
            let active_slice_counts = l2
                .slice_accesses
                .iter()
                .copied()
                .filter(|count| *count != 0)
                .collect::<Vec<_>>();
            if let (Some(&minimum), Some(&maximum)) = (
                active_slice_counts.iter().min(),
                active_slice_counts.iter().max(),
            ) {
                let total_slice_requests = active_slice_counts
                    .iter()
                    .copied()
                    .fold(0_u64, u64::saturating_add);
                let concentration_numerator = active_slice_counts
                    .iter()
                    .map(|count| u128::from(*count).saturating_mul(u128::from(*count)))
                    .fold(0_u128, u128::saturating_add)
                    .saturating_mul(1_000_000);
                let concentration_denominator = u128::from(total_slice_requests)
                    .saturating_mul(u128::from(total_slice_requests));
                m.insert(
                    "l2_slice_active_count".into(),
                    Metric::count(
                        None,
                        "slice",
                        u128::from(active_slice_counts.len() as u64),
                        "number of effective L2 slices receiving requests; observer only",
                    ),
                );
                m.insert(
                    "l2_slice_max_fraction_pct".into(),
                    Metric::ratio(
                        None,
                        "%",
                        u128::from(maximum) * 100,
                        u128::from(total_slice_requests.max(1)),
                        false,
                        "maximum effective L2-slice demand fraction; not a physical service rate",
                    ),
                );
                m.insert(
                "l2_slice_concentration_ppm".into(),
                Metric::ratio(
                    None,
                    "ppm",
                    concentration_numerator,
                    concentration_denominator.max(1),
                    false,
                    "normalized effective L2-slice demand concentration; route-local observer only",
                ),
            );
                m.insert(
                    "l2_slice_request_max_sectors".into(),
                    Metric::count(
                        None,
                        "sector",
                        u128::from(maximum),
                        "maximum effective L2-slice request demand; observer only",
                    ),
                );
                m.insert(
                    "l2_slice_request_min_sectors".into(),
                    Metric::count(
                        None,
                        "sector",
                        u128::from(minimum),
                        "minimum nonzero effective L2-slice request demand; observer only",
                    ),
                );
                m.insert(
                    "l2_slice_request_spread_sectors".into(),
                    Metric::count(
                        None,
                        "sector",
                        u128::from(maximum.saturating_sub(minimum)),
                        "effective L2-slice demand spread; not a queue-delay prediction",
                    ),
                );
                m.insert(
                "l2_slice_request_imbalance_ratio".into(),
                Metric::ratio(
                    None,
                    "ratio",
                    u128::from(maximum),
                    u128::from(minimum.max(1)),
                    true,
                    "effective L2-slice max/min demand ratio; not a physical hash or service-rate claim",
                ),
            );
                let active_count = active_slice_counts.len();
                let route_incomplete = active_count < 6;
                let route_balanced = !route_incomplete
                    && concentration_numerator
                        < 250_000_u128.saturating_mul(concentration_denominator.max(1));
                let route_concentrated = !route_incomplete && !route_balanced;
                for (name, enabled, semantics) in [
                    (
                        "l2_route_incomplete",
                        route_incomplete,
                        "effective L2 route has fewer than six active slices; observer only",
                    ),
                    (
                        "l2_route_balanced",
                        route_balanced,
                        "complete effective L2 route with low demand concentration; observer only",
                    ),
                    (
                        "l2_route_concentrated",
                        route_concentrated,
                        "complete effective L2 route with high demand concentration; observer only",
                    ),
                ] {
                    m.insert(
                        name.into(),
                        Metric::count(None, "launch", u128::from(enabled as u8), semantics),
                    );
                }
            }
        }
        // Constant-bank accesses are retained as an explicit trace-derived
        // demand interval. Their offsets are not device allocation identities;
        // do not inject them into L1/L2 state or the timing-active DRAM server.
        let constant = result.trace_metrics();
        m.insert(
            "dram_constant_memory_address_count".into(),
            Metric::count(
                None,
                "address",
                u128::from(constant.constant_memory_address_count),
                "dynamic constant-bank address observations; demand-only trace evidence",
            ),
        );
        m.insert(
            "dram_constant_memory_reference_count".into(),
            Metric::count(
                None,
                "inst",
                u128::from(constant.constant_memory_reference_count),
                "constant-bank instructions carrying dynamic addresses; demand-only trace evidence",
            ),
        );
        m.insert(
        "dram_constant_memory_sector_requests".into(),
        Metric::count(
            None,
            "sector",
            u128::from(constant.constant_memory_sector_requests),
            "sum of per-instruction distinct 32-byte constant sectors; conservative request upper bound",
        ),
    );
        m.insert(
        "dram_constant_memory_unique_sectors".into(),
        Metric::count(
            None,
            "sector",
            u128::from(constant.constant_memory_unique_sectors),
            "launch-wide distinct 32-byte bank-local constant-offset sectors; cold-domain address proxy, not a mathematical DRAM lower bound",
        ),
    );
        m.insert(
        "dram_constant_memory_unique_lines".into(),
        Metric::count(
            None,
            "line",
            u128::from(constant.constant_memory_unique_lines),
            "launch-wide distinct 64-byte bank-local constant-offset candidate lines; address-domain evidence only, not a cache fill or DRAM event",
        ),
    );
        m.insert(
        "dram_constant_memory_metadata_only_records".into(),
        Metric::count(
            None,
            "record",
            u128::from(constant.constant_memory_metadata_only_records),
            "constant records with bank/offset metadata but no active lane-offset vector; excluded from demand projections",
        ),
    );
        let constant_records = constant
            .constant_memory_reference_count
            .checked_add(constant.constant_memory_metadata_only_records)
            .ok_or_else(|| RunError::new("constant demand observer", "record count overflow"))?;
        m.insert(
        "dram_constant_memory_address_record_coverage_pct".into(),
        Metric::ratio(
            None,
            "%",
            u128::from(constant.constant_memory_reference_count) * 100,
            u128::from(constant_records),
            false,
            "dynamic constant records carrying active-lane address vectors divided by all records carrying constant metadata; transport completeness observer only",
        ),
    );
        m.insert(
        "dram_constant_memory_references_per_unique_sector".into(),
        Metric::ratio(
            None,
            "reference/sector",
            u128::from(constant.constant_memory_reference_count),
            u128::from(constant.constant_memory_unique_sectors),
            false,
            "dynamic constant reference reuse over launch-wide unique bank-local sectors; not a cache hit rate or DRAM multiplier",
        ),
    );
        m.insert(
        "dram_constant_memory_requests_per_unique_sector".into(),
        Metric::ratio(
            None,
            "request/sector",
            u128::from(constant.constant_memory_sector_requests),
            u128::from(constant.constant_memory_unique_sectors),
            false,
            "per-instruction constant-sector request reuse over launch-wide unique sectors; demand observer only",
        ),
    );
        for (key, value, unit, semantics) in [
            (
                "constant_global_load_instructions",
                constant.constant_global_load_instructions,
                "inst",
                "static LDG.*.CONSTANT global-load records; source-qualified candidate, not a GCC counter",
            ),
            (
                "constant_global_load_sectors",
                constant.constant_global_load_sectors,
                "sector",
                "sectors on statically classified LDG.*.CONSTANT loads; source-qualified candidate, not L2/DRAM service",
            ),
            (
                "constant_global_load_request_candidates",
                constant.constant_global_load_request_candidates,
                "request",
                "per-instruction distinct 128-byte line candidates for LDG.*.CONSTANT; not GCC request truth",
            ),
            (
                "constant_global_load_unique_lines",
                constant.constant_global_load_unique_lines,
                "line",
                "launch-wide CTA/warp-scoped 128-byte LDG.*.CONSTANT candidate lines; source attribution only",
            ),
            (
                "constant_global_load_cta_unique_lines",
                constant.constant_global_load_cta_unique_lines,
                "line",
                "launch-wide CTA-scoped 128-byte LDG.*.CONSTANT candidate lines; not GCC request truth",
            ),
            (
                "constant_global_load_uniform_instructions",
                constant.constant_global_load_uniform_instructions,
                "inst",
                "static LDG.*.CONSTANT records with one active-lane byte address/width; uniform GCC source witness only",
            ),
            (
                "constant_global_load_uniform_sectors",
                constant.constant_global_load_uniform_sectors,
                "sector",
                "sectors on warp-uniform LDG.*.CONSTANT records; source-class observer, not GCC/DRAM traffic",
            ),
            (
                "constant_global_load_divergent_instructions",
                constant.constant_global_load_divergent_instructions,
                "inst",
                "static LDG.*.CONSTANT records with lane-divergent addresses; TEX source witness only",
            ),
            (
                "constant_global_load_divergent_sectors",
                constant.constant_global_load_divergent_sectors,
                "sector",
                "sectors on lane-divergent LDG.*.CONSTANT records; source-class observer, not GCC/DRAM traffic",
            ),
        ] {
            add_count_row(&mut m, key, None, unit, value, semantics);
        }
        for (index, name) in ["ltc128b", "u16", "other"].into_iter().enumerate() {
            add_count_row(
                &mut m,
                format!("constant_global_load_{name}_instructions"),
                None,
                "inst",
                constant.constant_global_load_form_instructions[index],
                "static LDG.*.CONSTANT modifier-form observer; does not relocate global traffic or claim an L0C client",
            );
            add_count_row(
                &mut m,
                format!("constant_global_load_{name}_sectors"),
                None,
                "sector",
                constant.constant_global_load_form_sectors[index],
                "modifier-form sectors retained as a scope discriminator; not an NCU lookup or DRAM prediction",
            );
        }
        add_count_row(
            &mut m,
            "constant_client_ldc_instructions",
            None,
            "inst",
            constant.constant_client_ldc_instructions,
            "dynamic LDC static-SASS population; GCC request/sector coalescing is not inferred",
        );
        add_count_row(
            &mut m,
            "constant_client_uldc_instructions",
            None,
            "inst",
            constant.constant_client_uldc_instructions,
            "dynamic ULDC static-SASS population; GCC request/sector coalescing is not inferred",
        );
        for (key, value, semantics) in [
            (
                "atomic_conflict_instructions",
                constant.atomic_conflict_instructions,
                "atomic instructions with repeated traced sectors; conflict exposure only",
            ),
            (
                "atomic_distinct_instructions",
                constant.atomic_distinct_instructions,
                "atomic instructions whose traced sectors are distinct; no serialization claim",
            ),
            (
                "atomic_conflict_alias_sectors",
                constant.atomic_conflict_alias_sectors,
                "same-sector atomic aliases in the trace; timing/ordering remains unavailable",
            ),
        ] {
            add_count_row(
                &mut m,
                key,
                None,
                if key.ends_with("sectors") {
                    "sector"
                } else {
                    "inst"
                },
                value,
                semantics,
            );
        }
        for (key, value, unit, semantics) in [
            (
                "local_width_records",
                constant.local_width_records,
                "inst",
                "local-memory instructions with a positive producer data_width field",
            ),
            (
                "local_width_unknown_records",
                constant.local_width_unknown_records,
                "inst",
                "local-memory records without usable producer data_width; no width guessed",
            ),
            (
                "local_width_bytes",
                constant.local_width_bytes,
                "byte",
                "sum of producer local-memory instruction widths; source evidence only",
            ),
            (
                "local_width_candidate_sectors",
                constant.local_width_candidate_sectors,
                "sector",
                "predicate-lane times data_width converted to 32-byte candidate sectors; not coalesced L1 traffic",
            ),
        ] {
            add_count_row(&mut m, key, None, unit, value, semantics);
        }
        if let Some(dram) = result.dram() {
            let effective_service_rate = dram.service_rate_bytes_per_sec.or_else(|| {
                card.require_dram_bandwidth()
                    .ok()
                    .and_then(|bandwidth| bandwidth.validated_peak_bytes_per_second().ok())
            });
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
            let compulsory_instruction_sectors = if result.arch() == ArchId::Sm86 {
                u128::from(result.front_end().static_instruction_line_read_sectors())
            } else {
                u128::from(result.front_end().compulsory_instruction_read_sectors())
            };
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
                "modeled L2-missing read demand plus source-qualified front-end fetch sectors; SM86 uses aligned 128-byte instruction-line sectors, SM120 uses executed sectors/TMA payload; typed demand, not achieved DRAM service",
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
            let mut emit_dram_pressure = |prefix: &str, values: &[u64]| {
                for (index, value) in values.iter().copied().enumerate() {
                    add_count_row(
                        &mut m,
                        format!("{prefix}_{index}_candidate_sectors"),
                        None,
                        "sector",
                        value,
                        "observer-only DRAM partition/slice pressure candidate from L2-miss line IDs; physical address swizzle and service timing remain unknown",
                    );
                }
            };
            emit_dram_pressure("dram_partition_modulo", &dram.dram_partition_modulo_sectors);
            emit_dram_pressure("dram_partition_xor", &dram.dram_partition_xor_sectors);
            emit_dram_pressure("dram_slice_modulo", &dram.dram_slice_modulo_sectors);
            let concentration_ppm = |values: &[u64]| -> u64 {
                let total = values.iter().copied().sum::<u64>();
                if total == 0 {
                    return 0;
                }
                let maximum = values.iter().copied().max().unwrap_or(0);
                u64::try_from(
                    u128::from(maximum)
                        .saturating_mul(1_000_000)
                        .checked_div(u128::from(total))
                        .unwrap_or(0),
                )
                .unwrap_or(u64::MAX)
            };
            let modulo_concentration_ppm = concentration_ppm(&dram.dram_partition_modulo_sectors);
            let xor_concentration_ppm = concentration_ppm(&dram.dram_partition_xor_sectors);
            let slice_concentration_ppm = concentration_ppm(&dram.dram_slice_modulo_sectors);
            add_count_row(
                &mut m,
                "dram_partition_modulo_max_fraction_ppm",
                None,
                "ppm",
                modulo_concentration_ppm,
                "observer-only maximum candidate modulo-partition demand fraction; not a physical hash or service fraction",
            );
            add_count_row(
                &mut m,
                "dram_partition_xor_max_fraction_ppm",
                None,
                "ppm",
                xor_concentration_ppm,
                "observer-only maximum candidate XOR-partition demand fraction; not a physical hash or service fraction",
            );
            add_count_row(
                &mut m,
                "dram_slice_modulo_max_fraction_ppm",
                None,
                "ppm",
                slice_concentration_ppm,
                "observer-only maximum candidate L2-slice demand fraction; not a physical hash or service fraction",
            );
            let critical_frontier_present = u64::from(result.gpc_observation_frontier().is_some());
            add_count_row(
                &mut m,
                "dram_partition_critical_frontier_samples",
                None,
                "sample",
                critical_frontier_present,
                "joint route-concentration/GPC-frontier observation presence; no queue delay or timing adjustment",
            );
            add_count_row(
                &mut m,
                "dram_partition_critical_frontier_concentration_ppm",
                None,
                "ppm",
                if critical_frontier_present != 0 {
                    modulo_concentration_ppm
                        .max(xor_concentration_ppm)
                        .max(slice_concentration_ppm)
                } else {
                    0
                },
                "joint candidate concentration reported alongside the modeled GPC observation frontier; observer only",
            );
            // Near/far is intentionally an uncertainty envelope: without the
            // proprietary VA→FBP/slice swizzle, the two half-controller groups
            // are only candidate map domains. Do not feed this interval into the
            // G/D/1 server or relabel it as a measured latency.
            let mut near_far_low = u64::MAX;
            let mut near_far_high = 0_u64;
            for values in [
                dram.dram_partition_modulo_sectors,
                dram.dram_partition_xor_sectors,
            ] {
                let first = values[..6].iter().copied().sum::<u64>();
                let second = values[6..].iter().copied().sum::<u64>();
                near_far_low = near_far_low.min(first.min(second));
                near_far_high = near_far_high.max(first.max(second));
            }
            if near_far_low == u64::MAX {
                near_far_low = 0;
            }
            add_count_row(
                &mut m,
                "dram_near_far_candidate_low_sectors",
                None,
                "sector",
                near_far_low,
                "lower edge of candidate near/far half-controller pressure envelope; physical proximity is unknown",
            );
            add_count_row(
                &mut m,
                "dram_near_far_candidate_high_sectors",
                None,
                "sector",
                near_far_high,
                "upper edge of candidate near/far half-controller pressure envelope; physical proximity is unknown",
            );
            add_count_row(
                &mut m,
                "dram_partition_map_uncertainty_width_sectors",
                None,
                "sector",
                near_far_high.saturating_sub(near_far_low),
                "width between candidate near/far map assignments; not a service delay or confidence interval",
            );
            let l2_instruction_demand = l2_fetch_read_sectors
                .checked_add(compulsory_instruction_sectors)
                .ok_or_else(|| {
                    RunError::new("DRAM demand projection", "L2/instruction demand overflow")
                })?;
            m.insert(
            "dram_read_demand_l2_instruction_sectors".into(),
            Metric::count(
                None,
                "sector",
                l2_instruction_demand,
                "L2-missing data plus compulsory instruction sectors; constant-bank offsets remain a separate source-domain observer",
            ),
        );
            // GCC/L1.5 instruction and constant-client demand is kept as an
            // observer-only companion to TEX L2-miss demand.  These candidates do
            // not enter L2 state, DRAM service, or any timing frontier until an
            // independently captured source-client probe establishes scope and
            // completion ownership.
            let gcc_constant_sector_candidate = constant
                .constant_memory_unique_sectors
                .checked_add(
                    constant
                        .constant_global_load_unique_lines
                        .checked_mul(2)
                        .ok_or_else(|| {
                            RunError::new(
                                "DRAM GCC demand projection",
                                "constant-line sector overflow",
                            )
                        })?,
                )
                .ok_or_else(|| {
                    RunError::new("DRAM GCC demand projection", "constant-sector overflow")
                })?;
            let gcc_instruction_sector_candidate = compulsory_instruction_sectors;
            let gcc_candidate_low = u64::try_from(gcc_instruction_sector_candidate)
                .map_err(|_| {
                    RunError::new(
                        "DRAM GCC demand projection",
                        "instruction candidate overflow",
                    )
                })?
                .checked_add(gcc_constant_sector_candidate)
                .ok_or_else(|| {
                    RunError::new("DRAM GCC demand projection", "GCC demand overflow")
                })?;
            let multiclient_candidate_low = u64::try_from(l2_fetch_read_sectors)
                .map_err(|_| RunError::new("DRAM GCC demand projection", "TEX candidate overflow"))?
                .checked_add(gcc_candidate_low)
                .ok_or_else(|| {
                    RunError::new("DRAM GCC demand projection", "multi-client demand overflow")
                })?;
            add_count_row(
                &mut m,
                "dram_gcc_instruction_sector_candidate",
                None,
                "sector",
                u64::try_from(gcc_instruction_sector_candidate).map_err(|_| {
                    RunError::new(
                        "DRAM GCC demand projection",
                        "instruction candidate overflow",
                    )
                })?,
                "observer-only compulsory instruction/GCC-client sector candidate; not admitted to L2 or DRAM timing",
            );
            add_count_row(
                &mut m,
                "dram_gcc_constant_sector_candidate",
                None,
                "sector",
                gcc_constant_sector_candidate,
                "observer-only constant-bank/GCC-client sector candidate; metadata/domain proxy, not a modeled fill",
            );
            add_count_row(
                &mut m,
                "dram_read_demand_multiclient_candidate_low",
                None,
                "sector",
                multiclient_candidate_low,
                "TEX L2-miss demand plus observer-only GCC instruction/constant candidates; no timing or NCU fit",
            );
            m.insert(
            "dram_constant_memory_address_domain_proxy_sectors".into(),
            Metric::count(
                None,
                "sector",
                u128::from(constant.constant_memory_unique_sectors),
                "conditional cold address-domain constant-offset proxy; field 15 is CBANK metadata, not a DRAM transaction",
            ),
        );
            m.insert(
            "dram_constant_memory_request_domain_candidate_sectors".into(),
            Metric::count(
                None,
                "sector",
                u128::from(constant.constant_memory_sector_requests),
                "conditional per-instruction constant-offset packet candidate; field 15 is CBANK metadata, not a DRAM transaction and reload/fill semantics remain unavailable",
            ),
        );
            let source_candidate_low = l2_instruction_demand;
            let source_candidate_high = source_candidate_low
                .checked_add(u128::from(constant.constant_global_load_unique_lines))
                .and_then(|value| {
                    value.checked_add(u128::from(constant.constant_memory_unique_lines))
                })
                .ok_or_else(|| {
                    RunError::new(
                        "DRAM demand projection",
                        "source candidate envelope overflow",
                    )
                })?;
            m.insert(
            "dram_read_demand_source_candidate_low_sectors".into(),
            Metric::count(
                None,
                "sector",
                source_candidate_low,
                "L2-missing data plus compulsory instruction sectors; typed demand lower source edge, not achieved DRAM traffic",
            ),
        );
            m.insert(
            "dram_read_demand_source_candidate_high_sectors".into(),
            Metric::count(
                None,
                "sector",
                source_candidate_high,
                "L2/instruction demand plus static LDG.CONSTANT and CBANK candidate lines; uncertainty envelope, not a physical DRAM upper bound",
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
                effective_service_rate,
            ) {
                (Some(frontier), Some(hz), Some(peak_bytes_per_second)) => Metric::ratio(
                    None,
                    "%",
                    dram_read_bytes
                        .saturating_mul(u128::from(hz))
                        .saturating_mul(100),
                    u128::from(frontier.cycles).saturating_mul(u128::from(peak_bytes_per_second)),
                    true,
                    "read-only modeled demand divided by the configured carded service rate (runner default is the effective 825 GB/s hypothesis; direct-run fallback is the card peak); excludes writeback/coherence and is not NCU achieved bandwidth",
                ),
                _ => Metric::unavailable(
                    None,
                    "%",
                    "requires a shared-L2 DRAM observer, card SM clock, GPC frontier, and configured carded service rate",
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
            let writeback_service_cycles = effective_service_rate.and_then(|peak| {
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
                        effective_service_rate
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
                        effective_service_rate
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
            add_count_row(
                &mut m,
                "l2_frontier_samples",
                None,
                "sample",
                dram.l2_frontier_samples,
                "observer-only count of L2 requests with fill/response/service frontier data",
            );
            add_count_row(
                &mut m,
                "l2_issue_to_fill_cycles_sum",
                None,
                "cycle",
                dram.l2_issue_to_fill_cycles_sum,
                "observer-only sum of model issue-to-fill intervals; not hardware completion",
            );
            add_count_row(
                &mut m,
                "l2_issue_to_fill_cycles_max",
                None,
                "cycle",
                dram.l2_issue_to_fill_cycles_max,
                "observer-only maximum model issue-to-fill interval; not hardware completion",
            );
            add_count_row(
                &mut m,
                "l2_issue_to_response_cycles_sum",
                None,
                "cycle",
                dram.l2_issue_to_response_cycles_sum,
                "observer-only sum of model issue-to-response intervals; not hardware completion",
            );
            add_count_row(
                &mut m,
                "l2_issue_to_response_cycles_max",
                None,
                "cycle",
                dram.l2_issue_to_response_cycles_max,
                "observer-only maximum model issue-to-response interval; not hardware completion",
            );
            add_count_row(
                &mut m,
                "l2_service_delay_cycles_sum",
                None,
                "cycle",
                dram.l2_service_delay_cycles_sum,
                "observer-only sum of model bandwidth-service intervals; not a queue-delay law",
            );
            add_count_row(
                &mut m,
                "l2_service_delay_cycles_max",
                None,
                "cycle",
                dram.l2_service_delay_cycles_max,
                "observer-only maximum model bandwidth-service interval; not a queue-delay law",
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

        Ok(())
    }
}
