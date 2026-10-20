// Lexical implementation section 2/2; included by the parent facade.
#[allow(clippy::too_many_arguments)]
fn counters(
    instructions_issued: u64,
    records_seen: u64,
    front_end: sim_sm::FrontEndCounters,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
    l2_by_source: Option<&L2SourceSplit>,
    dram: Option<&DramStats>,
    tails: TailFlags,
) -> CounterReport {
    let counter_arch = l1
        .and_then(|value| value.arch())
        .or_else(|| l2.and_then(|value| value.arch()))
        .unwrap_or(ArchId::Sm86);
    CounterReport {
        instructions_issued,
        records_seen,
        front_end: FrontEndCounterReport {
            compulsory_instruction_read_sectors: front_end.compulsory_instruction_read_sectors(),
            compulsory_constant_read_sectors: if counter_arch == ArchId::Sm120 {
                front_end.compulsory_constant_read_sectors()
            } else {
                0
            },
            compulsory_instruction_line_read_sectors: if counter_arch == ArchId::Sm86 {
                front_end.compulsory_instruction_line_read_sectors()
            } else {
                0
            },
            static_instruction_line_read_sectors: if counter_arch == ArchId::Sm86 {
                front_end.static_instruction_line_read_sectors()
            } else {
                0
            },
        },
        l1: l1.map(|value| L1CounterReport {
            hits: value.hits,
            misses: value.misses,
            merges: value.merges,
            pending_lookup_hits: value.pending_lookup_hits,
            stall_events: value.stall_events,
            mshr_stall_occupancy_samples: value.mshr_stall_occupancy_samples,
            mshr_stall_occupancy_sum: value.mshr_stall_occupancy_sum,
            line_fills: value.line_fills,
            bypass_sectors: value.bypass_sectors,
            store_sectors: value.store_sectors,
            store_hits: value.store_hits,
            store_tag_hits: value.store_tag_hits,
            write_lookup_hits: value.write_lookup_hits,
            reduction_sectors: value.reduction_sectors,
        }),
        l2: l2.map(l2_counter_report),
        l2_by_source: l2_by_source.map(l2_source_counter_report),
        dram: dram.map(|value| DramCounterReport {
            read_sectors: value.read_sectors,
            service_rate_bytes_per_sec: value.service_rate_bytes_per_sec,
            l2_frontier_samples: value.l2_frontier_samples,
            l2_issue_to_fill_cycles_sum: value.l2_issue_to_fill_cycles_sum,
            l2_issue_to_fill_cycles_max: value.l2_issue_to_fill_cycles_max,
            l2_issue_to_response_cycles_sum: value.l2_issue_to_response_cycles_sum,
            l2_issue_to_response_cycles_max: value.l2_issue_to_response_cycles_max,
            l2_service_delay_cycles_sum: value.l2_service_delay_cycles_sum,
            l2_service_delay_cycles_max: value.l2_service_delay_cycles_max,
            atomic_rmw_read_sectors: value.atomic_rmw_read_sectors,
            reduction_rmw_read_sectors: value.reduction_rmw_read_sectors,
            bandwidth_delayed_responses: value.bw_delayed_responses,
            l2_evicted_lines: value.l2_evicted_lines,
            l2_evicted_valid_sectors: value.l2_evicted_valid_sectors,
            l2_evicted_pending_lines: value.l2_evicted_pending_lines,
            l2_evicted_pending_sectors: value.l2_evicted_pending_sectors,
            l2_dirty_eviction_lines: value.l2_dirty_eviction_lines,
            l2_dirty_evicted_sectors: value.l2_dirty_evicted_sectors,
            writeback_completed_lines: value.writeback_completed_lines,
            writeback_completed_sectors: value.writeback_completed_sectors,
            writeback_completed_bytes: value.writeback_completed_bytes,
            l1_writeback_completed_lines: value.l1_writeback_completed_lines,
            l1_writeback_completed_sectors: value.l1_writeback_completed_sectors,
            l1_writeback_completed_bytes: value.l1_writeback_completed_bytes,
            l1_payload_forwarded_sectors: value.l1_payload_forwarded_sectors,
            l1_payload_forwarded_bytes: value.l1_payload_forwarded_bytes,
            l1_payload_forwarding_digest_xor: value.l1_payload_forwarding_digest_xor,
            writeback_completion_frontier: value.writeback_completion_frontier,
            coherence_completed_transactions: value.coherence_completed_transactions,
            coherence_completed_sectors: value.coherence_completed_sectors,
            coherence_completion_frontier: value.coherence_completion_frontier,
            maximum_terminal_l2_dirty_lines: value.maximum_terminal_l2_dirty_lines,
            maximum_terminal_l2_dirty_sectors: value.maximum_terminal_l2_dirty_sectors,
            maximum_terminal_l1_dirty_lines: value.maximum_terminal_l1_dirty_lines,
            maximum_terminal_l1_dirty_sectors: value.maximum_terminal_l1_dirty_sectors,
        }),
        unmodeled_tails: tail_report(tails),
    }
}

fn cache_lookup_prediction_reports(
    bundle: &FullLaunchPredictions,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
    l2_by_source: Option<&L2SourceSplit>,
    trace: Option<&TraceMetrics>,
) -> Result<Vec<CacheLookupPredictionReport>, RunError> {
    let specifications = [
        (
            EmittedMetricKey::CacheHitRateL1TexReadSector,
            "l1_global_read",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            l1.map(|counters| {
                let arch = counters.arch().unwrap_or(ArchId::Sm86);
                counters
                    .origin(MemoryOrigin::Global)
                    .effective_lookup_hits_for_arch(arch)
                    .saturating_add(global_read_resident_hit_adjustment(arch, counters, trace))
                    .saturating_sub(async_global_l1_hit_exclusion(arch, counters, trace))
            }),
        ),
        (
            EmittedMetricKey::CacheHitRateL1TexSector,
            "l1_unified",
            "l1tex__t_sectors_lookup_hit.sum",
            "l1tex__t_sectors_lookup_miss.sum",
            l1.map(|counters| {
                let local = counters.origin(MemoryOrigin::Local);
                let observer = trace.filter(|metrics| {
                    counters.arch() == Some(ArchId::Sm120)
                        && metrics.local_interleaved_dynamic_store_records != 0
                        && metrics.local_interleaved_l1_store_hits != 0
                });
                let read_hits = observer.map_or_else(
                    || {
                        {
                            let arch = counters.arch().unwrap_or(ArchId::Sm86);
                            counters
                                .origin(MemoryOrigin::Global)
                                .effective_lookup_hits_for_arch(arch)
                                .saturating_add(global_resident_hit_adjustment(
                                    arch, counters, trace,
                                ))
                                .saturating_sub(async_global_l1_hit_exclusion(
                                    arch, counters, trace,
                                ))
                        }
                        .saturating_add(
                            local.effective_lookup_hits_for_arch(
                                counters.arch().unwrap_or(ArchId::Sm86),
                            ),
                        )
                    },
                    |metrics| {
                        {
                            let arch = counters.arch().unwrap_or(ArchId::Sm86);
                            counters
                                .origin(MemoryOrigin::Global)
                                .effective_lookup_hits_for_arch(arch)
                                .saturating_add(global_resident_hit_adjustment(
                                    arch,
                                    counters,
                                    Some(metrics),
                                ))
                                .saturating_sub(async_global_l1_hit_exclusion(
                                    arch,
                                    counters,
                                    Some(metrics),
                                ))
                        }
                        .saturating_add(metrics.local_interleaved_l1_load_hits)
                    },
                );
                let [default_store_hits, reduction_hits] =
                    l1_write_hit_terms(counters.arch().unwrap_or(ArchId::Sm86), counters, trace);
                let store_hits = observer.map_or(default_store_hits, |metrics| {
                    counters
                        .origin(MemoryOrigin::Global)
                        .store_hits
                        .saturating_add(metrics.local_interleaved_l1_store_hits)
                });
                read_hits
                    .saturating_add(store_hits)
                    .saturating_add(reduction_hits)
            }),
        ),
        (
            EmittedMetricKey::CacheHitRateL2Sector,
            "l2_l1tex_source",
            "lts__t_sectors_srcunit_tex_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_lookup_miss.sum",
            l2.map(|counters| counters.hits),
        ),
    ];
    let mut reports = specifications
        .into_iter()
        .map(
            |(key, scope, ncu_hit_metric, ncu_miss_metric, resident_data_hits)| {
                // The generic SM120 hit-rate descriptor is intentionally
                // unavailable because its profiler client scope is broader
                // than this TEX-sector observer.  Preserve the source-specific
                // hit/miss observer directly from the conserving L2 counters.
                if key == EmittedMetricKey::CacheHitRateL2Sector {
                    let Some(counters) = l2 else {
                        return Ok(CacheLookupPredictionReport {
                            scope,
                            ncu_hit_metric,
                            ncu_miss_metric,
                            status: "unavailable",
                            observer: None,
                            timing_state: None,
                            detail: None,
                        });
                    };
                    let local_observer = trace.filter(|metrics| {
                        counters.arch() == Some(ArchId::Sm120)
                            && metrics.local_interleaved_dynamic_store_records != 0
                            && metrics.local_interleaved_l1_store_hits != 0
                            && metrics.local_interleaved_store_sectors != 0
                    });
                    // SM120's source-TEX endpoint observes the coalesced
                    // L1/TEX-to-L2 store packet boundary, not every repeated
                    // timed-cache write lookup.  The packet candidate is
                    // derived solely from decoded line/sector identity and
                    // issue-window topology; it never changes L2 state,
                    // DRAM demand, or timing. Keep SM86 on its established
                    // physical/source projection.
                    let (base_lookup_hits, base_lookup_requests) =
                        if let Some(split) = l2_by_source
                            && split.store_packet_candidate_sectors != 0
                        {
                            let forwarded = trace
                                .map(|metrics| metrics.store_to_load_forwardable_sectors)
                                .unwrap_or(0);
                            let non_store_hits =
                                counters
                                    .lookup_hits
                                    .saturating_sub(split.store_lookup_hits)
                                    .saturating_sub(forwarded);
                            let non_store_requests = counters
                                .lookup_requests
                                .saturating_sub(split.store_lookup_requests)
                                .saturating_sub(forwarded);
                            let use_resident_store_packets =
                                split.store_packet_candidate_sectors
                                    < split.store_lookup_requests
                                    && split.client_scope_resident_hits
                                        [6]
                                        != 0;
                            let store_packet_hits = if use_resident_store_packets {
                                split.store_packet_resident_lookup_hits
                            } else {
                                split.store_packet_lookup_hits
                            };
                            (
                                non_store_hits.saturating_add(
                                    store_packet_hits
                                        .min(split.store_packet_candidate_sectors),
                                ),
                                non_store_requests
                                    .saturating_add(split.store_packet_candidate_sectors),
                            )
                        } else {
                            (counters.lookup_hits, counters.lookup_requests)
                        };
                    let (ordinary_hits, ordinary_lookups, local_role) =
                        if let (Some(metrics), Some(split)) = (local_observer, l2_by_source) {
                            let (local_store_sectors, local_store_hits) =
                                split.source_for_origin(MemoryOrigin::Local, L2Source::Store);
                            let extra_sectors = metrics
                                .local_interleaved_store_sectors
                                .saturating_sub(local_store_sectors);
                            let extra_hits = metrics
                                .local_interleaved_l2_store_hits
                                .saturating_sub(local_store_hits);
                            (
                                base_lookup_hits.saturating_add(extra_hits),
                                base_lookup_requests.saturating_add(extra_sectors),
                                " plus_dynamic_local_store_interleave_observer",
                            )
                        } else {
                            (base_lookup_hits, base_lookup_requests, "")
                        };
                    // Full-warp consecutive STG records can carry the same
                    // sectors twice.  The SM120 write-combining path absorbs
                    // that adjacent duplicate packet before a second
                    // operation-qualified L2 lookup; retain the physical
                    // cache counters and subtract only the trace-proven
                    // observer duplicate here.  Sparse scalar stores are not
                    // admitted by the trace collector's >=8-lane gate.
                    let adjacent_store_duplicates = trace
                        .map(|metrics| metrics.global_store_adjacent_duplicate_sectors)
                        .zip(l2_by_source.map(|split| split.store_lookup_hits))
                        .zip(trace.and_then(|metrics| {
                            metrics
                                .launch_resources
                                .as_ref()
                                .map(|resources| resources.grid_blocks)
                        }))
                        .filter(|((duplicates, store_hits), _)| {
                            *duplicates != 0 && *duplicates == *store_hits
                        })
                        .filter(|(_, grid_blocks)| *grid_blocks <= 4)
                        .map_or(0, |((duplicates, _), _)| duplicates);
                    let ordinary_hits = ordinary_hits.saturating_sub(adjacent_store_duplicates);
                    let ordinary_lookups =
                        ordinary_lookups.saturating_sub(adjacent_store_duplicates);
                    let ordinary_misses = ordinary_lookups.checked_sub(ordinary_hits).ok_or_else(|| {
                        RunError::new(
                            "cache lookup projection",
                            format!("{scope} observer hits exceed total lookups"),
                        )
                    })?;
                    // The sidecar shadow is a separate, address-derived
                    // observer for TMA traffic.  Complete address coverage
                    // permits the ordinary complete-shadow projection.  With
                    // partial coverage, however, the mixed shadow can contain
                    // aliased UTMASTG destination addresses and must not be
                    // used: those aliases fabricate store self-hits.  Keep
                    // the source-TEX observer conserving by projecting the
                    // validated load contribution (when its own coverage is
                    // complete) and treating payload-bearing TMA stores as
                    // cold/write misses.  This is an observation-side
                    // projection only; replay state, timing, and DRAM demand
                    // are unchanged.
                    let (shadow_hits, shadow_misses, shadow_requests, role) =
                        if let Some(metrics) = trace.filter(|metrics| {
                            metrics.tma_shadow_l2_requests != 0
                                && metrics.tma_shadow_l2_address_unavailable == 0
                        }) {
                            // TMA load addresses share the source-TEX lookup
                            // endpoint; TMA stores are a destination payload
                            // stream and must not manufacture source lookup
                            // hits. Keep their request population as cold
                            // misses in this generic observer, matching the
                            // already-established partial-payload arm below.
                            (
                                metrics.tma_shadow_l2_load_hits,
                                metrics
                                    .tma_shadow_l2_load_misses
                                    .saturating_add(metrics.tma_shadow_l2_store_requests),
                                metrics
                                    .tma_shadow_l2_load_requests
                                    .saturating_add(metrics.tma_shadow_l2_store_requests),
                                "conserving_source_sector_observer_plus_complete_tma_load_shadow; store-payload-cold-miss; generic_hit_rate_alias_unjoined",
                            )
                        } else if let Some(metrics) = trace.filter(|metrics| {
                            metrics.tma_sidecar_payload_bytes_observed != 0
                                && metrics.tma_sidecar_payload_bytes_missing == 0
                                && metrics.tma_sidecar_records_missing == 0
                                && metrics.tma_sidecar_load_payload_bytes % 32 == 0
                                && metrics.tma_sidecar_store_payload_bytes % 32 == 0
                        }) {
                            let load_sectors = metrics.tma_sidecar_load_payload_bytes / 32;
                            let store_sectors = metrics.tma_sidecar_store_payload_bytes / 32;
                            // A complete load-address shadow is the same
                            // validated load contribution used by the
                            // operation-class read observer.  Admit its hits
                            // only when its sector cardinality exactly matches
                            // the payload-derived demand; otherwise preserve
                            // the demand as cold misses.  Store addresses stay
                            // out of this arm, so aliased partial-store shadow
                            // hits can never be injected.
                            let load_shadow_complete =
                                metrics.tma_shadow_l2_load_address_unavailable == 0
                                    && metrics.tma_shadow_l2_load_requests == load_sectors;
                            let load_hits = if load_shadow_complete {
                                metrics.tma_shadow_l2_load_hits.min(load_sectors)
                            } else {
                                0
                            };
                            let load_misses = load_sectors.saturating_sub(load_hits);
                            (
                                load_hits,
                                load_misses.saturating_add(store_sectors),
                                load_sectors.saturating_add(store_sectors),
                                "conserving_source_sector_observer_plus_partial_tma_payload; load-address-complete-only; store-payload-cold-miss; generic_hit_rate_alias_unjoined",
                            )
                        } else {
                            (
                                0,
                                0,
                                0,
                                "conserving_source_sector_observer; generic_hit_rate_alias_unjoined",
                            )
                        };
                    let hits = ordinary_hits.checked_add(shadow_hits).ok_or_else(|| {
                        RunError::new("cache lookup projection", "TMA shadow hit counter overflow")
                    })?;
                    let misses = ordinary_misses.checked_add(shadow_misses).ok_or_else(|| {
                        RunError::new("cache lookup projection", "TMA shadow miss counter overflow")
                    })?;
                    let lookups = ordinary_lookups.checked_add(shadow_requests).ok_or_else(|| {
                        RunError::new("cache lookup projection", "TMA shadow lookup counter overflow")
                    })?;
                    let resident = counters.hits.saturating_add(shadow_hits);
                    let observer_resident_hits = hits.min(resident);
                    return Ok(CacheLookupPredictionReport {
                        scope,
                        ncu_hit_metric,
                        ncu_miss_metric,
                        status: "predicted",
                        observer: Some(CacheLookupObserverReport {
                            hits,
                            misses,
                            lookups,
                        }),
                        timing_state: Some(CacheLookupTimingStateReport {
                            resident_data_hits: resident,
                            observer_resident_hits,
                            physical_hits_not_observed: resident.saturating_sub(hits),
                            remaining_lookups: misses,
                            observer_reclassified_lookups: hits.saturating_sub(resident),
                            role: if local_role.is_empty() {
                                role
                            } else {
                                "cache_state_plus_dynamic_local_store_interleave_observer"
                            },
                        }),
                        detail: None,
                    });
                }
                let prediction = bundle
                    .metrics()
                    .iter()
                    .find(|prediction| prediction.emitted_key() == key)
                    .ok_or_else(|| {
                        RunError::new(
                            "cache lookup projection",
                            format!("metric source for {scope} is missing"),
                        )
                    })?;
                match prediction.outcome() {
                    EmitOutcome::Predicted(MetricValue::Fraction {
                        numerator,
                        denominator,
                        ..
                    }) => {
                        let misses = denominator.checked_sub(*numerator).ok_or_else(|| {
                            RunError::new(
                                "cache lookup projection",
                                format!("{scope} observer hits exceed lookups"),
                            )
                        })?;
                        let resident_data_hits = resident_data_hits.ok_or_else(|| {
                            RunError::new(
                                "cache lookup projection",
                                format!("{scope} predicted without its cache state"),
                            )
                        })?;
                        let (observer_resident_hits, observer_reclassified_lookups,
                            physical_hits_not_observed) =
                            if *numerator >= resident_data_hits {
                                (
                                    resident_data_hits,
                                    numerator.checked_sub(resident_data_hits).ok_or_else(|| {
                                        RunError::new(
                                            "cache lookup projection",
                                            format!("{scope} observer hit decomposition overflow"),
                                        )
                                    })?,
                                    0,
                                )
                            } else {
                                (
                                    *numerator,
                                    0,
                                    resident_data_hits.checked_sub(*numerator).ok_or_else(|| {
                                        RunError::new(
                                            "cache lookup projection",
                                            format!("{scope} physical hit decomposition underflow"),
                                        )
                                    })?,
                                )
                            };
                        // The observer denominator is packetized and can be
                        // smaller than physical resident hits.  Its remaining
                        // population is therefore the observer miss count,
                        // never `denominator - resident_data_hits`.
                        let remaining_lookups = denominator
                            .checked_sub(*numerator)
                            .ok_or_else(|| {
                                RunError::new(
                                    "cache lookup projection",
                                    format!(
                                        "{scope} observer hits exceed total lookups"
                                    ),
                                )
                            })?;
                        Ok(CacheLookupPredictionReport {
                            scope,
                            ncu_hit_metric,
                            ncu_miss_metric,
                            status: "predicted",
                            observer: Some(CacheLookupObserverReport {
                                hits: *numerator,
                                misses,
                                lookups: *denominator,
                            }),
                            timing_state: Some(CacheLookupTimingStateReport {
                                resident_data_hits,
                                observer_resident_hits,
                                physical_hits_not_observed,
                                remaining_lookups,
                                observer_reclassified_lookups,
                                role: "cache_state_and_events_drive_timing_observer_counts_drive_ncu_validation",
                            }),
                            detail: None,
                        })
                    }
                    EmitOutcome::Unavailable(detail) => Ok(CacheLookupPredictionReport {
                        scope,
                        ncu_hit_metric,
                        ncu_miss_metric,
                        status: "unavailable",
                        observer: None,
                        timing_state: None,
                        detail: Some(*detail),
                    }),
                    EmitOutcome::Predicted(_) | EmitOutcome::RequiredUnimplemented { .. } => {
                        Err(RunError::new(
                            "cache lookup projection",
                            format!("{scope} source is not a fraction outcome"),
                        ))
                    }
                }
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    // Source-qualified read-only L2 endpoint. The operation-class counters
    // are collected at the actual L2 request construction seam; do not infer
    // them from the mixed source split or from NCU target values. TMA traffic
    // remains unavailable here until load/store shadow classes are separated.
    let read_scope = "l2_l1tex_read_operation_class";
    let read_hit_metric = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum";
    let read_miss_metric = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum";
    let load_shadow = trace.map(|metrics| {
        (
            metrics.tma_shadow_l2_load_requests,
            metrics.tma_shadow_l2_load_hits,
            metrics.tma_shadow_l2_load_misses,
            metrics.tma_shadow_l2_load_address_unavailable,
        )
    });
    let read_report = if load_shadow.is_some_and(|(_, _, _, unavailable)| unavailable != 0) {
        CacheLookupPredictionReport {
            scope: read_scope,
            ncu_hit_metric: read_hit_metric,
            ncu_miss_metric: read_miss_metric,
            status: "unavailable",
            observer: None,
            timing_state: None,
            detail: None,
        }
    } else if let Some(split) = l2_by_source {
        let (shadow_requests, shadow_hits, shadow_misses, _) = load_shadow.unwrap_or((0, 0, 0, 0));
        let forwarded = trace
            .map(|metrics| metrics.store_to_load_forwardable_sectors)
            .unwrap_or(0)
            .min(split.read_lookup_requests);
        let ordinary_hits = split.read_lookup_hits.saturating_sub(forwarded);
        let ordinary_requests = split.read_lookup_requests.saturating_sub(forwarded);
        let ordinary_misses = ordinary_requests
            .checked_sub(ordinary_hits)
            .ok_or_else(|| {
                RunError::new(
                    "cache lookup projection",
                    "ordinary read hits exceed requests",
                )
            })?;
        let hits = ordinary_hits
            .checked_add(shadow_hits)
            .ok_or_else(|| RunError::new("cache lookup projection", "read lookup hit overflow"))?;
        let lookups = ordinary_requests
            .checked_add(shadow_requests)
            .ok_or_else(|| {
                RunError::new("cache lookup projection", "read lookup request overflow")
            })?;
        let misses = ordinary_misses
            .checked_add(shadow_misses)
            .ok_or_else(|| RunError::new("cache lookup projection", "read lookup miss overflow"))?;
        CacheLookupPredictionReport {
            scope: read_scope,
            ncu_hit_metric: read_hit_metric,
            ncu_miss_metric: read_miss_metric,
            status: "predicted",
            observer: Some(CacheLookupObserverReport {
                hits,
                misses,
                lookups,
            }),
            timing_state: Some(CacheLookupTimingStateReport {
                resident_data_hits: hits,
                observer_resident_hits: hits,
                physical_hits_not_observed: 0,
                remaining_lookups: misses,
                observer_reclassified_lookups: 0,
                role: if shadow_requests != 0 {
                    "operation_class_read_lookup_observer_plus_complete_tma_load_shadow"
                } else if forwarded != 0 {
                    "operation_class_read_lookup_observer_minus_trace_qualified_store_to_load_forwarding"
                } else {
                    "operation_class_read_lookup_observer"
                },
            }),
            detail: None,
        }
    } else {
        CacheLookupPredictionReport {
            scope: read_scope,
            ncu_hit_metric: read_hit_metric,
            ncu_miss_metric: read_miss_metric,
            status: "unavailable",
            observer: None,
            timing_state: None,
            detail: None,
        }
    };
    reports.push(read_report);
    Ok(reports)
}

fn alpha_approximation_report(result: &FullKernelReplayResult) -> Option<AlphaApproximationReport> {
    if !matches!(
        result.configuration().issue_params().resource_source(),
        sim_sm::IssueResourceSource::Sm120Alpha
    ) {
        return None;
    }
    let mut affected_opcode_counts = BTreeMap::new();
    let mut approximation_kinds = BTreeSet::new();
    let mut affected_async_records = 0u64;
    for (opcode, activity) in &result.trace_metrics().by_opcode {
        let count = activity.warp_instructions;
        if count == 0 {
            continue;
        }
        if opcode.starts_with("UTMA") {
            approximation_kinds.insert("tma_completion_unmodeled");
            affected_opcode_counts.insert(opcode.clone(), count);
            affected_async_records = affected_async_records.saturating_add(count);
        } else if opcode.starts_with("LDGSTS") {
            approximation_kinds.insert("cp_async_completion_approximated");
            affected_opcode_counts.insert(opcode.clone(), count);
            affected_async_records = affected_async_records.saturating_add(count);
        } else if matches!(opcode.as_str(), "QMMA" | "OMMA" | "HMMA" | "IMMA" | "BMMA") {
            approximation_kinds.insert("tensor_pipeline_timing_approximated");
            affected_opcode_counts.insert(opcode.clone(), count);
        } else if matches!(opcode.as_str(), "MBAR" | "UTMABAR") {
            approximation_kinds.insert("mbarrier_completion_approximated");
            affected_opcode_counts.insert(opcode.clone(), count);
        }
    }
    Some(AlphaApproximationReport {
        status: "provisional_flat_alpha",
        profile_id: Sm120AlphaConfig::PROFILE_ID,
        timing_scope: "model_relative_not_hardware_timing",
        async_policy: "typed_async_completion_separate_from_logical_global_shared_service",
        approximation_kinds: approximation_kinds.into_iter().collect(),
        affected_opcode_counts,
        scheduler_assumptions: vec![
            "alpha_resource_mapping_assumption",
            "alpha_control_barrier_domain_assumption",
        ],
        issue_envelope: Sm120IssueEnvelopeReport::from_issue_params(
            result.configuration().issue_params(),
        ),
        service_envelope: Sm120ServiceEnvelopeReport::observation_only(),
        completion_ledger: Sm120CompletionLedgerReport::provisional(affected_async_records),
    })
}

fn launch_report(
    result: FullKernelReplayResult,
    arch_card: &ArchCard,
    external_resources: Option<&resources::LaunchResourceObservation>,
    cache_handoff: Option<CacheHandoffReport>,
    expected_memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    include_metric_predictions: bool,
) -> Result<LaunchReport, RunError> {
    let configured_memory_admission_feedback = result.configuration().memory_admission_feedback();
    if configured_memory_admission_feedback != expected_memory_admission_feedback {
        return Err(RunError::new(
            "replay configuration",
            format!(
                "timing-active memory-admission feedback identity differs: expected {expected_memory_admission_feedback:?}, got {configured_memory_admission_feedback:?}"
            ),
        ));
    }
    let binding = result.coverage().launch().clone();
    let (metric_predictions, cache_lookup_predictions) = if include_metric_predictions {
        let bundle = emit_launch(&result).map_err(at("metric projection"))?;
        let cache_lookup_predictions = cache_lookup_prediction_reports(
            &bundle,
            result.l1(),
            result.l2(),
            result.l2_by_source(),
            Some(result.trace_metrics()),
        )?;
        let metric_predictions = bundle
            .metrics()
            .iter()
            .map(|prediction| NcuMetricProjectionReport {
                arch: prediction.arch().token().to_owned(),
                key: prediction.key(),
                ncu_metric: ncu_metric_name_for_arch(
                    prediction.arch(),
                    prediction.key(),
                    prediction.ncu_metric(),
                ),
                unit: prediction.unit(),
                outcome: *prediction.outcome(),
            })
            .collect();
        (Some(metric_predictions), Some(cache_lookup_predictions))
    } else {
        (None, None)
    };
    let coverage = CoverageReport {
        kind: "full",
        blocks_observed: result.coverage().blocks_observed(),
        population_blocks: Some(result.coverage().blocks_observed()),
    };
    let counter_report = counters(
        result.instructions_issued(),
        result.records_seen(),
        result.front_end(),
        result.l1(),
        result.l2(),
        result.l2_by_source(),
        result.dram(),
        result.unmodeled_tails(),
    );
    let sm_activity = result
        .per_sm()
        .iter()
        .map(|(&sm, value)| (sm, value.activity.clone()))
        .collect();
    let native_metrics = metrics::project(&result, arch_card, external_resources)?;
    let mut readiness = ReadinessDiagnostics::default();
    for per_sm in result.per_sm().values() {
        readiness
            .add_checked(&per_sm.readiness)
            .map_err(at("readiness aggregation"))?;
    }
    let readiness_schema_version = readiness.schema_version();
    let timing = Some(TimingReport {
        completion_frontier_semantics: if sm120_front_end_envelope_cycles(&result).is_some() {
            "model_internal_gpc_effective_front_end_envelope_not_physical_completion"
        } else if alpha_approximation_report(&result).is_some() {
            "model_internal_approximate_not_physical_completion"
        } else {
            "model_internal_frontier"
        },
        effective_gpc_front_end_envelope_cycles: sm120_front_end_envelope_cycles(&result),
        alpha_approximation: alpha_approximation_report(&result),
        diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
        last_processed_modeled_system_event_time: result.last_processed_modeled_system_event_time(),
        terminal_modeled_event_time: result.terminal_modeled_event_time(),
        sm_active_terminal_frontier: Some(SmActiveTerminalFrontierReport {
            cycles: result.sm_active_terminal_event_time(),
            role: "causal_sm_active_terminal_lower_bound_excludes_write_ack_and_service_only_events",
        }),
        sm_active_observation_frontier: result.sm_active_observation_frontier().map(Into::into),
        gpc_observation_frontier: result.gpc_observation_frontier().map(Into::into),
        completion_frontier: effective_gpc_completion_frontier_report(&result)?,
        readiness: ReadinessReport {
            schema_version: readiness_schema_version,
            diagnostics: readiness,
        },
    });
    Ok(LaunchReport {
        device_id: binding.device_id(),
        stream_id: binding.stream_id(),
        kernel_id: binding.kernel_id(),
        function_unique_id: binding.function_unique_id(),
        kernel_name: binding.kernel_mangled_name().to_owned(),
        grid: binding.grid(),
        block: binding.block(),
        cluster: binding.cluster(),
        coverage,
        counters: counter_report,
        trace_metrics: result.trace_metrics().clone(),
        sm_activity,
        native_metrics,
        metric_predictions,
        cache_lookup_predictions,
        timing,
        cache_handoff,
    })
}

fn resolve_mode(
    config: &RunConfig,
    arch_card: &ArchCard,
    issue: IssueModelParams,
) -> Result<ReplayMemMode, RunError> {
    if config.sm120_alpha
        && config.memory_mode != MemoryMode::Flat
        && config.sm120_effective_l1.is_none()
    {
        return Err(RunError::new(
            "arguments",
            "SM120 cache mode requires --sm120-effective-l1; use the flat mode only with --fast-frontier-only",
        ));
    }
    if config.sm120_effective_l1.is_some()
        && (!config.sm120_alpha || config.memory_mode == MemoryMode::Flat)
    {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l1 requires an sm_120 architecture card and --mode l1 or --mode l1-l2",
        ));
    }
    if config.sm120_effective_l2.is_some()
        && (!config.sm120_alpha || config.memory_mode != MemoryMode::L1L2)
    {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l2 requires an sm_120 architecture card and --mode l1-l2",
        ));
    }
    if config.dirty_writeback_capacity_sectors == Some(0) {
        return Err(RunError::new(
            "arguments",
            "dirty writeback capacity must be positive",
        ));
    }
    if config.dirty_writeback_capacity_sectors.is_some()
        && (config.memory_mode != MemoryMode::L1L2 || !config.carded_dram_bandwidth)
    {
        return Err(RunError::new(
            "arguments",
            "dirty writeback requires --mode l1-l2 together with --carded-dram",
        ));
    }
    if config.coherence_service_cycles == Some(0) || config.coherence_capacity_sectors == Some(0) {
        return Err(RunError::new(
            "arguments",
            "coherence service cycles and capacity must be positive",
        ));
    }
    if config.coherence_capacity_sectors.is_some() != config.coherence_service_cycles.is_some() {
        return Err(RunError::new(
            "arguments",
            "coherence service cycles and capacity must be supplied together",
        ));
    }
    if config.l1_visibility_protocol && config.coherence_service_cycles.is_none() {
        return Err(RunError::new(
            "arguments",
            "L1 visibility protocol requires --coherence-service-cycles and --coherence-capacity-sectors",
        ));
    }
    if config.l1_resident_writeback
        && (config.coherence_service_cycles.is_none()
            || config.dirty_writeback_capacity_sectors.is_none()
            || config.memory_mode != MemoryMode::L1L2
            || !config.carded_dram_bandwidth
            || !config.l1_visibility_protocol)
    {
        return Err(RunError::new(
            "arguments",
            "L1 resident writeback requires l1-l2, --carded-dram, --dirty-writeback-capacity-sectors, the coherence service pair, and --l1-visibility-protocol",
        ));
    }
    if config.coherence_service_cycles.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "coherence service requires --mode l1-l2",
        ));
    }
    if config.memory_mode == MemoryMode::Flat {
        if config.carded_dram_bandwidth {
            return Err(RunError::new(
                "arguments",
                "--carded-dram requires --mode l1-l2",
            ));
        }
        return Ok(ReplayMemMode::Flat);
    }

    let mut l1 = if let Some(effective) = config.sm120_effective_l1 {
        effective.as_l1_config()
    } else {
        let pchase_path = config.pchase_card.as_ref().ok_or_else(|| {
            RunError::new(
                "arguments",
                "--pchase-card is required for l1 and l1-l2 modes",
            )
        })?;
        let pchase = load_pchase_card(pchase_path)?;
        let geometry = pchase
            .resolve_walk_geometry_for_arch(issue.arch(), None)
            .map_err(at("L1 geometry"))?;
        issue.l1_config(geometry.capacity.line_capacity, geometry.mshr.count)
    };
    if issue.arch() == sim_arch::ArchId::Sm86 && config.sm120_effective_l1.is_none() {
        let geometry = arch_card
            .require_l1_geometry()
            .map_err(at("SM86 L1 set geometry"))?;
        // The card documents the set/way domain, while the p-chase resolver
        // retains the active carveout capacity.  Keep both sources explicit:
        // the pressure-bounded modulo/folded candidate is an effective
        // scenario and the proprietary hash remains unknown; global capacity
        // still follows the p-chase point.
        l1.set_count = Some(geometry.sets);
        l1.ways = Some(geometry.ways);
        l1.set_index = L1SetIndex::ModuloXorFallback;
    }
    if config.memory_mode == MemoryMode::L1 {
        if config.carded_dram_bandwidth {
            return Err(RunError::new(
                "arguments",
                "--carded-dram requires --mode l1-l2",
            ));
        }
        return Ok(ReplayMemMode::L1Const(l1));
    }

    let l2 = if let Some(effective) = config.sm120_effective_l2 {
        if effective.write_policy != L2_WRITE_ALLOCATE_POLICY {
            return Err(RunError::new(
                "SM120 effective L2 configuration",
                format!(
                    "unsupported effective L2 write policy {:?}; expected {:?}",
                    effective.write_policy, L2_WRITE_ALLOCATE_POLICY
                ),
            ));
        }
        L2Config::from_effective_geometry_for_arch(
            issue.arch(),
            effective.capacity_bytes,
            effective.line_bytes,
            effective.sector_bytes,
            effective.num_slices,
            effective.associativity,
            effective.replacement,
            effective.slice_hash,
        )
        .map_err(at("SM120 effective L2 configuration"))?
    } else {
        let l2_geometry = arch_card.require_l2_geometry().map_err(at("L2 geometry"))?;
        L2Config::from_carded_geometry_for_arch(
            issue.arch(),
            l2_geometry.capacity_bytes,
            l2_geometry.line_bytes,
            l2_geometry.sector_bytes,
            l2_geometry.num_slices,
            l2_geometry.policy.as_deref(),
        )
        .map_err(at("L2 configuration"))?
    };
    let dram = if config.carded_dram_bandwidth {
        let card = arch_card
            .require_dram_bandwidth()
            .map_err(at("DRAM bandwidth"))?;
        let bytes_per_sec = config
            .dram_service_rate_bytes_per_sec
            .or_else(|| card.validated_peak_bytes_per_second().ok())
            .ok_or_else(|| RunError::new("DRAM bandwidth", "no validated service rate"))?;
        if bytes_per_sec == 0 {
            return Err(RunError::new(
                "DRAM bandwidth",
                "service rate must be positive",
            ));
        }
        let _card_peak = card
            .validated_peak_bytes_per_second()
            .map_err(at("DRAM bandwidth"))?;
        let sm_clock_hz = card.locked_sm_clock_hz().map_err(at("DRAM clock"))?;
        match config.dirty_writeback_capacity_sectors {
            Some(capacity_sectors) => DramBw::CardedWriteback {
                arch: issue.arch(),
                bytes_per_sec,
                sm_clock_hz,
                capacity_sectors,
            },
            None => DramBw::Carded {
                arch: issue.arch(),
                bytes_per_sec,
                sm_clock_hz,
            },
        }
    } else {
        DramBw::Unmodeled { arch: issue.arch() }
    };
    let coherence = config
        .coherence_service_cycles
        .zip(config.coherence_capacity_sectors)
        .map(|(cycles, capacity)| CoherenceConfig::new(cycles, capacity))
        .transpose()
        .map_err(at("coherence configuration"))?
        .map(|value| {
            value
                .with_l1_visibility_protocol(config.l1_visibility_protocol)
                .with_l1_resident_writeback(config.l1_resident_writeback)
        });
    Ok(ReplayMemMode::L1SharedL2 {
        l1,
        l2,
        dram,
        coherence,
    })
}

pub fn run(config: &RunConfig) -> Result<RunReport, RunError> {
    let total_started = Instant::now();
    let arch_card = load_arch_card(&config.arch_card)?;
    let arch = arch_card.arch_id().map_err(at("architecture card"))?;
    let resolved_config = config.with_architecture_defaults(arch)?;
    let config = &resolved_config;
    #[cfg(feature = "performance-counters")]
    let performance_enabled = std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some();
    let candidate_inputs = usize::from(config.memory_service_candidates.is_some())
        + usize::from(config.memory_admission_candidates.is_some())
        + usize::from(config.memory_admission_feedback.is_some());
    if candidate_inputs != 0 && arch != sim_arch::ArchId::Sm86 {
        return Err(RunError::new(
            "arguments",
            "memory candidate documents are currently scoped to the sm_86 lane; an architecture-qualified candidate schema is required before applying one to sm_120",
        ));
    }
    if config.native_payload_events.is_some() && config.native_payload_events_path.is_some() {
        return Err(RunError::new(
            "arguments",
            "programmatic native payload events cannot combine with --native-payload-events",
        ));
    }
    let file_payload_events = config
        .native_payload_events_path
        .as_ref()
        .map(|path| load_native_payload_events(path))
        .transpose()?;
    let native_payload_events = config.native_payload_events.clone().or(file_payload_events);
    let native_payload_records = config
        .native_payload_sidecar
        .as_ref()
        .map(|path| load_native_payload_sidecar_records(path).map_err(at("native payload sidecar")))
        .transpose()?;
    let native_payload_sidecar_records = native_payload_records.as_ref().map(Vec::len);
    let native_payload_event_count = native_payload_events.as_ref().map(Vec::len);
    let identity_complete_records = native_payload_records.as_ref().map(|records| {
        records
            .iter()
            .filter(|record| record.identity_complete())
            .count()
    });
    let native_payloads = native_payload_records.as_ref().map(|records| {
        records
            .iter()
            .map(|record| record.payload())
            .collect::<Vec<_>>()
    });
    if candidate_inputs > 1 {
        return Err(RunError::new(
            "arguments",
            "memory-service, memory-admission, and feedback candidate files are mutually exclusive",
        ));
    }
    if native_payload_events
        .as_ref()
        .is_some_and(|events| !events.is_empty() && candidate_inputs != 0)
    {
        return Err(RunError::new(
            "arguments",
            "event-time payload admissions cannot combine with memory candidate lanes",
        ));
    }
    if config.fast_frontier_only && candidate_inputs != 0 {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with timing candidate documents",
        ));
    }
    if config.fast_frontier_only
        && (config.hypothetical_issue_ports.is_some() || config.architecture_spec.is_some())
    {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with timing/resource overrides",
        ));
    }
    if config.fast_frontier_only && config.native_payload_sidecar.is_some() {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with a native payload sidecar because it skips memory replay",
        ));
    }
    if config.fast_frontier_only && native_payload_events.is_some() {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with event-time payload admissions",
        ));
    }
    if config.fast_frontier_only
        && (config.memory_mode != MemoryMode::Flat
            || config.cache_scope != CacheScope::Launch
            || config.carded_dram_bandwidth
            || config.dirty_writeback_capacity_sectors.is_some()
            || config.coherence_service_cycles.is_some()
            || config.coherence_capacity_sectors.is_some())
    {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with native memory/cache, DRAM, writeback, or coherence settings",
        ));
    }
    if config.fast_frontier_only && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only requires --cache-scope launch",
        ));
    }
    if config.fast_frontier_only && config.ncu_shaped_metrics {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot emit native NCU-shaped metric projections",
        ));
    }
    if config.fast_frontier_only && config.latent_envelope {
        return Err(RunError::new(
            "arguments",
            "--latent-envelope requires native timing replay and cannot combine with --fast-frontier-only",
        ));
    }
    if config.predictive_service_profile.is_some() && !config.latent_envelope {
        return Err(RunError::new(
            "arguments",
            "--predictive-service-profile requires --latent-envelope",
        ));
    }
    if config.ncu_launch.is_some() && config.ncu_csv.is_none() {
        return Err(RunError::new(
            "arguments",
            "--ncu-launch requires --ncu-csv",
        ));
    }
    let ncu_observation = config
        .ncu_csv
        .as_ref()
        .map(|path| load_ncu_observation(path, config.ncu_launch))
        .transpose()?;
    let predictive_service_profile =
        load_predictive_service_profile(config.predictive_service_profile.as_ref())?;
    if config.cache_scope == CacheScope::StreamL2 && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--cache-scope stream-l2 requires --mode l1-l2",
        ));
    }
    if config.cache_scope == CacheScope::StreamL2 && config.kernel_id.is_some() {
        return Err(RunError::new(
            "arguments",
            "--kernel-id cannot skip launches in --cache-scope stream-l2",
        ));
    }
    if config.memory_service_candidates.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--memory-service-candidates requires --mode l1-l2",
        ));
    }
    if config.memory_service_candidates.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--memory-service-candidates currently requires --cache-scope launch",
        ));
    }
    if config.memory_admission_candidates.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-candidates requires --mode l1-l2",
        ));
    }
    if config.memory_admission_candidates.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-candidates currently requires --cache-scope launch",
        ));
    }
    if config.memory_admission_feedback.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-feedback requires --mode l1-l2",
        ));
    }
    if config.memory_admission_feedback.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-feedback currently requires --cache-scope launch",
        ));
    }
    if config.dirty_writeback_capacity_sectors.is_some() && config.cache_scope != CacheScope::Launch
    {
        return Err(RunError::new(
            "arguments",
            "dirty writeback currently requires --cache-scope launch",
        ));
    }
    if config.coherence_service_cycles.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "coherence service currently requires --cache-scope launch",
        ));
    }
    if config.l1_resident_writeback && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "L1 resident writeback currently requires --cache-scope launch",
        ));
    }
    if native_payload_events
        .as_ref()
        .is_some_and(|events| !events.is_empty())
        && (!matches!(config.memory_mode, MemoryMode::L1L2)
            || config.cache_scope != CacheScope::Launch
            || !config.carded_dram_bandwidth
            || config.dirty_writeback_capacity_sectors.is_none()
            || !config.l1_resident_writeback)
    {
        return Err(RunError::new(
            "arguments",
            "event-time payload admissions require launch-scoped l1-l2 replay, carded DRAM, resident writeback, and a dirty-writeback capacity",
        ));
    }
    let memory_service_candidates =
        load_memory_service_candidates(config.memory_service_candidates.as_ref())?;
    let memory_admission_candidates =
        load_memory_admission_candidates(config.memory_admission_candidates.as_ref())?;
    let memory_admission_feedback =
        load_memory_admission_feedback(config.memory_admission_feedback.as_ref())?;
    let specialized_loader_requested = std::env::var_os("SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD")
        .is_some()
        || std::env::var_os("SIM_RUN_STREAM_TRACE_PAYLOAD").is_some();
    let performance_diagnostics_requested =
        std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some();
    if candidate_inputs == 0
        && config.cache_scope == CacheScope::Launch
        && config.kernel_id.is_none()
        && !config.ncu_shaped_metrics
        && !config.fast_frontier_only
        && !config.sm120_alpha
        && !config.latent_envelope
        && config.dirty_writeback_capacity_sectors.is_none()
        && config.coherence_service_cycles.is_none()
        && !config.l1_visibility_protocol
        && !config.l1_resident_writeback
        && ncu_observation.is_none()
        && !specialized_loader_requested
        && !performance_diagnostics_requested
        && config.launch_resources.is_none()
        && config.native_payload_sidecar.is_none()
        && native_payload_events.is_none()
        && predictive_service_profile.is_none()
    {
        return PreparedReplay::prepare(config, CoverageAdmission::ExistingDirectBehavior)?
            .replay_all_report();
    }
    if config.kernel_id.is_some() && config.launch_resources.is_some() {
        return Err(RunError::new(
            "arguments",
            "--kernel-id cannot be combined with --launch-resources because CUPTI correlation identity is not present in the trace contract",
        ));
    }
    let mut external_resources = config
        .launch_resources
        .as_ref()
        .map(|path| resources::ResourceObservations::load(path))
        .transpose()?;
    let (issue, architecture_spec_issue_ports) = resolve_issue_with_metadata(config, &arch_card)?;
    let mode = resolve_mode(config, &arch_card, issue)?;

    #[cfg(feature = "performance-counters")]
    let setup_ns = total_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let trace_started = Instant::now();
    #[cfg(feature = "performance-counters")]
    let retain_raw_trace_payload = std::env::var_os("SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD").is_some();
    #[cfg(feature = "performance-counters")]
    let stream_trace_payload = std::env::var_os("SIM_RUN_STREAM_TRACE_PAYLOAD").is_some();
    #[cfg(not(feature = "performance-counters"))]
    let retain_raw_trace_payload = false;
    #[cfg(not(feature = "performance-counters"))]
    let stream_trace_payload = false;
    let trace = if retain_raw_trace_payload {
        TraceDir::open_direct(&config.trace_dir)
    } else if stream_trace_payload {
        TraceDir::open_direct_streaming(&config.trace_dir)
    } else {
        TraceDir::open_direct_compact(&config.trace_dir)
    }
    .map_err(at("trace"))?;
    #[cfg(feature = "performance-counters")]
    let trace_ns = trace_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let exec_started = Instant::now();
    let exec = trace.exec_info().ok_or_else(|| {
        RunError::new(
            "execution info",
            "direct trace loader did not retain execution metadata",
        )
    })?;
    #[cfg(feature = "performance-counters")]
    let exec_ns = exec_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let control_started = Instant::now();
    let control = load_control_sidecar(&config.control_bits, arch, config.sm120_alpha)?;
    validate_direct_control_identity(&config.control_bits, exec, &control)?;
    #[cfg(feature = "performance-counters")]
    let control_ns = control_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let schedule_started = Instant::now();
    let schedules = if config.sm120_alpha {
        sim_sm::build_schedules_sm120_alpha(exec, &control, alpha_config())
            .map_err(at("SM120 alpha schedule"))?
    } else {
        build_schedules(exec, &control).map_err(at("schedule"))?
    };
    #[cfg(feature = "performance-counters")]
    let schedule_ns = schedule_started.elapsed().as_nanos();

    let kernel_locs: Vec<_> = trace
        .kernels()
        .filter(|loc| config.kernel_id.is_none_or(|id| loc.kernel().id == id))
        .collect();
    if kernel_locs.is_empty() {
        return Err(RunError::new(
            "replay",
            match config.kernel_id {
                Some(id) => format!("kernel id {id} was not found"),
                None => "trace contains no kernel launches".to_owned(),
            },
        ));
    }
    if let Some(events) = native_payload_events.as_deref() {
        let launch_ids = kernel_locs
            .iter()
            .filter_map(|loc| u64::try_from(loc.kernel().id).ok())
            .collect::<BTreeSet<_>>();
        for (index, event) in events.iter().enumerate() {
            let Some(identity) = event.identity else {
                return Err(RunError::new(
                    "payload events",
                    format!("event {index} has no retained dynamic identity"),
                ));
            };
            if !launch_ids.contains(&identity.launch_id) {
                return Err(RunError::new(
                    "payload events",
                    format!(
                        "event {index} launch_id {} is absent from the selected trace launches",
                        identity.launch_id
                    ),
                ));
            }
        }
    }

    // A complete v2 sidecar is joined against the immutable trace snapshot
    // before replay. Historical/incomplete sidecars remain on the explicit
    // sidecar-only ledger path and never mutate native cache state.
    let payload_ledger = match (
        native_payload_records.as_deref(),
        native_payloads.as_deref(),
        identity_complete_records,
    ) {
        (Some(records), Some(payloads), Some(identity_complete)) => {
            let needs_trace_keys = identity_complete == records.len()
                || records.iter().all(|record| record.trace_matchable());
            let mut observed = BTreeSet::new();
            if needs_trace_keys {
                for loc in &kernel_locs {
                    observed.extend(
                        trace
                            .dynamic_memory_join_identities(loc)
                            .map_err(at("payload trace identity"))?,
                    );
                }
            }
            let validate_function_names = |records: &[NativePayloadRecordV1]| {
                for (index, record) in records.iter().enumerate() {
                    let Some(function_unique_id) = record.identity().function_unique_id else {
                        continue;
                    };
                    let Some(function_name) = record.identity().function_name.as_deref() else {
                        continue;
                    };
                    let Some(static_info) = exec.kernels.get(&function_unique_id) else {
                        return Err(RunError::new(
                            "payload identity join",
                            format!(
                                "payload record {index} function_unique_id {function_unique_id} is absent from exec-info"
                            ),
                        ));
                    };
                    if static_info.function_name != function_name {
                        return Err(RunError::new(
                            "payload identity join",
                            format!(
                                "payload record {index} function_name {function_name:?} disagrees with exec-info {:?}",
                                static_info.function_name
                            ),
                        ));
                    }
                }
                Ok::<(), RunError>(())
            };
            let (joined_payloads, join_status, identity_count, event_binding, event_count) =
                if identity_complete == records.len() {
                    validate_function_names(records)?;
                    (
                        join_native_payload_records(records, &observed)
                            .map_err(at("payload identity join"))?,
                        "matched_trace_identities",
                        records.len(),
                        "trace_locations_verified",
                        records.len(),
                    )
                } else if records.iter().all(|record| record.trace_matchable()) {
                    let enriched = enrich_native_payload_records(records, &observed)
                        .map_err(at("payload identity enrichment"))?;
                    validate_function_names(&enriched)?;
                    (
                        join_native_payload_records(&enriched, &observed)
                            .map_err(at("payload identity join"))?,
                        "enriched_matched_trace_identities",
                        enriched.len(),
                        "trace_locations_verified",
                        enriched.len(),
                    )
                } else {
                    (
                        payloads.to_vec(),
                        "not_attempted_incomplete_identity",
                        identity_complete,
                        "not_attempted_incomplete_identity",
                        0,
                    )
                };
            Some(build_payload_ledger(
                &joined_payloads,
                identity_count,
                join_status,
                event_binding,
                event_count,
            )?)
        }
        _ => None,
    };

    if config.fast_frontier_only {
        let fast_replay_started = Instant::now();
        let mut fast_frontier = Vec::with_capacity(kernel_locs.len());
        for loc in kernel_locs {
            let schedule = schedules
                .get(&loc.kernel().function_unique_id)
                .ok_or_else(|| {
                    RunError::new(
                        "core-only diagnostic",
                        format!(
                            "no schedule for function_unique_id {}",
                            loc.kernel().function_unique_id
                        ),
                    )
                })?;
            let summary = if config.sm120_alpha {
                sim_sm::sm120_fast_stage_for_trace(
                    &trace,
                    &loc,
                    schedule,
                    &arch_card,
                    alpha_config(),
                )
            } else {
                sim_sm::fast_frontier_for_trace(&trace, &loc, schedule, &arch_card)
            }
            .map_err(at("core-only diagnostic"))?;
            fast_frontier.push(FastFrontierLaunchReport {
                device_id: loc.device_id(),
                stream_id: loc.stream_id(),
                kernel_id: loc.kernel().id,
                function_unique_id: loc.kernel().function_unique_id,
                kernel_name: loc.kernel().name.clone(),
                grid: kernel_grid(loc.kernel()).map_err(at("fast launch geometry"))?,
                block: loc.kernel().block_dim.as_ref().map(|dim| {
                    // TraceDir already requires positive launch dimensions.
                    (dim.x as u32, dim.y as u32, dim.z as u32)
                }),
                summary,
            });
        }
        let fast_replay_wall_ns =
            u64::try_from(fast_replay_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let fast_total_wall_ns =
            u64::try_from(total_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        return Ok(RunReport {
            schema_version: 1,
            simulator_version: env!("CARGO_PKG_VERSION"),
            architecture: arch.token().to_owned(),
            memory_mode: config.memory_mode.token(),
            cache_scope: config.cache_scope.token(),
            carded_dram_bandwidth: config.carded_dram_bandwidth,
            native_payload_sidecar_records,
            native_payload_event_count,
            payload_ledger: None,
            dirty_writeback_capacity_sectors: config.dirty_writeback_capacity_sectors,
            coherence_service_cycles: config.coherence_service_cycles,
            coherence_capacity_sectors: config.coherence_capacity_sectors,
            l1_visibility_protocol: config.l1_visibility_protocol,
            l1_resident_writeback: config.l1_resident_writeback,
            fast_frontier_only: true,
            sm120_alpha: config.sm120_alpha,
            sm120_alpha_config: config.sm120_alpha.then(alpha_config),
            sm120_alpha_profile_id: config.sm120_alpha.then_some(Sm120AlphaConfig::PROFILE_ID),
            sm120_effective_l1: config.sm120_effective_l1,
            sm120_effective_l2: config.sm120_effective_l2,
            hypothetical_issue_ports: config
                .hypothetical_issue_ports
                .or(architecture_spec_issue_ports),
            architecture_spec: config
                .architecture_spec
                .as_ref()
                .map(|path| path.display().to_string()),
            memory_service_candidate_count: 0,
            memory_admission_candidate_count: 0,
            memory_admission_feedback_candidate_id: None,
            launches: Vec::new(),
            fast_frontier: Some(fast_frontier),
            cache_session: None,
            ncu_observation,
            latent_envelope: None,
            sim_total_wall_ns: Some(fast_total_wall_ns),
            sim_replay_wall_ns: Some(fast_replay_wall_ns),
        });
    }

    let mut launches = Vec::with_capacity(kernel_locs.len());
    let replay_started = Instant::now();
    let cache_session = match config.cache_scope {
        CacheScope::Launch => {
            for loc in kernel_locs {
                let launch_payload_events = native_payload_events
                    .as_deref()
                    .map(|events| {
                        let launch_id = u64::try_from(loc.kernel().id).unwrap_or_default();
                        events
                            .iter()
                            .filter(|event| {
                                event
                                    .identity
                                    .is_some_and(|identity| identity.launch_id == launch_id)
                            })
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let outcome = if let Some(candidate) = memory_admission_feedback {
                    replay_trace_kernel_with_memory_admission_feedback(
                        &trace, &loc, &schedules, &arch_card, &issue, mode, candidate,
                    )
                } else if let Some(candidates) = memory_admission_candidates.as_deref() {
                    replay_trace_kernel_with_memory_admission_candidates(
                        &trace, &loc, &schedules, &arch_card, &issue, mode, candidates,
                    )
                } else if let Some(candidates) = memory_service_candidates.as_deref() {
                    replay_trace_kernel_with_memory_service_candidates(
                        &trace, &loc, &schedules, &arch_card, &issue, mode, candidates,
                    )
                } else if !launch_payload_events.is_empty() {
                    replay_trace_kernel_with_l1_payload_events(
                        &trace,
                        &loc,
                        &schedules,
                        &arch_card,
                        &issue,
                        mode,
                        &launch_payload_events,
                    )
                } else {
                    replay_trace_kernel(&trace, &loc, &schedules, &arch_card, &issue, mode)
                }
                .map_err(at("replay"))?;
                let result = outcome.into_full().ok_or_else(|| {
                    RunError::new(
                        "replay",
                        "direct timing replay requires complete launch coverage",
                    )
                })?;
                let external = external_resources
                    .as_mut()
                    .map(|records| {
                        records.take(
                            loc.device_id(),
                            loc.stream_id(),
                            &loc.kernel().name,
                            kernel_grid(loc.kernel()).map_err(at("launch resource identity"))?,
                            loc.kernel()
                                .block_dim
                                .as_ref()
                                .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32)),
                        )
                    })
                    .transpose()?;
                launches.push(launch_report(
                    result,
                    &arch_card,
                    external.as_ref(),
                    None,
                    memory_admission_feedback,
                    config.ncu_shaped_metrics,
                )?);
            }
            None
        }
        CacheScope::StreamL2 => {
            let first = kernel_locs[0];
            if kernel_locs.iter().any(|loc| {
                loc.device_id() != first.device_id() || loc.stream_id() != first.stream_id()
            }) {
                return Err(RunError::new(
                    "arguments",
                    "--cache-scope stream-l2 requires every launch in the trace to belong to one device and stream",
                ));
            }
            let mut session = StreamL2ReplaySession::new(
                &trace,
                &arch_card,
                &issue,
                mode,
                first.device_id(),
                first.stream_id(),
            )
            .map_err(at("replay session"))?;
            let mut seeded_by_kernel_ids = Vec::new();
            for (sequence_index, loc) in kernel_locs.into_iter().enumerate() {
                let outcome = session
                    .replay_next(&trace, &loc, &schedules, &arch_card, &issue, mode)
                    .map_err(at("replay session"))?;
                let result = outcome.into_full().ok_or_else(|| {
                    RunError::new(
                        "replay",
                        "direct timing replay requires complete launch coverage",
                    )
                })?;
                let external = external_resources
                    .as_mut()
                    .map(|records| {
                        records.take(
                            loc.device_id(),
                            loc.stream_id(),
                            &loc.kernel().name,
                            kernel_grid(loc.kernel()).map_err(at("launch resource identity"))?,
                            loc.kernel()
                                .block_dim
                                .as_ref()
                                .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32)),
                        )
                    })
                    .transpose()?;
                launches.push(launch_report(
                    result,
                    &arch_card,
                    external.as_ref(),
                    Some(CacheHandoffReport {
                        scope: config.cache_scope.token(),
                        sequence_index,
                        seeded_by_kernel_ids: seeded_by_kernel_ids.clone(),
                    }),
                    None,
                    config.ncu_shaped_metrics,
                )?);
                seeded_by_kernel_ids.push(loc.kernel().id);
            }
            Some(CacheSessionReport {
                scope: config.cache_scope.token(),
                device_id: session.device_id(),
                stream_id: session.stream_id(),
                completed_launches: session.completed_launches(),
                cumulative_l2: l2_counter_report(&session.cumulative_l2_counters()),
            })
        }
    };
    let replay_ns = replay_started.elapsed().as_nanos();
    let replay_wall_ns = u64::try_from(replay_ns).unwrap_or(u64::MAX);
    let total_wall_ns = u64::try_from(total_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let latent_envelope = if config.latent_envelope {
        Some(build_latent_envelope(
            &launches,
            ncu_observation.is_some(),
            predictive_service_profile.as_ref(),
            config.service_stage_role,
        )?)
    } else {
        None
    };
    let report = RunReport {
        schema_version: 1,
        simulator_version: env!("CARGO_PKG_VERSION"),
        architecture: arch.token().to_owned(),
        memory_mode: config.memory_mode.token(),
        cache_scope: config.cache_scope.token(),
        carded_dram_bandwidth: config.carded_dram_bandwidth,
        native_payload_sidecar_records,
        native_payload_event_count,
        payload_ledger,
        dirty_writeback_capacity_sectors: config.dirty_writeback_capacity_sectors,
        coherence_service_cycles: config.coherence_service_cycles,
        coherence_capacity_sectors: config.coherence_capacity_sectors,
        l1_visibility_protocol: config.l1_visibility_protocol,
        l1_resident_writeback: config.l1_resident_writeback,
        fast_frontier_only: false,
        sm120_alpha: config.sm120_alpha,
        sm120_alpha_config: config.sm120_alpha.then(alpha_config),
        sm120_alpha_profile_id: config.sm120_alpha.then_some(Sm120AlphaConfig::PROFILE_ID),
        sm120_effective_l1: config.sm120_effective_l1,
        sm120_effective_l2: config.sm120_effective_l2,
        hypothetical_issue_ports: config
            .hypothetical_issue_ports
            .or(architecture_spec_issue_ports),
        architecture_spec: config
            .architecture_spec
            .as_ref()
            .map(|path| path.display().to_string()),
        memory_service_candidate_count: memory_service_candidates.as_ref().map_or(0, Vec::len),
        memory_admission_candidate_count: memory_admission_candidates.as_ref().map_or(0, Vec::len),
        memory_admission_feedback_candidate_id: memory_admission_feedback
            .map(|candidate| candidate.candidate_id),
        launches,
        fast_frontier: None,
        cache_session,
        ncu_observation,
        latent_envelope,
        sim_total_wall_ns: Some(total_wall_ns),
        sim_replay_wall_ns: Some(replay_wall_ns),
    };
    if let Some(resources) = external_resources.as_ref() {
        resources.finish()?;
    }
    #[cfg(feature = "performance-counters")]
    if performance_enabled {
        eprintln!(
            "SIM_PHASE_PERF setup_ns={setup_ns} trace_ns={trace_ns} exec_ns={exec_ns} \
             control_ns={control_ns} schedule_ns={schedule_ns} replay_ns={replay_ns} \
             run_total_ns={}",
            total_started.elapsed().as_nanos(),
        );
    }
    Ok(report)
}

