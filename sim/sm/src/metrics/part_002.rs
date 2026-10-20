// Lexical implementation section 3/3; included by the parent facade.
impl SmActivity {
    pub(crate) fn from_lifetimes(
        warp_lifetimes: Vec<WarpLifetime>,
        instructions_per_smsp: [u64; 4],
        issue_active_cycles_per_smsp: [u64; 4],
        max_resident_warps_per_sm: u32,
    ) -> Self {
        let warp_active_cycles: u128 = warp_lifetimes
            .iter()
            .map(|w| u128::from(w.retirement.saturating_sub(w.admission)))
            .sum();
        let mut intervals: Vec<_> = warp_lifetimes
            .iter()
            .map(|w| (w.admission, w.retirement))
            .collect();
        let sm_active_cycles = union_length(&mut intervals);
        let mut endpoints: Vec<(u64, i32)> = warp_lifetimes
            .iter()
            .flat_map(|w| [(w.admission, 1_i32), (w.retirement, -1_i32)])
            .collect();
        endpoints.sort_unstable_by_key(|&(time, delta)| (time, delta));
        let mut live = 0_i32;
        let mut peak_live_warps = 0_u32;
        for (_, delta) in endpoints {
            live = live.saturating_add(delta);
            peak_live_warps = peak_live_warps.max(u32::try_from(live).unwrap_or(0));
        }
        let smsp_active_cycles = std::array::from_fn(|sc| {
            intervals.clear();
            intervals.extend(
                warp_lifetimes
                    .iter()
                    .filter(|w| w.warp_slot as usize % 4 == sc)
                    .map(|w| (w.admission, w.retirement)),
            );
            union_length(&mut intervals)
        });
        Self {
            warp_active_cycles,
            sm_active_cycles,
            max_resident_warps_per_sm,
            peak_live_warps,
            smsp_active_cycles,
            instructions_per_smsp,
            issue_active_cycles_per_smsp,
            front_end_instruction_read_sectors: 0,
            warp_lifetimes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(off: usize, len: u16) -> AccessKind {
        AccessKind::GlobalEligible {
            off,
            len,
            token: CacheOpToken::LdgECa,
        }
    }

    #[test]
    fn reuse_observer_deduplicates_sectors_and_tracks_proxy_monotonicity() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 0, 0, 1];
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(1, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics
            .observe_reuse_memory(
                AccessKind::GlobalBypass {
                    off: 3,
                    len: 2,
                    token: CacheOpToken::Atom,
                    l2_requests: 1,
                },
                &sectors,
            )
            .unwrap();

        let reuse = &metrics.reuse_interval;
        assert_eq!(reuse.streams, 1);
        assert_eq!(reuse.line_references, 4);
        assert_eq!(reuse.stream_unique_lines, 2);
        assert_eq!(reuse.cold_line_references, 2);
        assert_eq!(reuse.reused_line_references, 2);
        assert_eq!(reuse.sector_references, 5);
        assert_eq!(reuse.cold_sector_references, 3);
        assert_eq!(reuse.reused_sector_references, 2);
        assert_eq!(reuse.line_reuse_interval_histogram[0], 1);
        assert_eq!(reuse.line_reuse_interval_histogram[1], 1);
        assert_eq!(reuse.sector_reuse_interval_histogram[0], 1);
        assert_eq!(reuse.sector_reuse_interval_histogram[1], 1);
        assert_eq!(reuse.line_proxy_miss_references[0], 4);
        assert_eq!(reuse.line_proxy_miss_references[1], 3);
        assert_eq!(reuse.line_proxy_miss_references[2], 2);
        assert!(
            reuse
                .line_proxy_miss_references
                .windows(2)
                .all(|window| window[0] >= window[1])
        );
        assert_eq!(reuse.by_class["global_load"].line_references, 3);
        assert_eq!(reuse.by_class["atomic_or_reduction"].sector_references, 2);
    }

    #[test]
    fn reuse_observer_resets_cold_state_at_warp_boundary() {
        let mut metrics = TraceMetrics::default();
        let sectors = [7_i64];
        metrics.begin_reuse_warp_stream();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.begin_reuse_warp_stream();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        assert_eq!(metrics.reuse_interval.streams, 2);
        assert_eq!(metrics.reuse_interval.line_references, 2);
        assert_eq!(metrics.reuse_interval.stream_unique_lines, 2);
        assert_eq!(metrics.reuse_interval.cold_line_references, 2);
        assert_eq!(metrics.reuse_interval.reused_line_references, 0);
    }

    #[test]
    fn exact_mrc_respects_same_tick_batch_and_sector_refill() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 0];
        // Two lines in one logical tick are both cold, but neither line is an
        // intervening reference for the next tick's reuse of line zero.
        metrics.observe_reuse_memory(load(0, 2), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        assert_eq!(mrc.line_references, 3);
        assert_eq!(mrc.cold_line_references, 2);
        assert_eq!(mrc.sector_references, 3);
        assert_eq!(mrc.cold_sector_references, 2);

        let capacity_one = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 1)
            .unwrap();
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        // The line-zero reuse has RD=0 because line one was in the same prior
        // tick. It therefore hits at capacity one and two.
        assert_eq!(mrc.line_miss_references[capacity_one], 2);
        assert_eq!(mrc.line_miss_references[capacity_two], 2);
        assert_eq!(mrc.sector_miss_references[capacity_one], 2);
        assert_eq!(mrc.sector_miss_references[capacity_two], 2);
        // The order-independent envelope admits line one before line zero in
        // the prior batch, so the final reuse can have RD=1.
        assert_eq!(mrc.line_miss_references_upper[capacity_one], 3);
        assert_eq!(mrc.sector_miss_references_upper[capacity_one], 3);
        assert_eq!(mrc.maximum_line_rd_envelope_width, 1);
    }

    #[test]
    fn exact_mrc_tracks_sector_validity_under_line_eviction() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 8, 1];
        // Tick 1: line zero sector zero; tick 2: line one; tick 3: line two;
        // tick 4: line zero sector one. At C=2 the line is evicted (two
        // intervening distinct lines), so the final reference is a line-tag
        // miss rather than a resident-line sector fill.
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(1, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(3, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        assert_eq!(mrc.line_miss_references[capacity_two], 4);
        assert_eq!(
            mrc.sector_line_tag_miss_references[capacity_two],
            mrc.sector_miss_references[capacity_two]
        );
    }

    #[test]
    fn exact_mrc_marks_reused_sector_miss_after_intervening_line_eviction() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 8];
        // A sector zero reference, two intervening distinct lines, then the
        // same sector. The line-level RD is two, so sector zero is not a hit
        // at a two-line fully-associative capacity.
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(1, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        assert_eq!(mrc.sector_miss_references[capacity_two], 4);
        assert_eq!(mrc.sector_line_tag_miss_references[capacity_two], 4);
    }

    #[test]
    fn exact_mrc_envelope_uses_the_reused_line_batch_not_only_previous_tick() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 8];
        // The reused line was last in a two-line batch. A different line is
        // then observed in a separate tick before line zero returns.
        metrics.observe_reuse_memory(load(0, 2), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        // Lower RD counts only line two; upper RD also admits line one from
        // the previous batch, so capacity two is an envelope boundary.
        assert_eq!(mrc.line_miss_references[capacity_two], 3);
        assert_eq!(mrc.line_miss_references_upper[capacity_two], 4);
    }

    #[test]
    fn exact_mrc_fenwick_resize_preserves_stack_counts() {
        let mut stream = LocalityMrcOriginStream {
            fenwick: LocalityMrcFenwick::new(),
            lines: BTreeMap::new(),
            batches: BTreeMap::new(),
            next_batch_id: 0,
            ..LocalityMrcOriginStream::default()
        };
        let mut report = LocalityMrcClassObservation::default();
        for line in 0_i64..5000 {
            let mut batch = BTreeMap::new();
            batch.insert(line, 1_u8);
            stream.observe(&batch, &mut report).unwrap();
        }
        let mut final_batch = BTreeMap::new();
        final_batch.insert(0_i64, 1_u8);
        stream.observe(&final_batch, &mut report).unwrap();
        let c4096 = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 4096)
            .unwrap();
        let c8192 = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 8192)
            .unwrap();
        // The final line-zero reuse has 4,999 distinct intervening lines.
        assert_eq!(report.line_miss_references[c4096], 5001);
        assert_eq!(report.line_miss_references[c8192], 5000);
        assert_eq!(report.cold_line_references, 5000);
    }
}

