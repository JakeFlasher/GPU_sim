// Lexical implementation section 7/8; included by the parent facade.
impl<'a> SmCore<'a> {fn preflight_issue_accounting(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
    ) -> Result<u64, IssueError> {
        let issue_add = u64::try_from(picks.iter().flatten().count()).map_err(|_| {
            IssueError::CounterOverflow {
                counter: "issued instruction",
            }
        })?;
        let next_issued = checked_u64_counter_add("issued instruction", self.issued, issue_add)?;
        if next_issued > self.instruction_limit {
            return Err(IssueError::InstructionCountExceedsReplay {
                issued: self.issued,
                add: issue_add,
                limit: self.instruction_limit,
            });
        }

        // (block, ordinary retirements, barrier arrivals, final-barrier
        // arrivals). Final barrier arrivals retire only after the generation
        // releases, so they must not be included in `ordinary retirements`.
        let mut block_deltas: [Option<(usize, usize, usize, usize)>; SUB_CORES] = [None; SUB_CORES];
        let mut released_blocks: [Option<usize>; SUB_CORES] = [None; SUB_CORES];
        let mut total_retirements = 0usize;
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let reaches_end = warp.next_issue_retires()?;
            let arrives = slot.op_class == OpClass::BlockBarrier;
            let retires = reaches_end && !arrives;
            let entry_index = block_deltas
                .iter()
                .position(|entry| entry.is_some_and(|(block, _, _, _)| block == warp.block))
                .or_else(|| block_deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "per-cycle block accounting",
                })?;
            let entry = block_deltas[entry_index].get_or_insert((warp.block, 0, 0, 0));
            if retires {
                entry.1 = checked_usize_counter_add("block retirement", entry.1, 1)?;
                total_retirements =
                    checked_usize_counter_add("unfinished warp retirement", total_retirements, 1)?;
            } else if arrives {
                entry.2 = checked_usize_counter_add("block barrier waiter", entry.2, 1)?;
                if reaches_end {
                    entry.3 = checked_usize_counter_add("final block barrier arrival", entry.3, 1)?;
                }
            }
        }

        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        let mut residency_released = 0u32;
        let mut shared_memory_released = 0u64;
        for (block, retirements, arrivals, final_arrivals) in block_deltas.into_iter().flatten() {
            let state = &self.block_states[block];
            let unfinished =
                state
                    .unfinished
                    .checked_sub(retirements)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "block unfinished warp",
                    })?;
            let waiting =
                checked_usize_counter_add("block barrier waiter", state.waiting, arrivals)?;
            if waiting > unfinished {
                return Err(IssueError::CounterOverflow {
                    counter: "block barrier waiter",
                });
            }
            let mut post_release_unfinished = unfinished;
            if waiting > 0 && waiting == unfinished {
                let existing_final_arrivals = self.warps
                    [state.first_warp..state.first_warp + state.admitted_warps]
                    .iter()
                    .filter(|warp| warp.at_block_barrier && warp.next == warp.insts.len())
                    .count();
                let final_retirements = checked_usize_counter_add(
                    "final block barrier retirement",
                    existing_final_arrivals,
                    final_arrivals,
                )?;
                post_release_unfinished = unfinished.checked_sub(final_retirements).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "final block barrier unfinished warp",
                    },
                )?;
                total_retirements = checked_usize_counter_add(
                    "unfinished warp retirement",
                    total_retirements,
                    final_retirements,
                )?;
                let mut last_arrival = self.t;
                for warp in &self.warps[state.first_warp..state.first_warp + state.admitted_warps] {
                    if let Some(arrival) = warp.block_barrier_arrival {
                        last_arrival = last_arrival.max(arrival);
                    }
                }
                for (subcore, pick) in picks.iter().enumerate() {
                    if pick.is_some_and(|warp| self.warps[warp].block == block) {
                        last_arrival =
                            last_arrival.max(barrier_arrival_times[subcore].unwrap_or(self.t));
                    }
                }
                checked_cycle_add("block-barrier release", last_arrival, 1)?;
                if let Some(issue) = self.barrier_issue {
                    checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        last_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
            }
            // A block can become empty as part of the barrier release when its
            // final instructions are BAR.SYNC. `post_release_unfinished`
            // accounts for that deferred retirement before releasing residency.
            if post_release_unfinished == 0 {
                for &warp_slot in &state.warp_slots {
                    if !self.resident_warp_slots.contains(&warp_slot) {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[state.first_warp].launch_index,
                            warp_slot,
                        });
                    }
                }
                let release_index = released_blocks.iter().position(Option::is_none).ok_or(
                    IssueError::CounterOverflow {
                        counter: "CTA barrier diagnostic released block",
                    },
                )?;
                released_blocks[release_index] = Some(block);
                residency_released = checked_u32_counter_add(
                    "resident warp retirement",
                    residency_released,
                    state.warp_count,
                )?;
                shared_memory_released = checked_u64_counter_add(
                    "resident shared-memory retirement",
                    shared_memory_released,
                    state.shared_memory_bytes,
                )?;
            }
        }
        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        self.resident_warps.checked_sub(residency_released).ok_or(
            IssueError::CounterUnderflow {
                counter: "resident warp",
            },
        )?;
        self.resident_shared_memory_bytes
            .checked_sub(shared_memory_released)
            .ok_or(IssueError::CounterUnderflow {
                counter: "resident shared-memory byte",
            })?;
        if released_blocks.iter().any(Option::is_some) {
            // `retire_warp` admits the next launch only after applying these
            // releases.  Validate that entire admission chain now, while the
            // issue cycle is still transactional; otherwise a deferred counter
            // overflow (for example, the arrival-stamp increment) could retire
            // an earlier warp before returning its error.
            let mut projected_resident_slots = (*self.resident_warp_slots).clone();
            let mut projected_resident_blocks = self
                .block_states
                .iter()
                .filter(|state| state.unfinished != 0)
                .count() as u32;
            for block in released_blocks.into_iter().flatten() {
                projected_resident_blocks = projected_resident_blocks.checked_sub(1).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "resident block",
                    },
                )?;
                for &warp_slot in &self.block_states[block].warp_slots {
                    let removed = projected_resident_slots.remove(&warp_slot);
                    if !removed {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[self.block_states[block].first_warp]
                                .launch_index,
                            warp_slot,
                        });
                    }
                }
            }
            self.preflight_block_admissions(
                self.resident_warps.checked_sub(residency_released).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "resident warp",
                    },
                )?,
                self.unfinished_warps.checked_sub(total_retirements).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "unfinished warp",
                    },
                )?,
                self.resident_shared_memory_bytes
                    .checked_sub(shared_memory_released)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "resident shared-memory byte",
                    })?,
                projected_resident_blocks,
                projected_resident_slots,
                self.arrival_stamp,
            )?;
        }
        Ok(next_issued)
    }

    /// Validate admissions that a successful retirement will trigger without
    /// mutating the actor.  The commit path in [`retire_warp`] repeats these
    /// already-checked operations; keeping this pass side-effect free preserves
    /// the issue-cycle transaction boundary when a checked counter can fail.
    fn preflight_block_admissions(
        &self,
        mut resident_warps: u32,
        mut unfinished_warps: usize,
        mut resident_shared_memory_bytes: u64,
        mut resident_blocks: u32,
        mut resident_warp_slots: HashSet<i32>,
        mut arrival_stamp: u64,
    ) -> Result<(), IssueError> {
        let mut next_block = self.next_block;
        while next_block < self.blocks.len() {
            let block = &self.blocks[next_block];
            let live_warps = self.block_live_warps[next_block];
            let shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let next_resident_blocks = resident_blocks
                .checked_add(u32::from(live_warps != 0))
                .ok_or(IssueError::CounterOverflow {
                    counter: "resident blocks",
                })?;
            let Some(next_resident_warps) = resident_warps.checked_add(live_warps) else {
                if resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) =
                resident_shared_memory_bytes.checked_add(shared_memory_bytes)
            else {
                if resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = block
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| resident_warp_slots.contains(slot));
            if resident_warps > 0
                && (next_resident_blocks > self.params.max_resident_blocks_per_sm
                    || next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
            {
                break;
            }

            // Keep the preflight in lockstep with `admit_blocks`: this
            // checked addition is otherwise the last fallible counter before
            // the first warp of a successor CTA is appended.
            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            unfinished_warps = unfinished_warps.checked_add(admitted_warps).ok_or(
                IssueError::CounterOverflow {
                    counter: "unfinished warp",
                },
            )?;

            // Construction validates every dynamic PC and physical slot, but
            // repeat the fallible joins here so this preflight remains a true
            // boundary if an in-memory replay is assembled by a future caller.
            for warp in &block.warps {
                let slot =
                    u32::try_from(warp.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })?;
                if slot >= self.physical_warp_slots {
                    return Err(IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    });
                }
                if !warp.insts.is_empty() {
                    lookup_dense(
                        &self.schedule_slots,
                        self.schedule,
                        warp.insts[0].pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                }
            }
            arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                arrival_stamp,
                u64::from(live_warps),
            )?;
            next_block = checked_usize_counter_add("next block", next_block, 1)?;
            resident_blocks = next_resident_blocks;
            resident_warps = next_resident_warps;
            resident_shared_memory_bytes = next_resident_shared_memory_bytes;
            for slot in pending_slots {
                resident_warp_slots.insert(slot);
            }
        }
        Ok(())
    }

    fn preflight_result_writeback_priority(
        &self,
        picks: &[Option<usize>; SUB_CORES],
    ) -> Result<(), IssueError> {
        let Some(state) = self.result_writebacks.as_ref() else {
            return Ok(());
        };
        let mut collisions = 0u64;
        let mut deferral_cycles = 0u64;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let slot = self.warps[wix]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let Some(ResultWritebackKind::FixedHadd2 { bank }) =
                result_writeback_kind(slot, self.params.arch)
            else {
                continue;
            };
            let fixed_at = checked_cycle_add(
                "HADD2 result writeback",
                self.t,
                state.timing.hadd2_writeback_cycles,
            )?;
            for event in &state.pending {
                if event.subcore == subcore as u8
                    && event.bank == bank
                    && event.writeback_at == fixed_at
                {
                    checked_cycle_add(
                        "deferred L1TEX writeback",
                        event.writeback_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    checked_cycle_add(
                        "deferred L1TEX visibility",
                        event.visible_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    collisions = checked_u64_counter_add(
                        "fixed-over-L1TEX collision preflight",
                        collisions,
                        1,
                    )?;
                    deferral_cycles = checked_u64_counter_add(
                        "L1TEX writeback deferral-cycle preflight",
                        deferral_cycles,
                        state.timing.variable_deferral_cycles,
                    )?;
                }
            }
        }
        checked_u64_counter_add(
            "fixed-over-L1TEX collision diagnostic",
            self.readiness.result_writeback.fixed_over_l1tex_collisions,
            collisions,
        )?;
        checked_u64_counter_add(
            "L1TEX writeback deferral-cycle diagnostic",
            self.readiness.result_writeback.l1tex_deferral_cycles,
            deferral_cycles,
        )?;
        Ok(())
    }

    fn register_l1tex_writeback(
        &mut self,
        subcore: usize,
        warp: usize,
        kind: ResultWritebackKind,
        barrier: Option<DepBarrier>,
        visible_at: Option<u64>,
    ) -> Result<(), IssueError> {
        let l1_hit_latency = self.params.l1_hit_latency;
        let (
            Some(state),
            ResultWritebackKind::L1TexStrongSmU32 { bank },
            Some(barrier),
            Some(visible_at),
        ) = (self.result_writebacks.as_mut(), kind, barrier, visible_at)
        else {
            return Ok(());
        };
        let hit_floor = checked_cycle_add("L1 hit-pipeline floor", self.t, l1_hit_latency)?;
        if visible_at != hit_floor
            || l1_hit_latency < state.timing.l1tex_writeback_to_visibility_cycles
        {
            return Ok(());
        }
        let writeback_at = visible_at - state.timing.l1tex_writeback_to_visibility_cycles;
        state.pending.push(PendingVariableWriteback {
            subcore: subcore as u8,
            bank,
            writeback_at,
            visible_at,
            warp,
            barrier,
        });
        Ok(())
    }

    fn apply_fixed_result_priority(
        &mut self,
        subcore: usize,
        kind: ResultWritebackKind,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let (Some(state), ResultWritebackKind::FixedHadd2 { bank }) =
            (self.result_writebacks.as_mut(), kind)
        else {
            return Ok(());
        };
        let fixed_at = checked_cycle_add(
            "HADD2 result writeback",
            self.t,
            state.timing.hadd2_writeback_cycles,
        )?;
        let (warps, diagnostics) = (&mut self.warps, &mut projected.result_writeback);
        for event in &mut state.pending {
            if event.subcore != subcore as u8
                || event.bank != bank
                || event.writeback_at != fixed_at
            {
                continue;
            }
            event.writeback_at = checked_cycle_add(
                "deferred L1TEX writeback",
                event.writeback_at,
                state.timing.variable_deferral_cycles,
            )?;
            event.visible_at = checked_cycle_add(
                "deferred L1TEX visibility",
                event.visible_at,
                state.timing.variable_deferral_cycles,
            )?;
            warps[event.warp].stamp_clear(
                Some(event.barrier),
                event.visible_at,
                DependencyOrigin::L1Tex,
            );
            diagnostics.fixed_over_l1tex_collisions = checked_u64_counter_add(
                "fixed-over-L1TEX collision diagnostic",
                diagnostics.fixed_over_l1tex_collisions,
                1,
            )?;
            diagnostics.l1tex_deferral_cycles = checked_u64_counter_add(
                "L1TEX writeback deferral-cycle diagnostic",
                diagnostics.l1tex_deferral_cycles,
                state.timing.variable_deferral_cycles,
            )?;
        }
        Ok(())
    }

    /// Probe the per-SMSP Immediate Constant Cache for the selected
    /// instructions.  A tag miss consumes an issue attempt but does not
    /// dispatch or advance the warp; the warp is parked until the line's
    /// deterministic GCC/L1C return time.  The state is cloned first so a
    /// later cycle-preflight error cannot leave a partially reserved constant
    /// request behind.
    fn preflight_immediate_constant(
        &self,
        picks: [Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<ImmediateConstantPlan, IssueError> {
        let Some(mut cache) = self.immediate_constant_cache.as_ref().cloned() else {
            return Ok(ImmediateConstantPlan {
                picks,
                cache: None,
                waits: [None; SUB_CORES],
                extra_waits: Vec::new(),
                miss_observed: false,
            });
        };
        let mut effective = picks;
        let mut waits = [None; SUB_CORES];
        let mut miss_observed = false;
        for (subcore, pick) in picks.into_iter().enumerate() {
            let Some(warp) = pick else { continue };
            let state = &self.warps[warp];
            let pc = state.insts[state.next].pc;
            let Some(lines) = self.immediate_constant_lines(pc) else {
                continue;
            };
            if lines.is_empty() {
                continue;
            }
            if state.imc_wait_until > now {
                effective[subcore] = None;
                waits[subcore] = Some((warp, state.imc_wait_until));
                miss_observed = true;
                continue;
            }
            if let Some(ready) = cache[subcore].reserve(lines, now, self.params.l1_miss_latency)?
                && ready > now
            {
                effective[subcore] = None;
                waits[subcore] = Some((warp, ready));
                miss_observed = true;
            }
        }
        Ok(ImmediateConstantPlan {
            picks: effective,
            cache: Some(cache),
            waits,
            extra_waits: Vec::new(),
            miss_observed,
        })
    }

    /// One issue cycle at `self.t` across the four sub-cores. Returns
    /// whether anything issued.
    fn issue_cycle(&mut self, out: &mut Vec<MemRequest>) -> Result<bool, IssueError> {
        let t = self.t;
        self.last_issue_picks = [None; SUB_CORES];
        let mut selected = if self.has_async_copy_protocol {
            self.pick_warps_with_memory_admission::<true>(t)
        } else {
            self.pick_warps_with_memory_admission::<false>(t)
        };
        if matches!(self.mem, MemModel::L1Shared(_))
            && let Some(subcore) = selected.iter().position(|pick| {
                pick.is_some_and(|warp| {
                    self.warps[warp]
                        .next_slot
                        .is_some_and(IssueSlot::is_cctl_invalidate_all)
                })
            })
        {
            // An invalidation is one cache-state boundary. Do not allow a
            // sibling picked in this same transaction to allocate a new fill
            // between the readiness check and invalidate_all().
            let warp = selected[subcore];
            selected = [None; SUB_CORES];
            selected[subcore] = warp;
        }
        let l0c_forced_hold: [bool; SUB_CORES] = std::array::from_fn(|subcore| {
            selected[subcore].is_some_and(|warp| {
                if self.has_async_copy_protocol {
                    self.l0c_greedy_hold_candidate::<true>(subcore, warp, t)
                } else {
                    self.l0c_greedy_hold_candidate::<false>(subcore, warp, t)
                }
            })
        });
        let imc_plan = self.preflight_immediate_constant(selected, t)?;
        let mut picks = imc_plan.picks;
        let effective_picks;
        let mut imc_cache = imc_plan.cache;
        let imc_waits = imc_plan.waits;
        let mut extra_imc_waits = imc_plan.extra_waits;
        let mut imc_miss_observed = imc_plan.miss_observed;
        if let Some(cache) = imc_cache.as_mut() {
            for (subcore, forced) in l0c_forced_hold.iter().copied().enumerate() {
                if forced && imc_waits[subcore].is_some() {
                    cache[subcore].diagnostics.greedy_switch_block_cycles = cache[subcore]
                        .diagnostics
                        .greedy_switch_block_cycles
                        .saturating_add(1);
                }
            }
        }

        // Huerta's issue loop continues through the remaining priority list
        // after a non-greedy warp misses in L0C. The previous Rust path
        // stopped after the first parked candidate, which could create a
        // false dispatch bubble when a younger sibling was also CBANK-bound.
        // Scan only when the four-cycle greedy hold did not apply, preserve
        // every line reservation made by parked candidates, and keep the
        // memory-admission candidate as a final transactional constraint.
        if imc_cache.is_some() {
            for subcore in 0..SUB_CORES {
                let Some(initial) = selected[subcore] else {
                    continue;
                };
                let held_active_window = self.hold[subcore] == Some(initial)
                    && self.immediate_constant_switch_remaining[subcore] > 0;
                if picks[subcore].is_some() || held_active_window || imc_waits[subcore].is_none() {
                    continue;
                }
                let Some(cache) = imc_cache.as_mut() else {
                    break;
                };
                let mut tried = vec![initial];
                loop {
                    let candidate = self.subcore_warps[subcore]
                        .iter()
                        .copied()
                        .filter(|warp| {
                            !tried.contains(warp)
                                && if self.has_async_copy_protocol {
                                    self.issue_ready_at_without_immediate_constant::<true>(
                                        subcore, *warp,
                                    )
                                } else {
                                    self.issue_ready_at_without_immediate_constant::<false>(
                                        subcore, *warp,
                                    )
                                }
                                .is_some_and(|ready| ready <= t)
                        })
                        .max_by_key(|warp| (self.warps[*warp].arrival, self.warps[*warp].warp_id));
                    let Some(candidate) = candidate else { break };
                    tried.push(candidate);

                    if let Some(feedback) = self.memory_admission_feedback.as_ref() {
                        let mut additions = [0usize; SUB_CORES];
                        for (other, pick) in picks.iter().enumerate() {
                            if other == subcore {
                                continue;
                            }
                            if let Some(warp) = *pick {
                                feedback.record_transaction_addition(
                                    other,
                                    self.warps[warp].insts[self.warps[warp].next].kind,
                                    &mut additions,
                                );
                            }
                        }
                        let kind = self.warps[candidate].insts[self.warps[candidate].next].kind;
                        if !feedback.permits_transaction_addition(subcore, kind, &additions, t) {
                            continue;
                        }
                    }

                    cache[subcore].diagnostics.non_greedy_scan_attempts = cache[subcore]
                        .diagnostics
                        .non_greedy_scan_attempts
                        .saturating_add(1);
                    let state = &self.warps[candidate];
                    let lines = self
                        .immediate_constant_lines(state.insts[state.next].pc)
                        .unwrap_or(&[]);
                    let wait = if state.imc_wait_until > t {
                        Some(state.imc_wait_until)
                    } else if lines.is_empty() {
                        None
                    } else {
                        cache[subcore].reserve(lines, t, self.params.l1_miss_latency)?
                    };
                    if let Some(ready) = wait.filter(|ready| *ready > t) {
                        cache[subcore].diagnostics.non_greedy_scan_misses = cache[subcore]
                            .diagnostics
                            .non_greedy_scan_misses
                            .saturating_add(1);
                        extra_imc_waits.push((candidate, ready));
                        imc_miss_observed = true;
                        continue;
                    }
                    cache[subcore].diagnostics.non_greedy_scan_successes = cache[subcore]
                        .diagnostics
                        .non_greedy_scan_successes
                        .saturating_add(1);
                    picks[subcore] = Some(candidate);
                    break;
                }
            }
        }
        effective_picks = picks;
        let memory_pipeline_plan =
            if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
                let mut selected = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    if matches!(
                        state.insts[state.next].kind,
                        AccessKind::PredicatedOffMemory
                    ) {
                        continue;
                    }
                    selected[subcore] =
                        self.memory_pipeline_slot(u64::from(state.insts[state.next].pc));
                }
                Some(memory_pipeline.plan(selected, t)?)
            } else {
                None
            };
        let scalar_execution_pipeline_plan =
            if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_ref() {
                let selected = std::array::from_fn(|subcore| {
                    picks[subcore].and_then(|warp| {
                        let slot = self.warps[warp].next_slot?;
                        matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. })
                            .then_some(slot.pipeline_class)
                    })
                });
                Some(scalar_pipeline.plan(selected, t)?)
            } else {
                None
            };
        let barrier_arrival_times = self.project_barrier_arrival_times(&picks, t)?;
        let mut async_destination_wavefronts = [0; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let kind = self.warps[wix].insts[self.warps[wix].next].kind;
            async_destination_wavefronts[subcore] =
                async_copy_destination_wavefronts(kind, &self.params)?.unwrap_or(0);
        }

        // A cycle is the caller-visible transaction boundary. Validate every
        // selected instruction's possible timing arms before any sub-core
        // mutates counters, cache state, barriers, requests, or retirement.
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let w = &self.warps[wix];
            if self.instruction_front_end.is_some() {
                w.preflight_buffered_issue(wix)?;
            }
            let slot = w
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            if slot.is_cctl_invalidate_all()
                && let MemModel::L1Shared(state) = &self.mem
            {
                // Dirty-resident mode still requires a real writeback path;
                // preserve its typed error before any issue-state mutation.
                state.l1.preflight_invalidate_all()?;
            }
            let reaches_end = w.next_issue_retires()?;
            // Final BAR.SYNC participates in the CTA rendezvous and retires
            // only once that generation releases.
            let retires_warp = reaches_end && slot.op_class != OpClass::BlockBarrier;
            let kind = w.insts[w.next].kind;
            let constant_deep = is_constant_load_opcode(slot.opcode_text())
                && self.is_constant_path_pc(u64::from(w.insts[w.next].pc));
            validate_async_copy_completion_work(kind, wix)?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(kind).is_some();
            if observe_async_copy && let Some(delivery_work) = async_copy_completion_work(kind) {
                let source_ready = match &self.mem {
                    MemModel::Flat => checked_cycle_add(
                        "flat async-copy source visibility",
                        self.t,
                        self.params.global_flat_latency,
                    )?,
                    MemModel::L1Const(_) => checked_cycle_add(
                        "constant-L1 async-copy source upper bound",
                        self.t,
                        self.params.l1_miss_latency,
                    )?,
                    MemModel::L1Shared(_) => checked_cycle_add(
                        "shared-L1 async-copy known-source floor",
                        self.t,
                        self.params.l1_hit_latency,
                    )?,
                };
                let pending_add = if matches!(self.mem, MemModel::L1Shared(_)) {
                    u32::from(async_copy_source_count(kind).unwrap_or(0))
                } else {
                    0
                };
                self.async_copies[wix].preflight_async_copy_add(
                    wix,
                    pending_add,
                    source_ready,
                    delivery_work,
                    async_destination_wavefronts[subcore],
                    (matches!(self.mem, MemModel::L1Shared(_))
                        && matches!(
                            kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::BypassL1,
                                ..
                            }
                        ))
                    .then_some(
                        bypass_async_copy_observed_topology(
                            kind,
                            w.next,
                            w.async_copy_cohorts
                                .and_then(|sidecar| sidecar.view(w.next)),
                        )
                        .unwrap_or(BypassAsyncCopyInput {
                            instruction: w.next,
                            sector_off: match kind {
                                AccessKind::GlobalAsyncCopy { off, .. } => off,
                                _ => 0,
                            },
                            sector_len: async_copy_source_count(kind).unwrap_or(0),
                            source_cohorts: 0,
                            destination_lines: 0,
                            max_bank_wavefronts: 0,
                            max_bypass_wavefronts: 0,
                            cohorts: None,
                            sector_membership_masks: None,
                        }),
                    ),
                    &self.params,
                )?;
            }
            self.preflight_issue_timing(
                kind,
                constant_deep,
                memory_pipeline_plan
                    .as_ref()
                    .and_then(|plan| plan.issues[subcore])
                    .map(|issue| issue.shared_accept_at),
                write_barrier,
                slot.control.read_barrier,
                if self.params.arch == ArchId::Sm120
                    && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text())
                {
                    0
                } else {
                    slot.control.stall
                },
                slot.control.yield_flag,
                slot.pipeline_class,
                retires_warp,
                observe_async_copy,
            )?;
        }
        let memory_times = std::array::from_fn(|subcore| {
            memory_pipeline_plan
                .as_ref()
                .and_then(|plan| plan.issues[subcore])
                .map_or(t, |issue| issue.shared_accept_at)
        });
        self.preflight_issue_memory(&picks, &async_destination_wavefronts, &memory_times)?;
        self.preflight_async_l1_hit_service(&picks, &memory_times)?;
        self.preflight_result_writeback_priority(&picks)?;
        let next_issued = self.preflight_issue_accounting(&picks, &barrier_arrival_times)?;
        // Diagnostics are preflighted last so an observational counter can
        // never mask an established execution/cache/domain error. No mutation
        // has occurred yet, so a diagnostic overflow remains transactional.
        let mut projected_readiness =
            self.project_picked_readiness(&picks, &barrier_arrival_times, t)?;
        let projected_memory_service_candidates =
            self.project_memory_service_candidates(&picks, t)?;
        if let Some(plan) = scalar_execution_pipeline_plan.as_ref() {
            projected_readiness.scalar_execution_pipeline = Some(Box::new(plan.next.diagnostics));
        }
        let memory_admission_feedback_plan =
            if let Some(feedback) = self.memory_admission_feedback.as_ref() {
                let mut issues = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    let slot = state
                        .next_slot
                        .ok_or(IssueError::MissingIssueSlot { warp })?;
                    issues[subcore] = Some(MemoryAdmissionFeedbackIssue {
                        subcore,
                        warp,
                        instruction: state.next,
                        kind: state.insts[state.next].kind,
                        prior_clock_reads: state.clock_reads_issued,
                        clock_read: is_clocklo_issue(slot),
                    });
                }
                Some(feedback.plan_issue(&issues, t)?)
            } else {
                None
            };

        if let Some(state) = self.result_writebacks.as_mut() {
            state.pending.retain(|event| event.writeback_at >= t);
        }

        // Reserve the SM86 RF read-port/RFC state only after every selected
        // instruction has passed the existing transaction preflight. One warp
        // per SMSP is selected, so these plans cannot alias another pick's
        // per-SMSP port timeline.
        let mut sm86_rf_plans: [Option<Sm86RegisterReadPlan>; SUB_CORES] = [None, None, None, None];
        if self.sm86_rf_enabled {
            for (subcore, pick) in picks.iter().enumerate() {
                let Some(warp) = *pick else { continue };
                let plan = self.sm86_register_read_plan_for(subcore, warp, t)?;
                sm86_rf_plans[subcore] = Some(plan);
            }
        }

        // Project the source-backed Huerta L0C miss-switch countdown only
        // after all ordinary preflight checks have passed.  A parked greedy
        // warp consumes one bounded opportunity per observed miss; every
        // effective instruction reloads the carded window for the next
        // instruction.  Sibling selection is therefore enabled only after
        // the four miss opportunities have elapsed.
        let mut projected_immediate_constant_switch = self.immediate_constant_switch_remaining;
        if self.params.arch == ArchId::Sm86
            && let Some(config) = self.params.immediate_constant_cache
        {
            for (subcore, selected_warp) in selected.iter().enumerate() {
                if self.hold[subcore] == *selected_warp
                    && imc_waits[subcore].is_some()
                    && projected_immediate_constant_switch[subcore] > 0
                {
                    projected_immediate_constant_switch[subcore] -= 1;
                }
            }
            for (subcore, pick) in picks.iter().enumerate() {
                if pick.is_some() {
                    projected_immediate_constant_switch[subcore] = config.switch_after_miss_cycles;
                }
            }
        }
        // An IMC miss consumes the SMSP issue opportunity even though no
        // traced instruction is dispatched.  Keep it separate from the exact
        // executed-instruction counter below.
        let mut issued_this_cycle = imc_miss_observed;
        for (sc, pick) in picks.into_iter().enumerate() {
            let Some(wix) = pick else { continue };
            issued_this_cycle = true;
            if let Some(plan) = sm86_rf_plans[sc].as_ref() {
                self.apply_sm86_register_read_plan(sc, wix, plan, &mut projected_readiness)?;
            }

            // Issue exactly one instruction from warp `wix`.
            let (
                blk_ix,
                launch_index,
                warp_slot,
                slot_class,
                pipeline_class,
                scalar_control_stall_suppressed,
                control,
                instruction,
                inst,
                warp_sectors,
                retires_warp,
                result_writeback,
                clock_read,
                constant_load,
                constant_deep,
                cctl_invalidate_all,
                fence_semantics,
            ) = {
                let w = &self.warps[wix];
                let slot = w
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
                (
                    w.block,
                    w.launch_index,
                    w.warp_slot,
                    slot.op_class,
                    slot.pipeline_class,
                    self.params.arch == ArchId::Sm120
                        && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text()),
                    slot.control,
                    w.next,
                    w.insts[w.next],
                    w.sectors,
                    {
                        let reaches_end = w.next_issue_retires()?;
                        reaches_end && slot.op_class != OpClass::BlockBarrier
                    },
                    result_writeback_kind(slot, self.params.arch),
                    is_clocklo_issue(slot),
                    is_constant_load_opcode(slot.opcode_text()),
                    self.is_constant_path_pc(u64::from(w.insts[w.next].pc)),
                    slot.is_cctl_invalidate_all(),
                    slot.visibility_fence_semantics(),
                )
            };
            let produces_dynamic_effects = !matches!(inst.kind, AccessKind::PredicatedOffMemory);
            let write_barrier = if !produces_dynamic_effects
                || (self.has_async_copy_protocol && slot_class == OpClass::AsyncCopyCommit)
            {
                None
            } else {
                control.write_barrier
            };
            let result_writeback = produces_dynamic_effects
                .then_some(result_writeback)
                .flatten();
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(inst.kind).is_some();
            let fixed_latency_status = (self.params.resource_source()
                == IssueResourceSource::Sm120Alpha)
                .then(|| {
                    self.warps[wix]
                        .next_slot
                        .map(|current| current.fixed_latency_operands())
                })
                .flatten();
            let observe_register_result = inst.effective_lane_mask() != 0
                && matches!(self.mem, MemModel::Flat)
                && fixed_latency_status
                    .and_then(|s| s.projected())
                    .is_some_and(|p| {
                        p.destinations()
                            .iter()
                            .any(|dst| !dst.register().is_constant())
                    });

            // The instruction has passed its IMC probe and is now dispatched;
            // touch each resident line in the cloned cache state.  A line that
            // was just filled is already present after `complete_ready` at the
            // top of this scheduler iteration.
            if let Some(cache) = imc_cache.as_mut()
                && let Some(lines) = self.immediate_constant_lines(inst.pc)
            {
                for &line in lines {
                    cache[sc].touch(line);
                }
            }
            self.warps[wix].imc_wait_until = 0;

            let memory_pipeline_issue = memory_pipeline_plan
                .as_ref()
                .and_then(|plan| plan.issues[sc]);
            let memory_time = memory_pipeline_issue.map_or(t, |issue| issue.shared_accept_at);
            let sm120_flat_service_completion = if self.sm120_flat_service_enabled {
                let opcode_text = self
                    .schedule
                    .issue_slot(u64::from(inst.pc))
                    .map(|slot| slot.opcode_text())
                    .unwrap_or("UNKNOWN");
                sm120_flat_service_work(inst.kind, opcode_text).map(|(global, work)| {
                    let ready = if global {
                        &mut self.sm120_flat_global_service_ready
                    } else {
                        &mut self.sm120_flat_shared_service_ready
                    };
                    let done = (*ready).max(t).saturating_add(work);
                    *ready = done;
                    done
                })
            } else if self.sm120_flat_shared_medium {
                let wavefronts = match inst.kind {
                    AccessKind::Shared { wavefronts, .. } => wavefronts,
                    _ => 0,
                };
                if wavefronts == 0 {
                    None
                } else {
                    let work = wavefronts
                        .saturating_mul(SM120_FLAT_SHARED_MEDIUM_EXTRA_CYCLES_PER_WAVEFRONT);
                    let done = self
                        .sm120_flat_shared_service_ready
                        .max(t)
                        .saturating_add(work);
                    self.sm120_flat_shared_service_ready = done;
                    Some(done)
                }
            } else if self.sm120_flat_multiwarp_service_enabled {
                let len = match inst.kind {
                    AccessKind::GlobalEligible { len, .. }
                    | AccessKind::GlobalBypass { len, .. }
                    | AccessKind::GlobalUntagged { len, .. }
                    | AccessKind::GlobalStore { len, .. }
                    | AccessKind::GlobalReduction { len, .. } => len,
                    _ => 0,
                };
                if len == 0 {
                    None
                } else {
                    let work = u64::from(len)
                        .saturating_mul(SM120_FLAT_MULTI_WARP_GLOBAL_SERVICE_CYCLES_PER_SECTOR);
                    let done = self
                        .sm120_flat_multiwarp_global_service_ready
                        .max(t)
                        .saturating_add(work)
                        .saturating_add(if self.sm120_flat_multiwarp_dense {
                            SM120_FLAT_MULTI_WARP_DENSE_EXTRA_CYCLES
                        } else if self.sm120_flat_multiwarp_medium {
                            SM120_FLAT_MULTI_WARP_MEDIUM_EXTRA_CYCLES
                        } else {
                            0
                        });
                    self.sm120_flat_multiwarp_global_service_ready = done;
                    Some(done)
                }
            } else {
                None
            };
            let sm120_flat_store_retire_completion = if self.sm120_flat_store_retire_enabled {
                let (len, per_sector) = match inst.kind {
                    AccessKind::GlobalStore { len, .. } => {
                        (u64::from(len), SM120_FLAT_STORE_RETIRE_CYCLES_PER_SECTOR)
                    }
                    AccessKind::GlobalReduction { len, .. } => (
                        u64::from(len),
                        SM120_FLAT_REDUCTION_RETIRE_CYCLES_PER_SECTOR,
                    ),
                    _ => (0, 0),
                };
                (len != 0).then(|| {
                    let done = self
                        .sm120_flat_store_retire_ready
                        .max(t)
                        .saturating_add(len.saturating_mul(per_sector));
                    self.sm120_flat_store_retire_ready = done;
                    done
                })
            } else {
                None
            };
            // Memory hierarchy mutation happens when the local address/operand
            // stages reach the SM-shared acceptor, not when the scheduler first
            // dispatches the instruction.
            let mut data_ready = if matches!(self.mem, MemModel::L1Shared(_)) {
                self.mem_step_shared(
                    sc,
                    wix,
                    instruction,
                    inst.kind,
                    constant_load,
                    constant_deep,
                    match inst.kind {
                        AccessKind::GlobalEligible { .. }
                            if opcode_has_qualifier(
                                self.schedule
                                    .issue_slot(u64::from(inst.pc))
                                    .map(|slot| slot.opcode_text())
                                    .unwrap_or(""),
                                "LTC128B",
                            ) =>
                        {
                            L2ClientScope::Ltc128b
                        }
                        AccessKind::GlobalEligible { .. } if constant_load => {
                            L2ClientScope::ConstantLdg
                        }
                        AccessKind::GlobalEligible { .. } => L2ClientScope::OrdinaryLdg,
                        AccessKind::LocalLoad { .. } => L2ClientScope::Local,
                        AccessKind::GlobalAsyncCopy {
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                            ..
                        } => L2ClientScope::AsyncLdgstsCache,
                        AccessKind::GlobalAsyncCopy { .. } => L2ClientScope::AsyncLdgstsBypass,
                        AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. } => {
                            L2ClientScope::Store
                        }
                        AccessKind::GlobalReduction { .. } => L2ClientScope::Reduction,
                        AccessKind::GlobalBypass {
                            token: CacheOpToken::Atom,
                            ..
                        } => L2ClientScope::Atomic,
                        _ => L2ClientScope::Unknown,
                    },
                    self.schedule
                        .issue_slot(u64::from(inst.pc))
                        .is_some_and(|slot| {
                            matches!(inst.kind, AccessKind::GlobalUntagged { .. })
                                && opcode_head(slot.opcode_text()) == "ST"
                        }),
                    self.schedule
                        .issue_slot(u64::from(inst.pc))
                        .is_some_and(|slot| {
                            matches!(inst.kind, AccessKind::GlobalUntagged { .. })
                                && opcode_head(slot.opcode_text()) == "ST"
                                && self
                                    .block_states
                                    .get(self.warps[wix].block)
                                    .is_some_and(|state| state.admitted_warps <= 8)
                        }),
                    warp_sectors,
                    write_barrier,
                    async_destination_wavefronts[sc],
                    memory_time,
                    out,
                )?
            } else {
                data_ready_time(
                    inst.kind,
                    warp_sectors,
                    memory_time,
                    &self.params,
                    self.mem.const_l1_mut(),
                    write_barrier,
                    observe_async_copy || observe_register_result,
                )?
            };
            if self.params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(self.mem, MemModel::Flat)
                && self.sm120_flat_async_dense
                && let AccessKind::GlobalAsyncCopy {
                    off,
                    len: len @ 1..,
                    l1_policy,
                    ..
                } = inst.kind
            {
                // Retain a bounded recent-source working set for non-bulk
                // cp.async. Repeated tile copies cannot all be cold DRAM
                // accesses. Existing fills carry their own ready timestamp.
                // 128 sectors is a logical 4-KiB window, not inferred L1 size.
                let mut source_ready = t;
                for &sector in &warp_sectors[off..off + usize::from(len)] {
                    let ready = if l1_policy == LdgstsL1Policy::CacheAtAllLevels {
                        if let Some(position) = self
                            .sm120_async_sources
                            .iter()
                            .position(|&(key, _)| key == sector)
                        {
                            let entry = self
                                .sm120_async_sources
                                .remove(position)
                                .expect("located source");
                            self.sm120_async_sources.push_back(entry);
                            entry.1.max(checked_cycle_add(
                                "warm async source",
                                t,
                                self.params.l1_hit_latency,
                            )?)
                        } else {
                            let latency = if self.sm120_flat_async_dense {
                                // Fresh group-depth probes show dense repeated
                                // cp.async source traffic is not a cold-DRAM
                                // ladder: NCU reports only 104/128/168 DRAM
                                // sectors for 64/128/256 groups. Use the
                                // logical L2 arm for this shape-conditioned
                                // source frontier; it is not an L2 geometry
                                // claim.
                                self.params.l1_miss_latency
                            } else {
                                self.params.global_flat_latency
                            };
                            let ready = checked_cycle_add("cold async source", t, latency)?;
                            self.sm120_async_sources.push_back((sector, ready));
                            if self.sm120_async_sources.len() > 128 {
                                self.sm120_async_sources.pop_front();
                            }
                            ready
                        }
                    } else {
                        checked_cycle_add(
                            "bypass async source",
                            t,
                            self.params.global_flat_latency,
                        )?
                    };
                    source_ready = source_ready.max(ready);
                }
                data_ready = Some(source_ready);
            }
            if let Some(service_done) = sm120_flat_service_completion
                && (write_barrier.is_some()
                    || control.read_barrier.is_some()
                    || observe_async_copy
                    || observe_register_result)
            {
                data_ready = Some(data_ready.map_or(service_done, |ready| ready.max(service_done)));
            }
            if let Some(service_done) = sm120_flat_store_retire_completion {
                data_ready = Some(data_ready.map_or(service_done, |ready| ready.max(service_done)));
            }
            // Register results and scoreboard counters must share the same
            // producer completion. A load is not ready at dispatch+1 simply
            // because its consumer lacks a compiler barrier annotation.
            if observe_register_result {
                let completion = match pipeline_class {
                    PipelineClass::ScalarAlu { latency_cycles, .. } => {
                        let base =
                            checked_cycle_add("scalar register completion", t, latency_cycles)?;
                        scalar_execution_pipeline_plan
                            .as_ref()
                            .and_then(|plan| plan.next.release_times[sc].back().copied())
                            .map_or(base, |ready| base.max(ready))
                    }
                    _ => data_ready.unwrap_or(self.params.non_memory_ready(t)?),
                };
                if let Some(status) = fixed_latency_status {
                    self.warps[wix]
                        .register_ready
                        .retain(|entry| entry.ready_at > t);
                    issue_record_fixed_register_writes(
                        &mut self.warps[wix].register_ready,
                        status,
                        inst.effective_lane_mask(),
                        completion,
                    );
                    self.local_completion_time =
                        Some(self.local_completion_time.unwrap_or(0).max(completion));
                }
            }
            if cctl_invalidate_all && matches!(self.mem, MemModel::L1Shared(_)) {
                self.invalidate_l1_all()?;
                if let MemModel::L1Shared(st) = &mut self.mem
                    && let Some(requester) = st.l1_visibility_requester
                {
                    let sequence = st.next_l1_visibility_sequence()?;
                    st.l1_invalidate_all_observations
                        .push(sim_mem::L1InvalidateAllObservation {
                            sequence,
                            requester,
                            issue_cycle: t,
                        });
                }
            }
            if let Some((scope, order)) = fence_semantics
                && let MemModel::L1Shared(st) = &mut self.mem
                && let Some(requester) = st.l1_visibility_requester
            {
                let sequence = st.next_l1_visibility_sequence()?;
                st.l1_fence_observations.push(sim_mem::L1FenceObservation {
                    sequence,
                    requester,
                    pc: inst.pc,
                    issue_cycle: t,
                    scope,
                    order,
                });
            }
            if observe_async_copy && let Some(source_ready) = data_ready {
                // Shared mode already attached the copy's delivery and exact
                // destination work to its pending group before walking source
                // sectors. This second update contributes only the known hit
                // floor; synchronous modes create the group here instead.
                let (delivery_work, destination_wavefronts) =
                    if matches!(self.mem, MemModel::L1Shared(_)) {
                        (0, 0)
                    } else {
                        (
                            if self.sm120_flat_async_dense {
                                0
                            } else {
                                async_copy_completion_work(inst.kind).unwrap_or(0)
                            },
                            async_destination_wavefronts[sc],
                        )
                    };
                self.async_copies[wix].add_known_async_copy(
                    wix,
                    source_ready,
                    delivery_work,
                    destination_wavefronts,
                    &self.params,
                )?;
            }

            match pipeline_class {
                PipelineClass::Branch { .. } => {}
                PipelineClass::Barrier { .. } => {
                    if let Some(arrival) = barrier_arrival_times[sc] {
                        self.barrier_pipe_ready = arrival;
                    }
                }
                PipelineClass::Tensor {
                    initiation_interval_cycles,
                } => {
                    self.tensor_pipe_ready[sc] = checked_cycle_add(
                        "tensor-pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                }
                PipelineClass::Sfu {
                    initiation_interval_cycles,
                } => {
                    self.sfu_pipe_ready[sc] = checked_cycle_add(
                        "uniform/SFU pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                    if self.params.arch == ArchId::Sm86
                        && self.warps[wix]
                            .next_slot
                            .is_some_and(IssueSlot::is_sfu_dependent_chain)
                    {
                        self.warps[wix].sfu_dependency_ready = checked_cycle_add(
                            "SM86 dependent SFU result readiness",
                            t,
                            SM86_DEPENDENT_SFU_LATENCY_CYCLES,
                        )?;
                    }
                }
                PipelineClass::ScalarAlu { .. } => {}
                PipelineClass::DispatchOnly => {}
            }
            // Yield semantics per the CGGTY paper: the yield bit means
            // "do not issue this warp in the NEXT cycle" — a one-cycle
            // bubble when no other warp is ready — on top of the stall
            // count. A retiring instruction has no next eligibility to observe.
            if !retires_warp {
                // The retained SM120 high-word control decoder is a
                // transport hypothesis. Dependent-chain calibration
                // shows that applying its scalar stall field on top of the
                // measured family latency double-counts readiness. Keep the
                // field for non-scalar classes, but let the calibrated
                // ScalarAlu pipeline own scalar readiness on SM120.
                let control_stall = if scalar_control_stall_suppressed {
                    0
                } else {
                    control.stall
                };
                let stall = u64::from(control_stall.max(1));
                let mut eligible = checked_cycle_add("warp stall eligibility", t, stall)?;
                if control.yield_flag {
                    eligible = eligible.max(checked_cycle_add("warp yield eligibility", t, 2)?);
                }
                if let Some(service_done) = sm120_flat_store_retire_completion {
                    eligible = eligible.max(service_done);
                }
                if (self.sm120_flat_multiwarp_medium || self.sm120_flat_shared_medium)
                    && let Some(service_done) = sm120_flat_service_completion
                {
                    // Dense multi-warp DRAM calibration exposes admission
                    // backpressure, not merely a terminal drain. Park the
                    // issuing warp at the logical service completion so the
                    // existing scheduler can overlap other warps naturally.
                    eligible = eligible.max(service_done);
                }
                self.warps[wix].eligible = eligible;
            }
            // Dependence barriers are COUNTERS in hardware (producer
            // increments at issue, decrements at write-back; the
            // consumer waits for zero), so with several in-flight
            // producers on one index the barrier clears at the LATEST
            // write-back: accumulate with max, never overwrite — a
            // later fast producer (L1 hit) must not shorten a pending
            // slow one (miss). Unknown write-backs (in-flight responses)
            // are tracked in `barrier_pending`.
            if let Some(dr) = data_ready {
                self.warps[wix].stamp_clear(write_barrier, dr, dependency_origin(inst.kind));
            }
            if let Some(result_writeback) = result_writeback {
                self.register_l1tex_writeback(
                    sc,
                    wix,
                    result_writeback,
                    write_barrier,
                    data_ready,
                )?;
                self.apply_fixed_result_priority(sc, result_writeback, &mut projected_readiness)?;
            }
            if produces_dynamic_effects && let Some(read_barrier) = control.read_barrier {
                let read_clear = memory_pipeline_issue.map_or_else(
                    || {
                        checked_cycle_add(
                            "read-dependence barrier visibility",
                            t,
                            self.params.read_barrier_latency,
                        )
                    },
                    |issue| Ok(issue.operand_ready_at),
                )?;
                let read_clear = sm120_flat_service_completion
                    .map_or(read_clear, |service_done| read_clear.max(service_done));
                self.warps[wix].stamp_clear(
                    Some(read_barrier),
                    read_clear,
                    operand_read_dependency_origin(inst.kind),
                );
            }
            if clock_read {
                let warp = &mut self.warps[wix];
                warp.clock_first_issue.get_or_insert(t);
                warp.clock_reads_issued = warp
                    .clock_reads_issued
                    .checked_add(1)
                    .expect("CLOCKLO issue counter was preflighted");
            }
            if self.has_async_copy_protocol {
                match slot_class {
                    OpClass::AsyncCopyCommit => {
                        let expected_nonempty = self.async_copies[wix]
                            .state
                            .as_deref()
                            .and_then(|state| state.open_group)
                            .is_some();
                        let committed = self.async_copies[wix].commit_async_copy_group(t);
                        debug_assert_eq!(
                            committed, expected_nonempty,
                            "async commit mutation disagrees with its pre-issue state"
                        );
                        if committed && self.sm120_flat_async_service_enabled {
                            let done = self
                                .sm120_flat_async_service_ready
                                .max(t)
                                .saturating_add(SM120_FLAT_ASYNC_GROUP_SERVICE_CYCLES);
                            self.sm120_flat_async_service_ready = done;
                            if let Some(state) = self.async_copies[wix].state.as_deref_mut()
                                && let Some(group) = state.committed_groups.back_mut()
                            {
                                group.destination_completion_time =
                                    group.destination_completion_time.max(done);
                                group.completion_time = group.completion_time.max(done);
                            }
                        }
                        if committed && self.sm120_flat_async_depth_service_per_group != 0 {
                            let done = self
                                .sm120_flat_async_depth_service_ready
                                .max(t)
                                .saturating_add(self.sm120_flat_async_depth_service_per_group);
                            self.sm120_flat_async_depth_service_ready = done;
                            if let Some(state) = self.async_copies[wix].state.as_deref_mut()
                                && let Some(group) = state.committed_groups.back_mut()
                            {
                                group.destination_completion_time =
                                    group.destination_completion_time.max(done);
                                group.completion_time = group.completion_time.max(done);
                            }
                        }
                    }
                    OpClass::AsyncCopyWait => {
                        let max_pending_groups = self.warps[wix]
                            .next_slot
                            .ok_or(IssueError::MissingIssueSlot { warp: wix })?
                            .async_copy_wait_max_pending_groups;
                        self.async_copies[wix].consume_async_wait(max_pending_groups, t);
                    }
                    // `Other` includes explicit SB1..SB5 DEPBAR operand sets.
                    // Readiness was enforced through ordinary barrier state;
                    // they never mutate asynchronous-copy groups.
                    OpClass::BlockBarrier | OpClass::Other | OpClass::MBarrierPhaseQuery => {}
                }
            }

            // Advance the warp's stream.
            if self.instruction_front_end.is_some() {
                self.warps[wix].consume_buffered_issue(wix)?;
            }
            self.warps[wix].next =
                checked_usize_counter_add("warp instruction cursor", self.warps[wix].next, 1)?;
            #[cfg(feature = "reference")]
            self.reference_issue_effects.push((
                self.t,
                u64::try_from(wix).unwrap_or(u64::MAX),
                u32::try_from(instruction).unwrap_or(u32::MAX),
            ));
            let final_block_barrier = slot_class == OpClass::BlockBarrier
                && self.warps[wix].next == self.warps[wix].insts.len();
            if retires_warp {
                self.retire_warp(sc, wix, t)?;
            } else if final_block_barrier {
                // Keep a final barrier warp unfinished and parked until the
                // CTA rendezvous. There is intentionally no next issue slot;
                // the release path retires it after all participants arrive.
                self.warps[wix].next_slot = None;
                self.warps[wix].ordinary_dependency_wait_mask = WaitMask::NONE;
                self.warps[wix].eligible = t;
            } else {
                let next_pc = self.warps[wix].insts[self.warps[wix].next].pc;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    next_pc,
                    launch_index,
                    warp_slot,
                )?;
                self.warps[wix].next_slot = Some(next_slot);
                self.warps[wix].ordinary_dependency_wait_mask =
                    next_slot.ordinary_dependency_wait_mask();
                self.warps[wix].current_since = t;
                self.warps[wix].imc_wait_until = 0;
                self.warps[wix].cta_barrier_ready = 0;
                self.warps[wix].barrier_pipeline_ready = 0;
                self.warps[wix].branch_pipeline_ready = 0;
                if matches!(pipeline_class, PipelineClass::Branch { .. })
                    && next_pc != inst.pc.saturating_add(16)
                    && let Some(front_end) = self.instruction_front_end.as_mut()
                {
                    front_end.diagnostics.control_flow_redirects = checked_u64_counter_add(
                        "instruction control-flow redirect observer",
                        front_end.diagnostics.control_flow_redirects,
                        1,
                    )?;
                }
            }
            if !retires_warp
                && let PipelineClass::Branch {
                    next_issue_interval_cycles,
                } = pipeline_class
            {
                self.warps[wix].branch_pipeline_ready =
                    checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            // Single hold write: never hold a yielded or finished warp.
            self.hold[sc] = (!control.yield_flag && !self.warps[wix].done).then_some(wix);
            self.fetch_last_issued[sc] = (!self.warps[wix].done).then_some(wix);

            // Block barrier arrival AFTER stream bookkeeping: a finished
            // warp never waits.
            if slot_class == OpClass::BlockBarrier && !self.warps[wix].done {
                let arrival = barrier_arrival_times[sc].unwrap_or(t);
                self.warps[wix].at_block_barrier = true;
                self.warps[wix].block_barrier_dispatch = Some(t);
                self.warps[wix].block_barrier_arrival = Some(arrival);
                self.block_states[blk_ix]
                    .barrier_first_dispatch
                    .get_or_insert(t);
                self.block_states[blk_ix]
                    .barrier_first_arrival
                    .get_or_insert(arrival);
                self.block_states[blk_ix].waiting = checked_usize_counter_add(
                    "block barrier waiter",
                    self.block_states[blk_ix].waiting,
                    1,
                )?;
            }
            // Release check: every unfinished warp of the block waiting.
            // The block's warps are contiguous in `warps` (admission
            // pushes them in one run), so scan only that range.
            let bs = &self.block_states[blk_ix];
            if bs.waiting > 0 && bs.waiting == bs.unfinished {
                let (first, count) = (bs.first_warp, bs.admitted_warps);
                let mut latest_arrival = 0u64;
                let mut second_latest_arrival = 0u64;
                let mut latest_count = 0usize;
                for warp in &self.warps[first..first + count] {
                    if !warp.at_block_barrier {
                        continue;
                    }
                    let arrival = warp
                        .block_barrier_arrival
                        .expect("barrier waiter owns an arrival timestamp");
                    if arrival > latest_arrival {
                        second_latest_arrival = latest_arrival;
                        latest_arrival = arrival;
                        latest_count = 1;
                    } else if arrival == latest_arrival {
                        latest_count += 1;
                    } else if arrival > second_latest_arrival {
                        second_latest_arrival = arrival;
                    }
                }
                self.block_states[blk_ix].waiting = 0;
                self.block_states[blk_ix].barrier_generation = self.block_states[blk_ix]
                    .barrier_generation
                    .checked_add(1)
                    .expect("barrier generation was preflighted");
                self.block_states[blk_ix].barrier_first_dispatch = None;
                self.block_states[blk_ix].barrier_first_arrival = None;
                if let Some(issue) = self.barrier_issue {
                    self.block_states[blk_ix].barrier_rearm_ready = checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        latest_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
                let release_at = checked_cycle_add("block-barrier release", latest_arrival, 1)?;
                self.local_completion_time = Some(
                    self.local_completion_time
                        .map_or(release_at, |current| current.max(release_at)),
                );
                // Final BAR.SYNC instructions advanced their warps to the end
                // of stream but deliberately left them unfinished above.
                // Collect those participants before clearing the barrier
                // flags, then retire them through the common residency path.
                let final_barrier_warps = self.warps[first..first + count]
                    .iter()
                    .enumerate()
                    .filter_map(|(offset, warp)| {
                        (warp.at_block_barrier && warp.next == warp.insts.len())
                            .then_some(first + offset)
                    })
                    .collect::<Vec<_>>();
                for w in &mut self.warps[first..first + count] {
                    if w.at_block_barrier {
                        let own_arrival = w
                            .block_barrier_arrival
                            .expect("barrier waiter owns an arrival timestamp");
                        let other_arrival = if count <= 1 {
                            None
                        } else if own_arrival == latest_arrival && latest_count == 1 {
                            Some(second_latest_arrival)
                        } else {
                            Some(latest_arrival)
                        };
                        w.at_block_barrier = false;
                        w.block_barrier_dispatch = None;
                        w.block_barrier_arrival = None;
                        if self.barrier_issue.is_some() {
                            w.barrier_pipeline_ready = checked_cycle_add(
                                "CTA-barrier own-arrival release",
                                own_arrival,
                                1,
                            )?;
                            w.cta_barrier_ready = other_arrival
                                .map(|arrival| {
                                    checked_cycle_add(
                                        "CTA-barrier other-participant rendezvous",
                                        arrival,
                                        1,
                                    )
                                })
                                .transpose()?
                                .unwrap_or(0);
                            debug_assert_eq!(
                                w.barrier_pipeline_ready.max(w.cta_barrier_ready),
                                release_at
                            );
                        } else {
                            w.barrier_pipeline_ready = 0;
                            w.cta_barrier_ready = release_at;
                        }
                    }
                }
                for warp_index in final_barrier_warps {
                    let subcore = self.warps[warp_index]
                        .warp_slot
                        .rem_euclid(SUB_CORES as i32) as usize;
                    // Admission of the next CTA is causally after the barrier
                    // release, not after the last warp's dispatch cycle.
                    self.retire_warp(subcore, warp_index, release_at)?;
                }
            }
        }
        if let Some(plan) = memory_pipeline_plan {
            self.memory_issue_pipeline = Some(plan.next);
        }
        if let Some(plan) = scalar_execution_pipeline_plan {
            self.scalar_execution_pipeline = Some(Box::new(plan.next));
        }
        for (warp, ready) in imc_waits
            .iter()
            .flatten()
            .copied()
            .chain(extra_imc_waits.iter().copied())
        {
            self.warps[warp].imc_wait_until = ready;
        }
        if let Some(cache) = imc_cache.as_mut() {
            for (warp, _) in imc_waits
                .iter()
                .flatten()
                .copied()
                .chain(extra_imc_waits.iter().copied())
            {
                let subcore = self.warps[warp].warp_slot.rem_euclid(SUB_CORES as i32) as usize;
                cache[subcore].diagnostics.miss_wait_events = cache[subcore]
                    .diagnostics
                    .miss_wait_events
                    .saturating_add(1);
                cache[subcore].diagnostics.miss_wait_cycles = cache[subcore]
                    .diagnostics
                    .miss_wait_cycles
                    .saturating_add(1);
            }
        }
        if imc_cache.is_some() {
            self.immediate_constant_switch_remaining = projected_immediate_constant_switch;
            self.immediate_constant_cache = imc_cache;
        }
        // Counts are bounded by next_issued, whose checked increment was
        // preflighted above. Multiple issue ports count only one active cycle.
        for (subcore, pick) in effective_picks.iter().enumerate() {
            if pick.is_some() {
                self.instructions_per_smsp[subcore] = self.instructions_per_smsp[subcore]
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "instructions per SMSP",
                    })?;
                if self.last_observed_issue[subcore] != Some(t) {
                    self.issue_active_cycles_per_smsp[subcore] = self.issue_active_cycles_per_smsp
                        [subcore]
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "issue-active cycles per SMSP",
                        })?;
                    self.last_observed_issue[subcore] = Some(t);
                }
            }
        }
        self.issued = next_issued;
        *self.readiness = projected_readiness;
        self.memory_service_candidates = projected_memory_service_candidates;
        if let Some(plan) = memory_admission_feedback_plan {
            self.memory_admission_feedback
                .as_mut()
                .expect("feedback issue plan requires installed feedback state")
                .apply_issue(plan, t);
        }
        if issued_this_cycle {
            self.last_issue_picks = effective_picks;
        }
        Ok(issued_this_cycle)
    }
}

/// Register a completion target that clears when `fill` completes. The
/// asynchronous-copy group already owns its pending source unit; an optional
/// ordinary barrier is counted here alongside ordinary load barriers.
fn add_fill_waiter(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    fill: FillKey,
    warp: usize,
    target: Option<FillTarget>,
    constant_deep: bool,
) -> Result<(), IssueError> {
    let Some(target) = target else {
        return Ok(());
    };
    warps[warp].add_pending(warp, target.barrier(), 1)?;
    match st.fill_waiters.entry(fill) {
        Entry::Vacant(v) => {
            v.insert(Waiters::One(FillWaiter {
                warp,
                target,
                units: 1,
                constant_deep,
            }));
        }
        Entry::Occupied(mut o) => o.get_mut().add(warp, target, constant_deep)?,
    }
    Ok(())
}

fn preflight_line_fill_mask(st: &SharedL2State, sector_id: i64) -> Result<(), IssueError> {
    let line = sector_id >> 2;
    let fill_in_flight = st.l1.is_line_fill_in_flight(line);
    let has_mask = st.fill_masks.contains_key(&line);
    match (fill_in_flight, has_mask) {
        (true, false) => Err(IssueError::FillMaskMissing { line }),
        (false, true) => Err(IssueError::FillMaskWithoutFill { line }),
        _ => Ok(()),
    }
}

fn register_response(st: &mut SharedL2State, reply: ReplyTo) -> Result<(), IssueError> {
    let response_id = reply
        .response_id()
        .ok_or(IssueError::UnexpectedMutationOnlyResponse)?;
    if let Some(&existing) = st.outstanding_response_ids.get(&response_id) {
        return Err(IssueError::ResponseTokenAlreadyOutstanding { reply: existing });
    }
    if !st.outstanding_replies.insert(reply) {
        return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
    }
    st.outstanding_response_ids.insert(response_id, reply);
    Ok(())
}

enum SharedL1Access {
    Hit { ready: Option<u64> },
    MissRequest { fill: FillKey },
    MergedPending { fill: FillKey },
    Stalled,
}

fn record_global_l1_observation(
    st: &mut SharedL2State,
    sector_id: i64,
    is_write: bool,
    l1_resident_after: bool,
    hit: bool,
    fill_generation: Option<u32>,
    issue_cycle: u64,
) -> Result<(), IssueError> {
    let Some(requester) = st.l1_visibility_requester else {
        return Ok(());
    };
    let sequence = st.next_l1_visibility_sequence()?;
    st.l1_visibility_observations
        .push(sim_mem::L1AccessObservation {
            sequence,
            line_id: sector_id >> 2,
            sector_mask: 1_u8 << (sector_id & 3) as u8,
            requester,
            is_write,
            l1_resident_after,
            fill_will_reside: fill_generation.is_some(),
            fill_generation,
            hit,
            issue_cycle,
        });
    Ok(())
}

fn fill_generation(fill: FillKey) -> u32 {
    match fill {
        FillKey::Line { generation, .. } | FillKey::Sector { generation, .. } => generation,
    }
}

fn access_shared_l1(
    l1: &mut L1Cache,
    origin: MemoryOrigin,
    sector_id: i64,
    now: u64,
    observe_ready: bool,
    mshr_floor: Option<NonZeroU32>,
) -> Result<SharedL1Access, L1AccessError> {
    if observe_ready {
        let access = match mshr_floor {
            Some(floor) => l1.access_eligible_ev_with_mshr_floor(origin, sector_id, now, floor)?,
            None => l1.access_eligible_ev(origin, sector_id, now)?,
        };
        return Ok(match access {
            EvAccess::Hit { ready } => SharedL1Access::Hit { ready: Some(ready) },
            EvAccess::MissRequest { fill } => SharedL1Access::MissRequest { fill },
            EvAccess::MergedPending { fill } => SharedL1Access::MergedPending { fill },
            EvAccess::Stalled => SharedL1Access::Stalled,
        });
    }
    let access = match mshr_floor {
        Some(floor) => {
            l1.access_eligible_ev_unobserved_with_mshr_floor(origin, sector_id, now, floor)?
        }
        None => l1.access_eligible_ev_unobserved(origin, sector_id, now)?,
    };
    Ok(match access {
        EvAccessUnobserved::Hit => SharedL1Access::Hit { ready: None },
        EvAccessUnobserved::MissRequest { fill } => SharedL1Access::MissRequest { fill },
        EvAccessUnobserved::MergedPending { fill } => SharedL1Access::MergedPending { fill },
        EvAccessUnobserved::Stalled => SharedL1Access::Stalled,
    })
}

#[derive(Debug)]
enum SharedAccessError {
    Stalled,
    Issue(IssueError),
}

impl From<IssueError> for SharedAccessError {
    fn from(source: IssueError) -> Self {
        Self::Issue(source)
    }
}

/// One eligible sector access in L1Shared mode at time `now`, emitting a
/// request on a fresh miss. Returns `Ok(known_ready)` for hits,
/// `Ok(None)` when the data waits on a response, and
/// `Err(SharedAccessError::Stalled)` on an MSHR stall (caller parks the
/// remainder). Accounting failures remain typed.
///
/// A sector merging onto a line fill emitted during the CURRENT
/// emission batch (`st.line_folds`, keyed by line and issue timestamp)
/// folds its bit into that request's `sector_mask`, so one L2
/// transaction carries same-cycle sectors and worst-of-mask pricing
/// covers every warp/instruction/retry-entry at that timestamp. A merge onto
/// an OLDER in-flight fill emits a follow-up sector fetch with an independent
/// `FillKey::Sector` response; the L2 must see every fetched sector exactly
/// once per fill and the waiter clears only at that sector's ready event.
fn access_shared(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    wix: usize,
    target: Option<FillTarget>,
    mshr_floor: Option<NonZeroU32>,
    origin: MemoryOrigin,
    constant_load: bool,
    constant_deep: bool,
    client_scope: L2ClientScope,
    sector_id: i64,
    response_id: ResponseId,
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<Option<u64>, SharedAccessError> {
    let sbit = 1u8 << (sector_id & 3) as u8;
    match access_shared_l1(
        &mut st.l1,
        origin,
        sector_id,
        now,
        target.is_some(),
        mshr_floor,
    )
    .map_err(IssueError::from)?
    {
        SharedL1Access::Hit { ready } => {
            if origin == MemoryOrigin::Global {
                record_global_l1_observation(st, sector_id, false, true, true, None, now)?;
            }
            Ok(ready)
        }
        SharedL1Access::MissRequest { fill } => {
            if origin == MemoryOrigin::Global {
                // The directory must not publish ownership before the L1 fill
                // response is visible. A later fill-visible event is required
                // to make this sector resident in the visibility protocol.
                record_global_l1_observation(
                    st,
                    sector_id,
                    false,
                    false,
                    false,
                    Some(fill_generation(fill)),
                    now,
                )?;
            }
            let request = MemRequest::l1_fill(
                now,
                sector_id,
                origin,
                fill,
                response_id,
                constant_load,
                constant_deep,
                client_scope,
            );
            register_response(st, request.reply)?;
            st.response_masks.insert(response_id, sbit);
            out.push(request);
            if let FillKey::Line { line, .. } = fill {
                st.fill_masks.insert(line, sbit);
                st.line_folds.entry((line, now)).or_insert(out.len() - 1);
            }
            add_fill_waiter(st, warps, fill, wix, target, constant_deep)?;
            Ok(None)
        }
        SharedL1Access::MergedPending { fill } => {
            if origin == MemoryOrigin::Global {
                record_global_l1_observation(
                    st,
                    sector_id,
                    false,
                    false,
                    false,
                    Some(fill_generation(fill)),
                    now,
                )?;
            }
            if let FillKey::Line { line, .. } = fill {
                // Pairing invariant (proven at the 2026-07-07 gate): a
                // line-pending merge always has a live fill_masks entry
                // (created at MissRequest, removed at the response that
                // also completes the fill). Fail LOUD if a future
                // change breaks it — a fabricated empty mask would
                // silently re-fetch already-sent sectors.
                let already_sent = st.fill_masks.get(&line).ok_or(SharedAccessError::Issue(
                    IssueError::FillMaskMissing { line },
                ))? & sbit
                    != 0;
                if !already_sent {
                    if let Some(&ix) = st.line_folds.get(&(line, now)) {
                        *st.fill_masks
                            .get_mut(&line)
                            .expect("fill mask checked above") |= sbit;
                        out[ix].fold_sector_origin(origin, sbit, constant_load, constant_deep);
                        if let Some(response_id) = out[ix].reply.response_id() {
                            *st.response_masks.entry(response_id).or_insert(0) |= sbit;
                        }
                    } else {
                        let sector_fill = st
                            .l1
                            .reserve_pending_line_sector(fill, (sector_id & 3) as u8)
                            .map_err(IssueError::from)?;
                        *st.fill_masks
                            .get_mut(&line)
                            .expect("fill mask checked above") |= sbit;
                        let request = MemRequest::l1_followup_fetch(
                            now,
                            sector_id,
                            origin,
                            sector_fill,
                            response_id,
                            constant_load,
                            constant_deep,
                            client_scope,
                        );
                        register_response(st, request.reply)?;
                        st.response_masks.insert(response_id, sbit);
                        out.push(request);
                        add_fill_waiter(st, warps, sector_fill, wix, target, constant_deep)?;
                        return Ok(None);
                    }
                }
                // Re-touch of an already-sent sector: MSHR merge, no
                // new fetch (matches NCU's xbar sector accounting).
            }
            // Sector-fill merges need nothing: that sector's fetch was
            // sent with its own request.
            add_fill_waiter(st, warps, fill, wix, target, constant_deep)?;
            Ok(None)
        }
        SharedL1Access::Stalled => Err(SharedAccessError::Stalled),
    }
}

fn drain_retries(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    async_copies: &mut [AsyncCopySlot],
    now: u64,
    params: &IssueModelParams,
    out: &mut Vec<MemRequest>,
) -> Result<(), IssueError> {
    while let Some(mut r) = st.retry.pop_front() {
        st.l1.resume_lookup_batch(r.lookup_batch);
        let sectors = warps[r.warp].sectors;
        let barrier = r.target.and_then(FillTarget::barrier);
        // Replace the retry entry's ordinary-barrier unit atomically with
        // whatever waiter units this attempt creates. Async-copy source units
        // remain owned by their group until a hit or fill completion resolves
        // each sector.
        warps[r.warp].release_pending(r.warp, barrier, 1)?;
        // The drain is one emission batch, but folds remain timestamp scoped:
        // a line fill emitted by an earlier retry entry only folds later
        // same-timestamp sectors; later-cycle sectors receive child tokens.
        while r.lo < r.hi {
            match access_shared(
                st,
                warps,
                r.warp,
                r.target,
                r.mshr_floor,
                r.origin,
                r.constant_load,
                r.constant_deep,
                r.client_scope,
                sectors[r.lo],
                ResponseId::new(st.response_owner, r.warp, r.instruction, r.lo),
                now,
                out,
            ) {
                Ok(Some(ready)) => {
                    let ready = checked_constant_return_time(
                        ready,
                        r.constant_deep && barrier.is_some(),
                        params,
                        "constant retry L1-hit return latency",
                    )?;
                    if let Some(group) = r.target.and_then(FillTarget::async_group) {
                        async_copies[r.warp]
                            .complete_async_copy_sources(r.warp, group, None, 1, ready, params)?;
                    }
                    warps[r.warp].stamp_clear(barrier, ready, DependencyOrigin::L1Tex);
                }
                Ok(None) => {}
                Err(SharedAccessError::Stalled) => {
                    // Still full: restore the retry entry's replacement unit
                    // and park it; nothing behind it can proceed either.
                    warps[r.warp].add_pending(r.warp, barrier, 1)?;
                    st.retry.push_front(r);
                    return Ok(());
                }
                Err(SharedAccessError::Issue(source)) => return Err(source),
            }
            r.lo = r.lo.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "retry sector cursor",
            })?;
        }
    }
    Ok(())
}

