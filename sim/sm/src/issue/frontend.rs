//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct InstructionLineKey {
    pub(super) function_unique_id: i32,
    pub(super) line: u64,
}

impl InstructionLineKey {
    pub(super) fn following(self) -> Option<Self> {
        Some(Self {
            function_unique_id: self.function_unique_id,
            line: self.line.checked_add(1)?,
        })
    }
}

/// Small deterministic set-associative tag store for the instruction path.
/// A pending line remains a tag hit with its future fill timestamp, matching
/// the merge behavior needed when many warps execute one code region.
#[derive(Debug, Clone)]
pub(super) struct SetAssociativeInstructionCache {
    pub(super) sets: Vec<Vec<InstructionCacheEntry>>,
    pub(super) ways: usize,
    pub(super) access_sequence: u64,
}

impl SetAssociativeInstructionCache {
    pub(super) fn new(
        component: &'static str,
        set_count: u32,
        ways: u32,
    ) -> Result<Self, IssueError> {
        if set_count == 0 {
            return Err(IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "sets",
                value: u64::from(set_count),
            });
        }
        if ways == 0 {
            return Err(IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "ways",
                value: u64::from(ways),
            });
        }
        let set_count = usize::try_from(set_count).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "sets",
                value: u64::from(set_count),
            }
        })?;
        let ways =
            usize::try_from(ways).map_err(|_| IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "ways",
                value: u64::from(ways),
            })?;
        Ok(Self {
            sets: (0..set_count).map(|_| Vec::with_capacity(ways)).collect(),
            ways,
            access_sequence: 0,
        })
    }

    pub(super) fn set_index(&self, key: InstructionLineKey) -> usize {
        let function = u64::from(key.function_unique_id.unsigned_abs());
        usize::try_from((key.line ^ function.rotate_left(17)) % self.sets.len() as u64)
            .expect("modulo by usize-sized set count fits usize")
    }

    pub(super) fn next_access_sequence(&mut self) -> Result<u64, IssueError> {
        let sequence = self.access_sequence;
        self.access_sequence =
            self.access_sequence
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "instruction-cache access sequence",
                })?;
        Ok(sequence)
    }

    pub(super) fn probe(&mut self, key: InstructionLineKey) -> Result<Option<u64>, IssueError> {
        let set = self.set_index(key);
        let Some(index) = self.sets[set].iter().position(|entry| entry.key == key) else {
            return Ok(None);
        };
        let sequence = self.next_access_sequence()?;
        let entry = &mut self.sets[set][index];
        entry.last_access = sequence;
        Ok(Some(entry.ready_at))
    }

    pub(super) fn reserve(
        &mut self,
        key: InstructionLineKey,
        ready_at: u64,
    ) -> Result<(), IssueError> {
        let sequence = self.next_access_sequence()?;
        let set = self.set_index(key);
        if let Some(entry) = self.sets[set].iter_mut().find(|entry| entry.key == key) {
            entry.ready_at = entry.ready_at.min(ready_at);
            entry.last_access = sequence;
            return Ok(());
        }
        let entries = &mut self.sets[set];
        if entries.len() == self.ways {
            let victim = entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_access, *index))
                .map(|(index, _)| index)
                .ok_or(IssueError::InstructionFrontEndGeometryInvalid {
                    component: "instruction_cache",
                    field: "full-set victim cardinality",
                    value: 0,
                })?;
            entries.swap_remove(victim);
        }
        entries.push(InstructionCacheEntry {
            key,
            ready_at,
            last_access: sequence,
        });
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct InstructionStreamBuffer {
    pub(super) lines: VecDeque<StreamBufferLine>,
    pub(super) next_line: Option<InstructionLineKey>,
}

impl InstructionStreamBuffer {
    pub(super) fn maintain(&mut self, now: u64) {
        if self
            .lines
            .front()
            .is_some_and(|line| line.demanded && line.ready_at <= now)
        {
            self.lines.pop_front();
        }
    }

    pub(super) fn demand(&mut self, key: InstructionLineKey) -> Option<u64> {
        let line = self.lines.front_mut()?;
        if line.key != key {
            return None;
        }
        line.demanded = true;
        Some(line.ready_at)
    }

    pub(super) fn can_restart(&self) -> bool {
        self.lines.front().is_none_or(|line| !line.demanded)
    }

    pub(super) fn restart_after(&mut self, key: InstructionLineKey) {
        if !self.can_restart() {
            return;
        }
        self.lines.clear();
        self.next_line = key.following();
    }

    pub(super) fn can_prefetch(&self, capacity: usize) -> bool {
        self.next_line.is_some() && self.lines.len() < capacity
    }

    pub(super) fn next_prefetch(&self) -> Option<InstructionLineKey> {
        self.next_line
    }

    pub(super) fn push_prefetch(
        &mut self,
        key: InstructionLineKey,
        ready_at: u64,
    ) -> Result<(), IssueError> {
        if self.next_line != Some(key) {
            return Err(IssueError::InstructionStreamOrder {
                expected: self
                    .next_line
                    .map(|line| (line.function_unique_id, line.line)),
                got: (key.function_unique_id, key.line),
            });
        }
        self.lines.push_back(StreamBufferLine {
            key,
            ready_at,
            demanded: false,
        });
        self.next_line = key.following();
        Ok(())
    }
}

/// One SM's active instruction hierarchy. Lower-level cold fills use the
/// independently carded dependent-DRAM latency; L0/L1 geometry and transport
/// come from the architecture capability above.
#[derive(Debug, Clone)]
pub(super) struct InstructionFrontEndState {
    pub(super) config: InstructionFrontEnd,
    pub(super) l0: [SetAssociativeInstructionCache; SUB_CORES],
    pub(super) l1: SetAssociativeInstructionCache,
    pub(super) streams: [InstructionStreamBuffer; SUB_CORES],
    pub(super) l1_request_ready: u64,
    pub(super) l1_reply_ready: u64,
    pub(super) l0_response_ready: [u64; SUB_CORES],
    pub(super) prefetch_priority: usize,
    pub(super) cold_fill_latency: u64,
    pub(super) diagnostics: InstructionFrontEndDiagnostics,
}

impl InstructionFrontEndState {
    pub(super) fn new(
        config: InstructionFrontEnd,
        cold_fill_latency: u64,
    ) -> Result<Self, IssueError> {
        for (field, value) in [
            ("line_bytes", u64::from(config.line_bytes)),
            ("l0_to_l1_cycles", config.l0_to_l1_cycles),
            ("l1_to_l0_cycles", config.l1_to_l0_cycles),
            ("l1_request_ports", u64::from(config.l1_request_ports)),
            ("l1_reply_ports", u64::from(config.l1_reply_ports)),
            ("stream_buffer_lines", u64::from(config.stream_buffer_lines)),
            (
                "stream_buffers_per_subcore",
                u64::from(config.stream_buffers_per_subcore),
            ),
            (
                "prefetches_per_cycle",
                u64::from(config.prefetches_per_cycle),
            ),
            (
                "instruction_buffer_entries",
                u64::from(config.instruction_buffer_entries),
            ),
            ("fetch_decode_width", u64::from(config.fetch_decode_width)),
            ("fetch_to_issue_stages", config.fetch_to_issue_stages),
            ("cold_fill_latency", cold_fill_latency),
        ] {
            if value == 0 {
                return Err(IssueError::InstructionFrontEndGeometryInvalid {
                    component: "instruction_front_end",
                    field,
                    value,
                });
            }
        }
        for (field, value) in [
            ("l1_request_ports", config.l1_request_ports),
            ("l1_reply_ports", config.l1_reply_ports),
            (
                "stream_buffers_per_subcore",
                config.stream_buffers_per_subcore,
            ),
            ("prefetches_per_cycle", config.prefetches_per_cycle),
            ("fetch_decode_width", config.fetch_decode_width),
        ] {
            if value != 1 {
                return Err(IssueError::InstructionFrontEndWidthUnsupported { field, value });
            }
        }
        let l0 = [
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
        ];
        Ok(Self {
            config,
            l0,
            l1: SetAssociativeInstructionCache::new("L1I", config.l1_sets, config.l1_ways)?,
            streams: std::array::from_fn(|_| InstructionStreamBuffer::default()),
            l1_request_ready: 0,
            l1_reply_ready: 0,
            l0_response_ready: [0; SUB_CORES],
            prefetch_priority: 0,
            cold_fill_latency,
            diagnostics: InstructionFrontEndDiagnostics::default(),
        })
    }

    #[cfg(feature = "reference")]
    pub(super) fn from_typed_boundary(
        boundary: &InstructionFrontEndBoundaryState,
    ) -> Result<Self, IssueError> {
        boundary
            .validate()
            .map_err(|_| IssueError::ReferenceMacroUnsupported {
                reason: "typed_front_end_exit_invalid",
            })?;
        let cache =
            |state: &InstructionCacheState| -> Result<SetAssociativeInstructionCache, IssueError> {
                let ways = usize::try_from(state.ways).map_err(|_| {
                    IssueError::ReferenceMacroUnsupported {
                        reason: "typed_front_end_cache_ways_out_of_range",
                    }
                })?;
                let sets = state
                    .sets
                    .iter()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| InstructionCacheEntry {
                                key: InstructionLineKey {
                                    function_unique_id: entry.function_unique_id,
                                    line: entry.line,
                                },
                                ready_at: entry.ready_at,
                                last_access: entry.last_access,
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(SetAssociativeInstructionCache {
                    sets,
                    ways,
                    access_sequence: state.access_sequence,
                })
            };
        let l0 = boundary
            .l0
            .iter()
            .map(cache)
            .collect::<Result<Vec<_>, _>>()?;
        let l0: [SetAssociativeInstructionCache; SUB_CORES] =
            l0.try_into()
                .map_err(|_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l0_shape_invalid",
                })?;
        let streams = boundary
            .streams
            .iter()
            .map(|stream| InstructionStreamBuffer {
                lines: stream
                    .lines
                    .iter()
                    .map(|line| StreamBufferLine {
                        key: InstructionLineKey {
                            function_unique_id: line.function_unique_id,
                            line: line.line,
                        },
                        ready_at: line.ready_at,
                        demanded: line.demanded,
                    })
                    .collect(),
                next_line: stream
                    .next_line
                    .map(|(function_unique_id, line)| InstructionLineKey {
                        function_unique_id,
                        line,
                    }),
            })
            .collect::<Vec<_>>();
        let streams: [InstructionStreamBuffer; SUB_CORES] =
            streams
                .try_into()
                .map_err(|_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_stream_shape_invalid",
                })?;
        let config = InstructionFrontEnd {
            line_bytes: boundary.line_bytes,
            l0_sets: u32::try_from(boundary.l0.first().map_or(0, |cache| cache.sets.len()))
                .map_err(|_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l0_sets_out_of_range",
                })?,
            l0_ways: u32::try_from(boundary.l0.first().map_or(0, |cache| cache.ways)).map_err(
                |_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l0_ways_out_of_range",
                },
            )?,
            l1_sets: u32::try_from(boundary.l1.sets.len()).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l1_sets_out_of_range",
                }
            })?,
            l1_ways: u32::try_from(boundary.l1.ways).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l1_ways_out_of_range",
                }
            })?,
            l0_to_l1_cycles: boundary.l0_to_l1_cycles,
            l1_to_l0_cycles: boundary.l1_to_l0_cycles,
            l1_request_ports: boundary.l1_request_ports,
            l1_reply_ports: boundary.l1_reply_ports,
            stream_buffer_lines: boundary.stream_buffer_lines,
            stream_buffers_per_subcore: boundary.stream_buffers_per_subcore,
            prefetches_per_cycle: boundary.prefetches_per_cycle,
            instruction_buffer_entries: boundary.instruction_buffer_entries,
            fetch_decode_width: boundary.fetch_decode_width,
            fetch_to_issue_stages: boundary.fetch_to_issue_stages,
        };
        Ok(Self {
            config,
            l0,
            l1: cache(&boundary.l1)?,
            streams,
            l1_request_ready: boundary.l1_request_ready,
            l1_reply_ready: boundary.l1_reply_ready,
            l0_response_ready: boundary.l0_response_ready.clone().try_into().map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_response_shape_invalid",
                }
            })?,
            prefetch_priority: usize::try_from(boundary.prefetch_priority).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_prefetch_priority_invalid",
                }
            })?,
            cold_fill_latency: boundary.cold_fill_latency,
            diagnostics: InstructionFrontEndDiagnostics {
                instructions_fetched: boundary.instructions_fetched,
                demand_l0_hits: boundary.demand_l0_hits,
                demand_stream_buffer_hits: boundary.demand_stream_buffer_hits,
                demand_l0_misses: boundary.demand_l0_misses,
                demand_l1_hits: boundary.demand_l1_hits,
                demand_l1_misses: boundary.demand_l1_misses,
                prefetch_requests: boundary.prefetch_requests,
                prefetch_l1_hits: boundary.prefetch_l1_hits,
                prefetch_l1_misses: boundary.prefetch_l1_misses,
                stream_restarts: boundary.stream_restarts,
                maximum_fetch_latency_cycles: boundary.maximum_fetch_latency_cycles,
                fetch_last_issued_continuations: 0,
                yield_hold_separations: 0,
                control_flow_redirects: 0,
            },
        })
    }

    pub(super) fn instruction_buffer_capacity(&self) -> Result<usize, IssueError> {
        usize::try_from(self.config.instruction_buffer_entries).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component: "instruction_front_end",
                field: "instruction_buffer_entries",
                value: u64::from(self.config.instruction_buffer_entries),
            }
        })
    }

    pub(super) fn line_key(&self, function_unique_id: i32, pc: u32) -> InstructionLineKey {
        InstructionLineKey {
            function_unique_id,
            line: u64::from(pc) / u64::from(self.config.line_bytes),
        }
    }

    pub(super) fn access_l1(
        &mut self,
        key: InstructionLineKey,
        now: u64,
        kind: InstructionL1RequestKind,
    ) -> Result<u64, IssueError> {
        let request_depart = now.max(self.l1_request_ready);
        self.l1_request_ready = checked_cycle_add(
            "instruction L1 request-port availability",
            request_depart,
            1,
        )?;
        let request_arrival = checked_cycle_add(
            "instruction L0-to-L1 transport",
            request_depart,
            self.config.l0_to_l1_cycles,
        )?;
        let line_ready = if let Some(ready_at) = self.l1.probe(key)? {
            match kind {
                InstructionL1RequestKind::Demand => {
                    self.diagnostics.demand_l1_hits = checked_u64_counter_add(
                        "instruction demand L1 hit",
                        self.diagnostics.demand_l1_hits,
                        1,
                    )?;
                }
                InstructionL1RequestKind::Prefetch => {
                    self.diagnostics.prefetch_l1_hits = checked_u64_counter_add(
                        "instruction prefetch L1 hit",
                        self.diagnostics.prefetch_l1_hits,
                        1,
                    )?;
                }
            }
            request_arrival.max(ready_at)
        } else {
            match kind {
                InstructionL1RequestKind::Demand => {
                    self.diagnostics.demand_l1_misses = checked_u64_counter_add(
                        "instruction demand L1 miss",
                        self.diagnostics.demand_l1_misses,
                        1,
                    )?;
                }
                InstructionL1RequestKind::Prefetch => {
                    self.diagnostics.prefetch_l1_misses = checked_u64_counter_add(
                        "instruction prefetch L1 miss",
                        self.diagnostics.prefetch_l1_misses,
                        1,
                    )?;
                }
            }
            let ready_at = checked_cycle_add(
                "instruction L1 cold fill",
                request_arrival,
                self.cold_fill_latency,
            )?;
            self.l1.reserve(key, ready_at)?;
            ready_at
        };
        let reply_depart = line_ready.max(self.l1_reply_ready);
        self.l1_reply_ready =
            checked_cycle_add("instruction L1 reply-port availability", reply_depart, 1)?;
        checked_cycle_add(
            "instruction L1-to-L0 transport",
            reply_depart,
            self.config.l1_to_l0_cycles,
        )
    }

    pub(super) fn fetch_instruction(
        &mut self,
        subcore: usize,
        function_unique_id: i32,
        pc: u32,
        now: u64,
    ) -> Result<u64, IssueError> {
        let key = self.line_key(function_unique_id, pc);
        self.diagnostics.instructions_fetched = checked_u64_counter_add(
            "instruction front-end fetch",
            self.diagnostics.instructions_fetched,
            1,
        )?;
        let line_ready = if let Some(ready_at) = self.l0[subcore].probe(key)? {
            self.diagnostics.demand_l0_hits = checked_u64_counter_add(
                "instruction demand L0 hit",
                self.diagnostics.demand_l0_hits,
                1,
            )?;
            ready_at
        } else if let Some(ready_at) = self.streams[subcore].demand(key) {
            self.diagnostics.demand_stream_buffer_hits = checked_u64_counter_add(
                "instruction stream-buffer hit",
                self.diagnostics.demand_stream_buffer_hits,
                1,
            )?;
            self.l0[subcore].reserve(key, ready_at)?;
            ready_at
        } else {
            self.diagnostics.demand_l0_misses = checked_u64_counter_add(
                "instruction demand L0 miss",
                self.diagnostics.demand_l0_misses,
                1,
            )?;
            let ready_at = self.access_l1(key, now, InstructionL1RequestKind::Demand)?;
            self.l0[subcore].reserve(key, ready_at)?;
            if self.streams[subcore].can_restart() {
                self.streams[subcore].restart_after(key);
                self.diagnostics.stream_restarts = checked_u64_counter_add(
                    "instruction stream restart",
                    self.diagnostics.stream_restarts,
                    1,
                )?;
            }
            ready_at
        };
        let response_at = now.max(line_ready).max(self.l0_response_ready[subcore]);
        self.l0_response_ready[subcore] =
            checked_cycle_add("instruction L0 response-port availability", response_at, 1)?;
        let ready_at = checked_cycle_add(
            "instruction fetch-to-issue pipeline",
            response_at,
            self.config.fetch_to_issue_stages,
        )?;
        self.diagnostics.maximum_fetch_latency_cycles = self
            .diagnostics
            .maximum_fetch_latency_cycles
            .max(ready_at.saturating_sub(now));
        Ok(ready_at)
    }

    pub(super) fn maintain(&mut self, now: u64) {
        for stream in &mut self.streams {
            stream.maintain(now);
        }
    }

    /// A demanded head can free a prefetch slot before any warp is eligible
    /// to issue. It remains a local wake source even when all instruction and
    /// stream buffers are currently full.
    pub(super) fn next_maintenance_time(&self) -> Option<u64> {
        self.streams
            .iter()
            .filter_map(|stream| stream.lines.front())
            .filter(|line| line.demanded)
            .map(|line| line.ready_at)
            .min()
    }

    pub(super) fn has_prefetch_work(&self) -> bool {
        let capacity = self.config.stream_buffer_lines as usize;
        self.streams
            .iter()
            .any(|stream| stream.can_prefetch(capacity))
    }

    /// Keep the normal fetch path allocation-free.  A shadow cycle is needed
    /// only near checked counter, cache-sequence, or timestamp boundaries,
    /// where a later fetch error could otherwise leave an earlier fetch
    /// partially committed.
    pub(super) fn needs_transactional_cycle(
        &self,
        fetches: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> bool {
        let fetch_count = fetches.iter().flatten().count();
        let prefetch_count = usize::from(self.has_prefetch_work());
        let access_count = fetch_count.saturating_add(prefetch_count);
        let fetches_u64 = u64::try_from(fetch_count).unwrap_or(u64::MAX);
        let accesses_u64 = u64::try_from(access_count).unwrap_or(u64::MAX);
        let sequence_increments = accesses_u64.saturating_mul(2);
        let counter_overflows = |value: u64, amount: u64| value.checked_add(amount).is_none();

        if counter_overflows(self.diagnostics.instructions_fetched, fetches_u64)
            || counter_overflows(self.diagnostics.demand_l0_hits, fetches_u64)
            || counter_overflows(self.diagnostics.demand_stream_buffer_hits, fetches_u64)
            || counter_overflows(self.diagnostics.demand_l0_misses, fetches_u64)
            || counter_overflows(self.diagnostics.demand_l1_hits, accesses_u64)
            || counter_overflows(self.diagnostics.demand_l1_misses, accesses_u64)
            || counter_overflows(
                self.diagnostics.prefetch_requests,
                u64::try_from(prefetch_count).unwrap_or(u64::MAX),
            )
            || counter_overflows(
                self.diagnostics.prefetch_l1_hits,
                u64::try_from(prefetch_count).unwrap_or(u64::MAX),
            )
            || counter_overflows(
                self.diagnostics.prefetch_l1_misses,
                u64::try_from(prefetch_count).unwrap_or(u64::MAX),
            )
            || counter_overflows(self.diagnostics.stream_restarts, fetches_u64)
            || self
                .l1
                .access_sequence
                .checked_add(sequence_increments)
                .is_none()
            || self.l1_request_ready.checked_add(accesses_u64).is_none()
            || self.l1_reply_ready.checked_add(accesses_u64).is_none()
        {
            return true;
        }
        for (subcore, fetch) in fetches.iter().enumerate() {
            if fetch.is_some()
                && (self.l0[subcore].access_sequence.checked_add(2).is_none()
                    || self.l0_response_ready[subcore].checked_add(1).is_none())
            {
                return true;
            }
        }
        let mut maximum_delay = 0_u64;
        for delta in [
            1,
            self.config.l0_to_l1_cycles,
            self.cold_fill_latency,
            1,
            self.config.l1_to_l0_cycles,
            1,
            self.config.fetch_to_issue_stages,
        ] {
            let Some(next) = maximum_delay.checked_add(delta) else {
                return true;
            };
            maximum_delay = next;
        }
        now.checked_add(maximum_delay).is_none()
            || self.l1_request_ready.checked_add(maximum_delay).is_none()
            || self.l1_reply_ready.checked_add(maximum_delay).is_none()
    }

    pub(super) fn prefetch_cycle(&mut self, now: u64) -> Result<(), IssueError> {
        let capacity = usize::try_from(self.config.stream_buffer_lines).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component: "instruction_front_end",
                field: "stream_buffer_lines",
                value: u64::from(self.config.stream_buffer_lines),
            }
        })?;
        let selected = (0..SUB_CORES)
            .map(|offset| (self.prefetch_priority + offset) % SUB_CORES)
            .find(|&subcore| self.streams[subcore].can_prefetch(capacity));
        let Some(subcore) = selected else {
            return Ok(());
        };
        let key =
            self.streams[subcore]
                .next_prefetch()
                .ok_or(IssueError::InstructionStreamOrder {
                    expected: None,
                    got: (0, 0),
                })?;
        self.diagnostics.prefetch_requests = checked_u64_counter_add(
            "instruction prefetch request",
            self.diagnostics.prefetch_requests,
            1,
        )?;
        let ready_at = self.access_l1(key, now, InstructionL1RequestKind::Prefetch)?;
        self.streams[subcore].push_prefetch(key, ready_at)?;
        self.prefetch_priority = (subcore + 1) % SUB_CORES;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(super) struct ImmediateConstantCacheState {
    pub(super) config: ImmediateConstantCache,
    /// Per-set LRU tags, oldest at the front.  A line key is used as its tag
    /// identity after the set index has been derived; this is sufficient for
    /// the closed constant-address domain and avoids address aliases.
    pub(super) sets: Vec<VecDeque<u64>>,
    pub(super) pending: Vec<ImmediateConstantPending>,
    /// The pipelined SM-shared L1C access path admits lines at its measured
    /// initiation interval while the MSHR count bounds lines outstanding.
    pub(super) next_service: u64,
    pub(super) diagnostics: ImmediateConstantCacheDiagnostics,
}

impl ImmediateConstantCacheState {
    pub(super) fn new(config: ImmediateConstantCache) -> Result<Self, IssueError> {
        if config.sets == 0
            || config.line_bytes == 0
            || config.ways == 0
            || config.mshr_entries == 0
            || config.access_queue_entries == 0
            || config.miss_initiation_interval_cycles == 0
            || config.sm_structure_latency_cycles == 0
            || config.l0_to_l1_cycles == 0
            || config.switch_after_miss_cycles == 0
        {
            return Err(IssueError::ImmediateConstantCacheConfigurationInvalid);
        }
        Ok(Self {
            config,
            sets: (0..config.sets).map(|_| VecDeque::new()).collect(),
            pending: Vec::new(),
            next_service: 0,
            diagnostics: ImmediateConstantCacheDiagnostics::default(),
        })
    }

    pub(super) fn set_index(&self, line: u64) -> usize {
        (line % u64::from(self.config.sets)) as usize
    }

    pub(super) fn contains(&self, line: u64) -> bool {
        self.sets[self.set_index(line)].contains(&line)
    }

    pub(super) fn pending_ready(&self, line: u64) -> Option<u64> {
        self.pending
            .iter()
            .find(|entry| entry.line == line)
            .map(|entry| entry.ready_at)
    }

    pub(super) fn touch(&mut self, line: u64) {
        let set_index = self.set_index(line);
        let set = &mut self.sets[set_index];
        if let Some(position) = set.iter().position(|&tag| tag == line) {
            set.remove(position);
        }
        set.push_back(line);
        while set.len() > self.config.ways as usize {
            set.pop_front();
        }
    }

    pub(super) fn complete_ready(&mut self, now: u64) {
        let mut ready = Vec::new();
        self.pending.retain(|entry| {
            if entry.ready_at <= now {
                ready.push(entry.line);
                false
            } else {
                true
            }
        });
        // `pending` is issue order. Preserve it when multiple lines have
        // become ready since the last scheduler step: sorting by line address
        // would change same-timestamp fill order and therefore LRU victims.
        // Pending-line identity is unique by `reserve`, so no deduplication is
        // needed here.
        // Diagnostic counters are observational; the execution state itself
        // remains checked. Saturation keeps a pathological trace from turning
        // reporting into a scheduler failure.
        self.diagnostics.lines_filled = self
            .diagnostics
            .lines_filled
            .saturating_add(u64::try_from(ready.len()).unwrap_or(u64::MAX));
        for line in ready {
            self.touch(line);
        }
    }

    /// Reserve every currently missing line referenced by one instruction.
    /// Returns the earliest time at which the instruction may be re-issued;
    /// `None` means all references are already resident.  New misses use one
    /// pipelined L1C service lane; an existing pending line is shared by all
    /// waiting warps on that SMSP.
    pub(super) fn reserve(
        &mut self,
        lines: &[u64],
        now: u64,
        upper_level_latency: u64,
    ) -> Result<Option<u64>, IssueError> {
        let mut wait_until: Option<u64> = None;
        // `upper_level_latency` is the carded L1/L2 response frontier used by
        // the shared event system.  The 11-stage local structure and its
        // 39-cycle transport are already represented in the NCU-calibrated
        // response frontier for this replay; adding them here would charge
        // the same path twice.  The local fields remain part of the capability
        // identity and diagnostics, while this actor waits on the single
        // externally observable response timestamp.
        let miss_latency = upper_level_latency;
        for &line in lines {
            self.diagnostics.references = self.diagnostics.references.saturating_add(1);
            if self.contains(line) {
                self.diagnostics.lookup_hits = self.diagnostics.lookup_hits.saturating_add(1);
                continue;
            }
            self.diagnostics.lookup_misses = self.diagnostics.lookup_misses.saturating_add(1);
            if let Some(ready) = self.pending_ready(line) {
                self.diagnostics.pending_tag_hits =
                    self.diagnostics.pending_tag_hits.saturating_add(1);
                self.diagnostics.pending_merges = self.diagnostics.pending_merges.saturating_add(1);
                wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
                continue;
            }
            if self.pending.len() >= self.config.mshr_entries as usize {
                if let Some(ready) = self.pending.iter().map(|entry| entry.ready_at).min() {
                    wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
                }
                continue;
            }
            let start = now.max(self.next_service);
            let ready = start
                .checked_add(miss_latency)
                .ok_or(IssueError::TimeOverflow {
                    operation: "immediate constant-cache line response",
                    base: start,
                    delta: miss_latency,
                })?;
            // The 11-stage L1C structure is a pipelined queue: its one-entry
            // access port accepts a new line each cycle while earlier misses
            // continue through the latency stages.  Serializing the port by
            // the full response latency would turn independent CBANK misses
            // into a bandwidth bottleneck that the released SM86 model does
            // not have.  Keep the response timestamp serialized only by the
            // measured initiation interval (one cycle for the characterized
            // queue), while `ready_at` still carries the full miss latency.
            self.next_service = start
                .checked_add(self.config.miss_initiation_interval_cycles)
                .ok_or(IssueError::TimeOverflow {
                    operation: "immediate constant-cache service initiation",
                    base: start,
                    delta: self.config.miss_initiation_interval_cycles,
                })?;
            self.pending.push(ImmediateConstantPending {
                line,
                ready_at: ready,
            });
            self.diagnostics.tag_misses = self.diagnostics.tag_misses.saturating_add(1);
            self.diagnostics.maximum_pending_lines = self
                .diagnostics
                .maximum_pending_lines
                .max(u64::try_from(self.pending.len()).unwrap_or(u64::MAX));
            wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
        }
        Ok(wait_until)
    }

    pub(super) fn diagnostics(&self) -> ImmediateConstantCacheDiagnostics {
        self.diagnostics
    }
}
