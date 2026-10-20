// Lexical implementation section 2/8; included by the parent facade.
/// One SM's active instruction hierarchy. Lower-level cold fills use the
/// independently carded dependent-DRAM latency; L0/L1 geometry and transport
/// come from the architecture capability above.
#[derive(Debug, Clone)]
struct InstructionFrontEndState {
    config: InstructionFrontEnd,
    l0: [SetAssociativeInstructionCache; SUB_CORES],
    l1: SetAssociativeInstructionCache,
    streams: [InstructionStreamBuffer; SUB_CORES],
    l1_request_ready: u64,
    l1_reply_ready: u64,
    l0_response_ready: [u64; SUB_CORES],
    prefetch_priority: usize,
    cold_fill_latency: u64,
    diagnostics: InstructionFrontEndDiagnostics,
}

impl InstructionFrontEndState {
    fn new(config: InstructionFrontEnd, cold_fill_latency: u64) -> Result<Self, IssueError> {
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
    fn from_typed_boundary(
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

    fn instruction_buffer_capacity(&self) -> Result<usize, IssueError> {
        usize::try_from(self.config.instruction_buffer_entries).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component: "instruction_front_end",
                field: "instruction_buffer_entries",
                value: u64::from(self.config.instruction_buffer_entries),
            }
        })
    }

    fn line_key(&self, function_unique_id: i32, pc: u32) -> InstructionLineKey {
        InstructionLineKey {
            function_unique_id,
            line: u64::from(pc) / u64::from(self.config.line_bytes),
        }
    }

    fn access_l1(
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

    fn fetch_instruction(
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

    fn maintain(&mut self, now: u64) {
        for stream in &mut self.streams {
            stream.maintain(now);
        }
    }

    /// A demanded head can free a prefetch slot before any warp is eligible
    /// to issue. It remains a local wake source even when all instruction and
    /// stream buffers are currently full.
    fn next_maintenance_time(&self) -> Option<u64> {
        self.streams
            .iter()
            .filter_map(|stream| stream.lines.front())
            .filter(|line| line.demanded)
            .map(|line| line.ready_at)
            .min()
    }

    fn has_prefetch_work(&self) -> bool {
        let capacity = self.config.stream_buffer_lines as usize;
        self.streams
            .iter()
            .any(|stream| stream.can_prefetch(capacity))
    }

    /// Keep the normal fetch path allocation-free.  A shadow cycle is needed
    /// only near checked counter, cache-sequence, or timestamp boundaries,
    /// where a later fetch error could otherwise leave an earlier fetch
    /// partially committed.
    fn needs_transactional_cycle(&self, fetches: &[Option<usize>; SUB_CORES], now: u64) -> bool {
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

    fn prefetch_cycle(&mut self, now: u64) -> Result<(), IssueError> {
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

#[derive(Debug, Clone, Copy)]
struct BufferedInstruction {
    instruction: usize,
    ready_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryOperandWidth {
    Bits32,
    Bits64,
    Bits128,
}

fn memory_operand_width(opcode: &str) -> MemoryOperandWidth {
    if opcode.split('.').any(|modifier| modifier == "128") {
        MemoryOperandWidth::Bits128
    } else if opcode.split('.').any(|modifier| modifier == "64") {
        MemoryOperandWidth::Bits64
    } else {
        MemoryOperandWidth::Bits32
    }
}

fn memory_address_is_uniform(opcode_text: &str) -> bool {
    opcode_text
        .split_once('[')
        .and_then(|(_, tail)| tail.split_once(']'))
        .is_some_and(|(address, _)| {
            address
                .split(|character: char| !character.is_ascii_alphanumeric())
                .any(|token| token.starts_with("UR"))
        })
}

/// Identify regular-register constant-cache-qualified loads from static SASS.
/// They continue through the normal global/L1/L2 allocation path; the marker
/// only selects the SM86 constant-return service when the shared system prices
/// the response.  Keeping this classification textual avoids treating dynamic
/// trace metadata or measured NCU values as simulator inputs.
fn is_constant_load_opcode(opcode_text: &str) -> bool {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    opcode.split('.').any(|modifier| modifier == "CONSTANT")
        && matches!(opcode_head(opcode_text), "LDG" | "LD" | "LDC" | "LDL")
}

fn is_u16_opcode(opcode_text: &str) -> bool {
    opcode_text
        .split_ascii_whitespace()
        .next()
        .is_some_and(|opcode| opcode.split('.').any(|modifier| modifier == "U16"))
}

/// Count distinct static constant-load request streams. The released SM86
/// memory model has a five-entry local memory dispatch window (four queue
/// entries plus its dispatch latch); a stream count beyond that window is the
/// conservative point at which the L0C/GCC transport can remain occupied.
/// This is derived solely from SASS shape, never from a workload id or an NCU
/// target.
fn constant_stream_count(schedule: &FunctionSchedule) -> usize {
    schedule
        .by_pc
        .values()
        .filter(|slot| is_constant_load_opcode(slot.opcode_text()))
        .count()
}

fn deep_constant_load(
    slot: &IssueSlot,
    stream_count: usize,
    queue_entries_with_dispatch: u32,
) -> bool {
    let pressure_threshold = usize::try_from(queue_entries_with_dispatch)
        .unwrap_or(usize::MAX)
        .saturating_add(1);
    let stream_pressure = stream_count > pressure_threshold;
    let halfword_pressure = is_u16_opcode(slot.opcode_text())
        && stream_count >= pressure_threshold
        && slot
            .control
            .write_barrier
            .is_some_and(|barrier| barrier.wire() > 2);
    is_constant_load_opcode(slot.opcode_text())
        && ((!is_u16_opcode(slot.opcode_text()) && stream_pressure) || halfword_pressure)
}

/// Select the observer-only write-tag window for SM86 global stores.  The
/// 256-set/2-tag-slot window is an architecture-level report-only capability;
/// it must be instantiated for every cacheable global-store stream so the
/// consumer cannot silently read a disabled counter on ordinary STG launches.
/// Bypass/untagged forms never reach this `GlobalStore` path and therefore do
/// not allocate observer tags.
pub(crate) fn stateful_write_lookup_observer_enabled(schedule: &FunctionSchedule) -> bool {
    if !matches!(
        schedule.arch.l1_write_lookup_window().status(),
        sim_arch::CapabilityStatus::Available
    ) {
        return false;
    }
    let mut has_global_store = false;
    for slot in schedule.by_pc.values() {
        let head = opcode_head(slot.opcode_text());
        has_global_store |= matches!(head, "STG" | "ST" | "STL");
    }
    has_global_store
}

#[derive(Debug, Clone, Copy)]
struct MemoryPipelineSlot {
    operand_ready_latency: u64,
    constant_pressure: bool,
}

fn memory_pipeline_slot(
    slot: &IssueSlot,
    config: MemoryIssuePipeline,
    deep_constant: bool,
) -> Option<MemoryPipelineSlot> {
    let head = opcode_head(slot.opcode_text());
    let uniform = memory_address_is_uniform(slot.opcode_text());
    let width = memory_operand_width(
        slot.opcode_text()
            .split_ascii_whitespace()
            .next()
            .unwrap_or(""),
    );
    let store_latency = |shared: bool| match (shared, uniform, width) {
        (false, true, MemoryOperandWidth::Bits32) => config.global_store_uniform_32_war_cycles,
        (false, true, MemoryOperandWidth::Bits64) => config.global_store_uniform_64_war_cycles,
        (false, true, MemoryOperandWidth::Bits128) => config.global_store_uniform_128_war_cycles,
        (false, false, MemoryOperandWidth::Bits32) => config.global_store_regular_32_war_cycles,
        (false, false, MemoryOperandWidth::Bits64) => config.global_store_regular_64_war_cycles,
        (false, false, MemoryOperandWidth::Bits128) => config.global_store_regular_128_war_cycles,
        (true, true, MemoryOperandWidth::Bits32) => config.shared_store_uniform_32_war_cycles,
        (true, true, MemoryOperandWidth::Bits64) => config.shared_store_uniform_64_war_cycles,
        (true, true, MemoryOperandWidth::Bits128) => config.shared_store_uniform_128_war_cycles,
        (true, false, MemoryOperandWidth::Bits32) => config.shared_store_regular_32_war_cycles,
        (true, false, MemoryOperandWidth::Bits64) => config.shared_store_regular_64_war_cycles,
        (true, false, MemoryOperandWidth::Bits128) => config.shared_store_regular_128_war_cycles,
    };
    let operand_ready_latency = match head {
        "LDGSTS" => config.ldgsts_war_cycles,
        "LDG" | "LD" | "LDC" | "LDL" => {
            if deep_constant {
                config.constant_load_war_cycles
            } else if uniform {
                config.global_load_uniform_war_cycles
            } else {
                config.global_load_regular_war_cycles
            }
        }
        "STG" | "STL" => store_latency(false),
        "LDS" | "LDSM" => config.shared_load_war_cycles,
        "STS" => store_latency(true),
        _ => return None,
    };
    Some(MemoryPipelineSlot {
        operand_ready_latency,
        constant_pressure: deep_constant,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct MemoryPipelineIssue {
    operand_ready_at: u64,
    shared_accept_at: u64,
}

#[derive(Debug, Clone)]
struct MemoryIssuePipelineState {
    config: MemoryIssuePipeline,
    queue_release_times: [VecDeque<u64>; SUB_CORES],
    local_completion_ready: [u64; SUB_CORES],
    shared_accept_ready: u64,
    diagnostics: MemoryIssuePipelineDiagnostics,
}

#[derive(Debug, Clone)]
struct MemoryIssuePipelinePlan {
    next: MemoryIssuePipelineState,
    issues: [Option<MemoryPipelineIssue>; SUB_CORES],
}

/// Scheduler-visible occupancy for the architecture's scalar INT/FP
/// execution stream.  Dispatch remains one instruction per SMSP cycle; an
/// entry reserves a future execution slot and leaves the window when that
/// slot starts.  This captures finite pipelined service without turning the
/// model into a register scoreboard or a per-cycle global scan.
/// Research-only scalar resource candidate. It is intentionally not wired
/// into production IssueModelParams; the bounded issue-port probe found that a
/// single generic queue does not transfer across independent and mixed
/// workloads. Keeping the implementation here makes future probes
/// reproducible without silently changing the deployed lane.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ScalarExecutionPipelineState {
    config: ScalarExecutionPipeline,
    release_times: [VecDeque<u64>; SUB_CORES],
    next_service: [u64; SUB_CORES],
    diagnostics: ScalarExecutionPipelineDiagnostics,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ScalarExecutionPipelineDiagnostics {
    pub instructions_admitted: u64,
    pub service_busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
}

impl ScalarExecutionPipelineDiagnostics {
    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        for (field, target, value) in [
            (
                "readiness.scalar_execution_pipeline.instructions_admitted",
                &mut next.instructions_admitted,
                addend.instructions_admitted,
            ),
            (
                "readiness.scalar_execution_pipeline.service_busy_cycles",
                &mut next.service_busy_cycles,
                addend.service_busy_cycles,
            ),
            (
                "readiness.scalar_execution_pipeline.queue_delay_cycles",
                &mut next.queue_delay_cycles,
                addend.queue_delay_cycles,
            ),
        ] {
            *target = checked_readiness_add(field, *target, value)?;
        }
        next.maximum_queue_occupancy = next
            .maximum_queue_occupancy
            .max(addend.maximum_queue_occupancy);
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ScalarExecutionPipelinePlan {
    next: ScalarExecutionPipelineState,
}

#[allow(dead_code)]
impl ScalarExecutionPipelineState {
    fn new(config: ScalarExecutionPipeline) -> Result<Self, IssueError> {
        if config.latency_cycles == 0
            || config.initiation_interval_cycles == 0
            || config.in_flight_capacity == 0
        {
            return Err(IssueError::ScalarExecutionPipelineConfigurationInvalid {
                latency: config.latency_cycles,
                initiation: config.initiation_interval_cycles,
                capacity: config.in_flight_capacity,
            });
        }
        Ok(Self {
            config,
            release_times: std::array::from_fn(|_| VecDeque::new()),
            next_service: [0; SUB_CORES],
            diagnostics: ScalarExecutionPipelineDiagnostics::default(),
        })
    }

    fn prune(&mut self, now: u64) {
        for queue in &mut self.release_times {
            while queue.front().is_some_and(|&release| release <= now) {
                queue.pop_front();
            }
        }
    }

    fn admission_ready_at(&self, subcore: usize) -> u64 {
        let queue = &self.release_times[subcore];
        if queue.len() < self.config.in_flight_capacity as usize {
            0
        } else {
            queue.front().copied().unwrap_or(0)
        }
    }

    fn plan(
        &self,
        selected: [Option<PipelineClass>; SUB_CORES],
        now: u64,
    ) -> Result<ScalarExecutionPipelinePlan, IssueError> {
        let mut next = self.clone();
        for (subcore, slot) in selected.into_iter().enumerate() {
            let Some(PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            }) = slot
            else {
                continue;
            };
            if next.release_times[subcore].len() >= self.config.in_flight_capacity as usize {
                return Err(IssueError::ScalarExecutionPipelineQueueFull { subcore, time: now });
            }
            let ready_for_service =
                checked_cycle_add("scalar execution pipeline latency", now, latency_cycles)?;
            let service_start = ready_for_service.max(next.next_service[subcore]);
            next.next_service[subcore] = checked_cycle_add(
                "scalar execution pipeline initiation",
                service_start,
                initiation_interval_cycles,
            )?;
            next.release_times[subcore].push_back(service_start);
            let diagnostics = &mut next.diagnostics;
            diagnostics.instructions_admitted = checked_u64_counter_add(
                "scalar execution pipeline admitted instruction",
                diagnostics.instructions_admitted,
                1,
            )?;
            diagnostics.service_busy_cycles = checked_u64_counter_add(
                "scalar execution pipeline service work",
                diagnostics.service_busy_cycles,
                initiation_interval_cycles,
            )?;
            diagnostics.queue_delay_cycles = checked_u64_counter_add(
                "scalar execution pipeline queue delay",
                diagnostics.queue_delay_cycles,
                service_start.saturating_sub(ready_for_service),
            )?;
            diagnostics.maximum_queue_occupancy = diagnostics.maximum_queue_occupancy.max(
                u64::try_from(next.release_times[subcore].len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "scalar execution pipeline queue occupancy",
                    }
                })?,
            );
        }
        Ok(ScalarExecutionPipelinePlan { next })
    }
}

impl MemoryIssuePipelineState {
    fn new(config: MemoryIssuePipeline) -> Result<Self, IssueError> {
        if config.queue_entries_with_dispatch == 0 {
            return Err(IssueError::MemoryIssuePipelineConfigurationInvalid {
                field: "queue_entries_with_dispatch",
                value: 0,
            });
        }
        for (field, value) in [
            (
                "local_initiation_interval_cycles",
                config.local_initiation_interval_cycles,
            ),
            (
                "shared_accept_interval_cycles",
                config.shared_accept_interval_cycles,
            ),
        ] {
            if value == 0 {
                return Err(IssueError::MemoryIssuePipelineConfigurationInvalid { field, value });
            }
        }
        Ok(Self {
            config,
            queue_release_times: std::array::from_fn(|_| VecDeque::new()),
            local_completion_ready: [0; SUB_CORES],
            shared_accept_ready: 0,
            diagnostics: MemoryIssuePipelineDiagnostics::default(),
        })
    }

    fn prune(&mut self, now: u64) {
        for queue in &mut self.queue_release_times {
            while queue.front().is_some_and(|&release| release <= now) {
                queue.pop_front();
            }
        }
    }

    fn admission_ready_at(&self, subcore: usize) -> u64 {
        let queue = &self.queue_release_times[subcore];
        if queue.len() < self.config.queue_entries_with_dispatch as usize {
            0
        } else {
            queue.front().copied().unwrap_or(0)
        }
    }

    fn plan(
        &self,
        selected: [Option<MemoryPipelineSlot>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryIssuePipelinePlan, IssueError> {
        let mut next = self.clone();
        let mut issues = [None; SUB_CORES];
        for (subcore, slot) in selected.into_iter().enumerate() {
            let Some(slot) = slot else { continue };
            if next.queue_release_times[subcore].len()
                >= next.config.queue_entries_with_dispatch as usize
            {
                return Err(IssueError::MemoryIssuePipelineQueueFull { subcore, time: now });
            }
            let nominal_operand_ready = checked_cycle_add(
                "memory operand/address readiness",
                now,
                slot.operand_ready_latency,
            )?;
            let operand_ready_at = nominal_operand_ready.max(next.local_completion_ready[subcore]);
            next.local_completion_ready[subcore] = checked_cycle_add(
                "memory local initiation interval",
                operand_ready_at,
                next.config.local_initiation_interval_cycles,
            )?;
            let shared_accept_at = operand_ready_at.max(next.shared_accept_ready);
            next.shared_accept_ready = checked_cycle_add(
                "memory SM-shared acceptance interval",
                shared_accept_at,
                next.config.shared_accept_interval_cycles,
            )?;
            next.queue_release_times[subcore].push_back(shared_accept_at);
            let diagnostics = &mut next.diagnostics;
            diagnostics.instructions_admitted = checked_u64_counter_add(
                "memory-pipeline admitted instruction",
                diagnostics.instructions_admitted,
                1,
            )?;
            if slot.constant_pressure {
                diagnostics.constant_pressure_instructions = checked_u64_counter_add(
                    "memory-pipeline constant-pressure instruction",
                    diagnostics.constant_pressure_instructions,
                    1,
                )?;
            }
            diagnostics.operand_pipeline_delay_cycles = checked_u64_counter_add(
                "memory-pipeline operand delay",
                diagnostics.operand_pipeline_delay_cycles,
                operand_ready_at - nominal_operand_ready,
            )?;
            diagnostics.shared_queue_delay_cycles = checked_u64_counter_add(
                "memory-pipeline shared queue delay",
                diagnostics.shared_queue_delay_cycles,
                shared_accept_at - operand_ready_at,
            )?;
            diagnostics.maximum_operand_pipeline_delay_cycles = diagnostics
                .maximum_operand_pipeline_delay_cycles
                .max(operand_ready_at - nominal_operand_ready);
            diagnostics.maximum_shared_queue_delay_cycles = diagnostics
                .maximum_shared_queue_delay_cycles
                .max(shared_accept_at - operand_ready_at);
            diagnostics.maximum_queue_occupancy = diagnostics.maximum_queue_occupancy.max(
                u64::try_from(next.queue_release_times[subcore].len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "memory-pipeline queue occupancy",
                    }
                })?,
            );
            diagnostics.terminal_shared_accept_cycle = diagnostics
                .terminal_shared_accept_cycle
                .max(shared_accept_at);
            issues[subcore] = Some(MemoryPipelineIssue {
                operand_ready_at,
                shared_accept_at,
            });
        }
        Ok(MemoryIssuePipelinePlan { next, issues })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultWritebackKind {
    FixedHadd2 { bank: u8 },
    L1TexStrongSmU32 { bank: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingVariableWriteback {
    subcore: u8,
    bank: u8,
    writeback_at: u64,
    visible_at: u64,
    warp: usize,
    barrier: DepBarrier,
}

#[derive(Debug)]
struct ResultWritebackState {
    timing: FixedL1TexWritebackPriority,
    pending: Vec<PendingVariableWriteback>,
}

fn scalar_regular_destination_bank(opcode_text: &str) -> Option<u8> {
    let (_, operands) = opcode_text.trim().split_once(char::is_whitespace)?;
    let destination = operands.split(',').next()?.trim();
    let index = destination.strip_prefix('R')?.parse::<u16>().ok()?;
    Some((index & 1) as u8)
}

fn is_clocklo_issue(slot: &IssueSlot) -> bool {
    slot.is_clocklo_read()
}

fn is_gpu_membar_issue(slot: &IssueSlot) -> bool {
    slot.is_gpu_membar()
}

fn result_writeback_kind(slot: &IssueSlot, arch: ArchId) -> Option<ResultWritebackKind> {
    if arch != ArchId::Sm86 {
        return None;
    }
    if let FixedLatencyOperandStatus::Projected(projection) = slot.fixed_latency_operands()
        && projection.opcode() == FixedLatencyOpcode::Hadd2
        && let [destination] = projection.destinations()
        && let Some(bank) = destination.register().sm86_regular_bank()
    {
        return Some(ResultWritebackKind::FixedHadd2 { bank });
    }
    if slot.mem_role() == MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm)
        && slot.opcode_text().split_whitespace().next() == Some("LDG.E.STRONG.SM")
    {
        return scalar_regular_destination_bank(slot.opcode_text())
            .map(|bank| ResultWritebackKind::L1TexStrongSmU32 { bank });
    }
    None
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DependencyOrigin {
    Read,
    GlobalLoadOperandRead,
    GlobalStoreOperandRead,
    GlobalAtomicOperandRead,
    GlobalReductionOperandRead,
    NonMemoryWrite,
    SharedWrite,
    L1Tex,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReadinessCause {
    InstructionFrontEnd,
    DependenceCounterRead,
    DependenceCounterGlobalLoadOperandRead,
    DependenceCounterGlobalStoreOperandRead,
    DependenceCounterGlobalAtomicOperandRead,
    DependenceCounterGlobalReductionOperandRead,
    DependenceCounterNonMemoryWrite,
    DependenceCounterSharedWrite,
    DependenceCounterL1Tex,
    DependenceCounterUnknownOrigin,
    AsyncSource,
    AsyncDestinationLocal,
    AsyncDestinationSmService,
    CtaBarrierRendezvous,
    PipelineBarrier,
    PipelineBranch,
    PipelineTensor,
    PipelineSfu,
    PipelineAlu,
    MemoryAdmissionQueue,
    MemoryIssueQueue,
    MemoryWriteCompletion,
    RegisterRaw,
    RegisterPredicate,
    RegisterWaw,
    ControlStall,
}

fn readiness_cause_name(cause: ReadinessCause) -> &'static str {
    match cause {
        ReadinessCause::AsyncSource => "async_source",
        ReadinessCause::AsyncDestinationLocal => "async_destination_local",
        ReadinessCause::AsyncDestinationSmService => "async_destination_sm_service",
        _ => "other_readiness_cause",
    }
}

impl From<DependencyOrigin> for ReadinessCause {
    fn from(origin: DependencyOrigin) -> Self {
        match origin {
            DependencyOrigin::Read => Self::DependenceCounterRead,
            DependencyOrigin::GlobalLoadOperandRead => Self::DependenceCounterGlobalLoadOperandRead,
            DependencyOrigin::GlobalStoreOperandRead => {
                Self::DependenceCounterGlobalStoreOperandRead
            }
            DependencyOrigin::GlobalAtomicOperandRead => {
                Self::DependenceCounterGlobalAtomicOperandRead
            }
            DependencyOrigin::GlobalReductionOperandRead => {
                Self::DependenceCounterGlobalReductionOperandRead
            }
            DependencyOrigin::NonMemoryWrite => Self::DependenceCounterNonMemoryWrite,
            DependencyOrigin::SharedWrite => Self::DependenceCounterSharedWrite,
            DependencyOrigin::L1Tex => Self::DependenceCounterL1Tex,
            DependencyOrigin::Unknown => Self::DependenceCounterUnknownOrigin,
        }
    }
}

impl ReadinessCauseCycles {
    fn add_cause(&mut self, cause: ReadinessCause, delta: u64) -> Result<(), IssueError> {
        let (field, counter): (&'static str, &mut u64) = match cause {
            ReadinessCause::InstructionFrontEnd => (
                "instruction front-end readiness",
                &mut self.instruction_front_end,
            ),
            ReadinessCause::RegisterRaw => ("register RAW", &mut self.register_raw),
            ReadinessCause::RegisterPredicate => {
                ("register predicate", &mut self.register_predicate)
            }
            ReadinessCause::RegisterWaw => ("register WAW", &mut self.register_waw),
            ReadinessCause::ControlStall => ("control stall readiness", &mut self.control_stall),
            ReadinessCause::DependenceCounterRead => (
                "read-dependence counter readiness",
                &mut self.dependence_counter_read,
            ),
            ReadinessCause::DependenceCounterGlobalLoadOperandRead => (
                "global-load operand-read counter readiness",
                &mut self.dependence_counter_global_load_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalStoreOperandRead => (
                "global-store operand-read counter readiness",
                &mut self.dependence_counter_global_store_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalAtomicOperandRead => (
                "global-atomic operand-read counter readiness",
                &mut self.dependence_counter_global_atomic_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalReductionOperandRead => (
                "global-reduction operand-read counter readiness",
                &mut self.dependence_counter_global_reduction_operand_read,
            ),
            ReadinessCause::DependenceCounterNonMemoryWrite => (
                "non-memory write-dependence counter readiness",
                &mut self.dependence_counter_non_memory_write,
            ),
            ReadinessCause::DependenceCounterSharedWrite => (
                "shared write-dependence counter readiness",
                &mut self.dependence_counter_shared_write,
            ),
            ReadinessCause::DependenceCounterL1Tex => (
                "L1TEX-dependence counter readiness",
                &mut self.dependence_counter_l1tex,
            ),
            ReadinessCause::DependenceCounterUnknownOrigin => (
                "unknown-origin dependence counter readiness",
                &mut self.dependence_counter_unknown_origin,
            ),
            ReadinessCause::AsyncSource => {
                ("asynchronous-copy source readiness", &mut self.async_source)
            }
            ReadinessCause::AsyncDestinationLocal => (
                "asynchronous-copy local destination readiness",
                &mut self.async_destination_local,
            ),
            ReadinessCause::AsyncDestinationSmService => (
                "asynchronous-copy SM destination-service readiness",
                &mut self.async_destination_sm_service,
            ),
            ReadinessCause::CtaBarrierRendezvous => (
                "CTA barrier rendezvous readiness",
                &mut self.cta_barrier_rendezvous,
            ),
            ReadinessCause::PipelineBarrier => {
                ("CTA-barrier pipeline readiness", &mut self.pipeline_barrier)
            }
            ReadinessCause::PipelineBranch => {
                ("branch-resolution readiness", &mut self.pipeline_branch)
            }
            ReadinessCause::PipelineTensor => {
                ("tensor-pipeline readiness", &mut self.pipeline_tensor)
            }
            ReadinessCause::PipelineSfu => {
                ("uniform/SFU-pipeline readiness", &mut self.pipeline_sfu)
            }
            ReadinessCause::PipelineAlu => {
                ("scalar-ALU pipeline readiness", &mut self.pipeline_alu)
            }
            ReadinessCause::MemoryAdmissionQueue => (
                "memory-admission queue readiness",
                &mut self.memory_admission_queue,
            ),
            ReadinessCause::MemoryIssueQueue => (
                "memory issue-pipeline queue readiness",
                &mut self.memory_issue_queue,
            ),
            ReadinessCause::MemoryWriteCompletion => (
                "memory write-completion readiness",
                &mut self.memory_write_completion,
            ),
        };
        *counter = checked_u64_counter_add(field, *counter, delta)?;
        Ok(())
    }

    fn add_eligible_not_selected(&mut self, delta: u64) -> Result<(), IssueError> {
        self.eligible_not_selected = checked_u64_counter_add(
            "eligible-not-selected readiness",
            self.eligible_not_selected,
            delta,
        )?;
        Ok(())
    }
}

fn checked_u64_counter_add(
    counter: &'static str,
    current: u64,
    add: u64,
) -> Result<u64, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

fn checked_usize_counter_add(
    counter: &'static str,
    current: usize,
    add: usize,
) -> Result<usize, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

fn checked_u32_counter_add(
    counter: &'static str,
    current: u32,
    add: u32,
) -> Result<u32, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

/// Warp-local identity of one nonempty asynchronous-copy group. The numeric
/// sequence is intentionally opaque outside the SM actor; shared-memory replies
/// may route an identity emitted by the actor but cannot construct one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncCopyGroupId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AsyncCopyGroup {
    id: AsyncCopyGroupId,
    /// Issue time of the `LDGDEPBAR` that made this group visible to a
    /// subsequent wait. Zero is also the valid initial-cycle timestamp; the
    /// group cannot be observed here until it is committed.
    committed_at: u64,
    /// Source sectors whose shared-L2 response time is not known yet.
    pending_sources: u32,
    /// Maximum known lower-level source completion time across flat/constant
    /// copies and shared-L2 responses.
    source_completion_time: u64,
    /// Route-weighted effective payload in the architecture capability's common
    /// fixed-point work domain. This is added once per issued copy, never inferred
    /// from or duplicated across its source sectors.
    delivery_work: u64,
    /// Exact cache-all destination wavefronts accumulated across this committed
    /// group. Zero retains the payload-only/bypass path.
    destination_wavefronts: u64,
    /// The aggregate SM service is assigned once, after commit and after every
    /// source unit has a known completion time.
    destination_service_queued: bool,
    /// Latest completion assigned by the aggregate per-SM destination service.
    destination_completion_time: u64,
    /// Maximum of the warp-local dependent-completion floor and the aggregate
    /// destination-service completion. Every mutation checks this value before
    /// publishing group state, leaving scheduler readiness infallible.
    completion_time: u64,
}

/// One source-bearing bypass LDGSTS instruction retained inside its
/// asynchronous-copy group. Topology-only zero-payload copies remain in the
/// group protocol but do not enter this source-response observation sidecar.
/// The production readiness path continues to use [`AsyncCopyGroup`]'s
/// established aggregate fields; this sidecar preserves the event ordering
/// needed to evaluate a future per-copy delivery model without changing time.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BypassAsyncCopyObservation {
    group: AsyncCopyGroupId,
    instruction: usize,
    pending_sources: u32,
    source_completion_time: u64,
    delivery_work: u64,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts: u64,
    max_bypass_wavefronts: u64,
    sector_off: usize,
    sector_membership_masks: Box<[u32]>,
    cohorts: Box<[BypassAsyncCopyCohortObservation]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassAsyncCopyInput<'a> {
    instruction: usize,
    sector_off: usize,
    sector_len: u16,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts: u64,
    max_bypass_wavefronts: u64,
    cohorts: Option<&'a [AsyncCopyCohortDescriptor]>,
    sector_membership_masks: Option<&'a [u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassAsyncCopyCohortObservation {
    pending_sources: u32,
    first_source_response: Option<u64>,
    last_source_response: Option<u64>,
    payload_work: u64,
    destination_lines: u64,
    bank_wavefronts: u64,
    bypass_wavefronts: u64,
    topology_penalty: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BypassAsyncCopyProjection {
    copies: u64,
    bank_wavefront_upper_bound: u64,
    bypass_wavefront_upper_bound: u64,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts_sum: u64,
    max_bypass_wavefronts_sum: u64,
    source_spread_cycles: u64,
    per_copy_completion_cycles: u64,
    group_aggregate_excess_cycles: u64,
    topology_candidate_penalty_cycles: u64,
    topology_candidate_completion_cycles: u64,
    topology_candidate_later_cycles: u64,
    topology_candidate_earlier_cycles: u64,
    pipeline_q1_completion_cycles: u64,
    pipeline_q1_exposure_cycles: u64,
    pipeline_q2_completion_cycles: u64,
    pipeline_q2_exposure_cycles: u64,
    pipeline_q4_completion_cycles: u64,
    pipeline_q4_exposure_cycles: u64,
    pipeline_q8_completion_cycles: u64,
    pipeline_q8_exposure_cycles: u64,
    pipeline_q16_completion_cycles: u64,
    pipeline_q16_exposure_cycles: u64,
    cohort_groups_observed: u64,
    cohort_copies_observed: u64,
    cohort_response_memberships_observed: u64,
    cohort_response_span_cycles: u64,
    cohort_topology_penalty_cycles: u64,
    cohort_candidate_completion_cycles: u64,
    cohort_candidate_exposure_cycles: u64,
    cohort_candidate_later_than_per_copy_cycles: u64,
    cohort_candidate_earlier_than_per_copy_cycles: u64,
    cohort_candidate_later_than_group_aggregate_cycles: u64,
    cohort_candidate_earlier_than_group_aggregate_cycles: u64,
    cohort_candidate_later_than_production_cycles: u64,
    cohort_candidate_earlier_than_production_cycles: u64,
    cohort_overlap_groups_observed: u64,
    cohort_response_envelope_cycles: u64,
    cohort_topology_prearrival_hidden_cycles: u64,
    cohort_topology_stream_hidden_cycles: u64,
    cohort_topology_residual_exposure_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassPipelineCopyCandidate {
    instruction: usize,
    payload_completion: u64,
    topology_penalty: u64,
}

const BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES: u64 = 4;

const BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES: u64 = 2;

const BYPASS_PIPELINE_CANDIDATE_Q: [u64; 5] = [1, 2, 4, 8, 16];

/// Independent RTX 3090 Ti clock64 dependent-MUFU probe endpoint. This is an
/// effective source-qualified latency for same-register SFU chains, not a
/// universal SFU latency or a claim about hidden pipeline stages.
const SM86_DEPENDENT_SFU_LATENCY_CYCLES: u64 = 30;

fn bypass_topology_candidate_penalty(
    bypass_wavefronts: u64,
    bank_wavefronts: u64,
) -> Result<u64, IssueError> {
    if bypass_wavefronts == 0 || bank_wavefronts == 0 {
        return Ok(0);
    }
    let wavefront_penalty = bypass_wavefronts
        .saturating_sub(1)
        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "bypass topology candidate wavefront penalty",
            lhs: bypass_wavefronts.saturating_sub(1),
            rhs: BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES,
        })?;
    let bank_penalty = bank_wavefronts
        .saturating_sub(1)
        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "bypass topology candidate bank penalty",
            lhs: bank_wavefronts.saturating_sub(1),
            rhs: BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES,
        })?;
    checked_cycle_add(
        "bypass topology candidate local penalty",
        wavefront_penalty,
        bank_penalty,
    )
}

fn bypass_pipeline_candidate_completion(
    copies: &[BypassPipelineCopyCandidate],
    q: u64,
) -> Result<u64, IssueError> {
    debug_assert!(q != 0, "bypass pipeline acceptance width must be nonzero");
    let mut next_accept = 0_u64;
    let mut group_completion = 0_u64;
    for copy in copies {
        let start = copy.payload_completion.max(next_accept);
        let finish = checked_cycle_add(
            "bypass pipeline candidate per-copy completion",
            start,
            copy.topology_penalty,
        )?;
        let accept_interval = copy.topology_penalty / q + u64::from(copy.topology_penalty % q != 0);
        next_accept = checked_cycle_add(
            "bypass pipeline candidate next acceptance",
            start,
            accept_interval,
        )?;
        group_completion = group_completion.max(finish);
    }
    Ok(group_completion)
}

fn bypass_pipeline_candidate_completions(
    copies: &mut [BypassPipelineCopyCandidate],
) -> Result<[u64; BYPASS_PIPELINE_CANDIDATE_Q.len()], IssueError> {
    copies.sort_by_key(|copy| (copy.payload_completion, copy.instruction));
    let mut completions = [0_u64; BYPASS_PIPELINE_CANDIDATE_Q.len()];
    for (completion, q) in completions.iter_mut().zip(BYPASS_PIPELINE_CANDIDATE_Q) {
        *completion = bypass_pipeline_candidate_completion(copies, q)?;
    }
    Ok(completions)
}

impl AsyncCopyGroup {
    const fn new(id: AsyncCopyGroupId) -> Self {
        Self {
            id,
            committed_at: 0,
            pending_sources: 0,
            source_completion_time: 0,
            delivery_work: 0,
            destination_wavefronts: 0,
            destination_service_queued: false,
            destination_completion_time: 0,
            completion_time: 0,
        }
    }

    const fn is_complete_at(self, now: u64) -> bool {
        self.pending_sources == 0
            && (self.destination_wavefronts == 0 || self.destination_service_queued)
            && self.completion_time <= now
    }

    fn project_delivery(
        self,
        source_ready: u64,
        work_add: u64,
        destination_wavefronts_add: u64,
        params: &IssueModelParams,
    ) -> Result<(u64, u64, u64, u64), IssueError> {
        let delivery_work = checked_u64_counter_add(
            "asynchronous-copy delivery work",
            self.delivery_work,
            work_add,
        )?;
        let source_completion_time = self.source_completion_time.max(source_ready);
        let destination_wavefronts = checked_u64_counter_add(
            "asynchronous-copy destination wavefront",
            self.destination_wavefronts,
            destination_wavefronts_add,
        )?;
        let local_completion_time =
            params.ldgsts_completion_ready(source_completion_time, delivery_work)?;
        let completion_time = local_completion_time.max(self.destination_completion_time);
        Ok((
            source_completion_time,
            delivery_work,
            destination_wavefronts,
            completion_time,
        ))
    }

    fn local_completion_time(self, params: &IssueModelParams) -> Result<u64, IssueError> {
        params.ldgsts_completion_ready(self.source_completion_time, self.delivery_work)
    }

    fn limiting_readiness_cause(
        self,
        params: &IssueModelParams,
    ) -> Result<ReadinessCause, IssueError> {
        let local = self.local_completion_time(params)?;
        Ok(if self.destination_completion_time > local {
            ReadinessCause::AsyncDestinationSmService
        } else if self.delivery_work != 0 || self.destination_wavefronts != 0 {
            ReadinessCause::AsyncDestinationLocal
        } else {
            ReadinessCause::AsyncSource
        })
    }
}

/// Allocated only for warps that issue an effective asynchronous-copy source.
/// Keeping this uncommon state out of every [`WarpState`] preserves the compact
/// scheduler scan used by workloads with no `LDGSTS` traffic.
#[derive(Debug)]
struct AsyncCopyState {
    /// Non-bulk `cp.async` copies issued since the most recent commit.
    open_group: Option<AsyncCopyGroup>,
    /// Nonempty groups closed by `LDGDEPBAR`, in warp program order.
    committed_groups: VecDeque<AsyncCopyGroup>,
    /// Bypass-only per-instruction source/delivery state. Entries share the
    /// owning group's opaque identity and are removed with the waited prefix.
    bypass_copies: Vec<BypassAsyncCopyObservation>,
    /// Next process-local sequence for this warp. `None` means the complete
    /// u64 domain has been allocated; the final `u64::MAX` identity remains
    /// valid, but another nonempty group fails closed.
    next_sequence: Option<u64>,
}

impl AsyncCopyState {
    const fn new() -> Self {
        Self {
            open_group: None,
            committed_groups: VecDeque::new(),
            bypass_copies: Vec::new(),
            next_sequence: Some(0),
        }
    }
}

/// Cold sidecar indexed by warp identity. Keeping even the nullable pointer out
/// of [`WarpState`] preserves that structure's pre-async stride for scheduler
/// scans while retaining direct, deterministic lookup for source responses.
#[derive(Debug, Default)]
struct AsyncCopySlot {
    state: Option<Box<AsyncCopyState>>,
}

struct WarpState<'a> {
    insts: &'a [ReplayInst],
    sectors: &'a [i64],
    async_copy_cohorts: Option<&'a AsyncCopyCohortReplay>,
    next: usize,
    /// Dynamic trace cursor already requested by the fetch scheduler and the
    /// three-entry instruction buffer (including in-flight fetches).
    front_end_fetch_cursor: usize,
    instruction_buffer: VecDeque<BufferedInstruction>,
    next_slot: Option<&'a IssueSlot>,
    /// Issue/admission timestamp at which `next_slot` became the current
    /// instruction. Causal readiness delays are measured from this frontier.
    current_since: u64,
    admission_time: u64,
    eligible: u64,
    /// Earliest cycle at which the current instruction may be re-issued after
    /// an immediate-constant-cache miss. Zero means no pending IMC request.
    imc_wait_until: u64,
    /// Compiler-emitted `DEPBAR.LE SB1..SB5, 0x0[, {..}]` operands for the
    /// current instruction. This is cached when `next_slot` changes so
    /// scheduler scans use the ordinary barrier counter/clear path without
    /// reparsing SASS.
    ordinary_dependency_wait_mask: WaitMask,
    barrier_clear: [u64; DEP_BARRIERS],
    barrier_origin: [DependencyOrigin; DEP_BARRIERS],
    /// Outstanding UNKNOWN write-backs per barrier index (L1Shared mode):
    /// hardware barriers are counters, and a response still in flight is
    /// an increment whose decrement time is unknown — a wait on the index
    /// cannot pass, and the warp has no computable ready time, until
    /// every pending unit resolves.
    barrier_pending: [u32; DEP_BARRIERS],
    /// Full replay's dynamic fixed-latency register recurrence. Native
    /// extraction currently carries no operand values in `ReplayInst`, so
    /// alpha uses the executing warp mask conservatively (all lanes for
    /// legacy captures); fast replay refines this with its compact masks.
    register_ready: Vec<RegisterReadyEntry>,
    block: usize,
    launch_index: i64,
    warp_id: i32,
    warp_slot: i32,
    /// Index into its sub-core's pick list while live (retired warps are
    /// swap-removed from the list; the swapped-in warp's position is
    /// fixed up). A warp is only ever picked from its own sub-core, so
    /// the retiring sub-core index is the loop's `sc`.
    pos_in_subcore: usize,
    arrival: u64,
    at_block_barrier: bool,
    /// CTA-release arm for the current post-barrier instruction. This is kept
    /// separate from compiler-control eligibility so attribution cannot confuse
    /// a CTA rendezvous with a SASS stall count.
    cta_barrier_ready: u64,
    /// Completion of this warp's own arrival through the SM-shared barrier
    /// service. The CTA arm above carries only other-participant rendezvous.
    barrier_pipeline_ready: u64,
    /// Effective dependent-SFU result frontier. Only the narrow static
    /// same-register MUFU/RCP/RSQ/EX2/LG2/SIN/COS shape updates this field.
    sfu_dependency_ready: u64,
    branch_pipeline_ready: u64,
    block_barrier_dispatch: Option<u64>,
    block_barrier_arrival: Option<u64>,
    clock_first_issue: Option<u64>,
    clock_reads_issued: u64,
    done: bool,
}

#[derive(Debug, Clone, Copy)]
struct RegisterReadyEntry {
    register: SassRegister,
    lane_mask: u32,
    ready_at: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct FixedRegisterReady {
    /// Data-register RAW readiness.
    raw: u64,
    /// Predicate-register readiness is tracked independently of data RAW.
    predicate: u64,
    /// Destination overlap is a WAW/partial-lane overwrite dependency.
    waw: u64,
}

impl FixedRegisterReady {
    fn latest(self) -> u64 {
        self.raw.max(self.predicate).max(self.waw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Sm86RfcKey {
    block: usize,
    warp_slot: i32,
    bank: u8,
    register: u16,
    operand: u8,
}

#[derive(Debug, Clone, Default)]
struct Sm86RegisterReadPlan {
    modeled: bool,
    next_port_ready: [u64; 2],
    reads: [u64; 2],
    port_conflict_cycles: u64,
    rfc_hits: u64,
    rfc_misses: u64,
    rfc_consumes: Vec<Sm86RfcKey>,
    rfc_inserts: Vec<Sm86RfcKey>,
    unsupported_shape: bool,
}

fn sm86_register_read_plan(
    status: &FixedLatencyOperandStatus,
    block: usize,
    warp_slot: i32,
    now: u64,
    port_ready: [u64; 2],
    rfc: &VecDeque<Sm86RfcKey>,
) -> Result<Sm86RegisterReadPlan, IssueError> {
    let projection = match status {
        FixedLatencyOperandStatus::Projected(projection) => projection,
        FixedLatencyOperandStatus::NotApplicable => {
            return Ok(Sm86RegisterReadPlan::default());
        }
        FixedLatencyOperandStatus::Gap(_) => {
            return Ok(Sm86RegisterReadPlan {
                unsupported_shape: true,
                ..Sm86RegisterReadPlan::default()
            });
        }
    };
    let mut plan = Sm86RegisterReadPlan {
        modeled: true,
        next_port_ready: port_ready,
        ..Sm86RegisterReadPlan::default()
    };
    for (operand_ix, source) in projection.sources().iter().enumerate() {
        let SassSourceOperand::Register(source) = source else {
            continue;
        };
        let register = source.register();
        if register.file() != SassRegisterFile::Regular || register.index().is_none() {
            continue;
        }
        // The first arm deliberately refuses vector-register RFC/port
        // attribution rather than inventing a multi-cell slot mapping.
        if register.span_registers() != 1 {
            plan.unsupported_shape = true;
            continue;
        }
        let bank = register
            .sm86_regular_bank()
            .ok_or(IssueError::CounterOverflow {
                counter: "SM86 register-read bank",
            })?;
        let key = Sm86RfcKey {
            block,
            warp_slot,
            bank,
            register: register.index().ok_or(IssueError::CounterOverflow {
                counter: "SM86 register-read index",
            })?,
            operand: u8::try_from(operand_ix).map_err(|_| IssueError::CounterOverflow {
                counter: "SM86 register-read operand position",
            })?,
        };
        if source.reused() {
            if rfc.contains(&key) {
                plan.rfc_hits =
                    plan.rfc_hits
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "SM86 register-file reuse hits",
                        })?;
                plan.rfc_consumes.push(key);
                continue;
            }
            plan.rfc_misses =
                plan.rfc_misses
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "SM86 register-file reuse misses",
                    })?;
            // A miss still reads the regular register and seeds the compiler
            // managed entry for the next same-key use.
            plan.rfc_inserts.push(key);
        }
        let bank_index = usize::from(bank);
        plan.reads[bank_index] =
            plan.reads[bank_index]
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "SM86 register-file source reads",
                })?;
    }
    if plan.reads.iter().any(|&reads| reads > 3) {
        plan.unsupported_shape = true;
        return Ok(plan);
    }
    let first_read = now.checked_add(1).ok_or(IssueError::TimeOverflow {
        operation: "SM86 register-read port window",
        base: now,
        delta: 1,
    })?;
    for bank in 0..2 {
        if plan.reads[bank] == 0 {
            continue;
        }
        let start = first_read.max(plan.next_port_ready[bank]);
        plan.port_conflict_cycles = plan.port_conflict_cycles.max(start - first_read);
        plan.next_port_ready[bank] =
            start
                .checked_add(plan.reads[bank])
                .ok_or(IssueError::TimeOverflow {
                    operation: "SM86 register-read port reservation",
                    base: start,
                    delta: plan.reads[bank],
                })?;
    }
    Ok(plan)
}

fn issue_register_overlap(left: SassRegister, right: SassRegister) -> bool {
    if left.file() != right.file() {
        return false;
    }
    let (Some(left_index), Some(right_index)) = (left.index(), right.index()) else {
        return false;
    };
    u32::from(left_index) < u32::from(right_index) + u32::from(right.span_registers())
        && u32::from(right_index) < u32::from(left_index) + u32::from(left.span_registers())
}

fn issue_register_lane_mask(register: SassRegister, mask: u32) -> u32 {
    match register.file() {
        SassRegisterFile::Uniform | SassRegisterFile::UniformPredicate if mask != 0 => u32::MAX,
        _ => mask,
    }
}

fn issue_fixed_register_ready(
    status: &FixedLatencyOperandStatus,
    mask: u32,
    active_mask: u32,
    guard: Option<SassRegister>,
    entries: &[RegisterReadyEntry],
) -> FixedRegisterReady {
    let mut ready = FixedRegisterReady::default();
    if let Some(register) = guard {
        let lanes = issue_register_lane_mask(register, active_mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.predicate = ready.predicate.max(entry.ready_at);
            }
        }
    }
    let FixedLatencyOperandStatus::Projected(projection) = status else {
        return ready;
    };
    for source in projection.sources() {
        let SassSourceOperand::Register(operand) = source else {
            continue;
        };
        let register = operand.register();
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.raw = ready.raw.max(entry.ready_at);
            }
        }
    }
    for operand in projection.predicate_sources() {
        let register = operand.register();
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.predicate = ready.predicate.max(entry.ready_at);
            }
        }
    }
    for operand in projection.destinations() {
        let register = operand.register();
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.waw = ready.waw.max(entry.ready_at);
            }
        }
    }
    ready
}

fn issue_record_fixed_register_writes(
    entries: &mut Vec<RegisterReadyEntry>,
    status: &FixedLatencyOperandStatus,
    mask: u32,
    ready_at: u64,
) {
    let FixedLatencyOperandStatus::Projected(projection) = status else {
        return;
    };
    for register in projection
        .destinations()
        .iter()
        .flat_map(|dst| dst.register().components())
    {
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries
            .iter_mut()
            .filter(|entry| issue_register_overlap(register, entry.register))
        {
            entry.lane_mask &= !lanes;
        }
        entries.retain(|entry| entry.lane_mask != 0);
        if lanes != 0 {
            entries.push(RegisterReadyEntry {
                register,
                lane_mask: lanes,
                ready_at,
            });
        }
    }
}

struct BlockState {
    unfinished: usize,
    waiting: usize,
    warp_count: u32,
    shared_memory_bytes: u64,
    /// Physical slots held while this block is resident. The trace does not carry
    /// CTA lifetime, so modeled overlap may admit another block only when its slot
    /// set is disjoint from every currently resident block.
    warp_slots: Box<[i32]>,
    /// This block's warps occupy `warps[first_warp..first_warp +
    /// admitted_warps]` (admission pushes them contiguously), so barrier
    /// release scans only the block's own range.
    first_warp: usize,
    admitted_warps: usize,
    barrier_generation: u64,
    barrier_first_dispatch: Option<u64>,
    barrier_first_arrival: Option<u64>,
    /// Earliest issue time of this CTA's next barrier generation. It gates
    /// barrier instructions only; ordinary work after rendezvous is unaffected.
    barrier_rearm_ready: u64,
}

impl WarpState<'_> {
    fn can_fetch_instruction(&self, capacity: usize) -> bool {
        !self.done
            && self.front_end_fetch_cursor < self.insts.len()
            && self.instruction_buffer.len() < capacity
    }

    fn instruction_front_end_ready_at(&self) -> Option<u64> {
        let buffered = self.instruction_buffer.front()?;
        (buffered.instruction == self.next).then_some(buffered.ready_at)
    }

    fn preflight_buffered_issue(&self, warp: usize) -> Result<(), IssueError> {
        let Some(buffered) = self.instruction_buffer.front() else {
            return Err(IssueError::InstructionBufferMissing {
                warp,
                instruction: self.next,
            });
        };
        if buffered.instruction != self.next {
            return Err(IssueError::InstructionBufferOrder {
                warp,
                expected: self.next,
                got: buffered.instruction,
            });
        }
        Ok(())
    }

    fn consume_buffered_issue(&mut self, warp: usize) -> Result<(), IssueError> {
        self.preflight_buffered_issue(warp)?;
        self.instruction_buffer.pop_front();
        Ok(())
    }

    fn next_issue_retires(&self) -> Result<bool, IssueError> {
        let next = checked_usize_counter_add("warp instruction cursor", self.next, 1)?;
        Ok(next == self.insts.len())
    }
}

fn invalid_cohort_projection(
    warp: usize,
    instruction: usize,
    field: &'static str,
    expected: u64,
    got: u64,
) -> IssueError {
    IssueError::AsyncCopyCohortProjectionInvalid {
        warp,
        instruction,
        field,
        expected,
        got,
    }
}

fn build_bypass_cohort_observation(
    warp: usize,
    pending_add: u32,
    delivery_work: u64,
    input: BypassAsyncCopyInput<'_>,
    params: &IssueModelParams,
) -> Result<(Box<[u32]>, Box<[BypassAsyncCopyCohortObservation]>), IssueError> {
    let (Some(descriptors), Some(masks)) = (input.cohorts, input.sector_membership_masks) else {
        if input.cohorts.is_some() != input.sector_membership_masks.is_some() {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "paired cohort descriptor/membership presence",
                1,
                0,
            ));
        }
        return Ok((Box::new([]), Box::new([])));
    };
    let sector_len = usize::from(input.sector_len);
    if masks.len() != sector_len {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "source-sector membership count",
            u64::from(input.sector_len),
            u64::try_from(masks.len()).unwrap_or(u64::MAX),
        ));
    }
    if u32::from(input.sector_len) != pending_add {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "pending source-sector count",
            u64::from(input.sector_len),
            u64::from(pending_add),
        ));
    }
    if descriptors.is_empty() || descriptors.len() > u32::BITS as usize {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "cohort count in u32 membership domain",
            input.source_cohorts,
            u64::try_from(descriptors.len()).unwrap_or(u64::MAX),
        ));
    }
    let descriptor_count = u64::try_from(descriptors.len()).unwrap_or(u64::MAX);
    if descriptor_count != input.source_cohorts {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "source cohort count",
            input.source_cohorts,
            descriptor_count,
        ));
    }

    let allowed_mask = if descriptors.len() == u32::BITS as usize {
        u32::MAX
    } else {
        (1_u32 << descriptors.len()) - 1
    };
    let mut pending = vec![0_u32; descriptors.len()];
    for &mask in masks {
        if mask == 0 || mask & !allowed_mask != 0 {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "source-sector cohort mask",
                u64::from(allowed_mask),
                u64::from(mask),
            ));
        }
        for (cohort, count) in pending.iter_mut().enumerate() {
            if mask & (1_u32 << cohort) != 0 {
                *count = count.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "bypass cohort pending source membership",
                })?;
            }
        }
    }

    let common_rate = params.ldgsts_delivery_units_per_cycle();
    let bypass_rate = params
        .ldgsts_completion_throughput
        .bypass_payload_bytes_per_cycle;
    debug_assert!(bypass_rate != 0 && common_rate.is_multiple_of(bypass_rate));
    let work_per_byte = common_rate / bypass_rate;
    let mut payload_work_sum = 0_u64;
    let mut destination_lines_sum = 0_u64;
    let mut max_bank_wavefronts = 0_u64;
    let mut max_bypass_wavefronts = 0_u64;
    let mut cohorts = Vec::with_capacity(descriptors.len());
    for (&descriptor, pending_sources) in descriptors.iter().zip(pending) {
        if pending_sources == 0 {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "cohort source-sector membership",
                1,
                0,
            ));
        }
        if descriptor.payload_bytes == 0
            || descriptor.destination_lines == 0
            || descriptor.bank_wavefronts == 0
            || descriptor.bypass_wavefronts == 0
        {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "nonzero active cohort descriptor",
                1,
                0,
            ));
        }
        let payload_work = u64::from(descriptor.payload_bytes)
            .checked_mul(work_per_byte)
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "bypass cohort payload work",
                lhs: u64::from(descriptor.payload_bytes),
                rhs: work_per_byte,
            })?;
        payload_work_sum = checked_u64_counter_add(
            "bypass cohort payload work sum",
            payload_work_sum,
            payload_work,
        )?;
        destination_lines_sum = checked_u64_counter_add(
            "bypass cohort destination line sum",
            destination_lines_sum,
            u64::from(descriptor.destination_lines),
        )?;
        max_bank_wavefronts = max_bank_wavefronts.max(u64::from(descriptor.bank_wavefronts));
        max_bypass_wavefronts = max_bypass_wavefronts.max(u64::from(descriptor.bypass_wavefronts));
        cohorts.push(BypassAsyncCopyCohortObservation {
            pending_sources,
            first_source_response: None,
            last_source_response: None,
            payload_work,
            destination_lines: u64::from(descriptor.destination_lines),
            bank_wavefronts: u64::from(descriptor.bank_wavefronts),
            bypass_wavefronts: u64::from(descriptor.bypass_wavefronts),
            topology_penalty: bypass_topology_candidate_penalty(
                u64::from(descriptor.bypass_wavefronts),
                u64::from(descriptor.bank_wavefronts),
            )?,
        });
    }
    if delivery_work != payload_work_sum {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "exact cohort payload work",
            delivery_work,
            payload_work_sum,
        ));
    }
    // The compact aggregate also includes executing source-ignored (zero-fill)
    // destination lanes.  Those lanes deliberately have no global source
    // response and therefore do not appear in the sparse source-cohort sidecar.
    // Require the detailed source-bearing projection to fit inside the compact
    // aggregate instead of requiring equality that would reject valid mixed
    // source/zero-fill instructions.  The residual remains topology-only and
    // is not converted into a completion timestamp.
    for (field, aggregate, source_only) in [
        (
            "source cohort destination line sum exceeds aggregate",
            input.destination_lines,
            destination_lines_sum,
        ),
        (
            "source cohort maximum bank wavefronts exceeds aggregate",
            input.max_bank_wavefronts,
            max_bank_wavefronts,
        ),
        (
            "source cohort maximum bypass wavefronts exceeds aggregate",
            input.max_bypass_wavefronts,
            max_bypass_wavefronts,
        ),
    ] {
        if source_only > aggregate {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                field,
                aggregate,
                source_only,
            ));
        }
    }
    Ok((masks.into(), cohorts.into_boxed_slice()))
}

