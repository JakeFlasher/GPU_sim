//! Carded DRAM bandwidth and service state; mutates DRAM event timing.

use super::*;

/// Aggregate DRAM read-bandwidth model (the §6.3 "DRAM
/// bandwidth-latency" increment). The configured carded rate is a model
/// covariate: the corpus runner defaults to the independently measured
/// 825 GB/s effective-rate hypothesis, while direct `sim-run` without an
/// override falls back to the architecture-card peak. Neither value is a
/// physical controller claim; the model is one G/D/1 byte server over the shared hierarchy
/// read-miss sector stream: each fetching sector demands 32 B, and a
/// reply-carrying request or intrinsically returning ATOM cannot complete
/// before BOTH its latency arm and the server's completion of ITS OWN request's
/// bytes. Late L1 merges carry their own sector response, so the waiting warp
/// clears at that child sector's ready event. Uncongested streams see pure
/// latency (the server only ever ADDS delay).
///
/// Default `Carded` scope remains DRAM READS only — write data validates Store
/// sectors without a fetch, while returning ATOM and non-returning RED charge
/// missing sectors for their internal RMW read. The opt-in `CardedWriteback`
/// variant adds a bounded dirty-sector service lane; coherence ownership and
/// visibility, and RED's value-dependent serialization, remain typed tails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DramBw {
    /// Latency-only (pre-increment behavior): the ablation mode and the
    /// event-vs-const equivalence-differential mode — the constant path
    /// has no bandwidth concept, so equivalence is proven here.
    Unmodeled { arch: ArchId },
    /// Configured carded aggregate service rate; runner/default provenance is
    /// carried by the direct-run configuration rather than inferred here.
    Carded {
        arch: ArchId,
        bytes_per_sec: u64,
        sm_clock_hz: u64,
    },
    /// Opt-in carded service with transactional dirty-sector writeback.
    /// Capacity is measured in 32-byte dirty sectors; this remains a bounded
    /// native replay experiment, not a coherence/ownership protocol.
    CardedWriteback {
        arch: ArchId,
        bytes_per_sec: u64,
        sm_clock_hz: u64,
        capacity_sectors: u32,
    },
}

impl DramBw {
    pub const fn arch(self) -> ArchId {
        match self {
            Self::Unmodeled { arch }
            | Self::Carded { arch, .. }
            | Self::CardedWriteback { arch, .. } => arch,
        }
    }

    pub const fn writeback_capacity_sectors(self) -> Option<u32> {
        match self {
            Self::CardedWriteback {
                capacity_sectors, ..
            } => Some(capacity_sectors),
            _ => None,
        }
    }

    pub const fn service_rate_bytes_per_sec(self) -> Option<u64> {
        match self {
            Self::Carded { bytes_per_sec, .. } | Self::CardedWriteback { bytes_per_sec, .. } => {
                Some(bytes_per_sec)
            }
            Self::Unmodeled { .. } => None,
        }
    }
}

/// The G/D/1 byte server. Fixed-point time in units of
/// `cycle × bytes_per_sec` (u128): serving `b` bytes takes
/// `b × sm_clock_hz` fp-units (= b × Hz / Bps cycles), so no division
/// happens until the final ceil back to cycles — bit-exact and
/// deterministic in queue order.
pub(super) struct DramServer {
    pub(super) free_fp: u128,
    pub(super) bytes_per_sec: u128,
    pub(super) sm_clock_hz: u128,
}

impl DramServer {
    #[cfg(feature = "reference")]
    pub(super) fn typed_boundary_state(&self) -> Result<DramBoundaryState, DramStateError> {
        DramBoundaryState::from_fixed_point(
            u64::try_from(self.bytes_per_sec).map_err(|_| DramStateError::InvalidRate)?,
            u64::try_from(self.sm_clock_hz).map_err(|_| DramStateError::InvalidRate)?,
            self.free_fp,
        )
    }

    /// The caller (`simulate_kernel_shared`) rejects zero rates with a
    /// typed error before construction — no clamping here (a clamp
    /// would turn a bad card into a silently absurd 1 B/s server and
    /// arm the fixed-point u64 ceiling on large traces; gate finding
    /// 2026-07-07).
    pub(super) fn new(bw: DramBw) -> Option<Self> {
        match bw {
            DramBw::Unmodeled { .. } => None,
            DramBw::Carded {
                bytes_per_sec,
                sm_clock_hz,
                ..
            } => Some(Self {
                free_fp: 0,
                bytes_per_sec: u128::from(bytes_per_sec),
                sm_clock_hz: u128::from(sm_clock_hz),
            }),
            // The opt-in writeback path owns the shared byte server through
            // `WritebackQueue`; no independent read server is created here.
            DramBw::CardedWriteback { .. } => None,
        }
    }

    /// Compute a service completion without mutating the server. The
    /// caller uses this to preflight the request's exact outcome-dependent
    /// fetch before any L2 or accounting state commits.
    pub(super) fn preview(&self, arrival: u64, bytes: u64) -> Result<(u128, u64), SystemError> {
        let arrival_fp = u128::from(arrival).checked_mul(self.bytes_per_sec).ok_or(
            SystemError::DramFixedPointOverflow {
                operation: "arrival scaling",
            },
        )?;
        let service_fp = u128::from(bytes).checked_mul(self.sm_clock_hz).ok_or(
            SystemError::DramFixedPointOverflow {
                operation: "service scaling",
            },
        )?;
        let start = arrival_fp.max(self.free_fp);
        let done = start
            .checked_add(service_fp)
            .ok_or(SystemError::DramFixedPointOverflow {
                operation: "service completion",
            })?;
        let ready_fp = done.div_ceil(self.bytes_per_sec);
        let ready = u64::try_from(ready_fp).map_err(|_| SystemError::DramCycleOverflow {
            completion_fp: done,
            bytes_per_sec: self.bytes_per_sec,
        })?;
        Ok((done, ready))
    }
}

/// Driver-side DRAM traffic and timing-neutral lower-level topology observer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DramStats {
    /// Architecture provenance bound by the shared-L2 system. Unbound
    /// synthetic stats are rejected before aggregation or emission.
    pub(super) arch: Option<ArchId>,
    pub service_rate_bytes_per_sec: Option<u64>,
    /// 32 B DRAM read fetches DEMANDED by the run (L2-missing sectors
    /// of read operations plus missing sectors of ATOM and RED internal RMW reads).
    /// Counted in BOTH modes: under `Unmodeled` it audits the ablation's demand
    /// with no server attached. Plain stores never enter this modeled
    /// read-fetch count.
    pub read_sectors: u64,
    /// Missing-sector read demand attributable to returning ATOM operations.
    /// This is a traffic observer, not a claim that the value-dependent RMW
    /// serialization is complete.
    pub atomic_rmw_read_sectors: u64,
    /// Missing-sector read demand attributable to non-returning RED/REDG
    /// operations. The reduction result/coherence protocol remains a typed
    /// tail even though its cold-line read arm is charged here.
    pub reduction_rmw_read_sectors: u64,
    /// Responses whose ready time the bandwidth server pushed PAST the
    /// latency arm (0 in uncongested runs and in `Unmodeled` mode).
    /// Late sector follow-ups carry their own completion response and are
    /// therefore included in the same latency-arm accounting as any other
    /// reply-bearing read.
    pub bw_delayed_responses: u64,
    /// L2 replacement topology. In the default read-only path dirty entries
    /// are candidate writeback work; the opt-in writeback variant also reports
    /// separate completion counters below.
    pub l2_evicted_lines: u64,
    pub l2_evicted_valid_sectors: u64,
    pub l2_evicted_pending_lines: u64,
    pub l2_evicted_pending_sectors: u64,
    pub l2_dirty_eviction_lines: u64,
    pub l2_dirty_evicted_sectors: u64,
    /// Opt-in scheduled writeback completion observations, including eviction
    /// victims and the terminal resident-L2 flush in CardedWriteback mode.
    pub writeback_completed_lines: u64,
    pub writeback_completed_sectors: u64,
    pub writeback_completed_bytes: u64,
    /// Subset of the writeback completion stream sourced from resident L1
    /// dirty sectors; L2 eviction/resident flush counters remain separate.
    pub l1_writeback_completed_lines: u64,
    pub l1_writeback_completed_sectors: u64,
    pub l1_writeback_completed_bytes: u64,
    /// Logical payload-forwarding records validated at terminal L1 flush.
    /// These counters prove model-local byte conservation only; they are not
    /// physical DRAM acknowledgements.
    pub l1_payload_forwarded_sectors: u64,
    pub l1_payload_forwarded_bytes: u64,
    pub l1_payload_forwarding_digest_xor: u64,
    pub writeback_completion_frontier: Option<u64>,
    /// Opt-in directory-service completion observations, separate from
    /// writeback and read-traffic counters.
    pub coherence_completed_transactions: u64,
    pub coherence_completed_sectors: u64,
    pub coherence_completion_frontier: Option<u64>,
    /// Maximum terminal dirty residency across the launches aggregated into
    /// this observer. A gauge is never summed as traffic.
    pub maximum_terminal_l2_dirty_lines: u64,
    pub maximum_terminal_l2_dirty_sectors: u64,
    pub maximum_terminal_l1_dirty_lines: u64,
    pub maximum_terminal_l1_dirty_sectors: u64,
    /// Observer-only effective partition/slice pressure candidates. The
    /// mapping uses line-id modulo/XOR hypotheses; it is not physical GA102
    /// swizzling and never feeds the byte server.
    pub dram_partition_modulo_sectors: [u64; 12],
    pub dram_partition_xor_sectors: [u64; 12],
    pub dram_slice_modulo_sectors: [u64; 6],
    /// Timing-neutral per-request frontier observers. Sums are model-domain
    /// intervals, not hardware completion timestamps or queue claims.
    pub l2_frontier_samples: u64,
    pub l2_issue_to_fill_cycles_sum: u64,
    pub l2_issue_to_fill_cycles_max: u64,
    pub l2_issue_to_response_cycles_sum: u64,
    pub l2_issue_to_response_cycles_max: u64,
    pub l2_service_delay_cycles_sum: u64,
    pub l2_service_delay_cycles_max: u64,
    pub(super) by_origin: [DramOriginStats; 2],
}

impl DramStats {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn with_service_rate(mut self, rate: Option<u64>) -> Self {
        self.service_rate_bytes_per_sec = rate;
        self
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn origin(&self, origin: MemoryOrigin) -> DramOriginStats {
        self.by_origin[origin.index()]
    }

    pub(super) fn with_l2_eviction(mut self, eviction: L2Eviction) -> Result<Self, SystemError> {
        macro_rules! add {
            ($field:ident, $value:expr, $name:literal) => {
                self.$field = self
                    .$field
                    .checked_add($value)
                    .ok_or(SystemError::CounterOverflow { counter: $name })?;
            };
        }
        add!(l2_evicted_lines, 1, "L2 evicted lines");
        add!(
            l2_evicted_valid_sectors,
            u64::from(eviction.valid_mask.count_ones()),
            "L2 evicted valid sectors"
        );
        if eviction.pending_mask != 0 {
            add!(l2_evicted_pending_lines, 1, "L2 evicted pending lines");
            add!(
                l2_evicted_pending_sectors,
                u64::from(eviction.pending_mask.count_ones()),
                "L2 evicted pending sectors"
            );
        }
        if eviction.dirty_mask != 0 {
            add!(l2_dirty_eviction_lines, 1, "L2 dirty eviction lines");
            add!(
                l2_dirty_evicted_sectors,
                u64::from(eviction.dirty_mask.count_ones()),
                "L2 dirty evicted sectors"
            );
        }
        Ok(self)
    }

    pub(super) fn with_terminal_l2_dirty_residency(
        mut self,
        residency: L2DirtyResidency,
    ) -> Result<Self, SystemError> {
        let lines = u64::try_from(residency.lines).map_err(|_| SystemError::CounterOverflow {
            counter: "terminal L2 dirty resident lines",
        })?;
        let sectors =
            u64::try_from(residency.sectors).map_err(|_| SystemError::CounterOverflow {
                counter: "terminal L2 dirty resident sectors",
            })?;
        self.maximum_terminal_l2_dirty_lines = self.maximum_terminal_l2_dirty_lines.max(lines);
        self.maximum_terminal_l2_dirty_sectors =
            self.maximum_terminal_l2_dirty_sectors.max(sectors);
        Ok(self)
    }

    pub(super) fn with_terminal_l1_dirty_residency(
        mut self,
        residency: sim_mem::L1DirtyResidency,
    ) -> Result<Self, SystemError> {
        self.maximum_terminal_l1_dirty_lines =
            self.maximum_terminal_l1_dirty_lines.max(residency.lines);
        self.maximum_terminal_l1_dirty_sectors = self
            .maximum_terminal_l1_dirty_sectors
            .max(residency.sectors);
        Ok(self)
    }

    pub(super) fn with_partition_candidate(
        mut self,
        line: i64,
        sectors: u64,
    ) -> Result<Self, SystemError> {
        let line = u64::try_from(line).unwrap_or_default();
        let modulo = usize::try_from(line % 12).unwrap_or(0);
        let xor = usize::try_from((line ^ (line >> 7)) % 12).unwrap_or(0);
        let slice = usize::try_from(line % 6).unwrap_or(0);
        self.dram_partition_modulo_sectors[modulo] = self.dram_partition_modulo_sectors[modulo]
            .checked_add(sectors)
            .ok_or(SystemError::CounterOverflow {
                counter: "DRAM modulo partition candidate sectors",
            })?;
        self.dram_partition_xor_sectors[xor] = self.dram_partition_xor_sectors[xor]
            .checked_add(sectors)
            .ok_or(SystemError::CounterOverflow {
                counter: "DRAM XOR partition candidate sectors",
            })?;
        self.dram_slice_modulo_sectors[slice] = self.dram_slice_modulo_sectors[slice]
            .checked_add(sectors)
            .ok_or(SystemError::CounterOverflow {
                counter: "DRAM slice candidate sectors",
            })?;
        Ok(self)
    }

    /// Build a coherent synthetic observer from explicit origin lanes.
    pub fn from_origins(
        arch: ArchId,
        global: DramOriginStats,
        local: DramOriginStats,
    ) -> Result<Self, SystemCounterError> {
        let read_sectors = global.read_sectors.checked_add(local.read_sectors).ok_or(
            SystemCounterError::DramOriginPartitionOverflow {
                counter: "read sectors",
                global: global.read_sectors,
                local: local.read_sectors,
            },
        )?;
        let bw_delayed_responses = global
            .bw_delayed_responses
            .checked_add(local.bw_delayed_responses)
            .ok_or(SystemCounterError::DramOriginPartitionOverflow {
                counter: "bandwidth-delayed responses",
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            })?;
        Ok(Self {
            arch: Some(arch),
            service_rate_bytes_per_sec: None,
            read_sectors,
            atomic_rmw_read_sectors: 0,
            reduction_rmw_read_sectors: 0,
            bw_delayed_responses,
            l2_evicted_lines: 0,
            l2_evicted_valid_sectors: 0,
            l2_evicted_pending_lines: 0,
            l2_evicted_pending_sectors: 0,
            l2_dirty_eviction_lines: 0,
            l2_dirty_evicted_sectors: 0,
            writeback_completed_lines: 0,
            writeback_completed_sectors: 0,
            writeback_completed_bytes: 0,
            l1_writeback_completed_lines: 0,
            l1_writeback_completed_sectors: 0,
            l1_writeback_completed_bytes: 0,
            l1_payload_forwarded_sectors: 0,
            l1_payload_forwarded_bytes: 0,
            l1_payload_forwarding_digest_xor: 0,
            writeback_completion_frontier: None,
            coherence_completed_transactions: 0,
            coherence_completed_sectors: 0,
            coherence_completion_frontier: None,
            maximum_terminal_l2_dirty_lines: 0,
            maximum_terminal_l2_dirty_sectors: 0,
            maximum_terminal_l1_dirty_lines: 0,
            maximum_terminal_l1_dirty_sectors: 0,
            dram_partition_modulo_sectors: [0; 12],
            dram_partition_xor_sectors: [0; 12],
            dram_slice_modulo_sectors: [0; 6],
            l2_frontier_samples: 0,
            l2_issue_to_fill_cycles_sum: 0,
            l2_issue_to_fill_cycles_max: 0,
            l2_issue_to_response_cycles_sum: 0,
            l2_issue_to_response_cycles_max: 0,
            l2_service_delay_cycles_sum: 0,
            l2_service_delay_cycles_max: 0,
            by_origin: [global, local],
        })
    }

    /// Explicitly attribute an already-built unified synthetic observer to one
    /// origin. Modeled requests always mutate an explicit lane directly.
    pub fn with_unified_origin(mut self, origin: MemoryOrigin) -> Self {
        let unified = DramOriginStats {
            read_sectors: self.read_sectors,
            bw_delayed_responses: self.bw_delayed_responses,
        };
        self.by_origin = [DramOriginStats::default(); 2];
        self.by_origin[origin.index()] = unified;
        self
    }

    pub fn validate_origin_partition(&self) -> Result<(), SystemCounterError> {
        let global = self.origin(MemoryOrigin::Global);
        let local = self.origin(MemoryOrigin::Local);
        let read_sectors = global.read_sectors.checked_add(local.read_sectors).ok_or(
            SystemCounterError::DramOriginPartitionOverflow {
                counter: "read sectors",
                global: global.read_sectors,
                local: local.read_sectors,
            },
        )?;
        if self.read_sectors != read_sectors {
            return Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "read sectors",
                unified: self.read_sectors,
                global: global.read_sectors,
                local: local.read_sectors,
            });
        }
        let bw_delayed_responses = global
            .bw_delayed_responses
            .checked_add(local.bw_delayed_responses)
            .ok_or(SystemCounterError::DramOriginPartitionOverflow {
                counter: "bandwidth-delayed responses",
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            })?;
        if self.bw_delayed_responses != bw_delayed_responses {
            return Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "bandwidth-delayed responses",
                unified: self.bw_delayed_responses,
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            });
        }
        Ok(())
    }

    pub fn add_checked(&mut self, other: &DramStats) -> Result<(), SystemCounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "DRAM stats",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "DRAM stats",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "DRAM stats",
                expected,
                got,
            }
            .into());
        }
        let read_sectors = self.read_sectors.checked_add(other.read_sectors).ok_or(
            SystemCounterError::Overflow {
                counter: "DRAM read sectors",
            },
        )?;
        let atomic_rmw_read_sectors = self
            .atomic_rmw_read_sectors
            .checked_add(other.atomic_rmw_read_sectors)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM atomic RMW read sectors",
            })?;
        let reduction_rmw_read_sectors = self
            .reduction_rmw_read_sectors
            .checked_add(other.reduction_rmw_read_sectors)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM reduction RMW read sectors",
            })?;
        let bw_delayed_responses = self
            .bw_delayed_responses
            .checked_add(other.bw_delayed_responses)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM bandwidth-delayed responses",
            })?;
        macro_rules! add_topology {
            ($field:ident, $name:literal) => {
                self.$field
                    .checked_add(other.$field)
                    .ok_or(SystemCounterError::Overflow { counter: $name })?
            };
        }
        let l2_evicted_lines = add_topology!(l2_evicted_lines, "L2 evicted lines");
        let l2_evicted_valid_sectors =
            add_topology!(l2_evicted_valid_sectors, "L2 evicted valid sectors");
        let l2_evicted_pending_lines =
            add_topology!(l2_evicted_pending_lines, "L2 evicted pending lines");
        let l2_evicted_pending_sectors =
            add_topology!(l2_evicted_pending_sectors, "L2 evicted pending sectors");
        let l2_dirty_eviction_lines =
            add_topology!(l2_dirty_eviction_lines, "L2 dirty eviction lines");
        let l2_dirty_evicted_sectors =
            add_topology!(l2_dirty_evicted_sectors, "L2 dirty evicted sectors");
        let mut dram_partition_modulo_sectors = [0; 12];
        let mut dram_partition_xor_sectors = [0; 12];
        let mut dram_slice_modulo_sectors = [0; 6];
        for index in 0..12 {
            dram_partition_modulo_sectors[index] = self.dram_partition_modulo_sectors[index]
                .checked_add(other.dram_partition_modulo_sectors[index])
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM modulo partition candidate sectors",
                })?;
            dram_partition_xor_sectors[index] = self.dram_partition_xor_sectors[index]
                .checked_add(other.dram_partition_xor_sectors[index])
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM XOR partition candidate sectors",
                })?;
        }
        for index in 0..6 {
            dram_slice_modulo_sectors[index] = self.dram_slice_modulo_sectors[index]
                .checked_add(other.dram_slice_modulo_sectors[index])
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM slice candidate sectors",
                })?;
        }
        macro_rules! add_frontier {
            ($field:ident, $name:literal) => {
                let $field = self
                    .$field
                    .checked_add(other.$field)
                    .ok_or(SystemCounterError::Overflow { counter: $name })?;
            };
        }
        add_frontier!(l2_frontier_samples, "L2 frontier samples");
        add_frontier!(l2_issue_to_fill_cycles_sum, "L2 issue-to-fill cycles");
        let l2_issue_to_fill_cycles_max = self
            .l2_issue_to_fill_cycles_max
            .max(other.l2_issue_to_fill_cycles_max);
        add_frontier!(
            l2_issue_to_response_cycles_sum,
            "L2 issue-to-response cycles"
        );
        let l2_issue_to_response_cycles_max = self
            .l2_issue_to_response_cycles_max
            .max(other.l2_issue_to_response_cycles_max);
        add_frontier!(l2_service_delay_cycles_sum, "L2 service-delay cycles");
        let l2_service_delay_cycles_max = self
            .l2_service_delay_cycles_max
            .max(other.l2_service_delay_cycles_max);
        let mut by_origin = self.by_origin;
        for origin in MemoryOrigin::ALL {
            let origin_ix = origin.index();
            by_origin[origin_ix].read_sectors = self.by_origin[origin_ix]
                .read_sectors
                .checked_add(other.by_origin[origin_ix].read_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM origin read sectors",
                })?;
            by_origin[origin_ix].bw_delayed_responses = self.by_origin[origin_ix]
                .bw_delayed_responses
                .checked_add(other.by_origin[origin_ix].bw_delayed_responses)
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM origin bandwidth-delayed responses",
                })?;
        }
        let next = Self {
            arch: self.arch,
            service_rate_bytes_per_sec: self
                .service_rate_bytes_per_sec
                .or(other.service_rate_bytes_per_sec),
            read_sectors,
            atomic_rmw_read_sectors,
            reduction_rmw_read_sectors,
            bw_delayed_responses,
            l2_evicted_lines,
            l2_evicted_valid_sectors,
            l2_evicted_pending_lines,
            l2_evicted_pending_sectors,
            l2_dirty_eviction_lines,
            l2_dirty_evicted_sectors,
            writeback_completed_lines: self
                .writeback_completed_lines
                .checked_add(other.writeback_completed_lines)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L2 writeback lines",
                })?,
            writeback_completed_sectors: self
                .writeback_completed_sectors
                .checked_add(other.writeback_completed_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L2 writeback sectors",
                })?,
            writeback_completed_bytes: self
                .writeback_completed_bytes
                .checked_add(other.writeback_completed_bytes)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L2 writeback bytes",
                })?,
            l1_writeback_completed_lines: self
                .l1_writeback_completed_lines
                .checked_add(other.l1_writeback_completed_lines)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L1 writeback lines",
                })?,
            l1_writeback_completed_sectors: self
                .l1_writeback_completed_sectors
                .checked_add(other.l1_writeback_completed_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L1 writeback sectors",
                })?,
            l1_writeback_completed_bytes: self
                .l1_writeback_completed_bytes
                .checked_add(other.l1_writeback_completed_bytes)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L1 writeback bytes",
                })?,
            l1_payload_forwarded_sectors: self
                .l1_payload_forwarded_sectors
                .checked_add(other.l1_payload_forwarded_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "logical L1 payload-forwarded sectors",
                })?,
            l1_payload_forwarded_bytes: self
                .l1_payload_forwarded_bytes
                .checked_add(other.l1_payload_forwarded_bytes)
                .ok_or(SystemCounterError::Overflow {
                    counter: "logical L1 payload-forwarded bytes",
                })?,
            l1_payload_forwarding_digest_xor: self.l1_payload_forwarding_digest_xor
                ^ other.l1_payload_forwarding_digest_xor,
            writeback_completion_frontier: self
                .writeback_completion_frontier
                .into_iter()
                .chain(other.writeback_completion_frontier)
                .max(),
            coherence_completed_transactions: self
                .coherence_completed_transactions
                .checked_add(other.coherence_completed_transactions)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed coherence transactions",
                })?,
            coherence_completed_sectors: self
                .coherence_completed_sectors
                .checked_add(other.coherence_completed_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed coherence sectors",
                })?,
            coherence_completion_frontier: self
                .coherence_completion_frontier
                .into_iter()
                .chain(other.coherence_completion_frontier)
                .max(),
            maximum_terminal_l2_dirty_lines: self
                .maximum_terminal_l2_dirty_lines
                .max(other.maximum_terminal_l2_dirty_lines),
            maximum_terminal_l2_dirty_sectors: self
                .maximum_terminal_l2_dirty_sectors
                .max(other.maximum_terminal_l2_dirty_sectors),
            maximum_terminal_l1_dirty_lines: self
                .maximum_terminal_l1_dirty_lines
                .max(other.maximum_terminal_l1_dirty_lines),
            maximum_terminal_l1_dirty_sectors: self
                .maximum_terminal_l1_dirty_sectors
                .max(other.maximum_terminal_l1_dirty_sectors),
            dram_partition_modulo_sectors,
            dram_partition_xor_sectors,
            dram_slice_modulo_sectors,
            l2_frontier_samples,
            l2_issue_to_fill_cycles_sum,
            l2_issue_to_fill_cycles_max,
            l2_issue_to_response_cycles_sum,
            l2_issue_to_response_cycles_max,
            l2_service_delay_cycles_sum,
            l2_service_delay_cycles_max,
            by_origin,
        };
        self.validate_origin_partition()?;
        other.validate_origin_partition()?;
        next.validate_origin_partition()?;
        *self = next;
        Ok(())
    }
}
