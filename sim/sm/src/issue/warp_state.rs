//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

pub(super) struct WarpState<'a> {
    pub(super) insts: &'a [ReplayInst],
    pub(super) sectors: &'a [i64],
    pub(super) async_copy_cohorts: Option<&'a AsyncCopyCohortReplay>,
    pub(super) next: usize,
    /// Dynamic trace cursor already requested by the fetch scheduler and the
    /// three-entry instruction buffer (including in-flight fetches).
    pub(super) front_end_fetch_cursor: usize,
    pub(super) instruction_buffer: VecDeque<BufferedInstruction>,
    pub(super) next_slot: Option<&'a IssueSlot>,
    /// Issue/admission timestamp at which `next_slot` became the current
    /// instruction. Causal readiness delays are measured from this frontier.
    pub(super) current_since: u64,
    pub(super) admission_time: u64,
    pub(super) eligible: u64,
    /// Earliest cycle at which the current instruction may be re-issued after
    /// an immediate-constant-cache miss. Zero means no pending IMC request.
    pub(super) imc_wait_until: u64,
    /// Compiler-emitted `DEPBAR.LE SB1..SB5, 0x0[, {..}]` operands for the
    /// current instruction. This is cached when `next_slot` changes so
    /// scheduler scans use the ordinary barrier counter/clear path without
    /// reparsing SASS.
    pub(super) ordinary_dependency_wait_mask: WaitMask,
    pub(super) barrier_clear: [u64; DEP_BARRIERS],
    pub(super) barrier_origin: [DependencyOrigin; DEP_BARRIERS],
    /// Outstanding UNKNOWN write-backs per barrier index (L1Shared mode):
    /// hardware barriers are counters, and a response still in flight is
    /// an increment whose decrement time is unknown — a wait on the index
    /// cannot pass, and the warp has no computable ready time, until
    /// every pending unit resolves.
    pub(super) barrier_pending: [u32; DEP_BARRIERS],
    /// Full replay's dynamic fixed-latency register recurrence. Native
    /// extraction currently carries no operand values in `ReplayInst`, so
    /// alpha uses the executing warp mask conservatively (all lanes for
    /// legacy captures); fast replay refines this with its compact masks.
    pub(super) register_ready: Vec<RegisterReadyEntry>,
    pub(super) block: usize,
    pub(super) launch_index: i64,
    pub(super) warp_id: i32,
    pub(super) warp_slot: i32,
    /// Index into its sub-core's pick list while live (retired warps are
    /// swap-removed from the list; the swapped-in warp's position is
    /// fixed up). A warp is only ever picked from its own sub-core, so
    /// the retiring sub-core index is the loop's `sc`.
    pub(super) pos_in_subcore: usize,
    pub(super) arrival: u64,
    pub(super) at_block_barrier: bool,
    /// CTA-release arm for the current post-barrier instruction. This is kept
    /// separate from compiler-control eligibility so attribution cannot confuse
    /// a CTA rendezvous with a SASS stall count.
    pub(super) cta_barrier_ready: u64,
    /// Completion of this warp's own arrival through the SM-shared barrier
    /// service. The CTA arm above carries only other-participant rendezvous.
    pub(super) barrier_pipeline_ready: u64,
    /// Effective dependent-SFU result frontier. Only the narrow static
    /// same-register MUFU/RCP/RSQ/EX2/LG2/SIN/COS shape updates this field.
    pub(super) sfu_dependency_ready: u64,
    pub(super) branch_pipeline_ready: u64,
    pub(super) block_barrier_dispatch: Option<u64>,
    pub(super) block_barrier_arrival: Option<u64>,
    pub(super) clock_first_issue: Option<u64>,
    pub(super) clock_reads_issued: u64,
    pub(super) done: bool,
}

/// Completion identity carried by an eligible L1 access. Ordinary loads only
/// observe a dependence barrier; cacheable asynchronous-copy sources also retain
/// their open group while optionally preserving an ordinary barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FillTarget {
    Barrier(DepBarrier),
    AsyncCopy {
        group: AsyncCopyGroupId,
        barrier: Option<DepBarrier>,
    },
}

impl FillTarget {
    pub(super) const fn barrier(self) -> Option<DepBarrier> {
        match self {
            Self::Barrier(barrier) => Some(barrier),
            Self::AsyncCopy { barrier, .. } => barrier,
        }
    }

    pub(super) const fn async_group(self) -> Option<AsyncCopyGroupId> {
        match self {
            Self::Barrier(_) => None,
            Self::AsyncCopy { group, .. } => Some(group),
        }
    }
}

/// Completion units cleared when one keyed L1 fill completes. Inline for the
/// dominant single-waiter identity and pre-aggregated by `(warp, target)`, so
/// response preflight only rebuilds counts when distinct targets share a fill.
pub(super) enum Waiters {
    One(FillWaiter),
    Many(Vec<FillWaiter>),
}

impl Waiters {
    pub(super) fn add(
        &mut self,
        warp: usize,
        target: FillTarget,
        constant_deep: bool,
    ) -> Result<(), IssueError> {
        let new_waiter = FillWaiter {
            warp,
            target,
            units: 1,
            constant_deep,
        };
        match self {
            Self::One(waiter)
                if waiter.warp == warp
                    && waiter.target == target
                    && waiter.constant_deep == constant_deep =>
            {
                waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
            }
            Self::One(waiter) => {
                *self = Self::Many(vec![*waiter, new_waiter]);
            }
            Self::Many(waiters) => {
                if let Some(waiter) = waiters.iter_mut().find(|waiter| {
                    waiter.warp == warp
                        && waiter.target == target
                        && waiter.constant_deep == constant_deep
                }) {
                    waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
                } else {
                    waiters.push(new_waiter);
                }
            }
        }
        Ok(())
    }

    pub(super) fn try_for_each(
        &self,
        mut visit: impl FnMut(FillWaiter) -> Result<(), IssueError>,
    ) -> Result<(), IssueError> {
        match self {
            Self::One(waiter) => visit(*waiter),
            Self::Many(waiters) => {
                for &waiter in waiters {
                    visit(waiter)?;
                }
                Ok(())
            }
        }
    }
}

/// State that EXISTS only in the shared-L2 mode — carrying it inside
/// the [`MemModel`] variant makes a mispaired construction
/// unrepresentable instead of an `expect()` deep in the memory step.
pub(super) struct SharedL2State {
    pub(super) l1: L1Cache,
    /// Monotonic tie-breaker for the per-SM visibility stream. Memory cycle
    /// timestamps alone cannot order equal-cycle fences and accesses.
    pub(super) l1_visibility_sequence: u64,
    /// Complete global eligible L1 observations, including resident hits.
    /// The native system currently drains this only through the explicit
    /// opt-in callback API; the legacy L2-only path remains unchanged.
    pub(super) l1_visibility_observations: Vec<sim_mem::L1AccessObservation>,
    pub(super) l1_fence_observations: Vec<sim_mem::L1FenceObservation>,
    pub(super) l1_fill_visibility_observations: Vec<sim_mem::L1FillVisibleObservation>,
    pub(super) l1_invalidate_all_observations: Vec<sim_mem::L1InvalidateAllObservation>,
    pub(super) l1_visibility_requester: Option<u8>,
    /// Only payload-event sectors are tracked; default replay allocates no
    /// value-observation map. Every later store to a watched sector replaces
    /// its writer, including stores with no corresponding payload event.
    pub(super) payload_store_writers:
        Option<Box<BTreeMap<(i64, u8), Option<PayloadStoreObservation>>>>,
    pub(super) response_owner: ResponseOwner,
    /// Exact response-bearing request tokens still expected from the system.
    pub(super) outstanding_replies: HashSet<ReplyTo>,
    /// Response identities indexed separately so admission checks do not scan
    /// every outstanding token for each sector in a wide instruction.
    pub(super) outstanding_response_ids: HashMap<ResponseId, ReplyTo>,
    /// Exact sector mask carried by each outstanding response-bearing request.
    /// This stays separate from `fill_masks`, which is the physical L2
    /// exactly-once traffic mask and may include a later child fill.
    pub(super) response_masks: HashMap<ResponseId, u8>,
    /// Barrier units cleared when the keyed L1 fill completes.
    pub(super) fill_waiters: HashMap<FillKey, Waiters>,
    /// Per in-flight LINE fill: the sector bits already sent to the L2
    /// (the fill request's mask plus any follow-up fetches). A merge
    /// onto a sector NOT yet in the mask sends a follow-up sector
    /// fetch with its own L1 sector response; a re-touch of a sent
    /// sector sends nothing, like the hardware MSHR. Entries die
    /// with the parent line response. Exactly-once is PER FILL, not
    /// per-address-epoch: if a resident line with an in-flight SECTOR
    /// fill is evicted and re-allocated, the new fill may re-fetch a
    /// sector whose zombie fetch is still outstanding (rare
    /// eviction-window corner, documented at the 2026-07-07 gate).
    pub(super) fill_masks: HashMap<i64, u8>,
    /// Line fills emitted during the CURRENT timestamped emission batch:
    /// `(line, issue_time) → first index in out`. A merge onto one of these
    /// folds its sector into the fill request's mask. Requests for the same
    /// line at a later local cycle receive an independent sector token; a
    /// whole `advance` call is not a same-cycle packet boundary.
    /// Cleared at every batch entry; indices are only valid because
    /// each batch starts with an empty `out`.
    pub(super) line_folds: HashMap<(i64, u64), usize>,
    pub(super) retry: VecDeque<Retry>,
}

impl SharedL2State {
    pub(super) fn next_l1_visibility_sequence(&mut self) -> Result<u64, IssueError> {
        let sequence = self.l1_visibility_sequence;
        self.l1_visibility_sequence =
            sequence.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "L1 visibility sequence",
            })?;
        Ok(sequence)
    }
}

/// The typed memory model an [`SmCore`] runs under (the public
/// [`SmMemMode`] resolved into owned state).
pub(super) enum MemModel {
    Flat,
    L1Const(L1Cache),
    L1Shared(SharedL2State),
}

impl MemModel {
    pub(super) fn l1(&self) -> Option<&L1Cache> {
        match self {
            MemModel::Flat => None,
            MemModel::L1Const(l1) => Some(l1),
            MemModel::L1Shared(st) => Some(&st.l1),
        }
    }

    /// The L1 the CONSTANT-latency data path drives (`None` in Flat and
    /// Shared modes — the shared path never uses `data_ready_time`).
    pub(super) fn const_l1_mut(&mut self) -> Option<&mut L1Cache> {
        match self {
            MemModel::L1Const(l1) => Some(l1),
            _ => None,
        }
    }

    pub(super) fn shared_l1_mut(&mut self) -> Option<&mut SharedL2State> {
        match self {
            MemModel::L1Shared(state) => Some(state),
            _ => None,
        }
    }
}
