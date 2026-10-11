use sim_mem::{AccessKind, CacheOpToken, L2OpClass};

/// Memory-hierarchy effects deliberately outside the modeled event stream.
/// These flags are capability boundaries, not synthetic traffic counters: a
/// set bit prevents consumers from mistaking a partial write-class model for a
/// complete RMW/writeback/coherence implementation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TailFlags {
    /// RED/REDG's value-dependent read/modify/serialize semantics are not
    /// modeled. Cold-sector read demand is charged separately by the shared
    /// L2/DRAM observer, but that traffic count is not a complete reduction
    /// protocol.
    pub red_rmw: bool,
    /// Dirty eviction/writeback traffic is not modeled.
    pub dirty_writeback: bool,
    /// Coherence propagation and invalidation traffic is not modeled.
    pub coherence: bool,
    /// The modeled LDGSTS commit/wait chain cannot establish destination
    /// visibility: either the trace omitted destination payload or the bypass
    /// route has no promoted per-copy destination timing mechanism.
    pub ldgsts_dst_shared_timing: bool,
}

impl TailFlags {
    /// True only when replay has no known timing-bearing hierarchy effect
    /// outside the modeled event stream.
    pub const fn is_empty(self) -> bool {
        !self.red_rmw && !self.dirty_writeback && !self.coherence && !self.ldgsts_dst_shared_timing
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            red_rmw: self.red_rmw || other.red_rmw,
            dirty_writeback: self.dirty_writeback || other.dirty_writeback,
            coherence: self.coherence || other.coherence,
            ldgsts_dst_shared_timing: self.ldgsts_dst_shared_timing
                || other.ldgsts_dst_shared_timing,
        }
    }

    pub(super) const fn for_access(kind: AccessKind) -> Self {
        match kind {
            AccessKind::GlobalReduction { .. } => Self {
                red_rmw: true,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            },
            AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. } => Self {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            },
            AccessKind::GlobalBypass {
                token: CacheOpToken::Atom,
                ..
            } => Self {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            },
            AccessKind::NonMemory
            | AccessKind::PredicatedOffMemory
            | AccessKind::Shared { .. }
            | AccessKind::LocalLoad { .. }
            | AccessKind::GlobalEligible { .. }
            | AccessKind::GlobalBypass { .. }
            | AccessKind::GlobalAsyncCopy { .. }
            | AccessKind::GlobalUntagged { .. } => Self {
                red_rmw: false,
                dirty_writeback: false,
                coherence: false,
                ldgsts_dst_shared_timing: false,
            },
        }
    }
}

/// Semantic operation carried through the shared-memory request protocol.
/// L2 hit policy is derived from this type so Store, returning ATOM, and
/// non-returning RED cannot collapse into one `WriteAtomic` request class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOperation {
    Read,
    Store,
    Atomic,
    Reduction,
}

impl MemoryOperation {
    pub(crate) const fn l2_class(self) -> L2OpClass {
        match self {
            Self::Read => L2OpClass::Read,
            Self::Store | Self::Atomic | Self::Reduction => L2OpClass::WriteAtomic,
        }
    }

    pub(crate) const fn fetches_missing_data(self) -> bool {
        // RED is non-returning at the ISA boundary, but the L2 write-class
        // request still needs the old value for its internal read-modify-
        // write.  The event model charges that missing-sector read through
        // the shared DRAM server without fabricating a warp response.
        matches!(self, Self::Read | Self::Atomic | Self::Reduction)
    }
}
