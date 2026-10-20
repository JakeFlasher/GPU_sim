//! Replay coverage bindings and completion frontiers; report-only launch accounting.

use super::*;

/// Launch identity and shape bound into a verified replay-coverage witness.
/// Declaration-side construction creates only an expected identity; the extraction
/// completeness gate remains the only constructor of observed coverage witnesses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReplayLaunchCoverageBinding {
    pub(super) device_id: i64,
    pub(super) stream_id: i64,
    pub(super) kernel_id: i32,
    pub(super) function_unique_id: i32,
    pub(super) kernel_mangled_name: String,
    pub(super) grid: (u32, u32, u32),
    pub(super) block: Option<(u32, u32, u32)>,
    pub(super) cluster: (u32, u32, u32),
}

impl ReplayLaunchCoverageBinding {
    pub fn device_id(&self) -> i64 {
        self.device_id
    }

    pub fn stream_id(&self) -> i64 {
        self.stream_id
    }

    pub fn kernel_id(&self) -> i32 {
        self.kernel_id
    }

    pub fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    pub fn kernel_mangled_name(&self) -> &str {
        &self.kernel_mangled_name
    }

    pub fn grid(&self) -> (u32, u32, u32) {
        self.grid
    }

    pub fn block(&self) -> Option<(u32, u32, u32)> {
        self.block
    }

    pub fn cluster(&self) -> (u32, u32, u32) {
        self.cluster
    }
}

/// Unforgeable-by-consumers witness that replay observed every block in one launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullLaunchCoverage {
    pub(super) launch: ReplayLaunchCoverageBinding,
    pub(super) blocks_observed: u64,
    pub(super) source: FullCoverageSource,
}

impl FullLaunchCoverage {
    pub fn launch(&self) -> &ReplayLaunchCoverageBinding {
        &self.launch
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn source(&self) -> &FullCoverageSource {
        &self.source
    }
}

/// Unforgeable-by-consumers witness that replay observed exactly the block roster
/// sealed by one validated subset sidecar.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampledLaunchCoverage {
    pub(super) launch: ReplayLaunchCoverageBinding,
    pub(super) population_blocks: u64,
    pub(super) blocks_observed: u64,
    pub(super) provenance: WaveSegmentSubsetProvenance,
}

#[cfg(feature = "campaign")]
impl SampledLaunchCoverage {
    pub fn launch(&self) -> &ReplayLaunchCoverageBinding {
        &self.launch
    }

    pub fn population_blocks(&self) -> u64 {
        self.population_blocks
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn provenance(&self) -> &WaveSegmentSubsetProvenance {
        &self.provenance
    }
}

/// Trace-derived compulsory front-end traffic. Every distinct executed 32-byte
/// SASS sector must enter the instruction/constant hierarchy at least once;
/// this observer is independent of the data-cache/L2 timing state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrontEndCounters {
    pub(super) compulsory_instruction_read_sectors: u64,
    /// Distinct 32-byte constant-bank sectors referenced by executed static
    /// SASS instructions. This is an observation-only non-TEX demand term;
    /// it never changes instruction timing or cache state.
    pub(super) compulsory_constant_read_sectors: u64,
    /// Number of 32-byte sectors in the unique 128-byte instruction-line
    /// envelope.  GCC/ICC fetches an aligned 128-byte line even when the
    /// traced control flow executes only one sector of that line.  This is a
    /// source-qualified traffic observer; it does not alter issue timing.
    pub(super) compulsory_instruction_line_read_sectors: u64,
    /// Static SASS instruction footprint rounded to aligned 128-byte GCC
    /// fetch lines. This remains separate from the executed-path ledger above
    /// so consumers can retain a bounded cold/static envelope explicitly.
    pub(super) static_instruction_line_read_sectors: u64,
}

impl FrontEndCounters {
    pub const fn compulsory_instruction_read_sectors(self) -> u64 {
        self.compulsory_instruction_read_sectors
    }

    pub const fn compulsory_constant_read_sectors(self) -> u64 {
        self.compulsory_constant_read_sectors
    }

    pub const fn compulsory_instruction_line_read_sectors(self) -> u64 {
        self.compulsory_instruction_line_read_sectors
    }

    pub const fn static_instruction_line_read_sectors(self) -> u64 {
        self.static_instruction_line_read_sectors
    }

    pub(super) fn add_checked(self, other: Self) -> Option<Self> {
        Some(Self {
            compulsory_instruction_read_sectors: self
                .compulsory_instruction_read_sectors
                .checked_add(other.compulsory_instruction_read_sectors)?,
            compulsory_constant_read_sectors: self
                .compulsory_constant_read_sectors
                .checked_add(other.compulsory_constant_read_sectors)?,
            compulsory_instruction_line_read_sectors: self
                .compulsory_instruction_line_read_sectors
                .checked_add(other.compulsory_instruction_line_read_sectors)?,
            static_instruction_line_read_sectors: self
                .static_instruction_line_read_sectors
                .checked_add(other.static_instruction_line_read_sectors)?,
        })
    }
}

/// Completion knowledge for one fully replayed launch.
///
/// `Complete` means every timing-bearing effect known to the replay model is
/// represented in its event stream. `Incomplete` preserves the event frontier
/// only as a lower bound and names the known retirement tails that remain
/// outside that stream. Neither variant includes the profiler launch envelope
/// required for `gpc__cycles_elapsed.max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeledCompletionFrontier {
    Complete {
        cycles: Option<u64>,
    },
    Incomplete {
        lower_bound_cycles: Option<u64>,
        unmodeled_tails: TailFlags,
    },
}

impl ModeledCompletionFrontier {
    pub const fn lower_bound_cycles(self) -> Option<u64> {
        match self {
            Self::Complete { cycles } => cycles,
            Self::Incomplete {
                lower_bound_cycles, ..
            } => lower_bound_cycles,
        }
    }

    pub const fn complete_cycles(self) -> Option<u64> {
        match self {
            Self::Complete { cycles } => cycles,
            Self::Incomplete { .. } => None,
        }
    }

    pub const fn unmodeled_tails(self) -> TailFlags {
        match self {
            Self::Complete { .. } => TailFlags {
                red_rmw: false,
                dirty_writeback: false,
                coherence: false,
                ldgsts_dst_shared_timing: false,
            },
            Self::Incomplete {
                unmodeled_tails, ..
            } => unmodeled_tails,
        }
    }

    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Complete { .. })
    }
}
