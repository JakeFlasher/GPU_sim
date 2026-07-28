//! Shared L2 slice model — a lockstep port of the sectored set-associative
//! replay core in `tools/cache_replay/sector_cache_replay.py`
//! (`replay_access_stream` + `_install_way`), the engine behind the
//! Phase-1 L2 predictor. Like the L1 port, only the CACHE STATE MACHINE
//! is taken: the Python module replayed a timeless trace-order stream,
//! while the sim drives this model with the globally (time, seq)-ordered
//! request stream the event engine produces — the ordering the flat
//! replay stood in for.
//!
//! Ported semantics (32 B sectors in 128 B lines) — these bullets
//! describe the READ class, the rule the Python engine implements; the
//! §6.4 [`L2OpClass::WriteAtomic`] class additionally hits on line-TAG
//! residency alone (same state trajectory, different counting — see
//! `L2OpClass`; the parity corpus pins the Read class only):
//! - a Read-class sector access HITS iff its line tag is resident in
//!   the addressed set AND that sector's valid bit is set;
//! - a resident line whose requested sector is not yet valid is a
//!   sector-fill miss for reads (the bit is set immediately — the
//!   Python replay is occupancy-only, fills are not in-flight at this
//!   level) and a tag-match HIT for writes/atomics;
//! - a non-resident line allocates a way, evicting per policy when the
//!   set is full (LRU / SRRIP 2-bit RRPV / deterministic BIP 1-in-32);
//! - slice/set addressing: `slice = hash(line_id)`,
//!   `set = (line_id // num_slices) % num_sets_per_slice`, with the six
//!   pre-registered slice-hash candidates ported bit-exactly.
//!
//! Rule 1 note: the Phase-1 calibrator SELECTED a config for sm_86 by
//! fitting against local NCU labels (capacity_scale, hash, policy). That
//! selection is a label-fitted artifact and is deliberately NOT consumed
//! here. The sim's geometry comes from the arch card's vendor-documented
//! `l2_geometry` (capacity, line/sector, slice count); associativity,
//! slice hash and replacement policy are DOCUMENTED-DEFAULT hypothesis
//! knobs below (same fail-closed pattern as the L1 MSHR default), to be
//! closed by microbenchmarks — never by label fits.
//!
//! MSHR/fill timing: v0 models L2 occupancy only (no L2-level MSHR or
//! in-flight fill state). The L1's MSHRs already bound the per-SM
//! outstanding stream, and the Python reference has no L2 fill model to
//! port — an L2 MSHR would be an uncarded invention at this point.

use sim_arch::{ArchError, ArchId, CapabilityError, L2Replacement as ArchL2Replacement};

/// Number of 32 B sectors per 128 B line (fixed physical granularity;
/// the `u8` way masks below hold exactly these four bits).
pub const L2_SECTORS_PER_LINE: u32 = 4;

/// Documented-default L2 associativity (research gap on every card):
/// 16 ways is the Accel-Sim Ampere config family's value and the common
/// literature assumption for NVIDIA L2s; not microbenched on this
/// project's cards yet. Same contract as the L1 MSHR default: a
/// documented hypothesis knob, never a silent invention.
pub const DOCUMENTED_DEFAULT_L2_ASSOCIATIVITY: u32 = 16;

/// The six pre-registered slice-hash candidates, ported bit-exactly from
/// `SLICE_HASH_REGISTRY`. The card records only the slice COUNT
/// (vendor-documented); which hash the crossbar uses is a research gap.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum L2SetIndexError {
    #[error("L2 slice count must be positive")]
    SlicesZero,
    #[error("L2 sets per slice must be positive")]
    SetsPerSliceZero,
    #[error("L2 sets per slice {sets_per_slice} does not fit i64")]
    SetsPerSliceOutOfSignedRange { sets_per_slice: u64 },
    #[error("L2 slice index {slice} does not fit u32")]
    SliceOutOfRange { slice: i64 },
    #[error("L2 set index {set_index} does not fit u64")]
    SetIndexOutOfRange { set_index: i64 },
    #[error(
        "L2 set key overflows u64: slice={slice}, sets_per_slice={sets_per_slice}, set_index={set_index}"
    )]
    SetKeyOverflow {
        slice: u32,
        sets_per_slice: u64,
        set_index: u64,
    },
    #[error("L2 set key {set_key} does not fit usize")]
    SetKeyOutOfRange { set_key: u64 },
    #[error("L2 set key {set_key} is outside {allocated_sets} allocated sets")]
    SetKeyOutsideStorage {
        set_key: usize,
        allocated_sets: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceHash {
    /// Documented-default candidate (v0 approximation).
    LineMod,
    SectorMod,
    XorFolded,
    KnuthMult,
    ShiftMod,
    BitReverse,
}

/// Documented-default slice hash: plain line-interleaving. The Python
/// registry existed to CALIBRATE this unknown; v0 pins the simplest
/// candidate and records it as an approximation.
pub const DOCUMENTED_DEFAULT_L2_SLICE_HASH: SliceHash = SliceHash::LineMod;

impl SliceHash {
    /// `slice_id` for a 128 B line address. Lockstep with the Python
    /// registry functions, INCLUDING negative line ids: Python's `%` is
    /// floored (→ `rem_euclid`), `>>` on negatives is an arithmetic
    /// shift, and `& 0xFFFF…` of a negative yields its two's-complement
    /// low bits — all matched here so a wire-admitted negative address
    /// (a `base_delta` stream summing below zero is structurally legal)
    /// replays identically instead of wrapping through `u64`.
    /// Pinned-value tests below cover both signs.
    pub fn slice_of(self, line_id: i64, num_slices: u32) -> Result<u32, L2SetIndexError> {
        if num_slices == 0 {
            return Err(L2SetIndexError::SlicesZero);
        }
        let n = i64::from(num_slices);
        let s = match self {
            SliceHash::LineMod => line_id.rem_euclid(n),
            // (line * 4) % n == ((line %e n) * 4) % n — overflow-free
            // and sign-correct under the euclidean remainder.
            SliceHash::SectorMod => {
                (line_id.rem_euclid(n) * i64::from(L2_SECTORS_PER_LINE)).rem_euclid(n)
            }
            SliceHash::XorFolded => (line_id ^ (line_id >> 5) ^ (line_id >> 11)).rem_euclid(n),
            // Knuth multiplicative hash masked to the low 32 bits — the
            // wrapping product's low bits equal Python's full-precision
            // product masked (two's complement, both signs).
            SliceHash::KnuthMult => {
                (line_id.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF).rem_euclid(n)
            }
            SliceHash::ShiftMod => (line_id >> 1).rem_euclid(n),
            SliceHash::BitReverse => i64::from((line_id as u16).reverse_bits()).rem_euclid(n),
        };
        u32::try_from(s).map_err(|_| L2SetIndexError::SliceOutOfRange { slice: s })
    }
}

/// Replacement policy. LRU is the documented-default approximation (the
/// same status as the L1 pool's LRU); SRRIP/BIP are ported so the
/// lockstep differential covers the full Python surface and calibration
/// can flip the knob later without a re-port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Replacement {
    Lru,
    Srrip,
    Bip,
}

pub const DOCUMENTED_DEFAULT_L2_REPLACEMENT: L2Replacement = L2Replacement::Lru;

const SRRIP_RRPV_MAX: i64 = 3;
const SRRIP_INSERT_RRPV: i64 = 2;
const BIP_MRU_INSERT_PERIOD: u64 = 32;

#[derive(Debug, thiserror::Error)]
pub enum L2ConfigError {
    #[error("architecture token is invalid: {0}")]
    Architecture(#[from] ArchError),
    #[error("architecture capability is unavailable: {0}")]
    Capability(#[from] CapabilityError),
    #[error("capacity {capacity} is below the {line}B line size")]
    CapacityBelowLine { capacity: u64, line: u32 },
    #[error(
        "line/sector geometry must be 128/32 (got line={line}, sector={sector}) — the way masks are 4-bit"
    )]
    UnsupportedGranularity { line: u32, sector: u32 },
    #[error("associativity must be >= 1")]
    AssociativityZero,
    #[error("num_slices must be >= 1")]
    SlicesZero,
    #[error(
        "arch {arch:?} has not reviewed the documented-default L2 knobs (assoc 16/LRU/line_mod \
         were reviewed for sm_86 only) — card the fields or extend the reviewed set deliberately"
    )]
    DefaultsUnreviewedForArch { arch: String },
    #[error(
        "card l2_geometry.policy is absent — the model write-allocates stores, and a card \
         without a measured policy must fail closed, never assume write-allocate"
    )]
    WritePolicyGap,
    #[error(
        "card l2_geometry.policy {policy:?} is not a write-allocate token this model implements"
    )]
    WritePolicyUnsupported { policy: String },
}

#[derive(Debug, thiserror::Error)]
pub enum L2BuildError {
    #[error(transparent)]
    Config(#[from] L2ConfigError),
    #[error("L2 set count overflows u64: num_slices={num_slices}, sets_per_slice={sets_per_slice}")]
    SetCountOverflow {
        num_slices: u32,
        sets_per_slice: u64,
    },
    #[error("L2 set count {set_count} does not fit usize")]
    SetCountOutOfRange { set_count: u64 },
    #[error("failed to allocate storage for {set_count} L2 sets: {source}")]
    SetStorageAllocation {
        set_count: usize,
        #[source]
        source: std::collections::TryReserveError,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum L2SnapshotError {
    #[error(transparent)]
    Counter(#[from] L2CounterError),
    #[error(
        "failed to reserve capacity {capacity} while snapshotting the outer storage for {set_count} L2 sets: {source}"
    )]
    SetStorageAllocation {
        set_count: usize,
        capacity: usize,
        #[source]
        source: std::collections::TryReserveError,
    },
    #[error(
        "failed to reserve capacity {capacity} while snapshotting L2 set {set_key} with {ways} ways: {source}"
    )]
    WayStorageAllocation {
        set_key: usize,
        ways: usize,
        capacity: usize,
        #[source]
        source: std::collections::TryReserveError,
    },
}

/// Arches whose documented-default L2 knobs (associativity 16, LRU,
/// line_mod) have been REVIEWED against available public evidence.
/// Same per-arch-set pattern as `cache_op::BARE_LDG_E_DEFAULTS_TO_CA`:
/// an unreviewed arch fails closed instead of inheriting another
/// arch's hypotheses (CLAUDE.md architecture-coupling discipline).
pub const L2_DEFAULTS_REVIEWED_ARCHES: [&str; 1] = ["sm_86"];

/// The one write policy the store path implements (stores allocate in
/// L2 like loads). Cards carrying anything else — or nothing — fail
/// closed at construction.
pub const L2_WRITE_ALLOCATE_POLICY: &str = "write_allocate_non_inclusive";

/// L2 geometry + policy. Construct via [`L2Config::from_carded_geometry`]
/// so the documented-default provenance is explicit at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Config {
    /// Deployment architecture whose reviewed defaults/geometry this config carries.
    pub arch: ArchId,
    pub capacity_bytes: u64,
    pub line_bytes: u32,
    pub sector_bytes: u32,
    pub num_slices: u32,
    pub associativity: u32,
    pub replacement: L2Replacement,
    pub slice_hash: SliceHash,
}

impl L2Config {
    /// Typed architecture entry point.  Reviewed defaults are obtained from
    /// the shared capability vocabulary; an sm_120 research gap cannot reach
    /// the model as an sm_86 configuration.
    pub fn from_carded_geometry_for_arch(
        arch: ArchId,
        capacity_bytes: u64,
        line_bytes: u32,
        sector_bytes: u32,
        num_slices: u32,
        policy: Option<&str>,
    ) -> Result<Self, L2ConfigError> {
        let defaults = arch.l2_defaults().require("l2_defaults").map_err(|_| {
            L2ConfigError::DefaultsUnreviewedForArch {
                arch: arch.token().to_owned(),
            }
        })?;
        match policy {
            None => return Err(L2ConfigError::WritePolicyGap),
            Some(L2_WRITE_ALLOCATE_POLICY) => {}
            Some(other) => {
                return Err(L2ConfigError::WritePolicyUnsupported {
                    policy: other.to_owned(),
                });
            }
        }
        let cfg = Self {
            arch,
            capacity_bytes,
            line_bytes,
            sector_bytes,
            num_slices,
            associativity: defaults.associativity,
            replacement: match defaults.replacement {
                ArchL2Replacement::Lru => L2Replacement::Lru,
            },
            slice_hash: match defaults.slice_hash {
                sim_arch::L2SliceHash::LineMod => SliceHash::LineMod,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Compatibility string-token entry point. New composition code should
    /// resolve a typed [`ArchId`] before constructing the cache.
    pub fn from_carded_geometry(
        arch_token: &str,
        capacity_bytes: u64,
        line_bytes: u32,
        sector_bytes: u32,
        num_slices: u32,
        policy: Option<&str>,
    ) -> Result<Self, L2ConfigError> {
        let arch = ArchId::parse_deploy(arch_token)?;
        Self::from_carded_geometry_for_arch(
            arch,
            capacity_bytes,
            line_bytes,
            sector_bytes,
            num_slices,
            policy,
        )
    }

    pub fn validate(&self) -> Result<(), L2ConfigError> {
        if self.arch != ArchId::Sm86 {
            return Err(L2ConfigError::DefaultsUnreviewedForArch {
                arch: self.arch.token().to_owned(),
            });
        }
        if self.line_bytes != 128 || self.sector_bytes != 32 {
            return Err(L2ConfigError::UnsupportedGranularity {
                line: self.line_bytes,
                sector: self.sector_bytes,
            });
        }
        if self.capacity_bytes < u64::from(self.line_bytes) {
            return Err(L2ConfigError::CapacityBelowLine {
                capacity: self.capacity_bytes,
                line: self.line_bytes,
            });
        }
        if self.associativity == 0 {
            return Err(L2ConfigError::AssociativityZero);
        }
        if self.num_slices == 0 {
            return Err(L2ConfigError::SlicesZero);
        }
        Ok(())
    }

    /// `max(1, capacity // line)` — Python `total_lines`.
    pub fn total_lines(&self) -> Result<u64, L2ConfigError> {
        self.validate()?;
        Ok((self.capacity_bytes / u64::from(self.line_bytes)).max(1))
    }

    /// `max(1, max(1, total_lines // num_slices) // associativity)` —
    /// Python `num_sets_per_slice`.
    pub fn num_sets_per_slice(&self) -> Result<u64, L2ConfigError> {
        let total_lines = self.total_lines()?;
        let per_slice = total_lines
            .checked_div(u64::from(self.num_slices))
            .ok_or(L2ConfigError::SlicesZero)?
            .max(1);
        Ok(per_slice
            .checked_div(u64::from(self.associativity))
            .ok_or(L2ConfigError::AssociativityZero)?
            .max(1))
    }
}

/// Who put the sector on the L2 stream. PROVENANCE ONLY: carried on
/// requests for the system layer's emission observer — the cache model
/// itself never consumes it (gate-3 altitude note: emission accounting
/// lives OUTSIDE the models). The BEHAVIORAL per-access dimension is
/// [`L2OpClass`], carried separately: source is 4-way who-sent-it,
/// class is 2-way how-it-hits (an ATOM rides `Bypass` provenance with
/// `WriteAtomic` class).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Source {
    /// An L1 fill request (line-allocation or resident-sector fetch).
    L1Fill,
    /// A manifest-ineligible load sector bypassing the L1 (.cg/.cs/…).
    Bypass,
    /// A global-class sector whose pc the classifier did not tag.
    Untagged,
    /// A global store sector (card policy `write_allocate_non_inclusive`
    /// → stores allocate in L2 exactly like loads).
    Store,
}

/// Behavioral ACCESS CLASS of one L2 sector access — the §6.4
/// per-access-class hit rule. NCU's `lts__t_sectors_lookup_hit`
/// accounting is hypothesized to be per-op-class (gate-3 error
/// attribution, validated on the histogram witness cell): a LOAD needs
/// the sector's DATA, so it hits only when the sector's valid bit is set
/// (the ported Python rule); a WRITE/ATOMIC/REDUCTION carries its own
/// data and coalesces into any resident line, so its lookup hits on
/// line-TAG residency alone. This is a COUNTING rule, not per-cell
/// tuning: the cache state trajectory is identical either way (a
/// write-class access to a resident line validates the sector exactly
/// as the sector-fill-miss path did).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2OpClass {
    /// Load-class sector (L1 fills, bypass loads, untagged, LDGSTS):
    /// hit iff the sector's valid bit is set.
    Read,
    /// Store/atomic/reduction sector (STG, RED, ATOM): hit iff the line
    /// TAG is resident, regardless of the sector's valid bit.
    WriteAtomic,
}

/// Outcome of one L2 sector access. `counted_hit` is the observation
/// (per-class NCU accounting, drives [`L2Counters`] and the driver's
/// emission observer); `sector_present` is the physics (the requested
/// sector's data was resident BEFORE this access — the response-pricing
/// arm: a tag-match write COUNTS as a hit but a dependent atomic reply
/// still fetches absent data from DRAM). For `Read` the two coincide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Outcome {
    pub counted_hit: bool,
    pub sector_present: bool,
}

/// Read-only prediction for an ascending sector-mask walk of one line.
/// [`L2Model::access_mask_transactional`] supplies this preview to its
/// pre-commit callback so outcome-dependent system work can fail before the
/// prepared cache state becomes visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2MaskPreview {
    pub sectors: u32,
    pub counted_hits: u32,
    pub present_sectors: u32,
}

impl L2MaskPreview {
    pub fn all_present(self) -> bool {
        self.present_sectors == self.sectors
    }
}

#[derive(Debug, thiserror::Error)]
pub enum L2AccessError {
    #[error("L2 sector {sector} is outside the 0..4 sector domain")]
    SectorOutOfRange { sector: u8 },
    #[error("L2 sector mask {sector_mask:#06b} must select one or more of bits 0..4")]
    SectorMaskInvalid { sector_mask: u8 },
    #[error(transparent)]
    SetIndex(#[from] L2SetIndexError),
    #[error("L2 access clock overflows i64 at {clock}")]
    ClockOverflow { clock: i64 },
    #[error("L2 BIP insertion counter overflows u64 at {counter}")]
    BipCounterOverflow { counter: u64 },
    #[error("L2 BIP recency underflows below minimum {minimum} in set {set_key}")]
    BipRecencyUnderflow { set_key: usize, minimum: i64 },
    #[error("L2 associativity {associativity} does not fit usize")]
    AssociativityOutOfRange { associativity: u32 },
    #[error("L2 set {set_key} has {ways} ways, above associativity {associativity}")]
    SetOverfull {
        set_key: usize,
        ways: usize,
        associativity: usize,
    },
    #[error("L2 replacement set {set_key} is unexpectedly empty")]
    ReplacementSetEmpty { set_key: usize },
    #[error("L2 SRRIP way {way} in set {set_key} has invalid recency {recency}")]
    SrripRecencyOutOfRange {
        set_key: usize,
        way: usize,
        recency: i64,
    },
    #[error("failed to reserve {additional} L2 way slot(s) in set {set_key}: {source}")]
    SetStorageAllocation {
        set_key: usize,
        additional: usize,
        #[source]
        source: std::collections::TryReserveError,
    },
    #[error(transparent)]
    Counter(#[from] L2CounterError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Counters {
    /// Architecture provenance bound by the owning [`L2Model`].
    arch: Option<ArchId>,
    pub hits: u64,
    pub misses: u64,
    /// Miss split (state-machine kinds, not the Python 3C diagnostic).
    pub line_misses: u64,
    pub sector_fill_misses: u64,
    /// Of `hits`: write-class hits granted on line-TAG residency while
    /// the sector's valid bit was still clear (the §6.4 rule's whole
    /// contribution — zero when every access is `Read`-class).
    pub write_tag_hits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2CounterField {
    Hits,
    Misses,
    LineMisses,
    SectorFillMisses,
    WriteTagHits,
}

impl std::fmt::Display for L2CounterField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Hits => "hits",
            Self::Misses => "misses",
            Self::LineMisses => "line_misses",
            Self::SectorFillMisses => "sector_fill_misses",
            Self::WriteTagHits => "write_tag_hits",
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum L2CounterError {
    #[error(transparent)]
    Architecture(#[from] sim_arch::ArchitectureProvenanceError),
    #[error("L2 counter '{field}' overflows u64 during aggregation")]
    Overflow { field: L2CounterField },
    #[error("L2 counter total overflows u64: hits={hits}, misses={misses}")]
    TotalOverflow { hits: u64, misses: u64 },
}

fn checked_counter_add(
    field: L2CounterField,
    left: u64,
    right: u64,
) -> Result<u64, L2CounterError> {
    left.checked_add(right)
        .ok_or(L2CounterError::Overflow { field })
}

impl Default for L2Counters {
    fn default() -> Self {
        Self {
            arch: None,
            hits: 0,
            misses: 0,
            line_misses: 0,
            sector_fill_misses: 0,
            write_tag_hits: 0,
        }
    }
}

impl L2Counters {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn add_checked(&mut self, other: &L2Counters) -> Result<(), L2CounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 counters",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 counters",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "L2 counters",
                expected,
                got,
            }
            .into());
        }

        let hits = checked_counter_add(L2CounterField::Hits, self.hits, other.hits)?;
        let misses = checked_counter_add(L2CounterField::Misses, self.misses, other.misses)?;
        let line_misses = checked_counter_add(
            L2CounterField::LineMisses,
            self.line_misses,
            other.line_misses,
        )?;
        let sector_fill_misses = checked_counter_add(
            L2CounterField::SectorFillMisses,
            self.sector_fill_misses,
            other.sector_fill_misses,
        )?;
        let write_tag_hits = checked_counter_add(
            L2CounterField::WriteTagHits,
            self.write_tag_hits,
            other.write_tag_hits,
        )?;

        self.hits = hits;
        self.misses = misses;
        self.line_misses = line_misses;
        self.sector_fill_misses = sector_fill_misses;
        self.write_tag_hits = write_tag_hits;
        Ok(())
    }

    pub fn total(&self) -> Result<u64, L2CounterError> {
        self.hits
            .checked_add(self.misses)
            .ok_or(L2CounterError::TotalOverflow {
                hits: self.hits,
                misses: self.misses,
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Way {
    /// Full line id as the tag (signed: negative wire-admitted ids
    /// replay with Python's floored semantics, never wrapped).
    tag: i64,
    /// Valid-sector bitmask (bit = sector index within the line, 0..4).
    mask: u8,
    /// LRU timestamp (LRU/BIP — BIP inserts can go BELOW the current
    /// minimum, hence signed) or RRPV (SRRIP).
    recency: i64,
}

/// The shared L2: one state machine partitioned into `num_slices ×
/// num_sets_per_slice` sets. Timeless by design — the caller (the event
/// system) presents accesses in the globally ordered request stream.
pub struct L2Model {
    cfg: L2Config,
    num_sets: u64,
    sets: Vec<Vec<Way>>,
    clock: i64,
    bip_counter: u64,
    counters: L2Counters,
}

fn checked_set_count(num_slices: u32, sets_per_slice: u64) -> Result<usize, L2BuildError> {
    let set_count = u64::from(num_slices).checked_mul(sets_per_slice).ok_or(
        L2BuildError::SetCountOverflow {
            num_slices,
            sets_per_slice,
        },
    )?;
    usize::try_from(set_count).map_err(|_| L2BuildError::SetCountOutOfRange { set_count })
}

fn allocate_set_storage(set_count: usize) -> Result<Vec<Vec<Way>>, L2BuildError> {
    let mut sets = Vec::new();
    sets.try_reserve_exact(set_count)
        .map_err(|source| L2BuildError::SetStorageAllocation { set_count, source })?;
    sets.resize_with(set_count, Vec::new);
    Ok(sets)
}

fn require_counter_arch(expected: ArchId, counters: &L2Counters) -> Result<(), L2CounterError> {
    let got = counters
        .arch()
        .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
            component: "L2 counters",
        })?;
    if got != expected {
        return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
            component: "L2 counters",
            expected,
            got,
        }
        .into());
    }
    Ok(())
}

impl L2Model {
    pub fn new(cfg: L2Config) -> Result<Self, L2BuildError> {
        cfg.validate()?;
        let num_sets = cfg.num_sets_per_slice()?;
        let set_count = checked_set_count(cfg.num_slices, num_sets)?;
        let sets = allocate_set_storage(set_count)?;
        let arch = cfg.arch;
        Ok(Self {
            cfg,
            num_sets,
            sets,
            clock: 0,
            bip_counter: 0,
            counters: L2Counters::for_arch(arch),
        })
    }

    pub fn config(&self) -> &L2Config {
        &self.cfg
    }

    /// Return the read-only counter snapshot only when its architecture
    /// provenance still matches the owning cache configuration.
    pub fn counters(&self) -> Result<&L2Counters, L2CounterError> {
        self.validate_counter_architecture()?;
        Ok(&self.counters)
    }

    fn validate_counter_architecture(&self) -> Result<(), L2CounterError> {
        require_counter_arch(self.cfg.arch, &self.counters)
    }

    /// Fallibly copy the complete cache state for a caller-level transaction.
    /// Spare vector capacity is retained so staging does not introduce an
    /// allocation on a path the caller's model had already reserved. No
    /// semantic state in `self` changes if either reservation fails.
    pub fn try_snapshot(&self) -> Result<Self, L2SnapshotError> {
        self.validate_counter_architecture()?;
        let set_count = self.sets.len();
        let set_capacity = self.sets.capacity();
        let mut sets = Vec::new();
        sets.try_reserve_exact(set_capacity).map_err(|source| {
            L2SnapshotError::SetStorageAllocation {
                set_count,
                capacity: set_capacity,
                source,
            }
        })?;
        for (set_key, ways) in self.sets.iter().enumerate() {
            let way_capacity = ways.capacity();
            let mut copied = Vec::new();
            copied.try_reserve_exact(way_capacity).map_err(|source| {
                L2SnapshotError::WayStorageAllocation {
                    set_key,
                    ways: ways.len(),
                    capacity: way_capacity,
                    source,
                }
            })?;
            copied.extend_from_slice(ways);
            sets.push(copied);
        }
        Ok(Self {
            cfg: self.cfg,
            num_sets: self.num_sets,
            sets,
            clock: self.clock,
            bip_counter: self.bip_counter,
            counters: self.counters,
        })
    }

    fn set_key(&self, line_id: i64) -> Result<usize, L2SetIndexError> {
        if self.cfg.num_slices == 0 {
            return Err(L2SetIndexError::SlicesZero);
        }
        if self.num_sets == 0 {
            return Err(L2SetIndexError::SetsPerSliceZero);
        }
        let signed_sets = i64::try_from(self.num_sets).map_err(|_| {
            L2SetIndexError::SetsPerSliceOutOfSignedRange {
                sets_per_slice: self.num_sets,
            }
        })?;
        let slice = self.cfg.slice_hash.slice_of(line_id, self.cfg.num_slices)?;
        let set_index_signed = line_id
            .div_euclid(i64::from(self.cfg.num_slices))
            .rem_euclid(signed_sets);
        let set_index =
            u64::try_from(set_index_signed).map_err(|_| L2SetIndexError::SetIndexOutOfRange {
                set_index: set_index_signed,
            })?;
        let set_key = u64::from(slice)
            .checked_mul(self.num_sets)
            .and_then(|base| base.checked_add(set_index))
            .ok_or(L2SetIndexError::SetKeyOverflow {
                slice,
                sets_per_slice: self.num_sets,
                set_index,
            })?;
        let set_key =
            usize::try_from(set_key).map_err(|_| L2SetIndexError::SetKeyOutOfRange { set_key })?;
        if set_key >= self.sets.len() {
            return Err(L2SetIndexError::SetKeyOutsideStorage {
                set_key,
                allocated_sets: self.sets.len(),
            });
        }
        Ok(set_key)
    }

    fn prepare_mask(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
    ) -> Result<PreparedMaskAccess, L2AccessError> {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        let set_key = self.set_key(line_id)?;
        let associativity = usize::try_from(self.cfg.associativity).map_err(|_| {
            L2AccessError::AssociativityOutOfRange {
                associativity: self.cfg.associativity,
            }
        })?;
        let ways = &mut self.sets[set_key];
        if ways.len() > associativity {
            return Err(L2AccessError::SetOverfull {
                set_key,
                ways: ways.len(),
                associativity,
            });
        }
        validate_srrip_recencies(ways, set_key, self.cfg.replacement)?;

        let resident = ways.iter().position(|way| way.tag == line_id);
        let mut update = resident.map(|way| PreparedSetUpdate::Update {
            way,
            value: ways[way],
        });
        let mut line_present = resident.is_some();
        let mut valid_mask = resident.map_or(0, |way| ways[way].mask);
        let mut clock = self.clock;
        let mut bip_counter = self.bip_counter;
        let mut counters = self.counters;
        let mut counted_mask = 0u8;
        let mut present_mask = 0u8;

        for sector in 0..4u8 {
            let bit = 1u8 << sector;
            if sector_mask & bit == 0 {
                continue;
            }
            clock = clock
                .checked_add(1)
                .ok_or(L2AccessError::ClockOverflow { clock })?;
            let sector_present = line_present && valid_mask & bit != 0;
            let counted_hit = sector_present || (line_present && class == L2OpClass::WriteAtomic);

            if !line_present {
                let next_bip_counter =
                    if self.cfg.replacement == L2Replacement::Bip {
                        Some(bip_counter.checked_add(1).ok_or(
                            L2AccessError::BipCounterOverflow {
                                counter: bip_counter,
                            },
                        )?)
                    } else {
                        None
                    };
                update = Some(prepare_install(
                    ways,
                    set_key,
                    line_id,
                    bit,
                    associativity,
                    self.cfg.replacement,
                    clock,
                    bip_counter,
                )?);
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.misses = 1;
                delta.line_misses = 1;
                counters.add_checked(&delta)?;
                if let Some(next) = next_bip_counter {
                    bip_counter = next;
                }
            } else {
                let value = update
                    .as_mut()
                    .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?
                    .value_mut();
                value.recency = match self.cfg.replacement {
                    L2Replacement::Srrip => 0,
                    _ => clock,
                };
                value.mask |= bit;
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                if counted_hit {
                    delta.hits = 1;
                    if !sector_present && class == L2OpClass::WriteAtomic {
                        delta.write_tag_hits = 1;
                    }
                } else {
                    delta.misses = 1;
                    delta.sector_fill_misses = 1;
                }
                counters.add_checked(&delta)?;
            }

            if counted_hit {
                counted_mask |= bit;
            }
            if sector_present {
                present_mask |= bit;
            }
            line_present = true;
            valid_mask |= bit;
        }

        Ok(PreparedMaskAccess {
            set_key,
            update: update.ok_or(L2AccessError::ReplacementSetEmpty { set_key })?,
            clock,
            bip_counter,
            counters,
            preview: L2MaskPreview {
                sectors: sector_mask.count_ones(),
                counted_hits: counted_mask.count_ones(),
                present_sectors: present_mask.count_ones(),
            },
        })
    }

    /// Predict an ascending walk of `sector_mask` for one line without
    /// changing recency, replacement, validity, or counters. The local mask
    /// tracks sectors validated by earlier bits in the same request, exactly
    /// matching repeated [`Self::access`] calls in ascending sector order.
    pub fn preview_mask(
        &self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
    ) -> Result<L2MaskPreview, L2AccessError> {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        let set_key = self.set_key(line_id)?;
        let resident = self.sets[set_key].iter().find(|w| w.tag == line_id);
        let mut line_present = resident.is_some();
        let mut valid_mask = resident.map_or(0, |w| w.mask);
        let mut counted_mask = 0u8;
        let mut present_mask = 0u8;
        for sector in 0..4u8 {
            let bit = 1u8 << sector;
            if sector_mask & bit == 0 {
                continue;
            }
            let sector_present = line_present && valid_mask & bit != 0;
            if sector_present {
                present_mask |= bit;
            }
            if sector_present || (line_present && class == L2OpClass::WriteAtomic) {
                counted_mask |= bit;
            }
            line_present = true;
            valid_mask |= bit;
        }
        Ok(L2MaskPreview {
            sectors: sector_mask.count_ones(),
            counted_hits: counted_mask.count_ones(),
            present_sectors: present_mask.count_ones(),
        })
    }

    /// One 32 B sector access in global request order. `line_id` is the
    /// global 128 B line address (`sector_id >> 2` in the L1's sector-id
    /// space); negative ids (wire-admissible) take Python's floored
    /// division/remainder. The STATE trajectory is identical for both
    /// [`L2OpClass`]es — the class only changes how a resident-line /
    /// invalid-sector access is COUNTED (Read: sector-fill miss, the
    /// ported Python rule; WriteAtomic: tag-match hit).
    pub fn access(
        &mut self,
        line_id: i64,
        sector: u8,
        class: L2OpClass,
    ) -> Result<L2Outcome, L2AccessError> {
        if sector >= 4 {
            return Err(L2AccessError::SectorOutOfRange { sector });
        }
        let sector_mask = 1u8 << sector;
        let (preview, ()) = self.access_mask_transactional(line_id, sector_mask, class, |_| {
            Ok::<(), L2AccessError>(())
        })?;
        Ok(L2Outcome {
            counted_hit: preview.counted_hits == 1,
            sector_present: preview.present_sectors == 1,
        })
    }

    /// Prepare one ascending mask walk, run all caller-owned fallible work
    /// against its exact outcome preview, and publish the cache changes only
    /// after the callback succeeds. A preparation or callback error leaves
    /// recency, validity, replacement metadata, and counters unchanged.
    pub fn access_mask_transactional<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        before_commit: impl FnOnce(L2MaskPreview) -> Result<T, E>,
    ) -> Result<(L2MaskPreview, T), E>
    where
        E: From<L2AccessError>,
    {
        let prepared = self
            .prepare_mask(line_id, sector_mask, class)
            .map_err(E::from)?;
        let preview = prepared.preview;
        let value = before_commit(preview)?;
        prepared.commit(self);
        Ok((preview, value))
    }
}

struct PreparedMaskAccess {
    set_key: usize,
    update: PreparedSetUpdate,
    clock: i64,
    bip_counter: u64,
    counters: L2Counters,
    preview: L2MaskPreview,
}

impl PreparedMaskAccess {
    fn commit(self, l2: &mut L2Model) {
        self.update.commit(&mut l2.sets[self.set_key]);
        l2.clock = self.clock;
        l2.bip_counter = self.bip_counter;
        l2.counters = self.counters;
    }
}

#[derive(Debug, Clone, Copy)]
enum PreparedSetUpdate {
    Update {
        way: usize,
        value: Way,
    },
    Insert {
        value: Way,
    },
    Replace {
        victim: usize,
        srrip_age: i64,
        value: Way,
    },
}

impl PreparedSetUpdate {
    fn value_mut(&mut self) -> &mut Way {
        match self {
            Self::Update { value, .. } | Self::Insert { value } | Self::Replace { value, .. } => {
                value
            }
        }
    }

    fn commit(self, ways: &mut Vec<Way>) {
        match self {
            Self::Update { way, value } => ways[way] = value,
            Self::Insert { value } => ways.push(value),
            Self::Replace {
                victim,
                srrip_age,
                value,
            } => {
                if srrip_age != 0 {
                    for way in ways.iter_mut() {
                        way.recency += srrip_age;
                    }
                }
                ways[victim] = value;
            }
        }
    }
}

fn validate_sector_mask(sector_mask: u8) -> Result<(), L2AccessError> {
    if sector_mask == 0 || sector_mask & !0b1111 != 0 {
        return Err(L2AccessError::SectorMaskInvalid { sector_mask });
    }
    Ok(())
}

fn validate_srrip_recencies(
    ways: &[Way],
    set_key: usize,
    policy: L2Replacement,
) -> Result<(), L2AccessError> {
    if policy == L2Replacement::Srrip {
        for (way, entry) in ways.iter().enumerate() {
            if !(0..=SRRIP_RRPV_MAX).contains(&entry.recency) {
                return Err(L2AccessError::SrripRecencyOutOfRange {
                    set_key,
                    way,
                    recency: entry.recency,
                });
            }
        }
    }
    Ok(())
}

/// `_install_way` port: prepare insertion or replacement without changing
/// semantic set state. A spare way slot is reserved before the caller's
/// pre-commit callback so the later insertion cannot allocate.
#[allow(clippy::too_many_arguments)]
fn prepare_install(
    ways: &mut Vec<Way>,
    set_key: usize,
    tag: i64,
    sector_bit: u8,
    associativity: usize,
    policy: L2Replacement,
    clock: i64,
    bip_counter: u64,
) -> Result<PreparedSetUpdate, L2AccessError> {
    if ways.len() > associativity {
        return Err(L2AccessError::SetOverfull {
            set_key,
            ways: ways.len(),
            associativity,
        });
    }
    if ways.len() < associativity {
        let recency = insertion_recency(ways, set_key, policy, clock, bip_counter)?;
        ways.try_reserve_exact(1)
            .map_err(|source| L2AccessError::SetStorageAllocation {
                set_key,
                additional: 1,
                source,
            })?;
        return Ok(PreparedSetUpdate::Insert {
            value: Way {
                tag,
                mask: sector_bit,
                recency,
            },
        });
    }
    if ways.is_empty() {
        return Err(L2AccessError::ReplacementSetEmpty { set_key });
    }

    let (victim, srrip_age) = if policy == L2Replacement::Srrip {
        let maximum = ways
            .iter()
            .map(|way| way.recency)
            .max()
            .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
        let age =
            SRRIP_RRPV_MAX
                .checked_sub(maximum)
                .ok_or(L2AccessError::SrripRecencyOutOfRange {
                    set_key,
                    way: 0,
                    recency: maximum,
                })?;
        let victim = ways
            .iter()
            .position(|way| way.recency == maximum)
            .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
        (victim, age)
    } else {
        let victim = ways
            .iter()
            .enumerate()
            .min_by_key(|(index, way)| (way.recency, *index))
            .map(|(index, _)| index)
            .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
        (victim, 0)
    };
    let recency = insertion_recency(ways, set_key, policy, clock, bip_counter)?;
    Ok(PreparedSetUpdate::Replace {
        victim,
        srrip_age,
        value: Way {
            tag,
            mask: sector_bit,
            recency,
        },
    })
}

fn insertion_recency(
    ways: &[Way],
    set_key: usize,
    policy: L2Replacement,
    clock: i64,
    bip_counter: u64,
) -> Result<i64, L2AccessError> {
    match policy {
        L2Replacement::Srrip => Ok(SRRIP_INSERT_RRPV),
        L2Replacement::Bip => {
            let mru = bip_counter.is_multiple_of(BIP_MRU_INSERT_PERIOD);
            if mru || ways.is_empty() {
                Ok(clock)
            } else {
                let minimum = ways
                    .iter()
                    .map(|way| way.recency)
                    .min()
                    .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
                minimum
                    .checked_sub(1)
                    .ok_or(L2AccessError::BipRecencyUnderflow { set_key, minimum })
            }
        }
        L2Replacement::Lru => Ok(clock),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Read-class access sugar: the ported Python rule, where the
    /// counted hit and the physical sector presence coincide.
    fn rd(l2: &mut L2Model, line: i64, sector: u8) -> bool {
        l2.access(line, sector, L2OpClass::Read)
            .unwrap()
            .counted_hit
    }

    /// Write-class access sugar (STG/RED/ATOM): counted hit under the
    /// §6.4 tag-match rule.
    fn wr(l2: &mut L2Model, line: i64, sector: u8) -> bool {
        l2.access(line, sector, L2OpClass::WriteAtomic)
            .unwrap()
            .counted_hit
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct L2State {
        cfg: L2Config,
        num_sets: u64,
        sets: Vec<Vec<Way>>,
        clock: i64,
        bip_counter: u64,
        counters: L2Counters,
    }

    fn state(l2: &L2Model) -> L2State {
        L2State {
            cfg: l2.cfg,
            num_sets: l2.num_sets,
            sets: l2.sets.clone(),
            clock: l2.clock,
            bip_counter: l2.bip_counter,
            counters: l2.counters,
        }
    }

    #[test]
    fn fallible_snapshot_copies_complete_state_and_is_independent() {
        for policy in [L2Replacement::Lru, L2Replacement::Srrip, L2Replacement::Bip] {
            let mut config = cfg(256, 2, 1);
            config.replacement = policy;
            let mut l2 = L2Model::new(config).unwrap();
            rd(&mut l2, 0, 0);
            wr(&mut l2, 1, 1);
            rd(&mut l2, 0, 2);
            // Deliberate spare capacity makes a length-only copy observable.
            l2.sets.reserve_exact(3);
            l2.sets[0].reserve_exact(3);
            let before = state(&l2);

            let mut snapshot = l2.try_snapshot().unwrap();
            assert_eq!(state(&snapshot), before, "{policy:?} snapshot state");
            assert!(snapshot.sets.capacity() >= l2.sets.capacity());
            for (copied, original) in snapshot.sets.iter().zip(&l2.sets) {
                assert!(copied.capacity() >= original.capacity());
            }

            rd(&mut snapshot, 7, 3);
            assert_eq!(state(&l2), before, "{policy:?} source was mutated");
            assert_ne!(state(&snapshot), before, "{policy:?} snapshot is aliased");
        }
    }

    fn assert_mask_error_preserves_state(
        l2: &mut L2Model,
        line: i64,
        sector_mask: u8,
        class: L2OpClass,
        expected: impl FnOnce(&L2AccessError) -> bool,
    ) {
        let before = state(l2);
        let error = l2
            .access_mask_transactional(line, sector_mask, class, |_| Ok::<(), L2AccessError>(()))
            .unwrap_err();
        assert!(expected(&error), "unexpected error: {error:?}");
        assert_eq!(state(l2), before, "failed mask access mutated L2 state");
    }

    #[test]
    fn transactional_mask_decomposition_matches_single_access_state() {
        for policy in [L2Replacement::Lru, L2Replacement::Srrip, L2Replacement::Bip] {
            for class in [L2OpClass::Read, L2OpClass::WriteAtomic] {
                for target_resident in [false, true] {
                    for requested_mask in 1u8..=0b1111 {
                        let mut c = cfg(256, 2, 1);
                        c.replacement = policy;
                        let mut transactional = L2Model::new(c.clone()).unwrap();
                        let mut sequential = L2Model::new(c).unwrap();
                        for l2 in [&mut transactional, &mut sequential] {
                            rd(l2, 0, 0);
                            rd(l2, 1, 0);
                            if target_resident {
                                rd(l2, 0, 0);
                            }
                        }
                        let line = if target_resident { 0 } else { 7 };
                        let (preview, ()) = transactional
                            .access_mask_transactional(line, requested_mask, class, |_| {
                                Ok::<(), L2AccessError>(())
                            })
                            .unwrap();
                        let outcomes = (0..4u8)
                            .filter(|sector| requested_mask & (1 << sector) != 0)
                            .map(|sector| sequential.access(line, sector, class).unwrap())
                            .collect::<Vec<_>>();
                        assert_eq!(
                            preview,
                            L2MaskPreview {
                                sectors: requested_mask.count_ones(),
                                counted_hits: outcomes
                                    .iter()
                                    .map(|outcome| u32::from(outcome.counted_hit))
                                    .sum(),
                                present_sectors: outcomes
                                    .iter()
                                    .map(|outcome| u32::from(outcome.sector_present))
                                    .sum(),
                            },
                            "preview mismatch for {policy:?}/{class:?}, resident={target_resident}, mask={requested_mask:#06b}"
                        );
                        assert_eq!(
                            state(&transactional),
                            state(&sequential),
                            "state mismatch for {policy:?}/{class:?}, resident={target_resident}, mask={requested_mask:#06b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn resident_and_full_set_transactions_reuse_set_storage() {
        for policy in [L2Replacement::Lru, L2Replacement::Srrip, L2Replacement::Bip] {
            let mut c = cfg(256, 2, 1);
            c.replacement = policy;
            let mut l2 = L2Model::new(c).unwrap();
            rd(&mut l2, 0, 0);
            rd(&mut l2, 1, 0);
            let pointer = l2.sets[0].as_ptr();
            let capacity = l2.sets[0].capacity();

            l2.access_mask_transactional(0, 0b1111, L2OpClass::Read, |_| {
                Ok::<(), L2AccessError>(())
            })
            .unwrap();
            assert_eq!(l2.sets[0].as_ptr(), pointer, "{policy:?} resident access");
            assert_eq!(l2.sets[0].capacity(), capacity);

            l2.access_mask_transactional(7, 0b1111, L2OpClass::Read, |_| {
                Ok::<(), L2AccessError>(())
            })
            .unwrap();
            assert_eq!(l2.sets[0].as_ptr(), pointer, "{policy:?} replacement");
            assert_eq!(l2.sets[0].capacity(), capacity);
        }
    }

    #[test]
    fn invalid_masks_and_sectors_are_typed_and_nonmutating() {
        let mut l2 = L2Model::new(cfg(256, 2, 1)).unwrap();
        rd(&mut l2, 0, 0);
        for sector_mask in [0, 0b1_0000, u8::MAX] {
            let before = state(&l2);
            let mut callback_called = false;
            assert!(matches!(
                l2.access_mask_transactional(
                    0,
                    sector_mask,
                    L2OpClass::Read,
                    |_| -> Result<(), L2AccessError> {
                        callback_called = true;
                        Ok(())
                    },
                ),
                Err(L2AccessError::SectorMaskInvalid { sector_mask: got }) if got == sector_mask
            ));
            assert!(!callback_called, "invalid mask invoked pre-commit callback");
            assert_eq!(state(&l2), before);
            assert!(matches!(
                l2.preview_mask(0, sector_mask, L2OpClass::Read),
                Err(L2AccessError::SectorMaskInvalid { sector_mask: got }) if got == sector_mask
            ));
            assert_eq!(state(&l2), before);
        }
        for sector in [4, u8::MAX] {
            let before = state(&l2);
            assert!(matches!(
                l2.access(0, sector, L2OpClass::Read),
                Err(L2AccessError::SectorOutOfRange { sector: got }) if got == sector
            ));
            assert_eq!(state(&l2), before);
        }
        assert_eq!(
            SliceHash::LineMod.slice_of(0, 0),
            Err(L2SetIndexError::SlicesZero)
        );
    }

    #[derive(Debug, PartialEq, Eq)]
    enum CallbackError {
        L2,
        Refused,
    }

    impl From<L2AccessError> for CallbackError {
        fn from(_: L2AccessError) -> Self {
            Self::L2
        }
    }

    #[test]
    fn callback_failure_rolls_back_complete_l2_state() {
        let mut l2 = L2Model::new(cfg(256, 2, 1)).unwrap();
        rd(&mut l2, 0, 0);
        rd(&mut l2, 1, 0);
        let before = state(&l2);
        let mut preview_seen = None;
        assert_eq!(
            l2.access_mask_transactional(7, 0b1111, L2OpClass::Read, |preview| {
                preview_seen = Some(preview);
                Err::<(), _>(CallbackError::Refused)
            }),
            Err(CallbackError::Refused)
        );
        assert_eq!(preview_seen.unwrap().sectors, 4);
        assert_eq!(state(&l2), before);
    }

    #[test]
    fn clock_bip_and_replacement_failures_are_typed_and_transactional() {
        let mut clock = L2Model::new(cfg(256, 2, 1)).unwrap();
        clock.clock = i64::MAX;
        assert_mask_error_preserves_state(&mut clock, 0, 1, L2OpClass::Read, |error| {
            matches!(error, L2AccessError::ClockOverflow { clock: i64::MAX })
        });

        let mut c = cfg(256, 2, 1);
        c.replacement = L2Replacement::Bip;
        let mut bip_counter = L2Model::new(c.clone()).unwrap();
        bip_counter.bip_counter = u64::MAX;
        assert_mask_error_preserves_state(&mut bip_counter, 0, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::BipCounterOverflow { counter: u64::MAX }
            )
        });

        let mut bip_recency = L2Model::new(c).unwrap();
        rd(&mut bip_recency, 0, 0);
        bip_recency.bip_counter = 1;
        bip_recency.sets[0][0].recency = i64::MIN;
        assert_mask_error_preserves_state(&mut bip_recency, 1, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::BipRecencyUnderflow {
                    set_key: 0,
                    minimum: i64::MIN,
                }
            )
        });

        let mut overfull = L2Model::new(cfg(128, 1, 1)).unwrap();
        rd(&mut overfull, 0, 0);
        overfull.sets[0].push(Way {
            tag: 1,
            mask: 1,
            recency: 0,
        });
        assert_mask_error_preserves_state(&mut overfull, 2, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::SetOverfull {
                    set_key: 0,
                    ways: 2,
                    associativity: 1,
                }
            )
        });

        let mut empty = L2Model::new(cfg(128, 1, 1)).unwrap();
        empty.cfg.associativity = 0;
        assert_mask_error_preserves_state(&mut empty, 0, 1, L2OpClass::Read, |error| {
            matches!(error, L2AccessError::ReplacementSetEmpty { set_key: 0 })
        });

        let mut c = cfg(256, 2, 1);
        c.replacement = L2Replacement::Srrip;
        let mut srrip = L2Model::new(c).unwrap();
        rd(&mut srrip, 0, 0);
        rd(&mut srrip, 1, 0);
        srrip.sets[0][0].recency = SRRIP_RRPV_MAX + 1;
        assert_mask_error_preserves_state(&mut srrip, 2, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::SrripRecencyOutOfRange {
                    set_key: 0,
                    recency: 4,
                    ..
                }
            )
        });
    }

    #[test]
    fn access_counter_failures_are_typed_and_transactional() {
        let mut hits = L2Model::new(cfg(256, 2, 1)).unwrap();
        rd(&mut hits, 0, 0);
        hits.counters.hits = u64::MAX;
        assert_mask_error_preserves_state(&mut hits, 0, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::Counter(L2CounterError::Overflow {
                    field: L2CounterField::Hits,
                })
            )
        });

        let mut misses = L2Model::new(cfg(256, 2, 1)).unwrap();
        misses.counters.misses = u64::MAX;
        assert_mask_error_preserves_state(&mut misses, 0, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::Counter(L2CounterError::Overflow {
                    field: L2CounterField::Misses,
                })
            )
        });

        let mut line_misses = L2Model::new(cfg(256, 2, 1)).unwrap();
        line_misses.counters.line_misses = u64::MAX;
        assert_mask_error_preserves_state(&mut line_misses, 0, 1, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::Counter(L2CounterError::Overflow {
                    field: L2CounterField::LineMisses,
                })
            )
        });

        let mut sector_fills = L2Model::new(cfg(256, 2, 1)).unwrap();
        rd(&mut sector_fills, 0, 0);
        sector_fills.counters.sector_fill_misses = u64::MAX;
        assert_mask_error_preserves_state(&mut sector_fills, 0, 0b0010, L2OpClass::Read, |error| {
            matches!(
                error,
                L2AccessError::Counter(L2CounterError::Overflow {
                    field: L2CounterField::SectorFillMisses,
                })
            )
        });

        let mut write_hits = L2Model::new(cfg(256, 2, 1)).unwrap();
        rd(&mut write_hits, 0, 0);
        write_hits.counters.write_tag_hits = u64::MAX;
        assert_mask_error_preserves_state(
            &mut write_hits,
            0,
            0b0010,
            L2OpClass::WriteAtomic,
            |error| {
                matches!(
                    error,
                    L2AccessError::Counter(L2CounterError::Overflow {
                        field: L2CounterField::WriteTagHits,
                    })
                )
            },
        );
    }

    #[test]
    fn mismatched_counter_architecture_fails_public_boundaries_transactionally() {
        let mut l2 = L2Model::new(cfg(256, 2, 1)).unwrap();
        l2.counters = L2Counters::for_arch(ArchId::Sm120);
        let before = state(&l2);

        assert!(matches!(
            l2.access(0, 0, L2OpClass::Read),
            Err(L2AccessError::Counter(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L2 counters",
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                }
            )))
        ));
        assert_eq!(state(&l2), before, "failed access mutated L2 state");

        assert!(matches!(
            l2.preview_mask(0, 1, L2OpClass::Read),
            Err(L2AccessError::Counter(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                    ..
                }
            )))
        ));
        assert!(matches!(
            l2.try_snapshot(),
            Err(L2SnapshotError::Counter(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                    ..
                }
            )))
        ));
        assert!(matches!(
            l2.counters(),
            Err(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                    ..
                }
            ))
        ));
        assert_eq!(state(&l2), before, "read-only boundaries mutated L2 state");
    }

    #[test]
    fn mask_preview_matches_sequential_access_outcomes() {
        for class in [L2OpClass::Read, L2OpClass::WriteAtomic] {
            for seeded_mask in 0u8..=0b1111 {
                for requested_mask in 1u8..=0b1111 {
                    let mut l2 = L2Model::new(cfg(128 * 4, 2, 1)).unwrap();
                    for sector in 0..4u8 {
                        if seeded_mask & (1 << sector) != 0 {
                            l2.access(7, sector, L2OpClass::WriteAtomic).unwrap();
                        }
                    }
                    let before = l2.counters;
                    let preview = l2.preview_mask(7, requested_mask, class).unwrap();
                    let outcomes = (0..4u8)
                        .filter(|sector| requested_mask & (1 << sector) != 0)
                        .map(|sector| l2.access(7, sector, class).unwrap())
                        .collect::<Vec<_>>();
                    let counted_hits = outcomes
                        .iter()
                        .map(|outcome| u32::from(outcome.counted_hit))
                        .sum::<u32>();
                    let present_sectors = outcomes
                        .iter()
                        .map(|outcome| u32::from(outcome.sector_present))
                        .sum::<u32>();
                    assert_eq!(preview.sectors, requested_mask.count_ones());
                    assert_eq!(preview.counted_hits, counted_hits);
                    assert_eq!(preview.present_sectors, present_sectors);
                    assert_eq!(
                        l2.counters.hits.checked_sub(before.hits).unwrap(),
                        u64::from(preview.counted_hits)
                    );
                }
            }
        }
    }

    fn cfg(capacity: u64, assoc: u32, slices: u32) -> L2Config {
        L2Config {
            arch: ArchId::Sm86,
            capacity_bytes: capacity,
            line_bytes: 128,
            sector_bytes: 32,
            num_slices: slices,
            associativity: assoc,
            replacement: L2Replacement::Lru,
            slice_hash: SliceHash::LineMod,
        }
    }

    /// Pinned against the Python registry (`SLICE_HASH_REGISTRY`) for
    /// lines {0, 1, 5, 12345, 987654321, 2^40+7, 2^56+123} at 6 and 7
    /// slices — regenerated verbatim from `sector_cache_replay.py`.
    #[test]
    fn slice_hashes_match_python_pins() {
        let lines: [i64; 7] = [0, 1, 5, 12345, 987_654_321, (1 << 40) + 7, (1 << 56) + 123];
        let cases: [(SliceHash, [u32; 7], [u32; 7]); 6] = [
            (
                SliceHash::LineMod,
                [0, 1, 5, 3, 3, 5, 1],
                [0, 1, 5, 4, 3, 2, 1],
            ),
            (
                SliceHash::SectorMod,
                [0, 4, 2, 0, 0, 2, 4],
                [0, 4, 6, 2, 5, 1, 4],
            ),
            (
                SliceHash::XorFolded,
                [0, 1, 5, 2, 5, 3, 2],
                [0, 1, 5, 1, 4, 3, 0],
            ),
            (
                SliceHash::KnuthMult,
                [0, 1, 5, 3, 3, 3, 5],
                [0, 5, 6, 3, 3, 5, 3],
            ),
            (
                SliceHash::ShiftMod,
                [0, 0, 2, 4, 4, 5, 3],
                [0, 0, 2, 5, 1, 4, 0],
            ),
            (
                SliceHash::BitReverse,
                [0, 2, 4, 0, 4, 2, 0],
                [0, 1, 3, 6, 5, 0, 6],
            ),
        ];
        for (hash, at6, at7) in cases {
            for (i, &line) in lines.iter().enumerate() {
                assert_eq!(
                    hash.slice_of(line, 6).unwrap(),
                    at6[i],
                    "{hash:?}({line}, 6)"
                );
                assert_eq!(
                    hash.slice_of(line, 7).unwrap(),
                    at7[i],
                    "{hash:?}({line}, 7)"
                );
            }
        }
    }

    /// NEGATIVE line ids (wire-admissible via base_delta streams summing
    /// below zero) must replay with Python's floored semantics, never a
    /// u64 wrap — pins regenerated from `sector_cache_replay.py` for
    /// lines {-1, -93, -12345, -(2^40)-7}.
    #[test]
    fn negative_line_ids_match_python_floored_semantics() {
        let lines: [i64; 4] = [-1, -93, -12345, -(1 << 40) - 7];
        let cases: [(SliceHash, [u32; 4], [u32; 4]); 6] = [
            (SliceHash::LineMod, [5, 3, 3, 1], [6, 5, 3, 5]),
            (SliceHash::SectorMod, [2, 0, 0, 4], [3, 6, 5, 6]),
            (SliceHash::XorFolded, [5, 1, 2, 3], [6, 3, 4, 4]),
            (SliceHash::KnuthMult, [3, 1, 1, 1], [6, 5, 1, 6]),
            (SliceHash::ShiftMod, [5, 1, 1, 0], [6, 2, 1, 2]),
            (SliceHash::BitReverse, [3, 5, 5, 3], [1, 0, 3, 2]),
        ];
        for (hash, at6, at7) in cases {
            for (i, &line) in lines.iter().enumerate() {
                assert_eq!(
                    hash.slice_of(line, 6).unwrap(),
                    at6[i],
                    "{hash:?}({line}, 6)"
                );
                assert_eq!(
                    hash.slice_of(line, 7).unwrap(),
                    at7[i],
                    "{hash:?}({line}, 7)"
                );
            }
        }
        // Set mapping floors too: Python (-93 // 6) % 512 == 496.
        let mut l2 = L2Model::new(cfg(6_291_456, 16, 6)).unwrap();
        assert!(!rd(&mut l2, -93, 1), "cold negative line");
        assert!(rd(&mut l2, -93, 1), "tagged and re-hit, no wrap");
    }

    /// Pinned against Python `CacheReplayConfig` derived properties.
    #[test]
    fn geometry_derivation_matches_python() {
        let c = cfg(6_291_456, 16, 6);
        assert_eq!(c.total_lines().unwrap(), 49_152);
        assert_eq!(c.num_sets_per_slice().unwrap(), 512);
        // Tiny degenerate config exercises both max(1, ·) clamps.
        let tiny = cfg(1024, 16, 6);
        assert_eq!(tiny.total_lines().unwrap(), 8);
        assert_eq!(tiny.num_sets_per_slice().unwrap(), 1);
    }

    #[test]
    fn invalid_geometry_and_set_storage_fail_typed() {
        let mut zero_line = cfg(1024, 16, 6);
        zero_line.line_bytes = 0;
        assert!(matches!(
            zero_line.total_lines(),
            Err(L2ConfigError::UnsupportedGranularity {
                line: 0,
                sector: 32,
            })
        ));
        assert!(matches!(
            L2Model::new(zero_line),
            Err(L2BuildError::Config(
                L2ConfigError::UnsupportedGranularity {
                    line: 0,
                    sector: 32,
                }
            ))
        ));

        let mut zero_slices = cfg(1024, 16, 1);
        zero_slices.num_slices = 0;
        assert!(matches!(
            zero_slices.num_sets_per_slice(),
            Err(L2ConfigError::SlicesZero)
        ));

        let mut zero_assoc = cfg(1024, 1, 1);
        zero_assoc.associativity = 0;
        assert!(matches!(
            zero_assoc.num_sets_per_slice(),
            Err(L2ConfigError::AssociativityZero)
        ));

        assert!(matches!(
            checked_set_count(u32::MAX, u64::MAX),
            Err(L2BuildError::SetCountOverflow {
                num_slices: u32::MAX,
                sets_per_slice: u64::MAX,
            })
        ));
        assert!(matches!(
            allocate_set_storage(usize::MAX),
            Err(L2BuildError::SetStorageAllocation {
                set_count: usize::MAX,
                ..
            })
        ));
    }

    #[test]
    fn carded_constructor_applies_documented_defaults() {
        let policy = Some(L2_WRITE_ALLOCATE_POLICY);
        let c = L2Config::from_carded_geometry("sm_86", 6_291_456, 128, 32, 6, policy).unwrap();
        assert_eq!(c.associativity, DOCUMENTED_DEFAULT_L2_ASSOCIATIVITY);
        assert_eq!(c.replacement, L2Replacement::Lru);
        assert_eq!(c.slice_hash, SliceHash::LineMod);
        assert!(matches!(
            L2Config::from_carded_geometry("sm_86", 6_291_456, 256, 32, 6, policy),
            Err(L2ConfigError::UnsupportedGranularity { .. })
        ));
        assert!(matches!(
            L2Config::from_carded_geometry("sm_86", 64, 128, 32, 6, policy),
            Err(L2ConfigError::CapacityBelowLine { .. })
        ));
    }

    /// Fail-closed gates on the carded constructor: an arch that never
    /// reviewed the documented defaults, and a card whose write policy
    /// is a gap (sm_120-laptop ships `policy: null`) or an unknown
    /// token, must all be typed errors — the store path write-allocates
    /// and must never assume it.
    #[test]
    fn carded_constructor_fails_closed_on_arch_and_policy() {
        let policy = Some(L2_WRITE_ALLOCATE_POLICY);
        assert!(matches!(
            L2Config::from_carded_geometry("sm_120", 33_554_432, 128, 32, 12, policy),
            Err(L2ConfigError::DefaultsUnreviewedForArch { .. })
        ));
        assert!(matches!(
            L2Config::from_carded_geometry("sm_86", 6_291_456, 128, 32, 6, None),
            Err(L2ConfigError::WritePolicyGap)
        ));
        assert!(matches!(
            L2Config::from_carded_geometry("sm_86", 6_291_456, 128, 32, 6, Some("write_back")),
            Err(L2ConfigError::WritePolicyUnsupported { .. })
        ));
    }

    #[test]
    fn sector_hit_requires_line_and_sector() {
        let mut l2 = L2Model::new(cfg(1024, 2, 1)).unwrap();
        assert!(!rd(&mut l2, 0, 0), "cold line miss");
        assert!(rd(&mut l2, 0, 0), "sector now valid");
        assert!(!rd(&mut l2, 0, 1), "resident line, new sector");
        assert!(rd(&mut l2, 0, 1));
        assert_eq!(l2.counters.line_misses, 1);
        assert_eq!(l2.counters.sector_fill_misses, 1);
        assert_eq!(l2.counters.hits, 2);
        assert_eq!(l2.counters.total().unwrap(), 4);
    }

    #[test]
    fn lru_evicts_least_recent_line() {
        // 1 slice, 1 set (capacity 256 B / 128 B line / assoc 2), LRU.
        let mut l2 = L2Model::new(cfg(256, 2, 1)).unwrap();
        rd(&mut l2, 0, 0);
        rd(&mut l2, 1, 0);
        rd(&mut l2, 0, 0); // line 0 → MRU
        rd(&mut l2, 2, 0); // evicts line 1
        assert!(rd(&mut l2, 0, 0), "line 0 survived");
        assert!(!rd(&mut l2, 1, 0), "line 1 was evicted");
    }

    #[test]
    fn eviction_drops_sector_residency_with_the_line() {
        let mut l2 = L2Model::new(cfg(128, 1, 1)).unwrap();
        rd(&mut l2, 0, 0);
        rd(&mut l2, 0, 1);
        rd(&mut l2, 9, 0); // same set (1 set total) → evict
        assert!(!rd(&mut l2, 0, 1), "sector died with the line");
        assert_eq!(l2.counters.line_misses, 3);
    }

    #[test]
    fn set_mapping_matches_python_form() {
        // 1 slice, 2 sets, assoc 1 (capacity 256, line 128, assoc 1):
        // lines 0 and 2 share set 0; line 1 lives in set 1.
        let mut l2 = L2Model::new(cfg(256, 1, 1)).unwrap();
        rd(&mut l2, 0, 0);
        rd(&mut l2, 1, 0);
        rd(&mut l2, 2, 0); // evicts line 0, not line 1
        assert!(rd(&mut l2, 1, 0), "set 1 undisturbed");
        assert!(!rd(&mut l2, 0, 0), "conflict-evicted");
    }

    #[test]
    fn srrip_ages_and_reinserts_at_long_interval() {
        let mut c = cfg(256, 2, 1);
        c.replacement = L2Replacement::Srrip;
        let mut l2 = L2Model::new(c).unwrap();
        rd(&mut l2, 0, 0); // insert rrpv=2
        rd(&mut l2, 2, 0); // insert rrpv=2 (same set)
        rd(&mut l2, 0, 0); // hit → rrpv=0
        // New line: no way at max → age all (0→1, 2→3), evict line 2.
        rd(&mut l2, 4, 0);
        assert!(rd(&mut l2, 0, 0), "recently-hit line survives");
        assert!(!rd(&mut l2, 2, 0), "aged line evicted");
    }

    #[test]
    fn bip_inserts_below_lru_except_periodically() {
        let mut c = cfg(256, 2, 1);
        c.replacement = L2Replacement::Bip;
        let mut l2 = L2Model::new(c).unwrap();
        // bip_counter=0 → first install at MRU; the next 31 at LRU-1.
        rd(&mut l2, 0, 0); // MRU insert (counter 0)
        rd(&mut l2, 2, 0); // below line 0 (counter 1)
        rd(&mut l2, 4, 0); // evicts line 2 (lowest recency)
        assert!(rd(&mut l2, 0, 0), "MRU-inserted line survives");
        assert!(!rd(&mut l2, 2, 0), "LRU-inserted line evicted");
    }

    /// Cross-language lockstep corpus (BL-20260507 pattern): the SAME
    /// deterministic LCG streams, replayed through the canonical Python
    /// engine, produce these (hits, total) pins —
    /// `tools/cache_replay/test_l2_rust_parity.py` asserts the Python
    /// side AND that these rows exist here. Regenerate BOTH files in
    /// one commit.
    #[test]
    fn python_parity_corpus_pins() {
        // (policy, hash, seed, capacity, assoc, slices, n, line_space,
        //  signed, hits, total)
        type Row = (
            L2Replacement,
            SliceHash,
            u64,
            u64,
            u32,
            u32,
            u64,
            i64,
            bool,
            u64,
            u64,
        );
        #[rustfmt::skip]
        let corpus: [Row; 7] = [
            (L2Replacement::Lru, SliceHash::LineMod, 1, 16384, 4, 3, 500, 128, false, 159, 500),
            (L2Replacement::Lru, SliceHash::KnuthMult, 2, 16384, 4, 3, 500, 128, false, 164, 500),
            (L2Replacement::Srrip, SliceHash::XorFolded, 3, 16384, 4, 3, 500, 128, false, 150, 500),
            (L2Replacement::Srrip, SliceHash::BitReverse, 4, 4096, 2, 2, 500, 256, false, 6, 500),
            (L2Replacement::Bip, SliceHash::SectorMod, 5, 16384, 4, 3, 500, 128, false, 181, 500),
            (L2Replacement::Bip, SliceHash::ShiftMod, 6, 4096, 2, 2, 500, 256, false, 11, 500),
            (L2Replacement::Lru, SliceHash::LineMod, 7, 4096, 2, 3, 400, 96, true, 14, 400),
        ];
        for (policy, hash, seed, cap, assoc, slices, n, space, signed, hits, total) in corpus {
            let mut l2 = L2Model::new(L2Config {
                arch: ArchId::Sm86,
                capacity_bytes: cap,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: slices,
                associativity: assoc,
                replacement: policy,
                slice_hash: hash,
            })
            .unwrap();
            let mut x: u64 = seed;
            for _ in 0..n {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let mut line = ((x >> 20) % space as u64) as i64;
                if signed && (x & 1) == 1 {
                    line = -line;
                }
                rd(&mut l2, line, ((x >> 10) & 3) as u8);
            }
            assert_eq!(
                (l2.counters.hits, l2.counters.total().unwrap()),
                (hits, total),
                "corpus row seed={seed} ({policy:?}, {hash:?})"
            );
        }
    }

    #[test]
    fn stores_allocate_like_loads() {
        let mut l2 = L2Model::new(cfg(1024, 2, 1)).unwrap();
        assert!(!wr(&mut l2, 3, 2), "cold store misses (no tag to match)");
        assert!(
            rd(&mut l2, 3, 2),
            "write-allocate: the store's line services a later load"
        );
        assert_eq!(l2.counters.line_misses, 1);
        assert_eq!(l2.counters.write_tag_hits, 0, "cold path grants no tag hit");
    }

    /// The §6.4 per-access-class rule: a write-class access to a
    /// RESIDENT line hits on the tag alone, even when the sector's
    /// valid bit is clear — and validates the sector (the write carries
    /// its own data), so a later read of that sector hits too. The same
    /// interleaving under Read class stays a sector-fill miss: the STATE
    /// trajectory is identical, only the counting differs.
    #[test]
    fn write_class_hits_on_tag_match_and_validates_the_sector() {
        let mut l2 = L2Model::new(cfg(1024, 2, 1)).unwrap();
        assert!(!wr(&mut l2, 3, 0), "cold write line-misses");
        assert!(
            wr(&mut l2, 3, 2),
            "write to a NEW sector of the resident line: tag-match hit"
        );
        assert_eq!(l2.counters.write_tag_hits, 1);
        assert_eq!(
            l2.counters.sector_fill_misses, 0,
            "the tag-match hit is not double-counted as a fill miss"
        );
        assert!(
            rd(&mut l2, 3, 2),
            "the write validated the sector — a later read hits"
        );
        // Physics stays honest for pricing: a tag-match hit to a fresh
        // sector reports it ABSENT before the access (an atomic reply
        // would still fetch from DRAM), while a re-read finds it present.
        let out = l2.access(3, 3, L2OpClass::WriteAtomic).unwrap();
        assert!(out.counted_hit && !out.sector_present);
        let out = l2.access(3, 3, L2OpClass::Read).unwrap();
        assert!(out.counted_hit && out.sector_present);
        // Mirror stream under Read class: same state, different counts.
        let mut r = L2Model::new(cfg(1024, 2, 1)).unwrap();
        assert!(!rd(&mut r, 3, 0));
        assert!(!rd(&mut r, 3, 2), "read of the invalid sector fill-misses");
        assert_eq!(r.counters.sector_fill_misses, 1);
        assert_eq!(r.counters.write_tag_hits, 0);
        assert!(rd(&mut r, 3, 2));
    }

    #[test]
    fn counter_aggregation_commits_every_field_and_total() {
        let mut total = L2Counters::for_arch(ArchId::Sm86);
        total.hits = 1;
        total.misses = 2;
        total.line_misses = 3;
        total.sector_fill_misses = 4;
        total.write_tag_hits = 5;

        let mut addend = L2Counters::for_arch(ArchId::Sm86);
        addend.hits = 10;
        addend.misses = 11;
        addend.line_misses = 12;
        addend.sector_fill_misses = 13;
        addend.write_tag_hits = 14;

        total.add_checked(&addend).unwrap();
        assert_eq!(total.arch(), Some(ArchId::Sm86));
        assert_eq!(total.hits, 11);
        assert_eq!(total.misses, 13);
        assert_eq!(total.line_misses, 15);
        assert_eq!(total.sector_fill_misses, 17);
        assert_eq!(total.write_tag_hits, 19);
        assert_eq!(total.total().unwrap(), 24);
    }

    #[test]
    fn counter_aggregation_overflow_is_typed_and_transactional() {
        type Setter = fn(&mut L2Counters, u64);
        let cases: [(L2CounterField, Setter); 5] = [
            (L2CounterField::Hits, |c, v| c.hits = v),
            (L2CounterField::Misses, |c, v| c.misses = v),
            (L2CounterField::LineMisses, |c, v| c.line_misses = v),
            (L2CounterField::SectorFillMisses, |c, v| {
                c.sector_fill_misses = v
            }),
            (L2CounterField::WriteTagHits, |c, v| c.write_tag_hits = v),
        ];
        for (field, set) in cases {
            let mut total = L2Counters::for_arch(ArchId::Sm86);
            total.hits = 7;
            set(&mut total, u64::MAX);
            let before = total;
            let mut addend = L2Counters::for_arch(ArchId::Sm86);
            addend.misses = 11;
            set(&mut addend, 1);
            assert!(matches!(
                total.add_checked(&addend),
                Err(L2CounterError::Overflow { field: got }) if got == field
            ));
            assert_eq!(total, before, "overflow in {field} partially committed");
        }
    }

    #[test]
    fn counter_total_overflow_is_typed() {
        let mut counters = L2Counters::for_arch(ArchId::Sm86);
        counters.hits = u64::MAX;
        counters.misses = 1;
        assert_eq!(
            counters.total(),
            Err(L2CounterError::TotalOverflow {
                hits: u64::MAX,
                misses: 1,
            })
        );
    }

    #[test]
    fn counter_error_precedence_and_maximum_total_are_pinned() {
        let mut unbound = L2Counters::default();
        unbound.hits = u64::MAX;
        let mut overflowing = L2Counters::for_arch(ArchId::Sm86);
        overflowing.hits = 1;
        let destination_precedence_addends = [
            L2Counters::default(),
            L2Counters::for_arch(ArchId::Sm120),
            overflowing,
        ];
        for addend in destination_precedence_addends {
            let before = unbound;
            assert!(matches!(
                unbound.add_checked(&addend),
                Err(L2CounterError::Architecture(
                    sim_arch::ArchitectureProvenanceError::Unbound {
                        component: "L2 counters",
                    }
                ))
            ));
            assert_eq!(unbound, before, "destination binding must fail first");
        }

        let mut bound = L2Counters::for_arch(ArchId::Sm86);
        bound.hits = u64::MAX;

        let mut unbound_addend = L2Counters::default();
        unbound_addend.hits = 1;
        let before = bound;
        assert!(matches!(
            bound.add_checked(&unbound_addend),
            Err(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Unbound {
                    component: "L2 counters",
                }
            ))
        ));
        assert_eq!(bound, before, "addend binding must precede overflow");

        let mut wrong_arch = L2Counters::for_arch(ArchId::Sm120);
        wrong_arch.hits = 1;
        assert!(matches!(
            bound.add_checked(&wrong_arch),
            Err(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L2 counters",
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                }
            ))
        ));
        assert_eq!(bound, before, "architecture mismatch must precede overflow");

        type Setter = fn(&mut L2Counters, u64);
        let order: [(L2CounterField, Setter); 5] = [
            (L2CounterField::Hits, |c, v| c.hits = v),
            (L2CounterField::Misses, |c, v| c.misses = v),
            (L2CounterField::LineMisses, |c, v| c.line_misses = v),
            (L2CounterField::SectorFillMisses, |c, v| {
                c.sector_fill_misses = v
            }),
            (L2CounterField::WriteTagHits, |c, v| c.write_tag_hits = v),
        ];
        for (first_overflow, &(expected, _)) in order.iter().enumerate() {
            let mut total = L2Counters::for_arch(ArchId::Sm86);
            let mut addend = L2Counters::for_arch(ArchId::Sm86);
            for &(_, set) in &order {
                set(&mut total, u64::MAX);
                set(&mut addend, 1);
            }
            for &(_, set) in &order[..first_overflow] {
                set(&mut total, 0);
            }
            let before = total;
            assert!(matches!(
                total.add_checked(&addend),
                Err(L2CounterError::Overflow { field }) if field == expected
            ));
            assert_eq!(total, before, "overflow order at {expected} mutated state");
        }

        for (hits, misses) in [(u64::MAX, 0), (0, u64::MAX)] {
            let mut counters = L2Counters::for_arch(ArchId::Sm86);
            counters.hits = hits;
            counters.misses = misses;
            assert_eq!(counters.total(), Ok(u64::MAX));
        }
    }

    #[test]
    fn counter_aggregation_rejects_mixed_architecture_transactionally() {
        let mut sm86 = L2Counters::for_arch(ArchId::Sm86);
        sm86.hits = 7;
        let before = sm86;
        let sm120 = L2Counters::for_arch(ArchId::Sm120);
        assert!(matches!(
            sm86.add_checked(&sm120),
            Err(L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L2 counters",
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                }
            ))
        ));
        assert_eq!(sm86, before);
    }
}
