//! Shared L2 slice model — a lockstep port of the sectored set-associative
//! replay core in `the native L2 replay model`
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
//!   sector-fill miss for reads. The immediate API sets the bit during the
//!   access to preserve Python parity; the event API keeps it pending until an
//!   explicit generation-bound completion. Writes validate immediately in both
//!   disciplines because they carry the sector data;
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
//! slice hash and replacement policy resolve through the deployment
//! architecture's `sim-arch` capability. These remain documented hypotheses
//! to be tested by microbenchmarks—never by same-cell label fits—and an
//! architecture without a model receives a typed capability error.
//!
//! Fill timing: the immediate API remains an occupancy-only Python-parity
//! surface. A separate event discipline tracks pending sectors and makes them
//! valid only through opaque, allocation-generation-bound completion tokens.
//! Pending reads always remain lookup misses. The compatibility event API mints
//! a distinct fetch, while the coalesced event API splits newly fetched sectors
//! from allocation/sector-matched pending merges so the system can reuse the
//! original ready time. This is visibility/ABA state, not an invented finite L2
//! MSHR capacity.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use sim_arch::{ArchError, ArchId, CapabilityError, L2Replacement as ArchL2Replacement};
use sim_semantics::{
    AddressDomain, CanonicalLineKey, L2AccessDisciplineState, L2AllocationState, L2BoundaryState,
    L2CounterState, L2GeometryState, L2SectorOwnership, L2SectorStatus, L2SetState, L2StateError,
    L2WayState, TYPED_STATE_SCHEMA_VERSION,
};

/// Number of 32 B sectors per 128 B line (fixed physical granularity;
/// the `u8` way masks below hold exactly these four bits).
pub const L2_SECTORS_PER_LINE: u32 = 4;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceHash {
    /// Documented-default candidate (v0 approximation).
    LineMod,
    SectorMod,
    XorFolded,
    KnuthMult,
    ShiftMod,
    BitReverse,
}

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

/// Replacement policy. LRU is the sm_86 approximation selected
/// through `sim-arch`; SRRIP/BIP are ported so the lockstep differential covers
/// the full Python surface and a future carded policy can flip the knob without
/// a re-port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum L2Replacement {
    Lru,
    Srrip,
    Bip,
}

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
        "line/sector geometry must match the configured {expected_line}/{expected_sector} layout \
         (got line={line}, sector={sector})"
    )]
    UnsupportedGranularity {
        line: u32,
        sector: u32,
        expected_line: u32,
        expected_sector: u32,
    },
    #[error("associativity must be >= 1")]
    AssociativityZero,
    #[error("num_slices must be >= 1")]
    SlicesZero,
    #[error("effective L2 scenarios are supported only for sm_120, got {arch}")]
    EffectiveArchitectureUnsupported { arch: ArchId },
    #[error("effective L2 capacity {capacity}B exceeds the bounded scenario limit {maximum}B")]
    EffectiveCapacityTooLarge { capacity: u64, maximum: u64 },
    #[error(
        "effective L2 geometry allocates {sets} sets, exceeding the bounded scenario limit {maximum}"
    )]
    EffectiveSetCountTooLarge { sets: u64, maximum: u64 },
    #[error("effective L2 scenarios require the implemented LRU/line-mod policy")]
    EffectivePolicyUnsupported,
    #[error(
        "card L2 geometry has non-integral sets per slice: capacity={capacity}B, line={line}B, \
         slices={slices}, associativity={associativity}"
    )]
    NonIntegralSetCount {
        capacity: u64,
        line: u32,
        slices: u32,
        associativity: u32,
    },
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
    #[error("process-local L2 model-owner identity space is exhausted")]
    ModelOwnerExhausted,
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

#[derive(Debug, thiserror::Error)]
pub enum L2ProjectionError {
    #[error("typed L2 projection is invalid: {0}")]
    State(#[from] L2StateError),
    #[error("typed L2 projection arithmetic overflowed")]
    Overflow,
}

/// The one write policy the store path implements (stores allocate in
/// L2 like loads). Cards carrying anything else — or nothing — fail
/// closed at construction.
pub const L2_WRITE_ALLOCATE_POLICY: &str = "write_allocate_non_inclusive";

/// The effective L2 model allocates one host-side `Vec` per set.  Keep the
/// scenario surface explicitly bounded in addition to bounding byte capacity;
/// a large number of slices/ways can otherwise exhaust the host even when the
/// nominal byte count looks moderate.
const MAX_EFFECTIVE_CAPACITY_BYTES: u64 = 1 << 40;
const MAX_EFFECTIVE_TOTAL_SETS: u64 = 4_194_304;

/// Provenance of an L2 configuration. Carded geometry is a physical card
/// declaration; effective geometry is an explicit scenario and must never be
/// reported as recovered SM120 hardware truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2ConfigOrigin {
    Carded,
    Effective,
}

/// L2 geometry + policy. Construct via [`L2Config::from_carded_geometry`] or
/// [`L2Config::from_effective_geometry_for_arch`] so provenance is explicit at
/// the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Config {
    /// Deployment architecture whose defaults/geometry this config carries.
    pub arch: ArchId,
    pub capacity_bytes: u64,
    pub line_bytes: u32,
    pub sector_bytes: u32,
    pub num_slices: u32,
    pub associativity: u32,
    pub replacement: L2Replacement,
    pub slice_hash: SliceHash,
    pub origin: L2ConfigOrigin,
}

impl L2Config {
    /// Typed architecture entry point. Architecture defaults are obtained from
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
        let defaults = arch.l2_defaults().require("l2_defaults")?;
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
            origin: L2ConfigOrigin::Carded,
        };
        cfg.validate()?;
        // Carded geometry is a physical declaration, so preserve the
        // set-associative contract used by the Python reference validator.
        // The general L2Config path intentionally retains its historical
        // max(1, floor(.)) behavior for analytical/degenerate inputs, but
        // a card must never silently discard capacity while deriving sets.
        let bytes_per_set = cfg.line_bytes.checked_mul(cfg.associativity).ok_or(
            L2ConfigError::NonIntegralSetCount {
                capacity: cfg.capacity_bytes,
                line: cfg.line_bytes,
                slices: cfg.num_slices,
                associativity: cfg.associativity,
            },
        )?;
        let slices = u64::from(cfg.num_slices);
        let bytes_per_slice = cfg.capacity_bytes / slices;
        if !cfg.capacity_bytes.is_multiple_of(slices)
            || !bytes_per_slice.is_multiple_of(u64::from(bytes_per_set))
        {
            return Err(L2ConfigError::NonIntegralSetCount {
                capacity: cfg.capacity_bytes,
                line: cfg.line_bytes,
                slices: cfg.num_slices,
                associativity: cfg.associativity,
            });
        }
        Ok(cfg)
    }

    /// Construct a deliberately explicit effective geometry. This path does
    /// not consult `ArchId::l2_defaults()` and therefore can represent an
    /// SM120 scenario while retaining its assumptions as `origin=Effective`.
    /// The current model supports the documented 128-byte/32-byte granularity,
    /// LRU replacement, and line-mod slice hypothesis only.
    pub fn from_effective_geometry_for_arch(
        arch: ArchId,
        capacity_bytes: u64,
        line_bytes: u32,
        sector_bytes: u32,
        num_slices: u32,
        associativity: u32,
        replacement: L2Replacement,
        slice_hash: SliceHash,
    ) -> Result<Self, L2ConfigError> {
        if arch != ArchId::Sm120 {
            return Err(L2ConfigError::EffectiveArchitectureUnsupported { arch });
        }
        if replacement != L2Replacement::Lru || slice_hash != SliceHash::LineMod {
            return Err(L2ConfigError::EffectivePolicyUnsupported);
        }
        let cfg = Self {
            arch,
            capacity_bytes,
            line_bytes,
            sector_bytes,
            num_slices,
            associativity,
            replacement,
            slice_hash,
            origin: L2ConfigOrigin::Effective,
        };
        cfg.validate()?;
        // Keep the scenario bounded and integral.  The cache model allocates
        // one vector per set, so accepting a floored or unbounded geometry
        // would silently change the declared capacity or exhaust the host.
        if capacity_bytes > MAX_EFFECTIVE_CAPACITY_BYTES {
            return Err(L2ConfigError::EffectiveCapacityTooLarge {
                capacity: capacity_bytes,
                maximum: MAX_EFFECTIVE_CAPACITY_BYTES,
            });
        }
        let bytes_per_set = u64::from(line_bytes)
            .checked_mul(u64::from(associativity))
            .ok_or(L2ConfigError::NonIntegralSetCount {
                capacity: capacity_bytes,
                line: line_bytes,
                slices: num_slices,
                associativity,
            })?;
        let bytes_per_slice = capacity_bytes
            .checked_div(u64::from(num_slices))
            .ok_or(L2ConfigError::SlicesZero)?;
        if !capacity_bytes.is_multiple_of(u64::from(num_slices))
            || !bytes_per_slice.is_multiple_of(bytes_per_set)
        {
            return Err(L2ConfigError::NonIntegralSetCount {
                capacity: capacity_bytes,
                line: line_bytes,
                slices: num_slices,
                associativity,
            });
        }
        let sets_per_slice = bytes_per_slice / bytes_per_set;
        let total_sets = sets_per_slice.checked_mul(u64::from(num_slices)).ok_or(
            L2ConfigError::EffectiveSetCountTooLarge {
                sets: u64::MAX,
                maximum: MAX_EFFECTIVE_TOTAL_SETS,
            },
        )?;
        if total_sets > MAX_EFFECTIVE_TOTAL_SETS {
            return Err(L2ConfigError::EffectiveSetCountTooLarge {
                sets: total_sets,
                maximum: MAX_EFFECTIVE_TOTAL_SETS,
            });
        }
        Ok(cfg)
    }

    pub const fn is_effective(&self) -> bool {
        matches!(self.origin, L2ConfigOrigin::Effective)
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
        if self.is_effective() {
            if self.arch != ArchId::Sm120 {
                return Err(L2ConfigError::EffectiveArchitectureUnsupported { arch: self.arch });
            }
            if self.replacement != L2Replacement::Lru || self.slice_hash != SliceHash::LineMod {
                return Err(L2ConfigError::EffectivePolicyUnsupported);
            }
        }
        let (expected_line, expected_sector) = if self.is_effective() {
            (128, 32)
        } else {
            let defaults = self.arch.l2_defaults().require("l2_defaults")?;
            (defaults.line_bytes, defaults.sector_bytes)
        };
        if self.line_bytes != expected_line || self.sector_bytes != expected_sector {
            return Err(L2ConfigError::UnsupportedGranularity {
                line: self.line_bytes,
                sector: self.sector_bytes,
                expected_line,
                expected_sector,
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
        if self.is_effective() {
            if self.capacity_bytes > MAX_EFFECTIVE_CAPACITY_BYTES {
                return Err(L2ConfigError::EffectiveCapacityTooLarge {
                    capacity: self.capacity_bytes,
                    maximum: MAX_EFFECTIVE_CAPACITY_BYTES,
                });
            }
            let bytes_per_set = u64::from(self.line_bytes)
                .checked_mul(u64::from(self.associativity))
                .ok_or(L2ConfigError::NonIntegralSetCount {
                    capacity: self.capacity_bytes,
                    line: self.line_bytes,
                    slices: self.num_slices,
                    associativity: self.associativity,
                })?;
            let bytes_per_slice = self
                .capacity_bytes
                .checked_div(u64::from(self.num_slices))
                .ok_or(L2ConfigError::SlicesZero)?;
            if !self
                .capacity_bytes
                .is_multiple_of(u64::from(self.num_slices))
                || !bytes_per_slice.is_multiple_of(bytes_per_set)
            {
                return Err(L2ConfigError::NonIntegralSetCount {
                    capacity: self.capacity_bytes,
                    line: self.line_bytes,
                    slices: self.num_slices,
                    associativity: self.associativity,
                });
            }
            let total_sets = (bytes_per_slice / bytes_per_set)
                .checked_mul(u64::from(self.num_slices))
                .ok_or(L2ConfigError::EffectiveSetCountTooLarge {
                    sets: u64::MAX,
                    maximum: MAX_EFFECTIVE_TOTAL_SETS,
                })?;
            if total_sets > MAX_EFFECTIVE_TOTAL_SETS {
                return Err(L2ConfigError::EffectiveSetCountTooLarge {
                    sets: total_sets,
                    maximum: MAX_EFFECTIVE_TOTAL_SETS,
                });
            }
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
    /// A store sector (card policy `write_allocate_non_inclusive` → stores
    /// allocate in L2 exactly like loads).
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
    /// Line-relative requested-sector bits whose access counts as an L2 hit.
    pub counted_hit_mask: u8,
    /// Line-relative requested-sector bits whose data was physically present
    /// before its access.
    pub present_mask: u8,
}

impl L2MaskPreview {
    pub fn all_present(self) -> bool {
        self.present_sectors == self.sectors
    }
}

/// Process-local identity of one L2 model lineage. Snapshots retain the owner
/// because they are transactional continuations of the same model; unrelated
/// models cannot accept each other's fill capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct L2ModelOwner(u64);

static NEXT_L2_MODEL_OWNER: AtomicU64 = AtomicU64::new(1);

fn allocate_l2_model_owner_from(next: &AtomicU64) -> Result<L2ModelOwner, L2BuildError> {
    next.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |current| match current {
            0 => None,
            u64::MAX => Some(0),
            _ => current.checked_add(1),
        },
    )
    .map(L2ModelOwner)
    .map_err(|_| L2BuildError::ModelOwnerExhausted)
}

fn allocate_l2_model_owner() -> Result<L2ModelOwner, L2BuildError> {
    allocate_l2_model_owner_from(&NEXT_L2_MODEL_OWNER)
}

/// Private identity of one event-discipline line allocation. Reinstalling the
/// same tag receives a fresh generation, preventing stale/ABA completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct L2AllocationGeneration(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextAllocationGeneration {
    Available(u64),
    Exhausted,
}

impl NextAllocationGeneration {
    fn allocate(self) -> Result<(L2AllocationGeneration, NextAllocationGeneration), L2AccessError> {
        match self {
            Self::Available(value) => Ok((
                L2AllocationGeneration(value),
                match value.checked_add(1) {
                    Some(next) => Self::Available(next),
                    None => Self::Exhausted,
                },
            )),
            Self::Exhausted => Err(L2AccessError::FillGenerationExhausted),
        }
    }
}

/// Which mutation family owns one model instance. Read-only previews and
/// snapshots are neutral; the first successful mutation latches the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum L2AccessDiscipline {
    Immediate,
    Event,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingReadPolicy {
    DistinctFetches,
    CoalesceByAllocationSector,
}

impl L2AccessDiscipline {
    const fn name(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::Event => "event",
        }
    }
}

/// Opaque identity of one sector in one event-mode L2 allocation. It carries no
/// completion authority; the shared system uses it only to join a pending read to
/// the ready time of the fetch that already owns that allocation/sector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L2PendingSectorKey {
    owner: L2ModelOwner,
    line_id: i64,
    generation: L2AllocationGeneration,
    sector: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L2PendingSectorIdentity {
    pub line_id: i64,
    pub generation: u64,
    pub sector: u8,
}

impl L2PendingSectorKey {
    /// Stable semantic identity for diagnostics. The process-local model owner
    /// remains private and is intentionally excluded from this projection.
    pub const fn boundary_identity(self) -> L2PendingSectorIdentity {
        L2PendingSectorIdentity {
            line_id: self.line_id,
            generation: self.generation.0,
            sector: self.sector,
        }
    }
}

/// Opaque capability for completing sectors fetched by one event-mode read.
/// Fields are private so callers can route but cannot mint or widen a fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L2FillToken {
    owner: L2ModelOwner,
    line_id: i64,
    generation: L2AllocationGeneration,
    sector_mask: u8,
}

impl L2FillToken {
    /// Sectors whose pending bits this capability may complete.
    pub const fn sector_mask(self) -> u8 {
        self.sector_mask
    }

    pub const fn boundary_identity(self) -> (i64, u64, u8) {
        (self.line_id, self.generation.0, self.sector_mask)
    }

    /// Identity used to register one authorized sector's pending ready time.
    pub const fn pending_sector_key(self, sector: u8) -> Option<L2PendingSectorKey> {
        if sector < 4 && self.sector_mask & (1 << sector) != 0 {
            Some(L2PendingSectorKey {
                owner: self.owner,
                line_id: self.line_id,
                generation: self.generation,
                sector,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedL2Fill {
    owner: L2ModelOwner,
    line_id: i64,
    generation: L2AllocationGeneration,
    sector_mask: u8,
}

impl PreparedL2Fill {
    fn issue(self) -> L2FillToken {
        L2FillToken {
            owner: self.owner,
            line_id: self.line_id,
            generation: self.generation,
            sector_mask: self.sector_mask,
        }
    }
}

/// Event-mode lookup preview available before mutation commits. `fill_mask` is
/// only a non-completable description: a failed callback may retain this value,
/// but no fill capability exists until the cache mutation commits successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2EventMaskPreview {
    pub lookup: L2MaskPreview,
    owner: L2ModelOwner,
    line_id: i64,
    generation: L2AllocationGeneration,
    fill_mask: u8,
    pending_merge_mask: u8,
    eviction: Option<L2Eviction>,
}

impl L2EventMaskPreview {
    /// Shared-L2 allocation generation that owns this access. This is a
    /// semantic identity used to join the eventual response with the exact
    /// allocation; it is independent of any L1 fill generation.
    pub const fn allocation_generation(self) -> u64 {
        self.generation.0
    }

    /// Missing read sectors that start a new lower-level fetch. Under the
    /// distinct-request discipline this includes already-pending sectors; under
    /// coalescing it excludes [`Self::pending_merge_mask`].
    pub const fn fill_mask(self) -> u8 {
        self.fill_mask
    }

    /// Requested read sectors that were already owned by a matching allocation's
    /// pending lower-level fetch before this access began. They remain lookup
    /// misses but issue no duplicate fetch. Pending bits opened by this access
    /// are never reported here.
    pub const fn pending_merge_mask(self) -> u8 {
        self.pending_merge_mask
    }

    /// Exact victim selected by this access, if the new line replaces a
    /// resident allocation. This is topology only: a dirty mask does not imply
    /// a scheduled or completed lower-level writeback.
    pub const fn eviction(self) -> Option<L2Eviction> {
        self.eviction
    }

    /// Opaque identity used to register one newly fetched sector's ready time.
    pub const fn fill_sector_key(self, sector: u8) -> Option<L2PendingSectorKey> {
        if sector < 4 && self.fill_mask & (1 << sector) != 0 {
            Some(L2PendingSectorKey {
                owner: self.owner,
                line_id: self.line_id,
                generation: self.generation,
                sector,
            })
        } else {
            None
        }
    }

    /// Opaque identity used to join one merged sector to its original ready time.
    pub const fn pending_sector_key(self, sector: u8) -> Option<L2PendingSectorKey> {
        if sector < 4 && self.pending_merge_mask & (1 << sector) != 0 {
            Some(L2PendingSectorKey {
                owner: self.owner,
                line_id: self.line_id,
                generation: self.generation,
                sector,
            })
        } else {
            None
        }
    }
}

/// Committed event-mode access outcome. Only this post-commit value can carry an
/// opaque fill capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2EventMaskAccess {
    pub lookup: L2MaskPreview,
    pub fill: Option<L2FillToken>,
    pub eviction: Option<L2Eviction>,
}

/// One exact L2 replacement victim. Masks are line-relative 32 B sectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Eviction {
    pub line_id: i64,
    pub valid_mask: u8,
    pub dirty_mask: u8,
    pub pending_mask: u8,
}

/// Terminal dirty occupancy of one shared L2 model. This is a gauge, not
/// additive traffic; launch-set consumers must never sum it across launches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L2DirtyResidency {
    pub lines: usize,
    pub sectors: usize,
}

/// One resident dirty L2 line selected for a terminal flush. The plan is a
/// state snapshot, not completed traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2DirtyLine {
    pub line_id: i64,
    pub dirty_mask: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2DirtyFlushPlan {
    lines: Box<[L2DirtyLine]>,
}

impl L2DirtyFlushPlan {
    pub fn lines(&self) -> &[L2DirtyLine] {
        &self.lines
    }
}

/// Deterministic result of an event fill completion. A stale allocation is a
/// no-op, while a matching allocation reports newly valid and duplicate bits
/// separately so partial and repeated deliveries are explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2FillCompletion {
    Completed {
        newly_valid_mask: u8,
        duplicate_mask: u8,
    },
    Stale,
}

#[derive(Debug, thiserror::Error)]
pub enum L2AccessError {
    #[error("L2 sector {sector} is outside the 0..4 sector domain")]
    SectorOutOfRange { sector: u8 },
    #[error("L2 sector mask {sector_mask:#06b} must select one or more of bits 0..4")]
    SectorMaskInvalid { sector_mask: u8 },
    #[error("L2 access discipline is {latched}; cannot perform a {requested} mutation")]
    AccessDisciplineMismatch {
        latched: &'static str,
        requested: &'static str,
    },
    #[error("L2 event allocation-generation space is exhausted")]
    FillGenerationExhausted,
    #[error("resident event-mode L2 line {line_id} has no allocation generation")]
    EventAllocationGenerationMissing { line_id: i64 },
    #[error("L2 fill token belongs to a different model")]
    FillTokenOwnerMismatch,
    #[error(
        "L2 completion mask {completion_mask:#06b} must be a nonempty subset of token mask {token_mask:#06b}"
    )]
    FillCompletionMaskInvalid { completion_mask: u8, token_mask: u8 },
    #[error(
        "L2 line {line_id} completion mask {completion_mask:#06b} is outside valid {valid_mask:#06b} plus pending {pending_mask:#06b} state"
    )]
    FillCompletionStateInvalid {
        line_id: i64,
        completion_mask: u8,
        valid_mask: u8,
        pending_mask: u8,
    },
    #[error(transparent)]
    SetIndex(#[from] L2SetIndexError),
    #[error("L2 access clock overflows i64 at {clock}")]
    ClockOverflow { clock: i64 },
    #[error("L2 BIP insertion counter overflows u64 at {counter}")]
    BipCounterOverflow { counter: u64 },
    #[error("L2 observer counter {field} overflows u64")]
    CounterOverflow { field: L2CounterField },
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
    #[error(
        "L2 way {way} in set {set_key} has invalid masks: valid={valid_mask:#06b}, dirty={dirty_mask:#06b}, pending={pending_mask:#06b}"
    )]
    DirtyMaskInvalid {
        set_key: usize,
        way: usize,
        valid_mask: u8,
        dirty_mask: u8,
        pending_mask: u8,
    },
    #[error(
        "L2 dirty flush plan is stale for line {line_id}: expected {expected:#06b}, current {current:#06b}"
    )]
    DirtyFlushStale {
        line_id: i64,
        expected: u8,
        current: u8,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L2Counters {
    /// Architecture provenance bound by the owning [`L2Model`].
    arch: Option<ArchId>,
    pub hits: u64,
    pub misses: u64,
    /// NCU-style L1TEX-origin lookup hits. In addition to completed-data hits,
    /// this counts a read merging into a pending allocation and every
    /// write-class sector lookup. Cache state and response pricing continue to
    /// use `hits`/`misses`; this is an observer-only classification.
    pub lookup_hits: u64,
    /// NCU-style lookup denominator.  It is deliberately separate from the
    /// modeled cache-state `hits + misses` stream because L2 write packetization
    /// can collapse several modeled sector-state mutations into one lookup observer
    /// unit.  Legacy/direct cache accesses populate it one sector at a time;
    /// the shared-system replay replaces the observer totals with its
    /// request/coalescing accounting after the event stream drains.
    pub lookup_requests: u64,
    /// Miss split (state-machine kinds, not the Python 3C diagnostic).
    pub line_misses: u64,
    pub sector_fill_misses: u64,
    /// Of `hits`: write-class hits granted on line-TAG residency while
    /// the sector's valid bit was still clear (the §6.4 rule's whole
    /// contribution — zero when every access is `Read`-class).
    pub write_tag_hits: u64,
    /// Read sectors that joined an already-pending fill owned by the same
    /// event-mode allocation. This is an observer of the bounded
    /// coalescing policy; it does not replace cache-state miss accounting.
    pub pending_merge_sectors: u64,
    /// Read sectors that acquired a new event-mode fill capability. Together
    /// with `pending_merge_sectors` this exposes the sectored fill-ownership
    /// split used by the shared-L2 replay.
    pub new_fill_sectors: u64,
    /// Trace-order request counts by the active effective slice hash. The
    /// fixed array is an observer surface; it does not assert a proprietary
    /// GA102 crossbar hash or add queue timing.
    pub slice_accesses: [u64; 16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2CounterField {
    Hits,
    Misses,
    LookupHits,
    LookupRequests,
    LineMisses,
    SectorFillMisses,
    WriteTagHits,
    PendingMergeSectors,
    NewFillSectors,
    SliceAccesses,
}

impl std::fmt::Display for L2CounterField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Hits => "hits",
            Self::Misses => "misses",
            Self::LookupHits => "lookup_hits",
            Self::LookupRequests => "lookup_requests",
            Self::LineMisses => "line_misses",
            Self::SectorFillMisses => "sector_fill_misses",
            Self::WriteTagHits => "write_tag_hits",
            Self::PendingMergeSectors => "pending_merge_sectors",
            Self::NewFillSectors => "new_fill_sectors",
            Self::SliceAccesses => "slice_accesses",
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum L2CounterError {
    #[error(transparent)]
    Architecture(#[from] sim_arch::ArchitectureProvenanceError),
    #[error("L2 counter '{field}' overflows u64 during aggregation")]
    Overflow { field: L2CounterField },
    #[error(
        "L2 counter '{field}' underflows while computing a launch delta: current={current}, baseline={baseline}"
    )]
    Underflow {
        field: L2CounterField,
        current: u64,
        baseline: u64,
    },
    #[error("L2 counter total overflows u64: hits={hits}, misses={misses}")]
    TotalOverflow { hits: u64, misses: u64 },
    #[error("L2 lookup observer hits exceed requests: hits={hits}, requests={requests}")]
    LookupHitsExceedRequests { hits: u64, requests: u64 },
}

fn checked_counter_add(
    field: L2CounterField,
    left: u64,
    right: u64,
) -> Result<u64, L2CounterError> {
    left.checked_add(right)
        .ok_or(L2CounterError::Overflow { field })
}

fn checked_counter_sub(
    field: L2CounterField,
    current: u64,
    baseline: u64,
) -> Result<u64, L2CounterError> {
    current
        .checked_sub(baseline)
        .ok_or(L2CounterError::Underflow {
            field,
            current,
            baseline,
        })
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
        let lookup_hits = checked_counter_add(
            L2CounterField::LookupHits,
            self.lookup_hits,
            other.lookup_hits,
        )?;
        let lookup_requests = checked_counter_add(
            L2CounterField::LookupRequests,
            self.lookup_requests,
            other.lookup_requests,
        )?;
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
        let pending_merge_sectors = checked_counter_add(
            L2CounterField::PendingMergeSectors,
            self.pending_merge_sectors,
            other.pending_merge_sectors,
        )?;
        let new_fill_sectors = checked_counter_add(
            L2CounterField::NewFillSectors,
            self.new_fill_sectors,
            other.new_fill_sectors,
        )?;
        let mut slice_accesses = self.slice_accesses;
        for (index, value) in slice_accesses.iter_mut().enumerate() {
            *value =
                value
                    .checked_add(other.slice_accesses[index])
                    .ok_or(L2CounterError::Overflow {
                        field: L2CounterField::SliceAccesses,
                    })?;
        }

        self.hits = hits;
        self.misses = misses;
        self.lookup_hits = lookup_hits;
        self.lookup_requests = lookup_requests;
        self.line_misses = line_misses;
        self.sector_fill_misses = sector_fill_misses;
        self.write_tag_hits = write_tag_hits;
        self.pending_merge_sectors = pending_merge_sectors;
        self.new_fill_sectors = new_fill_sectors;
        self.slice_accesses = slice_accesses;
        Ok(())
    }

    /// Compute checked per-launch counters from a cumulative cache-model
    /// snapshot. Both values must carry the same architecture provenance and
    /// every current field must be monotonic relative to `baseline`.
    pub fn checked_delta_since(&self, baseline: &L2Counters) -> Result<L2Counters, L2CounterError> {
        let current_arch = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 counters",
            })?;
        require_counter_arch(current_arch, baseline)?;
        Ok(L2Counters {
            arch: Some(current_arch),
            hits: checked_counter_sub(L2CounterField::Hits, self.hits, baseline.hits)?,
            misses: checked_counter_sub(L2CounterField::Misses, self.misses, baseline.misses)?,
            lookup_hits: checked_counter_sub(
                L2CounterField::LookupHits,
                self.lookup_hits,
                baseline.lookup_hits,
            )?,
            lookup_requests: checked_counter_sub(
                L2CounterField::LookupRequests,
                self.lookup_requests,
                baseline.lookup_requests,
            )?,
            line_misses: checked_counter_sub(
                L2CounterField::LineMisses,
                self.line_misses,
                baseline.line_misses,
            )?,
            sector_fill_misses: checked_counter_sub(
                L2CounterField::SectorFillMisses,
                self.sector_fill_misses,
                baseline.sector_fill_misses,
            )?,
            write_tag_hits: checked_counter_sub(
                L2CounterField::WriteTagHits,
                self.write_tag_hits,
                baseline.write_tag_hits,
            )?,
            pending_merge_sectors: checked_counter_sub(
                L2CounterField::PendingMergeSectors,
                self.pending_merge_sectors,
                baseline.pending_merge_sectors,
            )?,
            new_fill_sectors: checked_counter_sub(
                L2CounterField::NewFillSectors,
                self.new_fill_sectors,
                baseline.new_fill_sectors,
            )?,
            slice_accesses: {
                let mut values = [0_u64; 16];
                for index in 0..values.len() {
                    values[index] = checked_counter_sub(
                        L2CounterField::SliceAccesses,
                        self.slice_accesses[index],
                        baseline.slice_accesses[index],
                    )?;
                }
                values
            },
        })
    }

    pub fn total(&self) -> Result<u64, L2CounterError> {
        self.hits
            .checked_add(self.misses)
            .ok_or(L2CounterError::TotalOverflow {
                hits: self.hits,
                misses: self.misses,
            })
    }

    /// Validate the observer-only lookup pair.  The modeled cache-state stream
    /// is intentionally not required to equal this pair: packetization and
    /// source-side coalescing can make their denominators different.
    pub fn validate_lookup_observer(&self) -> Result<(), L2CounterError> {
        if self.lookup_hits > self.lookup_requests {
            return Err(L2CounterError::LookupHitsExceedRequests {
                hits: self.lookup_hits,
                requests: self.lookup_requests,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Way {
    /// Full line id as the tag (signed: negative wire-admitted ids
    /// replay with Python's floored semantics, never wrapped).
    tag: i64,
    /// Completed/valid-sector bitmask (bit = sector index within the line).
    mask: u8,
    /// Valid sectors modified by write-class traffic and not yet evicted.
    dirty_mask: u8,
    /// Event-mode read sectors requested but not yet completed. Pending bits are
    /// deliberately absent from `mask` and never count as physically present.
    pending_mask: u8,
    /// Event-mode allocation identity. Immediate-parity ways carry `None` and
    /// cannot coexist with event accesses because the discipline is latched.
    generation: Option<L2AllocationGeneration>,
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
    /// Last observer totals committed at a launch boundary.  During a launch
    /// the physical event path temporarily increments the same public counter
    /// fields; keeping this separate witness lets the boundary replace that
    /// provisional stream without erasing a retained stream-L2 history.
    observer_lookup_hits: u64,
    observer_lookup_requests: u64,
    owner: L2ModelOwner,
    next_generation: NextAllocationGeneration,
    access_discipline: Option<L2AccessDiscipline>,
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

fn architecture_name(arch: ArchId) -> &'static str {
    match arch {
        ArchId::Sm86 => "sm_86",
        ArchId::Sm120 => "sm_120",
    }
}

fn replacement_name(replacement: L2Replacement) -> &'static str {
    match replacement {
        L2Replacement::Lru => "lru",
        L2Replacement::Srrip => "srrip",
        L2Replacement::Bip => "bip",
    }
}

fn slice_hash_name(hash: SliceHash) -> &'static str {
    match hash {
        SliceHash::LineMod => "line_mod",
        SliceHash::SectorMod => "sector_mod",
        SliceHash::XorFolded => "xor_folded",
        SliceHash::KnuthMult => "knuth_mult",
        SliceHash::ShiftMod => "shift_mod",
        SliceHash::BitReverse => "bit_reverse",
    }
}

impl L2Model {
    pub fn new(cfg: L2Config) -> Result<Self, L2BuildError> {
        cfg.validate()?;
        let num_sets = cfg.num_sets_per_slice()?;
        let set_count = checked_set_count(cfg.num_slices, num_sets)?;
        let sets = allocate_set_storage(set_count)?;
        let owner = allocate_l2_model_owner()?;
        let arch = cfg.arch;
        Ok(Self {
            cfg,
            num_sets,
            sets,
            clock: 0,
            bip_counter: 0,
            counters: L2Counters::for_arch(arch),
            observer_lookup_hits: 0,
            observer_lookup_requests: 0,
            owner,
            next_generation: NextAllocationGeneration::Available(0),
            access_discipline: None,
        })
    }

    /// Stable structural projection for bounded differential diagnostics.
    /// Only occupied sets are emitted; empty geometry does not inflate a
    /// reference trace. Process-local owner identities are intentionally
    /// omitted because they do not affect modeled ordering.
    pub fn boundary_state(&self) -> String {
        let occupied = self
            .sets
            .iter()
            .enumerate()
            .filter(|(_, ways)| !ways.is_empty())
            .collect::<Vec<_>>();
        format!(
            "cfg={:?};num_sets={};clock={};bip_counter={};sets={:?};counters={:?};observer_lookup=({},{});next_generation={:?};discipline={:?}",
            self.cfg,
            self.num_sets,
            self.clock,
            self.bip_counter,
            occupied,
            self.counters,
            self.observer_lookup_hits,
            self.observer_lookup_requests,
            self.next_generation,
            self.access_discipline,
        )
    }

    /// Return a deterministic typed projection of the shared cache state.
    ///
    /// The cache model itself knows the line/set/way trajectory but does not
    /// know which SM response token produced or consumed a pending sector.
    /// Those requester edges are joined by `sim-sm` at the system boundary;
    /// this method therefore marks event-discipline ownership incomplete while
    /// still exposing every cache-resident field needed by that join.
    pub fn typed_boundary_state(&self) -> Result<L2BoundaryState, L2ProjectionError> {
        self.validate_counter_architecture()
            .map_err(|_| L2ProjectionError::Overflow)?;
        let sets_per_slice = self.num_sets;
        let mut sets = Vec::new();
        for (set_key, ways) in self.sets.iter().enumerate() {
            if ways.is_empty() {
                continue;
            }
            let set_key = u64::try_from(set_key).map_err(|_| L2ProjectionError::Overflow)?;
            let slice =
                u32::try_from(set_key / sets_per_slice).map_err(|_| L2ProjectionError::Overflow)?;
            let set = set_key % sets_per_slice;
            let mut projected_ways = Vec::with_capacity(ways.len());
            for way in ways {
                let line = CanonicalLineKey::VirtualAbsolute { line_id: way.tag };
                projected_ways.push(L2WayState {
                    line,
                    valid_mask: way.mask,
                    dirty_mask: way.dirty_mask,
                    pending_mask: way.pending_mask,
                    recency: way.recency,
                    generation: way.generation.map(|value| value.0),
                });
            }
            sets.push(L2SetState {
                slice,
                set,
                ways: projected_ways,
            });
        }

        let counters = self.counters;
        let counter_state = L2CounterState {
            hits: counters.hits,
            misses: counters.misses,
            lookup_hits: counters.lookup_hits,
            lookup_requests: counters.lookup_requests,
            line_misses: counters.line_misses,
            sector_fill_misses: counters.sector_fill_misses,
            write_tag_hits: counters.write_tag_hits,
        };
        let mut allocations = Vec::new();
        for set in &sets {
            for way in &set.ways {
                let Some(generation) = way.generation else {
                    continue;
                };
                let mut sectors = Vec::new();
                for sector in 0..L2_SECTORS_PER_LINE as u8 {
                    let bit = 1_u8 << sector;
                    let status = if way.pending_mask & bit != 0 {
                        Some(L2SectorStatus::Pending)
                    } else if way.dirty_mask & bit != 0 {
                        Some(L2SectorStatus::Dirty)
                    } else if way.valid_mask & bit != 0 {
                        Some(L2SectorStatus::Valid)
                    } else {
                        None
                    };
                    if let Some(status) = status {
                        sectors.push(L2SectorOwnership {
                            sector,
                            status,
                            producer: None,
                            consumers: Vec::new(),
                            ready_at: None,
                        });
                    }
                }
                allocations.push(L2AllocationState {
                    line: way.line.clone(),
                    generation,
                    sectors,
                });
            }
        }
        allocations.sort_by(|left, right| {
            left.line
                .cmp(&right.line)
                .then(left.generation.cmp(&right.generation))
        });
        let next_generation = match self.next_generation {
            NextAllocationGeneration::Available(value) => Some(value),
            NextAllocationGeneration::Exhausted => None,
        };
        let access_discipline = self.access_discipline.map(|discipline| match discipline {
            L2AccessDiscipline::Immediate => L2AccessDisciplineState::Immediate,
            L2AccessDiscipline::Event => L2AccessDisciplineState::Event,
        });
        let event_discipline = matches!(self.access_discipline, Some(L2AccessDiscipline::Event));
        let state = L2BoundaryState {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            architecture: architecture_name(self.cfg.arch).to_owned(),
            geometry: L2GeometryState {
                architecture: architecture_name(self.cfg.arch).to_owned(),
                capacity_bytes: self.cfg.capacity_bytes,
                line_bytes: self.cfg.line_bytes,
                sector_bytes: self.cfg.sector_bytes,
                num_slices: self.cfg.num_slices,
                sets_per_slice,
                associativity: self.cfg.associativity,
                replacement: replacement_name(self.cfg.replacement).to_owned(),
                slice_hash: slice_hash_name(self.cfg.slice_hash).to_owned(),
            },
            address_domain: AddressDomain::VirtualAbsolute,
            clock: self.clock,
            bip_counter: self.bip_counter,
            access_discipline,
            counters: counter_state,
            observer_lookup_hits: self.observer_lookup_hits,
            observer_lookup_requests: self.observer_lookup_requests,
            next_generation,
            sets,
            allocations,
            request_owners: Vec::new(),
            ownership_complete: !event_discipline,
            unknown_state: if event_discipline {
                vec!["requester_ownership_join_pending".to_owned()]
            } else {
                Vec::new()
            },
        };
        state.validate()?;
        Ok(state)
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

    /// Replace the observer-only lookup totals after a shared-system replay has
    /// applied its source/request coalescing rule.  Modeled cache state,
    /// replacement metadata, and the `hits`/`misses` response-pricing counters
    /// are untouched.  The caller supplies cumulative totals so retained
    /// stream sessions remain monotonic across launch boundaries.
    pub fn replace_lookup_observer(
        &mut self,
        lookup_hits: u64,
        lookup_requests: u64,
    ) -> Result<(), L2AccessError> {
        self.validate_counter_architecture()?;
        if lookup_hits > lookup_requests {
            return Err(L2AccessError::Counter(
                L2CounterError::LookupHitsExceedRequests {
                    hits: lookup_hits,
                    requests: lookup_requests,
                },
            ));
        }
        // This API receives cumulative observer totals at a retained
        // multi-launch boundary.  A lower replacement would silently erase
        // history and make the next `checked_delta_since` underflow (or, if
        // consumed directly, misreport a negative launch contribution).
        // Reject both regressions before changing either field.
        if lookup_hits < self.observer_lookup_hits {
            return Err(L2AccessError::Counter(L2CounterError::Underflow {
                field: L2CounterField::LookupHits,
                current: lookup_hits,
                baseline: self.observer_lookup_hits,
            }));
        }
        if lookup_requests < self.observer_lookup_requests {
            return Err(L2AccessError::Counter(L2CounterError::Underflow {
                field: L2CounterField::LookupRequests,
                current: lookup_requests,
                baseline: self.observer_lookup_requests,
            }));
        }
        self.counters.lookup_hits = lookup_hits;
        self.counters.lookup_requests = lookup_requests;
        self.observer_lookup_hits = lookup_hits;
        self.observer_lookup_requests = lookup_requests;
        Ok(())
    }

    /// Record the source-side sectored fill-ownership split for one committed
    /// event access.  This is intentionally separate from the cache state's
    /// physical `hits`/`misses`: a pending merge remains a lookup miss while
    /// reusing an existing lower-level fetch, whereas a new fill owns a new
    /// lower-level fetch capability.  The caller supplies only counts from a
    /// post-commit [`L2EventMaskPreview`], so callback failure cannot publish
    /// a partial observer update.
    pub fn record_fill_ownership_observer(
        &mut self,
        pending_merge_sectors: u64,
        new_fill_sectors: u64,
    ) -> Result<(), L2AccessError> {
        self.validate_counter_architecture()?;
        let pending_merge_total = self
            .counters
            .pending_merge_sectors
            .checked_add(pending_merge_sectors)
            .ok_or(L2AccessError::CounterOverflow {
                field: L2CounterField::PendingMergeSectors,
            })?;
        let new_fill_total = self
            .counters
            .new_fill_sectors
            .checked_add(new_fill_sectors)
            .ok_or(L2AccessError::CounterOverflow {
                field: L2CounterField::NewFillSectors,
            })?;
        self.counters.pending_merge_sectors = pending_merge_total;
        self.counters.new_fill_sectors = new_fill_total;
        Ok(())
    }

    /// Number of event-mode sectors requested but not yet completed. A launch
    /// boundary may carry replacement/valid-sector state only when this is zero.
    pub fn pending_sector_count(&self) -> usize {
        self.sets
            .iter()
            .flat_map(|ways| ways.iter())
            .map(|way| way.pending_mask.count_ones() as usize)
            .sum()
    }

    /// Dirty sectors still resident after the currently committed access
    /// stream. The result is a terminal gauge, not a traffic counter.
    pub fn dirty_residency(&self) -> L2DirtyResidency {
        let mut result = L2DirtyResidency::default();
        for way in self.sets.iter().flatten() {
            if way.dirty_mask != 0 {
                result.lines += 1;
                result.sectors += way.dirty_mask.count_ones() as usize;
            }
        }
        result
    }

    /// Snapshot all resident dirty sectors in deterministic line order. This
    /// is the terminal-flush admission surface; it is distinct from eviction
    /// topology and does not clear any cache state.
    pub fn preview_dirty_flush(&self) -> L2DirtyFlushPlan {
        let mut lines = self
            .sets
            .iter()
            .flatten()
            .filter(|way| way.dirty_mask != 0)
            .map(|way| L2DirtyLine {
                line_id: way.tag,
                dirty_mask: way.dirty_mask,
            })
            .collect::<Vec<_>>();
        lines.sort_unstable_by_key(|line| line.line_id);
        L2DirtyFlushPlan {
            lines: lines.into_boxed_slice(),
        }
    }

    /// Clear exactly the dirty masks accepted by a lower-level writeback
    /// transaction. A stale plan leaves every line unchanged.
    pub fn commit_dirty_flush(&mut self, plan: L2DirtyFlushPlan) -> Result<(), L2AccessError> {
        let mut locations = Vec::with_capacity(plan.lines.len());
        for line in plan.lines() {
            let set_key = self.set_key(line.line_id)?;
            let Some(way) = self.sets[set_key]
                .iter()
                .position(|way| way.tag == line.line_id)
            else {
                return Err(L2AccessError::DirtyFlushStale {
                    line_id: line.line_id,
                    expected: line.dirty_mask,
                    current: 0,
                });
            };
            let current = self.sets[set_key][way].dirty_mask;
            if current != line.dirty_mask {
                return Err(L2AccessError::DirtyFlushStale {
                    line_id: line.line_id,
                    expected: line.dirty_mask,
                    current,
                });
            }
            locations.push((set_key, way));
        }
        for (set_key, way) in locations {
            self.sets[set_key][way].dirty_mask = 0;
        }
        Ok(())
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
            observer_lookup_hits: self.observer_lookup_hits,
            observer_lookup_requests: self.observer_lookup_requests,
            owner: self.owner,
            next_generation: self.next_generation,
            access_discipline: self.access_discipline,
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

    fn preflight_access_discipline(
        &self,
        requested: L2AccessDiscipline,
    ) -> Result<(), L2AccessError> {
        if let Some(latched) = self.access_discipline
            && latched != requested
        {
            return Err(L2AccessError::AccessDisciplineMismatch {
                latched: latched.name(),
                requested: requested.name(),
            });
        }
        Ok(())
    }

    fn prepare_mask(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
    ) -> Result<PreparedMaskAccess, L2AccessError> {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        self.preflight_access_discipline(L2AccessDiscipline::Immediate)?;
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
        validate_way_masks(ways, set_key)?;

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
        let slice = usize::try_from(self.cfg.slice_hash.slice_of(line_id, self.cfg.num_slices)?)
            .map_err(|_| L2AccessError::CounterOverflow {
                field: L2CounterField::SliceAccesses,
            })?;
        if slice < counters.slice_accesses.len() {
            counters.slice_accesses[slice] = counters.slice_accesses[slice]
                .checked_add(u64::from(sector_mask.count_ones()))
                .ok_or(L2AccessError::CounterOverflow {
                    field: L2CounterField::SliceAccesses,
                })?;
        }
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
            // The sector lookup observer is distinct from physical write-tag
            // accounting: a write to a resident line's absent sector remains
            // a sector lookup miss.  Keep `counted_hit` for physical response
            // pricing and expose only present sectors here.
            let lookup_hit = sector_present;

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
                    if class == L2OpClass::WriteAtomic {
                        bit
                    } else {
                        0
                    },
                    0,
                    None,
                    associativity,
                    self.cfg.replacement,
                    clock,
                    bip_counter,
                )?);
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.misses = 1;
                delta.line_misses = 1;
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
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
                if class == L2OpClass::WriteAtomic {
                    value.dirty_mask |= bit;
                }
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
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
            next_generation: self.next_generation,
            discipline: L2AccessDiscipline::Immediate,
            preview: L2MaskPreview {
                sectors: sector_mask.count_ones(),
                counted_hits: counted_mask.count_ones(),
                present_sectors: present_mask.count_ones(),
                counted_hit_mask: counted_mask,
                present_mask,
            },
        })
    }

    fn prepare_event_mask(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        pending_read_policy: PendingReadPolicy,
    ) -> Result<
        (
            PreparedMaskAccess,
            L2EventMaskPreview,
            Option<PreparedL2Fill>,
        ),
        L2AccessError,
    > {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        self.preflight_access_discipline(L2AccessDiscipline::Event)?;
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
        validate_way_masks(ways, set_key)?;

        let resident = ways.iter().position(|way| way.tag == line_id);
        let mut update = resident.map(|way| PreparedSetUpdate::Update {
            way,
            value: ways[way],
        });
        let mut line_present = resident.is_some();
        let mut valid_mask = resident.map_or(0, |way| ways[way].mask);
        // Preserve the age boundary for this request. `pending_mask` is
        // mutated while the ascending sector walk constructs the new line
        // allocation; using that mutable value to classify a later sector
        // would make a fill opened by this request look like an older fill.
        // The coalescing endpoint is only valid for sectors pending before
        // this access began. Each sector bit is visited once today, but the
        // explicit snapshot keeps that protocol invariant local to the
        // classifier rather than relying on the walk shape.
        let preexisting_pending_mask = resident.map_or(0, |way| ways[way].pending_mask);
        let mut allocation_generation = match resident {
            Some(way) => Some(
                ways[way]
                    .generation
                    .ok_or(L2AccessError::EventAllocationGenerationMissing { line_id })?,
            ),
            None => None,
        };
        let mut next_generation = self.next_generation;
        let mut clock = self.clock;
        let mut bip_counter = self.bip_counter;
        let mut counters = self.counters;
        let slice = usize::try_from(self.cfg.slice_hash.slice_of(line_id, self.cfg.num_slices)?)
            .map_err(|_| L2AccessError::CounterOverflow {
                field: L2CounterField::SliceAccesses,
            })?;
        if slice < counters.slice_accesses.len() {
            counters.slice_accesses[slice] = counters.slice_accesses[slice]
                .checked_add(u64::from(sector_mask.count_ones()))
                .ok_or(L2AccessError::CounterOverflow {
                    field: L2CounterField::SliceAccesses,
                })?;
        }
        let mut counted_mask = 0u8;
        let mut present_mask = 0u8;
        let mut fill_mask = 0u8;
        let mut pending_merge_mask = 0u8;

        for sector in 0..4u8 {
            let bit = 1u8 << sector;
            if sector_mask & bit == 0 {
                continue;
            }
            clock = clock
                .checked_add(1)
                .ok_or(L2AccessError::ClockOverflow { clock })?;
            let sector_present = line_present && valid_mask & bit != 0;
            let sector_pending = line_present && preexisting_pending_mask & bit != 0;
            let counted_hit = sector_present || (line_present && class == L2OpClass::WriteAtomic);
            // L2's tag-stage lookup observer counts a pending allocation as a
            // hit.  This is distinct from `counted_hit`: the latter drives
            // data presence/response pricing, while the former mirrors the
            // operation-level NCU lookup counter (including same-mask sector
            // coalescing).
            // A pending read allocation can satisfy a lookup observer hit,
            // but operation class and write-tag accounting cannot turn an
            // absent sector into a lookup hit.
            let lookup_hit = sector_present || sector_pending;
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
                let (generation, next) = next_generation.allocate()?;
                next_generation = next;
                allocation_generation = Some(generation);
                let (initial_valid, initial_dirty, initial_pending) = match class {
                    L2OpClass::Read => (0, 0, bit),
                    L2OpClass::WriteAtomic => (bit, bit, 0),
                };
                update = Some(prepare_install(
                    ways,
                    set_key,
                    line_id,
                    initial_valid,
                    initial_dirty,
                    initial_pending,
                    Some(generation),
                    associativity,
                    self.cfg.replacement,
                    clock,
                    bip_counter,
                )?);
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.misses = 1;
                delta.line_misses = 1;
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
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
                match class {
                    L2OpClass::Read => {
                        if !sector_present {
                            value.pending_mask |= bit;
                        }
                    }
                    L2OpClass::WriteAtomic => {
                        value.mask |= bit;
                        value.dirty_mask |= bit;
                        value.pending_mask &= !bit;
                    }
                }
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
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
            } else if class == L2OpClass::Read {
                if sector_pending
                    && pending_read_policy == PendingReadPolicy::CoalesceByAllocationSector
                {
                    pending_merge_mask |= bit;
                } else {
                    fill_mask |= bit;
                }
            }
            if class == L2OpClass::WriteAtomic {
                valid_mask |= bit;
            }
            line_present = true;
        }

        let generation = allocation_generation
            .ok_or(L2AccessError::EventAllocationGenerationMissing { line_id })?;
        let lookup = L2MaskPreview {
            sectors: sector_mask.count_ones(),
            counted_hits: counted_mask.count_ones(),
            present_sectors: present_mask.count_ones(),
            counted_hit_mask: counted_mask,
            present_mask,
        };
        let event_preview = L2EventMaskPreview {
            lookup,
            owner: self.owner,
            line_id,
            generation,
            fill_mask,
            pending_merge_mask,
            eviction: update.as_ref().and_then(PreparedSetUpdate::eviction),
        };
        let prepared_fill = (fill_mask != 0).then_some(PreparedL2Fill {
            owner: self.owner,
            line_id,
            generation,
            sector_mask: fill_mask,
        });
        Ok((
            PreparedMaskAccess {
                set_key,
                update: update.ok_or(L2AccessError::ReplacementSetEmpty { set_key })?,
                clock,
                bip_counter,
                counters,
                next_generation,
                discipline: L2AccessDiscipline::Event,
                preview: lookup,
            },
            event_preview,
            prepared_fill,
        ))
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
            counted_hit_mask: counted_mask,
            present_mask,
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

    /// Event-discipline counterpart to [`Self::access_mask_transactional`].
    /// Read misses allocate/tag the line and set pending bits but remain absent
    /// from physical-presence and hit accounting until the returned opaque fill
    /// capability is completed. This compatibility surface keeps duplicate
    /// pending reads as distinct lower-level fetches. The callback receives only
    /// a non-completable preview; a fill token is minted after both the callback
    /// and cache commit succeed. Writes validate their sectors immediately.
    pub fn access_mask_event_transactional<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        before_commit: impl FnOnce(L2EventMaskPreview) -> Result<T, E>,
    ) -> Result<(L2EventMaskAccess, T), E>
    where
        E: From<L2AccessError>,
    {
        self.access_mask_event_with_policy(
            line_id,
            sector_mask,
            class,
            PendingReadPolicy::DistinctFetches,
            before_commit,
        )
    }

    /// Event-mode read access that merges a request for an already-pending sector
    /// onto the matching allocation-generation/sector. The lookup still counts a
    /// miss, but [`L2EventMaskPreview::pending_merge_mask`] identifies sectors
    /// that must reuse the original lower-level ready time instead of issuing a
    /// duplicate fetch. Preparation and callback failure are transactional.
    pub fn access_mask_event_coalesced_transactional<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        before_commit: impl FnOnce(L2EventMaskPreview) -> Result<T, E>,
    ) -> Result<(L2EventMaskAccess, T), E>
    where
        E: From<L2AccessError>,
    {
        self.access_mask_event_with_policy(
            line_id,
            sector_mask,
            class,
            PendingReadPolicy::CoalesceByAllocationSector,
            before_commit,
        )
    }

    fn access_mask_event_with_policy<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        pending_read_policy: PendingReadPolicy,
        before_commit: impl FnOnce(L2EventMaskPreview) -> Result<T, E>,
    ) -> Result<(L2EventMaskAccess, T), E>
    where
        E: From<L2AccessError>,
    {
        let (prepared, preview, prepared_fill) = self
            .prepare_event_mask(line_id, sector_mask, class, pending_read_policy)
            .map_err(E::from)?;
        let value = before_commit(preview)?;
        prepared.commit(self);
        let access = L2EventMaskAccess {
            lookup: preview.lookup,
            fill: prepared_fill.map(PreparedL2Fill::issue),
            eviction: preview.eviction,
        };
        Ok((access, value))
    }

    /// Complete every sector authorized by one event read capability.
    pub fn complete_fill(&mut self, token: L2FillToken) -> Result<L2FillCompletion, L2AccessError> {
        self.complete_fill_mask(token, token.sector_mask)
    }

    /// Complete a nonempty subset of one event read capability. Matching
    /// completions are idempotent; eviction or same-tag reallocation returns a
    /// stale no-op and can never resurrect the old allocation.
    pub fn complete_fill_mask(
        &mut self,
        token: L2FillToken,
        completion_mask: u8,
    ) -> Result<L2FillCompletion, L2AccessError> {
        self.validate_counter_architecture()?;
        self.preflight_access_discipline(L2AccessDiscipline::Event)?;
        if token.owner != self.owner {
            return Err(L2AccessError::FillTokenOwnerMismatch);
        }
        if completion_mask == 0
            || completion_mask & !0b1111 != 0
            || completion_mask & !token.sector_mask != 0
        {
            return Err(L2AccessError::FillCompletionMaskInvalid {
                completion_mask,
                token_mask: token.sector_mask,
            });
        }

        let set_key = self.set_key(token.line_id)?;
        validate_way_masks(&self.sets[set_key], set_key)?;
        let Some(way) = self.sets[set_key]
            .iter_mut()
            .find(|way| way.tag == token.line_id)
        else {
            return Ok(L2FillCompletion::Stale);
        };
        match way.generation {
            Some(generation) if generation == token.generation => {}
            Some(_) => return Ok(L2FillCompletion::Stale),
            None => {
                return Err(L2AccessError::EventAllocationGenerationMissing {
                    line_id: token.line_id,
                });
            }
        }
        if way.mask & way.pending_mask != 0 || completion_mask & !(way.mask | way.pending_mask) != 0
        {
            return Err(L2AccessError::FillCompletionStateInvalid {
                line_id: token.line_id,
                completion_mask,
                valid_mask: way.mask,
                pending_mask: way.pending_mask,
            });
        }

        let newly_valid_mask = completion_mask & way.pending_mask;
        let duplicate_mask = completion_mask & way.mask;
        way.mask |= newly_valid_mask;
        way.pending_mask &= !newly_valid_mask;
        Ok(L2FillCompletion::Completed {
            newly_valid_mask,
            duplicate_mask,
        })
    }
}

struct PreparedMaskAccess {
    set_key: usize,
    update: PreparedSetUpdate,
    clock: i64,
    bip_counter: u64,
    counters: L2Counters,
    next_generation: NextAllocationGeneration,
    discipline: L2AccessDiscipline,
    preview: L2MaskPreview,
}

impl PreparedMaskAccess {
    fn commit(self, l2: &mut L2Model) {
        self.update.commit(&mut l2.sets[self.set_key]);
        l2.clock = self.clock;
        l2.bip_counter = self.bip_counter;
        l2.counters = self.counters;
        l2.next_generation = self.next_generation;
        if l2.access_discipline.is_none() {
            l2.access_discipline = Some(self.discipline);
        }
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
        evicted: Way,
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

    fn eviction(&self) -> Option<L2Eviction> {
        let Self::Replace { evicted, .. } = self else {
            return None;
        };
        Some(L2Eviction {
            line_id: evicted.tag,
            valid_mask: evicted.mask,
            dirty_mask: evicted.dirty_mask,
            pending_mask: evicted.pending_mask,
        })
    }

    fn commit(self, ways: &mut Vec<Way>) {
        match self {
            Self::Update { way, value } => ways[way] = value,
            Self::Insert { value } => ways.push(value),
            Self::Replace {
                victim,
                srrip_age,
                evicted: _,
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

fn validate_way_masks(ways: &[Way], set_key: usize) -> Result<(), L2AccessError> {
    for (way, entry) in ways.iter().enumerate() {
        if entry.dirty_mask & !entry.mask != 0 || entry.dirty_mask & entry.pending_mask != 0 {
            return Err(L2AccessError::DirtyMaskInvalid {
                set_key,
                way,
                valid_mask: entry.mask,
                dirty_mask: entry.dirty_mask,
                pending_mask: entry.pending_mask,
            });
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
    valid_mask: u8,
    dirty_mask: u8,
    pending_mask: u8,
    generation: Option<L2AllocationGeneration>,
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
                mask: valid_mask,
                dirty_mask,
                pending_mask,
                generation,
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
    let evicted = ways[victim];
    Ok(PreparedSetUpdate::Replace {
        victim,
        srrip_age,
        evicted,
        value: Way {
            tag,
            mask: valid_mask,
            dirty_mask,
            pending_mask,
            generation,
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
