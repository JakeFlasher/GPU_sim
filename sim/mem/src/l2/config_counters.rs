//! L2 card configuration and source-split counters; counters are report-only.

use super::*;

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
    #[allow(clippy::too_many_arguments)]
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
    /// Store sectors whose 128-byte line tag was present before this packet.
    /// Same-packet sectors on a cold line therefore remain zero.
    pub pre_packet_write_tag_hit_mask: u8,
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
pub(super) struct L2ModelOwner(pub(super) u64);

/// Private identity of one event-discipline line allocation. Reinstalling the
/// same tag receives a fresh generation, preventing stale/ABA completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct L2AllocationGeneration(pub(super) u64);

/// Which mutation family owns one model instance. Read-only previews and
/// snapshots are neutral; the first successful mutation latches the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum L2AccessDiscipline {
    Immediate,
    Event,
}

impl L2AccessDiscipline {
    pub(super) const fn name(self) -> &'static str {
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
    pub(super) owner: L2ModelOwner,
    pub(super) line_id: i64,
    pub(super) generation: L2AllocationGeneration,
    pub(super) sector: u8,
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
    pub(super) owner: L2ModelOwner,
    pub(super) line_id: i64,
    pub(super) generation: L2AllocationGeneration,
    pub(super) sector_mask: u8,
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
pub(super) struct PreparedL2Fill {
    pub(super) owner: L2ModelOwner,
    pub(super) line_id: i64,
    pub(super) generation: L2AllocationGeneration,
    pub(super) sector_mask: u8,
}

impl PreparedL2Fill {
    pub(super) fn issue(self) -> L2FillToken {
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
    pub(super) owner: L2ModelOwner,
    pub(super) line_id: i64,
    pub(super) generation: L2AllocationGeneration,
    pub(super) fill_mask: u8,
    pub(super) pending_merge_mask: u8,
    pub(super) eviction: Option<L2Eviction>,
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
    pub(super) lines: Box<[L2DirtyLine]>,
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
    pub(super) arch: Option<ArchId>,
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
                for (index, value) in values.iter_mut().enumerate() {
                    *value = checked_counter_sub(
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
