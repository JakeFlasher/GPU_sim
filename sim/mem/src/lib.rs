//! sim-mem: the memory hierarchy models (design §6.3).
//!
//! v0 slice — the L1TEX layer:
//! - [`l1`]: sectored L1 cache core, a lockstep port of the validated
//!   Phase-1 warp-pool walker's cache state machine
//!   (`tools/cache_predictor/l1_warp_pool_walker.py`); the sim's SM issue
//!   model supplies the real warp interleaving the Python pool scheduler
//!   stood in for.
//! - [`cache_op`]: cache-operator classification from the control-bits
//!   sidecar's per-pc SASS text (closed 12-token vocabulary + eligibility
//!   partition in lockstep with `l1_cache_op_classifier.py`; the
//!   classifier's OWN manifest artifacts are not consumed — its section
//!   extraction mixes fatbin arch slices, see the module doc).
//! - [`pchase_card`]: geometry/latency resolvers over the committed
//!   P-chase static card, porting the fail-closed resolution contract
//!   (capacity has no default; typed gaps resolve only to DOCUMENTED
//!   defaults/approximations).
//!
//! - [`l2`]: the shared L2 slice model, a lockstep port of the sectored
//!   set-associative replay core in
//!   `tools/cache_replay/sector_cache_replay.py` (geometry from the arch
//!   card's vendor-documented `l2_geometry`; associativity/hash/policy
//!   are documented-default hypothesis knobs — the Phase-1 label-fitted
//!   calibration is never consumed, Rule 1).
//! - [`access`]: the shared `MemRole`/`AccessKind` record classification
//!   (token-preserving, so hierarchy levels can specialize per token).
//!
//! Later increment: the DRAM bandwidth model (v0 prices DRAM as the
//! carded flat latency — no bytes-per-cycle covariate is carded at the
//! locked operating point yet). Rule 1: inputs here are traces,
//! cuobjdump-derived sidecars and microbench cards — never NCU values.

pub mod access;
pub mod cache_op;
pub mod l1;
pub mod l2;
pub mod pchase_card;

pub use access::{
    AccessKind, MemRole, MemRoleError, is_global_store_opcode, is_known_untagged_opcode,
    mem_role_of_checked,
};
pub use cache_op::{
    BARE_LDG_E_DEFAULTS_TO_CA, CacheOpError, CacheOpToken, classify_opcode_checked,
    classify_opcode_for_arch, derive_l1_eligible, opcode_head,
};
pub use l1::{
    AccessOutcome, EvAccess, FillKey, L1Cache, L1Config, L1ConfigError, L1Counters, LINE_BYTES,
    SECTOR_BYTES, SECTORS_PER_LINE, push_unique_sectors,
};
pub use l2::{
    DOCUMENTED_DEFAULT_L2_ASSOCIATIVITY, DOCUMENTED_DEFAULT_L2_REPLACEMENT,
    DOCUMENTED_DEFAULT_L2_SLICE_HASH, L2Config, L2ConfigError, L2Counters, L2Model, L2OpClass,
    L2Outcome, L2Replacement, L2Source, SliceHash,
};
pub use pchase_card::{
    DOCUMENTED_DEFAULT_MSHR_COUNT, GeometryPolicyConfig, LatencyConfig, LineCapacityConfig,
    MshrConfig, PchaseCard, PchaseCardError, WalkGeometry,
};
