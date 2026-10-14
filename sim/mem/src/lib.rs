//! sim-mem: the memory hierarchy models (design §6.3).
//!
//! v0 slice — the L1TEX layer:
//! - [`l1`]: sectored L1 cache core, a lockstep port of the validated
//!   Phase-1 warp-pool walker's cache state machine
//!   (`the native L1 replay model`); the sim's SM issue
//!   model supplies the real warp interleaving the Python pool scheduler
//!   stood in for.
//! - [`cache_op`]: cache-operator classification from the control-bits
//!   sidecar's per-pc SASS text (closed 12-token vocabulary + eligibility
//!   partition in lockstep with `l1_cache_op_classifier.py`; the
//!   classifier's OWN manifest artifacts are not consumed — its section
//!   extraction mixes fatbin arch slices, see the module doc).
//!   The token remains literal across architectures; [`access`] applies the
//!   measured sm_86 exception that CUDA's `LDG.E.STRONG.SM` lowering is L1
//!   eligible, without changing Blackwell strong-scope routing.
//! - [`pchase_card`]: geometry/latency resolvers over the committed
//!   P-chase static card, porting the fail-closed resolution contract
//!   (capacity has no default; typed gaps resolve only to DOCUMENTED
//!   defaults/approximations).
//!
//! - [`l2`]: the shared L2 slice model, a lockstep port of the sectored
//!   set-associative replay core in
//!   `the native L2 replay model` (geometry from the arch
//!   card's vendor-documented `l2_geometry`; associativity/hash/policy
//!   are documented-default hypothesis knobs — the Phase-1 label-fitted
//!   calibration is never consumed, Rule 1).
//! - [`access`]: the shared `MemRole`/`AccessKind` record classification
//!   (token-preserving, so hierarchy levels can specialize per token).
//! - [`writeback`]: an opt-in transactional dirty-sector service primitive;
//!   it is not native SM86 coherence completion.
//! - [`coherence`]: an opt-in ordered-L2 directory/service experiment;
//!   it is not a protocol implementation.
//! - [`l1_visibility`]: an opt-in stateful per-SM L1 ownership/invalidation
//!   queue. It requires observations for resident hits and is not attached to
//!   native replay until the fence/boundary wire contract is complete.
//!
//! Later increment: the DRAM bandwidth model (v0 prices DRAM as the
//! carded flat latency — no bytes-per-cycle covariate is carded at the
//! locked operating point yet). Rule 1: inputs here are traces,
//! cuobjdump-derived sidecars and microbench cards — never NCU values.

pub mod access;
pub mod cache_op;
pub mod coherence;
pub mod l1;
pub mod l1_visibility;
pub mod l2;
pub mod memory_order;
pub mod payload;
pub mod pchase_card;
pub mod writeback;

pub use access::{
    ASYNC_COPY_COMPLETION_VALUE_MASK, ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX, AccessKind,
    AsyncCopyCompletion, MemRole, MemRoleError, MemoryOrigin, decode_async_copy_completion,
    encode_async_copy_destination_work, encode_async_copy_payload_and_topology,
    encode_async_copy_payload_work, is_global_reduction_opcode, is_global_store_opcode,
    is_known_untagged_opcode, mem_role_of_checked,
};
pub use cache_op::{
    BARE_LDG_E_DEFAULTS_TO_CA, CacheOpError, CacheOpToken, LdgstsL1Policy,
    classify_ldgsts_l1_policy, classify_opcode_checked, classify_opcode_for_arch,
    derive_l1_eligible, opcode_head,
};
pub use coherence::{
    CoherenceCompletion, CoherenceConfig, CoherenceError, CoherencePlan, CoherenceQueue,
    CoherenceRequest,
};
pub use l1::{
    AccessOutcome, EvAccess, EvAccessUnobserved, FillKey, L1AccessError, L1Cache, L1Config,
    L1ConfigError, L1CounterError, L1CounterField, L1Counters, L1DirtyFlushPlan, L1DirtyLine,
    L1DirtyPayloadAdmissionPlan, L1DirtyResidency, L1InvalidationResult, L1LookupBatch,
    L1MutationBudget, L1OriginCounters, L1PayloadResidencyWitness, L1StorePolicy,
    L1WriteLookupObserver, L1WriteLookupObserverPlan, LINE_BYTES, PreparedSectorExpansion,
    SECTOR_BYTES, SECTORS_PER_LINE, SectorExpansionError, SectorExpansionPlan,
};
pub use l1_visibility::{
    L1AccessObservation, L1FenceObservation, L1FillVisibleObservation, L1InvalidateAllObservation,
    L1InvalidationTarget, L1VisibilityCompletion, L1VisibilityConfig, L1VisibilityError,
    L1VisibilityEvent, L1VisibilityPlan, L1VisibilityQueue, VisibilityOrder, VisibilityScope,
};
pub use l2::{
    L2AccessError, L2BuildError, L2Config, L2ConfigError, L2CounterError, L2CounterField,
    L2Counters, L2DirtyFlushPlan, L2DirtyLine, L2DirtyResidency, L2EventMaskAccess,
    L2EventMaskPreview, L2Eviction, L2FillCompletion, L2FillToken, L2MaskPreview, L2Model,
    L2OpClass, L2Outcome, L2PendingSectorIdentity, L2PendingSectorKey, L2ProjectionError,
    L2Replacement, L2SetIndexError, L2SnapshotError, L2Source, SliceHash,
    L2_WRITE_ALLOCATE_POLICY,
};
pub use memory_order::{
    MemoryOrder, MemoryOrderError, MemoryOrderLedgerV1, MemoryOrderScope, MemoryRead,
    MemoryVisibilityQualifier, MemoryWrite,
};
pub use payload::{
    NativePayloadIdentityV1, NativePayloadJoinKey, NativePayloadRecordV1, PayloadCommitLedgerV1,
    PayloadError, PayloadForwardingLedgerV1, SECTOR_BYTES as PAYLOAD_SECTOR_BYTES, SectorPayloadV1,
    coalesce_payloads, enrich_native_payload_records, join_native_payload_records,
    load_native_payload_sidecar, load_native_payload_sidecar_records,
    require_native_payload_identity,
};
pub use pchase_card::{
    GeometryPolicyConfig, LatencyConfig, LineCapacityConfig, MshrConfig, PchaseCard,
    PchaseCardError, WalkGeometry,
};
pub use writeback::{
    PayloadWritebackPlan, ServicePlan, ServiceRequest, WritebackCompletion, WritebackError,
    WritebackPlan, WritebackQueue,
};
