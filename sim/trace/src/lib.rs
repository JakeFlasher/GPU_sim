//! sim-trace: reader for NVBit dynamic traces (the simulator's primary input).
//!
//! Owns: the committed prost bindings for the tracer's protobuf schema
//! ([`pb`]), per-lane address reconstruction ([`lanes`]), and the
//! trace-directory walker plus reader-level wire validation ([`reader`]).
//!
//! This crate consumes NVBit traces only. NCU observations are comparison
//! targets and never appear here.

pub mod anchored;
pub mod async_sidecars;
#[cfg(feature = "campaign")]
pub mod digest;
pub mod exec_info;
pub mod json;
pub mod lanes;
pub mod obligations;
pub mod phase_sidecars;
pub mod pb;
pub mod reader;
pub mod semantic;
pub mod submission;
mod wire_preflight;

pub use anchored::{
    AnchoredDir, AnchoredFile, AnchoredReadError, FileGenerationIdentity, FileIdentity,
    RetainedFile,
};
#[cfg(feature = "campaign")]
pub use digest::{
    CanonicalJsonError, Sha256Digest, Sha256DigestError, canonical_json, canonical_json_utf8,
};
pub use exec_info::{ExecInfo, ExecInfoError, KernelStaticInfo};
pub use json::strict_json_from_slice;
pub use lanes::{LANE_INACTIVE_SENTINEL, LANES_PER_WARP, LaneError, reconstruct_lane_vector};
pub use reader::{
    AddressAccess, AllocationRegistry, DynamicMemoryJoinIdentity, DynamicMemoryJoinLocation,
    KernelLoc, MemoryAccess, MemoryAccessMeta, TraceDir, TraceError, TraceLineage, TraceStats,
    TraceValidationSummary, block_launch_index, check_allocation_identity, kernel_grid,
    memory_access, validate_threadblock_contract, visit_memory_access,
};
#[cfg(feature = "campaign")]
pub use reader::{
    ArtifactDigestIdentity, LosslessFullProvenance, ThreadblockPayloadIdentity,
    TraceArtifactIdentity, TraceDirSnapshot, TraceSamplingMethod, TraceSamplingSidecar,
    TraceSnapshotLimits, WaveSegmentSubsetProvenance,
};
