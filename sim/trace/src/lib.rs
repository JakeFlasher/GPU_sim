//! sim-trace: reader for NVBit dynamic traces (the simulator's primary input).
//!
//! Owns: the committed prost bindings for the tracer's protobuf schema
//! ([`pb`]), per-lane address reconstruction in lockstep with the Python
//! admission gate ([`lanes`]), and the trace-directory walker + the
//! reader-level wire contract ([`reader`]).
//!
//! Rule 1 (targets-never-features) note: this crate consumes NVBit traces
//! only. NCU label artifacts are prediction targets and never appear here.

pub mod anchored;
pub mod digest;
pub mod exec_info;
pub mod lanes;
pub mod pb;
pub mod reader;
mod wire_preflight;

pub use anchored::{AnchoredDir, AnchoredFile, AnchoredReadError, FileIdentity, RetainedFile};
pub use digest::{
    CanonicalJsonError, Sha256Digest, Sha256DigestError, canonical_json, canonical_json_utf8,
};
pub use exec_info::{ExecInfo, ExecInfoError, KernelStaticInfo};
pub use lanes::{LANE_INACTIVE_SENTINEL, LANES_PER_WARP, LaneError, reconstruct_lane_vector};
pub use reader::{
    AddressAccess, AllocationRegistry, ArtifactDigestIdentity, KernelLoc, LosslessFullProvenance,
    MemoryAccess, MemoryAccessMeta, ThreadblockPayloadIdentity, TraceArtifactIdentity, TraceDir,
    TraceDirSnapshot, TraceError, TraceSamplingMethod, TraceSamplingSidecar, TraceSnapshotLimits,
    TraceStats, TraceValidationSummary, WaveSegmentSubsetProvenance, block_launch_index,
    check_allocation_identity, kernel_grid, memory_access, validate_threadblock_contract,
    visit_memory_access,
};
