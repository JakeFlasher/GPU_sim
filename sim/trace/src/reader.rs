//! Trace-directory reader for the NVBit tracer's on-disk layout:
//!
//! ```text
//! <trace_dir>/dynamic_trace.pb                    # dynamic_trace.Trace
//! <trace_dir>/threadblocks/device_<d>/stream_<s>/kernel_<k>/*.pb
//!                                                 # dynamic_trace.threadblock
//! <trace_dir>/extra_info/allocations.json         # allocation-identity registry
//! <trace_dir>/trace_sampling.json                 # sampling-provenance sidecar
//! ```
//!
//! [`TraceDir::open_direct`] reads ordinary replay inputs and ignores campaign
//! provenance. With the explicit `campaign` feature, [`TraceDir::open`] retains
//! the historical sealed-campaign behavior for the opt-in campaign skill. Both
//! loaders validate protobuf and simulator semantics before exposing replay
//! inputs.

use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs;
use std::path::{Path, PathBuf};

use prost::Message;
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::anchored::{AnchoredDir, AnchoredFile, AnchoredReadError};
#[cfg(feature = "campaign")]
use crate::anchored::{FileGenerationIdentity, FileIdentity};
#[cfg(feature = "campaign")]
use crate::digest::{Sha256, Sha256Digest, canonical_json};
use crate::lanes::{self, LaneError};
use crate::pb;
use crate::wire_preflight::{self, WirePreflightError};

const MIB: u64 = 1024 * 1024;
const DYNAMIC_TRACE_PB_MAX_BYTES: u64 = 1024 * MIB;
const ALLOCATIONS_JSON_MAX_BYTES: u64 = 16 * MIB;
#[cfg(feature = "campaign")]
const TRACE_SAMPLING_JSON_MAX_BYTES: u64 = 64 * MIB;
const STATS_CSV_MAX_BYTES: u64 = 64 * MIB;
const DIRECT_THREADBLOCK_FILE_MAX_COUNT: u64 = 1_000_000;
const DIRECT_THREADBLOCK_TOTAL_MAX_BYTES: u64 = 32 * 1024 * MIB;
const ALLOCATION_ENTRY_MAX_COUNT: u64 = 131_072;
const ALLOCATIONS_JSON_MAX_NODES: u64 = 1 + 7 * ALLOCATION_ENTRY_MAX_COUNT;
#[cfg(feature = "campaign")]
const TRACE_SAMPLING_JSON_MAX_NODES: u64 = 1_000_000;

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("descriptor-anchored trace snapshot failed: {0}")]
    AnchoredSnapshot(#[from] AnchoredReadError),
    #[error("descriptor-anchored trace snapshot artifact {artifact} read failed: {source}")]
    SnapshotArtifactRead {
        artifact: &'static str,
        #[source]
        source: AnchoredReadError,
    },
    #[error(
        "trace snapshot artifact {artifact} presence mismatch: expected {expected}, actual {actual}"
    )]
    SnapshotArtifactPresenceMismatch {
        artifact: &'static str,
        expected: bool,
        actual: bool,
    },
    #[error(
        "trace snapshot artifact {artifact} size mismatch: expected {expected}, actual {actual}"
    )]
    SnapshotArtifactSizeMismatch {
        artifact: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error(
        "trace snapshot artifact {artifact} file count mismatch: expected {expected}, actual {actual}"
    )]
    SnapshotArtifactCardinalityMismatch {
        artifact: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error(
        "trace artifact {artifact} at {path} has {resource} {observed}, exceeding implementation limit {limit}"
    )]
    TraceArtifactResourceLimitExceeded {
        path: PathBuf,
        artifact: &'static str,
        resource: &'static str,
        observed: u64,
        limit: u64,
    },
    #[error("trace root protobuf {path} is empty")]
    TraceRootEmpty { path: PathBuf },
    #[error("could not parse {path} as {message_type}: {source}")]
    Decode {
        path: PathBuf,
        message_type: &'static str,
        #[source]
        source: prost::DecodeError,
    },
    #[error("could not preflight {path} as {message_type}: malformed protobuf wire: {detail}")]
    ProtobufWireMalformed {
        path: PathBuf,
        message_type: &'static str,
        detail: &'static str,
    },
    #[error(
        "{path}: {message_type} decoded resource {resource} exceeds implementation limit {limit} (observed at least {observed})"
    )]
    ProtobufDecodedResourceLimitExceeded {
        path: PathBuf,
        message_type: &'static str,
        resource: &'static str,
        observed: u64,
        limit: u64,
    },
    #[error(
        "{path}: unsupported trace_contract_id {got:?}; expected empty legacy identity, sm86_addr_carry_v1, sm86_addr_carry_shmem_split_v2, or sm86_addr_carry_ldgsts_dst_v3"
    )]
    TraceContractUnsupported { path: PathBuf, got: String },
    #[error(
        "launch ({device_id}, {stream_id}, {kernel_id}) has negative combined shared-memory bytes {combined}"
    )]
    TraceLaunchSharedMemoryInvalid {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        combined: i32,
    },
    #[error(
        "legacy launch ({device_id}, {stream_id}, {kernel_id}) carries split shared-memory fields"
    )]
    SharedMemorySplitUnadvertised {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "launch ({device_id}, {stream_id}, {kernel_id}) has a nonzero combined shared-memory count without an authenticated split"
    )]
    SharedMemorySplitUnavailable {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "split-capable launch ({device_id}, {stream_id}, {kernel_id}) does not carry both shared-memory components"
    )]
    SharedMemorySplitMissing {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "launch ({device_id}, {stream_id}, {kernel_id}) combined shared-memory bytes {combined} disagree with static {static_bytes} plus dynamic {dynamic_bytes}"
    )]
    SharedMemorySplitMismatch {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        combined: i32,
        static_bytes: u32,
        dynamic_bytes: u32,
    },
    #[error(
        "{path}: trace contract {contract_id:?} carries invalid memory_payload_classes {got:?}; expected canonical values [GLOBAL, SHARED, LOCAL, ATOMIC, ASYNC_COPY], in that exact order for split-v2"
    )]
    MemoryPayloadCapabilitiesInvalid {
        path: PathBuf,
        contract_id: String,
        got: Vec<i32>,
    },
    #[error(
        "trace threadblocks directory {dir} is missing or contains zero protobuf payload files"
    )]
    TraceThreadblocksEmpty { dir: PathBuf },
    #[error("root launch has no threadblock payload under {dir}")]
    ThreadblocksMissing { dir: PathBuf },
    #[error("threadblock payload entry {path} is not a regular file")]
    ThreadblockPayloadNotRegular { path: PathBuf },
    #[error("threadblock payload snapshot under {root} is invalid: {message}")]
    ThreadblockPayloadSnapshotInvalid { root: PathBuf, message: String },
    #[error("verified sampling payload snapshot is missing expected file {path}")]
    SamplingPayloadSnapshotMissing { path: PathBuf },
    #[error("raw threadblock payload bytes for {path} were released by the compact replay loader")]
    RawThreadblockPayloadUnavailable { path: PathBuf },
    #[error(
        "{threadblock}: warp map key {map_key} disagrees with embedded Warp.id {warp_id} (canonical: warp_map_key_mismatch)"
    )]
    WarpMapKeyMismatch {
        threadblock: String,
        map_key: i32,
        warp_id: i32,
    },
    #[error("{record}: warpid_tb absent on the wire (canonical: warpid_tb_missing)")]
    WarpidTbMissing { record: String },
    #[error(
        "{record}: warpid_tb={actual} disagrees with enclosing warp id {expected} (canonical: warpid_tb_mismatch)"
    )]
    WarpidTbMismatch {
        record: String,
        expected: i32,
        actual: i32,
    },
    #[error(
        "{record}: function_unique_id={got} is not strictly positive (canonical: function_unique_id_invalid)"
    )]
    FunctionUniqueIdInvalid { record: String, got: i32 },
    #[error("{record}: active_mask absent on the wire (canonical: active_mask_missing)")]
    ActiveMaskMissing { record: String },
    #[error("{record}: predicate_mask absent on the wire (canonical: predicate_mask_missing)")]
    PredicateMaskMissing { record: String },
    #[error("{record}: vpc absent on the wire (canonical: vpc_missing)")]
    VpcMissing { record: String },
    #[error(
        "{record}: vpc low 32 bits 0x{vpc_low_32:x} disagree with pc 0x{pc:x} (canonical: vpc_inconsistent_with_pc)"
    )]
    VpcInconsistentWithPc {
        record: String,
        vpc_low_32: u32,
        pc: u32,
    },
    #[error(
        "{record}: lane address payload contradicts active_mask & predicate_mask (canonical: lane_mask_inconsistent)"
    )]
    LaneMaskInconsistent { record: String },
    #[error(
        "{record}: instruction has {n_addresses} address record(s) but memory_op_class is absent"
    )]
    MemoryOpClassMissing { record: String, n_addresses: usize },
    #[error(
        "{record}: memory_op_class is present but UNSPECIFIED; the tracer should have emitted an unsupported_memory_path exclusion instead"
    )]
    MemoryOpClassUnspecified { record: String },
    #[error(
        "{record}: memory_op_class {class:?} is not declared by the address-carrying v1 payload capability"
    )]
    MemoryPayloadClassUndeclared {
        record: String,
        class: pb::MemoryOpClass,
    },
    #[error(
        "{record}: address-carrying v1 {class:?} operand shape is invalid: {detail} (got {got})"
    )]
    MemoryPayloadShapeInvalid {
        record: String,
        class: pb::MemoryOpClass,
        detail: &'static str,
        got: usize,
    },
    #[error("{record}: memory_op_access_kind is present but UNSPECIFIED on a memory instruction")]
    MemoryOpAccessKindUnspecified { record: String },
    #[error("{record}: instruction_cache_op is present but UNSPECIFIED")]
    InstructionCacheOpUnspecified { record: String },
    #[error("{record}: non-memory instruction carries additive memory metadata in {field}")]
    MemoryMetadataOnNonMemoryInstruction { record: String, field: &'static str },
    #[error(
        "{record}: address operand {operand} explicitly carries MEMORY_SPACE_UNSPECIFIED (canonical: memory_space_unspecified)"
    )]
    MemorySpaceUnspecified { record: String, operand: usize },
    #[error(
        "{record}: address operand {operand} carries {space:?}, incompatible with memory_op_class {class:?} (memory_op_class_address_space_conflict)"
    )]
    MemoryOpClassAddressSpaceConflict {
        record: String,
        operand: usize,
        class: pb::MemoryOpClass,
        space: pb::MemorySpace,
    },
    #[error(
        "{record}: {field} carries value {raw} unknown to this reader's committed schema — \
         wire-schema skew; re-run sim/trace/regen_pb.sh against the current .proto and rebuild"
    )]
    UnknownEnumValue {
        record: String,
        field: &'static str,
        raw: i32,
    },
    #[error(
        "{record}: data_width absent on a memory instruction (canonical contract: data_width_missing)"
    )]
    DataWidthMissing { record: String },
    #[error(
        "{record}: data_width={got} is not strictly positive (canonical contract: data_width_invalid)"
    )]
    DataWidthInvalid { record: String, got: i32 },
    #[error("{record}: effective memory payload-byte projection failed: {source}")]
    EffectiveDataBytes {
        record: String,
        #[source]
        source: EffectiveDataBytesError,
    },
    #[error("{record}: shared-memory 32-bank/4-byte projection failed: {source}")]
    SharedBankProjection {
        record: String,
        #[source]
        source: SharedBankProjectionError,
    },
    #[error("{record}: LDGSTS destination-wavefront projection failed: {source}")]
    LdgstsDestinationProjection {
        record: String,
        #[source]
        source: LdgstsDestinationProjectionError,
    },
    #[error("{path}: threadblock proto is missing block_id (producer MUST populate it)")]
    BlockIdMissing { path: PathBuf },
    #[error(
        "{path}: block_id ({x},{y},{z}) flattens to {index}, outside the kernel's grid range [0, {grid})"
    )]
    BlockIdOutOfRange {
        path: PathBuf,
        x: i32,
        y: i32,
        z: i32,
        index: i128,
        grid: i64,
    },
    #[error(
        "{path}: flattened block index {index} already seen in this kernel dir (duplicate threadblock)"
    )]
    BlockIdDuplicate { path: PathBuf, index: i64 },
    #[error("kernel {kernel_id}: grid_dim absent or degenerate on the wire")]
    GridDimMissing { kernel_id: i32 },
    #[error("kernel {kernel_id}: {operation} overflows the signed launch-index domain")]
    GridArithmeticOverflow {
        kernel_id: i32,
        operation: &'static str,
    },
    #[error("{record}: {source}")]
    Lane {
        record: String,
        #[source]
        source: LaneError,
    },
    #[error(
        "{record}: GLOBAL/GENERIC-space record has active lanes but allocation_id \
         is absent or zero (canonical: allocation_id_missing)"
    )]
    AllocationIdMissing { record: String },
    #[error(
        "{record}: record carries a non-zero allocation_id but \
         <trace_dir>/extra_info/allocations.json is absent (allocations_sidecar_missing)"
    )]
    AllocationsSidecarMissing { record: String },
    #[error(
        "could not parse allocations sidecar {path}: {message} (allocations_sidecar_malformed)"
    )]
    AllocationsSidecarMalformed { path: PathBuf, message: String },
    #[error(
        "allocations sidecar {path} repeats allocation_id {allocation_id} (allocation_id_duplicate)"
    )]
    AllocationIdDuplicate { path: PathBuf, allocation_id: u32 },
    #[cfg(feature = "campaign")]
    #[error(
        "could not parse trace-sampling sidecar {path}: {message} (sampling_sidecar_malformed)"
    )]
    SamplingSidecarMalformed { path: PathBuf, message: String },
    #[error(
        "{record}: allocation_id {allocation_id} is not present in the allocations \
         sidecar (allocation_id_unknown)"
    )]
    AllocationIdUnknown { record: String, allocation_id: u32 },
    #[error(
        "{record}: lane {lane} address 0x{lane_addr:x} is outside allocation \
         {allocation_id} range [0x{base:x}, 0x{end:x}) (lane_address_outside_allocation_range)"
    )]
    LaneOutsideAllocationRange {
        record: String,
        lane: usize,
        lane_addr: u64,
        allocation_id: u32,
        base: u64,
        end: u128,
    },
    #[error("root Trace contains zero kernel launches")]
    TraceLaunchesEmpty,
    #[error("root device map key {map_key} disagrees with embedded gpu_device.id {embedded_id}")]
    DeviceMapKeyMismatch { map_key: i64, embedded_id: u32 },
    #[error("root stream map key {map_key} disagrees with embedded cuda_stream.id {embedded_id}")]
    StreamMapKeyMismatch { map_key: i64, embedded_id: i64 },
    #[error("root launch identity ({device_id}, {stream_id}, {kernel_id}) occurs more than once")]
    LaunchIdentityDuplicate {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "kernel locator ({device_id}, {stream_id}, {kernel_id}) does not belong to trace root {root}"
    )]
    KernelLocatorForeign {
        root: PathBuf,
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "root launch ({device_id}, {stream_id}, {kernel_id}) has absent or non-positive {field} geometry"
    )]
    LaunchGeometryInvalid {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        field: &'static str,
    },
    #[error("threadblock payload {path} has zero instruction records")]
    ThreadblockRecordsEmpty { path: PathBuf },
    #[error("threadblock payload {path} does not belong to any root launch")]
    ThreadblockPayloadOrphan { path: PathBuf },
    #[error(
        "{record}: function_unique_id {actual} disagrees with root launch function_unique_id {expected}"
    )]
    FunctionUniqueIdMismatch {
        record: String,
        expected: i32,
        actual: i32,
    },
    #[error("stats CSV {path} is empty")]
    StatsEmpty { path: PathBuf },
    #[error("stats CSV {path} has invalid schema: {message}")]
    StatsSchemaInvalid { path: PathBuf, message: String },
    #[error(
        "stats CSV {path} row {row} has invalid counters: expected={expected}, admitted={admitted}"
    )]
    StatsCountersInvalid {
        path: PathBuf,
        row: u64,
        expected: i128,
        admitted: i128,
    },
    #[error(
        "stats CSV {path} proves an empty payload: rows={rows}, expected={expected}, admitted={admitted}"
    )]
    TracePayloadEmpty {
        path: PathBuf,
        rows: u64,
        expected: u64,
        admitted: u64,
    },
    #[error("trace validation count {counter} overflows u64")]
    TraceCountOverflow { counter: &'static str },
}

impl TraceError {
    /// Stable cross-runtime diagnosis for failures owned by the forward wire
    /// contract. Filesystem and simulator-only errors intentionally return
    /// `None` rather than being mistaken for Python semantic parity.
    pub fn typed_reason(&self) -> Option<&'static str> {
        match self {
            Self::TraceRootEmpty { .. } => Some("trace_root_empty"),
            Self::Decode { message_type, .. }
            | Self::ProtobufWireMalformed { message_type, .. } => match *message_type {
                "dynamic_trace.Trace" => Some("trace_root_unparseable"),
                "dynamic_trace.threadblock" => Some("trace_threadblock_unparseable"),
                _ => None,
            },
            Self::ProtobufDecodedResourceLimitExceeded { .. } => {
                Some("protobuf_decoded_resource_limit_exceeded")
            }
            Self::TraceArtifactResourceLimitExceeded { .. } => {
                Some("trace_artifact_resource_limit_exceeded")
            }
            Self::TraceContractUnsupported { .. } => Some("trace_contract_mismatch"),
            Self::TraceLaunchSharedMemoryInvalid { .. } => {
                Some("trace_launch_shared_memory_invalid")
            }
            Self::SharedMemorySplitUnadvertised { .. } => Some("shared_memory_split_unadvertised"),
            Self::SharedMemorySplitUnavailable { .. } => Some("shared_memory_split_unavailable"),
            Self::SharedMemorySplitMissing { .. } => Some("shared_memory_split_missing"),
            Self::SharedMemorySplitMismatch { .. } => Some("shared_memory_split_mismatch"),
            Self::MemoryPayloadCapabilitiesInvalid { .. } => Some("trace_capabilities_invalid"),
            Self::WarpMapKeyMismatch { .. } => Some("warp_map_key_mismatch"),
            Self::WarpidTbMissing { .. } => Some("warpid_tb_missing"),
            Self::WarpidTbMismatch { .. } => Some("warpid_tb_mismatch"),
            Self::FunctionUniqueIdInvalid { .. } => Some("function_unique_id_invalid"),
            Self::ActiveMaskMissing { .. } => Some("active_mask_missing"),
            Self::PredicateMaskMissing { .. } => Some("predicate_mask_missing"),
            Self::VpcMissing { .. } => Some("vpc_missing"),
            Self::VpcInconsistentWithPc { .. } => Some("vpc_inconsistent_with_pc"),
            Self::LaneMaskInconsistent { .. } => Some("lane_mask_inconsistent"),
            Self::MemoryOpClassMissing { .. } => {
                Some("memory_op_class_missing_on_memory_instruction")
            }
            Self::MemoryOpClassUnspecified { .. } => Some("memory_op_class_unspecified"),
            Self::MemoryPayloadClassUndeclared { .. } => Some("memory_payload_class_undeclared"),
            Self::MemoryPayloadShapeInvalid { .. } => Some("memory_payload_shape_invalid"),
            Self::MemoryOpAccessKindUnspecified { .. } => Some("memory_op_access_kind_unspecified"),
            Self::InstructionCacheOpUnspecified { .. } => Some("instruction_cache_op_unspecified"),
            Self::MemoryMetadataOnNonMemoryInstruction { .. } => {
                Some("memory_metadata_on_non_memory_instruction")
            }
            Self::MemorySpaceUnspecified { .. } => Some("memory_space_unspecified"),
            Self::MemoryOpClassAddressSpaceConflict { .. } => {
                Some("memory_op_class_address_space_conflict")
            }
            Self::UnknownEnumValue { .. } => Some("unknown_enum_value"),
            Self::DataWidthMissing { .. } => Some("data_width_missing"),
            Self::DataWidthInvalid { .. } => Some("data_width_invalid"),
            Self::BlockIdMissing { .. } => Some("block_id_missing"),
            Self::BlockIdOutOfRange { .. } => Some("block_id_out_of_range"),
            Self::BlockIdDuplicate { .. } => Some("block_id_duplicate"),
            Self::GridDimMissing { .. } => Some("trace_launch_geometry_invalid"),
            Self::GridArithmeticOverflow { .. } => Some("grid_arithmetic_overflow"),
            Self::Lane { source, .. } => match source {
                LaneError::ListAllLengthInvalid { .. } => Some("lane_vector_length_invalid"),
                LaneError::LanePositionMismatch { .. } => Some("lane_position_mismatch"),
                LaneError::BaseStrideAddrsNonEmpty { .. }
                | LaneError::BaseStrideSingleLane
                | LaneError::BaseStrideNonContiguous { .. }
                | LaneError::BaseDeltaCountInvalid { .. }
                | LaneError::AddressOverflow { .. }
                | LaneError::UnknownFormat(_) => Some("compressed_format_malformed"),
            },
            Self::AllocationIdMissing { .. } => Some("allocation_id_missing"),
            Self::AllocationsSidecarMissing { .. } => Some("allocations_sidecar_missing"),
            Self::AllocationsSidecarMalformed { .. } => Some("allocations_sidecar_malformed"),
            Self::AllocationIdDuplicate { .. } => Some("allocation_id_duplicate"),
            Self::AllocationIdUnknown { .. } => Some("allocation_id_unknown"),
            Self::LaneOutsideAllocationRange { .. } => {
                Some("lane_address_outside_allocation_range")
            }
            Self::TraceLaunchesEmpty => Some("trace_launches_empty"),
            Self::TraceThreadblocksEmpty { .. } => Some("trace_threadblocks_empty"),
            Self::ThreadblocksMissing { .. } => Some("trace_launch_payload_missing"),
            Self::DeviceMapKeyMismatch { .. }
            | Self::StreamMapKeyMismatch { .. }
            | Self::LaunchIdentityDuplicate { .. } => Some("trace_launch_identity_mismatch"),
            Self::KernelLocatorForeign { .. } => Some("kernel_locator_foreign"),
            Self::LaunchGeometryInvalid { .. } => Some("trace_launch_geometry_invalid"),
            Self::ThreadblockRecordsEmpty { .. } => Some("trace_threadblock_empty"),
            Self::ThreadblockPayloadOrphan { .. } => Some("trace_payload_orphan"),
            Self::FunctionUniqueIdMismatch { .. } => Some("function_unique_id_mismatch"),
            Self::StatsEmpty { .. } => Some("stats_empty"),
            Self::StatsSchemaInvalid { .. } => Some("stats_schema_invalid"),
            Self::StatsCountersInvalid { .. } => Some("stats_counters_invalid"),
            Self::TracePayloadEmpty { .. } => Some("trace_payload_empty"),
            Self::TraceCountOverflow { .. } => Some("trace_counter_overflow"),
            _ => None,
        }
    }
}

fn map_wire_preflight_error(
    path: &Path,
    message_type: &'static str,
    error: WirePreflightError,
) -> TraceError {
    match error {
        WirePreflightError::Malformed(detail) => TraceError::ProtobufWireMalformed {
            path: path.to_path_buf(),
            message_type,
            detail,
        },
        WirePreflightError::ResourceLimit {
            resource,
            observed,
            limit,
        } => TraceError::ProtobufDecodedResourceLimitExceeded {
            path: path.to_path_buf(),
            message_type,
            resource,
            observed,
            limit,
        },
    }
}

/// One kernel launch located inside the root `Trace` message. Fields are private
/// so callers cannot pair arbitrary geometry with a trace payload; consuming
/// `TraceDir` methods also verify pointer identity against their own root.
///
/// ```compile_fail
/// use sim_trace::{KernelLoc, pb};
///
/// fn forge(kernel: &pb::Kernel) -> KernelLoc<'_> {
///     KernelLoc {
///         device_id: 0,
///         stream_id: 0,
///         kernel,
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KernelLoc<'a> {
    device_id: i64,
    stream_id: i64,
    kernel: &'a pb::Kernel,
}

impl<'a> KernelLoc<'a> {
    pub fn device_id(self) -> i64 {
        self.device_id
    }

    pub fn stream_id(self) -> i64 {
        self.stream_id
    }

    pub fn kernel(self) -> &'a pb::Kernel {
        self.kernel
    }
}

/// Non-vacuous producer counters parsed from snapshotted `stats.csv` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceStats {
    row_count: u64,
    expected_record_count: u64,
    admitted_record_count: u64,
}

impl TraceStats {
    pub fn row_count(self) -> u64 {
        self.row_count
    }

    pub fn expected_record_count(self) -> u64 {
        self.expected_record_count
    }

    pub fn admitted_record_count(self) -> u64 {
        self.admitted_record_count
    }
}

/// Counts produced by the complete immutable-root and payload semantic walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceValidationSummary {
    launch_count: u64,
    threadblock_count: u64,
    instruction_count: u64,
}

impl TraceValidationSummary {
    pub fn launch_count(self) -> u64 {
        self.launch_count
    }

    pub fn threadblock_count(self) -> u64 {
        self.threadblock_count
    }

    pub fn instruction_count(self) -> u64 {
        self.instruction_count
    }
}

pub const ADDRESS_CARRYING_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_v1";
pub const SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_shmem_split_v2";
pub const LDGSTS_DESTINATION_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_ldgsts_dst_v3";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPayloadContract {
    LegacyExclusion,
    AddressCarryingV1,
    AddressCarryingSharedMemorySplitV2,
    AddressCarryingLdgstsDestinationV3,
}

impl MemoryPayloadContract {
    const fn carries_atomic_addresses(self) -> bool {
        !matches!(self, Self::LegacyExclusion)
    }

    const fn carries_shared_memory_split(self) -> bool {
        matches!(
            self,
            Self::AddressCarryingSharedMemorySplitV2 | Self::AddressCarryingLdgstsDestinationV3
        )
    }

    const fn carries_ldgsts_destination(self) -> bool {
        matches!(self, Self::AddressCarryingLdgstsDestinationV3)
    }
}

const ADDRESS_CARRYING_PAYLOAD_CLASSES: [pb::MemoryOpClass; 5] = [
    pb::MemoryOpClass::Global,
    pb::MemoryOpClass::Shared,
    pb::MemoryOpClass::Local,
    pb::MemoryOpClass::Atomic,
    pb::MemoryOpClass::AsyncCopy,
];
const ADDRESS_CARRYING_PAYLOAD_CAPABILITY_VECTOR: [i32; 5] = [1, 2, 3, 6, 8];

fn classify_memory_payload_contract(
    trace: &pb::Trace,
    path: &Path,
) -> Result<MemoryPayloadContract, TraceError> {
    if trace.trace_contract_id.is_empty() {
        if trace.memory_payload_classes.is_empty() {
            return Ok(MemoryPayloadContract::LegacyExclusion);
        }
        return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
            path: path.to_path_buf(),
            contract_id: trace.trace_contract_id.clone(),
            got: trace.memory_payload_classes.clone(),
        });
    }
    let contract = match trace.trace_contract_id.as_str() {
        ADDRESS_CARRYING_TRACE_CONTRACT_ID => MemoryPayloadContract::AddressCarryingV1,
        SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID => {
            MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
        }
        LDGSTS_DESTINATION_TRACE_CONTRACT_ID => {
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
        }
        _ => {
            return Err(TraceError::TraceContractUnsupported {
                path: path.to_path_buf(),
                got: trace.trace_contract_id.clone(),
            });
        }
    };

    // V2 and V3 use the repeated enum field as an ordered capability vector,
    // not a set. Keep v1's historical set-based admission below unchanged.
    if contract.carries_shared_memory_split()
        && !trace
            .memory_payload_classes
            .iter()
            .copied()
            .eq(ADDRESS_CARRYING_PAYLOAD_CAPABILITY_VECTOR)
    {
        return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
            path: path.to_path_buf(),
            contract_id: trace.trace_contract_id.clone(),
            got: trace.memory_payload_classes.clone(),
        });
    }

    let expected: BTreeSet<_> = ADDRESS_CARRYING_PAYLOAD_CLASSES.into_iter().collect();
    let mut found = BTreeSet::new();
    for &raw in &trace.memory_payload_classes {
        let Ok(class) = pb::MemoryOpClass::try_from(raw) else {
            return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
                path: path.to_path_buf(),
                contract_id: trace.trace_contract_id.clone(),
                got: trace.memory_payload_classes.clone(),
            });
        };
        if !found.insert(class) {
            return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
                path: path.to_path_buf(),
                contract_id: trace.trace_contract_id.clone(),
                got: trace.memory_payload_classes.clone(),
            });
        }
    }
    if found != expected {
        return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
            path: path.to_path_buf(),
            contract_id: trace.trace_contract_id.clone(),
            got: trace.memory_payload_classes.clone(),
        });
    }
    Ok(contract)
}

/// Raw-byte identity of one snapshotted trace artifact.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactDigestIdentity {
    size_bytes: u64,
    sha256: Sha256Digest,
}

#[cfg(feature = "campaign")]
impl ArtifactDigestIdentity {
    fn from_bytes(bytes: &[u8], path: &Path) -> Result<Self, TraceError> {
        let size_bytes = u64::try_from(bytes.len()).map_err(|_| {
            TraceError::ThreadblockPayloadSnapshotInvalid {
                root: path.to_path_buf(),
                message: "artifact byte length exceeds u64".to_owned(),
            }
        })?;
        Ok(Self {
            size_bytes,
            sha256: Sha256Digest::of(bytes),
        })
    }

    pub fn size_bytes(self) -> u64 {
        self.size_bytes
    }

    pub fn sha256(self) -> Sha256Digest {
        self.sha256
    }
}

/// Deterministic identity of every threadblock payload file in a trace snapshot.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadblockPayloadIdentity {
    file_count: u64,
    raw_bytes: u64,
    sorted_path_manifest_sha256: Sha256Digest,
    sorted_path_stream_sha256: Sha256Digest,
}

#[cfg(feature = "campaign")]
impl ThreadblockPayloadIdentity {
    pub fn file_count(&self) -> u64 {
        self.file_count
    }

    pub fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    pub fn sorted_path_manifest_sha256(&self) -> Sha256Digest {
        self.sorted_path_manifest_sha256
    }

    pub fn sorted_path_stream_sha256(&self) -> Sha256Digest {
        self.sorted_path_stream_sha256
    }
}

/// Raw identities captured before any trace artifact is decoded.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceArtifactIdentity {
    dynamic_trace_pb: ArtifactDigestIdentity,
    allocations_json: Option<ArtifactDigestIdentity>,
    trace_sampling_json: Option<ArtifactDigestIdentity>,
    stats_csv: Option<ArtifactDigestIdentity>,
    threadblock_payload: ThreadblockPayloadIdentity,
}

#[cfg(feature = "campaign")]
impl TraceArtifactIdentity {
    pub fn dynamic_trace_pb(&self) -> ArtifactDigestIdentity {
        self.dynamic_trace_pb
    }

    pub fn allocations_json(&self) -> Option<ArtifactDigestIdentity> {
        self.allocations_json
    }

    pub fn trace_sampling_json(&self) -> Option<ArtifactDigestIdentity> {
        self.trace_sampling_json
    }

    pub fn stats_csv(&self) -> Option<ArtifactDigestIdentity> {
        self.stats_csv
    }

    pub fn threadblock_payload(&self) -> &ThreadblockPayloadIdentity {
        &self.threadblock_payload
    }
}

#[cfg(feature = "campaign")]
struct VerifiedSamplingPayload {
    by_path: BTreeMap<PathBuf, Vec<u8>>,
    identity: ThreadblockPayloadIdentity,
}

/// Declared byte/cardinality bounds used before a manifest snapshot reads data.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceSnapshotLimits {
    dynamic_trace_pb_size: u64,
    allocations_json_size: Option<u64>,
    trace_sampling_json_size: Option<u64>,
    stats_csv_size: Option<u64>,
    threadblock_file_count: u64,
    threadblock_raw_bytes: u64,
}

#[cfg(feature = "campaign")]
impl TraceSnapshotLimits {
    pub fn new(
        dynamic_trace_pb_size: u64,
        allocations_json_size: Option<u64>,
        trace_sampling_json_size: Option<u64>,
        stats_csv_size: Option<u64>,
        threadblock_file_count: u64,
        threadblock_raw_bytes: u64,
    ) -> Self {
        Self {
            dynamic_trace_pb_size,
            allocations_json_size,
            trace_sampling_json_size,
            stats_csv_size,
            threadblock_file_count,
            threadblock_raw_bytes,
        }
    }
}

/// One-read snapshot used by manifest admission. Artifact identities are
/// available before parsing, and [`TraceDir::from_snapshot`] consumes these exact
/// bytes so a hash-then-reopen substitution cannot change replay inputs.
#[cfg(feature = "campaign")]
pub struct TraceDirSnapshot {
    root: PathBuf,
    dynamic_trace_pb: Vec<u8>,
    allocations_json: Option<Vec<u8>>,
    trace_sampling_json: Option<Vec<u8>>,
    stats_csv: Option<Vec<u8>>,
    threadblock_payload: VerifiedSamplingPayload,
    identity: TraceArtifactIdentity,
    file_identities: BTreeMap<FileIdentity, (PathBuf, FileGenerationIdentity)>,
}

#[cfg(feature = "campaign")]
impl TraceDirSnapshot {
    pub fn capture(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        let root = root.into();
        let reader = AnchoredDir::open(&root)?;
        Self::capture_from_anchored(&reader, Path::new(""), None)
    }

    pub fn capture_anchored(
        reader: &AnchoredDir,
        root_relative: &Path,
        limits: TraceSnapshotLimits,
    ) -> Result<Self, TraceError> {
        Self::capture_from_anchored(reader, root_relative, Some(limits))
    }

    fn capture_from_anchored(
        reader: &AnchoredDir,
        root_relative: &Path,
        limits: Option<TraceSnapshotLimits>,
    ) -> Result<Self, TraceError> {
        let root = reader.root().join(root_relative);
        let preserve_manifest_artifact_role = limits.is_some();
        let dynamic_relative = root_relative.join("dynamic_trace.pb");
        let dynamic_file = match limits {
            Some(limits) => reader
                .read_exact_regular(&dynamic_relative, limits.dynamic_trace_pb_size)
                .map_err(|source| map_manifest_snapshot_read_error("dynamic_trace_pb", source))?,
            None => reader.read_bounded_regular(&dynamic_relative, u64::MAX)?,
        };
        let dynamic_path = root.join("dynamic_trace.pb");
        let mut file_identities = BTreeMap::new();
        insert_snapshot_file_identity(
            &mut file_identities,
            "dynamic_trace_pb",
            preserve_manifest_artifact_role,
            dynamic_file.generation_identity(),
            dynamic_path.clone(),
        )?;
        let dynamic_trace_pb = dynamic_file.into_raw();
        // Root-byte failures precede payload-tree failures in both validators.
        // Keep this byte-level check at snapshot capture so an empty root cannot
        // be masked by an independently empty/missing payload tree.
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty { path: dynamic_path });
        }

        let allocations_relative = root_relative.join("extra_info/allocations.json");
        let allocations_path = root.join("extra_info/allocations.json");
        let allocations_file = capture_optional_snapshot_artifact(
            reader,
            &allocations_relative,
            limits.map(|limits| limits.allocations_json_size),
            "allocations_json",
            ALLOCATIONS_JSON_MAX_BYTES,
        )?;
        if let Some(file) = &allocations_file {
            insert_snapshot_file_identity(
                &mut file_identities,
                "allocations_json",
                preserve_manifest_artifact_role,
                file.generation_identity(),
                allocations_path.clone(),
            )?;
        }
        let allocations_json = allocations_file.map(AnchoredFile::into_raw);

        let sampling_relative = root_relative.join("trace_sampling.json");
        let sampling_path = root.join("trace_sampling.json");
        let sampling_file = capture_optional_snapshot_artifact(
            reader,
            &sampling_relative,
            limits.map(|limits| limits.trace_sampling_json_size),
            "trace_sampling_json",
            TRACE_SAMPLING_JSON_MAX_BYTES,
        )?;
        if let Some(file) = &sampling_file {
            insert_snapshot_file_identity(
                &mut file_identities,
                "trace_sampling_json",
                preserve_manifest_artifact_role,
                file.generation_identity(),
                sampling_path.clone(),
            )?;
        }
        let trace_sampling_json = sampling_file.map(AnchoredFile::into_raw);

        let stats_relative = root_relative.join("stats.csv");
        let stats_path = root.join("stats.csv");
        let stats_file = capture_optional_snapshot_artifact(
            reader,
            &stats_relative,
            limits.map(|limits| limits.stats_csv_size),
            "stats_csv",
            STATS_CSV_MAX_BYTES,
        )?;
        if let Some(file) = &stats_file {
            insert_snapshot_file_identity(
                &mut file_identities,
                "stats_csv",
                preserve_manifest_artifact_role,
                file.generation_identity(),
                stats_path.clone(),
            )?;
        }
        let stats_csv = stats_file.map(AnchoredFile::into_raw);

        let (maximum_files, maximum_bytes) = limits.map_or((u64::MAX, u64::MAX), |limits| {
            (limits.threadblock_file_count, limits.threadblock_raw_bytes)
        });
        let tree_relative = root_relative.join("threadblocks");
        let mut tree = reader
            .read_pb_tree(&tree_relative, maximum_files, maximum_bytes)
            .map_err(|source| {
                map_threadblock_tree_read_error(&root, source, preserve_manifest_artifact_role)
            })?;
        let actual_file_count = u64::try_from(tree.files.len()).map_err(|_| {
            TraceError::SnapshotArtifactCardinalityMismatch {
                artifact: "threadblock_payload",
                expected: maximum_files,
                actual: u64::MAX,
            }
        })?;
        if let Some(limits) = limits {
            if actual_file_count != limits.threadblock_file_count {
                return Err(TraceError::SnapshotArtifactCardinalityMismatch {
                    artifact: "threadblock_payload",
                    expected: limits.threadblock_file_count,
                    actual: actual_file_count,
                });
            }
            if tree.bytes != limits.threadblock_raw_bytes {
                return Err(TraceError::SnapshotArtifactSizeMismatch {
                    artifact: "threadblock_payload",
                    expected: limits.threadblock_raw_bytes,
                    actual: tree.bytes,
                });
            }
        }
        if tree.files.is_empty() {
            return Err(TraceError::TraceThreadblocksEmpty {
                dir: root.join("threadblocks"),
            });
        }
        // Python's producer hashes slash-separated manifest path spellings, not
        // component-wise `Path` order (`a.pb` sorts before `a/b.pb` there).
        // Re-establish that exact order after the anchored tree walk.
        tree.files
            .sort_by(|left, right| left.relative.to_str().cmp(&right.relative.to_str()));
        let mut raw_bytes = 0_u64;
        let mut stream_digest = Sha256::new();
        let mut manifest = String::from("trace-payload-file-manifest-v1\n");
        let mut by_path = BTreeMap::new();
        for entry in tree.files {
            let relative = entry.relative.strip_prefix(root_relative).map_err(|_| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!(
                        "descriptor-anchored payload path is outside trace root: {}",
                        entry.relative.display()
                    ),
                }
            })?;
            let relative =
                relative
                    .to_str()
                    .ok_or_else(|| TraceError::ThreadblockPayloadSnapshotInvalid {
                        root: root.clone(),
                        message: format!("payload path is not UTF-8: {}", entry.relative.display()),
                    })?;
            if !safe_payload_manifest_path(relative) {
                return Err(TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!(
                        "payload path is not canonical for the per-file manifest: {relative:?}"
                    ),
                });
            }
            let path = root.join(relative);
            insert_snapshot_file_identity(
                &mut file_identities,
                "threadblock_payload",
                preserve_manifest_artifact_role,
                entry.file.generation_identity(),
                path.clone(),
            )?;
            let raw = entry.file.into_raw();
            let size = u64::try_from(raw.len()).map_err(|_| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!("payload {} length exceeds u64", path.display()),
                }
            })?;
            raw_bytes = raw_bytes.checked_add(size).ok_or_else(|| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: "payload byte count overflows u64".to_owned(),
                }
            })?;
            let file_digest = Sha256Digest::of(&raw);
            manifest.push_str(&format!("{relative}\t{size}\t{file_digest}\n"));
            stream_digest.update(&raw);
            by_path.insert(path, raw);
        }
        let threadblock_payload = VerifiedSamplingPayload {
            by_path,
            identity: ThreadblockPayloadIdentity {
                file_count: actual_file_count,
                raw_bytes,
                sorted_path_manifest_sha256: Sha256Digest::of(manifest.as_bytes()),
                sorted_path_stream_sha256: stream_digest.finish_digest(),
            },
        };
        let identity = TraceArtifactIdentity {
            dynamic_trace_pb: ArtifactDigestIdentity::from_bytes(&dynamic_trace_pb, &dynamic_path)?,
            allocations_json: allocations_json
                .as_deref()
                .map(|raw| ArtifactDigestIdentity::from_bytes(raw, &allocations_path))
                .transpose()?,
            trace_sampling_json: trace_sampling_json
                .as_deref()
                .map(|raw| ArtifactDigestIdentity::from_bytes(raw, &sampling_path))
                .transpose()?,
            stats_csv: stats_csv
                .as_deref()
                .map(|raw| ArtifactDigestIdentity::from_bytes(raw, &stats_path))
                .transpose()?,
            threadblock_payload: threadblock_payload.identity.clone(),
        };
        Ok(Self {
            root,
            dynamic_trace_pb,
            allocations_json,
            trace_sampling_json,
            stats_csv,
            threadblock_payload,
            identity,
            file_identities,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn artifact_identity(&self) -> &TraceArtifactIdentity {
        &self.identity
    }

    pub fn file_identities(&self) -> impl Iterator<Item = (FileIdentity, &Path)> {
        self.file_identities
            .iter()
            .map(|(identity, (path, _))| (*identity, path.as_path()))
    }

    /// Every file generation whose exact bytes entered this snapshot.
    pub fn file_generations(&self) -> impl Iterator<Item = (FileGenerationIdentity, &Path)> {
        self.file_identities
            .values()
            .map(|(path, generation)| (*generation, path.as_path()))
    }
}

fn map_threadblock_tree_read_error(
    root: &Path,
    source: AnchoredReadError,
    _preserve_manifest_artifact_role: bool,
) -> TraceError {
    match source {
        AnchoredReadError::Io { path, source }
            if path == root.join("threadblocks")
                && source.kind() == std::io::ErrorKind::NotFound =>
        {
            TraceError::TraceThreadblocksEmpty { dir: path }
        }
        #[cfg(feature = "campaign")]
        other if _preserve_manifest_artifact_role => {
            map_manifest_snapshot_read_error("threadblock_payload", other)
        }
        other => TraceError::AnchoredSnapshot(other),
    }
}

fn map_snapshot_read_error_with(
    artifact: &'static str,
    source: AnchoredReadError,
    anchored: impl FnOnce(AnchoredReadError) -> TraceError,
) -> TraceError {
    match source {
        AnchoredReadError::SizeMismatch {
            expected, actual, ..
        } => TraceError::SnapshotArtifactSizeMismatch {
            artifact,
            expected,
            actual,
        },
        AnchoredReadError::SizeLimitExceeded {
            path,
            limit,
            actual,
        } => TraceError::TraceArtifactResourceLimitExceeded {
            path,
            artifact,
            resource: "bytes",
            observed: actual,
            limit,
        },
        other => anchored(other),
    }
}

fn map_snapshot_read_error(artifact: &'static str, source: AnchoredReadError) -> TraceError {
    map_snapshot_read_error_with(artifact, source, TraceError::AnchoredSnapshot)
}

#[cfg(feature = "campaign")]
fn map_manifest_snapshot_read_error(
    artifact: &'static str,
    source: AnchoredReadError,
) -> TraceError {
    map_snapshot_read_error_with(artifact, source, |source| {
        TraceError::SnapshotArtifactRead { artifact, source }
    })
}

#[cfg(feature = "campaign")]
fn capture_optional_snapshot_artifact(
    reader: &AnchoredDir,
    relative: &Path,
    expected: Option<Option<u64>>,
    artifact: &'static str,
    maximum_size: u64,
) -> Result<Option<AnchoredFile>, TraceError> {
    let map_read_error = |source| {
        if expected.is_some() {
            map_manifest_snapshot_read_error(artifact, source)
        } else {
            map_snapshot_read_error(artifact, source)
        }
    };
    match expected {
        None => reader
            .read_optional_bounded_regular(relative, maximum_size)
            .map_err(map_read_error),
        Some(Some(size)) => {
            if size > maximum_size {
                return Err(TraceError::TraceArtifactResourceLimitExceeded {
                    path: reader.root().join(relative),
                    artifact,
                    resource: "bytes",
                    observed: size,
                    limit: maximum_size,
                });
            }
            reader
                .read_optional_exact_regular(relative, size)
                .map_err(map_read_error)?
                .ok_or(TraceError::SnapshotArtifactPresenceMismatch {
                    artifact,
                    expected: true,
                    actual: false,
                })
                .map(Some)
        }
        Some(None) => {
            if reader
                .probe_optional_regular(relative)
                .map_err(map_read_error)?
                .is_some()
            {
                return Err(TraceError::SnapshotArtifactPresenceMismatch {
                    artifact,
                    expected: false,
                    actual: true,
                });
            }
            Ok(None)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JsonShape {
    nodes: u64,
    top_level_sequence_entries: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JsonResourceLimit {
    resource: &'static str,
    observed: u64,
    limit: u64,
}

#[derive(Debug)]
enum JsonPreflightError {
    Malformed(String),
    ResourceLimit(JsonResourceLimit),
}

struct JsonShapeCounter {
    nodes: u64,
    top_level_sequence_entries: u64,
    maximum_nodes: u64,
    maximum_top_level_sequence_entries: Option<u64>,
    exceeded: Option<JsonResourceLimit>,
}

impl JsonShapeCounter {
    fn bump_nodes<E: serde::de::Error>(&mut self) -> Result<(), E> {
        self.nodes = self.nodes.checked_add(1).unwrap_or(u64::MAX);
        if self.nodes > self.maximum_nodes {
            let exceeded = JsonResourceLimit {
                resource: "json_nodes",
                observed: self.nodes,
                limit: self.maximum_nodes,
            };
            self.exceeded = Some(exceeded);
            return Err(E::custom("JSON node limit exceeded"));
        }
        Ok(())
    }

    fn bump_top_level_sequence_entries<E: serde::de::Error>(&mut self) -> Result<(), E> {
        self.top_level_sequence_entries = self
            .top_level_sequence_entries
            .checked_add(1)
            .unwrap_or(u64::MAX);
        if let Some(limit) = self.maximum_top_level_sequence_entries
            && self.top_level_sequence_entries > limit
        {
            let exceeded = JsonResourceLimit {
                resource: "top_level_entries",
                observed: self.top_level_sequence_entries,
                limit,
            };
            self.exceeded = Some(exceeded);
            return Err(E::custom("JSON top-level entry limit exceeded"));
        }
        Ok(())
    }
}

struct JsonValueSeed<'a> {
    counter: &'a mut JsonShapeCounter,
    root: bool,
    top_level_sequence_entry: bool,
}

impl<'de> DeserializeSeed<'de> for JsonValueSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.top_level_sequence_entry {
            self.counter.bump_top_level_sequence_entries()?;
        }
        self.counter.bump_nodes()?;
        deserializer.deserialize_any(JsonValueVisitor {
            counter: self.counter,
            root: self.root,
        })
    }
}

struct JsonValueVisitor<'a> {
    counter: &'a mut JsonShapeCounter,
    root: bool,
}

impl<'de> Visitor<'de> for JsonValueVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        JsonValueSeed {
            counter: self.counter,
            root: false,
            top_level_sequence_entry: false,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence
            .next_element_seed(JsonValueSeed {
                counter: self.counter,
                root: false,
                top_level_sequence_entry: self.root,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while object.next_key::<IgnoredAny>()?.is_some() {
            object.next_value_seed(JsonValueSeed {
                counter: self.counter,
                root: false,
                top_level_sequence_entry: false,
            })?;
        }
        Ok(())
    }
}

fn preflight_json_shape(
    raw: &[u8],
    maximum_nodes: u64,
    maximum_top_level_sequence_entries: Option<u64>,
) -> Result<JsonShape, JsonPreflightError> {
    let mut counter = JsonShapeCounter {
        nodes: 0,
        top_level_sequence_entries: 0,
        maximum_nodes,
        maximum_top_level_sequence_entries,
        exceeded: None,
    };
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let parsed = JsonValueSeed {
        counter: &mut counter,
        root: true,
        top_level_sequence_entry: false,
    }
    .deserialize(&mut deserializer);
    if let Some(exceeded) = counter.exceeded {
        return Err(JsonPreflightError::ResourceLimit(exceeded));
    }
    parsed.map_err(|error| JsonPreflightError::Malformed(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| JsonPreflightError::Malformed(error.to_string()))?;
    Ok(JsonShape {
        nodes: counter.nodes,
        top_level_sequence_entries: counter.top_level_sequence_entries,
    })
}

fn sidecar_byte_limit(
    path: &Path,
    artifact: &'static str,
    raw: &[u8],
    maximum_bytes: u64,
) -> Result<(), TraceError> {
    let observed = u64::try_from(raw.len()).unwrap_or(u64::MAX);
    if observed > maximum_bytes {
        return Err(TraceError::TraceArtifactResourceLimitExceeded {
            path: path.to_path_buf(),
            artifact,
            resource: "bytes",
            observed,
            limit: maximum_bytes,
        });
    }
    Ok(())
}

fn map_json_preflight_limit(
    path: &Path,
    artifact: &'static str,
    error: JsonResourceLimit,
) -> TraceError {
    TraceError::TraceArtifactResourceLimitExceeded {
        path: path.to_path_buf(),
        artifact,
        resource: error.resource,
        observed: error.observed,
        limit: error.limit,
    }
}

#[cfg(feature = "campaign")]
fn insert_snapshot_file_identity(
    identities: &mut BTreeMap<FileIdentity, (PathBuf, FileGenerationIdentity)>,
    artifact: &'static str,
    preserve_manifest_artifact_role: bool,
    generation: FileGenerationIdentity,
    path: PathBuf,
) -> Result<(), TraceError> {
    if let Some((first, _)) =
        identities.insert(generation.file_identity(), (path.clone(), generation))
    {
        let source = AnchoredReadError::HardlinkAlias {
            first,
            second: path,
        };
        return Err(if preserve_manifest_artifact_role {
            TraceError::SnapshotArtifactRead { artifact, source }
        } else {
            TraceError::AnchoredSnapshot(source)
        });
    }
    Ok(())
}

fn parse_ascii_signed_decimal(text: &str) -> Option<i128> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

fn parse_trace_stats(path: &Path, raw: &[u8]) -> Result<TraceStats, TraceError> {
    if raw.is_empty() {
        return Err(TraceError::StatsEmpty {
            path: path.to_path_buf(),
        });
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(raw);
    let headers = reader
        .headers()
        .map_err(|error| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?
        .iter()
        .map(str::trim)
        .collect::<Vec<_>>();
    let expected_index = headers
        .iter()
        .position(|header| *header == "total_insts")
        .ok_or_else(|| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: "stats.csv lacks total_insts".to_owned(),
        })?;
    let admitted_index = headers
        .iter()
        .position(|header| *header == "total_reported_insts")
        .ok_or_else(|| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: "stats.csv lacks total_reported_insts".to_owned(),
        })?;

    let mut rows = 0_u64;
    let mut expected_total = 0_u64;
    let mut admitted_total = 0_u64;
    for (record_index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        if record.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        let row = u64::try_from(record_index)
            .ok()
            .and_then(|index| index.checked_add(2))
            .ok_or(TraceError::TraceCountOverflow {
                counter: "stats row number",
            })?;
        let expected_text =
            record
                .get(expected_index)
                .ok_or_else(|| TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row} is truncated"),
                })?;
        let admitted_text =
            record
                .get(admitted_index)
                .ok_or_else(|| TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row} is truncated"),
                })?;
        let expected_signed =
            parse_ascii_signed_decimal(expected_text.trim()).ok_or_else(|| {
                TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row} has a non-integer or out-of-domain total_insts"),
                }
            })?;
        let admitted_signed =
            parse_ascii_signed_decimal(admitted_text.trim()).ok_or_else(|| {
                TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "row {row} has a non-integer or out-of-domain total_reported_insts"
                    ),
                }
            })?;
        if expected_signed < 0 || admitted_signed < 0 || admitted_signed > expected_signed {
            return Err(TraceError::StatsCountersInvalid {
                path: path.to_path_buf(),
                row,
                expected: expected_signed,
                admitted: admitted_signed,
            });
        }
        let expected =
            u64::try_from(expected_signed).map_err(|_| TraceError::StatsSchemaInvalid {
                path: path.to_path_buf(),
                message: format!("row {row} total_insts exceeds u64"),
            })?;
        let admitted =
            u64::try_from(admitted_signed).map_err(|_| TraceError::StatsSchemaInvalid {
                path: path.to_path_buf(),
                message: format!("row {row} total_reported_insts exceeds u64"),
            })?;
        expected_total =
            expected_total
                .checked_add(expected)
                .ok_or(TraceError::TraceCountOverflow {
                    counter: "stats expected total",
                })?;
        admitted_total =
            admitted_total
                .checked_add(admitted)
                .ok_or(TraceError::TraceCountOverflow {
                    counter: "stats admitted total",
                })?;
        rows = rows.checked_add(1).ok_or(TraceError::TraceCountOverflow {
            counter: "stats row count",
        })?;
    }
    if rows == 0 || expected_total == 0 || admitted_total == 0 {
        return Err(TraceError::TracePayloadEmpty {
            path: path.to_path_buf(),
            rows,
            expected: expected_total,
            admitted: admitted_total,
        });
    }
    Ok(TraceStats {
        row_count: rows,
        expected_record_count: expected_total,
        admitted_record_count: admitted_total,
    })
}

fn positive_geometry(dim: Option<&pb::Dim3d>) -> bool {
    dim.is_some_and(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
}

fn validate_shared_memory_split(
    device_id: i64,
    stream_id: i64,
    kernel: &pb::Kernel,
    contract: MemoryPayloadContract,
) -> Result<(), TraceError> {
    if kernel.size_shared_memory < 0 {
        return Err(TraceError::TraceLaunchSharedMemoryInvalid {
            device_id,
            stream_id,
            kernel_id: kernel.id,
            combined: kernel.size_shared_memory,
        });
    }

    match contract {
        MemoryPayloadContract::LegacyExclusion | MemoryPayloadContract::AddressCarryingV1 => {
            if kernel.shared_memory_static_bytes.is_some()
                || kernel.shared_memory_dynamic_bytes.is_some()
            {
                return Err(TraceError::SharedMemorySplitUnadvertised {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                });
            }
            if kernel.size_shared_memory != 0 {
                return Err(TraceError::SharedMemorySplitUnavailable {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                });
            }
        }
        MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
        | MemoryPayloadContract::AddressCarryingLdgstsDestinationV3 => {
            let (Some(static_bytes), Some(dynamic_bytes)) = (
                kernel.shared_memory_static_bytes,
                kernel.shared_memory_dynamic_bytes,
            ) else {
                return Err(TraceError::SharedMemorySplitMissing {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                });
            };
            let sum = u64::from(static_bytes) + u64::from(dynamic_bytes);
            if sum != kernel.size_shared_memory as u64 {
                return Err(TraceError::SharedMemorySplitMismatch {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                    combined: kernel.size_shared_memory,
                    static_bytes,
                    dynamic_bytes,
                });
            }
        }
    }
    Ok(())
}

/// Dynamic memory route proven from the validated instruction class and operand
/// spaces. Static SASS routing is still checked by `sim-sm` during replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedMemoryRoute {
    Global,
    Local,
    Shared,
    Unsupported,
}

/// Compact, context-free form of dynamic operand-route failures. Replay adds
/// path/record context and preserves its established typed diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedMemoryRouteError {
    GlobalOperandCount {
        got: usize,
    },
    GlobalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    SharedLocalOperandCount {
        got: usize,
    },
    SharedLocalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    AtomicOperandCount {
        got: usize,
    },
    AtomicSpaceMissing {
        operand: usize,
    },
    AtomicSpaceUnspecified {
        operand: usize,
    },
    AtomicSpacesConflict,
    AsyncCopySourceCount {
        got: usize,
    },
    AsyncCopySourceSpaceMissing,
    AsyncCopySourceSpaceInvalid {
        space: pb::MemorySpace,
    },
    AsyncCopyOperandCount {
        got: usize,
        expected: usize,
    },
    AsyncCopyDestinationSpaceMissing,
    AsyncCopyDestinationSpaceInvalid {
        space: pb::MemorySpace,
    },
}

#[derive(Debug, Clone, Copy)]
struct CompactMemoryRoute(u32);

impl CompactMemoryRoute {
    const GLOBAL: u32 = 0;
    const ERROR_BASE: u32 = 4;

    const fn ready(route: ValidatedMemoryRoute) -> Self {
        Self(match route {
            ValidatedMemoryRoute::Global => Self::GLOBAL,
            ValidatedMemoryRoute::Local => 1,
            ValidatedMemoryRoute::Shared => 2,
            ValidatedMemoryRoute::Unsupported => 3,
        })
    }

    fn error(index: usize) -> Self {
        let index = u32::try_from(index)
            .unwrap_or_else(|_| unreachable!("validated route error arena exceeded u32"));
        Self(
            index
                .checked_add(Self::ERROR_BASE)
                .unwrap_or_else(|| unreachable!("validated route error index overflowed")),
        )
    }
}

/// Packed memory-route and protobuf-enum metadata. The wire preflight limits one
/// threadblock to 1,000,000 instructions, so a route-error arena index plus the
/// four ready-route discriminants fits in 20 bits. Packing the two small closed
/// enums into the remaining bits keeps each compact memory record within 24
/// bytes; exact u64 async payloads live in the sparse owning-warp arena.
#[derive(Debug, Clone, Copy)]
struct CompactMemoryMeta(u32);

impl CompactMemoryMeta {
    const ROUTE_BITS: u32 = 20;
    const ROUTE_MASK: u32 = (1 << Self::ROUTE_BITS) - 1;
    const CLASS_SHIFT: u32 = Self::ROUTE_BITS;
    const CLASS_MASK: u32 = 0xf;
    const ACCESS_SHIFT: u32 = Self::CLASS_SHIFT + 4;
    const ACCESS_MASK: u32 = 0x7;
    const EFFECTIVE_SHIFT: u32 = Self::ACCESS_SHIFT + 3;

    fn new(
        op_class: pb::MemoryOpClass,
        access_kind: Option<pb::MemoryOpAccessKind>,
        route: CompactMemoryRoute,
        has_effective_lanes: bool,
    ) -> Self {
        if route.0 > Self::ROUTE_MASK {
            unreachable!("bounded route-error arena exceeded compact encoding");
        }
        let access = access_kind.map_or(0, |kind| kind as u32 + 1);
        debug_assert!((op_class as u32) <= Self::CLASS_MASK);
        debug_assert!(access <= Self::ACCESS_MASK);
        Self(
            route.0
                | ((op_class as u32) << Self::CLASS_SHIFT)
                | (access << Self::ACCESS_SHIFT)
                | (u32::from(has_effective_lanes) << Self::EFFECTIVE_SHIFT),
        )
    }

    const fn route(self) -> CompactMemoryRoute {
        CompactMemoryRoute(self.0 & Self::ROUTE_MASK)
    }

    fn op_class(self) -> pb::MemoryOpClass {
        let encoded = ((self.0 >> Self::CLASS_SHIFT) & Self::CLASS_MASK) as i32;
        pb::MemoryOpClass::try_from(encoded)
            .unwrap_or_else(|_| unreachable!("validated memory class escaped compact encoding"))
    }

    fn access_kind(self) -> Option<pb::MemoryOpAccessKind> {
        let encoded = (self.0 >> Self::ACCESS_SHIFT) & Self::ACCESS_MASK;
        if encoded == 0 {
            return None;
        }
        pb::MemoryOpAccessKind::try_from((encoded - 1) as i32)
            .map(Some)
            .unwrap_or_else(|_| unreachable!("validated access kind escaped compact encoding"))
    }

    const fn has_effective_lanes(self) -> bool {
        self.0 & (1 << Self::EFFECTIVE_SHIFT) != 0
    }
}

/// Stage at which deferred sector expansion would have failed in the replay
/// classifier. Retaining the stage preserves its diagnostic precedence relative
/// to static SASS role checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedSectorErrorPhase {
    BeforeStaticRole,
    AfterStaticRole,
}

/// Failure while reducing active lane widths to exact effective payload bytes.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum EffectiveDataBytesError {
    #[error("effective memory payload byte count overflows u64")]
    Overflow,
}

/// Failure while reducing explicit shared addresses to the bounded projection
/// used by the sm_86 bank model. This is an implementation resource boundary,
/// not a wire-contract reinterpretation.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SharedBankProjectionError {
    #[error(
        "lane {lane} spans {words} 4-byte words, exceeding the per-lane projection limit {limit}"
    )]
    LaneWordLimit { lane: usize, words: u64, limit: u64 },
    #[error(
        "instruction projects {words} 4-byte words, exceeding the per-instruction limit {limit}"
    )]
    InstructionWordLimit { words: u64, limit: u64 },
    #[error("projection scratch could not reserve {additional} additional words")]
    CapacityUnavailable { additional: usize },
    #[error("bank {bank} wavefront count overflowed u64")]
    CountOverflow { bank: u8 },
}

/// Compact per-instruction LDGSTS source/destination topology derived only
/// from the lane-paired trace operands. Timing policy remains outside the
/// trace reader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LdgstsDestinationTopology {
    /// Number of flexible 128-byte source cohorts formed by the active lane
    /// transfers. Zero is the exact result for a predicated-off instruction.
    pub source_cohorts: u32,
    /// Sum of distinct 128-byte destination lines touched independently by
    /// each source cohort.
    pub destination_line_fanout_total: u32,
    /// Largest distinct destination-line fanout of any one source cohort.
    pub destination_line_fanout_max: u32,
    /// Existing cache-all projection: the sum, over source cohorts, of the
    /// maximum distinct 4-byte-word multiplicity on any of 32 shared banks.
    pub cache_all_bank_wavefronts_total: u32,
    /// Largest cache-all bank-word multiplicity of any one source cohort.
    pub cache_all_bank_wavefronts_max: u32,
    /// Trace-derived bypass projection. For each source cohort this adds
    /// `min(active transfers, destination lines * bank wavefronts)` before
    /// summing, so line fanout and bank multiplicity are not collapsed before
    /// their cohort-local interaction is observed.
    pub bypass_wavefronts_total: u32,
    /// Largest cohort-local contribution to `bypass_wavefronts_total`.
    pub bypass_wavefronts_max: u32,
}

/// Exact topology and payload work retained for one flexible 128-byte LDGSTS
/// source cohort. Cohorts are ordered by their smallest uncovered source
/// address, matching [`project_ldgsts_source_cohort_topology_128b_32x4`].
/// Timing policy remains outside the trace reader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LdgstsSourceCohort {
    /// Sum of active source-transfer widths assigned to this cohort.
    pub payload_bytes: u32,
    /// Distinct 128-byte shared destination lines touched by this cohort.
    pub destination_lines: u32,
    /// Maximum distinct 4-byte-word multiplicity on any shared bank.
    pub bank_wavefronts: u32,
    /// Cohort-local bypass projection:
    /// `min(active transfers, destination_lines * bank_wavefronts)`.
    pub bypass_wavefronts: u32,
}

/// Failure while pairing one LDGSTS global-source transfer with its
/// lane-corresponding shared destination and reducing flexible 128-byte source
/// cohorts to destination-line and shared-bank topology.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LdgstsDestinationProjectionError {
    #[error("expected exactly one GLOBAL/GENERIC source operand, got {got}")]
    SourceOperandCount { got: usize },
    #[error("expected exactly one SHARED destination operand, got {got}")]
    DestinationOperandCount { got: usize },
    #[error(
        "source and destination operand widths differ: source={source_bytes} bytes, destination={destination_bytes} bytes"
    )]
    OperandWidthMismatch {
        source_bytes: i32,
        destination_bytes: i32,
    },
    #[error("lane {lane} source transfer width {width_bytes} exceeds one 128-byte cohort")]
    SourceTransferExceedsCohort { lane: usize, width_bytes: u64 },
    #[error(
        "lane {lane} source interval overflows the u64 address domain: base=0x{base:x}, width={width_bytes}"
    )]
    SourceIntervalOverflow {
        lane: usize,
        base: u64,
        width_bytes: u64,
    },
    #[error(
        "lane {lane} destination interval overflows the i64 address domain: base={base}, width={width_bytes}"
    )]
    DestinationIntervalOverflow {
        lane: usize,
        base: i64,
        width_bytes: u64,
    },
    #[error(
        "instruction projects {words} destination words, exceeding the per-instruction limit {limit}"
    )]
    InstructionWordLimit { words: u64, limit: u64 },
    #[error("destination projection scratch could not reserve {additional} additional words")]
    CapacityUnavailable { additional: usize },
    #[error("destination bank {bank} wavefront count overflowed u32")]
    BankWavefrontCountOverflow { bank: u8 },
    #[error("instruction destination wavefront sum overflowed u32")]
    WavefrontCountOverflow,
    #[error("instruction source-cohort count overflowed u32")]
    SourceCohortCountOverflow,
    #[error("instruction destination-line fanout sum overflowed u32")]
    DestinationLineFanoutOverflow,
    #[error(
        "cohort bypass projection overflowed u32: destination-line fanout {destination_lines} * bank wavefronts {bank_wavefronts}"
    )]
    BypassWavefrontProductOverflow {
        destination_lines: u32,
        bank_wavefronts: u32,
    },
    #[error("instruction bypass destination-wavefront sum overflowed u32")]
    BypassWavefrontCountOverflow,
}

/// Compact, context-free form of a sector-expansion failure. Replay restores the
/// path/warp/instruction context when this deferred error is observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedSectorError {
    DataWidthInvalid {
        data_width_bytes: i32,
    },
    AddressOverflow {
        lane: usize,
        base: i64,
        data_width_bytes: i32,
    },
    SectorCountOverflow {
        sectors: usize,
    },
    UniqueSectorCountOutOfRange {
        sectors: u128,
    },
    CardinalityMismatch {
        expected: usize,
        produced: usize,
    },
    CapacityUnavailable {
        buffer: &'static str,
        additional: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct CompactSectorPlan {
    offset_or_error: usize,
    len: u16,
    error_phase: u8,
}

impl CompactSectorPlan {
    const READY: u8 = 0;
    const BEFORE_STATIC_ROLE: u8 = 1;
    const AFTER_STATIC_ROLE: u8 = 2;

    const fn ready(start: usize, len: u16) -> Self {
        Self {
            offset_or_error: start,
            len,
            error_phase: Self::READY,
        }
    }

    const fn error(phase: ValidatedSectorErrorPhase, index: usize) -> Self {
        Self {
            offset_or_error: index,
            len: 0,
            error_phase: match phase {
                ValidatedSectorErrorPhase::BeforeStaticRole => Self::BEFORE_STATIC_ROLE,
                ValidatedSectorErrorPhase::AfterStaticRole => Self::AFTER_STATIC_ROLE,
            },
        }
    }
}

/// Borrowed sector payload or deferred expansion failure for one validated
/// memory instruction.
#[derive(Debug, Clone, Copy)]
pub enum ValidatedSectorPlan<'a> {
    Sectors(&'a [i64]),
    Error {
        phase: ValidatedSectorErrorPhase,
        error: &'a ValidatedSectorError,
    },
}

/// Compact memory metadata retained after the full Prost tree has been
/// validated and discarded. `projection` is route-disjoint: it stores a shared
/// bank-wavefront count for Shared records, an index into the owning warp's
/// sparse async-copy projection arena for Global async-copy records, or the
/// already-validated allocation id for address-carrying Global atomics.
pub struct ValidatedMemoryInstruction {
    sectors: CompactSectorPlan,
    projection: u32,
    meta: CompactMemoryMeta,
}

#[derive(Debug, Clone, Copy)]
struct AsyncCopyProjection {
    effective_data_bytes: u64,
    destination_topology: CompactLdgstsDestinationTopology,
    cohort_span: CompactArenaSpan,
    source_sector_mask_span: CompactArenaSpan,
}

#[derive(Debug, Clone, Copy, Default)]
struct CompactArenaSpan {
    start: u32,
    len: u16,
}

impl CompactArenaSpan {
    fn new(start: usize, len: usize) -> Self {
        Self {
            start: u32::try_from(start)
                .unwrap_or_else(|_| unreachable!("validated sparse arena exceeded u32")),
            len: u16::try_from(len)
                .unwrap_or_else(|_| unreachable!("validated sparse span exceeded u16")),
        }
    }

    fn slice<'a, T>(self, arena: &'a [T]) -> &'a [T] {
        let start = usize::try_from(self.start)
            .unwrap_or_else(|_| unreachable!("validated sparse span escaped usize"));
        let end = start
            .checked_add(usize::from(self.len))
            .unwrap_or_else(|| unreachable!("validated sparse span overflowed"));
        arena
            .get(start..end)
            .unwrap_or_else(|| unreachable!("validated sparse span escaped its arena"))
    }
}

#[derive(Debug, Clone, Copy)]
struct CompactLdgstsDestinationTopology {
    source_cohorts: u8,
    destination_line_fanout_total: u8,
    destination_line_fanout_max: u8,
    cache_all_bank_wavefronts_total: u8,
    cache_all_bank_wavefronts_max: u8,
    bypass_wavefronts_total: u8,
    bypass_wavefronts_max: u8,
}

impl CompactLdgstsDestinationTopology {
    const UNAVAILABLE: u8 = u8::MAX;

    fn new(topology: Option<LdgstsDestinationTopology>) -> Self {
        let Some(topology) = topology else {
            return Self {
                source_cohorts: Self::UNAVAILABLE,
                destination_line_fanout_total: 0,
                destination_line_fanout_max: 0,
                cache_all_bank_wavefronts_total: 0,
                cache_all_bank_wavefronts_max: 0,
                bypass_wavefronts_total: 0,
                bypass_wavefronts_max: 0,
            };
        };
        let compact = |value: u32| {
            u8::try_from(value).expect("one 32-lane LDGSTS topology fits the compact u8 domain")
        };
        Self {
            source_cohorts: compact(topology.source_cohorts),
            destination_line_fanout_total: compact(topology.destination_line_fanout_total),
            destination_line_fanout_max: compact(topology.destination_line_fanout_max),
            cache_all_bank_wavefronts_total: compact(topology.cache_all_bank_wavefronts_total),
            cache_all_bank_wavefronts_max: compact(topology.cache_all_bank_wavefronts_max),
            bypass_wavefronts_total: compact(topology.bypass_wavefronts_total),
            bypass_wavefronts_max: compact(topology.bypass_wavefronts_max),
        }
    }

    const fn expand(self) -> Option<LdgstsDestinationTopology> {
        if self.source_cohorts == Self::UNAVAILABLE {
            return None;
        }
        Some(LdgstsDestinationTopology {
            source_cohorts: self.source_cohorts as u32,
            destination_line_fanout_total: self.destination_line_fanout_total as u32,
            destination_line_fanout_max: self.destination_line_fanout_max as u32,
            cache_all_bank_wavefronts_total: self.cache_all_bank_wavefronts_total as u32,
            cache_all_bank_wavefronts_max: self.cache_all_bank_wavefronts_max as u32,
            bypass_wavefronts_total: self.bypass_wavefronts_total as u32,
            bypass_wavefronts_max: self.bypass_wavefronts_max as u32,
        })
    }
}

impl AsyncCopyProjection {
    fn new(
        effective_data_bytes: u64,
        destination_topology: Option<LdgstsDestinationTopology>,
        cohort_span: CompactArenaSpan,
        source_sector_mask_span: CompactArenaSpan,
    ) -> Self {
        Self {
            effective_data_bytes,
            destination_topology: CompactLdgstsDestinationTopology::new(destination_topology),
            cohort_span,
            source_sector_mask_span,
        }
    }

    const fn destination_topology(self) -> Option<LdgstsDestinationTopology> {
        self.destination_topology.expand()
    }
}

impl ValidatedMemoryInstruction {
    /// Reconstruct only the metadata replay still consumes. Cache-op and mask
    /// fields were validated before compaction but need not remain resident.
    pub fn replay_meta(&self) -> MemoryAccessMeta {
        MemoryAccessMeta {
            op_class: self.meta.op_class(),
            access_kind: self.meta.access_kind(),
            cache_op: None,
            and_mask: u32::from(self.meta.has_effective_lanes()),
        }
    }

    /// Whether at least one lane executes this dynamic memory instruction.
    pub const fn has_effective_lanes(&self) -> bool {
        self.meta.has_effective_lanes()
    }

    /// Maximum distinct 4-byte words targeting one of 32 banks after collapsing
    /// broadcasts. Zero means the record had no effective shared address.
    /// Architecture timing must explicitly admit this named projection before
    /// interpreting it as hardware wavefronts. Callers resolve the record's
    /// Shared route before reading this route-disjoint projection.
    pub const fn shared_bank_wavefronts_32x4(&self) -> u32 {
        self.projection
    }

    /// Backing allocation identity for an address-carrying global atomic.
    ///
    /// The full reader has already required this id to resolve to the
    /// allocation registry and contain every effective lane address. Zero is
    /// retained only for predicated-off or legacy/non-atomic records and is
    /// exposed as `None`. This is timing-neutral identity: it does not assign a
    /// physical LTS class.
    pub fn global_atomic_allocation_id(&self) -> Option<u32> {
        (self.meta.op_class() == pb::MemoryOpClass::Atomic
            && self.meta.route().0 == CompactMemoryRoute::GLOBAL
            && self.projection != 0)
            .then_some(self.projection)
    }
}

/// Compact immutable threadblock produced by the full payload validation walk.
pub struct ValidatedThreadblock {
    launch_index: i64,
    warps: Vec<ValidatedWarp>,
}

impl ValidatedThreadblock {
    fn from_message(
        launch_index: i64,
        threadblock: pb::Threadblock,
        payload_contract: MemoryPayloadContract,
        label: &str,
    ) -> Result<Self, TraceError> {
        let warps = threadblock
            .warps
            .into_iter()
            .map(|(map_key, warp)| {
                ValidatedWarp::from_message(map_key, warp, payload_contract, label)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            launch_index,
            warps,
        })
    }

    pub const fn launch_index(&self) -> i64 {
        self.launch_index
    }

    pub fn warps(&self) -> &[ValidatedWarp] {
        &self.warps
    }
}

/// One validated warp in protobuf map-key order.
pub struct ValidatedWarp {
    map_key: i32,
    warp_id: i32,
    function_unique_id: i32,
    instructions: Vec<ValidatedInstruction>,
    memories: Vec<ValidatedMemoryInstruction>,
    async_copy_projections: Vec<AsyncCopyProjection>,
    ldgsts_source_cohorts: Vec<LdgstsSourceCohort>,
    ldgsts_source_sector_cohort_masks: Vec<u32>,
    route_errors: Vec<ValidatedMemoryRouteError>,
    sectors: Vec<i64>,
    sector_errors: Vec<ValidatedSectorError>,
}

impl ValidatedWarp {
    fn from_message(
        map_key: i32,
        warp: pb::Warp,
        payload_contract: MemoryPayloadContract,
        label: &str,
    ) -> Result<Self, TraceError> {
        let function_unique_id = warp
            .instructions
            .first()
            .map_or(0, |instruction| instruction.function_unique_id);
        let mut instructions = Vec::with_capacity(warp.instructions.len());
        let mut memories = Vec::new();
        let mut async_copy_projections = Vec::new();
        let mut ldgsts_source_cohorts = Vec::new();
        let mut ldgsts_source_sector_cohort_masks = Vec::new();
        let mut route_errors = Vec::new();
        let mut sectors = Vec::new();
        let mut sector_errors = Vec::new();
        let mut shared_word_scratch = Vec::new();
        for (record_ix, instruction) in warp.instructions.into_iter().enumerate() {
            let memory_index = if instruction.memory_op_class.is_some() {
                let no_error_context = || String::new();
                let preflight = match preflight_memory_access(&instruction, no_error_context) {
                    Ok(Some(preflight)) => preflight,
                    Ok(None) | Err(_) => {
                        unreachable!("validated memory instruction failed compact preflight")
                    }
                };
                let meta = preflight.meta();
                let resolved_route = compact_memory_route(
                    &instruction,
                    meta.op_class,
                    payload_contract,
                    meta.and_mask,
                );
                let shared_bank_wavefronts_32x4 =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Shared)) {
                        project_shared_bank_wavefronts_32x4(preflight, &mut shared_word_scratch)
                            .map_err(|source| TraceError::SharedBankProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            })?
                    } else {
                        0
                    };
                let is_async_copy_source = meta.op_class == pb::MemoryOpClass::AsyncCopy
                    && matches!(resolved_route, Ok(ValidatedMemoryRoute::Global));
                let effective_data_bytes = if is_async_copy_source {
                    if payload_contract.carries_ldgsts_destination() {
                        project_operand_effective_data_bytes(preflight, 0)
                    } else {
                        project_effective_data_bytes(preflight)
                    }
                    .map_err(|source| TraceError::EffectiveDataBytes {
                        record: format!("{label} warp={map_key} inst={record_ix}"),
                        source,
                    })?
                } else {
                    0
                };
                let global_atomic_transactions = meta.op_class == pb::MemoryOpClass::Atomic
                    && matches!(resolved_route, Ok(ValidatedMemoryRoute::Global))
                    && payload_contract.carries_atomic_addresses();
                let sector_projection = if matches!(
                    resolved_route,
                    Ok(ValidatedMemoryRoute::Global | ValidatedMemoryRoute::Local)
                ) {
                    Some(if global_atomic_transactions {
                        compact_atomic_sector_transaction_vector_32b(preflight)
                    } else {
                        compact_sector_vector_for_operand(
                            preflight,
                            (is_async_copy_source && payload_contract.carries_ldgsts_destination())
                                .then_some(0),
                        )
                    })
                } else {
                    None
                };
                let destination_projection =
                    if is_async_copy_source && payload_contract.carries_ldgsts_destination() {
                        let source_sectors = sector_projection
                            .as_ref()
                            .and_then(|projection| projection.as_ref().ok())
                            .map_or(&[][..], Vec::as_slice);
                        Some(
                            project_ldgsts_source_cohorts_128b_32x4(
                                preflight,
                                source_sectors,
                                &mut shared_word_scratch,
                            )
                            .map_err(|source| {
                                TraceError::LdgstsDestinationProjection {
                                    record: format!("{label} warp={map_key} inst={record_ix}"),
                                    source,
                                }
                            })?,
                        )
                    } else {
                        None
                    };
                let projection = if global_atomic_transactions {
                    preflight
                        .instruction()
                        .addresses
                        .first()
                        .and_then(|address| address.allocation_id)
                        .filter(|allocation_id| *allocation_id != 0)
                        .unwrap_or(0)
                } else if is_async_copy_source {
                    let index = match u32::try_from(async_copy_projections.len()) {
                        Ok(index) if index != u32::MAX => index,
                        Ok(_) | Err(_) => {
                            unreachable!("validated async-copy projection arena exceeded u32")
                        }
                    };
                    let cohort_span = CompactArenaSpan::new(
                        ldgsts_source_cohorts.len(),
                        destination_projection
                            .as_ref()
                            .map_or(0, |projection| projection.cohorts.len()),
                    );
                    let source_sector_mask_span = CompactArenaSpan::new(
                        ldgsts_source_sector_cohort_masks.len(),
                        destination_projection
                            .as_ref()
                            .map_or(0, |projection| projection.source_sector_cohort_masks.len()),
                    );
                    let destination_topology = destination_projection
                        .as_ref()
                        .map(|projection| projection.topology);
                    if let Some(destination_projection) = destination_projection {
                        ldgsts_source_cohorts.extend(destination_projection.cohorts);
                        ldgsts_source_sector_cohort_masks
                            .extend(destination_projection.source_sector_cohort_masks);
                    }
                    async_copy_projections.push(AsyncCopyProjection::new(
                        effective_data_bytes,
                        destination_topology,
                        cohort_span,
                        source_sector_mask_span,
                    ));
                    index
                } else {
                    shared_bank_wavefronts_32x4
                };
                let sector_plan = if let Some(sector_projection) = sector_projection {
                    match sector_projection {
                        Ok(staged) => {
                            let start = sectors.len();
                            let len = match u16::try_from(staged.len()) {
                                Ok(len) => len,
                                Err(_) => unreachable!("bounded sector vector exceeded u16"),
                            };
                            match sectors.try_reserve(staged.len()) {
                                Ok(()) => {
                                    sectors.extend_from_slice(&staged);
                                    CompactSectorPlan::ready(start, len)
                                }
                                Err(_) => {
                                    let index = sector_errors.len();
                                    sector_errors.push(ValidatedSectorError::CapacityUnavailable {
                                        buffer: "warp sector arena",
                                        additional: staged.len(),
                                    });
                                    CompactSectorPlan::error(
                                        ValidatedSectorErrorPhase::AfterStaticRole,
                                        index,
                                    )
                                }
                            }
                        }
                        Err((phase, error)) => {
                            let index = sector_errors.len();
                            sector_errors.push(error);
                            CompactSectorPlan::error(phase, index)
                        }
                    }
                } else {
                    CompactSectorPlan::ready(sectors.len(), 0)
                };
                let route = match resolved_route {
                    Ok(route) => CompactMemoryRoute::ready(route),
                    Err(error) => {
                        let route = CompactMemoryRoute::error(route_errors.len());
                        route_errors.push(error);
                        route
                    }
                };
                let memory_index = match u32::try_from(memories.len()) {
                    Ok(index) if index != u32::MAX => index,
                    Ok(_) | Err(_) => unreachable!("validated memory arena exceeded u32"),
                };
                memories.push(ValidatedMemoryInstruction {
                    sectors: sector_plan,
                    projection,
                    meta: CompactMemoryMeta::new(
                        meta.op_class,
                        meta.access_kind,
                        route,
                        meta.and_mask != 0,
                    ),
                });
                memory_index
            } else {
                u32::MAX
            };
            let mut identity_presence = 0_u8;
            if instruction.sm_id.is_some() {
                identity_presence |= ValidatedInstruction::SM_ID_PRESENT;
            }
            if instruction.warpid_sm.is_some() {
                identity_presence |= ValidatedInstruction::WARP_SLOT_PRESENT;
            }
            instructions.push(ValidatedInstruction {
                pc: instruction.pc,
                memory_index,
                sm_id: instruction.sm_id.unwrap_or_default(),
                warpid_sm: instruction.warpid_sm.unwrap_or_default(),
                identity_presence,
            });
        }
        Ok(Self {
            map_key,
            warp_id: warp.id,
            function_unique_id,
            instructions,
            memories,
            async_copy_projections,
            ldgsts_source_cohorts,
            ldgsts_source_sector_cohort_masks,
            route_errors,
            sectors,
            sector_errors,
        })
    }

    pub const fn map_key(&self) -> i32 {
        self.map_key
    }

    pub const fn warp_id(&self) -> i32 {
        self.warp_id
    }

    pub fn instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = ValidatedInstructionRef<'_>> + DoubleEndedIterator + '_ {
        self.instructions.iter().map(|instruction| {
            let memory =
                if instruction.memory_index == u32::MAX {
                    None
                } else {
                    let index = usize::try_from(instruction.memory_index)
                        .unwrap_or_else(|_| unreachable!("validated memory index escaped usize"));
                    Some(self.memories.get(index).unwrap_or_else(|| {
                        unreachable!("validated memory index escaped its arena")
                    }))
                };
            ValidatedInstructionRef {
                instruction,
                memory,
                function_unique_id: self.function_unique_id,
                warpid_tb: self.map_key,
            }
        })
    }

    pub fn memory_route<'a>(
        &'a self,
        memory: &ValidatedMemoryInstruction,
    ) -> Result<ValidatedMemoryRoute, &'a ValidatedMemoryRouteError> {
        match memory.meta.route().0 {
            0 => Ok(ValidatedMemoryRoute::Global),
            1 => Ok(ValidatedMemoryRoute::Local),
            2 => Ok(ValidatedMemoryRoute::Shared),
            3 => Ok(ValidatedMemoryRoute::Unsupported),
            encoded => {
                let index = encoded
                    .checked_sub(CompactMemoryRoute::ERROR_BASE)
                    .unwrap_or_else(|| unreachable!("validated route encoding underflowed"));
                let index = usize::try_from(index)
                    .unwrap_or_else(|_| unreachable!("validated route error escaped usize"));
                Err(self
                    .route_errors
                    .get(index)
                    .unwrap_or_else(|| unreachable!("validated route error escaped its arena")))
            }
        }
    }

    fn async_copy_projection(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<AsyncCopyProjection> {
        if memory.meta.op_class() != pb::MemoryOpClass::AsyncCopy
            || memory.meta.route().0 != CompactMemoryRoute::GLOBAL
        {
            return None;
        }
        let index = usize::try_from(memory.projection)
            .unwrap_or_else(|_| unreachable!("validated async-copy index escaped usize"));
        Some(*self.async_copy_projections.get(index).unwrap_or_else(|| {
            unreachable!("validated async-copy index escaped its projection arena")
        }))
    }

    /// Sum of active-lane source widths for one Global async-copy record. Exact
    /// u64 values live in a sparse warp-local arena so ordinary memory metadata
    /// remains compact. Other routes and operation classes return zero.
    pub fn effective_data_bytes(&self, memory: &ValidatedMemoryInstruction) -> u64 {
        self.async_copy_projection(memory)
            .map_or(0, |projection| projection.effective_data_bytes)
    }

    /// Exact LDGSTS source/destination topology when the trace contract carries
    /// the lane-paired SHARED operand. `None` preserves the explicit source-only
    /// fallback boundary; `Some(default())` is an exact zero-active result.
    pub fn ldgsts_destination_topology_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<LdgstsDestinationTopology> {
        self.async_copy_projection(memory)
            .and_then(AsyncCopyProjection::destination_topology)
    }

    /// Exact flexible 128-byte source-cohort descriptors for a destination-
    /// bearing LDGSTS record. `None` is the legacy source-only fallback;
    /// predicated-off destination-bearing records return `Some(&[])`.
    pub fn ldgsts_source_cohorts_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<&[LdgstsSourceCohort]> {
        let projection = self.async_copy_projection(memory)?;
        projection.destination_topology()?;
        Some(projection.cohort_span.slice(&self.ldgsts_source_cohorts))
    }

    /// One cohort-membership bitmask per compact source sector, in the exact
    /// order returned by [`Self::sector_plan`]. A bit is set when that sector
    /// intersects source bytes assigned to the corresponding cohort; boundary
    /// sectors may therefore carry multiple bits. `None` preserves the legacy
    /// source-only fallback.
    pub fn ldgsts_source_sector_cohort_masks_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<&[u32]> {
        let projection = self.async_copy_projection(memory)?;
        projection.destination_topology()?;
        Some(
            projection
                .source_sector_mask_span
                .slice(&self.ldgsts_source_sector_cohort_masks),
        )
    }

    /// Compatibility view of the established cache-all bank-wavefront total.
    /// New bypass work must consume the multidimensional topology above rather
    /// than treating this scalar as a transferable completion coordinate.
    pub fn ldgsts_destination_wavefronts_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<u32> {
        self.ldgsts_destination_topology_128b_32x4(memory)
            .map(|topology| topology.cache_all_bank_wavefronts_total)
    }

    pub fn sector_plan<'a>(
        &'a self,
        memory: &ValidatedMemoryInstruction,
    ) -> ValidatedSectorPlan<'a> {
        match memory.sectors.error_phase {
            CompactSectorPlan::READY => {
                let start = memory.sectors.offset_or_error;
                let end = match start.checked_add(usize::from(memory.sectors.len)) {
                    Some(end) => end,
                    None => unreachable!("validated sector span overflowed"),
                };
                let Some(sectors) = self.sectors.get(start..end) else {
                    unreachable!("validated sector span escaped its arena")
                };
                ValidatedSectorPlan::Sectors(sectors)
            }
            code @ (CompactSectorPlan::BEFORE_STATIC_ROLE
            | CompactSectorPlan::AFTER_STATIC_ROLE) => {
                let index = memory.sectors.offset_or_error;
                let Some(error) = self.sector_errors.get(index) else {
                    unreachable!("validated sector error index escaped its arena")
                };
                let phase = if code == CompactSectorPlan::BEFORE_STATIC_ROLE {
                    ValidatedSectorErrorPhase::BeforeStaticRole
                } else {
                    ValidatedSectorErrorPhase::AfterStaticRole
                };
                ValidatedSectorPlan::Error { phase, error }
            }
            _ => unreachable!("validated sector plan carried an unknown phase"),
        }
    }
}

/// Packed scheduler fields retained from one validated dynamic instruction.
struct ValidatedInstruction {
    pc: u32,
    memory_index: u32,
    sm_id: i32,
    warpid_sm: i32,
    identity_presence: u8,
}

impl ValidatedInstruction {
    const SM_ID_PRESENT: u8 = 1 << 0;
    const WARP_SLOT_PRESENT: u8 = 1 << 1;
}

/// One instruction borrowed together with the compact memory metadata from its
/// owning warp. The pairing prevents memory-arena indices from crossing warps.
#[derive(Clone, Copy)]
pub struct ValidatedInstructionRef<'a> {
    instruction: &'a ValidatedInstruction,
    memory: Option<&'a ValidatedMemoryInstruction>,
    function_unique_id: i32,
    warpid_tb: i32,
}

impl<'a> ValidatedInstructionRef<'a> {
    pub const fn pc(&self) -> u32 {
        self.instruction.pc
    }

    pub const fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    pub const fn warpid_tb(&self) -> i32 {
        self.warpid_tb
    }

    pub const fn warpid_sm(&self) -> Option<i32> {
        if self.instruction.identity_presence & ValidatedInstruction::WARP_SLOT_PRESENT != 0 {
            Some(self.instruction.warpid_sm)
        } else {
            None
        }
    }

    pub const fn sm_id(&self) -> Option<i32> {
        if self.instruction.identity_presence & ValidatedInstruction::SM_ID_PRESENT != 0 {
            Some(self.instruction.sm_id)
        } else {
            None
        }
    }

    pub const fn memory(self) -> Option<&'a ValidatedMemoryInstruction> {
        self.memory
    }
}

fn compact_memory_route(
    instruction: &pb::Instruction,
    class: pb::MemoryOpClass,
    payload_contract: MemoryPayloadContract,
    and_mask: u32,
) -> Result<ValidatedMemoryRoute, ValidatedMemoryRouteError> {
    let space = |address: &pb::Address| {
        address
            .address_space
            .map(|raw| match pb::MemorySpace::try_from(raw) {
                Ok(space) => space,
                Err(_) => unreachable!("validated address carried an unknown space"),
            })
    };
    let zero_effective_mask = and_mask == 0;
    match class {
        pb::MemoryOpClass::Unspecified => Ok(ValidatedMemoryRoute::Unsupported),
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if instruction.addresses.is_empty() && !zero_effective_mask {
                return Err(ValidatedMemoryRouteError::SharedLocalOperandCount { got: 0 });
            }
            let expected = if class == pb::MemoryOpClass::Shared {
                pb::MemorySpace::Shared
            } else {
                pb::MemorySpace::Local
            };
            for (operand, address) in instruction.addresses.iter().enumerate() {
                let Some(space) = space(address) else {
                    continue;
                };
                if space != expected {
                    return Err(ValidatedMemoryRouteError::SharedLocalSpaceInvalid {
                        operand,
                        space,
                    });
                }
            }
            Ok(if class == pb::MemoryOpClass::Shared {
                ValidatedMemoryRoute::Shared
            } else {
                ValidatedMemoryRoute::Local
            })
        }
        pb::MemoryOpClass::Global => {
            if instruction.addresses.is_empty() && !zero_effective_mask {
                return Err(ValidatedMemoryRouteError::GlobalOperandCount { got: 0 });
            }
            for (operand, address) in instruction.addresses.iter().enumerate() {
                let Some(space) = space(address) else {
                    continue;
                };
                if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                    return Err(ValidatedMemoryRouteError::GlobalSpaceInvalid { operand, space });
                }
            }
            Ok(ValidatedMemoryRoute::Global)
        }
        pb::MemoryOpClass::Atomic => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                let mut resolved = None;
                for (operand, address) in instruction.addresses.iter().enumerate() {
                    let Some(space) = space(address) else {
                        continue;
                    };
                    let candidate = match space {
                        pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                            ValidatedMemoryRoute::Global
                        }
                        pb::MemorySpace::Shared => ValidatedMemoryRoute::Shared,
                        pb::MemorySpace::Local => ValidatedMemoryRoute::Local,
                        pb::MemorySpace::Unspecified => {
                            return Err(ValidatedMemoryRouteError::AtomicSpaceUnspecified {
                                operand,
                            });
                        }
                    };
                    match resolved {
                        Some(previous) if previous != candidate => {
                            return Err(ValidatedMemoryRouteError::AtomicSpacesConflict);
                        }
                        None => resolved = Some(candidate),
                        Some(_) => {}
                    }
                }
                return Ok(resolved.unwrap_or(ValidatedMemoryRoute::Global));
            }
            if instruction.addresses.is_empty() && zero_effective_mask {
                return Ok(ValidatedMemoryRoute::Global);
            }
            if instruction.addresses.len() != 1 {
                return Err(ValidatedMemoryRouteError::AtomicOperandCount {
                    got: instruction.addresses.len(),
                });
            }
            let Some(space) = space(&instruction.addresses[0]) else {
                return Err(ValidatedMemoryRouteError::AtomicSpaceMissing { operand: 0 });
            };
            match space {
                pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                    Ok(ValidatedMemoryRoute::Global)
                }
                pb::MemorySpace::Shared => Ok(ValidatedMemoryRoute::Shared),
                pb::MemorySpace::Local => Ok(ValidatedMemoryRoute::Local),
                pb::MemorySpace::Unspecified => {
                    Err(ValidatedMemoryRouteError::AtomicSpaceUnspecified { operand: 0 })
                }
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                for address in &instruction.addresses {
                    let Some(space) = space(address) else {
                        continue;
                    };
                    if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                        return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space,
                        });
                    }
                }
                return Ok(ValidatedMemoryRoute::Global);
            }
            if payload_contract.carries_ldgsts_destination() {
                if instruction.addresses.len() != 2 {
                    return Err(ValidatedMemoryRouteError::AsyncCopyOperandCount {
                        got: instruction.addresses.len(),
                        expected: 2,
                    });
                }
                let Some(source_space) = space(&instruction.addresses[0]) else {
                    return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing);
                };
                if !matches!(
                    source_space,
                    pb::MemorySpace::Global | pb::MemorySpace::Generic
                ) {
                    return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid {
                        space: source_space,
                    });
                }
                let Some(destination_space) = space(&instruction.addresses[1]) else {
                    return Err(ValidatedMemoryRouteError::AsyncCopyDestinationSpaceMissing);
                };
                if destination_space != pb::MemorySpace::Shared {
                    return Err(
                        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceInvalid {
                            space: destination_space,
                        },
                    );
                }
                return Ok(ValidatedMemoryRoute::Global);
            }
            if instruction.addresses.len() != 1 {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceCount {
                    got: instruction.addresses.len(),
                });
            }
            let Some(space) = space(&instruction.addresses[0]) else {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing);
            };
            if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid { space });
            }
            Ok(ValidatedMemoryRoute::Global)
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => Ok(ValidatedMemoryRoute::Unsupported),
    }
}

fn compact_sector_vector_for_operand(
    preflight: PreflightedMemoryAccess<'_>,
    selected_operand: Option<usize>,
) -> Result<Vec<i64>, (ValidatedSectorErrorPhase, ValidatedSectorError)> {
    let mut intervals = Vec::new();
    let mut early_error = None;
    preflight.visit_addresses_enumerated(|operand, access| {
        if early_error.is_some() || selected_operand.is_some_and(|selected| selected != operand) {
            return;
        }
        if access.data_width_bytes <= 0 {
            early_error = Some(ValidatedSectorError::DataWidthInvalid {
                data_width_bytes: access.data_width_bytes,
            });
            return;
        }
        let width_offset = i64::from(access.data_width_bytes) - 1;
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let Some(last_address) = base.checked_add(width_offset) else {
                early_error = Some(ValidatedSectorError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes: access.data_width_bytes,
                });
                return;
            };
            if intervals.try_reserve(1).is_err() {
                early_error = Some(ValidatedSectorError::CapacityUnavailable {
                    buffer: "sector interval plan",
                    additional: 1,
                });
                return;
            }
            intervals.push((base >> 5, last_address >> 5));
        }
    });
    if let Some(error) = early_error {
        return Err((ValidatedSectorErrorPhase::BeforeStaticRole, error));
    }

    let mut sorted = Vec::new();
    sorted.try_reserve(intervals.len()).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sorted sector intervals",
                additional: intervals.len(),
            },
        )
    })?;
    sorted.extend_from_slice(&intervals);
    sorted.sort_unstable();
    let mut union_count = 0_u128;
    if let Some(&(initial_first, initial_last)) = sorted.first() {
        let mut first = initial_first;
        let mut last = initial_last;
        for &(next_first, next_last) in &sorted[1..] {
            if next_first <= last.saturating_add(1) {
                last = last.max(next_last);
            } else {
                union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
                first = next_first;
                last = next_last;
            }
        }
        union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
    }
    let unique_count = usize::try_from(union_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::UniqueSectorCountOutOfRange {
                sectors: union_count,
            },
        )
    })?;
    if unique_count > usize::from(u16::MAX) {
        return Err((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::SectorCountOverflow {
                sectors: unique_count,
            },
        ));
    }

    let mut staged = Vec::new();
    staged.try_reserve(unique_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sector staging",
                additional: unique_count,
            },
        )
    })?;
    let covered_capacity = intervals.len().min(unique_count);
    let mut covered: Vec<(i64, i64)> = Vec::new();
    covered.try_reserve(covered_capacity).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "covered sector intervals",
                additional: covered_capacity,
            },
        )
    })?;
    for (first, last) in intervals {
        let start = covered.partition_point(|&(_, covered_last)| covered_last < first);
        let mut cursor = Some(first);
        for &(covered_first, covered_last) in &covered[start..] {
            let Some(next) = cursor else { break };
            if covered_first > last {
                break;
            }
            if next < covered_first {
                staged.extend(next..=last.min(covered_first.saturating_sub(1)));
            }
            cursor = if covered_last >= last {
                None
            } else {
                covered_last
                    .checked_add(1)
                    .map(|after_covered| next.max(after_covered))
            };
        }
        if let Some(next) = cursor {
            staged.extend(next..=last);
        }

        let left =
            covered.partition_point(|&(_, covered_last)| covered_last.saturating_add(1) < first);
        let right = left
            + covered[left..]
                .partition_point(|&(covered_first, _)| covered_first <= last.saturating_add(1));
        if left == right {
            covered.insert(left, (first, last));
        } else {
            let merged = (first.min(covered[left].0), last.max(covered[right - 1].1));
            covered[left] = merged;
            covered.drain(left + 1..right);
        }
    }
    if staged.len() != unique_count {
        return Err((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CardinalityMismatch {
                expected: unique_count,
                produced: staged.len(),
            },
        ));
    }
    Ok(staged)
}

/// Project one address-carrying global atomic warp instruction onto the
/// 32-byte sector transactions observed at L1TEX/LTS on sm_86.
///
/// Distinct atomic words in one sector share a transaction, while lanes that
/// alias the same atomic word serialize. Consequently each touched sector is
/// repeated by the largest exact-address multiplicity among lanes touching
/// that sector. This retains ordinary coalescing (`lane_word`: four sectors),
/// distinct-line traffic (`lane_line`: 32 sectors), and same-word
/// serialization (`warp_word`: one sector repeated 32 times) without retaining
/// the full lane vector after trace validation.
fn compact_atomic_sector_transaction_vector_32b(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<Vec<i64>, (ValidatedSectorErrorPhase, ValidatedSectorError)> {
    let unique = compact_sector_vector_for_operand(preflight, None)?;
    if unique.is_empty() {
        return Ok(unique);
    }

    let mut intervals = Vec::new();
    let mut early_error = None;
    preflight.visit_addresses(|access| {
        if early_error.is_some() {
            return;
        }
        if access.data_width_bytes <= 0 {
            early_error = Some(ValidatedSectorError::DataWidthInvalid {
                data_width_bytes: access.data_width_bytes,
            });
            return;
        }
        let width_offset = i64::from(access.data_width_bytes) - 1;
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let Some(last_address) = base.checked_add(width_offset) else {
                early_error = Some(ValidatedSectorError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes: access.data_width_bytes,
                });
                return;
            };
            if intervals.try_reserve(1).is_err() {
                early_error = Some(ValidatedSectorError::CapacityUnavailable {
                    buffer: "atomic address intervals",
                    additional: 1,
                });
                return;
            }
            intervals.push((base, base >> 5, last_address >> 5));
        }
    });
    if let Some(error) = early_error {
        return Err((ValidatedSectorErrorPhase::BeforeStaticRole, error));
    }
    intervals.sort_unstable();

    let mut multiplicities = Vec::new();
    multiplicities.try_reserve(unique.len()).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "atomic sector multiplicities",
                additional: unique.len(),
            },
        )
    })?;
    let mut transaction_count = 0_usize;
    for &sector in &unique {
        let mut previous_base = None;
        let mut aliases = 0_usize;
        let mut maximum_aliases = 0_usize;
        for &(base, first_sector, last_sector) in &intervals {
            if sector < first_sector || sector > last_sector {
                continue;
            }
            if previous_base == Some(base) {
                aliases += 1;
            } else {
                previous_base = Some(base);
                aliases = 1;
            }
            maximum_aliases = maximum_aliases.max(aliases);
        }
        debug_assert!(
            maximum_aliases > 0,
            "unique atomic sector has no lane interval"
        );
        transaction_count = transaction_count.checked_add(maximum_aliases).ok_or((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::SectorCountOverflow {
                sectors: usize::MAX,
            },
        ))?;
        if transaction_count > usize::from(u16::MAX) {
            return Err((
                ValidatedSectorErrorPhase::AfterStaticRole,
                ValidatedSectorError::SectorCountOverflow {
                    sectors: transaction_count,
                },
            ));
        }
        multiplicities.push(maximum_aliases);
    }

    let mut staged = Vec::new();
    staged.try_reserve(transaction_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "atomic sector transaction staging",
                additional: transaction_count,
            },
        )
    })?;
    for (&sector, &multiplicity) in unique.iter().zip(&multiplicities) {
        staged.extend(std::iter::repeat_n(sector, multiplicity));
    }
    Ok(staged)
}

struct ValidatedPayload {
    block_index: i64,
    records: u64,
    block: ValidatedThreadblock,
}

fn validate_threadblock_payload(
    raw: &[u8],
    path: &Path,
    kernel: &pb::Kernel,
    allocations: Option<&AllocationRegistry>,
    memory_payload_contract: MemoryPayloadContract,
) -> Result<ValidatedPayload, TraceError> {
    if raw.is_empty() {
        return Err(TraceError::ThreadblockRecordsEmpty {
            path: path.to_path_buf(),
        });
    }
    let threadblock = decode_threadblock(raw, path)?;
    let block_index = block_launch_index(kernel, &threadblock, path)?;
    let records = threadblock.warps.values().try_fold(0_u64, |total, warp| {
        let count =
            u64::try_from(warp.instructions.len()).map_err(|_| TraceError::TraceCountOverflow {
                counter: "threadblock instruction count",
            })?;
        total
            .checked_add(count)
            .ok_or(TraceError::TraceCountOverflow {
                counter: "threadblock instruction count",
            })
    })?;
    if records == 0 {
        return Err(TraceError::ThreadblockRecordsEmpty {
            path: path.to_path_buf(),
        });
    }
    let label = path.to_string_lossy();
    validate_threadblock_contract(
        &threadblock,
        &label,
        allocations,
        memory_payload_contract,
        kernel.function_unique_id,
    )?;
    let block = ValidatedThreadblock::from_message(
        block_index,
        threadblock,
        memory_payload_contract,
        &label,
    )?;
    Ok(ValidatedPayload {
        block_index,
        records,
        block,
    })
}

fn validate_trace_semantics(
    root: &Path,
    trace: &pb::Trace,
    payload_paths: impl IntoIterator<Item = PathBuf>,
    memory_payload_contract: MemoryPayloadContract,
    mut validate_payload: impl FnMut(&Path, &pb::Kernel) -> Result<ValidatedPayload, TraceError>,
) -> Result<
    (
        TraceValidationSummary,
        BTreeMap<PathBuf, ValidatedThreadblock>,
    ),
    TraceError,
> {
    let payload_paths = payload_paths.into_iter().collect::<BTreeSet<_>>();
    let mut launch_ids = BTreeSet::new();
    let mut consumed_payloads = BTreeSet::new();
    let mut validated_payload = BTreeMap::new();
    let mut payloads_by_parent: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for path in &payload_paths {
        let parent =
            path.parent()
                .ok_or_else(|| TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.to_path_buf(),
                    message: format!("payload path has no launch parent: {}", path.display()),
                })?;
        payloads_by_parent
            .entry(parent.to_path_buf())
            .or_default()
            .push(path.clone());
    }
    let mut launch_count = 0_u64;
    let mut threadblock_count = 0_u64;
    let mut instruction_count = 0_u64;

    for (&device_key, device) in &trace.gpu_device {
        if u32::try_from(device_key).ok() != Some(device.id) {
            return Err(TraceError::DeviceMapKeyMismatch {
                map_key: device_key,
                embedded_id: device.id,
            });
        }
        for (&stream_key, stream) in &device.streams {
            if stream_key != stream.id {
                return Err(TraceError::StreamMapKeyMismatch {
                    map_key: stream_key,
                    embedded_id: stream.id,
                });
            }
            for kernel in &stream.kernels {
                if !launch_ids.insert((device_key, stream_key, kernel.id)) {
                    return Err(TraceError::LaunchIdentityDuplicate {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                    });
                }
                launch_count =
                    launch_count
                        .checked_add(1)
                        .ok_or(TraceError::TraceCountOverflow {
                            counter: "launch count",
                        })?;
                if kernel.function_unique_id <= 0 {
                    return Err(TraceError::FunctionUniqueIdInvalid {
                        record: format!(
                            "root launch device={device_key} stream={stream_key} kernel={}",
                            kernel.id
                        ),
                        got: kernel.function_unique_id,
                    });
                }
                if !positive_geometry(kernel.grid_dim.as_ref()) {
                    return Err(TraceError::LaunchGeometryInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        field: "grid_dim",
                    });
                }
                if !positive_geometry(kernel.block_dim.as_ref()) {
                    return Err(TraceError::LaunchGeometryInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        field: "block_dim",
                    });
                }
                validate_shared_memory_split(
                    device_key,
                    stream_key,
                    kernel,
                    memory_payload_contract,
                )?;

                let launch_dir = root
                    .join("threadblocks")
                    .join(format!("device_{device_key}"))
                    .join(format!("stream_{stream_key}"))
                    .join(format!("kernel_{}", kernel.id));
                let Some(paths) = payloads_by_parent.get(launch_dir.as_path()) else {
                    return Err(TraceError::ThreadblocksMissing { dir: launch_dir });
                };
                let mut block_ids = BTreeSet::new();
                for path in paths {
                    let validated = validate_payload(path, kernel)?;
                    if !block_ids.insert(validated.block_index) {
                        return Err(TraceError::BlockIdDuplicate {
                            path: path.to_path_buf(),
                            index: validated.block_index,
                        });
                    }
                    validated_payload.insert(path.clone(), validated.block);
                    consumed_payloads.insert(path.clone());
                    threadblock_count =
                        threadblock_count
                            .checked_add(1)
                            .ok_or(TraceError::TraceCountOverflow {
                                counter: "threadblock count",
                            })?;
                    instruction_count = instruction_count.checked_add(validated.records).ok_or(
                        TraceError::TraceCountOverflow {
                            counter: "instruction count",
                        },
                    )?;
                }
            }
        }
    }
    if launch_count == 0 {
        return Err(TraceError::TraceLaunchesEmpty);
    }
    if let Some(path) = payload_paths
        .iter()
        .find(|path| !consumed_payloads.contains(path.as_path()))
    {
        return Err(TraceError::ThreadblockPayloadOrphan { path: path.clone() });
    }
    Ok((
        TraceValidationSummary {
            launch_count,
            threadblock_count,
            instruction_count,
        },
        validated_payload,
    ))
}

fn validate_trace_snapshot_semantics(
    root: &Path,
    trace: &pb::Trace,
    payload: &BTreeMap<PathBuf, Vec<u8>>,
    allocations: Option<&AllocationRegistry>,
    memory_payload_contract: MemoryPayloadContract,
) -> Result<
    (
        TraceValidationSummary,
        BTreeMap<PathBuf, ValidatedThreadblock>,
    ),
    TraceError,
> {
    validate_trace_semantics(
        root,
        trace,
        payload.keys().cloned(),
        memory_payload_contract,
        |path, kernel| {
            let raw =
                payload
                    .get(path)
                    .ok_or_else(|| TraceError::SamplingPayloadSnapshotMissing {
                        path: path.to_path_buf(),
                    })?;
            validate_threadblock_payload(raw, path, kernel, allocations, memory_payload_contract)
        },
    )
}

pub struct TraceDir {
    root: PathBuf,
    trace: pb::Trace,
    memory_payload_contract: MemoryPayloadContract,
    allocations: Option<AllocationRegistry>,
    #[cfg(feature = "campaign")]
    sampling: Option<TraceSamplingSidecar>,
    stats: Option<TraceStats>,
    validation: TraceValidationSummary,
    payload_by_path: BTreeMap<PathBuf, Vec<u8>>,
    raw_payload_retained: bool,
    threadblock_by_path: BTreeMap<PathBuf, ValidatedThreadblock>,
}

impl TraceDir {
    /// Snapshot and fully validate one trace directory before exposing any
    /// decoded launch or payload. All paths are descriptor-anchored, every
    /// included payload is semantically validated, and its decoded tree is then
    /// compacted before exposure.
    #[cfg(feature = "campaign")]
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        Self::from_snapshot(TraceDirSnapshot::capture(root)?)
    }

    /// Load a trace directly for interactive research.
    ///
    /// This path reads only the simulator inputs, does not read or validate a
    /// campaign `trace_sampling.json`, and does not calculate artifact digests.
    /// It retains protobuf, launch, allocation, and payload-semantic checks so
    /// malformed data cannot silently become a simulation result.
    pub fn open_direct(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        let root = root.into();
        let reader = AnchoredDir::open(&root)?;
        let dynamic_trace_pb = reader
            .read_bounded_regular(Path::new("dynamic_trace.pb"), DYNAMIC_TRACE_PB_MAX_BYTES)?
            .into_raw();
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty {
                path: root.join("dynamic_trace.pb"),
            });
        }
        let allocations_json = reader
            .read_optional_bounded_regular(
                Path::new("extra_info/allocations.json"),
                ALLOCATIONS_JSON_MAX_BYTES,
            )?
            .map(AnchoredFile::into_raw);
        let stats_csv = reader
            .read_optional_bounded_regular(Path::new("stats.csv"), STATS_CSV_MAX_BYTES)?
            .map(AnchoredFile::into_raw);
        let tree = reader
            .read_pb_tree(
                Path::new("threadblocks"),
                DIRECT_THREADBLOCK_FILE_MAX_COUNT,
                DIRECT_THREADBLOCK_TOTAL_MAX_BYTES,
            )
            .map_err(|source| map_threadblock_tree_read_error(&root, source, false))?;
        if tree.files.is_empty() {
            return Err(TraceError::TraceThreadblocksEmpty {
                dir: root.join("threadblocks"),
            });
        }
        let mut payload_by_path = BTreeMap::new();
        for entry in tree.files {
            let relative = entry.relative.to_str().ok_or_else(|| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!("payload path is not UTF-8: {}", entry.relative.display()),
                }
            })?;
            if !safe_payload_manifest_path(relative) {
                return Err(TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!("payload path is not canonical: {relative:?}"),
                });
            }
            payload_by_path.insert(root.join(relative), entry.file.into_raw());
        }
        Self::from_loaded_bytes(
            root,
            dynamic_trace_pb,
            allocations_json,
            stats_csv,
            payload_by_path,
        )
    }

    /// Use the complete direct-loader validation path, then release immutable
    /// raw threadblock protobuf bytes before replay constructs per-SM state.
    /// Compact validated threadblocks and the canonical path roster remain.
    pub fn open_direct_compact(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        let mut trace = Self::open_direct(root)?;
        trace.payload_by_path.clear();
        trace.raw_payload_retained = false;
        Ok(trace)
    }

    /// Capture a complete bounded payload roster, then read, validate, and
    /// compact one exact rostered file generation at a time. Raw payload bytes
    /// are dropped before the next file is opened. Campaign and raw-retaining
    /// loaders remain separate APIs.
    pub fn open_direct_streaming(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        let root = root.into();
        let reader = AnchoredDir::open(&root)?;
        let dynamic_trace_pb = reader
            .read_bounded_regular(Path::new("dynamic_trace.pb"), DYNAMIC_TRACE_PB_MAX_BYTES)?
            .into_raw();
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty {
                path: root.join("dynamic_trace.pb"),
            });
        }
        let allocations_json = reader
            .read_optional_bounded_regular(
                Path::new("extra_info/allocations.json"),
                ALLOCATIONS_JSON_MAX_BYTES,
            )?
            .map(AnchoredFile::into_raw);
        let stats_csv = reader
            .read_optional_bounded_regular(Path::new("stats.csv"), STATS_CSV_MAX_BYTES)?
            .map(AnchoredFile::into_raw);
        let roster = reader
            .roster_pb_tree(
                Path::new("threadblocks"),
                DIRECT_THREADBLOCK_FILE_MAX_COUNT,
                DIRECT_THREADBLOCK_TOTAL_MAX_BYTES,
            )
            .map_err(|source| map_threadblock_tree_read_error(&root, source, false))?;
        if roster.files.is_empty() {
            return Err(TraceError::TraceThreadblocksEmpty {
                dir: root.join("threadblocks"),
            });
        }
        let mut roster_by_path = BTreeMap::new();
        for entry in roster.files {
            let relative = entry.relative.to_str().ok_or_else(|| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!("payload path is not UTF-8: {}", entry.relative.display()),
                }
            })?;
            if !safe_payload_manifest_path(relative) {
                return Err(TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!("payload path is not canonical: {relative:?}"),
                });
            }
            roster_by_path.insert(root.join(relative), (entry.relative, entry.generation));
        }

        let dynamic_path = root.join("dynamic_trace.pb");
        wire_preflight::preflight_trace(&dynamic_trace_pb).map_err(|error| {
            map_wire_preflight_error(&dynamic_path, "dynamic_trace.Trace", error)
        })?;
        let trace = pb::Trace::decode(dynamic_trace_pb.as_slice()).map_err(|source| {
            TraceError::Decode {
                path: dynamic_path.clone(),
                message_type: "dynamic_trace.Trace",
                source,
            }
        })?;
        let memory_payload_contract = classify_memory_payload_contract(&trace, &dynamic_path)?;
        let allocations_path = root.join("extra_info/allocations.json");
        let allocations = allocations_json
            .as_deref()
            .map(|raw| AllocationRegistry::parse(&allocations_path, raw))
            .transpose()?;
        let (validation, threadblock_by_path) = validate_trace_semantics(
            &root,
            &trace,
            roster_by_path.keys().cloned(),
            memory_payload_contract,
            |path, kernel| {
                let (relative, generation) = roster_by_path.get(path).ok_or_else(|| {
                    TraceError::SamplingPayloadSnapshotMissing {
                        path: path.to_path_buf(),
                    }
                })?;
                let file = reader
                    .read_rostered_regular(relative, *generation)
                    .map_err(|source| map_threadblock_tree_read_error(&root, source, false))?;
                validate_threadblock_payload(
                    file.raw(),
                    path,
                    kernel,
                    allocations.as_ref(),
                    memory_payload_contract,
                )
            },
        )?;
        let stats_path = root.join("stats.csv");
        let stats = stats_csv
            .as_deref()
            .map(|raw| parse_trace_stats(&stats_path, raw))
            .transpose()?;
        Ok(Self {
            root,
            trace,
            memory_payload_contract,
            allocations,
            #[cfg(feature = "campaign")]
            sampling: None,
            stats,
            validation,
            payload_by_path: BTreeMap::new(),
            raw_payload_retained: false,
            threadblock_by_path,
        })
    }

    /// Decode a one-read manifest snapshot without reopening any artifact.
    #[cfg(feature = "campaign")]
    pub fn from_snapshot(snapshot: TraceDirSnapshot) -> Result<Self, TraceError> {
        let TraceDirSnapshot {
            root,
            dynamic_trace_pb,
            allocations_json,
            trace_sampling_json,
            stats_csv,
            threadblock_payload,
            identity: _,
            file_identities: _,
        } = snapshot;
        let sampling_path = root.join("trace_sampling.json");
        let sampling = trace_sampling_json
            .as_deref()
            .map(|raw| TraceSamplingSidecar::parse(&sampling_path, raw))
            .transpose()?;
        if let Some(sidecar) = sampling.as_ref() {
            validate_sampling_payload_snapshot(&root, sidecar, &threadblock_payload)?;
        }
        Self::from_loaded_bytes(
            root,
            dynamic_trace_pb,
            allocations_json,
            stats_csv,
            threadblock_payload.by_path,
        )
        .map(|mut trace| {
            trace.sampling = sampling;
            trace
        })
    }

    fn from_loaded_bytes(
        root: PathBuf,
        dynamic_trace_pb: Vec<u8>,
        allocations_json: Option<Vec<u8>>,
        stats_csv: Option<Vec<u8>>,
        payload_by_path: BTreeMap<PathBuf, Vec<u8>>,
    ) -> Result<Self, TraceError> {
        let dynamic_path = root.join("dynamic_trace.pb");
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty { path: dynamic_path });
        }
        wire_preflight::preflight_trace(&dynamic_trace_pb).map_err(|error| {
            map_wire_preflight_error(&dynamic_path, "dynamic_trace.Trace", error)
        })?;
        let trace = pb::Trace::decode(dynamic_trace_pb.as_slice()).map_err(|source| {
            TraceError::Decode {
                path: dynamic_path.clone(),
                message_type: "dynamic_trace.Trace",
                source,
            }
        })?;
        let memory_payload_contract = classify_memory_payload_contract(&trace, &dynamic_path)?;
        let allocations_path = root.join("extra_info/allocations.json");
        let allocations = allocations_json
            .as_deref()
            .map(|raw| AllocationRegistry::parse(&allocations_path, raw))
            .transpose()?;
        let (validation, threadblock_by_path) = validate_trace_snapshot_semantics(
            &root,
            &trace,
            &payload_by_path,
            allocations.as_ref(),
            memory_payload_contract,
        )?;
        let stats_path = root.join("stats.csv");
        let stats = stats_csv
            .as_deref()
            .map(|raw| parse_trace_stats(&stats_path, raw))
            .transpose()?;
        Ok(Self {
            root,
            trace,
            memory_payload_contract,
            allocations,
            #[cfg(feature = "campaign")]
            sampling: None,
            stats,
            validation,
            payload_by_path,
            raw_payload_retained: true,
            threadblock_by_path,
        })
    }

    /// The loaded allocation registry, if `allocations.json` was present.
    pub fn allocations(&self) -> Option<&AllocationRegistry> {
        self.allocations.as_ref()
    }

    /// The validated trace-sampling sidecar (`trace_sampling.json`), if present.
    /// Supported sidecars distinguish sampled-subset provenance from a retained
    /// lossless-compression declaration; neither alone proves full coverage.
    #[cfg(feature = "campaign")]
    pub fn sampling_sidecar(&self) -> Option<&TraceSamplingSidecar> {
        self.sampling.as_ref()
    }

    pub fn stats(&self) -> Option<TraceStats> {
        self.stats
    }

    pub fn validation_summary(&self) -> TraceValidationSummary {
        self.validation
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn trace(&self) -> &pb::Trace {
        &self.trace
    }

    /// Validated payload-generation contract for every instruction in this trace.
    pub fn memory_payload_contract(&self) -> MemoryPayloadContract {
        self.memory_payload_contract
    }

    /// All kernel launches, in deterministic (device, stream, launch) order.
    /// Map fields are BTreeMaps (see pbgen), so iteration order is stable.
    pub fn kernels(&self) -> impl Iterator<Item = KernelLoc<'_>> {
        self.trace.gpu_device.iter().flat_map(|(&device_id, dev)| {
            dev.streams.iter().flat_map(move |(&stream_id, stream)| {
                stream.kernels.iter().map(move |kernel| KernelLoc {
                    device_id,
                    stream_id,
                    kernel,
                })
            })
        })
    }

    fn require_kernel_loc(&self, loc: &KernelLoc<'_>) -> Result<(), TraceError> {
        let belongs_to_root = self
            .trace
            .gpu_device
            .get(&loc.device_id)
            .and_then(|device| device.streams.get(&loc.stream_id))
            .is_some_and(|stream| {
                stream
                    .kernels
                    .iter()
                    .any(|kernel| std::ptr::eq(kernel, loc.kernel))
            });
        if !belongs_to_root {
            return Err(TraceError::KernelLocatorForeign {
                root: self.root.clone(),
                device_id: loc.device_id,
                stream_id: loc.stream_id,
                kernel_id: loc.kernel.id,
            });
        }
        Ok(())
    }

    /// Paths of the per-threadblock `.pb` files for one kernel launch, in
    /// deterministic FILENAME order — which is NOT flattened-block launch
    /// order (filenames embed unpadded decimal coordinates, so "10" sorts
    /// before "2"). Block identity/order must come from the proto's
    /// `block_id`: use [`TraceDir::validated_threadblocks_in_launch_order`] for
    /// replay's compact view, or [`TraceDir::read_threadblocks_in_launch_order`]
    /// when the selected loader retained raw protobuf bytes. The list always
    /// comes from the descriptor-anchored payload snapshot captured by the
    /// loader, never a second directory walk.
    ///
    /// A kernel launch with no snapshotted payload is a typed error.
    pub fn threadblock_paths(&self, loc: &KernelLoc<'_>) -> Result<Vec<PathBuf>, TraceError> {
        self.require_kernel_loc(loc)?;
        let dir = self
            .root
            .join("threadblocks")
            .join(format!("device_{}", loc.device_id))
            .join(format!("stream_{}", loc.stream_id))
            .join(format!("kernel_{}", loc.kernel.id));
        let mut paths: Vec<_> = self
            .threadblock_by_path
            .keys()
            .filter(|path| path.parent() == Some(dir.as_path()))
            .cloned()
            .collect();
        if paths.is_empty() {
            return Err(TraceError::ThreadblocksMissing { dir });
        }
        paths.sort();
        Ok(paths)
    }

    /// Borrow the compact threadblock produced by the semantic validation walk.
    /// Replay uses this view so non-memory Prost instruction trees are not retained
    /// and no payload is decoded a second time.
    pub fn validated_threadblock(&self, path: &Path) -> Result<&ValidatedThreadblock, TraceError> {
        self.threadblock_by_path.get(path).ok_or_else(|| {
            TraceError::SamplingPayloadSnapshotMissing {
                path: path.to_path_buf(),
            }
        })
    }

    /// Borrow every compact validated threadblock for one kernel launch in
    /// flattened CUDA block order. This works for every loader, including
    /// [`TraceDir::open_direct_compact`], because it does not require retained
    /// raw protobuf bytes.
    pub fn validated_threadblocks_in_launch_order<'a>(
        &'a self,
        loc: &KernelLoc<'_>,
    ) -> Result<Vec<(i64, &'a ValidatedThreadblock)>, TraceError> {
        let mut blocks = self
            .threadblock_paths(loc)?
            .into_iter()
            .map(|path| {
                self.validated_threadblock(&path)
                    .map(|block| (block.launch_index(), block))
            })
            .collect::<Result<Vec<_>, _>>()?;
        blocks.sort_by_key(|(index, _)| *index);
        Ok(blocks)
    }

    /// Decode one threadblock from the immutable raw bytes captured by the
    /// selected loader. Paths outside that snapshot fail as missing inputs.
    pub fn read_threadblock(&self, path: &Path) -> Result<pb::Threadblock, TraceError> {
        if !self.threadblock_by_path.contains_key(path) {
            return Err(TraceError::SamplingPayloadSnapshotMissing {
                path: path.to_path_buf(),
            });
        }
        if !self.raw_payload_retained {
            return Err(TraceError::RawThreadblockPayloadUnavailable {
                path: path.to_path_buf(),
            });
        }
        let raw = self.payload_by_path.get(path).ok_or_else(|| {
            TraceError::SamplingPayloadSnapshotMissing {
                path: path.to_path_buf(),
            }
        })?;
        decode_threadblock(raw, path)
    }

    /// Read every threadblock of one kernel launch, keyed and sorted by the
    /// FLATTENED block index derived from the proto's canonical `block_id`
    /// (CUDA row-major: `x + y*gx + z*gx*gy` — the same rule as the Python
    /// consumer `l1_production_helper._flatten_block_id`). Fail-closed on
    /// missing `block_id`, out-of-grid coordinates, and duplicates.
    ///
    /// Loads the whole kernel's blocks into memory (fine for the current
    /// roster; replay uses [`TraceDir::validated_threadblock`] instead). This
    /// raw-decoding API requires a raw-retaining loader; a trace returned by
    /// [`TraceDir::open_direct_compact`] reports
    /// [`TraceError::RawThreadblockPayloadUnavailable`]. Blocks decode from the
    /// immutable bytes captured by the loader, so later path replacement cannot
    /// change the returned content.
    pub fn read_threadblocks_in_launch_order(
        &self,
        loc: &KernelLoc<'_>,
    ) -> Result<Vec<(i64, pb::Threadblock)>, TraceError> {
        let mut blocks = Vec::new();
        for path in self.threadblock_paths(loc)? {
            let index = self.validated_threadblock(&path)?.launch_index();
            let tb = self.read_threadblock(&path)?;
            // Reader-side allocation-identity gate (address-carrying contract): a GLOBAL/GENERIC-
            // space active record must carry a resolvable, in-range allocation
            // id — parity with the Python validator. Fail-closed here so the sim
            // never consumes a malformed global / global-atomic / LDGSTS-source
            // record even if it bypassed the offline validator.
            for (&warp_id, warp) in &tb.warps {
                for (ix, inst) in warp.instructions.iter().enumerate() {
                    check_allocation_identity(inst, self.allocations.as_ref(), || {
                        format!("{} warp={warp_id} inst={ix}", path.display())
                    })?;
                }
            }
            blocks.push((index, path, tb));
        }
        blocks.sort_by_key(|(index, _, _)| *index);
        for w in blocks.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(TraceError::BlockIdDuplicate {
                    path: w[1].1.clone(),
                    index: w[1].0,
                });
            }
        }
        Ok(blocks
            .into_iter()
            .map(|(index, _, tb)| (index, tb))
            .collect())
    }
}

/// Flattened launch index of one threadblock from the proto's canonical
/// `block_id` (CUDA row-major: `x + y*gx + z*gx*gy` — the same rule as the
/// Python consumer `l1_production_helper._flatten_block_id`). Fail-closed
/// on missing `block_id` and out-of-grid coordinates; duplicate detection
/// is the caller's job (it needs the full block set).
/// The validated `(gx, gy, gz)` grid of a launch record — the same presence +
/// strictly-positive check `block_launch_index` applies. Used by the reader/replay
/// completeness gate to compute `grid_total` and cross-check `gridDim` against a
/// trace-sampling sidecar's `original_grid`.
pub fn kernel_grid(kernel: &pb::Kernel) -> Result<(u32, u32, u32), TraceError> {
    let g = kernel
        .grid_dim
        .as_ref()
        .filter(|g| g.x > 0 && g.y > 0 && g.z > 0)
        .ok_or(TraceError::GridDimMissing {
            kernel_id: kernel.id,
        })?;
    // The filter guarantees strictly-positive i32 dims, so the u32 cast is safe.
    Ok((g.x as u32, g.y as u32, g.z as u32))
}

/// Method-specific provenance retained from a supported v2 sampling sidecar.
/// Unsupported schema methods never construct this enum.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceSamplingMethod {
    /// Sealed lossless-compression provenance. Replay must independently observe
    /// full launch coverage; this variant never sanctions incompleteness.
    LosslessCompressedFull(LosslessFullProvenance),
    /// A deeply validated fixed-point subset envelope. The selected membership
    /// and seals remain bound as one variant instead of a boolean plus optional
    /// fields that callers could combine inconsistently.
    WaveSegmentStratifiedSubset(WaveSegmentSubsetProvenance),
}

#[cfg(feature = "campaign")]
impl TraceSamplingMethod {
    pub fn selected_block_ids(&self) -> Option<&BTreeSet<i64>> {
        match self {
            Self::LosslessCompressedFull(_) => None,
            Self::WaveSegmentStratifiedSubset(provenance) => Some(&provenance.selected_block_ids),
        }
    }

    pub fn sanctions_incomplete(&self) -> bool {
        matches!(self, Self::WaveSegmentStratifiedSubset(_))
    }
}

/// Hash identity retained from a sealed lossless-compression sidecar. This
/// declaration remains distinct from the launch-bound full-coverage witness
/// that replay establishes from the decoded block roster.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LosslessFullProvenance {
    selected_block_ids_sha256: String,
    sidecar_sha256: String,
}

#[cfg(feature = "campaign")]
impl LosslessFullProvenance {
    pub fn selected_block_ids_sha256(&self) -> &str {
        &self.selected_block_ids_sha256
    }

    pub fn sidecar_sha256(&self) -> &str {
        &self.sidecar_sha256
    }
}

/// Identity retained from a Python-canonical, deeply validated subset sidecar.
/// Payload bytes are additionally verified by [`TraceDir::open`] before callers
/// can obtain this provenance through [`TraceDir::sampling_sidecar`].
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveSegmentSubsetProvenance {
    selected_block_ids: BTreeSet<i64>,
    selected_block_ids_sha256: String,
    sidecar_sha256: String,
}

#[cfg(feature = "campaign")]
impl WaveSegmentSubsetProvenance {
    pub fn selected_block_ids(&self) -> &BTreeSet<i64> {
        &self.selected_block_ids
    }

    pub fn selected_block_ids_sha256(&self) -> &str {
        &self.selected_block_ids_sha256
    }

    pub fn sidecar_sha256(&self) -> &str {
        &self.sidecar_sha256
    }
}

/// One canonical entry from `hashes.per_block_raw_sha256_manifest`.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PayloadFileManifestEntry {
    size_bytes: u64,
    sha256: String,
}

/// Full-launch block-to-SM relation retained only after the sidecar parser proves
/// that every launch index occurs exactly once. This is spatial placement; it is
/// not evidence of temporal block co-residency.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedPerSmSlotMap {
    sm_by_launch_index: Box<[u64]>,
}

#[cfg(feature = "campaign")]
impl ValidatedPerSmSlotMap {
    fn retain(
        roster_by_sm: &BTreeMap<u64, Vec<u64>>,
        expected_blocks: u64,
    ) -> Result<Self, String> {
        let mut placement_by_launch_index = BTreeMap::new();
        for (&sm_id, blocks) in roster_by_sm {
            for &block_id in blocks {
                if block_id >= expected_blocks {
                    return Err(format!(
                        "block {block_id} is outside retained per_sm_slot_map cardinality {expected_blocks}"
                    ));
                }
                if placement_by_launch_index.insert(block_id, sm_id).is_some() {
                    return Err(format!(
                        "block {block_id} occurs more than once in retained per_sm_slot_map"
                    ));
                }
            }
        }
        let actual_blocks = u64::try_from(placement_by_launch_index.len())
            .map_err(|_| "per_sm_slot_map cardinality exceeds u64".to_string())?;
        if actual_blocks != expected_blocks {
            let first_missing = placement_by_launch_index
                .keys()
                .copied()
                .zip(0_u64..)
                .find_map(|(actual, expected)| (actual != expected).then_some(expected))
                .unwrap_or(actual_blocks);
            return Err(format!(
                "launch index {first_missing} is absent from retained per_sm_slot_map"
            ));
        }
        Ok(Self {
            sm_by_launch_index: placement_by_launch_index
                .into_values()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }
}

/// Reader-side view of a supported `<root>/trace_sampling.json` v2 sidecar.
/// Fields are private so method-specific invariants can only be constructed by
/// the closed parser. A sealed lossless declaration is not itself a
/// full-coverage witness. Launch-bound full versus sampled coverage is
/// established later by replay after observing block membership and record counts.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceSamplingSidecar {
    cell_id: String,
    kernel_mangled_name: String,
    kernel_args_hash: String,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_grid: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    expected_full_records: u64,
    captured_records: u64,
    decompressed_payload_sha256: String,
    payload_file_manifest: BTreeMap<String, PayloadFileManifestEntry>,
    raw_payload_bytes: u64,
    per_sm_slot_map: ValidatedPerSmSlotMap,
    method: TraceSamplingMethod,
}

#[cfg(feature = "campaign")]
impl TraceSamplingSidecar {
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }

    pub fn kernel_mangled_name(&self) -> &str {
        &self.kernel_mangled_name
    }

    pub fn kernel_args_hash(&self) -> &str {
        &self.kernel_args_hash
    }

    pub fn original_grid(&self) -> (u32, u32, u32) {
        self.original_grid
    }

    pub fn original_block(&self) -> (u32, u32, u32) {
        self.original_block
    }

    pub fn captured_grid(&self) -> (u32, u32, u32) {
        self.captured_grid
    }

    pub fn captured_block(&self) -> (u32, u32, u32) {
        self.captured_block
    }

    pub fn expected_full_records(&self) -> u64 {
        self.expected_full_records
    }

    pub fn captured_records(&self) -> u64 {
        self.captured_records
    }

    pub fn decompressed_payload_sha256(&self) -> &str {
        &self.decompressed_payload_sha256
    }

    pub fn raw_payload_bytes(&self) -> u64 {
        self.raw_payload_bytes
    }

    /// Iterate the sidecar's validated full-launch spatial placement in launch
    /// index order. The map says where each block was observed, not which blocks
    /// were resident concurrently.
    pub fn declared_block_placements(&self) -> impl ExactSizeIterator<Item = (u64, u64)> + '_ {
        self.per_sm_slot_map
            .sm_by_launch_index
            .iter()
            .copied()
            .enumerate()
            .map(|(launch_index, sm_id)| (launch_index as u64, sm_id))
    }

    pub fn declared_sm_for_launch_index(&self, launch_index: i64) -> Option<u64> {
        usize::try_from(launch_index)
            .ok()
            .and_then(|index| self.per_sm_slot_map.sm_by_launch_index.get(index))
            .copied()
    }

    pub fn method(&self) -> &TraceSamplingMethod {
        &self.method
    }
}

#[cfg(feature = "campaign")]
fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(feature = "campaign")]
const PAYLOAD_FILE_MANIFEST_PREFIX: &str = "trace-payload-file-manifest-v1\n";

fn safe_payload_manifest_path(path: &str) -> bool {
    if path.is_empty()
        || !path.is_ascii()
        || path
            .bytes()
            .any(|byte| !(b'!'..=b'~').contains(&byte) || byte == b'\\')
        || !path.ends_with(".pb")
    {
        return false;
    }
    let Some(file_name) = path.rsplit('/').next() else {
        return false;
    };
    let Some(file_stem) = file_name.strip_suffix(".pb") else {
        return false;
    };
    if file_stem.bytes().all(|byte| byte == b'.') {
        return false;
    }
    let mut parts = path.split('/');
    if parts.next() != Some("threadblocks") {
        return false;
    }
    let mut saw_payload_component = false;
    for part in parts {
        if part.is_empty() || part == "." || part == ".." {
            return false;
        }
        saw_payload_component = true;
    }
    saw_payload_component
}

#[cfg(feature = "campaign")]
fn parse_payload_file_manifest(
    manifest: &str,
) -> Result<BTreeMap<String, PayloadFileManifestEntry>, String> {
    let body = manifest
        .strip_prefix(PAYLOAD_FILE_MANIFEST_PREFIX)
        .and_then(|value| value.strip_suffix('\n'))
        .ok_or_else(|| {
            "per_block_raw_sha256_manifest must use the canonical v1 header and trailing newline"
                .to_string()
        })?;
    if body.is_empty() {
        return Err("per_block_raw_sha256_manifest must contain at least one .pb file".to_string());
    }

    let mut entries = BTreeMap::new();
    let mut previous_path: Option<&str> = None;
    for line in body.split('\n') {
        let mut fields = line.split('\t');
        let path = fields.next().unwrap_or_default();
        let size_text = fields.next().ok_or_else(|| {
            "per_block_raw_sha256_manifest entries must be path<TAB>size<TAB>sha256".to_string()
        })?;
        let sha256 = fields.next().ok_or_else(|| {
            "per_block_raw_sha256_manifest entries must be path<TAB>size<TAB>sha256".to_string()
        })?;
        if fields.next().is_some() {
            return Err(
                "per_block_raw_sha256_manifest entries must be path<TAB>size<TAB>sha256"
                    .to_string(),
            );
        }
        if !safe_payload_manifest_path(path) {
            return Err(format!(
                "per_block_raw_sha256_manifest path is not a canonical threadblocks .pb path: {path:?}"
            ));
        }
        if previous_path.is_some_and(|previous| path <= previous) {
            return Err(
                "per_block_raw_sha256_manifest paths must be unique and byte-sorted".to_string(),
            );
        }
        if size_text.is_empty()
            || !size_text.bytes().all(|byte| byte.is_ascii_digit())
            || (size_text.len() > 1 && size_text.starts_with('0'))
        {
            return Err(format!(
                "per_block_raw_sha256_manifest size is not canonical decimal: {size_text:?}"
            ));
        }
        let size_bytes = size_text.parse::<u64>().map_err(|_| {
            format!("per_block_raw_sha256_manifest size exceeds u64: {size_text:?}")
        })?;
        if !is_lower_hex_sha256(sha256) {
            return Err(
                "per_block_raw_sha256_manifest hashes must be lowercase SHA-256".to_string(),
            );
        }
        previous_path = Some(path);
        entries.insert(
            path.to_string(),
            PayloadFileManifestEntry {
                size_bytes,
                sha256: sha256.to_string(),
            },
        );
    }
    Ok(entries)
}

#[cfg(feature = "campaign")]
fn sha256_hex(input: &[u8]) -> String {
    Sha256Digest::of(input).to_string()
}

#[cfg(feature = "campaign")]
fn exact_keys(
    object: &serde_json::Map<String, serde_json::Value>,
    required: &[&str],
    allowed: &[&str],
    context: &str,
) -> Result<(), String> {
    for key in required {
        if !object.contains_key(*key) {
            return Err(format!("{context} is missing required field {key}"));
        }
    }
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{context} contains unknown field {key}"));
        }
    }
    Ok(())
}

#[cfg(feature = "campaign")]
fn object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

#[cfg(feature = "campaign")]
fn required<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
    context: &str,
) -> Result<&'a serde_json::Value, String> {
    object
        .get(field)
        .ok_or_else(|| format!("{context} is missing required field {field}"))
}

#[cfg(feature = "campaign")]
fn positive_u32(value: &serde_json::Value, context: &str) -> Result<u32, String> {
    let value = value
        .as_u64()
        .and_then(|raw| u32::try_from(raw).ok())
        .filter(|raw| *raw > 0)
        .ok_or_else(|| format!("{context} must be a positive u32"))?;
    Ok(value)
}

#[cfg(feature = "campaign")]
fn parse_dim3(value: &serde_json::Value, context: &str) -> Result<(u32, u32, u32), String> {
    let dim = object(value, context)?;
    exact_keys(dim, &["x", "y", "z"], &["x", "y", "z"], context)?;
    Ok((
        positive_u32(required(dim, "x", context)?, &format!("{context}.x"))?,
        positive_u32(required(dim, "y", context)?, &format!("{context}.y"))?,
        positive_u32(required(dim, "z", context)?, &format!("{context}.z"))?,
    ))
}

#[cfg(feature = "campaign")]
fn canonical_sidecar_sha256(sidecar: &serde_json::Value) -> Result<String, String> {
    let mut unsigned = sidecar.clone();
    let root = unsigned
        .as_object_mut()
        .ok_or_else(|| "trace_sampling.json root must be an object".to_string())?;
    let hashes = root
        .get_mut("hashes")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| "hashes must be an object".to_string())?;
    if hashes.remove("sidecar_sha256").is_none() {
        return Err("hashes is missing required field sidecar_sha256".to_string());
    }
    let canonical = canonical_json(&unsigned).map_err(|error| error.to_string())?;
    Ok(sha256_hex(canonical.as_bytes()))
}

#[cfg(feature = "campaign")]
fn parse_unique_ids(
    value: &serde_json::Value,
    context: &str,
    allow_empty: bool,
) -> Result<BTreeSet<u64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    if !allow_empty && values.is_empty() {
        return Err(format!("{context} must not be empty"));
    }
    let mut ids = BTreeSet::new();
    for value in values {
        let id = value
            .as_u64()
            .ok_or_else(|| format!("{context} entries must be nonnegative integers"))?;
        if !ids.insert(id) {
            return Err(format!("{context} contains duplicate id {id}"));
        }
    }
    Ok(ids)
}

#[cfg(feature = "campaign")]
fn unsigned(value: &serde_json::Value, context: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{context} must be a nonnegative integer"))
}

#[cfg(feature = "campaign")]
fn positive(value: &serde_json::Value, context: &str) -> Result<u64, String> {
    unsigned(value, context).and_then(|value| {
        if value == 0 {
            Err(format!("{context} must be positive"))
        } else {
            Ok(value)
        }
    })
}

#[cfg(feature = "campaign")]
fn boolean(value: &serde_json::Value, context: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{context} must be boolean"))
}

#[cfg(feature = "campaign")]
fn string<'a>(value: &'a serde_json::Value, context: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{context} must be a string"))
}

#[cfg(feature = "campaign")]
fn nonnegative_number(value: &serde_json::Value, context: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{context} must be a finite nonnegative number"))
}

#[cfg(feature = "campaign")]
fn string_array(value: &serde_json::Value, context: &str) -> Result<(), String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    for value in values {
        string(value, &format!("{context} entry"))?;
    }
    Ok(())
}

#[cfg(feature = "campaign")]
fn sampling_cardinalities_match(sample: u64, ids_len: u64) -> bool {
    sample == ids_len
}

/// Convert the complete `u64` domain through two exactly representable `u32`
/// limbs. The final addition performs the one unavoidable binary64 rounding;
/// no unchecked integer-to-float narrowing hides which source bits participate.
#[cfg(feature = "campaign")]
fn sampling_u64_as_f64(value: u64) -> f64 {
    let bytes = value.to_be_bytes();
    let high = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let low = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

#[cfg(feature = "campaign")]
fn relative_close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12 * left.abs().max(right.abs())
}

#[cfg(feature = "campaign")]
fn close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12_f64.max(1e-12 * left.abs().max(right.abs()))
}

#[cfg(feature = "campaign")]
fn validate_lossless_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_grid: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_block_ids_sha256: &str,
) -> Result<(u64, ValidatedPerSmSlotMap), String> {
    const LOSSLESS_FIELDS: &[&str] = &[
        "schema_version",
        "cell_id",
        "kernel",
        "method",
        "reason",
        "shape",
        "gates",
        "wave_model",
        "per_sm_slot_map",
        "rng",
        "label_pairing",
        "tracer",
        "bytes",
        "records",
        "hashes",
    ];
    exact_keys(
        root,
        LOSSLESS_FIELDS,
        LOSSLESS_FIELDS,
        "lossless compressed sidecar",
    )?;

    let reason = string(
        required(root, "reason", "lossless compressed sidecar")?,
        "reason",
    )?;
    if ![
        "byte_budget",
        "walltime_budget",
        "gate_forced",
        "not_breaching",
    ]
    .contains(&reason)
    {
        return Err("reason is not a schema-v2 value".to_string());
    }
    if original_grid != captured_grid || original_block != captured_block {
        return Err("lossless compression cannot rewrite the captured launch shape".to_string());
    }
    let shape = object(
        required(root, "shape", "lossless compressed sidecar")?,
        "shape",
    )?;
    if !shape
        .get("reduced_shape_variant")
        .is_some_and(serde_json::Value::is_null)
    {
        return Err("lossless compression requires reduced_shape_variant=null".to_string());
    }
    if string(
        required(root, "label_pairing", "lossless compressed sidecar")?,
        "label_pairing",
    )? != "full_shape_label"
    {
        return Err("lossless compression requires full_shape_label pairing".to_string());
    }

    let gates = object(
        required(root, "gates", "lossless compressed sidecar")?,
        "gates",
    )?;
    const GATE_FIELDS: &[&str] = &[
        "coop",
        "wave_count",
        "n_measured_waves",
        "tail_measured_waves",
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
        "n_phases",
        "skew_heavy_block_count",
        "certainty_bytes",
        "max_trace_bytes",
        "certainty_budget_exceeded",
        "verdict",
    ];
    exact_keys(gates, GATE_FIELDS, GATE_FIELDS, "gates")?;
    boolean(required(gates, "coop", "gates")?, "gates.coop")?;
    let gate_wave_count = unsigned(required(gates, "wave_count", "gates")?, "gates.wave_count")?;
    let gate_measured = unsigned(
        required(gates, "n_measured_waves", "gates")?,
        "gates.n_measured_waves",
    )?;
    let gate_tail_measured = unsigned(
        required(gates, "tail_measured_waves", "gates")?,
        "gates.tail_measured_waves",
    )?;
    for field in [
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
    ] {
        nonnegative_number(required(gates, field, "gates")?, &format!("gates.{field}"))?;
    }
    positive(required(gates, "n_phases", "gates")?, "gates.n_phases")?;
    unsigned(
        required(gates, "skew_heavy_block_count", "gates")?,
        "gates.skew_heavy_block_count",
    )?;
    let certainty_bytes = match required(gates, "certainty_bytes", "gates")? {
        serde_json::Value::Null => None,
        value => Some(unsigned(value, "gates.certainty_bytes")?),
    };
    let max_trace_bytes = match required(gates, "max_trace_bytes", "gates")? {
        serde_json::Value::Null => None,
        value => Some(positive(value, "gates.max_trace_bytes")?),
    };
    let budget_exceeded = boolean(
        required(gates, "certainty_budget_exceeded", "gates")?,
        "gates.certainty_budget_exceeded",
    )?;
    match certainty_bytes {
        Some(certainty) => {
            let maximum = max_trace_bytes.ok_or_else(|| {
                "gates.certainty_bytes requires gates.max_trace_bytes".to_string()
            })?;
            let expected = 5_u128 * u128::from(certainty) > 2_u128 * u128::from(maximum);
            if budget_exceeded != expected {
                return Err(
                    "gates.certainty_budget_exceeded disagrees with the exact 2/5 rule".to_string(),
                );
            }
        }
        None if budget_exceeded => {
            return Err(
                "gates.certainty_budget_exceeded cannot be true without byte evidence".to_string(),
            );
        }
        None => {}
    }
    let verdict = string(required(gates, "verdict", "gates")?, "gates.verdict")?;
    if ![
        "subset_admissible",
        "coop_forbid",
        "wave_forbid",
        "tail_measurement_forbid",
        "atomic_forbid",
        "skew_to_reduced_shape",
        "certainty_budget_to_reduced_shape",
        "homogeneity_to_reduced_shape",
    ]
    .contains(&verdict)
    {
        return Err("gates.verdict is not a schema-v2 value".to_string());
    }

    let tracer = object(
        required(root, "tracer", "lossless compressed sidecar")?,
        "tracer",
    )?;
    exact_keys(
        tracer,
        &["version", "features", "stats_pass_version"],
        &["version", "features", "stats_pass_version"],
        "tracer",
    )?;
    if unsigned(required(tracer, "version", "tracer")?, "tracer.version")? != 5
        || unsigned(
            required(tracer, "stats_pass_version", "tracer")?,
            "tracer.stats_pass_version",
        )? != 2
    {
        return Err("lossless compression requires tracer v5 and stats pass v2".to_string());
    }
    let features = object(required(tracer, "features", "tracer")?, "tracer.features")?;
    exact_keys(
        features,
        &["atomic_addr", "ldgsts_addr"],
        &["atomic_addr", "ldgsts_addr"],
        "tracer.features",
    )?;
    if !boolean(
        required(features, "atomic_addr", "tracer.features")?,
        "tracer.features.atomic_addr",
    )? || !boolean(
        required(features, "ldgsts_addr", "tracer.features")?,
        "tracer.features.ldgsts_addr",
    )? {
        return Err(
            "lossless compression requires atomic_addr and ldgsts_addr features".to_string(),
        );
    }

    let rng = object(required(root, "rng", "lossless compressed sidecar")?, "rng")?;
    exact_keys(rng, &["algo", "seed"], &["algo", "seed"], "rng")?;
    string(required(rng, "algo", "rng")?, "rng.algo")?;
    if required(rng, "seed", "rng")?.as_i64().is_none() {
        return Err("rng.seed must be an integer".to_string());
    }

    let bytes = object(
        required(root, "bytes", "lossless compressed sidecar")?,
        "bytes",
    )?;
    const BYTE_FIELDS: &[&str] = &["raw", "compressed", "codec", "level", "long", "dict_id"];
    exact_keys(bytes, &["raw", "compressed", "codec"], BYTE_FIELDS, "bytes")?;
    let raw_bytes = unsigned(required(bytes, "raw", "bytes")?, "bytes.raw")?;
    unsigned(required(bytes, "compressed", "bytes")?, "bytes.compressed")?;
    string(required(bytes, "codec", "bytes")?, "bytes.codec")?;
    for field in ["level", "long"] {
        if bytes
            .get(field)
            .is_some_and(|value| value.as_i64().is_none() && value.as_u64().is_none())
        {
            return Err(format!("bytes.{field} must be an integer"));
        }
    }
    if bytes.get("dict_id").is_some_and(|value| {
        !value.is_null() && value.as_i64().is_none() && value.as_u64().is_none()
    }) {
        return Err("bytes.dict_id must be an integer or null".to_string());
    }

    let wave = object(
        required(root, "wave_model", "lossless compressed sidecar")?,
        "wave_model",
    )?;
    const WAVE_FIELDS: &[&str] = &[
        "reconstruction",
        "per_sm_resident_blocks",
        "n_waves",
        "segment_len",
        "ghost_depth",
    ];
    exact_keys(wave, WAVE_FIELDS, WAVE_FIELDS, "wave_model")?;
    if string(
        required(wave, "reconstruction", "wave_model")?,
        "wave_model.reconstruction",
    )? != "launch_index"
    {
        return Err("wave_model.reconstruction must be launch_index".to_string());
    }
    let declared_n_waves = unsigned(
        required(wave, "n_waves", "wave_model")?,
        "wave_model.n_waves",
    )?;
    let segment_len = positive(
        required(wave, "segment_len", "wave_model")?,
        "wave_model.segment_len",
    )?;
    let ghost_depth = unsigned(
        required(wave, "ghost_depth", "wave_model")?,
        "wave_model.ghost_depth",
    )?;
    let residency_values = required(wave, "per_sm_resident_blocks", "wave_model")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "wave_model.per_sm_resident_blocks must be a nonempty array".to_string())?;
    let mut resident_by_sm = BTreeMap::new();
    for (index, value) in residency_values.iter().enumerate() {
        let context = format!("wave_model.per_sm_resident_blocks[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "resident_blocks"],
            &["sm_id", "resident_blocks"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let resident = positive(
            required(row, "resident_blocks", &context)?,
            &format!("{context}.resident_blocks"),
        )?;
        if resident_by_sm.insert(sm, resident).is_some() {
            return Err(format!("duplicate resident-block entry for sm {sm}"));
        }
    }

    let slot_values = required(root, "per_sm_slot_map", "lossless compressed sidecar")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "per_sm_slot_map must be a nonempty array".to_string())?;
    let mut roster_by_sm: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut roster = BTreeSet::new();
    for (index, value) in slot_values.iter().enumerate() {
        let context = format!("per_sm_slot_map[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "block_ids"],
            &["sm_id", "block_ids"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let values = required(row, "block_ids", &context)?
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("{context}.block_ids must be a nonempty array"))?;
        let mut block_ids = Vec::with_capacity(values.len());
        for value in values {
            let block = unsigned(value, &format!("{context}.block_ids entry"))?;
            if block > i64::MAX as u64 {
                return Err(format!(
                    "lossless block id {block} exceeds the replay index range"
                ));
            }
            if block_ids.last().is_some_and(|previous| *previous >= block) {
                return Err(format!(
                    "{context}.block_ids must be strictly launch-index sorted"
                ));
            }
            if !roster.insert(block) {
                return Err(format!(
                    "block {block} occurs more than once in per_sm_slot_map"
                ));
            }
            block_ids.push(block);
        }
        if roster_by_sm.insert(sm, block_ids).is_some() {
            return Err(format!("duplicate per_sm_slot_map entry for sm {sm}"));
        }
    }
    if resident_by_sm.keys().ne(roster_by_sm.keys()) {
        return Err("resident-block SM keys must exactly equal per_sm_slot_map keys".to_string());
    }
    let grid_total = u64::from(original_grid.0)
        .checked_mul(u64::from(original_grid.1))
        .and_then(|value| value.checked_mul(u64::from(original_grid.2)))
        .ok_or_else(|| "original grid size overflows u64".to_string())?;
    let roster_len = u64::try_from(roster.len())
        .map_err(|_| "lossless block roster cardinality exceeds u64".to_string())?;
    if roster_len != grid_total || roster.iter().copied().ne(0..grid_total) {
        return Err("per_sm_slot_map must exhaust launch indices 0..original_grid".to_string());
    }
    let reconstructed_n_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            u64::try_from(blocks.len())
                .map(|count| count.div_ceil(resident_by_sm[sm]))
                .map_err(|_| "per-SM block roster cardinality exceeds u64".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    if reconstructed_n_waves != declared_n_waves || gate_wave_count != declared_n_waves {
        return Err("wave count disagrees with reconstructed per-SM cohorts".to_string());
    }
    let complete_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            u64::try_from(blocks.len())
                .map(|count| count / resident_by_sm[sm])
                .map_err(|_| "per-SM block roster cardinality exceeds u64".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .unwrap_or(0);
    let complete_cutoff = (complete_waves / segment_len) * segment_len;
    let residual_tail = reconstructed_n_waves > complete_cutoff;
    let tail_wave_count = if residual_tail {
        reconstructed_n_waves - complete_cutoff
    } else {
        segment_len.min(complete_cutoff)
    };
    let expected_tail_measured = tail_wave_count.saturating_sub(ghost_depth);
    let segment_count = (complete_cutoff / segment_len)
        .checked_add(u64::from(residual_tail))
        .ok_or_else(|| "lossless segment count overflows u64".to_string())?;
    let ghost_waves = ghost_depth
        .checked_mul(segment_count)
        .ok_or_else(|| "lossless ghost-wave accounting overflows u64".to_string())?;
    let expected_measured = reconstructed_n_waves.saturating_sub(ghost_waves);
    if gate_tail_measured != expected_tail_measured || gate_measured != expected_measured {
        return Err("measured-wave gates disagree with reconstructed ghost accounting".to_string());
    }

    let sorted_blocks: Vec<u64> = roster.into_iter().collect();
    let block_json = serde_json::to_string(&sorted_blocks)
        .map_err(|error| format!("could not canonicalize lossless block ids: {error}"))?;
    if selected_block_ids_sha256 != sha256_hex(block_json.as_bytes()) {
        return Err("selected_block_ids_sha256 does not match the full launch roster".to_string());
    }

    let records = object(
        required(root, "records", "lossless compressed sidecar")?,
        "records",
    )?;
    let expected_records = unsigned(
        required(records, "expected_full", "records")?,
        "records.expected_full",
    )?;
    let captured_records = unsigned(
        required(records, "captured", "records")?,
        "records.captured",
    )?;
    if captured_records != expected_records {
        return Err("lossless records.captured must equal records.expected_full".to_string());
    }
    Ok((
        raw_bytes,
        ValidatedPerSmSlotMap::retain(&roster_by_sm, grid_total)?,
    ))
}

#[cfg(feature = "campaign")]
fn validate_subset_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_segments: &BTreeSet<u64>,
    certainty_segments: &BTreeSet<u64>,
    segment_blocks: &BTreeMap<u64, BTreeSet<u64>>,
) -> Result<ValidatedPerSmSlotMap, String> {
    let reason = string(required(root, "reason", "sampled subset")?, "reason")?;
    if ![
        "byte_budget",
        "walltime_budget",
        "gate_forced",
        "not_breaching",
    ]
    .contains(&reason)
    {
        return Err("reason is not a schema-v2 value".to_string());
    }
    if original_block != captured_block {
        return Err(
            "shape.captured_block must equal shape.original_block for a subset".to_string(),
        );
    }
    let shape = object(required(root, "shape", "sampled subset")?, "shape")?;
    if !shape
        .get("reduced_shape_variant")
        .is_some_and(serde_json::Value::is_null)
    {
        return Err("sampled subset requires reduced_shape_variant=null".to_string());
    }

    let gates = object(required(root, "gates", "sampled subset")?, "gates")?;
    const GATE_FIELDS: &[&str] = &[
        "coop",
        "wave_count",
        "n_measured_waves",
        "tail_measured_waves",
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
        "n_phases",
        "skew_heavy_block_count",
        "certainty_bytes",
        "max_trace_bytes",
        "certainty_budget_exceeded",
        "verdict",
    ];
    exact_keys(gates, GATE_FIELDS, GATE_FIELDS, "gates")?;
    if boolean(required(gates, "coop", "gates")?, "gates.coop")? {
        return Err("a cooperative launch cannot sanction a sampled subset".to_string());
    }
    let gate_wave_count = unsigned(required(gates, "wave_count", "gates")?, "gates.wave_count")?;
    let gate_measured = unsigned(
        required(gates, "n_measured_waves", "gates")?,
        "gates.n_measured_waves",
    )?;
    let gate_tail_measured = unsigned(
        required(gates, "tail_measured_waves", "gates")?,
        "gates.tail_measured_waves",
    )?;
    for field in [
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
    ] {
        nonnegative_number(required(gates, field, "gates")?, &format!("gates.{field}"))?;
    }
    positive(required(gates, "n_phases", "gates")?, "gates.n_phases")?;
    let skew_heavy_count = unsigned(
        required(gates, "skew_heavy_block_count", "gates")?,
        "gates.skew_heavy_block_count",
    )?;
    let gate_certainty_bytes = unsigned(
        required(gates, "certainty_bytes", "gates")?,
        "gates.certainty_bytes",
    )?;
    let gate_max_bytes = positive(
        required(gates, "max_trace_bytes", "gates")?,
        "gates.max_trace_bytes",
    )?;
    if boolean(
        required(gates, "certainty_budget_exceeded", "gates")?,
        "gates.certainty_budget_exceeded",
    )? {
        return Err("a certainty-budget-exceeded gate cannot sanction a subset".to_string());
    }
    if string(required(gates, "verdict", "gates")?, "gates.verdict")? != "subset_admissible" {
        return Err("a sampled subset requires gates.verdict=subset_admissible".to_string());
    }

    let tracer = object(required(root, "tracer", "sampled subset")?, "tracer")?;
    exact_keys(
        tracer,
        &["version", "features", "stats_pass_version"],
        &["version", "features", "stats_pass_version"],
        "tracer",
    )?;
    if unsigned(required(tracer, "version", "tracer")?, "tracer.version")? != 5
        || unsigned(
            required(tracer, "stats_pass_version", "tracer")?,
            "tracer.stats_pass_version",
        )? != 2
    {
        return Err("sampled subset requires tracer v5 and stats pass v2".to_string());
    }
    let features = object(required(tracer, "features", "tracer")?, "tracer.features")?;
    exact_keys(
        features,
        &["atomic_addr", "ldgsts_addr"],
        &["atomic_addr", "ldgsts_addr"],
        "tracer.features",
    )?;
    if !boolean(
        required(features, "atomic_addr", "tracer.features")?,
        "tracer.features.atomic_addr",
    )? || !boolean(
        required(features, "ldgsts_addr", "tracer.features")?,
        "tracer.features.ldgsts_addr",
    )? {
        return Err("sampled subset requires atomic_addr and ldgsts_addr features".to_string());
    }

    let rng = object(required(root, "rng", "sampled subset")?, "rng")?;
    exact_keys(rng, &["algo", "seed"], &["algo", "seed"], "rng")?;
    if string(required(rng, "algo", "rng")?, "rng.algo")? != "pcg64"
        || required(rng, "seed", "rng")?.as_i64().is_none()
    {
        return Err("sampled subset requires pcg64 with an integer seed".to_string());
    }
    if string(
        required(root, "label_pairing", "sampled subset")?,
        "label_pairing",
    )? != "full_shape_label"
    {
        return Err("sampled subset requires full_shape_label pairing".to_string());
    }

    let wave = object(
        required(root, "wave_model", "sampled subset")?,
        "wave_model",
    )?;
    const WAVE_FIELDS: &[&str] = &[
        "reconstruction",
        "per_sm_resident_blocks",
        "n_waves",
        "segment_len",
        "ghost_depth",
    ];
    exact_keys(wave, WAVE_FIELDS, WAVE_FIELDS, "wave_model")?;
    if string(
        required(wave, "reconstruction", "wave_model")?,
        "wave_model.reconstruction",
    )? != "launch_index"
    {
        return Err("wave_model.reconstruction must be launch_index".to_string());
    }
    let declared_n_waves = unsigned(
        required(wave, "n_waves", "wave_model")?,
        "wave_model.n_waves",
    )?;
    let segment_len = positive(
        required(wave, "segment_len", "wave_model")?,
        "wave_model.segment_len",
    )?;
    let ghost_depth = unsigned(
        required(wave, "ghost_depth", "wave_model")?,
        "wave_model.ghost_depth",
    )?;
    if segment_len != 3 || ghost_depth != 1 {
        return Err(
            "current schema-v2 producer requires segment_len=3 and ghost_depth=1".to_string(),
        );
    }

    let residency_values = required(wave, "per_sm_resident_blocks", "wave_model")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "wave_model.per_sm_resident_blocks must be a nonempty array".to_string())?;
    let mut resident_by_sm = BTreeMap::new();
    for (index, value) in residency_values.iter().enumerate() {
        let context = format!("wave_model.per_sm_resident_blocks[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "resident_blocks"],
            &["sm_id", "resident_blocks"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let resident = positive(
            required(row, "resident_blocks", &context)?,
            &format!("{context}.resident_blocks"),
        )?;
        if resident_by_sm.insert(sm, resident).is_some() {
            return Err(format!("duplicate resident-block entry for sm {sm}"));
        }
    }

    let slot_values = required(root, "per_sm_slot_map", "sampled subset")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "per_sm_slot_map must be a nonempty array".to_string())?;
    let mut roster_by_sm: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut roster = BTreeSet::new();
    for (index, value) in slot_values.iter().enumerate() {
        let context = format!("per_sm_slot_map[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "block_ids"],
            &["sm_id", "block_ids"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let values = required(row, "block_ids", &context)?
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("{context}.block_ids must be a nonempty array"))?;
        let mut block_ids = Vec::with_capacity(values.len());
        for value in values {
            let block = unsigned(value, &format!("{context}.block_ids entry"))?;
            if block > i64::MAX as u64 {
                return Err(format!(
                    "sampled block id {block} exceeds the replay index range"
                ));
            }
            if block_ids.last().is_some_and(|previous| *previous >= block) {
                return Err(format!(
                    "{context}.block_ids must be strictly launch-index sorted"
                ));
            }
            if !roster.insert(block) {
                return Err(format!(
                    "block {block} occurs more than once in per_sm_slot_map"
                ));
            }
            block_ids.push(block);
        }
        if roster_by_sm.insert(sm, block_ids).is_some() {
            return Err(format!("duplicate per_sm_slot_map entry for sm {sm}"));
        }
    }
    if resident_by_sm.keys().ne(roster_by_sm.keys()) {
        return Err("resident-block SM keys must exactly equal per_sm_slot_map keys".to_string());
    }
    let grid_total = u64::from(original_grid.0)
        .checked_mul(u64::from(original_grid.1))
        .and_then(|value| value.checked_mul(u64::from(original_grid.2)))
        .ok_or_else(|| "original grid size overflows u64".to_string())?;
    if roster.len() as u64 != grid_total || roster.iter().copied().ne(0..grid_total) {
        return Err("per_sm_slot_map must exhaust launch indices 0..original_grid".to_string());
    }

    let reconstructed_n_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            let resident = resident_by_sm[sm];
            (blocks.len() as u64).div_ceil(resident)
        })
        .max()
        .unwrap_or(0);
    if reconstructed_n_waves != declared_n_waves || gate_wave_count != declared_n_waves {
        return Err("wave count disagrees with reconstructed per-SM cohorts".to_string());
    }
    let complete_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| blocks.len() as u64 / resident_by_sm[sm])
        .min()
        .unwrap_or(0);
    let complete_cutoff = (complete_waves / segment_len) * segment_len;
    let residual_tail = reconstructed_n_waves > complete_cutoff;
    let tail_wave_count = if residual_tail {
        reconstructed_n_waves - complete_cutoff
    } else {
        segment_len.min(complete_cutoff)
    };
    // Deliberate semantic clamp: the Python producer defines ghost removal as
    // max(0, waves - ghosts), so the saturating subtractions here are contract
    // arithmetic rather than overflow recovery. The ghost product itself must
    // still be checked.
    let expected_tail_measured = tail_wave_count.saturating_sub(ghost_depth);
    let segment_count = complete_cutoff / segment_len + u64::from(residual_tail);
    let ghost_waves = ghost_depth
        .checked_mul(segment_count)
        .ok_or_else(|| "ghost-wave accounting overflows u64".to_string())?;
    let expected_measured = reconstructed_n_waves.saturating_sub(ghost_waves);
    if gate_tail_measured != expected_tail_measured || gate_measured != expected_measured {
        return Err("measured-wave gates disagree with reconstructed ghost accounting".to_string());
    }

    let mut expected_segments: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (sm, blocks) in &roster_by_sm {
        let resident = resident_by_sm[sm];
        for (sequence, block) in blocks.iter().copied().enumerate() {
            let wave_index = sequence as u64 / resident;
            let segment_id = if wave_index < complete_cutoff {
                wave_index / segment_len
            } else {
                complete_cutoff / segment_len
            };
            expected_segments
                .entry(segment_id)
                .or_default()
                .insert(block);
        }
    }
    for (segment_id, blocks) in segment_blocks {
        if expected_segments.get(segment_id) != Some(blocks) {
            return Err(format!(
                "segment {segment_id} is not a complete reconstructed wave cohort"
            ));
        }
    }
    let cold_segment = *expected_segments
        .keys()
        .next()
        .ok_or_else(|| "wave reconstruction produced no segments".to_string())?;
    let candidate_tail = complete_cutoff / segment_len;
    let tail_segment = if expected_segments.contains_key(&candidate_tail) {
        candidate_tail
    } else {
        *expected_segments
            .keys()
            .next_back()
            .expect("nonempty above")
    };
    if !certainty_segments.contains(&cold_segment) || !certainty_segments.contains(&tail_segment) {
        return Err(
            "certainty_segments must contain reconstructed cold and tail cohorts".to_string(),
        );
    }

    let force_blocks = parse_unique_ids(
        required(root, "force_include_blocks", "sampled subset")?,
        "force_include_blocks",
        true,
    )?;
    if force_blocks.len() as u64 != skew_heavy_count {
        return Err("skew_heavy_block_count disagrees with force_include_blocks".to_string());
    }
    let certainty_blocks: BTreeSet<u64> = certainty_segments
        .iter()
        .flat_map(|segment| segment_blocks[segment].iter().copied())
        .collect();
    if !force_blocks.is_subset(&certainty_blocks) {
        return Err("force_include_blocks must be contained in certainty segments".to_string());
    }

    let strata_values = required(root, "strata", "sampled subset")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "strata must be a nonempty array".to_string())?;
    let mut stratum_ids = BTreeSet::new();
    let mut stratum_keys = BTreeSet::new();
    let mut sampled_ordinary = BTreeSet::new();
    const STRATUM_FIELDS: &[&str] = &[
        "id",
        "phase_id",
        "launch_tercile",
        "segment_ids",
        "N_h",
        "n_h",
        "pi",
        "sector_stddev",
        "neyman_weight",
    ];
    for (index, value) in strata_values.iter().enumerate() {
        let context = format!("strata[{index}]");
        let stratum = object(value, &context)?;
        exact_keys(stratum, STRATUM_FIELDS, STRATUM_FIELDS, &context)?;
        let id = unsigned(required(stratum, "id", &context)?, &format!("{context}.id"))?;
        let phase = unsigned(
            required(stratum, "phase_id", &context)?,
            &format!("{context}.phase_id"),
        )?;
        let tercile = unsigned(
            required(stratum, "launch_tercile", &context)?,
            &format!("{context}.launch_tercile"),
        )?;
        if tercile > 2 {
            return Err(format!("{context} id does not encode phase/launch tercile"));
        }
        let encoded_id = phase
            .checked_mul(3)
            .and_then(|value| value.checked_add(tercile))
            .ok_or_else(|| format!("{context} phase/launch tercile encoding overflows u64"))?;
        if id != encoded_id {
            return Err(format!("{context} id does not encode phase/launch tercile"));
        }
        if !stratum_ids.insert(id) || !stratum_keys.insert((phase, tercile)) {
            return Err(format!("{context} duplicates a stratum id or key"));
        }
        let ids = parse_unique_ids(
            required(stratum, "segment_ids", &context)?,
            &format!("{context}.segment_ids"),
            false,
        )?;
        if ids.len() < 2 {
            return Err(format!(
                "{context}.segment_ids must contain at least two ids"
            ));
        }
        let population = positive(
            required(stratum, "N_h", &context)?,
            &format!("{context}.N_h"),
        )?;
        let sample = positive(
            required(stratum, "n_h", &context)?,
            &format!("{context}.n_h"),
        )?;
        let ids_len = u64::try_from(ids.len())
            .map_err(|_| format!("{context}.segment_ids cardinality exceeds u64"))?;
        if sample < 2 || sample > population || !sampling_cardinalities_match(sample, ids_len) {
            return Err(format!("{context} has inconsistent N_h/n_h/segment_ids"));
        }
        let pi = nonnegative_number(required(stratum, "pi", &context)?, &format!("{context}.pi"))?;
        let stddev = nonnegative_number(
            required(stratum, "sector_stddev", &context)?,
            &format!("{context}.sector_stddev"),
        )?;
        let weight = nonnegative_number(
            required(stratum, "neyman_weight", &context)?,
            &format!("{context}.neyman_weight"),
        )?;
        if pi <= 0.0 || pi > 1.0 {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        let population_f64 = sampling_u64_as_f64(population);
        let expected_pi = sampling_u64_as_f64(sample) / population_f64;
        // An absolute floor would admit arbitrarily wrong probabilities as N_h
        // grows. Keep only a relative allowance for cross-runtime float spelling.
        if !relative_close_enough(pi, expected_pi) {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        let expected_weight = population_f64 * stddev;
        if !expected_weight.is_finite() {
            return Err(format!("{context} derived neyman_weight is not finite"));
        }
        if !close_enough(weight, expected_weight) {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        if ids.iter().any(|id| !sampled_ordinary.insert(*id)) {
            return Err("ordinary segment occurs in multiple strata".to_string());
        }
    }
    if !sampled_ordinary.is_disjoint(certainty_segments) {
        return Err("certainty segments cannot also be sampled ordinary segments".to_string());
    }
    let selected_partition: BTreeSet<u64> = sampled_ordinary
        .union(certainty_segments)
        .copied()
        .collect();
    if &selected_partition != selected_segments {
        return Err(
            "strata sampled ids plus certainty ids must partition selected ids".to_string(),
        );
    }

    let budget = object(
        required(root, "selection_budget", "sampled subset")?,
        "selection_budget",
    )?;
    const BUDGET_FIELDS: &[&str] = &[
        "max_trace_bytes",
        "basis",
        "certainty_fraction_numerator",
        "certainty_fraction_denominator",
        "certainty_bytes",
    ];
    exact_keys(budget, BUDGET_FIELDS, BUDGET_FIELDS, "selection_budget")?;
    let max_bytes = positive(
        required(budget, "max_trace_bytes", "selection_budget")?,
        "selection_budget.max_trace_bytes",
    )?;
    let certainty_bytes = unsigned(
        required(budget, "certainty_bytes", "selection_budget")?,
        "selection_budget.certainty_bytes",
    )?;
    if string(
        required(budget, "basis", "selection_budget")?,
        "selection_budget.basis",
    )? != "raw_threadblock_pb"
        || unsigned(
            required(budget, "certainty_fraction_numerator", "selection_budget")?,
            "selection_budget.certainty_fraction_numerator",
        )? != 2
        || unsigned(
            required(budget, "certainty_fraction_denominator", "selection_budget")?,
            "selection_budget.certainty_fraction_denominator",
        )? != 5
        || max_bytes != gate_max_bytes
        || certainty_bytes != gate_certainty_bytes
        || 5_u128 * u128::from(certainty_bytes) > 2_u128 * u128::from(max_bytes)
    {
        return Err("selection budget disagrees with the exact raw-byte 2/5 contract".to_string());
    }

    let auxiliary = object(
        required(root, "selection_auxiliary", "sampled subset")?,
        "selection_auxiliary",
    )?;
    const AUX_FIELDS: &[&str] = &["name", "sector_bytes", "scope", "expected_full", "selected"];
    exact_keys(auxiliary, AUX_FIELDS, AUX_FIELDS, "selection_auxiliary")?;
    let expected_aux = unsigned(
        required(auxiliary, "expected_full", "selection_auxiliary")?,
        "selection_auxiliary.expected_full",
    )?;
    let selected_aux = unsigned(
        required(auxiliary, "selected", "selection_auxiliary")?,
        "selection_auxiliary.selected",
    )?;
    if string(
        required(auxiliary, "name", "selection_auxiliary")?,
        "selection_auxiliary.name",
    )? != "l2_sector_transactions_32b"
        || unsigned(
            required(auxiliary, "sector_bytes", "selection_auxiliary")?,
            "selection_auxiliary.sector_bytes",
        )? != 32
        || string(
            required(auxiliary, "scope", "selection_auxiliary")?,
            "selection_auxiliary.scope",
        )? != "pass2_all_blocks"
        || selected_aux > expected_aux
    {
        return Err("selection auxiliary envelope is inconsistent".to_string());
    }

    let bytes = object(required(root, "bytes", "sampled subset")?, "bytes")?;
    const BYTE_FIELDS: &[&str] = &["raw", "compressed", "codec", "level", "long", "dict_id"];
    exact_keys(bytes, BYTE_FIELDS, BYTE_FIELDS, "bytes")?;
    let raw_bytes = unsigned(required(bytes, "raw", "bytes")?, "bytes.raw")?;
    unsigned(required(bytes, "compressed", "bytes")?, "bytes.compressed")?;
    if raw_bytes > max_bytes
        || string(required(bytes, "codec", "bytes")?, "bytes.codec")? != "zstd"
        || required(bytes, "level", "bytes")?.as_i64() != Some(19)
        || required(bytes, "long", "bytes")?.as_i64() != Some(27)
        || !required(bytes, "dict_id", "bytes")?.is_null()
    {
        return Err("bytes envelope is not the current bounded zstd contract".to_string());
    }

    let estimator = object(required(root, "estimator", "sampled subset")?, "estimator")?;
    exact_keys(
        estimator,
        &["mode", "targets", "not_recoverable", "convergence"],
        &[
            "mode",
            "form",
            "ci",
            "targets",
            "not_recoverable",
            "convergence",
        ],
        "estimator",
    )?;
    if string(required(estimator, "mode", "estimator")?, "estimator.mode")?
        != "weighted_full_shape_estimate"
    {
        return Err("subset estimator mode must be weighted_full_shape_estimate".to_string());
    }
    let targets = required(estimator, "targets", "estimator")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "estimator.targets must be a nonempty array".to_string())?;
    for target in targets {
        string(target, "estimator.targets entry")?;
    }
    string_array(
        required(estimator, "not_recoverable", "estimator")?,
        "estimator.not_recoverable",
    )?;
    if let Some(form) = estimator.get("form") {
        string(form, "estimator.form")?;
    }
    if let Some(ci_value) = estimator.get("ci") {
        let ci = object(ci_value, "estimator.ci")?;
        exact_keys(ci, &[], &["method", "B", "alpha"], "estimator.ci")?;
        if let Some(method) = ci.get("method") {
            string(method, "estimator.ci.method")?;
        }
        if let Some(samples) = ci.get("B") {
            positive(samples, "estimator.ci.B")?;
        }
        if let Some(alpha) = ci.get("alpha") {
            nonnegative_number(alpha, "estimator.ci.alpha")?;
        }
    }
    let convergence = object(
        required(estimator, "convergence", "estimator")?,
        "estimator.convergence",
    )?;
    exact_keys(
        convergence,
        &["status"],
        &["status", "per_target"],
        "estimator.convergence",
    )?;
    let status = string(
        required(convergence, "status", "estimator.convergence")?,
        "estimator.convergence.status",
    )?;
    if !["converged", "budget_limited_wide_ci", "unavailable"].contains(&status) {
        return Err("estimator convergence status is invalid".to_string());
    }
    if convergence
        .get("per_target")
        .is_some_and(|value| !value.is_object())
    {
        return Err("estimator.convergence.per_target must be an object".to_string());
    }

    let records = object(required(root, "records", "sampled subset")?, "records")?;
    let expected_records = unsigned(
        required(records, "expected_full", "records")?,
        "records.expected_full",
    )?;
    let captured_records = unsigned(
        required(records, "captured", "records")?,
        "records.captured",
    )?;
    if captured_records > expected_records {
        return Err("records.captured cannot exceed records.expected_full".to_string());
    }
    ValidatedPerSmSlotMap::retain(&roster_by_sm, grid_total)
}

#[cfg(feature = "campaign")]
fn validate_sampling_payload_snapshot(
    root: &Path,
    sidecar: &TraceSamplingSidecar,
    snapshot: &VerifiedSamplingPayload,
) -> Result<(), TraceError> {
    let sidecar_path = root.join("trace_sampling.json");
    let malformed = |message: String| TraceError::SamplingSidecarMalformed {
        path: sidecar_path.clone(),
        message,
    };
    let mut actual_entries = BTreeMap::new();
    for (path, raw) in &snapshot.by_path {
        let relative = path
            .strip_prefix(root)
            .ok()
            .and_then(Path::to_str)
            .ok_or_else(|| {
                malformed(format!(
                    "snapshotted payload path is not canonical under the trace root: {}",
                    path.display()
                ))
            })?;
        actual_entries.insert(
            relative,
            (
                u64::try_from(raw.len())
                    .map_err(|_| malformed(format!("payload {relative:?} length exceeds u64")))?,
                Sha256Digest::of(raw),
            ),
        );
    }
    let actual_paths: BTreeSet<_> = actual_entries.keys().copied().collect();
    let declared_paths: BTreeSet<_> = sidecar
        .payload_file_manifest
        .keys()
        .map(String::as_str)
        .collect();
    if actual_paths != declared_paths {
        let missing: Vec<_> = declared_paths.difference(&actual_paths).take(4).collect();
        let unexpected: Vec<_> = actual_paths.difference(&declared_paths).take(4).collect();
        return Err(malformed(format!(
            "per_block_raw_sha256_manifest path set disagrees with threadblock payload: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    for (relative, (size, digest)) in actual_entries {
        let declared = &sidecar.payload_file_manifest[relative];
        if size != declared.size_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest size for {relative:?} is {}, actual {size}",
                declared.size_bytes
            )));
        }
        if digest.to_string() != declared.sha256 {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest hash for {relative:?} does not match raw payload (actual {digest})"
            )));
        }
    }
    if snapshot.identity.raw_bytes != sidecar.raw_payload_bytes {
        return Err(malformed(format!(
            "bytes.raw {} does not match threadblock payload bytes {}",
            sidecar.raw_payload_bytes, snapshot.identity.raw_bytes
        )));
    }
    if snapshot.identity.sorted_path_stream_sha256.to_string()
        != sidecar.decompressed_payload_sha256
    {
        return Err(malformed(format!(
            "decompressed_payload_sha256 does not match sorted threadblock bytes (actual {})",
            snapshot.identity.sorted_path_stream_sha256
        )));
    }
    Ok(())
}

#[cfg(feature = "campaign")]
impl TraceSamplingSidecar {
    /// Parse `<root>/trace_sampling.json` if present (`Ok(None)` if absent).
    /// This internal stage validates and retains the JSON envelope; callers only
    /// receive it through [`TraceDir::sampling_sidecar`] after `TraceDir::open`
    /// has also bound the decompressed threadblock bytes.
    #[cfg(test)]
    #[cfg(feature = "campaign")]
    fn load(root: &Path) -> Result<Option<Self>, TraceError> {
        let path = root.join("trace_sampling.json");
        let reader = AnchoredDir::open(root)?;
        let raw = match reader
            .read_optional_bounded_regular(
                Path::new("trace_sampling.json"),
                TRACE_SAMPLING_JSON_MAX_BYTES,
            )
            .map_err(|source| map_snapshot_read_error("trace_sampling_json", source))?
        {
            Some(file) => file.into_raw(),
            None => return Ok(None),
        };
        Self::parse(&path, &raw).map(Some)
    }

    fn parse(path: &Path, raw: &[u8]) -> Result<Self, TraceError> {
        Self::parse_with_limits(
            path,
            raw,
            TRACE_SAMPLING_JSON_MAX_BYTES,
            TRACE_SAMPLING_JSON_MAX_NODES,
        )
    }

    fn parse_with_limits(
        path: &Path,
        raw: &[u8],
        maximum_bytes: u64,
        maximum_nodes: u64,
    ) -> Result<Self, TraceError> {
        sidecar_byte_limit(path, "trace_sampling_json", raw, maximum_bytes)?;
        match preflight_json_shape(raw, maximum_nodes, None) {
            Ok(_) => {}
            Err(JsonPreflightError::Malformed(message)) => {
                return Err(TraceError::SamplingSidecarMalformed {
                    path: path.to_path_buf(),
                    message,
                });
            }
            Err(JsonPreflightError::ResourceLimit(error)) => {
                return Err(map_json_preflight_limit(path, "trace_sampling_json", error));
            }
        }
        let parsed: serde_json::Value =
            serde_json::from_slice(raw).map_err(|e| TraceError::SamplingSidecarMalformed {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        let malformed = |message: String| TraceError::SamplingSidecarMalformed {
            path: path.to_path_buf(),
            message,
        };
        macro_rules! get {
            ($object:expr, $field:expr, $context:expr) => {
                required($object, $field, $context).map_err(&malformed)?
            };
        }
        let root = object(&parsed, "trace_sampling.json root").map_err(&malformed)?;
        const TOP_LEVEL_FIELDS: &[&str] = &[
            "schema_version",
            "cell_id",
            "kernel",
            "method",
            "reason",
            "shape",
            "gates",
            "wave_model",
            "segments",
            "selected_segment_ids",
            "certainty_segments",
            "force_include_blocks",
            "per_sm_slot_map",
            "strata",
            "selection_reconciliation",
            "selection_budget",
            "selection_auxiliary",
            "rng",
            "estimator",
            "label_pairing",
            "tracer",
            "bytes",
            "records",
            "hashes",
        ];
        const BASE_REQUIRED: &[&str] = &[
            "schema_version",
            "cell_id",
            "kernel",
            "method",
            "reason",
            "shape",
            "gates",
            "records",
            "tracer",
            "hashes",
        ];
        exact_keys(
            root,
            BASE_REQUIRED,
            TOP_LEVEL_FIELDS,
            "trace_sampling.json root",
        )
        .map_err(&malformed)?;
        if get!(root, "schema_version", "trace_sampling.json root").as_u64() != Some(2) {
            return Err(malformed("schema_version must be 2".to_string()));
        }
        let method = get!(root, "method", "trace_sampling.json root")
            .as_str()
            .ok_or_else(|| malformed("method must be a string".to_string()))?;

        let cell_id = get!(root, "cell_id", "trace_sampling.json root")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("cell_id must be a non-empty string".to_string()))?
            .to_string();
        let kernel = object(get!(root, "kernel", "trace_sampling.json root"), "kernel")
            .map_err(&malformed)?;
        exact_keys(
            kernel,
            &["mangled_name", "args_hash"],
            &["mangled_name", "args_hash"],
            "kernel",
        )
        .map_err(&malformed)?;
        let kernel_mangled_name = get!(kernel, "mangled_name", "kernel")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("kernel.mangled_name must be nonempty".to_string()))?
            .to_string();
        let kernel_args_hash = get!(kernel, "args_hash", "kernel")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("kernel.args_hash must be nonempty".to_string()))?
            .to_string();
        get!(root, "reason", "trace_sampling.json root")
            .as_str()
            .ok_or_else(|| malformed("reason must be a string".to_string()))?;
        object(get!(root, "gates", "trace_sampling.json root"), "gates").map_err(&malformed)?;
        object(get!(root, "tracer", "trace_sampling.json root"), "tracer").map_err(&malformed)?;

        let shape =
            object(get!(root, "shape", "trace_sampling.json root"), "shape").map_err(&malformed)?;
        exact_keys(
            shape,
            &[
                "original_grid",
                "original_block",
                "captured_grid",
                "captured_block",
            ],
            &[
                "original_grid",
                "original_block",
                "captured_grid",
                "captured_block",
                "reduced_shape_variant",
            ],
            "shape",
        )
        .map_err(&malformed)?;
        let original_grid =
            parse_dim3(get!(shape, "original_grid", "shape"), "shape.original_grid")
                .map_err(&malformed)?;
        let captured_grid =
            parse_dim3(get!(shape, "captured_grid", "shape"), "shape.captured_grid")
                .map_err(&malformed)?;
        let original_block = parse_dim3(
            get!(shape, "original_block", "shape"),
            "shape.original_block",
        )
        .map_err(&malformed)?;
        let captured_block = parse_dim3(
            get!(shape, "captured_block", "shape"),
            "shape.captured_block",
        )
        .map_err(&malformed)?;

        let records = object(get!(root, "records", "trace_sampling.json root"), "records")
            .map_err(&malformed)?;
        exact_keys(
            records,
            &["expected_full", "captured"],
            &["expected_full", "captured"],
            "records",
        )
        .map_err(&malformed)?;
        let expected_full_records = get!(records, "expected_full", "records")
            .as_u64()
            .ok_or_else(|| malformed("records.expected_full must be a u64".to_string()))?;
        let captured_records = get!(records, "captured", "records")
            .as_u64()
            .ok_or_else(|| malformed("records.captured must be a u64".to_string()))?;

        let hashes = object(get!(root, "hashes", "trace_sampling.json root"), "hashes")
            .map_err(&malformed)?;
        exact_keys(
            hashes,
            &[
                "selected_block_ids_sha256",
                "decompressed_payload_sha256",
                "per_block_raw_sha256_manifest",
                "sidecar_sha256",
            ],
            &[
                "selected_block_ids_sha256",
                "decompressed_payload_sha256",
                "per_block_raw_sha256_manifest",
                "sidecar_sha256",
            ],
            "hashes",
        )
        .map_err(&malformed)?;
        for field in [
            "selected_block_ids_sha256",
            "decompressed_payload_sha256",
            "sidecar_sha256",
        ] {
            let value = get!(hashes, field, "hashes")
                .as_str()
                .ok_or_else(|| malformed(format!("hashes.{field} must be a string")))?;
            if !is_lower_hex_sha256(value) {
                return Err(malformed(format!(
                    "hashes.{field} must be lowercase SHA-256"
                )));
            }
        }
        let selected_block_ids_sha256 = get!(hashes, "selected_block_ids_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let decompressed_payload_sha256 = get!(hashes, "decompressed_payload_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let payload_file_manifest = parse_payload_file_manifest(
            get!(hashes, "per_block_raw_sha256_manifest", "hashes")
                .as_str()
                .ok_or_else(|| {
                    malformed("hashes.per_block_raw_sha256_manifest must be a string".to_string())
                })?,
        )
        .map_err(&malformed)?;
        let sidecar_sha256 = get!(hashes, "sidecar_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let expected_seal = canonical_sidecar_sha256(&parsed).map_err(&malformed)?;
        if sidecar_sha256 != expected_seal {
            return Err(malformed(
                "hashes.sidecar_sha256 does not match Python-canonical sidecar".to_string(),
            ));
        }

        let (method, raw_payload_bytes, per_sm_slot_map) = match method {
            "wave_segment_stratified_subset" => {
                const SUBSET_REQUIRED: &[&str] = &[
                    "wave_model",
                    "segments",
                    "selected_segment_ids",
                    "certainty_segments",
                    "force_include_blocks",
                    "per_sm_slot_map",
                    "strata",
                    "selection_reconciliation",
                    "selection_budget",
                    "selection_auxiliary",
                    "rng",
                    "estimator",
                    "label_pairing",
                    "bytes",
                ];
                exact_keys(root, SUBSET_REQUIRED, TOP_LEVEL_FIELDS, "sampled subset")
                    .map_err(&malformed)?;
                for field in [
                    "wave_model",
                    "selection_budget",
                    "selection_auxiliary",
                    "rng",
                    "estimator",
                    "bytes",
                ] {
                    object(get!(root, field, "sampled subset"), field).map_err(&malformed)?;
                }
                for field in ["force_include_blocks", "per_sm_slot_map", "strata"] {
                    get!(root, field, "sampled subset")
                        .as_array()
                        .ok_or_else(|| malformed(format!("{field} must be an array")))?;
                }
                get!(root, "label_pairing", "sampled subset")
                    .as_str()
                    .ok_or_else(|| malformed("label_pairing must be a string".to_string()))?;
                if captured_grid != original_grid {
                    return Err(malformed(
                        "shape.captured_grid must equal shape.original_grid for a subset"
                            .to_string(),
                    ));
                }

                let reconciliation = object(
                    get!(root, "selection_reconciliation", "sampled subset"),
                    "selection_reconciliation",
                )
                .map_err(&malformed)?;
                exact_keys(
                    reconciliation,
                    &["status", "attempts", "max_attempts", "capture_stats_sha256"],
                    &["status", "attempts", "max_attempts", "capture_stats_sha256"],
                    "selection_reconciliation",
                )
                .map_err(&malformed)?;
                let status = get!(reconciliation, "status", "selection_reconciliation").as_str();
                let attempts =
                    get!(reconciliation, "attempts", "selection_reconciliation").as_u64();
                let max_attempts =
                    get!(reconciliation, "max_attempts", "selection_reconciliation").as_u64();
                let capture_hash = get!(
                    reconciliation,
                    "capture_stats_sha256",
                    "selection_reconciliation"
                )
                .as_str();
                if status != Some("fixed_point")
                    || attempts.is_none_or(|value| value == 0)
                    || max_attempts.is_none_or(|value| value == 0)
                    || attempts > max_attempts
                    || capture_hash.is_none_or(|value| !is_lower_hex_sha256(value))
                {
                    return Err(malformed(
                        "invalid fixed-point reconciliation envelope".to_string(),
                    ));
                }

                let selected_segments = parse_unique_ids(
                    get!(root, "selected_segment_ids", "sampled subset"),
                    "selected_segment_ids",
                    false,
                )
                .map_err(&malformed)?;
                let certainty_segments = parse_unique_ids(
                    get!(root, "certainty_segments", "sampled subset"),
                    "certainty_segments",
                    false,
                )
                .map_err(&malformed)?;
                if !certainty_segments.is_subset(&selected_segments) {
                    return Err(malformed(
                        "certainty_segments must be a subset of selected_segment_ids".to_string(),
                    ));
                }

                let segment_values = get!(root, "segments", "sampled subset")
                    .as_array()
                    .ok_or_else(|| malformed("segments must be an array".to_string()))?;
                let mut segment_ids = BTreeSet::new();
                let mut segment_blocks_by_id = BTreeMap::new();
                let mut blocks = BTreeSet::new();
                for (index, segment_value) in segment_values.iter().enumerate() {
                    let context = format!("segments[{index}]");
                    let segment = object(segment_value, &context).map_err(&malformed)?;
                    exact_keys(
                        segment,
                        &["id", "block_ids"],
                        &["id", "block_ids"],
                        &context,
                    )
                    .map_err(&malformed)?;
                    let segment_id = get!(segment, "id", &context).as_u64().ok_or_else(|| {
                        malformed(format!("{context}.id must be a nonnegative integer"))
                    })?;
                    if !segment_ids.insert(segment_id) {
                        return Err(malformed(format!(
                            "segments contains duplicate id {segment_id}"
                        )));
                    }
                    let blocks_for_segment = parse_unique_ids(
                        get!(segment, "block_ids", &context),
                        &format!("{context}.block_ids"),
                        false,
                    )
                    .map_err(&malformed)?;
                    for block_id in &blocks_for_segment {
                        if *block_id > i64::MAX as u64 {
                            return Err(malformed(format!(
                                "selected block id {block_id} exceeds the replay index range"
                            )));
                        }
                        if !blocks.insert(*block_id) {
                            return Err(malformed(format!(
                                "selected block id {block_id} occurs in multiple segments"
                            )));
                        }
                    }
                    segment_blocks_by_id.insert(segment_id, blocks_for_segment);
                }
                if segment_ids != selected_segments {
                    return Err(malformed(
                        "segments ids must exactly equal selected_segment_ids".to_string(),
                    ));
                }

                let per_sm_slot_map = validate_subset_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_block,
                    &selected_segments,
                    &certainty_segments,
                    &segment_blocks_by_id,
                )
                .map_err(&malformed)?;

                let sorted_blocks: Vec<u64> = blocks.iter().copied().collect();
                let block_json = serde_json::to_string(&sorted_blocks).map_err(|error| {
                    malformed(format!(
                        "could not canonicalize selected block ids: {error}"
                    ))
                })?;
                let expected_block_hash = sha256_hex(block_json.as_bytes());
                if selected_block_ids_sha256 != expected_block_hash {
                    return Err(malformed(
                        "selected_block_ids_sha256 does not match segment membership".to_string(),
                    ));
                }

                (
                    TraceSamplingMethod::WaveSegmentStratifiedSubset(WaveSegmentSubsetProvenance {
                        selected_block_ids: blocks.into_iter().map(|value| value as i64).collect(),
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    unsigned(
                        required(
                            object(get!(root, "bytes", "sampled subset"), "bytes")
                                .map_err(&malformed)?,
                            "raw",
                            "bytes",
                        )
                        .map_err(&malformed)?,
                        "bytes.raw",
                    )
                    .map_err(&malformed)?,
                    per_sm_slot_map,
                )
            }
            // A lossless-compression declaration is sealed and payload-bound,
            // but replay must still observe complete launch coverage itself.
            "lossless_compressed_full" => {
                let (raw, per_sm_slot_map) = validate_lossless_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_grid,
                    captured_block,
                    &selected_block_ids_sha256,
                )
                .map_err(&malformed)?;
                (
                    TraceSamplingMethod::LosslessCompressedFull(LosslessFullProvenance {
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    raw,
                    per_sm_slot_map,
                )
            }
            _ => {
                return Err(malformed(
                    "method must be a sampled subset or lossless compressed full".to_string(),
                ));
            }
        };
        let manifest_bytes = payload_file_manifest
            .values()
            .try_fold(0_u64, |total, entry| {
                total.checked_add(entry.size_bytes).ok_or_else(|| {
                    malformed("per_block_raw_sha256_manifest byte total overflows u64".to_string())
                })
            })?;
        if manifest_bytes != raw_payload_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest byte total {manifest_bytes} disagrees with bytes.raw {raw_payload_bytes}"
            )));
        }
        Ok(TraceSamplingSidecar {
            cell_id,
            kernel_mangled_name,
            kernel_args_hash,
            original_grid,
            original_block,
            captured_grid,
            captured_block,
            expected_full_records,
            captured_records,
            decompressed_payload_sha256,
            payload_file_manifest,
            raw_payload_bytes,
            per_sm_slot_map,
            method,
        })
    }
}

pub fn block_launch_index(
    kernel: &pb::Kernel,
    tb: &pb::Threadblock,
    path: &Path,
) -> Result<i64, TraceError> {
    let grid = kernel
        .grid_dim
        .as_ref()
        .filter(|g| g.x > 0 && g.y > 0 && g.z > 0)
        .ok_or(TraceError::GridDimMissing {
            kernel_id: kernel.id,
        })?;
    let (gx, gy, gz) = (i64::from(grid.x), i64::from(grid.y), i64::from(grid.z));
    let grid_plane = gx
        .checked_mul(gy)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "grid plane size",
        })?;
    let grid_total = grid_plane
        .checked_mul(gz)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "grid volume",
        })?;
    let Some(bid) = tb.block_id.as_ref() else {
        return Err(TraceError::BlockIdMissing {
            path: path.to_path_buf(),
        });
    };
    let (x, y, z) = (bid.x, bid.y, bid.z);
    // Reject invalid coordinates before signed launch-index arithmetic. A malformed
    // out-of-axis coordinate must remain a range error even when its flattened
    // value exceeds i64; i128 covers every i32-coordinate/i32-grid product.
    if x < 0 || y < 0 || z < 0 || i64::from(x) >= gx || i64::from(y) >= gy || i64::from(z) >= gz {
        let index =
            i128::from(x) + i128::from(y) * i128::from(gx) + i128::from(z) * i128::from(grid_plane);
        return Err(TraceError::BlockIdOutOfRange {
            path: path.to_path_buf(),
            x,
            y,
            z,
            index,
            grid: grid_total,
        });
    }
    let y_offset = i64::from(y)
        .checked_mul(gx)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "block y offset",
        })?;
    let z_offset =
        i64::from(z)
            .checked_mul(grid_plane)
            .ok_or(TraceError::GridArithmeticOverflow {
                kernel_id: kernel.id,
                operation: "block z offset",
            })?;
    let index = i64::from(x)
        .checked_add(y_offset)
        .and_then(|value| value.checked_add(z_offset))
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "flattened block index",
        })?;
    // The per-axis checks above prevent an invalid CTA from flattening into a
    // valid launch index. Retain the flattened range check as a fail-closed
    // invariant over the checked arithmetic.
    if index >= grid_total {
        return Err(TraceError::BlockIdOutOfRange {
            path: path.to_path_buf(),
            x,
            y,
            z,
            index: i128::from(index),
            grid: grid_total,
        });
    }
    Ok(index)
}

/// Read one per-threadblock `.pb` file.
fn decode_threadblock(raw: &[u8], path: &Path) -> Result<pb::Threadblock, TraceError> {
    wire_preflight::preflight_threadblock(raw)
        .map_err(|error| map_wire_preflight_error(path, "dynamic_trace.threadblock", error))?;
    pb::Threadblock::decode(raw).map_err(|source| TraceError::Decode {
        path: path.to_path_buf(),
        message_type: "dynamic_trace.threadblock",
        source,
    })
}

#[cfg(test)]
fn read_threadblock(path: &Path) -> Result<pb::Threadblock, TraceError> {
    let raw = fs::read(path).map_err(|source| TraceError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    decode_threadblock(&raw, path)
}

/// One wire address record decoded for the memory hierarchy: the
/// reconstructed 32-lane vector plus the per-lane access width in bytes.
/// Width resolution mirrors the canonical Python consumer
/// (`l1_production_helper`): the address record's own `data_width` wins
/// when strictly positive, else the instruction-level width.
#[derive(Debug, Clone, Copy)]
pub struct AddressAccess {
    pub lane_vector: [i64; lanes::LANES_PER_WARP],
    pub data_width_bytes: i32,
    /// Per-address memory space (address-carrying contract). Absent on legacy
    /// pre-transition traces (the field is unset), where consumers fall back to
    /// the instruction-level [`MemoryAccessMeta::op_class`]. A record carrying
    /// an explicit space lets a consumer route the global source of an atomic /
    /// async-copy to the L2 stream while exempting a shared destination.
    pub address_space: Option<pb::MemorySpace>,
    /// Backing global allocation id (absent/0 = unresolved). The identity gate
    /// requires a non-zero id resolving to a live sidecar allocation for a
    /// GLOBAL/GENERIC-space active record; SHARED/LOCAL are exempt.
    pub allocation_id: Option<u32>,
}

/// The instruction-level memory identity, without the per-record payloads —
/// what [`visit_memory_access`] returns after streaming the payloads into
/// the caller's sink.
#[derive(Debug, Clone, Copy)]
pub struct MemoryAccessMeta {
    pub op_class: pb::MemoryOpClass,
    /// Absent on legacy traces (pre-tag-12); consumers route absent to their
    /// UNSPECIFIED bucket, mirroring the Python consumers.
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// Absent or UNSPECIFIED means "tracer did not classify the cache
    /// operator" — downstream L1 modeling must treat that as typed missing
    /// provenance, not as a default operator. (A wire value UNKNOWN to the
    /// committed schema is a typed reader error instead — schema skew.)
    pub cache_op: Option<pb::InstructionCacheOp>,
    /// `active_mask & predicate_mask`. May be 0: a predicated-off memory
    /// instruction still ISSUES (the SM model sees it); it just touches no
    /// lanes — the sink receives no records in that case, mirroring the
    /// canonical consumers, which skip lane reconstruction on zero masks.
    pub and_mask: u32,
}

/// A successful full structural preflight tied to one immutable instruction.
/// Consumers may apply semantic gates to [`Self::meta`] before invoking
/// [`Self::visit_addresses`], which preserves structural-error precedence without
/// decoding the complete wire record again.
#[derive(Debug, Clone, Copy)]
pub struct PreflightedMemoryAccess<'a> {
    inst: &'a pb::Instruction,
    meta: MemoryAccessMeta,
    inst_width: i32,
}

impl<'a> PreflightedMemoryAccess<'a> {
    pub fn instruction(self) -> &'a pb::Instruction {
        self.inst
    }

    pub fn meta(self) -> MemoryAccessMeta {
        self.meta
    }

    /// Reconstruct each already-validated address into stack storage in wire
    /// order. The instruction borrow prevents mutation between preflight and
    /// this pass, making a second decode failure an unreachable invariant.
    pub fn visit_addresses<F>(self, mut sink: F)
    where
        F: FnMut(&AddressAccess),
    {
        self.visit_addresses_enumerated(|_, access| sink(access));
    }

    fn visit_addresses_enumerated<F>(self, mut sink: F)
    where
        F: FnMut(usize, &AddressAccess),
    {
        if self.meta.and_mask == 0 {
            return;
        }
        let no_error_context = || String::new();
        for (operand, addr) in self.inst.addresses.iter().enumerate() {
            let access = match decode_address_access(
                addr,
                self.meta.and_mask,
                self.inst_width,
                &no_error_context,
            ) {
                Ok(access) => access,
                Err(_) => unreachable!("preflighted immutable address failed a second decode"),
            };
            sink(operand, &access);
        }
    }

    fn decoded_address(self, operand: usize) -> Option<AddressAccess> {
        let addr = self.inst.addresses.get(operand)?;
        let no_error_context = || String::new();
        Some(
            decode_address_access(addr, self.meta.and_mask, self.inst_width, &no_error_context)
                .unwrap_or_else(|_| {
                    unreachable!("preflighted immutable address failed a second decode")
                }),
        )
    }
}

/// Sum each decoded address operand's per-lane width over its effective lanes.
/// This preserves payload volume independently of source-sector coalescing.
pub fn project_effective_data_bytes(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u64, EffectiveDataBytesError> {
    let mut bytes = 0_u64;
    let mut overflowed = false;
    preflight.visit_addresses(|access| {
        if overflowed {
            return;
        }
        match effective_data_bytes_for_access(access) {
            Ok(operand_bytes) => match bytes.checked_add(operand_bytes) {
                Some(total) => bytes = total,
                None => overflowed = true,
            },
            Err(EffectiveDataBytesError::Overflow) => overflowed = true,
        }
    });
    if overflowed {
        Err(EffectiveDataBytesError::Overflow)
    } else {
        Ok(bytes)
    }
}

fn project_operand_effective_data_bytes(
    preflight: PreflightedMemoryAccess<'_>,
    operand: usize,
) -> Result<u64, EffectiveDataBytesError> {
    let access = preflight
        .decoded_address(operand)
        .unwrap_or_else(|| unreachable!("validated async-copy source operand disappeared"));
    effective_data_bytes_for_access(&access)
}

fn effective_data_bytes_for_access(access: &AddressAccess) -> Result<u64, EffectiveDataBytesError> {
    let active_lanes = access
        .lane_vector
        .iter()
        .filter(|&&address| address != lanes::LANE_INACTIVE_SENTINEL)
        .count() as u64;
    let width =
        u64::try_from(access.data_width_bytes).map_err(|_| EffectiveDataBytesError::Overflow)?;
    width
        .checked_mul(active_lanes)
        .ok_or(EffectiveDataBytesError::Overflow)
}

const SHARED_BANK_COUNT_32X4: i64 = 32;
const SHARED_BANK_WORD_BYTES_32X4: i128 = 4;
const MAX_SHARED_WORDS_PER_LANE: u64 = 1_024;
const MAX_SHARED_WORDS_PER_INSTRUCTION: u64 = 65_536;

/// Reduce effective shared addresses to the exact bank-wavefront observable for
/// 32 banks with 4-byte words. Identical word addresses collapse as broadcasts;
/// the result is the maximum distinct-word count on any bank.
pub fn project_shared_bank_wavefronts_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    word_scratch.clear();
    let mut projected_words = 0_u64;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        let width = i128::from(access.data_width_bytes);
        for (lane, &address) in access.lane_vector.iter().enumerate() {
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let first_word = i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let last_byte = i128::from(address) + width - 1;
            let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let words = u64::try_from(last_word - first_word + 1)
                .unwrap_or_else(|_| unreachable!("positive data width produced a negative span"));
            if words > MAX_SHARED_WORDS_PER_LANE {
                failure = Some(SharedBankProjectionError::LaneWordLimit {
                    lane,
                    words,
                    limit: MAX_SHARED_WORDS_PER_LANE,
                });
                return;
            }
            let Some(next_projected_words) = projected_words.checked_add(words) else {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            };
            if next_projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: next_projected_words,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            }
            let additional = usize::try_from(words)
                .unwrap_or_else(|_| unreachable!("bounded word span escaped usize"));
            if word_scratch.try_reserve(additional).is_err() {
                failure = Some(SharedBankProjectionError::CapacityUnavailable { additional });
                return;
            }
            for word in first_word..=last_word {
                word_scratch.push(
                    i64::try_from(word)
                        .unwrap_or_else(|_| unreachable!("i64 byte address word escaped i64")),
                );
            }
            projected_words = next_projected_words;
        }
    });
    if let Some(failure) = failure {
        return Err(failure);
    }

    word_scratch.sort_unstable();
    word_scratch.dedup();
    let mut bank_words = [0_u64; SHARED_BANK_COUNT_32X4 as usize];
    for &word in word_scratch.iter() {
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank]
            .checked_add(1)
            .ok_or(SharedBankProjectionError::CountOverflow { bank: bank as u8 })?;
    }
    let wavefronts = bank_words.into_iter().max().unwrap_or(0);
    Ok(
        u32::try_from(wavefronts)
            .unwrap_or_else(|_| unreachable!("bounded projection escaped u32")),
    )
}

#[derive(Debug, Clone, Copy)]
struct LdgstsLaneTransfer {
    source_start: u64,
    source_end: u64,
    source_first_sector: i64,
    source_last_sector: i64,
    destination_first_word: i64,
    destination_last_word: i64,
    lane: u8,
    cohort: u8,
}

struct ProjectedLdgstsSourceCohorts {
    topology: LdgstsDestinationTopology,
    cohorts: Vec<LdgstsSourceCohort>,
    source_sector_cohort_masks: Vec<u32>,
}

const LDGSTS_DESTINATION_LINE_WORDS_128B: i64 = 128 / 4;

fn finish_ldgsts_destination_cohort(
    word_scratch: &mut Vec<i64>,
    active_transfers: u32,
    payload_bytes: u32,
    cohorts: &mut Vec<LdgstsSourceCohort>,
) -> Result<(), LdgstsDestinationProjectionError> {
    if active_transfers == 0 {
        debug_assert!(word_scratch.is_empty());
        debug_assert_eq!(payload_bytes, 0);
        return Ok(());
    }
    word_scratch.sort_unstable();
    word_scratch.dedup();
    let mut bank_words = [0_u32; SHARED_BANK_COUNT_32X4 as usize];
    let mut destination_lines = 0_u32;
    let mut previous_line = None;
    for &word in word_scratch.iter() {
        let line = word.div_euclid(LDGSTS_DESTINATION_LINE_WORDS_128B);
        if previous_line != Some(line) {
            destination_lines = destination_lines
                .checked_add(1)
                .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
            previous_line = Some(line);
        }
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank].checked_add(1).ok_or(
            LdgstsDestinationProjectionError::BankWavefrontCountOverflow { bank: bank as u8 },
        )?;
    }
    let bank_wavefronts = bank_words.into_iter().max().unwrap_or(0);
    let bypass_product = destination_lines.checked_mul(bank_wavefronts).ok_or(
        LdgstsDestinationProjectionError::BypassWavefrontProductOverflow {
            destination_lines,
            bank_wavefronts,
        },
    )?;
    let bypass_wavefronts = active_transfers.min(bypass_product);
    cohorts
        .try_reserve(1)
        .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable { additional: 1 })?;
    cohorts.push(LdgstsSourceCohort {
        payload_bytes,
        destination_lines,
        bank_wavefronts,
        bypass_wavefronts,
    });
    word_scratch.clear();
    Ok(())
}

fn recompose_ldgsts_destination_topology(
    cohorts: &[LdgstsSourceCohort],
) -> Result<LdgstsDestinationTopology, LdgstsDestinationProjectionError> {
    let mut topology = LdgstsDestinationTopology {
        source_cohorts: u32::try_from(cohorts.len())
            .map_err(|_| LdgstsDestinationProjectionError::SourceCohortCountOverflow)?,
        ..LdgstsDestinationTopology::default()
    };
    for cohort in cohorts {
        topology.destination_line_fanout_total = topology
            .destination_line_fanout_total
            .checked_add(cohort.destination_lines)
            .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
        topology.destination_line_fanout_max = topology
            .destination_line_fanout_max
            .max(cohort.destination_lines);
        topology.cache_all_bank_wavefronts_total = topology
            .cache_all_bank_wavefronts_total
            .checked_add(cohort.bank_wavefronts)
            .ok_or(LdgstsDestinationProjectionError::WavefrontCountOverflow)?;
        topology.cache_all_bank_wavefronts_max = topology
            .cache_all_bank_wavefronts_max
            .max(cohort.bank_wavefronts);
        topology.bypass_wavefronts_total = topology
            .bypass_wavefronts_total
            .checked_add(cohort.bypass_wavefronts)
            .ok_or(LdgstsDestinationProjectionError::BypassWavefrontCountOverflow)?;
        topology.bypass_wavefronts_max =
            topology.bypass_wavefronts_max.max(cohort.bypass_wavefronts);
    }
    Ok(topology)
}

fn project_ldgsts_source_cohorts_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    source_sectors: &[i64],
    word_scratch: &mut Vec<i64>,
) -> Result<ProjectedLdgstsSourceCohorts, LdgstsDestinationProjectionError> {
    let mut source_index = None;
    let mut source_count = 0_usize;
    let mut destination_index = None;
    let mut destination_count = 0_usize;
    for (operand, address) in preflight.instruction().addresses.iter().enumerate() {
        let space = address
            .address_space
            .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
        match space {
            Some(pb::MemorySpace::Global | pb::MemorySpace::Generic) => {
                source_count += 1;
                source_index.get_or_insert(operand);
            }
            Some(pb::MemorySpace::Shared) => {
                destination_count += 1;
                destination_index.get_or_insert(operand);
            }
            Some(pb::MemorySpace::Unspecified | pb::MemorySpace::Local) | None => {}
        }
    }
    if source_count != 1 {
        return Err(LdgstsDestinationProjectionError::SourceOperandCount { got: source_count });
    }
    if destination_count != 1 {
        return Err(LdgstsDestinationProjectionError::DestinationOperandCount {
            got: destination_count,
        });
    }
    let source = preflight
        .decoded_address(source_index.expect("count proved source index"))
        .expect("source index came from instruction addresses");
    let destination = preflight
        .decoded_address(destination_index.expect("count proved destination index"))
        .expect("destination index came from instruction addresses");
    if source.data_width_bytes != destination.data_width_bytes {
        return Err(LdgstsDestinationProjectionError::OperandWidthMismatch {
            source_bytes: source.data_width_bytes,
            destination_bytes: destination.data_width_bytes,
        });
    }
    let width = u64::try_from(source.data_width_bytes)
        .unwrap_or_else(|_| unreachable!("preflight admitted a non-positive operand width"));
    let width_u32 = u32::try_from(width)
        .unwrap_or_else(|_| unreachable!("one source transfer width escaped u32"));

    const EMPTY_TRANSFER: LdgstsLaneTransfer = LdgstsLaneTransfer {
        source_start: 0,
        source_end: 0,
        source_first_sector: 0,
        source_last_sector: 0,
        destination_first_word: 0,
        destination_last_word: 0,
        lane: 0,
        cohort: 0,
    };
    let mut transfers = [EMPTY_TRANSFER; lanes::LANES_PER_WARP];
    let mut transfer_count = 0_usize;
    for lane in 0..lanes::LANES_PER_WARP {
        let source_address = source.lane_vector[lane];
        let destination_address = destination.lane_vector[lane];
        if source_address == lanes::LANE_INACTIVE_SENTINEL {
            debug_assert_eq!(destination_address, lanes::LANE_INACTIVE_SENTINEL);
            continue;
        }
        if width > 128 {
            return Err(
                LdgstsDestinationProjectionError::SourceTransferExceedsCohort {
                    lane,
                    width_bytes: width,
                },
            );
        }
        let source_start = source_address as u64;
        let source_end = source_start.checked_add(width).ok_or(
            LdgstsDestinationProjectionError::SourceIntervalOverflow {
                lane,
                base: source_start,
                width_bytes: width,
            },
        )?;
        let destination_width_offset = i64::try_from(width - 1)
            .unwrap_or_else(|_| unreachable!("128-byte transfer width escaped i64"));
        let destination_last_byte = destination_address
            .checked_add(destination_width_offset)
            .ok_or(
                LdgstsDestinationProjectionError::DestinationIntervalOverflow {
                    lane,
                    base: destination_address,
                    width_bytes: width,
                },
            )?;
        transfers[transfer_count] = LdgstsLaneTransfer {
            source_start,
            source_end,
            source_first_sector: source_address >> 5,
            source_last_sector: ((source_end - 1) as i64) >> 5,
            destination_first_word: destination_address.div_euclid(4),
            destination_last_word: destination_last_byte.div_euclid(4),
            lane: lane as u8,
            cohort: 0,
        };
        transfer_count += 1;
    }
    transfers[..transfer_count].sort_unstable_by_key(|transfer| {
        (transfer.source_start, transfer.source_end, transfer.lane)
    });

    word_scratch.clear();
    let mut cohort_start = None;
    let mut cohort_active_transfers = 0_u32;
    let mut cohort_payload_bytes = 0_u32;
    let mut cohort_ordinal = 0_u8;
    let mut cohorts = Vec::new();
    let mut projected_words = 0_u64;
    for transfer in &mut transfers[..transfer_count] {
        if cohort_start.is_some_and(|start| transfer.source_end - start > 128) {
            finish_ldgsts_destination_cohort(
                word_scratch,
                cohort_active_transfers,
                cohort_payload_bytes,
                &mut cohorts,
            )?;
            cohort_start = None;
            cohort_active_transfers = 0;
            cohort_payload_bytes = 0;
            cohort_ordinal = cohort_ordinal
                .checked_add(1)
                .ok_or(LdgstsDestinationProjectionError::SourceCohortCountOverflow)?;
        }
        cohort_start.get_or_insert(transfer.source_start);
        transfer.cohort = cohort_ordinal;
        cohort_active_transfers = cohort_active_transfers
            .checked_add(1)
            .expect("one source cohort cannot exceed the 32-lane warp");
        cohort_payload_bytes = cohort_payload_bytes
            .checked_add(width_u32)
            .expect("one source cohort payload cannot exceed 4096 bytes");
        let words =
            u64::try_from(transfer.destination_last_word - transfer.destination_first_word + 1)
                .unwrap_or_else(|_| {
                    unreachable!("checked destination interval produced negative words")
                });
        projected_words = projected_words.checked_add(words).ok_or(
            LdgstsDestinationProjectionError::InstructionWordLimit {
                words: u64::MAX,
                limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
            },
        )?;
        if projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
            return Err(LdgstsDestinationProjectionError::InstructionWordLimit {
                words: projected_words,
                limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
            });
        }
        let additional = usize::try_from(words)
            .unwrap_or_else(|_| unreachable!("bounded destination words escaped usize"));
        word_scratch
            .try_reserve(additional)
            .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable { additional })?;
        word_scratch.extend(transfer.destination_first_word..=transfer.destination_last_word);
    }
    finish_ldgsts_destination_cohort(
        word_scratch,
        cohort_active_transfers,
        cohort_payload_bytes,
        &mut cohorts,
    )?;

    let topology = recompose_ldgsts_destination_topology(&cohorts)?;
    debug_assert_eq!(
        cohorts
            .iter()
            .map(|cohort| u64::from(cohort.payload_bytes))
            .sum::<u64>(),
        u64::from(width_u32) * transfer_count as u64
    );

    let mut source_sector_cohort_masks = Vec::new();
    source_sector_cohort_masks
        .try_reserve(source_sectors.len())
        .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable {
            additional: source_sectors.len(),
        })?;
    for &sector in source_sectors {
        let mut mask = 0_u32;
        for transfer in &transfers[..transfer_count] {
            if sector >= transfer.source_first_sector && sector <= transfer.source_last_sector {
                mask |= 1_u32
                    .checked_shl(u32::from(transfer.cohort))
                    .unwrap_or_else(|| unreachable!("32-lane cohort ordinal escaped u32 mask"));
            }
        }
        debug_assert_ne!(mask, 0, "compact source sector must belong to a cohort");
        source_sector_cohort_masks.push(mask);
    }
    Ok(ProjectedLdgstsSourceCohorts {
        topology,
        cohorts,
        source_sector_cohort_masks,
    })
}

/// Pair each active lane's LDGSTS source and destination transfer, form
/// flexible source cohorts whose complete byte span is at most 128 bytes, and
/// retain destination-line fanout separately from 32-bank/4-byte-word
/// multiplicity. The bypass projection is derived inside each cohort before
/// aggregation; it is an observational topology coordinate, not a timing rule.
/// Cohorts are anchored at the smallest uncovered source address rather than at
/// absolute 128-byte cache-line boundaries.
pub fn project_ldgsts_source_cohort_topology_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<LdgstsDestinationTopology, LdgstsDestinationProjectionError> {
    Ok(project_ldgsts_source_cohorts_128b_32x4(preflight, &[], word_scratch)?.topology)
}

/// Compatibility wrapper for the established cache-all bank-wavefront scalar.
pub fn project_ldgsts_source_cohort_wavefronts_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, LdgstsDestinationProjectionError> {
    Ok(
        project_ldgsts_source_cohort_topology_128b_32x4(preflight, word_scratch)?
            .cache_all_bank_wavefronts_total,
    )
}

/// A memory instruction decoded to what the memory hierarchy consumes —
/// the ALLOCATING convenience form (one heap `Vec` per memory record).
/// Hot walk paths (millions of records per kernel) use
/// [`visit_memory_access`] instead, which streams [`AddressAccess`] values
/// through a sink with zero per-record heap allocation.
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub op_class: pb::MemoryOpClass,
    /// See [`MemoryAccessMeta::access_kind`].
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// See [`MemoryAccessMeta::cache_op`].
    pub cache_op: Option<pb::InstructionCacheOp>,
    /// See [`MemoryAccessMeta::and_mask`].
    pub and_mask: u32,
    /// One decoded record per wire address record (empty when and_mask==0
    /// or the producer emitted no address payload).
    pub accesses: Vec<AddressAccess>,
}

fn known_enum<T: TryFrom<i32>>(
    raw: i32,
    field: &'static str,
    record: &impl Fn() -> String,
) -> Result<T, TraceError> {
    T::try_from(raw).map_err(|_| TraceError::UnknownEnumValue {
        record: record(),
        field,
        raw,
    })
}

fn memory_space_is_compatible(class: pb::MemoryOpClass, space: pb::MemorySpace) -> bool {
    match class {
        pb::MemoryOpClass::Global => {
            matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic)
        }
        pb::MemoryOpClass::Shared => space == pb::MemorySpace::Shared,
        pb::MemoryOpClass::Local => space == pb::MemorySpace::Local,
        pb::MemoryOpClass::Atomic => matches!(
            space,
            pb::MemorySpace::Global
                | pb::MemorySpace::Generic
                | pb::MemorySpace::Shared
                | pb::MemorySpace::Local
        ),
        // The generic structural pass admits the v3 destination space; the
        // enclosing payload-contract shape gate subsequently requires the exact
        // ordered (GLOBAL|GENERIC source, SHARED destination) pair. Historical
        // source-only contracts retain their stricter one-operand shape there.
        pb::MemoryOpClass::AsyncCopy => matches!(
            space,
            pb::MemorySpace::Global | pb::MemorySpace::Generic | pb::MemorySpace::Shared
        ),
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => true,
    }
}

fn decode_address_access(
    addr: &pb::Address,
    and_mask: u32,
    inst_width: i32,
    record: &impl Fn() -> String,
) -> Result<AddressAccess, TraceError> {
    // Match the Python gate's aggregate mask diagnosis before positional
    // reconstruction. A malformed list_all length has higher precedence there;
    // every other per-slot disagreement remains lane_position_mismatch.
    if addr.compression_format != lanes::FORMAT_LIST_ALL
        || addr.addrs.len() == lanes::LANES_PER_WARP
    {
        let real_count = if addr.compression_format == lanes::FORMAT_LIST_ALL {
            addr.addrs
                .iter()
                .filter(|&&slot| slot != lanes::LANE_INACTIVE_SENTINEL)
                .count()
        } else {
            addr.addrs.len()
        };
        if (and_mask == 0 && real_count > 0)
            || (and_mask != 0
                && addr.compression_format == lanes::FORMAT_LIST_ALL
                && real_count == 0)
        {
            return Err(TraceError::LaneMaskInconsistent { record: record() });
        }
    }

    let lane_vector =
        lanes::reconstruct_lane_vector(addr, and_mask).map_err(|source| TraceError::Lane {
            record: record(),
            source,
        })?;
    let data_width_bytes = if addr.data_width > 0 {
        addr.data_width
    } else {
        inst_width
    };
    let address_space = match addr.address_space {
        None => None,
        Some(raw) => Some(known_enum::<pb::MemorySpace>(raw, "address_space", record)?),
    };
    Ok(AddressAccess {
        lane_vector,
        data_width_bytes,
        address_space,
        allocation_id: addr.allocation_id,
    })
}

/// Validate one traced instruction's complete memory wire contract without
/// invoking a consumer callback. A successful token borrows the immutable
/// instruction and may later stream its address records without another full
/// preflight.
///
/// "Memory instruction" uses the canonical Python predicate: address records OR
/// a present `memory_op_class`. Data-width alone does not reclassify a non-memory
/// record; access-kind or cache-op metadata on such a record is rejected as a
/// producer contradiction. Every instruction must carry `warpid_tb`, a
/// positive function id, both masks, and a low-32-consistent `vpc`. Memory records
/// additionally require a specified class and positive instruction width; unknown
/// enum values are typed schema skew, explicit memory-space UNSPECIFIED is rejected,
/// and every address record must reconstruct positionally. Non-memory instructions
/// return `Ok(None)` after their universal fields have been validated.
pub fn preflight_memory_access<'a>(
    inst: &'a pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<PreflightedMemoryAccess<'a>>, TraceError> {
    let n_addresses = inst.addresses.len();
    // Canonical Python predicate: data_width and additive provenance fields do
    // not by themselves reclassify an otherwise non-memory instruction. The
    // additive fields are nevertheless contradictory on that record and are
    // rejected after their enum values receive complete wire validation.
    let looks_like_memory = n_addresses > 0 || inst.memory_op_class.is_some();

    // Keep the Python validator's fail-fast precedence: memory width first,
    // followed by the fields required on every instruction.
    let inst_width = if looks_like_memory {
        let Some(inst_width) = inst.data_width else {
            return Err(TraceError::DataWidthMissing { record: record() });
        };
        if inst_width <= 0 {
            return Err(TraceError::DataWidthInvalid {
                record: record(),
                got: inst_width,
            });
        }
        inst_width
    } else {
        0
    };

    if inst.warpid_tb.is_none() {
        return Err(TraceError::WarpidTbMissing { record: record() });
    }
    if inst.function_unique_id <= 0 {
        return Err(TraceError::FunctionUniqueIdInvalid {
            record: record(),
            got: inst.function_unique_id,
        });
    }
    let Some(active_mask) = inst.active_mask else {
        return Err(TraceError::ActiveMaskMissing { record: record() });
    };
    let Some(predicate_mask) = inst.predicate_mask else {
        return Err(TraceError::PredicateMaskMissing { record: record() });
    };
    let Some(vpc) = inst.vpc else {
        return Err(TraceError::VpcMissing { record: record() });
    };
    let vpc_low_32 = vpc as u32;
    if vpc_low_32 != inst.pc {
        return Err(TraceError::VpcInconsistentWithPc {
            record: record(),
            vpc_low_32,
            pc: inst.pc,
        });
    }

    if !looks_like_memory {
        let access_kind = match inst.memory_op_access_kind {
            None => None,
            Some(raw) => {
                let kind: pb::MemoryOpAccessKind =
                    known_enum(raw, "memory_op_access_kind", &record)?;
                if kind == pb::MemoryOpAccessKind::Unspecified {
                    return Err(TraceError::MemoryOpAccessKindUnspecified { record: record() });
                }
                Some(kind)
            }
        };
        let cache_op = match inst.instruction_cache_op {
            None => None,
            Some(raw) => {
                let op: pb::InstructionCacheOp = known_enum(raw, "instruction_cache_op", &record)?;
                if op == pb::InstructionCacheOp::Unspecified {
                    return Err(TraceError::InstructionCacheOpUnspecified { record: record() });
                }
                Some(op)
            }
        };
        if access_kind.is_some() {
            return Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                record: record(),
                field: "memory_op_access_kind",
            });
        }
        if cache_op.is_some() {
            return Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                record: record(),
                field: "instruction_cache_op",
            });
        }
        return Ok(None);
    }

    let Some(op_class_raw) = inst.memory_op_class else {
        return Err(TraceError::MemoryOpClassMissing {
            record: record(),
            n_addresses,
        });
    };
    let op_class: pb::MemoryOpClass = known_enum(op_class_raw, "memory_op_class", &record)?;
    if op_class == pb::MemoryOpClass::Unspecified {
        return Err(TraceError::MemoryOpClassUnspecified { record: record() });
    }

    let and_mask = active_mask & predicate_mask;
    // Validate every address, including zero-mask records. Those stream no
    // callbacks, but malformed lane payloads and unknown enums remain failures.
    for addr in &inst.addresses {
        decode_address_access(addr, and_mask, inst_width, &record)?;
    }
    // Structural preflight above intentionally completes before this semantic
    // compatibility gate. Explicit UNSPECIFIED is never a legacy fallback;
    // absence alone retains the class-derived behavior.
    for (operand, addr) in inst.addresses.iter().enumerate() {
        let Some(raw) = addr.address_space else {
            continue;
        };
        let space: pb::MemorySpace = known_enum(raw, "address_space", &record)?;
        if space == pb::MemorySpace::Unspecified {
            return Err(TraceError::MemorySpaceUnspecified {
                record: record(),
                operand,
            });
        }
        if !memory_space_is_compatible(op_class, space) {
            return Err(TraceError::MemoryOpClassAddressSpaceConflict {
                record: record(),
                operand,
                class: op_class,
                space,
            });
        }
    }

    let access_kind = match inst.memory_op_access_kind {
        None => None,
        Some(raw) => {
            let kind: pb::MemoryOpAccessKind = known_enum(raw, "memory_op_access_kind", &record)?;
            if kind == pb::MemoryOpAccessKind::Unspecified {
                return Err(TraceError::MemoryOpAccessKindUnspecified { record: record() });
            }
            Some(kind)
        }
    };
    let cache_op = match inst.instruction_cache_op {
        None => None,
        Some(raw) => {
            let op: pb::InstructionCacheOp = known_enum(raw, "instruction_cache_op", &record)?;
            if op == pb::InstructionCacheOp::Unspecified {
                return Err(TraceError::InstructionCacheOpUnspecified { record: record() });
            }
            Some(op)
        }
    };

    Ok(Some(PreflightedMemoryAccess {
        inst,
        meta: MemoryAccessMeta {
            op_class,
            access_kind,
            cache_op,
            and_mask,
        },
        inst_width,
    }))
}

/// Interpret one traced instruction per the wire contract, streaming each
/// decoded address record into `sink` — the zero-per-record-allocation form for
/// hot walk paths. Full preflight completes before the first callback, so an
/// error never exposes a partial callback prefix. The sink borrows a
/// stack-resident [`AddressAccess`]; copy it out (`Copy`) only if it must outlive
/// the call.
pub fn visit_memory_access<F>(
    inst: &pb::Instruction,
    record: impl Fn() -> String,
    sink: F,
) -> Result<Option<MemoryAccessMeta>, TraceError>
where
    F: FnMut(&AddressAccess),
{
    let Some(preflight) = preflight_memory_access(inst, record)? else {
        return Ok(None);
    };
    let meta = preflight.meta();
    preflight.visit_addresses(sink);
    Ok(Some(meta))
}

/// One backing allocation's device VA range.
#[derive(Debug, Clone, Copy)]
struct AllocRange {
    base: u64,
    size: u64,
}

/// The `<trace_dir>/extra_info/allocations.json` registry — the same sidecar the
/// Python `import_trace_contract` validator reads (an array of
/// `{allocation_id, base_address (hex string), size_bytes, ...}`). Used to
/// enforce the address-carrying contract's allocation identity in the Rust
/// reader, at parity with the Python gate.
#[derive(Debug, Clone, Default)]
pub struct AllocationRegistry {
    by_id: BTreeMap<u32, AllocRange>,
}

/// The EXACT six-field allocation-sidecar entry, at parity with the Python
/// validator's shape gate (`import_trace_contract.py`: all six fields required,
/// no extras). `deny_unknown_fields` + all-required non-Option fields make serde
/// reject a missing field, an unexpected key, a negative/boolean `allocation_id`
/// or `size_bytes`, and a non-bool `alive`. The explicit VALUE checks (positive
/// id/size, 0x-prefixed base, duplicate id) run in `from_str` after.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAllocation {
    allocation_id: u32,
    // context_id / device_id / alive are producer metadata not consumed by the
    // range check (Python does not gate on `alive` — a freed-at-trace-end
    // allocation was valid during the trace), but each MUST be present and
    // correctly typed for exact-shape parity. `device_id` may be 0 or negative.
    #[allow(dead_code)]
    context_id: i64,
    #[allow(dead_code)]
    device_id: i64,
    base_address: String,
    size_bytes: u64,
    #[allow(dead_code)]
    alive: bool,
}

fn parse_hex_u64(s: &str) -> Option<u64> {
    // Canonical cross-runtime grammar: an exact 0x/0X prefix followed by one or
    // more ASCII hex digits. In particular, Python-style digit separators are
    // not wire syntax.
    let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

impl AllocationRegistry {
    /// Return an address's byte offset within a validated allocation.
    ///
    /// This exposes allocation-relative identity without exposing or inventing
    /// a physical address. `None` means the id is unknown, the subtraction
    /// underflows, or the address is at/after the allocation end.
    pub fn relative_byte_offset(&self, allocation_id: u32, address: u64) -> Option<u64> {
        let range = self.by_id.get(&allocation_id)?;
        let offset = address.checked_sub(range.base)?;
        (offset < range.size).then_some(offset)
    }

    /// Return a compact 32-byte sector's offset within a sector-aligned
    /// allocation.
    ///
    /// Global CUDA allocations are sector aligned in admitted traces. A
    /// malformed/non-aligned synthetic registry fails closed rather than
    /// silently rounding its base.
    pub fn relative_sector_offset(&self, allocation_id: u32, absolute_sector: i64) -> Option<u64> {
        let range = self.by_id.get(&allocation_id)?;
        if range.base & 31 != 0 {
            return None;
        }
        let sector = u64::try_from(absolute_sector).ok()?;
        let base_sector = range.base >> 5;
        let offset = sector.checked_sub(base_sector)?;
        let byte_offset = offset.checked_mul(32)?;
        (byte_offset < range.size).then_some(offset)
    }

    /// Load from `<root>/extra_info/allocations.json`. `Ok(None)` if the file is
    /// absent (legacy traces, or traces with no GLOBAL/GENERIC active-lane record
    /// requiring identity). `Err(AllocationsSidecarMalformed)` on bad JSON, a
    /// non-hex `base_address`, or a duplicate `allocation_id`.
    pub fn load(root: &Path) -> Result<Option<Self>, TraceError> {
        let path = root.join("extra_info").join("allocations.json");
        let reader = AnchoredDir::open(root)?;
        let raw = match reader
            .read_optional_bounded_regular(
                Path::new("extra_info/allocations.json"),
                ALLOCATIONS_JSON_MAX_BYTES,
            )
            .map_err(|source| map_snapshot_read_error("allocations_json", source))?
        {
            Some(file) => file.into_raw(),
            None => return Ok(None),
        };
        Self::parse(&path, &raw).map(Some)
    }

    #[cfg(test)]
    fn from_str(raw: &str, path: &Path) -> Result<Self, TraceError> {
        Self::parse(path, raw.as_bytes())
    }

    fn parse(path: &Path, raw: &[u8]) -> Result<Self, TraceError> {
        Self::parse_with_limits(
            path,
            raw,
            ALLOCATIONS_JSON_MAX_BYTES,
            ALLOCATIONS_JSON_MAX_NODES,
            ALLOCATION_ENTRY_MAX_COUNT,
        )
    }

    fn parse_with_limits(
        path: &Path,
        raw: &[u8],
        maximum_bytes: u64,
        maximum_nodes: u64,
        maximum_entries: u64,
    ) -> Result<Self, TraceError> {
        sidecar_byte_limit(path, "allocations_json", raw, maximum_bytes)?;
        match preflight_json_shape(raw, maximum_nodes, Some(maximum_entries)) {
            Ok(_) => {}
            Err(JsonPreflightError::Malformed(message)) => {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message,
                });
            }
            Err(JsonPreflightError::ResourceLimit(error)) => {
                return Err(map_json_preflight_limit(path, "allocations_json", error));
            }
        }
        let entries: Vec<RawAllocation> =
            serde_json::from_slice(raw).map_err(|e| TraceError::AllocationsSidecarMalformed {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        let mut by_id = BTreeMap::new();
        for e in entries {
            // Python parity: allocation ids start at 1 (0 is the unresolved
            // sentinel) and every allocation has a positive size.
            if e.allocation_id == 0 {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: "allocation_id must be positive (0 is the unresolved sentinel)"
                        .to_string(),
                });
            }
            if e.size_bytes == 0 {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!("allocation_id {} has size_bytes == 0", e.allocation_id),
                });
            }
            let base = parse_hex_u64(&e.base_address).ok_or_else(|| {
                TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!(
                        "allocation_id {} has a non-0x-hex base_address {:?}",
                        e.allocation_id, e.base_address
                    ),
                }
            })?;
            if by_id
                .insert(
                    e.allocation_id,
                    AllocRange {
                        base,
                        size: e.size_bytes,
                    },
                )
                .is_some()
            {
                return Err(TraceError::AllocationIdDuplicate {
                    path: path.to_path_buf(),
                    allocation_id: e.allocation_id,
                });
            }
        }
        Ok(Self { by_id })
    }
}

/// Whether an effective per-address space requires global allocation identity.
/// A record's own `address_space` is authoritative (GLOBAL/GENERIC require it,
/// SHARED/LOCAL are exempt); absent falls back to the instruction-level GLOBAL
/// class — exactly the Python gate's rule.
fn record_requires_identity(
    space: Option<pb::MemorySpace>,
    inst_class: Option<pb::MemoryOpClass>,
) -> bool {
    match space {
        Some(pb::MemorySpace::Global) | Some(pb::MemorySpace::Generic) => true,
        Some(_) => false,
        None => inst_class == Some(pb::MemoryOpClass::Global),
    }
}

fn validate_address_carrying_shape(
    preflight: PreflightedMemoryAccess<'_>,
    contract: MemoryPayloadContract,
    record: &impl Fn() -> String,
) -> Result<(), TraceError> {
    let meta = preflight.meta();
    let instruction = preflight.instruction();
    if !ADDRESS_CARRYING_PAYLOAD_CLASSES.contains(&meta.op_class) {
        return Err(TraceError::MemoryPayloadClassUndeclared {
            record: record(),
            class: meta.op_class,
        });
    }

    let operand_count = instruction.addresses.len();
    let invalid = |detail| {
        Err(TraceError::MemoryPayloadShapeInvalid {
            record: record(),
            class: meta.op_class,
            detail,
            got: operand_count,
        })
    };
    match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if meta.and_mask != 0 && operand_count == 0 {
                return invalid("an active record requires at least one operand");
            }
        }
        pb::MemoryOpClass::Atomic => {
            if meta.and_mask == 0 && operand_count == 0 {
                return Ok(());
            }
            if operand_count != 1 {
                return invalid("expected exactly one operand, except zero-active may carry none");
            }
            if instruction.addresses[0].address_space.is_none() {
                return invalid("the sole operand requires an explicit concrete address_space");
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if contract.carries_ldgsts_destination() {
                if operand_count != 2 {
                    return invalid(
                        "expected exactly an ordered GLOBAL/GENERIC source and SHARED destination",
                    );
                }
                let source_space = instruction.addresses[0]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if !matches!(
                    source_space,
                    Some(pb::MemorySpace::Global | pb::MemorySpace::Generic)
                ) {
                    return invalid("operand 0 requires explicit GLOBAL or GENERIC source space");
                }
                let destination_space = instruction.addresses[1]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if destination_space != Some(pb::MemorySpace::Shared) {
                    return invalid("operand 1 requires explicit SHARED destination space");
                }
            } else {
                if operand_count != 1 {
                    return invalid("expected exactly one source operand");
                }
                let source_space = instruction.addresses[0]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if !matches!(
                    source_space,
                    Some(pb::MemorySpace::Global | pb::MemorySpace::Generic)
                ) {
                    return invalid("the source operand requires explicit GLOBAL or GENERIC space");
                }
            }
        }
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => {
            unreachable!("undeclared classes return before shape validation")
        }
    }
    Ok(())
}

/// Enforce the Python-equivalent allocation identity on one instruction against
/// the loaded registry (`None` = no sidecar). The complete memory wire record is
/// preflighted before the first identity diagnosis, so an unknown enum or malformed
/// later lane vector cannot be masked by an earlier missing allocation id. A
/// GLOBAL/GENERIC-space record with active lanes MUST carry a non-zero
/// `allocation_id` that exists in the registry and whose range contains every
/// active non-sentinel lane address. SHARED/LOCAL records (e.g. a shared atomic
/// ATOMS) are exempt. This is the reader-side identity gate — parity with
/// `import_trace_contract._check_instruction`.
pub fn check_allocation_identity(
    inst: &pb::Instruction,
    registry: Option<&AllocationRegistry>,
    record: impl Fn() -> String,
) -> Result<(), TraceError> {
    let Some(preflight) = preflight_memory_access(inst, &record)? else {
        return Ok(());
    };
    check_preflighted_allocation_identity(preflight, registry, record)
}

/// Apply the complete per-instruction contract and enclosing warp-identity
/// contract to every record in one decoded threadblock. Iteration follows sorted
/// protobuf map keys and then instruction order, matching the full-snapshot
/// admission walk.
pub fn validate_threadblock_contract(
    threadblock: &pb::Threadblock,
    label: impl AsRef<str>,
    registry: Option<&AllocationRegistry>,
    payload_contract: MemoryPayloadContract,
    expected_function_unique_id: i32,
) -> Result<(), TraceError> {
    let label = label.as_ref();
    for (&warp_id, warp) in &threadblock.warps {
        if warp.id != warp_id {
            return Err(TraceError::WarpMapKeyMismatch {
                threadblock: label.to_owned(),
                map_key: warp_id,
                warp_id: warp.id,
            });
        }
        for (instruction_index, instruction) in warp.instructions.iter().enumerate() {
            let record = || format!("{label} warp={warp_id} inst={instruction_index}");
            let preflight = preflight_memory_access(instruction, &record)?;
            let Some(instruction_warp_id) = instruction.warpid_tb else {
                return Err(TraceError::WarpidTbMissing { record: record() });
            };
            if instruction_warp_id != warp_id {
                return Err(TraceError::WarpidTbMismatch {
                    record: record(),
                    expected: warp_id,
                    actual: instruction_warp_id,
                });
            }
            if instruction.function_unique_id != expected_function_unique_id {
                return Err(TraceError::FunctionUniqueIdMismatch {
                    record: record(),
                    expected: expected_function_unique_id,
                    actual: instruction.function_unique_id,
                });
            }
            let Some(preflight) = preflight else {
                continue;
            };
            if matches!(
                payload_contract,
                MemoryPayloadContract::AddressCarryingV1
                    | MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
                    | MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
            ) {
                validate_address_carrying_shape(preflight, payload_contract, &record)?;
            }
            check_preflighted_allocation_identity(preflight, registry, record)?;
        }
    }
    Ok(())
}

/// Apply allocation identity to an instruction whose complete wire structure was
/// already validated. This split lets replay enforce its producer-shape route
/// before identity without rerunning fallible structural preflight or allocating an
/// intermediate decoded-payload vector; immutable address records are revisited by
/// the allocation-free visitor.
pub fn check_preflighted_allocation_identity(
    preflight: PreflightedMemoryAccess<'_>,
    registry: Option<&AllocationRegistry>,
    record: impl Fn() -> String,
) -> Result<(), TraceError> {
    let meta = preflight.meta();
    let inst_class = Some(meta.op_class);
    for addr in &preflight.instruction().addresses {
        let address_space = match addr.address_space {
            None => None,
            Some(raw) => Some(known_enum::<pb::MemorySpace>(
                raw,
                "address_space",
                &record,
            )?),
        };
        if !record_requires_identity(address_space, inst_class) {
            continue;
        }

        let allocation_id = addr.allocation_id.unwrap_or(0);
        if allocation_id == 0 {
            if meta.and_mask != 0 {
                return Err(TraceError::AllocationIdMissing { record: record() });
            }
            continue;
        }
        let Some(registry) = registry else {
            return Err(TraceError::AllocationsSidecarMissing { record: record() });
        };
        let Some(range) = registry.by_id.get(&allocation_id) else {
            return Err(TraceError::AllocationIdUnknown {
                record: record(),
                allocation_id,
            });
        };
        if meta.and_mask == 0 {
            continue;
        }

        let access = decode_address_access(addr, meta.and_mask, preflight.inst_width, &record)?;
        let base = u128::from(range.base);
        let end = base + u128::from(range.size);
        for (lane, &slot) in access.lane_vector.iter().enumerate() {
            if slot == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let lane_addr = slot as u64;
            let lane_addr_wide = u128::from(lane_addr);
            if !(base <= lane_addr_wide && lane_addr_wide < end) {
                return Err(TraceError::LaneOutsideAllocationRange {
                    record: record(),
                    lane,
                    lane_addr,
                    allocation_id,
                    base: range.base,
                    end,
                });
            }
        }
    }
    Ok(())
}

/// Allocating convenience wrapper over [`visit_memory_access`]: collects the
/// streamed records into a `Vec`. Identical contract and typed errors —
/// implemented on top of the streaming core so the two forms cannot drift.
pub fn memory_access(
    inst: &pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<MemoryAccess>, TraceError> {
    let mut accesses = Vec::new();
    let meta = visit_memory_access(inst, record, |a| accesses.push(*a))?;
    Ok(meta.map(|m| MemoryAccess {
        op_class: m.op_class,
        access_kind: m.access_kind,
        cache_op: m.cache_op,
        and_mask: m.and_mask,
        accesses,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // unwrap in tests is an assertion, not error handling
mod tests {
    use super::*;
    use prost::Message;

    fn address_carrying_trace() -> pb::Trace {
        pb::Trace {
            trace_contract_id: ADDRESS_CARRYING_TRACE_CONTRACT_ID.into(),
            memory_payload_classes: ADDRESS_CARRYING_PAYLOAD_CLASSES
                .into_iter()
                .map(|class| class as i32)
                .collect(),
            ..Default::default()
        }
    }

    fn split_v2_trace() -> pb::Trace {
        pb::Trace {
            trace_contract_id: SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID.into(),
            memory_payload_classes: ADDRESS_CARRYING_PAYLOAD_CAPABILITY_VECTOR.to_vec(),
            ..Default::default()
        }
    }

    fn ldgsts_destination_v3_trace() -> pb::Trace {
        pb::Trace {
            trace_contract_id: LDGSTS_DESTINATION_TRACE_CONTRACT_ID.into(),
            memory_payload_classes: ADDRESS_CARRYING_PAYLOAD_CAPABILITY_VECTOR.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn memory_payload_contract_distinguishes_legacy_and_address_carrying_traces() {
        assert_eq!(
            classify_memory_payload_contract(&pb::Trace::default(), Path::new("legacy.pb"))
                .expect("empty capability is the legacy signal"),
            MemoryPayloadContract::LegacyExclusion
        );
        assert_eq!(
            classify_memory_payload_contract(
                &address_carrying_trace(),
                Path::new("address-carrying.pb")
            )
            .expect("complete address-carrying capability"),
            MemoryPayloadContract::AddressCarryingV1
        );
        assert_eq!(
            classify_memory_payload_contract(&split_v2_trace(), Path::new("split-v2.pb"))
                .expect("canonical ordered split-v2 capability vector"),
            MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
        );
        assert_eq!(
            classify_memory_payload_contract(
                &ldgsts_destination_v3_trace(),
                Path::new("ldgsts-destination-v3.pb")
            )
            .expect("canonical ordered destination-v3 capability vector"),
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
        );
    }

    #[test]
    fn split_v2_requires_order_while_v1_retains_historical_set_semantics() {
        for (name, capabilities) in [
            ("reversed", vec![8, 6, 3, 2, 1]),
            ("representative-permutation", vec![1, 3, 2, 6, 8]),
        ] {
            let mut trace = split_v2_trace();
            trace.memory_payload_classes = capabilities.clone();
            assert!(
                matches!(
                    classify_memory_payload_contract(&trace, Path::new("split-v2.pb")),
                    Err(TraceError::MemoryPayloadCapabilitiesInvalid {
                        contract_id,
                        got,
                        ..
                    }) if contract_id == SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID && got == capabilities
                ),
                "{name} split-v2 vector must fail closed"
            );
        }

        let mut v1 = address_carrying_trace();
        v1.memory_payload_classes.reverse();
        assert_eq!(
            classify_memory_payload_contract(&v1, Path::new("historical-v1.pb"))
                .expect("v1 retains its historical set-based capability check"),
            MemoryPayloadContract::AddressCarryingV1
        );
    }

    #[test]
    fn memory_payload_contract_rejects_partial_or_unbound_capabilities() {
        let mut partial = address_carrying_trace();
        partial
            .memory_payload_classes
            .retain(|&raw| raw != pb::MemoryOpClass::Atomic as i32);
        assert!(matches!(
            classify_memory_payload_contract(&partial, Path::new("partial.pb")),
            Err(TraceError::MemoryPayloadCapabilitiesInvalid { .. })
        ));

        let mut duplicate = address_carrying_trace();
        duplicate
            .memory_payload_classes
            .push(pb::MemoryOpClass::Global as i32);
        assert!(matches!(
            classify_memory_payload_contract(&duplicate, Path::new("duplicate.pb")),
            Err(TraceError::MemoryPayloadCapabilitiesInvalid { .. })
        ));

        let mut unbound = address_carrying_trace();
        unbound.trace_contract_id.clear();
        assert!(matches!(
            classify_memory_payload_contract(&unbound, Path::new("unbound.pb")),
            Err(TraceError::MemoryPayloadCapabilitiesInvalid { .. })
        ));

        let mut unknown = address_carrying_trace();
        unknown.trace_contract_id = "future_contract".into();
        assert!(matches!(
            classify_memory_payload_contract(&unknown, Path::new("unknown.pb")),
            Err(TraceError::TraceContractUnsupported { .. })
        ));
    }

    fn shared_projection_inst(stride_words: i64, width: i32) -> pb::Instruction {
        pb::Instruction {
            pc: 32,
            active_mask: Some(u32::MAX),
            predicate_mask: Some(u32::MAX),
            function_unique_id: 1,
            addresses: vec![pb::Address {
                data_width: width,
                compression_format: crate::lanes::FORMAT_LIST_ALL,
                addrs: (0_i64..32).map(|lane| lane * stride_words * 4).collect(),
                address_space: Some(pb::MemorySpace::Shared as i32),
                ..Default::default()
            }],
            memory_op_class: Some(pb::MemoryOpClass::Shared as i32),
            data_width: Some(width),
            warpid_tb: Some(0),
            vpc: Some(32),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        }
    }

    fn atomic_projection_inst<F>(active_mask: u32, address: F) -> pb::Instruction
    where
        F: Fn(i64) -> i64,
    {
        pb::Instruction {
            pc: 48,
            active_mask: Some(active_mask),
            predicate_mask: Some(active_mask),
            function_unique_id: 1,
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: crate::lanes::FORMAT_LIST_ALL,
                addrs: (0_i64..32)
                    .map(|lane| {
                        if active_mask & (1_u32 << lane) != 0 {
                            address(lane)
                        } else {
                            crate::lanes::LANE_INACTIVE_SENTINEL
                        }
                    })
                    .collect(),
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            memory_op_class: Some(pb::MemoryOpClass::Atomic as i32),
            data_width: Some(4),
            warpid_tb: Some(0),
            vpc: Some(48),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Atomic as i32),
            ..Default::default()
        }
    }

    #[test]
    fn global_atomic_sector_projection_preserves_same_word_serialization() {
        let base = 0x1000_i64;
        for (label, instruction, expected) in [
            (
                "lane_line",
                atomic_projection_inst(u32::MAX, |lane| base + lane * 128),
                (0_i64..32).map(|lane| (base + lane * 128) >> 5).collect(),
            ),
            (
                "lane_word",
                atomic_projection_inst(u32::MAX, |lane| base + lane * 4),
                vec![base >> 5, (base >> 5) + 1, (base >> 5) + 2, (base >> 5) + 3],
            ),
            (
                "warp_word",
                atomic_projection_inst(u32::MAX, |_| base),
                vec![base >> 5; 32],
            ),
            (
                "two_way_alias",
                atomic_projection_inst(u32::MAX, |lane| base + (lane % 16) * 4),
                vec![base >> 5, base >> 5, (base >> 5) + 1, (base >> 5) + 1],
            ),
            (
                "one_lane",
                atomic_projection_inst(1, |_| base),
                vec![base >> 5],
            ),
        ] {
            let preflight = preflight_memory_access(&instruction, || label.to_owned())
                .unwrap()
                .unwrap();
            assert_eq!(
                compact_atomic_sector_transaction_vector_32b(preflight).unwrap(),
                expected,
                "{label}"
            );
        }

        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![atomic_projection_inst(u32::MAX, |_| base)],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingV1,
            "compact atomic",
        )
        .unwrap();
        let memory = compact.instructions().next().unwrap().memory().unwrap();
        assert_eq!(memory.global_atomic_allocation_id(), None);
        assert!(matches!(
            compact.sector_plan(memory),
            ValidatedSectorPlan::Sectors(sectors)
                if sectors == vec![base >> 5; 32].as_slice()
        ));

        let mut identified = atomic_projection_inst(u32::MAX, |_| base);
        identified.addresses[0].allocation_id = Some(7);
        let identified_compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![identified],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingV1,
            "identified compact atomic",
        )
        .unwrap();
        let identified_memory = identified_compact
            .instructions()
            .next()
            .unwrap()
            .memory()
            .unwrap();
        assert_eq!(identified_memory.global_atomic_allocation_id(), Some(7));
        assert!(std::mem::size_of::<ValidatedMemoryInstruction>() <= 24);
    }

    #[test]
    fn compact_instruction_identity_preserves_optional_i32_values() {
        let instruction = |pc, sm_id, warpid_sm| pb::Instruction {
            pc,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 7,
            warpid_tb: Some(3),
            sm_id,
            warpid_sm,
            ..Default::default()
        };
        let compact = ValidatedWarp::from_message(
            3,
            pb::Warp {
                id: 3,
                instructions: vec![
                    instruction(4, Some(i32::MIN), Some(i32::MAX)),
                    instruction(8, None, None),
                ],
                ..Default::default()
            },
            MemoryPayloadContract::LegacyExclusion,
            "compact identity",
        )
        .unwrap();
        let instructions = compact.instructions().collect::<Vec<_>>();
        assert_eq!(std::mem::size_of::<ValidatedInstruction>(), 20);
        assert_eq!(instructions[0].function_unique_id(), 7);
        assert_eq!(instructions[0].warpid_tb(), 3);
        assert_eq!(instructions[0].sm_id(), Some(i32::MIN));
        assert_eq!(instructions[0].warpid_sm(), Some(i32::MAX));
        assert_eq!(instructions[1].sm_id(), None);
        assert_eq!(instructions[1].warpid_sm(), None);
    }

    #[test]
    fn shared_bank_projection_counts_conflicts_width_and_broadcasts() {
        for (stride_words, width, expected) in [
            (0, 4, 1),
            (1, 4, 1),
            (2, 4, 2),
            (4, 4, 4),
            (4, 16, 4),
            (8, 4, 8),
            (16, 4, 16),
            (32, 4, 32),
        ] {
            let inst = shared_projection_inst(stride_words, width);
            let preflight = preflight_memory_access(&inst, || "shared projection".to_owned())
                .unwrap()
                .unwrap();
            let mut scratch = Vec::new();
            assert_eq!(
                project_shared_bank_wavefronts_32x4(preflight, &mut scratch).unwrap(),
                expected,
                "stride_words={stride_words}, width={width}"
            );
        }

        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![shared_projection_inst(4, 4)],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingV1,
            "compact shared",
        )
        .unwrap();
        let memory = compact.instructions().next().unwrap().memory().unwrap();
        assert_eq!(memory.shared_bank_wavefronts_32x4(), 4);

        let mut zero_active = shared_projection_inst(32, 4);
        zero_active.predicate_mask = Some(0);
        zero_active.addresses[0]
            .addrs
            .fill(crate::lanes::LANE_INACTIVE_SENTINEL);
        let preflight = preflight_memory_access(&zero_active, || "zero shared".to_owned())
            .unwrap()
            .unwrap();
        assert_eq!(
            project_shared_bank_wavefronts_32x4(preflight, &mut Vec::new()).unwrap(),
            0
        );
    }

    fn async_payload_inst(active_lanes: u32, width: i32, stride: i64) -> pb::Instruction {
        let mask = if active_lanes == 32 {
            u32::MAX
        } else {
            (1_u32 << active_lanes) - 1
        };
        let addrs = (0_i64..32)
            .map(|lane| {
                if lane < i64::from(active_lanes) {
                    0x1000 + lane * stride
                } else {
                    crate::lanes::LANE_INACTIVE_SENTINEL
                }
            })
            .collect();
        pb::Instruction {
            pc: 64,
            active_mask: Some(mask),
            predicate_mask: Some(mask),
            function_unique_id: 1,
            addresses: vec![pb::Address {
                data_width: width,
                compression_format: crate::lanes::FORMAT_LIST_ALL,
                addrs,
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            memory_op_class: Some(pb::MemoryOpClass::AsyncCopy as i32),
            data_width: Some(width),
            warpid_tb: Some(0),
            vpc: Some(64),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::AsyncCopy as i32),
            ..Default::default()
        }
    }

    #[test]
    fn async_payload_projection_is_exact_and_independent_of_unique_sectors() {
        let repeated = async_payload_inst(16, 16, 0);
        let spread = async_payload_inst(16, 16, 64);
        let zero = async_payload_inst(0, 16, 0);
        for (label, instruction, expected) in [
            ("repeated", &repeated, 256),
            ("spread", &spread, 256),
            ("zero", &zero, 0),
        ] {
            let preflight = preflight_memory_access(instruction, || label.to_owned())
                .unwrap()
                .unwrap();
            assert_eq!(project_effective_data_bytes(preflight).unwrap(), expected);
        }

        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![repeated, spread, zero],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingV1,
            "compact async payload",
        )
        .unwrap();
        let memories = compact
            .instructions()
            .map(|instruction| instruction.memory().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(compact.effective_data_bytes(memories[0]), 256);
        assert_eq!(compact.effective_data_bytes(memories[1]), 256);
        assert_eq!(compact.effective_data_bytes(memories[2]), 0);
        assert!(matches!(
            compact.sector_plan(memories[0]),
            ValidatedSectorPlan::Sectors(sectors) if sectors.len() == 1
        ));
        assert!(matches!(
            compact.sector_plan(memories[1]),
            ValidatedSectorPlan::Sectors(sectors) if sectors.len() == 16
        ));
        assert!(matches!(
            compact.sector_plan(memories[2]),
            ValidatedSectorPlan::Sectors(sectors) if sectors.is_empty()
        ));
    }

    fn ldgsts_destination_inst(
        transpose_source_pairing: bool,
        source_offset: i64,
    ) -> pb::Instruction {
        let mut instruction = async_payload_inst(32, 16, 16);
        instruction.addresses[0].addrs = (0_i64..32)
            .map(|lane| {
                let source_index = if transpose_source_pairing {
                    (lane % 4) * 8 + lane / 4
                } else {
                    lane
                };
                0x1000 + source_offset + source_index * 16
            })
            .collect();
        instruction.addresses.push(pb::Address {
            data_width: 16,
            compression_format: crate::lanes::FORMAT_LIST_ALL,
            addrs: (0_i64..32).map(|lane| lane * 16).collect(),
            address_space: Some(pb::MemorySpace::Shared as i32),
            ..Default::default()
        });
        instruction
    }

    fn ldgsts_destination_layout_inst(destination_lane_stride: i64) -> pb::Instruction {
        let mut instruction = async_payload_inst(32, 16, 32);
        instruction.addresses.push(pb::Address {
            data_width: 16,
            compression_format: crate::lanes::FORMAT_LIST_ALL,
            addrs: (0_i64..32)
                .map(|lane| lane * destination_lane_stride)
                .collect(),
            address_space: Some(pb::MemorySpace::Shared as i32),
            ..Default::default()
        });
        instruction
    }

    fn ldgsts_boundary_overlap_inst() -> pb::Instruction {
        let mut instruction = async_payload_inst(3, 16, 16);
        instruction.addresses[0].addrs[0] = 0x1008;
        instruction.addresses[0].addrs[1] = 0x1078;
        instruction.addresses[0].addrs[2] = 0x1088;
        instruction.addresses.push(pb::Address {
            data_width: 16,
            compression_format: crate::lanes::FORMAT_LIST_ALL,
            addrs: (0_i64..32)
                .map(|lane| match lane {
                    0 => 0,
                    1 => 128,
                    2 => 16,
                    _ => crate::lanes::LANE_INACTIVE_SENTINEL,
                })
                .collect(),
            address_space: Some(pb::MemorySpace::Shared as i32),
            ..Default::default()
        });
        instruction
    }

    #[test]
    fn ldgsts_source_sector_masks_preserve_boundary_multi_membership() {
        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![ldgsts_boundary_overlap_inst()],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
            "boundary-overlap destination-bearing async copy",
        )
        .unwrap();
        let memory = compact.instructions().next().unwrap().memory().unwrap();

        assert_eq!(compact.effective_data_bytes(memory), 48);
        assert_eq!(
            compact.ldgsts_destination_topology_128b_32x4(memory),
            Some(LdgstsDestinationTopology {
                source_cohorts: 2,
                destination_line_fanout_total: 3,
                destination_line_fanout_max: 2,
                cache_all_bank_wavefronts_total: 3,
                cache_all_bank_wavefronts_max: 2,
                bypass_wavefronts_total: 3,
                bypass_wavefronts_max: 2,
            })
        );
        assert_eq!(
            compact.ldgsts_source_cohorts_128b_32x4(memory),
            Some(
                [
                    LdgstsSourceCohort {
                        payload_bytes: 32,
                        destination_lines: 2,
                        bank_wavefronts: 2,
                        bypass_wavefronts: 2,
                    },
                    LdgstsSourceCohort {
                        payload_bytes: 16,
                        destination_lines: 1,
                        bank_wavefronts: 1,
                        bypass_wavefronts: 1,
                    },
                ]
                .as_slice()
            )
        );
        assert!(matches!(
            compact.sector_plan(memory),
            ValidatedSectorPlan::Sectors(sectors) if sectors == [0x1000 >> 5, 0x1060 >> 5, 0x1080 >> 5]
        ));
        assert_eq!(
            compact.ldgsts_source_sector_cohort_masks_128b_32x4(memory),
            Some([0b01, 0b01, 0b11].as_slice())
        );
    }

    #[test]
    fn ldgsts_destination_projection_is_source_cohort_scoped_and_pairing_exact() {
        for (transpose, source_offset, expected) in [
            (
                false,
                0,
                LdgstsDestinationTopology {
                    source_cohorts: 4,
                    destination_line_fanout_total: 4,
                    destination_line_fanout_max: 1,
                    cache_all_bank_wavefronts_total: 4,
                    cache_all_bank_wavefronts_max: 1,
                    bypass_wavefronts_total: 4,
                    bypass_wavefronts_max: 1,
                },
            ),
            (
                false,
                16,
                LdgstsDestinationTopology {
                    source_cohorts: 4,
                    destination_line_fanout_total: 4,
                    destination_line_fanout_max: 1,
                    cache_all_bank_wavefronts_total: 4,
                    cache_all_bank_wavefronts_max: 1,
                    bypass_wavefronts_total: 4,
                    bypass_wavefronts_max: 1,
                },
            ),
            (
                true,
                0,
                LdgstsDestinationTopology {
                    source_cohorts: 4,
                    destination_line_fanout_total: 16,
                    destination_line_fanout_max: 4,
                    cache_all_bank_wavefronts_total: 16,
                    cache_all_bank_wavefronts_max: 4,
                    bypass_wavefronts_total: 32,
                    bypass_wavefronts_max: 8,
                },
            ),
        ] {
            let instruction = ldgsts_destination_inst(transpose, source_offset);
            let preflight = preflight_memory_access(&instruction, || "LDGSTS pair".to_owned())
                .unwrap()
                .unwrap();
            assert_eq!(
                project_ldgsts_source_cohort_topology_128b_32x4(preflight, &mut Vec::new())
                    .unwrap(),
                expected,
                "transpose={transpose}, source_offset={source_offset}"
            );
            assert_eq!(
                project_ldgsts_source_cohort_wavefronts_128b_32x4(preflight, &mut Vec::new(),)
                    .unwrap(),
                expected.cache_all_bank_wavefronts_total,
                "compatibility wrapper transpose={transpose}, source_offset={source_offset}"
            );
        }

        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![
                    ldgsts_destination_inst(false, 0),
                    ldgsts_destination_inst(true, 0),
                ],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
            "compact destination-bearing async copy",
        )
        .unwrap();
        let memories = compact
            .instructions()
            .map(|instruction| instruction.memory().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(compact.effective_data_bytes(memories[0]), 512);
        assert_eq!(compact.effective_data_bytes(memories[1]), 512);
        assert_eq!(
            compact.ldgsts_destination_topology_128b_32x4(memories[0]),
            Some(LdgstsDestinationTopology {
                source_cohorts: 4,
                destination_line_fanout_total: 4,
                destination_line_fanout_max: 1,
                cache_all_bank_wavefronts_total: 4,
                cache_all_bank_wavefronts_max: 1,
                bypass_wavefronts_total: 4,
                bypass_wavefronts_max: 1,
            })
        );
        assert_eq!(
            compact.ldgsts_destination_topology_128b_32x4(memories[1]),
            Some(LdgstsDestinationTopology {
                source_cohorts: 4,
                destination_line_fanout_total: 16,
                destination_line_fanout_max: 4,
                cache_all_bank_wavefronts_total: 16,
                cache_all_bank_wavefronts_max: 4,
                bypass_wavefronts_total: 32,
                bypass_wavefronts_max: 8,
            })
        );
        assert_eq!(
            compact.ldgsts_destination_wavefronts_128b_32x4(memories[0]),
            Some(4)
        );
        assert_eq!(
            compact.ldgsts_destination_wavefronts_128b_32x4(memories[1]),
            Some(16)
        );
        for memory in memories {
            assert!(matches!(
                compact.sector_plan(memory),
                ValidatedSectorPlan::Sectors(sectors) if sectors.len() == 16
            ));
        }
    }

    #[test]
    fn ldgsts_destination_topology_separates_compact_skew_conflict_and_severe() {
        for (label, destination_lane_stride, expected) in [
            (
                "compact",
                32,
                LdgstsDestinationTopology {
                    source_cohorts: 8,
                    destination_line_fanout_total: 8,
                    destination_line_fanout_max: 1,
                    cache_all_bank_wavefronts_total: 8,
                    cache_all_bank_wavefronts_max: 1,
                    bypass_wavefronts_total: 8,
                    bypass_wavefronts_max: 1,
                },
            ),
            (
                "skew",
                96,
                LdgstsDestinationTopology {
                    source_cohorts: 8,
                    destination_line_fanout_total: 24,
                    destination_line_fanout_max: 3,
                    cache_all_bank_wavefronts_total: 8,
                    cache_all_bank_wavefronts_max: 1,
                    bypass_wavefronts_total: 24,
                    bypass_wavefronts_max: 3,
                },
            ),
            (
                "conflict",
                64,
                LdgstsDestinationTopology {
                    source_cohorts: 8,
                    destination_line_fanout_total: 16,
                    destination_line_fanout_max: 2,
                    cache_all_bank_wavefronts_total: 16,
                    cache_all_bank_wavefronts_max: 2,
                    bypass_wavefronts_total: 32,
                    bypass_wavefronts_max: 4,
                },
            ),
            (
                "severe",
                128,
                LdgstsDestinationTopology {
                    source_cohorts: 8,
                    destination_line_fanout_total: 32,
                    destination_line_fanout_max: 4,
                    cache_all_bank_wavefronts_total: 32,
                    cache_all_bank_wavefronts_max: 4,
                    bypass_wavefronts_total: 32,
                    bypass_wavefronts_max: 4,
                },
            ),
        ] {
            let instruction = ldgsts_destination_layout_inst(destination_lane_stride);
            let preflight = preflight_memory_access(&instruction, || label.to_owned())
                .unwrap()
                .unwrap();
            assert_eq!(
                project_ldgsts_source_cohort_topology_128b_32x4(preflight, &mut Vec::new(),)
                    .unwrap(),
                expected,
                "layout={label}"
            );
            assert_eq!(
                project_ldgsts_source_cohort_wavefronts_128b_32x4(preflight, &mut Vec::new(),)
                    .unwrap(),
                expected.cache_all_bank_wavefronts_total,
                "cache-all compatibility layout={label}"
            );
        }

        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![ldgsts_destination_layout_inst(32)],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
            "uniform compact destination-bearing async copy",
        )
        .unwrap();
        let memory = compact.instructions().next().unwrap().memory().unwrap();
        let expected_cohort = LdgstsSourceCohort {
            payload_bytes: 64,
            destination_lines: 1,
            bank_wavefronts: 1,
            bypass_wavefronts: 1,
        };
        assert_eq!(
            compact.ldgsts_source_cohorts_128b_32x4(memory),
            Some([expected_cohort; 8].as_slice())
        );
        let masks = compact
            .ldgsts_source_sector_cohort_masks_128b_32x4(memory)
            .unwrap();
        assert_eq!(masks.len(), 32);
        for (cohort, sectors) in masks.chunks_exact(4).enumerate() {
            assert_eq!(sectors, [1_u32 << cohort; 4]);
        }
    }

    #[test]
    fn async_copy_projection_keeps_multidimensional_topology_compact() {
        assert!(std::mem::size_of::<AsyncCopyProjection>() <= 32);
        assert!(std::mem::size_of::<CompactLdgstsDestinationTopology>() <= 8);
        assert!(std::mem::size_of::<CompactArenaSpan>() <= 8);
    }

    #[test]
    fn source_only_async_copy_retains_explicit_destination_projection_fallback() {
        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![async_payload_inst(32, 16, 16)],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingSharedMemorySplitV2,
            "source-only async copy",
        )
        .unwrap();
        let memory = compact.instructions().next().unwrap().memory().unwrap();
        assert_eq!(compact.effective_data_bytes(memory), 512);
        assert_eq!(
            compact.ldgsts_destination_wavefronts_128b_32x4(memory),
            None
        );
        assert_eq!(compact.ldgsts_source_cohorts_128b_32x4(memory), None);
        assert_eq!(
            compact.ldgsts_source_sector_cohort_masks_128b_32x4(memory),
            None
        );

        let mut predicated_off = ldgsts_destination_layout_inst(32);
        predicated_off.predicate_mask = Some(0);
        for address in &mut predicated_off.addresses {
            address.addrs.fill(crate::lanes::LANE_INACTIVE_SENTINEL);
        }
        let compact = ValidatedWarp::from_message(
            0,
            pb::Warp {
                id: 0,
                instructions: vec![predicated_off],
                ..Default::default()
            },
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
            "predicated-off destination-bearing async copy",
        )
        .unwrap();
        let memory = compact.instructions().next().unwrap().memory().unwrap();
        assert_eq!(
            compact.ldgsts_destination_topology_128b_32x4(memory),
            Some(LdgstsDestinationTopology::default())
        );
        assert_eq!(
            compact.ldgsts_source_cohorts_128b_32x4(memory),
            Some([].as_slice())
        );
        assert_eq!(
            compact.ldgsts_source_sector_cohort_masks_128b_32x4(memory),
            Some([].as_slice())
        );
        assert!(matches!(
            compact.sector_plan(memory),
            ValidatedSectorPlan::Sectors(sectors) if sectors.is_empty()
        ));
    }

    #[test]
    fn ldgsts_destination_projection_failures_are_typed() {
        let mut mismatched = ldgsts_destination_inst(false, 0);
        mismatched.addresses[1].data_width = 8;
        let preflight = preflight_memory_access(&mismatched, || "width mismatch".to_owned())
            .unwrap()
            .unwrap();
        assert!(matches!(
            project_ldgsts_source_cohort_wavefronts_128b_32x4(preflight, &mut Vec::new()),
            Err(LdgstsDestinationProjectionError::OperandWidthMismatch {
                source_bytes: 16,
                destination_bytes: 8,
            })
        ));

        let mut too_wide = ldgsts_destination_inst(false, 0);
        too_wide.active_mask = Some(1);
        too_wide.predicate_mask = Some(1);
        too_wide.data_width = Some(129);
        for address in &mut too_wide.addresses {
            address.data_width = 129;
            address.addrs[1..].fill(crate::lanes::LANE_INACTIVE_SENTINEL);
        }
        let preflight = preflight_memory_access(&too_wide, || "wide transfer".to_owned())
            .unwrap()
            .unwrap();
        assert!(matches!(
            project_ldgsts_source_cohort_wavefronts_128b_32x4(preflight, &mut Vec::new()),
            Err(
                LdgstsDestinationProjectionError::SourceTransferExceedsCohort {
                    lane: 0,
                    width_bytes: 129,
                }
            )
        ));

        let mut overflow = ldgsts_destination_inst(false, 0);
        overflow.active_mask = Some(1);
        overflow.predicate_mask = Some(1);
        for address in &mut overflow.addresses {
            address.addrs[1..].fill(crate::lanes::LANE_INACTIVE_SENTINEL);
        }
        overflow.addresses[0].addrs[0] = -2;
        let preflight = preflight_memory_access(&overflow, || "source overflow".to_owned())
            .unwrap()
            .unwrap();
        assert!(matches!(
            project_ldgsts_source_cohort_wavefronts_128b_32x4(
                preflight,
                &mut Vec::new()
            ),
            Err(LdgstsDestinationProjectionError::SourceIntervalOverflow {
                lane: 0,
                base,
                width_bytes: 16,
            }) if base == u64::MAX - 1
        ));
    }

    #[test]
    fn destination_v3_requires_exact_ordered_async_copy_pair() {
        let valid = ldgsts_destination_inst(false, 0);
        let preflight = preflight_memory_access(&valid, || "valid v3".to_owned())
            .unwrap()
            .unwrap();
        validate_address_carrying_shape(
            preflight,
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
            &|| "valid v3".to_owned(),
        )
        .unwrap();

        let source_only = async_payload_inst(32, 16, 16);
        let preflight = preflight_memory_access(&source_only, || "source only".to_owned())
            .unwrap()
            .unwrap();
        assert!(matches!(
            validate_address_carrying_shape(
                preflight,
                MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
                &|| "source only".to_owned(),
            ),
            Err(TraceError::MemoryPayloadShapeInvalid { got: 1, .. })
        ));

        let mut reversed = valid;
        reversed.addresses.reverse();
        let preflight = preflight_memory_access(&reversed, || "reversed pair".to_owned())
            .unwrap()
            .unwrap();
        assert!(matches!(
            validate_address_carrying_shape(
                preflight,
                MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
                &|| "reversed pair".to_owned(),
            ),
            Err(TraceError::MemoryPayloadShapeInvalid { .. })
        ));
    }

    #[test]
    fn shared_bank_projection_word_limit_is_typed() {
        let mut inst = shared_projection_inst(1, 4_097);
        inst.active_mask = Some(1);
        inst.predicate_mask = Some(1);
        inst.addresses[0].addrs[1..].fill(crate::lanes::LANE_INACTIVE_SENTINEL);
        let preflight = preflight_memory_access(&inst, || "wide shared".to_owned())
            .unwrap()
            .unwrap();
        assert!(matches!(
            project_shared_bank_wavefronts_32x4(preflight, &mut Vec::new()),
            Err(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words: 1_025,
                limit: MAX_SHARED_WORDS_PER_LANE,
            })
        ));
    }

    fn mem_inst(_bx: i32) -> pb::Instruction {
        pb::Instruction {
            pc: 16,
            active_mask: Some(0b11),
            predicate_mask: Some(0b11),
            function_unique_id: 1,
            addresses: vec![pb::Address {
                data_width: 4,
                udesc_value: 0,
                compression_format: crate::lanes::FORMAT_BASE_STRIDE,
                base_address: 0x7000_0000,
                stride: 4,
                addrs: vec![],
                // Valid global identity: id 1 resolves to the sidecar allocation
                // write_synthetic_trace emits (base 0x7000_0000, 4 KiB), which
                // contains the base_stride lane addresses. The reader identity
                // gate requires this for a GLOBAL active record.
                allocation_id: Some(1),
                address_space: None,
            }],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(4),
            warpid_tb: Some(0),
            warpid_sm: Some(0),
            sm_id: Some(0),
            vpc: Some(16),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            instruction_cache_op: Some(pb::InstructionCacheOp::LdgECa as i32),
        }
    }

    fn write_synthetic_trace(dir: &Path, grid_x: i32) {
        let kernel = pb::Kernel {
            id: 1,
            name: "synthetic_kernel".into(),
            function_unique_id: 1,
            size_shared_memory: 0,
            number_of_registers: 32,
            shared_memory_base_address: 0,
            local_memory_base_address: 0,
            grid_dim: Some(pb::Dim3d {
                x: grid_x,
                y: 1,
                z: 1,
            }),
            block_dim: Some(pb::Dim3d { x: 64, y: 1, z: 1 }),
            cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
            shared_memory_static_bytes: None,
            shared_memory_dynamic_bytes: None,
        };
        let stream = pb::CudaStream {
            id: 0,
            kernels: vec![kernel],
            ordered_cuda_events: vec![],
        };
        let device = pb::GpuDevice {
            id: 0,
            streams: [(0i64, stream)].into_iter().collect(),
        };
        let trace = pb::Trace {
            name: "synthetic".into(),
            binary_version: 1,
            nvbit_version: "1.8".into(),
            accelsim_version: 0,
            is_gathered_registers_values: false,
            gpu_device: [(0i64, device)].into_iter().collect(),
            trace_contract_id: String::new(),
            memory_payload_classes: vec![],
        };
        std::fs::write(dir.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();

        // Allocations sidecar covering mem_inst's base_stride addresses so the
        // reader-side identity gate admits its GLOBAL records (allocation_id 1).
        let extra = dir.join("extra_info");
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(
            extra.join("allocations.json"),
            r#"[{"allocation_id": 1, "context_id": 1, "device_id": 0, "base_address": "0x70000000", "size_bytes": 4096, "alive": true}]"#,
        )
        .unwrap();

        let tb_dir = dir.join("threadblocks/device_0/stream_0/kernel_1");
        std::fs::create_dir_all(&tb_dir).unwrap();
        for bx in 0..grid_x {
            let inst_alu = pb::Instruction {
                pc: 0,
                active_mask: Some(0b11),
                predicate_mask: Some(0b11),
                function_unique_id: 1,
                warpid_tb: Some(0),
                vpc: Some(0),
                ..Default::default()
            };
            let warp = pb::Warp {
                id: 0,
                instructions: vec![inst_alu, mem_inst(bx)],
            };
            let tb = pb::Threadblock {
                block_id: Some(pb::Dim3d { x: bx, y: 0, z: 0 }),
                warps: [(0i32, warp)].into_iter().collect(),
            };
            std::fs::write(
                tb_dir.join(format!("d_0_s_0_k_1_{bx},0,0.pb")),
                tb.encode_to_vec(),
            )
            .unwrap();
        }
    }

    fn read_synthetic_root(dir: &Path) -> pb::Trace {
        let raw = std::fs::read(dir.join("dynamic_trace.pb")).unwrap();
        pb::Trace::decode(raw.as_slice()).unwrap()
    }

    fn write_synthetic_root(dir: &Path, trace: &pb::Trace) {
        std::fs::write(dir.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();
    }

    fn synthetic_threadblock_path(dir: &Path, kernel_id: i32) -> PathBuf {
        dir.join(format!(
            "threadblocks/device_0/stream_0/kernel_{kernel_id}/d_0_s_0_k_{kernel_id}_0,0,0.pb"
        ))
    }

    #[cfg(feature = "campaign")]
    fn write_nonselected_second_launch(dir: &Path) -> PathBuf {
        let mut trace = read_synthetic_root(dir);
        let stream = trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap();
        let mut second = stream.kernels[0].clone();
        second.id = 2;
        second.function_unique_id = 2;
        stream.kernels.push(second);
        write_synthetic_root(dir, &trace);

        let first_path = synthetic_threadblock_path(dir, 1);
        let raw = std::fs::read(first_path).unwrap();
        let mut threadblock = pb::Threadblock::decode(raw.as_slice()).unwrap();
        for instruction in &mut threadblock.warps.get_mut(&0).unwrap().instructions {
            instruction.function_unique_id = 2;
        }
        let second_path = synthetic_threadblock_path(dir, 2);
        std::fs::create_dir_all(second_path.parent().unwrap()).unwrap();
        std::fs::write(&second_path, threadblock.encode_to_vec()).unwrap();
        second_path
    }

    #[test]
    #[cfg(feature = "campaign")]
    fn manifest_snapshot_root_allocation_failure_retains_artifact_role() {
        let path = PathBuf::from("trace/dynamic_trace.pb");
        assert!(matches!(
            map_manifest_snapshot_read_error(
                "dynamic_trace_pb",
                AnchoredReadError::AllocationFailed {
                    path: path.clone(),
                    bytes: u64::MAX,
                },
            ),
            TraceError::SnapshotArtifactRead {
                artifact: "dynamic_trace_pb",
                source: AnchoredReadError::AllocationFailed {
                    path: actual_path,
                    bytes: u64::MAX,
                },
            } if actual_path == path
        ));
        assert!(matches!(
            map_snapshot_read_error(
                "dynamic_trace_pb",
                AnchoredReadError::AllocationFailed {
                    path: path.clone(),
                    bytes: u64::MAX,
                },
            ),
            TraceError::AnchoredSnapshot(AnchoredReadError::AllocationFailed {
                path: actual_path,
                bytes: u64::MAX,
            }) if actual_path == path
        ));
    }

    #[test]
    fn root_map_and_function_identities_fail_closed() {
        for mutation in ["device", "stream", "function"] {
            let tmp = tempfile::tempdir().unwrap();
            write_synthetic_trace(tmp.path(), 1);
            let mut trace = read_synthetic_root(tmp.path());
            match mutation {
                "device" => trace.gpu_device.get_mut(&0).unwrap().id = 1,
                "stream" => {
                    trace
                        .gpu_device
                        .get_mut(&0)
                        .unwrap()
                        .streams
                        .get_mut(&0)
                        .unwrap()
                        .id = 1;
                }
                "function" => {
                    trace
                        .gpu_device
                        .get_mut(&0)
                        .unwrap()
                        .streams
                        .get_mut(&0)
                        .unwrap()
                        .kernels[0]
                        .function_unique_id = 0;
                }
                _ => unreachable!(),
            }
            write_synthetic_root(tmp.path(), &trace);
            let error = TraceDir::open_direct(tmp.path()).err().unwrap();
            let expected = if mutation == "function" {
                "function_unique_id_invalid"
            } else {
                "trace_launch_identity_mismatch"
            };
            assert_eq!(
                error.typed_reason(),
                Some(expected),
                "{mutation}: {error:?}"
            );
        }
    }

    #[test]
    fn duplicate_launch_identity_is_typed() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        let mut trace = read_synthetic_root(tmp.path());
        let stream = trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap();
        stream.kernels.push(stream.kernels[0].clone());
        write_synthetic_root(tmp.path(), &trace);
        let error = TraceDir::open_direct(tmp.path()).err().unwrap();
        assert_eq!(error.typed_reason(), Some("trace_launch_identity_mismatch"));
    }

    #[test]
    fn every_payload_is_validated_and_joins_one_root_launch() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        let mut trace = read_synthetic_root(tmp.path());
        let stream = trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap();
        let mut second = stream.kernels[0].clone();
        second.id = 2;
        second.function_unique_id = 2;
        stream.kernels.push(second);
        write_synthetic_root(tmp.path(), &trace);

        let first_path = synthetic_threadblock_path(tmp.path(), 1);
        let raw = std::fs::read(&first_path).unwrap();
        let mut threadblock = pb::Threadblock::decode(raw.as_slice()).unwrap();
        let instructions = &mut threadblock.warps.get_mut(&0).unwrap().instructions;
        for instruction in instructions.iter_mut() {
            instruction.function_unique_id = 2;
        }
        instructions[0].warpid_tb = None;
        let second_path = synthetic_threadblock_path(tmp.path(), 2);
        std::fs::create_dir_all(second_path.parent().unwrap()).unwrap();
        std::fs::write(&second_path, threadblock.encode_to_vec()).unwrap();
        let error = TraceDir::open_direct(tmp.path()).err().unwrap();
        assert_eq!(error.typed_reason(), Some("warpid_tb_missing"));

        std::fs::remove_file(second_path).unwrap();
        let error = TraceDir::open_direct(tmp.path()).err().unwrap();
        assert_eq!(error.typed_reason(), Some("trace_launch_payload_missing"));
    }

    #[test]
    #[cfg(feature = "campaign")]
    fn full_snapshot_rejects_warp_map_identity_in_nonselected_launch() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        let second_path = write_nonselected_second_launch(tmp.path());
        let raw = std::fs::read(&second_path).unwrap();
        let mut threadblock = pb::Threadblock::decode(raw.as_slice()).unwrap();
        let warp = threadblock.warps.remove(&0).unwrap();
        assert_eq!(warp.id, 0);
        threadblock.warps.insert(1, warp);
        std::fs::write(&second_path, threadblock.encode_to_vec()).unwrap();

        let snapshot = TraceDirSnapshot::capture(tmp.path()).unwrap();
        let error = TraceDir::from_snapshot(snapshot).err().unwrap();
        assert!(matches!(
            &error,
            TraceError::WarpMapKeyMismatch {
                threadblock,
                map_key: 1,
                warp_id: 0,
            } if threadblock == second_path.to_string_lossy().as_ref()
        ));
        assert_eq!(error.typed_reason(), Some("warp_map_key_mismatch"));
    }

    #[test]
    #[cfg(feature = "campaign")]
    fn full_snapshot_rejects_instruction_warp_identity_in_nonselected_launch() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        let second_path = write_nonselected_second_launch(tmp.path());
        let raw = std::fs::read(&second_path).unwrap();
        let mut threadblock = pb::Threadblock::decode(raw.as_slice()).unwrap();
        threadblock.warps.get_mut(&0).unwrap().instructions[1].warpid_tb = Some(1);
        std::fs::write(&second_path, threadblock.encode_to_vec()).unwrap();

        let snapshot = TraceDirSnapshot::capture(tmp.path()).unwrap();
        let error = TraceDir::from_snapshot(snapshot).err().unwrap();
        assert!(matches!(
            &error,
            TraceError::WarpidTbMismatch {
                record,
                expected: 0,
                actual: 1,
            } if record.contains("kernel_2") && record.ends_with("warp=0 inst=1")
        ));
        assert_eq!(error.typed_reason(), Some("warpid_tb_mismatch"));
    }

    #[test]
    fn orphan_payload_and_function_join_are_typed() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        let first_path = synthetic_threadblock_path(tmp.path(), 1);
        let orphan_path = synthetic_threadblock_path(tmp.path(), 9);
        std::fs::create_dir_all(orphan_path.parent().unwrap()).unwrap();
        std::fs::copy(&first_path, &orphan_path).unwrap();
        let error = TraceDir::open_direct(tmp.path()).err().unwrap();
        assert_eq!(error.typed_reason(), Some("trace_payload_orphan"));

        std::fs::remove_file(orphan_path).unwrap();
        let raw = std::fs::read(&first_path).unwrap();
        let mut threadblock = pb::Threadblock::decode(raw.as_slice()).unwrap();
        threadblock.warps.get_mut(&0).unwrap().instructions[0].function_unique_id = 2;
        std::fs::write(first_path, threadblock.encode_to_vec()).unwrap();
        let error = TraceDir::open_direct(tmp.path()).err().unwrap();
        assert_eq!(error.typed_reason(), Some("function_unique_id_mismatch"));
    }

    #[test]
    fn stats_parser_enforces_uint64_rows_and_checked_totals() {
        let path = Path::new("stats.csv");
        let maximum = format!(
            "total_insts,total_reported_insts\n{},{}\n",
            u64::MAX,
            u64::MAX
        );
        assert_eq!(
            parse_trace_stats(path, maximum.as_bytes()).unwrap(),
            TraceStats {
                row_count: 1,
                expected_record_count: u64::MAX,
                admitted_record_count: u64::MAX,
            }
        );

        let too_wide = b"total_insts,total_reported_insts\n18446744073709551616,1\n";
        assert!(matches!(
            parse_trace_stats(path, too_wide),
            Err(TraceError::StatsSchemaInvalid { .. })
        ));
        let overflow = format!("total_insts,total_reported_insts\n{},1\n1,1\n", u64::MAX);
        let error = parse_trace_stats(path, overflow.as_bytes()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("trace_counter_overflow"));
    }

    #[test]
    fn synthetic_trace_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 2);

        let td = TraceDir::open_direct(tmp.path()).unwrap();
        let kernels: Vec<_> = td.kernels().collect();
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].kernel.name, "synthetic_kernel");

        let paths = td.threadblock_paths(&kernels[0]).unwrap();
        assert_eq!(paths.len(), 2);

        let mut mem_insts = 0usize;
        let mut alu_insts = 0usize;
        for p in &paths {
            let tb = read_threadblock(p).unwrap();
            for (warp_id, warp) in &tb.warps {
                for (i, inst) in warp.instructions.iter().enumerate() {
                    let rec = || format!("{} warp={warp_id} inst={i}", p.display());
                    match memory_access(inst, rec).unwrap() {
                        Some(acc) => {
                            mem_insts += 1;
                            assert_eq!(acc.op_class, pb::MemoryOpClass::Global);
                            assert_eq!(acc.accesses.len(), 1);
                            assert_eq!(acc.accesses[0].lane_vector[0], 0x7000_0000);
                            assert_eq!(acc.accesses[0].lane_vector[1], 0x7000_0004);
                            assert_eq!(acc.accesses[0].data_width_bytes, 4);
                        }
                        None => alu_insts += 1,
                    }
                }
            }
        }
        assert_eq!(mem_insts, 2);
        assert_eq!(alu_insts, 2);
    }

    #[test]
    fn compact_threadblock_is_borrowed_from_the_immutable_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);

        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let kernel = trace.kernels().next().unwrap();
        let path = trace.threadblock_paths(&kernel).unwrap().pop().unwrap();
        let first = trace.validated_threadblock(&path).unwrap();
        let first_warp = &first.warps()[0];
        let first_pc = first_warp
            .instructions()
            .next()
            .expect("synthetic warp is nonempty")
            .pc();
        let memory = first_warp
            .instructions()
            .find_map(|instruction| instruction.memory())
            .expect("synthetic warp carries memory");
        assert_eq!(memory.replay_meta().op_class, pb::MemoryOpClass::Global);
        assert_eq!(
            first_warp.memory_route(memory),
            Ok(ValidatedMemoryRoute::Global)
        );
        assert!(matches!(
            first_warp.sector_plan(memory),
            ValidatedSectorPlan::Sectors(sectors)
                if sectors == [0x7000_0000_i64 >> 5]
        ));
        assert!(
            first_warp
                .instructions()
                .any(|instruction| instruction.memory().is_none())
        );
        assert!(std::mem::size_of::<ValidatedInstruction>() <= 20);
        assert!(std::mem::size_of::<ValidatedMemoryInstruction>() <= 24);

        let mut replacement = pb::Threadblock::decode(std::fs::read(&path).unwrap().as_slice())
            .expect("synthetic payload decodes");
        replacement.warps.get_mut(&0).unwrap().instructions[0].pc = first_pc + 4;
        std::fs::write(&path, replacement.encode_to_vec()).unwrap();

        let second = trace.validated_threadblock(&path).unwrap();
        assert_eq!(first.launch_index(), 0);
        assert!(std::ptr::eq(first, second));
        assert_eq!(
            second.warps()[0]
                .instructions()
                .next()
                .expect("synthetic warp remains nonempty")
                .pc(),
            first_pc
        );
        assert_eq!(
            trace.read_threadblock(&path).unwrap().warps[&0].instructions[0].pc,
            first_pc
        );
        assert_eq!(trace.validation_summary().threadblock_count(), 1);
    }

    #[test]
    fn compact_direct_loader_releases_only_raw_payload_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 2);

        let raw = TraceDir::open_direct(tmp.path()).unwrap();
        let compact = TraceDir::open_direct_compact(tmp.path()).unwrap();
        let streaming = TraceDir::open_direct_streaming(tmp.path()).unwrap();
        assert_eq!(compact.validation_summary(), raw.validation_summary());
        assert_eq!(streaming.validation_summary(), raw.validation_summary());
        assert_eq!(
            compact.memory_payload_contract(),
            raw.memory_payload_contract()
        );
        assert_eq!(
            streaming.memory_payload_contract(),
            raw.memory_payload_contract()
        );
        let raw_kernel = raw.kernels().next().unwrap();
        let compact_kernel = compact.kernels().next().unwrap();
        let streaming_kernel = streaming.kernels().next().unwrap();
        let raw_paths = raw.threadblock_paths(&raw_kernel).unwrap();
        let compact_paths = compact.threadblock_paths(&compact_kernel).unwrap();
        let streaming_paths = streaming.threadblock_paths(&streaming_kernel).unwrap();
        assert_eq!(compact_paths, raw_paths);
        assert_eq!(streaming_paths, raw_paths);
        assert_eq!(
            compact
                .validated_threadblocks_in_launch_order(&compact_kernel)
                .unwrap()
                .into_iter()
                .map(|(index, _)| index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        for path in raw_paths {
            let raw_block = raw.validated_threadblock(&path).unwrap();
            let compact_block = compact.validated_threadblock(&path).unwrap();
            let streaming_block = streaming.validated_threadblock(&path).unwrap();
            assert_eq!(compact_block.launch_index(), raw_block.launch_index());
            assert_eq!(streaming_block.launch_index(), raw_block.launch_index());
            let instruction_counts = |block: &ValidatedThreadblock| {
                block
                    .warps()
                    .iter()
                    .map(|warp| warp.instructions().count())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                instruction_counts(compact_block),
                instruction_counts(raw_block)
            );
            assert_eq!(
                instruction_counts(streaming_block),
                instruction_counts(raw_block)
            );
            assert!(raw.read_threadblock(&path).is_ok());
            assert!(matches!(
                compact.read_threadblock(&path),
                Err(TraceError::RawThreadblockPayloadUnavailable { path: missing })
                    if missing == path
            ));
            assert!(matches!(
                streaming.read_threadblock(&path),
                Err(TraceError::RawThreadblockPayloadUnavailable { path: missing })
                    if missing == path
            ));
        }
        let foreign = tmp.path().join("threadblocks/not-in-snapshot.pb");
        assert!(matches!(
            compact.read_threadblock(&foreign),
            Err(TraceError::SamplingPayloadSnapshotMissing { path }) if path == foreign
        ));
    }

    #[test]
    fn direct_loader_ignores_campaign_sampling_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        std::fs::write(
            tmp.path().join("trace_sampling.json"),
            b"this is deliberately not campaign JSON",
        )
        .unwrap();

        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        assert_eq!(trace.validation_summary().launch_count(), 1);
        assert_eq!(trace.validation_summary().threadblock_count(), 1);
    }

    #[test]
    fn kernel_locator_is_bound_to_its_trace_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        write_synthetic_trace(first.path(), 1);
        write_synthetic_trace(second.path(), 1);

        let first_trace = TraceDir::open_direct(first.path()).unwrap();
        let second_trace = TraceDir::open_direct(second.path()).unwrap();
        let foreign = first_trace.kernels().next().unwrap();
        let error = second_trace.threadblock_paths(&foreign).unwrap_err();
        assert!(matches!(
            &error,
            TraceError::KernelLocatorForeign {
                device_id: 0,
                stream_id: 0,
                kernel_id: 1,
                ..
            }
        ));
        assert_eq!(error.typed_reason(), Some("kernel_locator_foreign"));
    }

    /// Lexicographic filename order is NOT launch order once a coordinate
    /// hits double digits; read_threadblocks_in_launch_order must return
    /// flattened-block_id order regardless.
    #[test]
    fn launch_order_is_numeric_not_lexicographic() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 12);
        let td = TraceDir::open_direct(tmp.path()).unwrap();
        let kernels: Vec<_> = td.kernels().collect();

        // Filename order goes 0, 1, 10, 11, 2, ...
        let paths = td.threadblock_paths(&kernels[0]).unwrap();
        let names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            names[2].contains("_10,"),
            "lexicographic order expected in raw paths"
        );

        let ordered = td.read_threadblocks_in_launch_order(&kernels[0]).unwrap();
        let indices: Vec<i64> = ordered.iter().map(|(i, _)| *i).collect();
        assert_eq!(indices, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn missing_block_id_is_typed() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        // Overwrite the block file with one lacking block_id.
        let tb = pb::Threadblock {
            block_id: None,
            warps: [(
                0,
                pb::Warp {
                    id: 0,
                    instructions: vec![pb::Instruction {
                        active_mask: Some(0),
                        predicate_mask: Some(0),
                        function_unique_id: 1,
                        warpid_tb: Some(0),
                        vpc: Some(0),
                        ..Default::default()
                    }],
                },
            )]
            .into_iter()
            .collect(),
        };
        std::fs::write(
            tmp.path()
                .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
            tb.encode_to_vec(),
        )
        .unwrap();
        assert!(matches!(
            TraceDir::open_direct(tmp.path()),
            Err(TraceError::BlockIdMissing { .. })
        ));
    }

    #[test]
    fn out_of_grid_block_id_is_typed() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 1);
        let tb = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 5, y: 0, z: 0 }),
            warps: Default::default(),
        };
        std::fs::write(
            tmp.path()
                .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
            tb.encode_to_vec(),
        )
        .unwrap();
        assert!(matches!(
            TraceDir::open_direct(tmp.path()),
            Err(TraceError::BlockIdOutOfRange {
                index: 5,
                grid: 1,
                ..
            })
        ));
    }

    #[test]
    fn per_axis_out_of_grid_block_id_is_typed() {
        // Grid (2,2,1): CTA (2,0,0) flattens to index 2 (inside [0,4)) but x=2 is
        // outside gx=2 — it must NOT alias into a valid launch index.
        let kernel = pb::Kernel {
            id: 1,
            name: "k".into(),
            function_unique_id: 1,
            size_shared_memory: 0,
            number_of_registers: 0,
            shared_memory_base_address: 0,
            local_memory_base_address: 0,
            grid_dim: Some(pb::Dim3d { x: 2, y: 2, z: 1 }),
            block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
            cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
            shared_memory_static_bytes: None,
            shared_memory_dynamic_bytes: None,
        };
        let tb = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 2, y: 0, z: 0 }),
            warps: Default::default(),
        };
        let err = block_launch_index(&kernel, &tb, Path::new("t.pb")).unwrap_err();
        assert!(
            matches!(
                err,
                TraceError::BlockIdOutOfRange {
                    index: 2,
                    grid: 4,
                    ..
                }
            ),
            "expected BlockIdOutOfRange, got {err:?}"
        );
    }

    #[test]
    fn out_of_axis_precedes_flattening_overflow() {
        let kernel = pb::Kernel {
            id: 8,
            grid_dim: Some(pb::Dim3d {
                x: i32::MAX,
                y: i32::MAX,
                z: 1,
            }),
            ..Default::default()
        };
        let tb = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 0, y: 0, z: 3 }),
            warps: Default::default(),
        };
        let expected_index = 3_i128 * i128::from(i32::MAX) * i128::from(i32::MAX);
        assert!(matches!(
            block_launch_index(&kernel, &tb, Path::new("t.pb")),
            Err(TraceError::BlockIdOutOfRange {
                index,
                grid,
                ..
            }) if index == expected_index && grid == i64::from(i32::MAX) * i64::from(i32::MAX)
        ));
    }

    #[test]
    fn grid_volume_overflow_is_typed() {
        let kernel = pb::Kernel {
            id: 9,
            grid_dim: Some(pb::Dim3d {
                x: i32::MAX,
                y: i32::MAX,
                z: i32::MAX,
            }),
            ..Default::default()
        };
        let tb = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
            warps: Default::default(),
        };
        assert!(matches!(
            block_launch_index(&kernel, &tb, Path::new("t.pb")),
            Err(TraceError::GridArithmeticOverflow {
                kernel_id: 9,
                operation: "grid volume",
            })
        ));
    }

    #[test]
    fn missing_threadblock_dir_is_typed_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 2);
        std::fs::remove_dir_all(tmp.path().join("threadblocks")).unwrap();

        assert!(matches!(
            TraceDir::open_direct(tmp.path()),
            Err(TraceError::TraceThreadblocksEmpty { .. })
        ));
    }

    #[test]
    fn unspecified_op_class_rejected() {
        let mut inst = mem_inst(0);
        inst.memory_op_class = Some(pb::MemoryOpClass::Unspecified as i32);
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::MemoryOpClassUnspecified { .. })
        ));
    }

    /// A wire value beyond the committed schema is SCHEMA SKEW, not an
    /// UNSPECIFIED-contract violation — the diagnostic must say so.
    #[test]
    fn unknown_op_class_value_is_schema_skew() {
        let mut inst = mem_inst(0);
        inst.memory_op_class = Some(99);
        match memory_access(&inst, || "t".into()) {
            Err(TraceError::UnknownEnumValue { field, raw, .. }) => {
                assert_eq!(field, "memory_op_class");
                assert_eq!(raw, 99);
            }
            other => panic!("expected UnknownEnumValue, got {other:?}"),
        }
    }

    #[test]
    fn unknown_cache_op_value_is_schema_skew() {
        let mut inst = mem_inst(0);
        inst.instruction_cache_op = Some(42);
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::UnknownEnumValue {
                field: "instruction_cache_op",
                raw: 42,
                ..
            })
        ));
    }

    #[test]
    fn unspecified_cache_op_rejected() {
        let mut inst = mem_inst(0);
        inst.instruction_cache_op = Some(pb::InstructionCacheOp::Unspecified as i32);
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::InstructionCacheOpUnspecified { .. })
        ));
    }

    #[test]
    fn masks_required_on_every_instruction() {
        let mut inst = mem_inst(0);
        inst.active_mask = None;
        inst.predicate_mask = None;
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::ActiveMaskMissing { .. })
        ));

        let mut inst = mem_inst(0);
        inst.predicate_mask = None;
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::PredicateMaskMissing { .. })
        ));
    }

    /// Canonical memory-ness: class present with NO addresses is still a
    /// memory instruction (predicated-off case), not an ALU record.
    #[test]
    fn class_present_without_addresses_is_memory() {
        let mut inst = mem_inst(0);
        inst.addresses.clear();
        inst.active_mask = Some(0);
        inst.predicate_mask = Some(0);
        let acc = memory_access(&inst, || "t".into())
            .unwrap()
            .expect("memory identity");
        assert_eq!(acc.and_mask, 0);
        assert!(acc.accesses.is_empty());
    }

    #[test]
    fn additive_memory_metadata_rejected_on_nonmemory() {
        let universal = pb::Instruction {
            pc: 0,
            active_mask: Some(0),
            predicate_mask: Some(0),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            ..Default::default()
        };
        let mut width_only = universal.clone();
        width_only.data_width = Some(4);
        assert!(
            memory_access(&width_only, || "width-only".into())
                .unwrap()
                .is_none()
        );

        let mut kind_only = universal.clone();
        kind_only.data_width = Some(4);
        kind_only.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        assert!(matches!(
            memory_access(&kind_only, || "kind-only".into()),
            Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                field: "memory_op_access_kind",
                ..
            })
        ));

        let mut cache_op_only = universal;
        cache_op_only.data_width = Some(4);
        cache_op_only.instruction_cache_op = Some(pb::InstructionCacheOp::LdgECa as i32);
        assert!(matches!(
            memory_access(&cache_op_only, || "cache-op-only".into()),
            Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                field: "instruction_cache_op",
                ..
            })
        ));
    }

    #[test]
    fn data_width_contract_enforced() {
        let mut inst = mem_inst(0);
        inst.data_width = None;
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::DataWidthMissing { .. })
        ));
        let mut inst = mem_inst(0);
        inst.data_width = Some(0);
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::DataWidthInvalid { got: 0, .. })
        ));
    }

    #[test]
    fn universal_instruction_contract_matches_python_reasons() {
        let mut inst = mem_inst(0);
        inst.warpid_tb = None;
        let error = memory_access(&inst, || "t".into()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("warpid_tb_missing"));

        let mut inst = mem_inst(0);
        inst.function_unique_id = 0;
        let error = memory_access(&inst, || "t".into()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("function_unique_id_invalid"));

        let mut inst = mem_inst(0);
        inst.vpc = None;
        let error = memory_access(&inst, || "t".into()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("vpc_missing"));

        let mut inst = mem_inst(0);
        inst.vpc = Some((1_u64 << 32) | 17);
        let error = memory_access(&inst, || "t".into()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("vpc_inconsistent_with_pc"));

        let mut inst = mem_inst(0);
        inst.vpc = Some((1_u64 << 32) | 16);
        assert!(memory_access(&inst, || "t".into()).is_ok());
    }

    #[test]
    fn lane_mask_and_explicit_unspecified_space_match_python_reasons() {
        let mut inst = mem_inst(0);
        inst.active_mask = Some(0);
        inst.predicate_mask = Some(0);
        inst.addresses[0].compression_format = lanes::FORMAT_BASE_DELTA;
        inst.addresses[0].addrs = vec![4];
        let error = memory_access(&inst, || "t".into()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("lane_mask_inconsistent"));

        let mut inst = mem_inst(0);
        inst.memory_op_class = Some(pb::MemoryOpClass::Atomic as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Unspecified as i32);
        let error = memory_access(&inst, || "t".into()).unwrap_err();
        assert_eq!(error.typed_reason(), Some("memory_space_unspecified"));
    }

    /// Per-address width wins over the instruction-level width when the
    /// record carries its own (canonical l1_production_helper priority).
    #[test]
    fn per_address_width_priority() {
        let mut inst = mem_inst(0);
        inst.addresses[0].data_width = 16;
        inst.data_width = Some(4);
        let acc = memory_access(&inst, || "t".into()).unwrap().unwrap();
        assert_eq!(acc.accesses[0].data_width_bytes, 16);

        let mut inst = mem_inst(0);
        inst.addresses[0].data_width = 0; // absent at record level → inst width
        let acc = memory_access(&inst, || "t".into()).unwrap().unwrap();
        assert_eq!(acc.accesses[0].data_width_bytes, 4);
    }

    /// The allocating wrapper and the streaming core must agree exactly —
    /// the wrapper is implemented ON the core, so this guards the meta
    /// mapping and record ordering.
    #[test]
    fn streaming_and_collecting_forms_agree() {
        let mut inst = mem_inst(0);
        inst.addresses.push(pb::Address {
            data_width: 16,
            udesc_value: 0,
            compression_format: crate::lanes::FORMAT_BASE_DELTA,
            base_address: 0x9000,
            stride: 0,
            addrs: vec![64],
            allocation_id: None,
            address_space: None,
        });
        let collected = memory_access(&inst, || "t".into()).unwrap().unwrap();

        let mut streamed: Vec<AddressAccess> = Vec::new();
        let meta = visit_memory_access(&inst, || "t".into(), |a| streamed.push(*a))
            .unwrap()
            .unwrap();
        assert_eq!(meta.op_class, collected.op_class);
        assert_eq!(meta.access_kind, collected.access_kind);
        assert_eq!(meta.cache_op, collected.cache_op);
        assert_eq!(meta.and_mask, collected.and_mask);
        assert_eq!(streamed.len(), collected.accesses.len());
        for (s_rec, c_rec) in streamed.iter().zip(&collected.accesses) {
            assert_eq!(s_rec.lane_vector, c_rec.lane_vector);
            assert_eq!(s_rec.data_width_bytes, c_rec.data_width_bytes);
        }
        assert_eq!(streamed[1].data_width_bytes, 16);
        assert_eq!(streamed[1].lane_vector[0], 0x9000);
        assert_eq!(streamed[1].lane_vector[1], 0x9040);
    }

    #[test]
    fn later_lane_error_emits_no_callback_prefix() {
        let mut inst = mem_inst(0);
        inst.addresses.push(pb::Address {
            data_width: 4,
            compression_format: crate::lanes::FORMAT_LIST_ALL,
            addrs: Vec::new(),
            ..Default::default()
        });
        let mut callbacks = 0;
        let err = visit_memory_access(&inst, || "t".into(), |_| callbacks += 1).unwrap_err();
        assert!(matches!(
            err,
            TraceError::Lane {
                source: LaneError::ListAllLengthInvalid { got: 0 },
                ..
            }
        ));
        assert_eq!(callbacks, 0);
    }

    #[test]
    fn later_unknown_address_space_emits_no_callback_prefix() {
        let mut inst = mem_inst(0);
        let mut invalid = inst.addresses[0].clone();
        invalid.address_space = Some(99);
        inst.addresses.push(invalid);
        let mut callbacks = 0;
        let err = visit_memory_access(&inst, || "t".into(), |_| callbacks += 1).unwrap_err();
        assert!(matches!(
            err,
            TraceError::UnknownEnumValue {
                field: "address_space",
                raw: 99,
                ..
            }
        ));
        assert_eq!(callbacks, 0);
    }

    #[test]
    fn address_space_is_decoded_onto_access() {
        // Address-carrying contract: a per-address address_space decodes onto
        // the AddressAccess; absent stays None (legacy fallback).
        let mut inst = mem_inst(0);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Global as i32);
        let ma = memory_access(&inst, || "t".into()).unwrap().unwrap();
        assert_eq!(ma.accesses[0].address_space, Some(pb::MemorySpace::Global));

        let legacy = mem_inst(0);
        let ml = memory_access(&legacy, || "t".into()).unwrap().unwrap();
        assert_eq!(ml.accesses[0].address_space, None);
    }

    #[test]
    fn unknown_address_space_value_is_schema_skew() {
        // Fail-closed like op_class: an out-of-range address_space raw rejects.
        let mut inst = mem_inst(0);
        inst.addresses[0].address_space = Some(99);
        assert!(memory_access(&inst, || "t".into()).is_err());
    }

    #[test]
    fn operation_aware_memory_space_matrix_admits() {
        let cases = [
            (pb::MemoryOpClass::Global, pb::MemorySpace::Global),
            (pb::MemoryOpClass::Global, pb::MemorySpace::Generic),
            (pb::MemoryOpClass::Shared, pb::MemorySpace::Shared),
            (pb::MemoryOpClass::Local, pb::MemorySpace::Local),
            (pb::MemoryOpClass::Atomic, pb::MemorySpace::Global),
            (pb::MemoryOpClass::Atomic, pb::MemorySpace::Generic),
            (pb::MemoryOpClass::Atomic, pb::MemorySpace::Shared),
            (pb::MemoryOpClass::Atomic, pb::MemorySpace::Local),
            (pb::MemoryOpClass::AsyncCopy, pb::MemorySpace::Global),
            (pb::MemoryOpClass::AsyncCopy, pb::MemorySpace::Generic),
            // Structurally admitted for the destination-bearing v3 contract;
            // validate_address_carrying_shape rejects a lone/shared-first form.
            (pb::MemoryOpClass::AsyncCopy, pb::MemorySpace::Shared),
        ];
        for (class, space) in cases {
            let inst = identity_inst(class, Some(space), None, 0x40);
            memory_access(&inst, || format!("{class:?}/{space:?}"))
                .unwrap_or_else(|error| panic!("{class:?}/{space:?}: {error}"));
        }
    }

    #[test]
    fn missing_memory_space_retains_legacy_fallback() {
        for class in [
            pb::MemoryOpClass::Global,
            pb::MemoryOpClass::Shared,
            pb::MemoryOpClass::Local,
            pb::MemoryOpClass::Atomic,
            pb::MemoryOpClass::AsyncCopy,
        ] {
            let inst = identity_inst(class, None, None, 0x40);
            memory_access(&inst, || format!("legacy {class:?}"))
                .unwrap_or_else(|error| panic!("legacy {class:?}: {error}"));
        }
    }

    #[test]
    fn crossed_and_non_source_memory_spaces_are_typed_conflicts() {
        let cases = [
            (pb::MemoryOpClass::Shared, pb::MemorySpace::Local),
            (pb::MemoryOpClass::Local, pb::MemorySpace::Shared),
            (pb::MemoryOpClass::Global, pb::MemorySpace::Shared),
            (pb::MemoryOpClass::Global, pb::MemorySpace::Local),
        ];
        for (class, space) in cases {
            let inst = identity_inst(class, Some(space), None, 0x40);
            assert!(matches!(
                memory_access(&inst, || "t".into()),
                Err(TraceError::MemoryOpClassAddressSpaceConflict {
                    operand: 0,
                    class: got_class,
                    space: got_space,
                    ..
                }) if got_class == class && got_space == space
            ));
        }
    }

    #[test]
    fn structural_error_precedes_space_conflict() {
        let mut inst = identity_inst(
            pb::MemoryOpClass::Shared,
            Some(pb::MemorySpace::Local),
            None,
            0x40,
        );
        let mut malformed_later = inst.addresses[0].clone();
        malformed_later.addrs.clear();
        inst.addresses.push(malformed_later);
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::Lane {
                source: LaneError::ListAllLengthInvalid { got: 0 },
                ..
            })
        ));
    }

    #[test]
    fn non_memory_instruction_passes_through() {
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(0),
            predicate_mask: Some(0),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            ..Default::default()
        };
        assert!(memory_access(&inst, || "t".into()).unwrap().is_none());
    }

    // ---- allocation-identity gate (parity with the Python validator) --

    fn registry() -> AllocationRegistry {
        AllocationRegistry::from_str(
            r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x70000000","size_bytes":4096,"alive":true}]"#,
            Path::new("t"),
        )
        .unwrap()
    }

    #[test]
    fn allocation_registry_projects_checked_relative_offsets() {
        let registry = registry();
        assert_eq!(registry.relative_byte_offset(1, 0x7000_0000), Some(0));
        assert_eq!(registry.relative_byte_offset(1, 0x7000_007f), Some(0x7f));
        assert_eq!(registry.relative_byte_offset(1, 0x7000_1000), None);
        assert_eq!(registry.relative_byte_offset(2, 0x7000_0000), None);
        assert_eq!(
            registry.relative_sector_offset(1, 0x7000_0040_i64 >> 5),
            Some(2)
        );
        assert_eq!(
            registry.relative_sector_offset(1, 0x7000_1000_i64 >> 5),
            None
        );
        assert_eq!(registry.relative_sector_offset(1, -1), None);

        let unaligned = AllocationRegistry::from_str(
            r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x70000001","size_bytes":4096,"alive":true}]"#,
            Path::new("unaligned"),
        )
        .unwrap();
        assert_eq!(
            unaligned.relative_sector_offset(1, 0x7000_0020_i64 >> 5),
            None
        );
    }

    fn identity_inst(
        op_class: pb::MemoryOpClass,
        space: Option<pb::MemorySpace>,
        alloc_id: Option<u32>,
        base: u64,
    ) -> pb::Instruction {
        // list_all: lane 0 active (and_mask 0b1) carries `base`, lanes 1..31 the
        // inactive sentinel.
        let mut addrs = vec![base as i64];
        addrs.extend(std::iter::repeat(crate::lanes::LANE_INACTIVE_SENTINEL).take(31));
        pb::Instruction {
            pc: 0,
            active_mask: Some(0b1),
            predicate_mask: Some(0b1),
            function_unique_id: 1,
            memory_op_class: Some(op_class as i32),
            data_width: Some(4),
            warpid_tb: Some(0),
            vpc: Some(0),
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: crate::lanes::FORMAT_LIST_ALL,
                addrs,
                allocation_id: alloc_id,
                address_space: space.map(|s| s as i32),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn global_atomic_missing_id_rejected() {
        let inst = identity_inst(
            pb::MemoryOpClass::Atomic,
            Some(pb::MemorySpace::Global),
            None,
            0x7000_0000,
        );
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::AllocationIdMissing { .. })
        ));
    }

    #[test]
    fn ldgsts_global_source_missing_id_rejected() {
        let inst = identity_inst(
            pb::MemoryOpClass::AsyncCopy,
            Some(pb::MemorySpace::Global),
            None,
            0x7000_0000,
        );
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::AllocationIdMissing { .. })
        ));
    }

    #[test]
    fn identity_gate_reports_unknown_space_before_missing_id() {
        let mut inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            None,
            0x7000_0000,
        );
        inst.addresses[0].address_space = Some(99);
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::UnknownEnumValue {
                field: "address_space",
                raw: 99,
                ..
            })
        ));
    }

    #[test]
    fn identity_gate_reports_space_conflict_before_missing_id() {
        let inst = identity_inst(
            pb::MemoryOpClass::Shared,
            Some(pb::MemorySpace::Global),
            None,
            0x7000_0000,
        );
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::MemoryOpClassAddressSpaceConflict {
                operand: 0,
                class: pb::MemoryOpClass::Shared,
                space: pb::MemorySpace::Global,
                ..
            })
        ));
    }

    #[test]
    fn identity_gate_reports_later_lane_error_before_earlier_missing_id() {
        let mut inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            None,
            0x7000_0000,
        );
        let mut malformed_later = inst.addresses[0].clone();
        malformed_later.addrs.clear();
        inst.addresses.push(malformed_later);
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::Lane {
                source: LaneError::ListAllLengthInvalid { got: 0 },
                ..
            })
        ));
    }

    #[test]
    fn shared_atomic_is_exempt_from_identity() {
        // ATOMS: op_class ATOMIC, address_space SHARED -> exempt, admits with no id.
        let inst = identity_inst(
            pb::MemoryOpClass::Atomic,
            Some(pb::MemorySpace::Shared),
            None,
            0x40,
        );
        assert!(check_allocation_identity(&inst, Some(&registry()), || "t".into()).is_ok());
    }

    #[test]
    fn out_of_range_lane_rejected() {
        // base 0x90000000 is outside the [0x70000000, +4096) allocation.
        let inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            Some(1),
            0x9000_0000,
        );
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::LaneOutsideAllocationRange { .. })
        ));
    }

    #[test]
    fn allocation_crossing_u64_boundary_uses_wide_range_arithmetic() {
        let registry = AllocationRegistry::from_str(
            r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0xfffffffffffffff0","size_bytes":32,"alive":true}]"#,
            Path::new("t"),
        )
        .unwrap();
        let inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            Some(1),
            u64::MAX - 4,
        );
        assert!(check_allocation_identity(&inst, Some(&registry), || "t".into()).is_ok());
    }

    #[test]
    fn unknown_id_rejected() {
        let inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            Some(99),
            0x7000_0000,
        );
        assert!(matches!(
            check_allocation_identity(&inst, Some(&registry()), || "t".into()),
            Err(TraceError::AllocationIdUnknown { .. })
        ));
    }

    #[test]
    fn missing_sidecar_rejects_global_with_id() {
        let inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            Some(1),
            0x7000_0000,
        );
        assert!(matches!(
            check_allocation_identity(&inst, None, || "t".into()),
            Err(TraceError::AllocationsSidecarMissing { .. })
        ));
    }

    #[test]
    fn legacy_global_no_space_falls_back_to_class() {
        // Absent address_space -> legacy fallback to op_class==GLOBAL -> requires id.
        let no_id = identity_inst(pb::MemoryOpClass::Global, None, None, 0x7000_0000);
        assert!(matches!(
            check_allocation_identity(&no_id, Some(&registry()), || "t".into()),
            Err(TraceError::AllocationIdMissing { .. })
        ));
        let ok = identity_inst(pb::MemoryOpClass::Global, None, Some(1), 0x7000_0000);
        assert!(check_allocation_identity(&ok, Some(&registry()), || "t".into()).is_ok());
    }

    #[test]
    fn valid_global_source_passes() {
        let inst = identity_inst(
            pb::MemoryOpClass::Global,
            Some(pb::MemorySpace::Global),
            Some(1),
            0x7000_0000,
        );
        assert!(check_allocation_identity(&inst, Some(&registry()), || "t".into()).is_ok());
    }

    #[test]
    fn allocation_sidecar_limits_precede_vec_materialization() {
        let path = Path::new("allocations.json");
        let raw = br#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true},{"allocation_id":2,"context_id":1,"device_id":0,"base_address":"0x2000","size_bytes":16,"alive":true}]"#;
        let bytes = u64::try_from(raw.len()).unwrap();
        let nodes = 1 + 2 * 7;

        AllocationRegistry::parse_with_limits(path, raw, bytes, nodes, 2).unwrap();

        for (maximum_bytes, maximum_nodes, maximum_entries, resource) in [
            (bytes - 1, nodes, 2, "bytes"),
            (bytes, nodes - 1, 2, "json_nodes"),
            (bytes, nodes, 1, "top_level_entries"),
        ] {
            assert!(matches!(
                AllocationRegistry::parse_with_limits(
                    path,
                    raw,
                    maximum_bytes,
                    maximum_nodes,
                    maximum_entries,
                ),
                Err(TraceError::TraceArtifactResourceLimitExceeded {
                    artifact: "allocations_json",
                    resource: actual,
                    ..
                }) if actual == resource
            ));
        }
    }

    #[test]
    fn allocation_sidecar_load_rejects_oversized_sparse_file_before_reading() {
        let tmp = tempfile::tempdir().unwrap();
        let extra = tmp.path().join("extra_info");
        std::fs::create_dir(&extra).unwrap();
        let file = std::fs::File::create(extra.join("allocations.json")).unwrap();
        file.set_len(ALLOCATIONS_JSON_MAX_BYTES + 1).unwrap();

        assert!(matches!(
            AllocationRegistry::load(tmp.path()),
            Err(TraceError::TraceArtifactResourceLimitExceeded {
                artifact: "allocations_json",
                resource: "bytes",
                observed,
                limit: ALLOCATIONS_JSON_MAX_BYTES,
                ..
            }) if observed == ALLOCATIONS_JSON_MAX_BYTES + 1
        ));
    }

    #[test]
    fn sidecar_parser_enforces_python_parity() {
        let p = Path::new("t");
        // The canonical valid six-field entry admits.
        let ok = r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true}]"#;
        assert!(
            AllocationRegistry::from_str(ok, p).is_ok(),
            "valid six-field entry"
        );
        // device_id may be 0 or negative (uninstrumented device).
        let neg_dev = r#"[{"allocation_id":1,"context_id":1,"device_id":-1,"base_address":"0x1000","size_bytes":16,"alive":true}]"#;
        assert!(
            AllocationRegistry::from_str(neg_dev, p).is_ok(),
            "negative device_id"
        );

        // A helper building a full entry with one JSON substitution.
        let bad = |body: &str| AllocationRegistry::from_str(&format!("[{body}]"), p).is_err();
        // Value checks.
        assert!(
            bad(
                r#"{"allocation_id":0,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true}"#
            ),
            "id 0"
        );
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":0,"alive":true}"#
            ),
            "size 0"
        );
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"1000","size_bytes":16,"alive":true}"#
            ),
            "non-0x base"
        );
        // Exact-shape: missing each metadata field is rejected.
        assert!(
            bad(
                r#"{"allocation_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true}"#
            ),
            "missing context_id"
        );
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"base_address":"0x1000","size_bytes":16,"alive":true}"#
            ),
            "missing device_id"
        );
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16}"#
            ),
            "missing alive"
        );
        // Exact-shape: an unexpected extra field is rejected.
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true,"extra":1}"#
            ),
            "extra key"
        );
        // Wrong types: alive as int, id as bool, size negative.
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":1}"#
            ),
            "alive as int"
        );
        assert!(
            bad(
                r#"{"allocation_id":true,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true}"#
            ),
            "id as bool"
        );
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":-16,"alive":true}"#
            ),
            "negative size"
        );
        // Duplicate allocation_id is rejected.
        assert!(
            bad(
                r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":16,"alive":true},
                      {"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x2000","size_bytes":16,"alive":true}"#
            ),
            "duplicate id"
        );
    }
}

#[cfg(all(test, feature = "campaign"))]
mod sampling_sidecar_tests {
    use super::*;

    // Generated by `tools/trace_sampling/tests/test_sidecar.py::_subset`, then
    // sealed by Python `json.dumps(sort_keys=True,separators=(",",":"))`.
    // Keeping the producer's literal seal here prevents a Rust-hashes-Rust
    // tautology in the canonicalization test.
    const PYTHON_SEALED_SUBSET: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"ab"},"method":"wave_segment_stratified_subset","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":100,"max_trace_bytes":2000,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":12},"hashes":{"selected_block_ids_sha256":"84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","per_block_raw_sha256_manifest":"trace-payload-file-manifest-v1\nthreadblocks/device_0/stream_0/kernel_1/b_0.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n","sidecar_sha256":"12b7fcf51dd0d4277c9268b7eda67bca6dec410d36953098f181ef0f7ce7a907"},"label_pairing":"full_shape_label","segments":[{"id":0,"block_ids":[0,1,2]},{"id":19,"block_ids":[57,58,59]},{"id":1,"block_ids":[3,4,5]},{"id":2,"block_ids":[6,7,8]}],"selected_segment_ids":[0,19,1,2],"certainty_segments":[0,19],"force_include_blocks":[0],"strata":[{"id":0,"phase_id":0,"launch_tercile":0,"segment_ids":[1,2],"N_h":6,"n_h":2,"pi":0.3333333333333333,"sector_stddev":2.0,"neyman_weight":12.0}],"selection_reconciliation":{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"selection_budget":{"max_trace_bytes":2000,"basis":"raw_threadblock_pb","certainty_fraction_numerator":2,"certainty_fraction_denominator":5,"certainty_bytes":100},"selection_auxiliary":{"name":"l2_sector_transactions_32b","sector_bytes":32,"scope":"pass2_all_blocks","expected_full":400,"selected":200},"estimator":{"mode":"weighted_full_shape_estimate","targets":["sectors"],"not_recoverable":["cycles"],"convergence":{"status":"converged"}}}"#;

    const PYTHON_SEALED_LOSSLESS: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"deadbeef"},"method":"lossless_compressed_full","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":null,"max_trace_bytes":null,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":60,"captured":60},"hashes":{"selected_block_ids_sha256":"3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","per_block_raw_sha256_manifest":"trace-payload-file-manifest-v1\nthreadblocks/device_0/stream_0/kernel_1/b_0.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n","sidecar_sha256":"3c8a94decd2c56db2c72b25aa9b163fcc9d2a1dec13bf075546378c3ec874869"},"label_pairing":"full_shape_label"}"#;

    fn subset_value() -> serde_json::Value {
        serde_json::from_str(PYTHON_SEALED_SUBSET).unwrap()
    }

    fn lossless_value() -> serde_json::Value {
        serde_json::from_str(PYTHON_SEALED_LOSSLESS).unwrap()
    }

    fn reseal(value: &mut serde_json::Value) {
        let seal = canonical_sidecar_sha256(value).unwrap();
        value["hashes"]["sidecar_sha256"] = serde_json::Value::String(seal);
    }

    fn rebind_blocks_and_reseal(value: &mut serde_json::Value) {
        let mut blocks: Vec<u64> = value["segments"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|segment| segment["block_ids"].as_array().unwrap())
            .map(|block| block.as_u64().unwrap())
            .collect();
        blocks.sort_unstable();
        value["hashes"]["selected_block_ids_sha256"] = serde_json::Value::String(sha256_hex(
            serde_json::to_string(&blocks).unwrap().as_bytes(),
        ));
        reseal(value);
    }

    fn load_json(json: impl AsRef<[u8]>) -> Result<TraceSamplingSidecar, TraceError> {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("trace_sampling.json"), json).unwrap();
        TraceSamplingSidecar::load(tmp.path()).map(|value| value.unwrap())
    }

    #[test]
    fn absent_sidecar_loads_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(TraceSamplingSidecar::load(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn sampling_sidecar_limits_precede_value_materialization() {
        let raw = PYTHON_SEALED_LOSSLESS.as_bytes();
        let bytes = u64::try_from(raw.len()).unwrap();
        let shape = preflight_json_shape(raw, u64::MAX, None).unwrap();
        let path = Path::new("trace_sampling.json");

        TraceSamplingSidecar::parse_with_limits(path, raw, bytes, shape.nodes).unwrap();

        for (maximum_bytes, maximum_nodes, resource) in [
            (bytes - 1, shape.nodes, "bytes"),
            (bytes, shape.nodes - 1, "json_nodes"),
        ] {
            assert!(matches!(
                TraceSamplingSidecar::parse_with_limits(
                    path,
                    raw,
                    maximum_bytes,
                    maximum_nodes,
                ),
                Err(TraceError::TraceArtifactResourceLimitExceeded {
                    artifact: "trace_sampling_json",
                    resource: actual,
                    ..
                }) if actual == resource
            ));
        }
    }

    #[test]
    fn sampling_sidecar_load_rejects_oversized_sparse_file_before_reading() {
        let tmp = tempfile::tempdir().unwrap();
        let file = std::fs::File::create(tmp.path().join("trace_sampling.json")).unwrap();
        file.set_len(TRACE_SAMPLING_JSON_MAX_BYTES + 1).unwrap();

        assert!(matches!(
            TraceSamplingSidecar::load(tmp.path()),
            Err(TraceError::TraceArtifactResourceLimitExceeded {
                artifact: "trace_sampling_json",
                resource: "bytes",
                observed,
                limit: TRACE_SAMPLING_JSON_MAX_BYTES,
                ..
            }) if observed == TRACE_SAMPLING_JSON_MAX_BYTES + 1
        ));
    }

    #[test]
    fn arbitrary_precision_estimator_integer_retains_python_seal_domain() {
        let mut value = subset_value();
        value["estimator"]["convergence"]["per_target"] =
            serde_json::from_str(r#"{"sectors":{"population":18446744073709551616}}"#).unwrap();
        reseal(&mut value);
        load_json(serde_json::to_vec(&value).unwrap()).unwrap();
    }

    #[test]
    fn python_sealed_subset_parses_and_reconstructs_membership() {
        let sc = load_json(PYTHON_SEALED_SUBSET).unwrap();
        assert_eq!(sc.cell_id(), "c");
        assert_eq!(sc.kernel_mangled_name(), "_Z1kv");
        assert_eq!(sc.kernel_args_hash(), "ab");
        assert_eq!(sc.original_grid(), (60, 1, 1));
        assert_eq!(sc.original_block(), (32, 1, 1));
        assert_eq!(sc.captured_grid(), (60, 1, 1));
        assert_eq!(sc.captured_block(), (32, 1, 1));
        assert_eq!(sc.expected_full_records(), 6000);
        assert_eq!(sc.captured_records(), 12);
        assert_eq!(sc.decompressed_payload_sha256(), "a".repeat(64));
        assert_eq!(sc.raw_payload_bytes(), 1000);
        assert_eq!(sc.payload_file_manifest.len(), 1);
        assert!(
            sc.declared_block_placements()
                .eq((0..60).map(|block| (block, 0)))
        );
        assert_eq!(sc.declared_sm_for_launch_index(0), Some(0));
        assert_eq!(sc.declared_sm_for_launch_index(59), Some(0));
        assert_eq!(sc.declared_sm_for_launch_index(60), None);
        assert_eq!(
            sc.payload_file_manifest["threadblocks/device_0/stream_0/kernel_1/b_0.pb"].size_bytes,
            1000
        );
        let TraceSamplingMethod::WaveSegmentStratifiedSubset(provenance) = sc.method() else {
            panic!("subset sidecar lost its sampling method");
        };
        assert_eq!(
            provenance.selected_block_ids(),
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59]
                .into_iter()
                .collect()
        );
        assert_eq!(
            provenance.selected_block_ids_sha256(),
            "84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf"
        );
        assert_eq!(
            provenance.sidecar_sha256(),
            "12b7fcf51dd0d4277c9268b7eda67bca6dec410d36953098f181ef0f7ce7a907"
        );
    }

    #[test]
    fn validated_sidecar_retains_nonzero_spatial_placement() {
        let mut value = subset_value();
        value["wave_model"]["per_sm_resident_blocks"][0]["sm_id"] = serde_json::json!(83);
        value["per_sm_slot_map"][0]["sm_id"] = serde_json::json!(83);
        reseal(&mut value);

        let sidecar = load_json(serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            sidecar
                .declared_block_placements()
                .eq((0..60).map(|block| (block, 83)))
        );
        assert_eq!(sidecar.declared_sm_for_launch_index(8), Some(83));
        assert_eq!(sidecar.declared_sm_for_launch_index(-1), None);
    }

    #[test]
    fn retained_slot_map_rejects_duplicate_launch_indices() {
        let roster = BTreeMap::from([(0, vec![0]), (1, vec![0])]);
        let error = ValidatedPerSmSlotMap::retain(&roster, 2).unwrap_err();
        assert!(error.contains("block 0 occurs more than once"), "{error}");
    }

    #[test]
    fn retained_slot_map_rejects_missing_launch_indices() {
        let roster = BTreeMap::from([(0, vec![0]), (1, vec![2])]);
        let error = ValidatedPerSmSlotMap::retain(&roster, 3).unwrap_err();
        assert!(error.contains("launch index 1 is absent"), "{error}");
    }

    #[test]
    fn retained_slot_map_rejects_out_of_cardinality_launch_indices() {
        let roster = BTreeMap::from([(0, vec![0, 3])]);
        let error = ValidatedPerSmSlotMap::retain(&roster, 3).unwrap_err();
        assert!(
            error.contains("block 3 is outside retained per_sm_slot_map cardinality 3"),
            "{error}"
        );
    }

    #[test]
    fn stratum_id_overflow_is_rejected_not_saturated() {
        let mut value = subset_value();
        value["strata"][0]["id"] = serde_json::json!(u64::MAX);
        value["strata"][0]["phase_id"] = serde_json::json!(u64::MAX);
        reseal(&mut value);
        match load_json(serde_json::to_vec(&value).unwrap()) {
            Err(TraceError::SamplingSidecarMalformed { message, .. }) => {
                assert!(message.contains("encoding overflows u64"), "{message}");
            }
            other => panic!("expected stratum overflow rejection, got {other:?}"),
        }
    }

    #[test]
    fn stratum_cardinality_comparison_stays_in_u64() {
        let large_sample = 4_294_967_298_u64;
        assert!(!sampling_cardinalities_match(
            large_sample,
            u64::from(2_u32)
        ));

        let mut value = subset_value();
        value["strata"][0]["N_h"] = serde_json::json!(large_sample);
        value["strata"][0]["n_h"] = serde_json::json!(large_sample);
        value["strata"][0]["pi"] = serde_json::json!(1.0);
        value["strata"][0]["neyman_weight"] = serde_json::json!(8_589_934_596.0);
        reseal(&mut value);
        match load_json(serde_json::to_vec(&value).unwrap()) {
            Err(TraceError::SamplingSidecarMalformed { message, .. }) => {
                assert!(
                    message.contains("inconsistent N_h/n_h/segment_ids"),
                    "{message}"
                );
            }
            other => panic!("expected stratum cardinality rejection, got {other:?}"),
        }
    }

    #[test]
    fn max_population_probability_has_no_absolute_tolerance_floor() {
        let population = sampling_u64_as_f64(u64::MAX);
        assert!(!relative_close_enough(1e-13, 2.0 / population));

        let mut value = subset_value();
        value["strata"][0]["N_h"] = serde_json::json!(u64::MAX);
        value["strata"][0]["pi"] = serde_json::json!(1e-13);
        value["strata"][0]["sector_stddev"] = serde_json::json!(0.0);
        value["strata"][0]["neyman_weight"] = serde_json::json!(0.0);
        reseal(&mut value);

        match load_json(serde_json::to_vec(&value).unwrap()) {
            Err(TraceError::SamplingSidecarMalformed { message, .. }) => {
                assert!(
                    message.contains("inconsistent design provenance"),
                    "{message}"
                );
            }
            other => panic!("expected inconsistent probability rejection, got {other:?}"),
        }
    }

    #[test]
    fn overflowed_derived_stratum_weight_is_rejected() {
        assert!(!close_enough(f64::MAX, f64::INFINITY));

        let mut value = subset_value();
        let population = sampling_u64_as_f64(u64::MAX);
        value["strata"][0]["N_h"] = serde_json::json!(u64::MAX);
        value["strata"][0]["pi"] = serde_json::json!(2.0 / population);
        value["strata"][0]["sector_stddev"] = serde_json::json!(f64::MAX);
        // Before the finite-derived-value gate, any finite claim compared equal
        // to the +inf product because both the delta and tolerance became +inf.
        value["strata"][0]["neyman_weight"] = serde_json::json!(f64::MAX);
        reseal(&mut value);

        match load_json(serde_json::to_vec(&value).unwrap()) {
            Err(TraceError::SamplingSidecarMalformed { message, .. }) => {
                assert!(
                    message.contains("derived neyman_weight is not finite"),
                    "{message}"
                );
            }
            other => panic!("expected non-finite derived-weight rejection, got {other:?}"),
        }
    }

    #[test]
    fn invalid_tercile_precedes_stratum_id_overflow() {
        let mut value = subset_value();
        value["strata"][0]["id"] = serde_json::json!(u64::MAX);
        value["strata"][0]["phase_id"] = serde_json::json!(u64::MAX);
        value["strata"][0]["launch_tercile"] = serde_json::json!(3);
        reseal(&mut value);
        match load_json(serde_json::to_vec(&value).unwrap()) {
            Err(TraceError::SamplingSidecarMalformed { message, .. }) => {
                assert!(
                    message.contains("id does not encode phase/launch tercile"),
                    "{message}"
                );
                assert!(!message.contains("overflows"), "{message}");
            }
            other => panic!("expected invalid tercile rejection, got {other:?}"),
        }
    }

    #[test]
    fn lossless_full_sidecar_loads_but_does_not_sanction_sampling() {
        let sc = load_json(PYTHON_SEALED_LOSSLESS).unwrap();
        assert_eq!(sc.cell_id(), "c");
        assert_eq!(sc.kernel_mangled_name(), "_Z1kv");
        assert_eq!(sc.kernel_args_hash(), "deadbeef");
        assert_eq!(sc.original_grid(), (60, 1, 1));
        assert_eq!(sc.original_block(), (32, 1, 1));
        assert_eq!(sc.captured_grid(), (60, 1, 1));
        assert_eq!(sc.captured_block(), (32, 1, 1));
        assert_eq!(sc.expected_full_records(), 60);
        assert_eq!(sc.captured_records(), 60);
        assert_eq!(sc.decompressed_payload_sha256(), "a".repeat(64));
        assert_eq!(sc.raw_payload_bytes(), 1000);
        assert!(
            sc.declared_block_placements()
                .eq((0..60).map(|block| (block, 0)))
        );
        assert_eq!(sc.declared_sm_for_launch_index(31), Some(0));
        let TraceSamplingMethod::LosslessCompressedFull(provenance) = sc.method() else {
            panic!("lossless sidecar lost its method-specific provenance");
        };
        assert_eq!(
            provenance.selected_block_ids_sha256(),
            "3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a"
        );
        assert_eq!(
            provenance.sidecar_sha256(),
            "3c8a94decd2c56db2c72b25aa9b163fcc9d2a1dec13bf075546378c3ec874869"
        );
    }

    #[test]
    fn lossless_rejects_stale_seal_fresh_record_drop_and_wrong_pairing() {
        let mut stale = lossless_value();
        stale["records"]["expected_full"] = serde_json::json!(59);
        stale["records"]["captured"] = serde_json::json!(59);
        let err = load_json(serde_json::to_vec(&stale).unwrap()).unwrap_err();
        assert!(err.to_string().contains("sidecar_sha256"));

        let mut record_drop = lossless_value();
        record_drop["records"]["captured"] = serde_json::json!(59);
        reseal(&mut record_drop);
        let err = load_json(serde_json::to_vec(&record_drop).unwrap()).unwrap_err();
        assert!(err.to_string().contains("must equal records.expected_full"));

        let mut wrong_pairing = lossless_value();
        wrong_pairing["label_pairing"] = serde_json::json!("matched_reduced_shape_label");
        reseal(&mut wrong_pairing);
        let err = load_json(serde_json::to_vec(&wrong_pairing).unwrap()).unwrap_err();
        assert!(err.to_string().contains("full_shape_label"));
    }

    #[test]
    fn lossless_rejects_freshly_sealed_malformed_or_unbound_metadata() {
        let mut cases = Vec::new();

        let mut unknown_bytes = lossless_value();
        unknown_bytes["bytes"]["surprise"] = serde_json::json!(1);
        cases.push(unknown_bytes);

        let mut malformed_feature = lossless_value();
        malformed_feature["tracer"]["features"]["atomic_addr"] = serde_json::json!("yes");
        cases.push(malformed_feature);

        let mut rewritten_shape = lossless_value();
        rewritten_shape["shape"]["captured_grid"]["x"] = serde_json::json!(59);
        cases.push(rewritten_shape);

        let mut wrong_roster_hash = lossless_value();
        wrong_roster_hash["hashes"]["selected_block_ids_sha256"] =
            serde_json::Value::String("0".repeat(64));
        cases.push(wrong_roster_hash);

        for mut case in cases {
            reseal(&mut case);
            assert!(matches!(
                load_json(serde_json::to_vec(&case).unwrap()),
                Err(TraceError::SamplingSidecarMalformed { .. })
            ));
        }
    }

    #[test]
    fn lossless_rejects_freshly_sealed_stale_or_unknown_producers_and_missing_features() {
        let mut tracer_version = lossless_value();
        tracer_version["tracer"]["version"] = serde_json::json!(999);

        let mut stale_stats_pass_version = lossless_value();
        stale_stats_pass_version["tracer"]["stats_pass_version"] = serde_json::json!(1);

        let mut unknown_stats_pass_version = lossless_value();
        unknown_stats_pass_version["tracer"]["stats_pass_version"] = serde_json::json!(999);

        let mut atomic_addr = lossless_value();
        atomic_addr["tracer"]["features"]["atomic_addr"] = serde_json::json!(false);

        let mut ldgsts_addr = lossless_value();
        ldgsts_addr["tracer"]["features"]["ldgsts_addr"] = serde_json::json!(false);

        for (mut sidecar, expected) in [
            (tracer_version, "tracer v5 and stats pass v2"),
            (stale_stats_pass_version, "tracer v5 and stats pass v2"),
            (unknown_stats_pass_version, "tracer v5 and stats pass v2"),
            (atomic_addr, "atomic_addr and ldgsts_addr features"),
            (ldgsts_addr, "atomic_addr and ldgsts_addr features"),
        ] {
            reseal(&mut sidecar);
            let error = load_json(serde_json::to_vec(&sidecar).unwrap()).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn supported_sidecars_require_a_canonical_per_file_payload_manifest() {
        for mut sidecar in [subset_value(), lossless_value()] {
            sidecar["hashes"]
                .as_object_mut()
                .unwrap()
                .remove("per_block_raw_sha256_manifest");
            reseal(&mut sidecar);
            let error = load_json(serde_json::to_vec(&sidecar).unwrap()).unwrap_err();
            assert!(
                error.to_string().contains("per_block_raw_sha256_manifest"),
                "{error}"
            );
        }

        for manifest in [
            "trace-payload-file-manifest-v1\nthreadblocks/../escape.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
            "trace-payload-file-manifest-v1\nthreadblocks/..pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        ] {
            let mut unsafe_path = subset_value();
            unsafe_path["hashes"]["per_block_raw_sha256_manifest"] = serde_json::json!(manifest);
            reseal(&mut unsafe_path);
            let error = load_json(serde_json::to_vec(&unsafe_path).unwrap()).unwrap_err();
            assert!(
                error.to_string().contains("canonical threadblocks"),
                "{error}"
            );
        }

        let mut huge_size = lossless_value();
        huge_size["hashes"]["per_block_raw_sha256_manifest"] = serde_json::json!(format!(
            "trace-payload-file-manifest-v1\nthreadblocks/a.pb\t{}\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
            "9".repeat(5_000)
        ));
        reseal(&mut huge_size);
        let error = load_json(serde_json::to_vec(&huge_size).unwrap()).unwrap_err();
        assert!(error.to_string().contains("exceeds u64"), "{error}");

        let mut wrong_total = lossless_value();
        wrong_total["hashes"]["per_block_raw_sha256_manifest"] = serde_json::json!(
            "trace-payload-file-manifest-v1\nthreadblocks/a.pb\t999\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        );
        reseal(&mut wrong_total);
        let error = load_json(serde_json::to_vec(&wrong_total).unwrap()).unwrap_err();
        assert!(error.to_string().contains("byte total"), "{error}");
    }

    #[test]
    fn lossless_equality_retains_the_exact_sealed_sidecar_identity() {
        let original = load_json(PYTHON_SEALED_LOSSLESS).unwrap();
        let mut changed = lossless_value();
        changed["cell_id"] = serde_json::json!("other-cell");
        reseal(&mut changed);
        let changed = load_json(serde_json::to_vec(&changed).unwrap()).unwrap();
        assert_ne!(original, changed);
        let (
            TraceSamplingMethod::LosslessCompressedFull(original_provenance),
            TraceSamplingMethod::LosslessCompressedFull(changed_provenance),
        ) = (original.method(), changed.method())
        else {
            panic!("lossless method provenance was not retained");
        };
        assert_ne!(
            original_provenance.sidecar_sha256(),
            changed_provenance.sidecar_sha256()
        );
    }

    #[test]
    fn sha256_implementation_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut split = Sha256::new();
        split.update(b"a");
        split.update(b"b");
        split.update(b"c");
        assert_eq!(
            split.finish_digest().to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_json_matches_python_unicode_and_exponent_spelling() {
        let value = serde_json::json!({
            "tiny": 1e-7,
            "huge": 1e20,
            "unicode": "é😀\u{7f}",
        });
        let actual = canonical_json(&value).unwrap();
        assert_eq!(
            actual,
            r#"{"huge":1e+20,"tiny":1e-07,"unicode":"\u00e9\ud83d\ude00\u007f"}"#
        );

        for (raw, expected) in [
            ("1e-4", "0.0001"),
            ("1e-5", "1e-05"),
            ("1e15", "1000000000000000.0"),
            ("1e16", "1e+16"),
            ("-0.0", "-0.0"),
        ] {
            let value: serde_json::Value = serde_json::from_str(raw).unwrap();
            let actual = canonical_json(&value).unwrap();
            assert_eq!(actual, expected, "Python spelling for {raw}");
        }
    }

    #[test]
    fn non_object_and_minimal_fake_subset_are_rejected_typed() {
        let cases = [
            "[]".to_string(),
            format!(
                r#"{{"schema_version":2,"method":"wave_segment_stratified_subset","shape":{{"original_grid":{{"x":60,"y":1,"z":1}}}},"records":{{"captured":12}},"hashes":{{"selected_block_ids_sha256":"{a}","decompressed_payload_sha256":"{a}","sidecar_sha256":"{a}"}},"selection_reconciliation":{{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"{a}"}}}}"#,
                a = "a".repeat(64)
            ),
        ];
        for json in cases {
            let err = load_json(json).unwrap_err();
            assert!(
                matches!(err, TraceError::SamplingSidecarMalformed { .. }),
                "expected SamplingSidecarMalformed, got {err:?}"
            );
        }
    }

    #[test]
    fn subset_projection_rejects_unknown_and_missing_binding_fields() {
        let mut cases = Vec::new();

        let mut unknown_root = subset_value();
        unknown_root["surprise"] = serde_json::json!(1);
        cases.push(unknown_root);

        let mut unknown_segment = subset_value();
        unknown_segment["segments"][0]["surprise"] = serde_json::json!(1);
        cases.push(unknown_segment);

        for field in [
            "segments",
            "selected_segment_ids",
            "certainty_segments",
            "selection_reconciliation",
        ] {
            let mut missing = subset_value();
            missing.as_object_mut().unwrap().remove(field);
            cases.push(missing);
        }

        let mut missing_grid = subset_value();
        missing_grid["shape"]
            .as_object_mut()
            .unwrap()
            .remove("captured_grid");
        cases.push(missing_grid);

        for case in cases {
            assert!(matches!(
                load_json(serde_json::to_vec(&case).unwrap()),
                Err(TraceError::SamplingSidecarMalformed { .. })
            ));
        }
    }

    #[test]
    fn subset_rejects_bad_seal_and_bad_membership_hash_independently() {
        let mut bad_seal = subset_value();
        bad_seal["records"]["captured"] = serde_json::json!(13);
        assert!(matches!(
            load_json(serde_json::to_vec(&bad_seal).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));

        let mut bad_block_hash = subset_value();
        bad_block_hash["hashes"]["selected_block_ids_sha256"] =
            serde_json::Value::String("0".repeat(64));
        reseal(&mut bad_block_hash);
        assert!(matches!(
            load_json(serde_json::to_vec(&bad_block_hash).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn subset_rejects_duplicate_or_mismatched_segment_membership() {
        let mut duplicate_block = subset_value();
        duplicate_block["segments"][2]["block_ids"] = serde_json::json!([2, 4, 5]);
        reseal(&mut duplicate_block);
        assert!(matches!(
            load_json(serde_json::to_vec(&duplicate_block).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));

        let mut mismatched_segment_ids = subset_value();
        mismatched_segment_ids["selected_segment_ids"] = serde_json::json!([0, 19, 1]);
        reseal(&mut mismatched_segment_ids);
        assert!(matches!(
            load_json(serde_json::to_vec(&mismatched_segment_ids).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn subset_cannot_rewrite_captured_grid_even_with_a_fresh_seal() {
        let mut rewritten = subset_value();
        rewritten["shape"]["captured_grid"]["x"] = serde_json::json!(12);
        reseal(&mut rewritten);
        assert!(matches!(
            load_json(serde_json::to_vec(&rewritten).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn freshly_sealed_empty_or_arbitrary_nested_envelopes_are_rejected() {
        let mut cases = Vec::new();
        for field in [
            "gates",
            "tracer",
            "wave_model",
            "selection_budget",
            "selection_auxiliary",
            "rng",
            "estimator",
            "bytes",
        ] {
            let mut value = subset_value();
            value[field] = serde_json::json!({});
            reseal(&mut value);
            cases.push(value);
        }
        for field in ["per_sm_slot_map", "strata"] {
            let mut value = subset_value();
            value[field] = serde_json::json!([]);
            reseal(&mut value);
            cases.push(value);
        }
        let mut wrong_label = subset_value();
        wrong_label["label_pairing"] = serde_json::json!("no_comparable_label");
        reseal(&mut wrong_label);
        cases.push(wrong_label);
        let mut wrong_reason = subset_value();
        wrong_reason["reason"] = serde_json::json!("invented");
        reseal(&mut wrong_reason);
        cases.push(wrong_reason);

        for case in cases {
            assert!(matches!(
                load_json(serde_json::to_vec(&case).unwrap()),
                Err(TraceError::SamplingSidecarMalformed { .. })
            ));
        }
    }

    #[test]
    fn freshly_rebound_decimated_segments_and_nonexhaustive_rosters_are_rejected() {
        let mut decimated = subset_value();
        decimated["segments"][0]["block_ids"] = serde_json::json!([0, 1]);
        rebind_blocks_and_reseal(&mut decimated);
        assert!(matches!(
            load_json(serde_json::to_vec(&decimated).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));

        let mut incomplete_roster = subset_value();
        incomplete_roster["per_sm_slot_map"][0]["block_ids"]
            .as_array_mut()
            .unwrap()
            .pop();
        reseal(&mut incomplete_roster);
        assert!(matches!(
            load_json(serde_json::to_vec(&incomplete_roster).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));

        let mut mismatched_residency = subset_value();
        mismatched_residency["wave_model"]["per_sm_resident_blocks"][0]["sm_id"] =
            serde_json::json!(1);
        reseal(&mut mismatched_residency);
        assert!(matches!(
            load_json(serde_json::to_vec(&mismatched_residency).unwrap()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));
    }
}
