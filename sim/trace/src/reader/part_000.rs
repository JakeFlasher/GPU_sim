// Lexical implementation section 1/5; included by the parent facade.
use std::collections::{BTreeMap, BTreeSet, HashSet};

use std::path::{Path, PathBuf};

use std::sync::{Arc, OnceLock};

use prost::Message;

use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};

#[cfg(feature = "campaign")]
use crate::anchored::FileIdentity;

use crate::anchored::{AnchoredDir, AnchoredFile, AnchoredReadError, FileGenerationIdentity};

#[cfg(feature = "campaign")]
use crate::digest::{Sha256, Sha256Digest, canonical_json};

use crate::exec_info::ExecInfo;

use crate::lanes::{self, LaneError};

use crate::pb;

use crate::wire_preflight::{self, WirePreflightError};

/// Canonical key emitted by the validated trace for a per-lane global memory
/// record. The alias keeps the trace projection and value-sidecar join on one
/// identity type without exposing cache or timing semantics here.
pub type DynamicMemoryJoinIdentity = sim_mem::NativePayloadJoinKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicMemoryJoinLocation {
    pub key: DynamicMemoryJoinIdentity,
    pub block_id: [u32; 3],
    pub warp: u32,
    pub instruction_index: u32,
    pub launch_index: i64,
    pub warp_slot: Option<i32>,
    /// Trace-declared direction for this dynamic GLOBAL occurrence.  A
    /// payload admission may bind only to `Store`; `None` is retained for
    /// legacy traces and must fail closed at the value-bearing ingress.
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// Trace-observed SM placement for the dynamic instruction. This is a
    /// placement identity, not a physical pipe/SMSP/slot observation.
    pub sm_id: Option<i32>,
}

const MIB: u64 = 1024 * 1024;

const DYNAMIC_TRACE_PB_MAX_BYTES: u64 = 1024 * MIB;

const ALLOCATIONS_JSON_MAX_BYTES: u64 = 16 * MIB;

#[cfg(feature = "campaign")]
const TRACE_SAMPLING_JSON_MAX_BYTES: u64 = 64 * MIB;

#[cfg(feature = "campaign")]
const CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES: u64 = 768 * MIB;

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
    #[error("execution-info admission failed: {0}")]
    ExecInfo(#[from] crate::exec_info::ExecInfoError),
    #[error("execution-info file {path} changed after its snapshot")]
    ExecInfoChanged { path: PathBuf },
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
        "{path}: unsupported trace_contract_id {got:?}; expected empty legacy identity, sm86_addr_carry_v1, sm86_addr_carry_shmem_split_v2, sm86_addr_carry_ldgsts_dst_v3, or sm120_addr_carry_ldgsts_dst_v3"
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
        "launch ({device_id}, {stream_id}, {kernel_id}) has negative number_of_registers {registers}"
    )]
    TraceLaunchRegistersInvalid {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        registers: i32,
    },
    #[error(
        "launch ({device_id}, {stream_id}, {kernel_id}) has invalid shared-memory carveout preference {preference}"
    )]
    TraceLaunchCarveoutInvalid {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        preference: i32,
    },
    #[error(
        "launch ({device_id}, {stream_id}, {kernel_id}) has invalid driver occupancy limit {blocks}"
    )]
    TraceLaunchOccupancyInvalid {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        blocks: u32,
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
        "{record}: instrumentation_order requires same-record {identity} identity (canonical: instrumentation_order_identity_missing)"
    )]
    InstrumentationOrderIdentityMissing {
        record: String,
        identity: &'static str,
    },
    #[error(
        "{record}: instrumentation_order is present with reserved value zero; producer order is 1-based (canonical: instrumentation_order_zero)"
    )]
    InstrumentationOrderZero { record: String },
    #[error(
        "{record}: instrumentation_order={current} does not strictly follow prior warp-local order {previous} (canonical: instrumentation_order_not_increasing)"
    )]
    InstrumentationOrderNotIncreasing {
        record: String,
        previous: u64,
        current: u64,
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
        "{record}: address lane mask 0x{lane_mask:x} is not a subset of instruction mask 0x{instruction_mask:x} (address_lane_mask_outside_instruction_mask)"
    )]
    AddressLaneMaskOutsideInstructionMask {
        record: String,
        lane_mask: u32,
        instruction_mask: u32,
    },
    #[error(
        "{record}: GLOBAL/GENERIC-space record has active lanes but allocation_id \
         is absent or zero (canonical: allocation_id_missing)"
    )]
    AllocationIdMissing { record: String },
    #[error(
        "{record}: allocation_identity_status={status:?} is not replayable (allocation_identity_status_invalid)"
    )]
    AllocationIdentityStatusInvalid { record: String, status: String },
    #[error(
        "{record}: unresolved_module_global cannot carry allocation_id {allocation_id} (allocation_identity_status_conflict)"
    )]
    AllocationIdentityStatusConflict { record: String, allocation_id: u32 },
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
        "root launch identity ({device_id}, {stream_id}, {kernel_id}) has a non-positive kernel id"
    )]
    LaunchIdentityInvalid {
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
    #[error("dynamic memory join identity is duplicated in the validated trace")]
    DynamicMemoryJoinIdentityDuplicate,
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
            Self::TraceLaunchRegistersInvalid { .. } => Some("trace_launch_registers_invalid"),
            Self::TraceLaunchCarveoutInvalid { .. } => Some("trace_launch_carveout_invalid"),
            Self::TraceLaunchOccupancyInvalid { .. } => Some("trace_launch_occupancy_invalid"),
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
            Self::InstrumentationOrderIdentityMissing { .. } => {
                Some("instrumentation_order_identity_missing")
            }
            Self::InstrumentationOrderZero { .. } => Some("instrumentation_order_zero"),
            Self::InstrumentationOrderNotIncreasing { .. } => {
                Some("instrumentation_order_not_increasing")
            }
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
                LaneError::NegativeAddress { .. } => Some("compressed_format_malformed"),
            },
            Self::AddressLaneMaskOutsideInstructionMask { .. } => {
                Some("address_lane_mask_outside_instruction_mask")
            }
            Self::AllocationIdMissing { .. } => Some("allocation_id_missing"),
            Self::AllocationIdentityStatusInvalid { .. } => {
                Some("allocation_identity_status_invalid")
            }
            Self::AllocationIdentityStatusConflict { .. } => {
                Some("allocation_identity_status_conflict")
            }
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
            | Self::LaunchIdentityDuplicate { .. }
            | Self::LaunchIdentityInvalid { .. } => Some("trace_launch_identity_mismatch"),
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

pub const SM120_LDGSTS_DESTINATION_TRACE_CONTRACT_ID: &str = "sm120_addr_carry_ldgsts_dst_v3";

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
        LDGSTS_DESTINATION_TRACE_CONTRACT_ID | SM120_LDGSTS_DESTINATION_TRACE_CONTRACT_ID => {
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
            None => reader.read_bounded_regular(&dynamic_relative, DYNAMIC_TRACE_PB_MAX_BYTES)?,
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

        let (maximum_files, maximum_bytes) = limits.map_or(
            (
                DIRECT_THREADBLOCK_FILE_MAX_COUNT,
                CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES,
            ),
            |limits| (limits.threadblock_file_count, limits.threadblock_raw_bytes),
        );
        let maximum_file_bytes = limits.map_or(CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES, |limits| {
            limits
                .threadblock_raw_bytes
                .min(CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES)
        });
        let tree_relative = root_relative.join("threadblocks");
        let mut tree = reader
            .read_pb_tree_with_file_limit(
                &tree_relative,
                maximum_files,
                maximum_bytes,
                maximum_file_bytes,
            )
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

const DIRECT_EXEC_INFO_RELATIVE: &str = "extra_info/enhanced_execution_info.json";

/// Match the exec-info loader bound for full-workload direct replay.
const DIRECT_EXEC_INFO_MAX_BYTES: u64 = 256 * 1024 * 1024;

fn read_direct_exec_info(
    reader: &AnchoredDir,
) -> Result<Option<(ExecInfo, FileGenerationIdentity)>, TraceError> {
    let relative = Path::new(DIRECT_EXEC_INFO_RELATIVE);
    let path = reader.root().join(relative);
    let file = match reader
        .read_optional_bounded_regular(relative, DIRECT_EXEC_INFO_MAX_BYTES)
        .map_err(TraceError::AnchoredSnapshot)?
    {
        Some(file) => file,
        None => return Ok(None),
    };
    let generation = file.generation_identity();
    let raw = file.into_raw();
    let exec = ExecInfo::parse(&path, &raw).map_err(TraceError::ExecInfo)?;
    let actual = reader
        .inspect_regular_generation(relative)
        .map_err(TraceError::AnchoredSnapshot)?;
    if actual != generation {
        return Err(TraceError::ExecInfoChanged { path });
    }
    Ok(Some((exec, generation)))
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
        self.nodes = self.nodes.saturating_add(1);
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
        self.top_level_sequence_entries = self.top_level_sequence_entries.saturating_add(1);
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
        let mut keys = BTreeSet::new();
        while let Some(key) = object.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key {key:?}"
                )));
            }
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
/// threadblock to 3,000,000 instructions, so a route-error arena index plus the
/// four ready-route discriminants fits in 22 bits. Packing the two small closed
/// enums into the remaining bits keeps each compact memory record within 24
/// bytes; exact u64 async payloads live in the sparse owning-warp arena.
#[derive(Debug, Clone, Copy)]
struct CompactMemoryMeta(u32);

impl CompactMemoryMeta {
    const ROUTE_BITS: u32 = 22;
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
    /// Sum of distinct 128-byte destination lines touched by source-coupled
    /// cohorts and executing source-ignored zero-fill lanes.
    pub destination_line_fanout_total: u32,
    /// Largest distinct destination-line fanout of any one source cohort.
    pub destination_line_fanout_max: u32,
    /// Existing cache-all projection: the sum, over source cohorts and
    /// zero-fill destination cohorts, of the maximum distinct 4-byte-word
    /// multiplicity on any of 32 shared banks.
    pub cache_all_bank_wavefronts_total: u32,
    /// Largest cache-all bank-word multiplicity of any destination cohort.
    pub cache_all_bank_wavefronts_max: u32,
    /// Trace-derived bypass projection. For each destination cohort this adds
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

