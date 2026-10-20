//! Trace validation errors and summary records; report-only.

use super::*;

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
    pub(super) device_id: i64,
    pub(super) stream_id: i64,
    pub(super) kernel: &'a pb::Kernel,
}

/// Non-vacuous producer counters parsed from snapshotted `stats.csv` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceStats {
    pub(super) row_count: u64,
    pub(super) expected_record_count: u64,
    pub(super) admitted_record_count: u64,
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
    pub(super) launch_count: u64,
    pub(super) threadblock_count: u64,
    pub(super) instruction_count: u64,
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
