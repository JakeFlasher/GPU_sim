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
//! Both sidecars ARE read here: `allocations.json` feeds the allocation-identity
//! gate, and `trace_sampling.json` (present for a supported subset or retained
//! lossless-compression provenance) feeds the completeness gate
//! (`E_INCOMPLETE_TRACE` / never-rewrite-gridDim).
//!
//! Admission-level validation is owned by the Python gate
//! (`tools/nvbit_tracer/import_trace_contract.py`) and already ran on every
//! M1-admitted trace (allocation-id/range identity, vpc/warpid presence, and
//! the full per-instruction walk stay THERE). This reader re-enforces the
//! invariants the simulator itself depends on to replay correctly — memory
//! identity, masks, data widths, lane reconstruction, block identity — and
//! it fail-closes on each of them: a malformed record is a typed error,
//! never a skip.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use prost::Message;

use crate::lanes::{self, LaneError};
use crate::pb;

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse {path} as {message_type}: {source}")]
    Decode {
        path: PathBuf,
        message_type: &'static str,
        #[source]
        source: prost::DecodeError,
    },
    #[error(
        "{path}: unsupported trace_contract_id {got:?}; expected empty legacy identity or sm86_addr_carry_v1"
    )]
    TraceContractUnsupported { path: PathBuf, got: String },
    #[error(
        "{path}: trace contract {contract_id:?} carries invalid memory_payload_classes {got:?}; expected exactly [GLOBAL, SHARED, LOCAL, ATOMIC, ASYNC_COPY]"
    )]
    MemoryPayloadCapabilitiesInvalid {
        path: PathBuf,
        contract_id: String,
        got: Vec<i32>,
    },
    #[error("kernel threadblock dir {dir} is missing or empty — the trace tree is truncated")]
    ThreadblocksMissing { dir: PathBuf },
    #[error("threadblock payload entry {path} is not a regular file")]
    ThreadblockPayloadNotRegular { path: PathBuf },
    #[error("verified sampling payload snapshot is missing expected file {path}")]
    SamplingPayloadSnapshotMissing { path: PathBuf },
    #[error(
        "{record}: instruction has memory identity but active_mask/predicate_mask absent on the wire"
    )]
    MaskMissing { record: String },
    #[error(
        "{record}: instruction has {n_addresses} address record(s) but memory_op_class is absent"
    )]
    MemoryOpClassMissing { record: String, n_addresses: usize },
    #[error(
        "{record}: memory_op_class is present but UNSPECIFIED; the tracer should have emitted an unsupported_memory_path exclusion instead"
    )]
    MemoryOpClassUnspecified { record: String },
    #[error("{record}: memory_op_access_kind is present but UNSPECIFIED on a memory instruction")]
    MemoryOpAccessKindUnspecified { record: String },
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
}

/// One kernel launch located inside the root `Trace` message.
#[derive(Debug, Clone, Copy)]
pub struct KernelLoc<'a> {
    pub device_id: i64,
    pub stream_id: i64,
    pub kernel: &'a pb::Kernel,
}

pub const ADDRESS_CARRYING_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPayloadContract {
    LegacyExclusion,
    AddressCarryingV1,
}

const ADDRESS_CARRYING_PAYLOAD_CLASSES: [pb::MemoryOpClass; 5] = [
    pb::MemoryOpClass::Global,
    pb::MemoryOpClass::Shared,
    pb::MemoryOpClass::Local,
    pb::MemoryOpClass::Atomic,
    pb::MemoryOpClass::AsyncCopy,
];

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
    if trace.trace_contract_id != ADDRESS_CARRYING_TRACE_CONTRACT_ID {
        return Err(TraceError::TraceContractUnsupported {
            path: path.to_path_buf(),
            got: trace.trace_contract_id.clone(),
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
    Ok(MemoryPayloadContract::AddressCarryingV1)
}

struct VerifiedSamplingPayload {
    by_path: BTreeMap<PathBuf, Vec<u8>>,
}

pub struct TraceDir {
    root: PathBuf,
    trace: pb::Trace,
    memory_payload_contract: MemoryPayloadContract,
    allocations: Option<AllocationRegistry>,
    sampling: Option<TraceSamplingSidecar>,
    sampling_payload: Option<VerifiedSamplingPayload>,
}

impl TraceDir {
    /// Open `<root>/dynamic_trace.pb` and hold the decoded launch metadata, plus
    /// the `extra_info/allocations.json` registry (absent for legacy/no-identity
    /// traces) used by the reader-side allocation-identity gate.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        let root = root.into();
        let pb_path = root.join("dynamic_trace.pb");
        let raw = fs::read(&pb_path).map_err(|source| TraceError::Io {
            path: pb_path.clone(),
            source,
        })?;
        let trace = pb::Trace::decode(raw.as_slice()).map_err(|source| TraceError::Decode {
            path: pb_path.clone(),
            message_type: "dynamic_trace.Trace",
            source,
        })?;
        let memory_payload_contract = classify_memory_payload_contract(&trace, &pb_path)?;
        let allocations = AllocationRegistry::load(&root)?;
        let sampling = TraceSamplingSidecar::load(&root)?;
        let sampling_payload = sampling
            .as_ref()
            .map(|sidecar| verify_sampling_payload(&root, sidecar))
            .transpose()?;
        Ok(Self {
            root,
            trace,
            memory_payload_contract,
            allocations,
            sampling,
            sampling_payload,
        })
    }

    /// The loaded allocation registry, if the sidecar was present.
    pub fn allocations(&self) -> Option<&AllocationRegistry> {
        self.allocations.as_ref()
    }

    /// The validated trace-sampling sidecar (`trace_sampling.json`), if present.
    /// Supported sidecars distinguish sampled-subset provenance from a retained
    /// lossless-compression declaration; neither alone proves full coverage.
    pub fn sampling_sidecar(&self) -> Option<&TraceSamplingSidecar> {
        self.sampling.as_ref()
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

    /// Paths of the per-threadblock `.pb` files for one kernel launch, in
    /// deterministic FILENAME order — which is NOT flattened-block launch
    /// order (filenames embed unpadded decimal coordinates, so "10" sorts
    /// before "2"). Block identity/order must come from the proto's
    /// `block_id`: use [`TraceDir::read_threadblocks_in_launch_order`]. When a
    /// sampling sidecar is present, this list comes from the payload snapshot
    /// sealed during [`TraceDir::open`], not a second directory walk.
    ///
    /// A kernel launch with a missing/empty threadblock dir is a typed
    /// error: every launched kernel has at least one block, so absence
    /// means the trace tree is truncated. Directory-entry read failures
    /// propagate as typed I/O errors — a dropped entry would silently
    /// shrink the replay set.
    pub fn threadblock_paths(&self, loc: &KernelLoc<'_>) -> Result<Vec<PathBuf>, TraceError> {
        let dir = self
            .root
            .join("threadblocks")
            .join(format!("device_{}", loc.device_id))
            .join(format!("stream_{}", loc.stream_id))
            .join(format!("kernel_{}", loc.kernel.id));
        let mut paths = if let Some(snapshot) = &self.sampling_payload {
            snapshot
                .by_path
                .keys()
                .filter(|path| path.parent() == Some(dir.as_path()))
                .cloned()
                .collect()
        } else {
            let entries = match fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(TraceError::ThreadblocksMissing { dir });
                }
                Err(source) => return Err(TraceError::Io { path: dir, source }),
            };
            let mut paths = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|source| TraceError::Io {
                    path: dir.clone(),
                    source,
                })?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "pb") {
                    let file_type = entry.file_type().map_err(|source| TraceError::Io {
                        path: path.clone(),
                        source,
                    })?;
                    if !file_type.is_file() {
                        return Err(TraceError::ThreadblockPayloadNotRegular { path });
                    }
                    paths.push(path);
                }
            }
            paths
        };
        if paths.is_empty() {
            return Err(TraceError::ThreadblocksMissing { dir });
        }
        paths.sort();
        Ok(paths)
    }

    /// Decode one threadblock path owned by this trace directory. Sidecar-bound
    /// traces read only the immutable bytes captured and hashed by
    /// [`TraceDir::open`]; paths outside that snapshot fail closed. Traces without
    /// a sampling sidecar retain the ordinary filesystem reader.
    pub fn read_threadblock(&self, path: &Path) -> Result<pb::Threadblock, TraceError> {
        match &self.sampling_payload {
            Some(snapshot) => {
                let raw = snapshot.by_path.get(path).ok_or_else(|| {
                    TraceError::SamplingPayloadSnapshotMissing {
                        path: path.to_path_buf(),
                    }
                })?;
                decode_threadblock(raw, path)
            }
            None => read_threadblock(path),
        }
    }

    /// Read every threadblock of one kernel launch, keyed and sorted by the
    /// FLATTENED block index derived from the proto's canonical `block_id`
    /// (CUDA row-major: `x + y*gx + z*gx*gy` — the same rule as the Python
    /// consumer `l1_production_helper._flatten_block_id`). Fail-closed on
    /// missing `block_id`, out-of-grid coordinates, and duplicates.
    ///
    /// Loads the whole kernel's blocks into memory (fine for the current
    /// roster; a streaming variant can land with the SM scheduler when
    /// needed). Sidecar-bound traces decode the immutable byte snapshot captured
    /// and hashed by [`TraceDir::open`], so later path replacement cannot change
    /// replay under already-issued provenance.
    pub fn read_threadblocks_in_launch_order(
        &self,
        loc: &KernelLoc<'_>,
    ) -> Result<Vec<(i64, pb::Threadblock)>, TraceError> {
        let mut blocks = Vec::new();
        for path in self.threadblock_paths(loc)? {
            let tb = self.read_threadblock(&path)?;
            let index = block_launch_index(loc.kernel, &tb, &path)?;
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LosslessFullProvenance {
    selected_block_ids_sha256: String,
    sidecar_sha256: String,
}

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveSegmentSubsetProvenance {
    selected_block_ids: BTreeSet<i64>,
    selected_block_ids_sha256: String,
    sidecar_sha256: String,
}

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
#[derive(Debug, Clone, PartialEq, Eq)]
struct PayloadFileManifestEntry {
    size_bytes: u64,
    sha256: String,
}

/// Reader-side view of a supported `<root>/trace_sampling.json` v2 sidecar.
/// Fields are private so method-specific invariants can only be constructed by
/// the closed parser. A sealed lossless declaration is not itself a
/// full-coverage witness. Launch-bound full versus sampled coverage is
/// established later by replay after observing block membership and record counts.
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
    method: TraceSamplingMethod,
}

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

    pub fn method(&self) -> &TraceSamplingMethod {
        &self.method
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

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

const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

fn sha256_compress(state: &mut [u32; 8], chunk: &[u8]) {
    debug_assert_eq!(chunk.len(), 64);
    let mut schedule = [0u32; 64];
    for (index, word) in schedule[..16].iter_mut().enumerate() {
        let offset = index * 4;
        *word = u32::from_be_bytes([
            chunk[offset],
            chunk[offset + 1],
            chunk[offset + 2],
            chunk[offset + 3],
        ]);
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(SHA256_K[index])
            .wrapping_add(schedule[index]);
        let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h].into_iter()) {
        *slot = slot.wrapping_add(value);
    }
}

struct Sha256 {
    state: [u32; 8],
    tail: [u8; 64],
    tail_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: SHA256_INITIAL,
            tail: [0; 64],
            tail_len: 0,
            total_bytes: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(input.len() as u64);
        if self.tail_len > 0 {
            let copied = (64 - self.tail_len).min(input.len());
            self.tail[self.tail_len..self.tail_len + copied].copy_from_slice(&input[..copied]);
            self.tail_len += copied;
            input = &input[copied..];
            if self.tail_len < 64 {
                return;
            }
            sha256_compress(&mut self.state, &self.tail);
            self.tail_len = 0;
        }
        let mut chunks = input.chunks_exact(64);
        for chunk in &mut chunks {
            sha256_compress(&mut self.state, chunk);
        }
        let remainder = chunks.remainder();
        self.tail[..remainder.len()].copy_from_slice(remainder);
        self.tail_len = remainder.len();
    }

    fn finish(mut self) -> String {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.tail[self.tail_len] = 0x80;
        self.tail_len += 1;
        if self.tail_len > 56 {
            self.tail[self.tail_len..].fill(0);
            sha256_compress(&mut self.state, &self.tail);
            self.tail = [0; 64];
        } else {
            self.tail[self.tail_len..56].fill(0);
        }
        self.tail[56..].copy_from_slice(&bit_len.to_be_bytes());
        sha256_compress(&mut self.state, &self.tail);
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect()
    }
}

fn sha256_hex(input: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(input);
    digest.finish()
}

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

fn object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

fn required<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
    context: &str,
) -> Result<&'a serde_json::Value, String> {
    object
        .get(field)
        .ok_or_else(|| format!("{context} is missing required field {field}"))
}

fn positive_u32(value: &serde_json::Value, context: &str) -> Result<u32, String> {
    let value = value
        .as_u64()
        .and_then(|raw| u32::try_from(raw).ok())
        .filter(|raw| *raw > 0)
        .ok_or_else(|| format!("{context} must be a positive u32"))?;
    Ok(value)
}

fn parse_dim3(value: &serde_json::Value, context: &str) -> Result<(u32, u32, u32), String> {
    let dim = object(value, context)?;
    exact_keys(dim, &["x", "y", "z"], &["x", "y", "z"], context)?;
    Ok((
        positive_u32(required(dim, "x", context)?, &format!("{context}.x"))?,
        positive_u32(required(dim, "y", context)?, &format!("{context}.y"))?,
        positive_u32(required(dim, "z", context)?, &format!("{context}.z"))?,
    ))
}

fn write_python_json_number(value: &serde_json::Number, out: &mut String) {
    let rendered = value.to_string();
    if value.is_i64() || value.is_u64() {
        out.push_str(&rendered);
        return;
    }

    // serde_json and CPython use shortest-roundtrip decimal digits, but choose
    // fixed/scientific notation at different cutovers.  Re-layout the common
    // shortest significand using CPython's repr rule: fixed for decimal
    // exponents -4..15, scientific otherwise, with a signed two-digit minimum
    // exponent.  This is the spelling used by json.dumps for finite floats.
    let (negative, unsigned) = rendered
        .strip_prefix('-')
        .map_or((false, rendered.as_str()), |rest| (true, rest));
    let (mantissa, explicit_exponent) = match unsigned.find(['e', 'E']) {
        Some(at) => (
            &unsigned[..at],
            unsigned[at + 1..].parse::<i32>().unwrap_or(0),
        ),
        None => (unsigned, 0),
    };
    let decimal_at = mantissa.find('.').unwrap_or(mantissa.len());
    let digits: String = mantissa.chars().filter(|ch| *ch != '.').collect();
    let Some(first) = digits.bytes().position(|byte| byte != b'0') else {
        out.push_str(if negative { "-0.0" } else { "0.0" });
        return;
    };
    let last = digits
        .bytes()
        .rposition(|byte| byte != b'0')
        .expect("a first nonzero digit exists");
    let significant = &digits[first..=last];
    let exponent = explicit_exponent + decimal_at as i32 - first as i32 - 1;

    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent < 0 {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            out.push_str(significant);
        } else {
            let integer_digits = exponent as usize + 1;
            if significant.len() <= integer_digits {
                out.push_str(significant);
                for _ in significant.len()..integer_digits {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&significant[..integer_digits]);
                out.push('.');
                out.push_str(&significant[integer_digits..]);
            }
        }
    } else {
        out.push(significant.as_bytes()[0] as char);
        if significant.len() > 1 {
            out.push('.');
            out.push_str(&significant[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let absolute = exponent.unsigned_abs();
        if absolute < 10 {
            out.push('0');
        }
        out.push_str(&absolute.to_string());
    }
}

fn canonical_python_json(value: &serde_json::Value, out: &mut String) -> Result<(), String> {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => write_python_json_number(value, out),
        serde_json::Value::String(value) => {
            out.push('"');
            for ch in value.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\u{08}' => out.push_str("\\b"),
                    '\u{09}' => out.push_str("\\t"),
                    '\n' => out.push_str("\\n"),
                    '\u{0c}' => out.push_str("\\f"),
                    '\r' => out.push_str("\\r"),
                    '\u{20}'..='\u{7e}' => out.push(ch),
                    ch if u32::from(ch) <= 0xffff => {
                        out.push_str(&format!("\\u{:04x}", u32::from(ch)));
                    }
                    ch => {
                        let code = u32::from(ch) - 0x1_0000;
                        let high = 0xd800 + (code >> 10);
                        let low = 0xdc00 + (code & 0x3ff);
                        out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                    }
                }
            }
            out.push('"');
        }
        serde_json::Value::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical_python_json(value, out)?;
            }
            out.push(']');
        }
        serde_json::Value::Object(values) => {
            out.push('{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical_python_json(&serde_json::Value::String(key.clone()), out)?;
                out.push(':');
                canonical_python_json(value, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

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
    let mut canonical = String::new();
    canonical_python_json(&unsigned, &mut canonical)?;
    Ok(sha256_hex(canonical.as_bytes()))
}

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

fn unsigned(value: &serde_json::Value, context: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{context} must be a nonnegative integer"))
}

fn positive(value: &serde_json::Value, context: &str) -> Result<u64, String> {
    unsigned(value, context).and_then(|value| {
        if value == 0 {
            Err(format!("{context} must be positive"))
        } else {
            Ok(value)
        }
    })
}

fn boolean(value: &serde_json::Value, context: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{context} must be boolean"))
}

fn string<'a>(value: &'a serde_json::Value, context: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{context} must be a string"))
}

fn nonnegative_number(value: &serde_json::Value, context: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{context} must be a finite nonnegative number"))
}

fn string_array(value: &serde_json::Value, context: &str) -> Result<(), String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    for value in values {
        string(value, &format!("{context} entry"))?;
    }
    Ok(())
}

fn sampling_cardinalities_match(sample: u64, ids_len: u64) -> bool {
    sample == ids_len
}

/// Convert the complete `u64` domain through two exactly representable `u32`
/// limbs. The final addition performs the one unavoidable binary64 rounding;
/// no unchecked integer-to-float narrowing hides which source bits participate.
fn sampling_u64_as_f64(value: u64) -> f64 {
    let bytes = value.to_be_bytes();
    let high = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let low = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

fn relative_close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12 * left.abs().max(right.abs())
}

fn close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12_f64.max(1e-12 * left.abs().max(right.abs()))
}

fn validate_lossless_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_grid: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_block_ids_sha256: &str,
) -> Result<u64, String> {
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
    if captured_records > expected_records {
        return Err("records.captured cannot exceed records.expected_full".to_string());
    }
    Ok(raw_bytes)
}

fn validate_subset_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_segments: &BTreeSet<u64>,
    certainty_segments: &BTreeSet<u64>,
    segment_blocks: &BTreeMap<u64, BTreeSet<u64>>,
) -> Result<(), String> {
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
    Ok(())
}

fn collect_threadblock_pb_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), TraceError> {
    let entries = fs::read_dir(dir).map_err(|source| TraceError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| TraceError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| TraceError::Io {
            path: path.clone(),
            source,
        })?;
        let pb_named = path.extension().is_some_and(|suffix| suffix == "pb")
            || path.file_name().is_some_and(|name| name == ".pb");
        if file_type.is_dir() {
            if pb_named {
                return Err(TraceError::ThreadblockPayloadNotRegular { path });
            }
            collect_threadblock_pb_paths(&path, paths)?;
        } else if pb_named {
            if !file_type.is_file() {
                return Err(TraceError::ThreadblockPayloadNotRegular { path });
            }
            paths.push(path);
        }
    }
    Ok(())
}

fn verify_sampling_payload(
    root: &Path,
    sidecar: &TraceSamplingSidecar,
) -> Result<VerifiedSamplingPayload, TraceError> {
    let mut paths = Vec::new();
    collect_threadblock_pb_paths(&root.join("threadblocks"), &mut paths)?;
    let sidecar_path = root.join("trace_sampling.json");
    let malformed = |message: String| TraceError::SamplingSidecarMalformed {
        path: sidecar_path.clone(),
        message,
    };
    if paths.is_empty() {
        return Err(malformed(
            "threadblock payload snapshot contains no regular .pb files".to_string(),
        ));
    }

    let mut payload_paths = Vec::with_capacity(paths.len());
    for path in paths {
        let relative = path.strip_prefix(root).map_err(|_| {
            malformed(format!(
                "threadblock payload path is outside trace root: {}",
                path.display()
            ))
        })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| {
                malformed(format!(
                    "threadblock payload path is not UTF-8: {}",
                    path.display()
                ))
            })?
            .to_string();
        if !safe_payload_manifest_path(&relative) {
            return Err(malformed(format!(
                "threadblock payload path is not canonical for the per-file manifest: {relative:?}"
            )));
        }
        payload_paths.push((path, relative));
    }
    payload_paths.sort_by(|(_, left), (_, right)| left.cmp(right));
    let actual_paths: BTreeSet<_> = payload_paths
        .iter()
        .map(|(_, relative)| relative.as_str())
        .collect();
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

    let mut actual_bytes = 0_u64;
    let mut digest = Sha256::new();
    let mut by_path = BTreeMap::new();
    for (path, relative) in payload_paths {
        let raw = fs::read(&path).map_err(|source| TraceError::Io {
            path: path.clone(),
            source,
        })?;
        let size = u64::try_from(raw.len()).map_err(|_| {
            malformed(format!(
                "threadblock payload {} length exceeds u64",
                path.display()
            ))
        })?;
        let declared = &sidecar.payload_file_manifest[&relative];
        if size != declared.size_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest size for {relative:?} is {}, actual {size}",
                declared.size_bytes
            )));
        }
        let file_hash = sha256_hex(&raw);
        if file_hash != declared.sha256 {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest hash for {relative:?} does not match raw payload (actual {file_hash})"
            )));
        }
        actual_bytes = actual_bytes
            .checked_add(size)
            .ok_or_else(|| malformed("threadblock payload byte count overflows u64".to_string()))?;
        digest.update(&raw);
        by_path.insert(path, raw);
    }
    if actual_bytes != sidecar.raw_payload_bytes {
        return Err(malformed(format!(
            "bytes.raw {} does not match threadblock payload bytes {actual_bytes}",
            sidecar.raw_payload_bytes
        )));
    }
    let actual_hash = digest.finish();
    if actual_hash != sidecar.decompressed_payload_sha256 {
        return Err(malformed(format!(
            "decompressed_payload_sha256 does not match sorted threadblock bytes (actual {actual_hash})"
        )));
    }
    Ok(VerifiedSamplingPayload { by_path })
}

impl TraceSamplingSidecar {
    /// Parse `<root>/trace_sampling.json` if present (`Ok(None)` if absent).
    /// This internal stage validates and retains the JSON envelope; callers only
    /// receive it through [`TraceDir::sampling_sidecar`] after `TraceDir::open`
    /// has also bound the decompressed threadblock bytes.
    fn load(root: &Path) -> Result<Option<Self>, TraceError> {
        let path = root.join("trace_sampling.json");
        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(TraceError::Io { path, source }),
        };
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| TraceError::SamplingSidecarMalformed {
                path: path.clone(),
                message: e.to_string(),
            })?;
        let malformed = |message: String| TraceError::SamplingSidecarMalformed {
            path: path.clone(),
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

        let (method, raw_payload_bytes) = match method {
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

                validate_subset_producer_contract(
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

                let expected_seal = canonical_sidecar_sha256(&parsed).map_err(&malformed)?;
                if sidecar_sha256 != expected_seal {
                    return Err(malformed(
                        "hashes.sidecar_sha256 does not match Python-canonical sidecar".to_string(),
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
                )
            }
            // A lossless-compression declaration is sealed and payload-bound,
            // but replay must still observe complete launch coverage itself.
            "lossless_compressed_full" => {
                let raw = validate_lossless_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_grid,
                    captured_block,
                    &selected_block_ids_sha256,
                )
                .map_err(&malformed)?;
                let expected_seal = canonical_sidecar_sha256(&parsed).map_err(&malformed)?;
                if sidecar_sha256 != expected_seal {
                    return Err(malformed(
                        "hashes.sidecar_sha256 does not match Python-canonical sidecar".to_string(),
                    ));
                }
                (
                    TraceSamplingMethod::LosslessCompressedFull(LosslessFullProvenance {
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    raw,
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
        Ok(Some(TraceSamplingSidecar {
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
            method,
        }))
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
    pb::Threadblock::decode(raw).map_err(|source| TraceError::Decode {
        path: path.to_path_buf(),
        message_type: "dynamic_trace.threadblock",
        source,
    })
}

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
        if self.meta.and_mask == 0 {
            return;
        }
        let no_error_context = || String::new();
        for addr in &self.inst.addresses {
            let access = match decode_address_access(
                addr,
                self.meta.and_mask,
                self.inst_width,
                &no_error_context,
            ) {
                Ok(access) => access,
                Err(_) => unreachable!("preflighted immutable address failed a second decode"),
            };
            sink(&access);
        }
    }
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
        pb::MemoryOpClass::AsyncCopy => {
            matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic)
        }
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
/// "Memory instruction" starts with the canonical gate (`_check_instruction`):
/// address records OR a present `memory_op_class`. This replay consumer also
/// treats present data-width, access-kind, or cache-op metadata as memory evidence
/// so a malformed partial producer record cannot silently downgrade to
/// `NonMemory`. For those records, masks and `memory_op_class` must be present, the class must be
/// specified, `data_width` must be present and strictly positive, unknown enum
/// wire values are typed schema-skew errors, and every address record must
/// reconstruct positionally. Non-memory instructions return `Ok(None)`.
pub fn preflight_memory_access<'a>(
    inst: &'a pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<PreflightedMemoryAccess<'a>>, TraceError> {
    let n_addresses = inst.addresses.len();
    let looks_like_memory = n_addresses > 0
        || inst.memory_op_class.is_some()
        || inst.data_width.is_some()
        || inst.memory_op_access_kind.is_some()
        || inst.instruction_cache_op.is_some();
    if !looks_like_memory {
        return Ok(None);
    }

    let (Some(active_mask), Some(predicate_mask)) = (inst.active_mask, inst.predicate_mask) else {
        return Err(TraceError::MaskMissing { record: record() });
    };

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
        Some(raw) => Some(known_enum::<pb::InstructionCacheOp>(
            raw,
            "instruction_cache_op",
            &record,
        )?),
    };

    // Canonical data_width contract: present and strictly positive on every
    // memory instruction (data_width_missing / data_width_invalid).
    let Some(inst_width) = inst.data_width else {
        return Err(TraceError::DataWidthMissing { record: record() });
    };
    if inst_width <= 0 {
        return Err(TraceError::DataWidthInvalid {
            record: record(),
            got: inst_width,
        });
    }

    let and_mask = active_mask & predicate_mask;
    // Validate every address, including zero-mask records. Those stream no
    // callbacks, but malformed lane payloads and unknown enums remain failures.
    for addr in &inst.addresses {
        decode_address_access(addr, and_mask, inst_width, &record)?;
    }
    // Structural preflight above intentionally completes before this semantic
    // compatibility gate. An explicit per-address space may refine ATOMIC, but
    // it cannot cross GLOBAL/SHARED/LOCAL classes or turn ASYNC_COPY into a
    // shared-destination record. Absence remains the admitted legacy fallback.
    for (operand, addr) in inst.addresses.iter().enumerate() {
        let Some(raw) = addr.address_space else {
            continue;
        };
        let space: pb::MemorySpace = known_enum(raw, "address_space", &record)?;
        if !memory_space_is_compatible(op_class, space) {
            return Err(TraceError::MemoryOpClassAddressSpaceConflict {
                record: record(),
                operand,
                class: op_class,
                space,
            });
        }
    }

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
    // Python parity: base_address MUST be a 0x-prefixed hex string; a bare or
    // decimal value is rejected (returns None → allocations_sidecar_malformed).
    let t = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    u64::from_str_radix(t, 16).ok()
}

impl AllocationRegistry {
    /// Load from `<root>/extra_info/allocations.json`. `Ok(None)` if the file is
    /// absent (legacy traces, or traces with no GLOBAL/GENERIC active-lane record
    /// requiring identity). `Err(AllocationsSidecarMalformed)` on bad JSON, a
    /// non-hex `base_address`, or a duplicate `allocation_id`.
    pub fn load(root: &Path) -> Result<Option<Self>, TraceError> {
        let path = root.join("extra_info").join("allocations.json");
        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(TraceError::Io { path, source }),
        };
        Self::from_str(&raw, &path).map(Some)
    }

    fn from_str(raw: &str, path: &Path) -> Result<Self, TraceError> {
        let entries: Vec<RawAllocation> =
            serde_json::from_str(raw).map_err(|e| TraceError::AllocationsSidecarMalformed {
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
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!("duplicate allocation_id {}", e.allocation_id),
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
    let inst_class = Some(preflight.meta().op_class);
    let mut identity_error = None;
    preflight.visit_addresses(|access| {
        // Preserve the first identity diagnosis. Complete structural validation
        // already succeeded when the preflight token was constructed.
        if identity_error.is_some() || !record_requires_identity(access.address_space, inst_class) {
            return;
        }
        let alloc_id = access.allocation_id.unwrap_or(0);
        if alloc_id == 0 {
            identity_error = Some(TraceError::AllocationIdMissing { record: record() });
            return;
        }
        let Some(reg) = registry else {
            identity_error = Some(TraceError::AllocationsSidecarMissing { record: record() });
            return;
        };
        let Some(range) = reg.by_id.get(&alloc_id) else {
            identity_error = Some(TraceError::AllocationIdUnknown {
                record: record(),
                allocation_id: alloc_id,
            });
            return;
        };
        let base = u128::from(range.base);
        let end = base + u128::from(range.size);
        for (lane, &slot) in access.lane_vector.iter().enumerate() {
            if slot == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let lane_addr = slot as u64;
            let lane_addr_wide = u128::from(lane_addr);
            if !(base <= lane_addr_wide && lane_addr_wide < end) {
                identity_error = Some(TraceError::LaneOutsideAllocationRange {
                    record: record(),
                    lane,
                    lane_addr,
                    allocation_id: alloc_id,
                    base: range.base,
                    end,
                });
                return;
            }
        }
    });
    match identity_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
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

    #[test]
    fn synthetic_trace_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 2);

        let td = TraceDir::open(tmp.path()).unwrap();
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

    /// Lexicographic filename order is NOT launch order once a coordinate
    /// hits double digits; read_threadblocks_in_launch_order must return
    /// flattened-block_id order regardless.
    #[test]
    fn launch_order_is_numeric_not_lexicographic() {
        let tmp = tempfile::tempdir().unwrap();
        write_synthetic_trace(tmp.path(), 12);
        let td = TraceDir::open(tmp.path()).unwrap();
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
            warps: Default::default(),
        };
        std::fs::write(
            tmp.path()
                .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
            tb.encode_to_vec(),
        )
        .unwrap();
        let td = TraceDir::open(tmp.path()).unwrap();
        let kernels: Vec<_> = td.kernels().collect();
        assert!(matches!(
            td.read_threadblocks_in_launch_order(&kernels[0]),
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
        let td = TraceDir::open(tmp.path()).unwrap();
        let kernels: Vec<_> = td.kernels().collect();
        assert!(matches!(
            td.read_threadblocks_in_launch_order(&kernels[0]),
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

        let td = TraceDir::open(tmp.path()).unwrap();
        let kernels: Vec<_> = td.kernels().collect();
        assert!(matches!(
            td.threadblock_paths(&kernels[0]),
            Err(TraceError::ThreadblocksMissing { .. })
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
    fn masks_required_on_memory_instruction() {
        let mut inst = mem_inst(0);
        inst.active_mask = None;
        inst.predicate_mask = None;
        assert!(matches!(
            memory_access(&inst, || "t".into()),
            Err(TraceError::MaskMissing { .. })
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
    fn memory_only_metadata_without_class_fails_closed() {
        let width_only = pb::Instruction {
            active_mask: Some(0),
            predicate_mask: Some(0),
            data_width: Some(4),
            ..Default::default()
        };
        assert!(matches!(
            memory_access(&width_only, || "width-only".into()),
            Err(TraceError::MemoryOpClassMissing { n_addresses: 0, .. })
        ));

        let kind_only = pb::Instruction {
            active_mask: Some(0),
            predicate_mask: Some(0),
            data_width: Some(4),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        };
        assert!(matches!(
            memory_access(&kind_only, || "kind-only".into()),
            Err(TraceError::MemoryOpClassMissing { n_addresses: 0, .. })
        ));

        let cache_op_only = pb::Instruction {
            active_mask: Some(0),
            predicate_mask: Some(0),
            data_width: Some(4),
            instruction_cache_op: Some(pb::InstructionCacheOp::LdgECa as i32),
            ..Default::default()
        };
        assert!(matches!(
            memory_access(&cache_op_only, || "cache-op-only".into()),
            Err(TraceError::MemoryOpClassMissing { n_addresses: 0, .. })
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
            (pb::MemoryOpClass::AsyncCopy, pb::MemorySpace::Shared),
            (pb::MemoryOpClass::Atomic, pb::MemorySpace::Unspecified),
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
        let inst = pb::Instruction::default();
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
            active_mask: Some(0b1),
            predicate_mask: Some(0b1),
            memory_op_class: Some(op_class as i32),
            data_width: Some(4),
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

#[cfg(test)]
mod sampling_sidecar_tests {
    use super::*;

    // Generated by `tools/trace_sampling/tests/test_sidecar.py::_subset`, then
    // sealed by Python `json.dumps(sort_keys=True,separators=(",",":"))`.
    // Keeping the producer's literal seal here prevents a Rust-hashes-Rust
    // tautology in the canonicalization test.
    const PYTHON_SEALED_SUBSET: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"ab"},"method":"wave_segment_stratified_subset","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":100,"max_trace_bytes":2000,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":12},"hashes":{"selected_block_ids_sha256":"84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","per_block_raw_sha256_manifest":"trace-payload-file-manifest-v1\nthreadblocks/device_0/stream_0/kernel_1/b_0.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n","sidecar_sha256":"12b7fcf51dd0d4277c9268b7eda67bca6dec410d36953098f181ef0f7ce7a907"},"label_pairing":"full_shape_label","segments":[{"id":0,"block_ids":[0,1,2]},{"id":19,"block_ids":[57,58,59]},{"id":1,"block_ids":[3,4,5]},{"id":2,"block_ids":[6,7,8]}],"selected_segment_ids":[0,19,1,2],"certainty_segments":[0,19],"force_include_blocks":[0],"strata":[{"id":0,"phase_id":0,"launch_tercile":0,"segment_ids":[1,2],"N_h":6,"n_h":2,"pi":0.3333333333333333,"sector_stddev":2.0,"neyman_weight":12.0}],"selection_reconciliation":{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"selection_budget":{"max_trace_bytes":2000,"basis":"raw_threadblock_pb","certainty_fraction_numerator":2,"certainty_fraction_denominator":5,"certainty_bytes":100},"selection_auxiliary":{"name":"l2_sector_transactions_32b","sector_bytes":32,"scope":"pass2_all_blocks","expected_full":400,"selected":200},"estimator":{"mode":"weighted_full_shape_estimate","targets":["sectors"],"not_recoverable":["cycles"],"convergence":{"status":"converged"}}}"#;

    const PYTHON_SEALED_LOSSLESS: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"deadbeef"},"method":"lossless_compressed_full","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":null,"max_trace_bytes":null,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":60},"hashes":{"selected_block_ids_sha256":"3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","per_block_raw_sha256_manifest":"trace-payload-file-manifest-v1\nthreadblocks/device_0/stream_0/kernel_1/b_0.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n","sidecar_sha256":"e6be7bb4738e3f831213a6f2eea33a0f9198909b21a5a39ce24cacf8a9e53fa9"},"label_pairing":"full_shape_label"}"#;

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
        assert_eq!(sc.expected_full_records(), 6000);
        assert_eq!(sc.captured_records(), 60);
        assert_eq!(sc.decompressed_payload_sha256(), "a".repeat(64));
        assert_eq!(sc.raw_payload_bytes(), 1000);
        let TraceSamplingMethod::LosslessCompressedFull(provenance) = sc.method() else {
            panic!("lossless sidecar lost its method-specific provenance");
        };
        assert_eq!(
            provenance.selected_block_ids_sha256(),
            "3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a"
        );
        assert_eq!(
            provenance.sidecar_sha256(),
            "e6be7bb4738e3f831213a6f2eea33a0f9198909b21a5a39ce24cacf8a9e53fa9"
        );
    }

    #[test]
    fn lossless_rejects_stale_seal_and_freshly_sealed_wrong_pairing() {
        let mut stale = lossless_value();
        stale["records"]["captured"] = serde_json::json!(59);
        let err = load_json(serde_json::to_vec(&stale).unwrap()).unwrap_err();
        assert!(err.to_string().contains("sidecar_sha256"));

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
            split.finish(),
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
        let mut actual = String::new();
        canonical_python_json(&value, &mut actual).unwrap();
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
            let mut actual = String::new();
            canonical_python_json(&value, &mut actual).unwrap();
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
