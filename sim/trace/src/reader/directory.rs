//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

pub struct TraceDir {
    pub(super) async_sidecars: OnceLock<
        Result<crate::async_sidecars::AsyncSidecars, crate::async_sidecars::AsyncSidecarError>,
    >,
    pub(super) lineage: TraceLineage,
    pub(super) root: PathBuf,
    pub(super) trace: pb::Trace,
    /// Direct loaders retain the parsed execution-info join alongside the
    /// dynamic snapshot so replay cannot reopen a mutable sidecar by path.
    /// Campaign evidence carries this join separately in its manifest.
    pub(super) exec_info: Option<ExecInfo>,
    pub(super) memory_payload_contract: MemoryPayloadContract,
    pub(super) allocations: Option<AllocationRegistry>,
    #[cfg(feature = "campaign")]
    pub(super) sampling: Option<TraceSamplingSidecar>,
    pub(super) stats: Option<TraceStats>,
    pub(super) validation: TraceValidationSummary,
    pub(super) payload_by_path: BTreeMap<PathBuf, Vec<u8>>,
    pub(super) raw_payload_retained: bool,
    pub(super) threadblock_by_path: BTreeMap<PathBuf, ValidatedThreadblock>,
}

/// In-process identity of one fully loaded trace snapshot. A `TraceDir` owns
/// all decoded replay inputs, so retaining this token prevents a stream-L2
/// session from combining its retained cache state with a separately opened
/// snapshot that happens to use the same filesystem path and launch IDs.
#[derive(Debug, Clone)]
pub struct TraceLineage(pub(super) Arc<()>);

impl TraceLineage {
    pub(super) fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl PartialEq for TraceLineage {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TraceLineage {}

impl TraceDir {
    /// Return the immutable in-process lineage token for this loaded snapshot.
    pub fn lineage(&self) -> TraceLineage {
        self.lineage.clone()
    }

    /// Parsed static execution metadata captured with a direct trace load.
    /// Campaign snapshots return `None` because their evidence admission owns
    /// the execution-info artifact and generation binding.
    pub fn exec_info(&self) -> Option<&ExecInfo> {
        self.exec_info.as_ref()
    }

    pub(super) fn verify_direct_generations(
        &self,
        reader: &AnchoredDir,
        expected: &[(PathBuf, FileGenerationIdentity)],
    ) -> Result<(), TraceError> {
        for (relative, expected_generation) in expected {
            let actual = reader.inspect_regular_generation(relative)?;
            if actual != *expected_generation {
                return Err(TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: self.root.clone(),
                    message: format!(
                        "direct trace artifact changed while loading: {}",
                        relative.display()
                    ),
                });
            }
        }
        Ok(())
    }
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
        let dynamic_file = reader
            .read_bounded_regular(Path::new("dynamic_trace.pb"), DYNAMIC_TRACE_PB_MAX_BYTES)?;
        let mut generations = vec![(
            PathBuf::from("dynamic_trace.pb"),
            dynamic_file.generation_identity(),
        )];
        let dynamic_trace_pb = dynamic_file.into_raw();
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty {
                path: root.join("dynamic_trace.pb"),
            });
        }
        let exec_info = read_direct_exec_info(&reader)?;
        if let Some((_, generation)) = &exec_info {
            generations.push((PathBuf::from(DIRECT_EXEC_INFO_RELATIVE), *generation));
        }
        let allocations_file = reader.read_optional_bounded_regular(
            Path::new("extra_info/allocations.json"),
            ALLOCATIONS_JSON_MAX_BYTES,
        )?;
        if let Some(file) = allocations_file.as_ref() {
            generations.push((
                PathBuf::from("extra_info/allocations.json"),
                file.generation_identity(),
            ));
        }
        let allocations_json = allocations_file.map(AnchoredFile::into_raw);
        let stats_file =
            reader.read_optional_bounded_regular(Path::new("stats.csv"), STATS_CSV_MAX_BYTES)?;
        if let Some(file) = stats_file.as_ref() {
            generations.push((PathBuf::from("stats.csv"), file.generation_identity()));
        }
        let stats_csv = stats_file.map(AnchoredFile::into_raw);
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
            generations.push((PathBuf::from(relative), entry.file.generation_identity()));
            payload_by_path.insert(root.join(relative), entry.file.into_raw());
        }
        let trace = Self::from_loaded_bytes(
            root,
            dynamic_trace_pb,
            exec_info.map(|(info, _)| info),
            allocations_json,
            stats_csv,
            payload_by_path,
        )?;
        trace.verify_direct_generations(&reader, &generations)?;
        Ok(trace)
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
        let dynamic_file = reader
            .read_bounded_regular(Path::new("dynamic_trace.pb"), DYNAMIC_TRACE_PB_MAX_BYTES)?;
        let mut generations = vec![(
            PathBuf::from("dynamic_trace.pb"),
            dynamic_file.generation_identity(),
        )];
        let dynamic_trace_pb = dynamic_file.into_raw();
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty {
                path: root.join("dynamic_trace.pb"),
            });
        }
        let exec_info = read_direct_exec_info(&reader)?;
        if let Some((_, generation)) = &exec_info {
            generations.push((PathBuf::from(DIRECT_EXEC_INFO_RELATIVE), *generation));
        }
        let allocations_file = reader.read_optional_bounded_regular(
            Path::new("extra_info/allocations.json"),
            ALLOCATIONS_JSON_MAX_BYTES,
        )?;
        if let Some(file) = allocations_file.as_ref() {
            generations.push((
                PathBuf::from("extra_info/allocations.json"),
                file.generation_identity(),
            ));
        }
        let allocations_json = allocations_file.map(AnchoredFile::into_raw);
        let stats_file =
            reader.read_optional_bounded_regular(Path::new("stats.csv"), STATS_CSV_MAX_BYTES)?;
        if let Some(file) = stats_file.as_ref() {
            generations.push((PathBuf::from("stats.csv"), file.generation_identity()));
        }
        let stats_csv = stats_file.map(AnchoredFile::into_raw);
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
        generations.extend(
            roster_by_path
                .values()
                .map(|(relative, generation)| (relative.clone(), *generation)),
        );

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
        let trace = Self {
            lineage: TraceLineage::new(),
            async_sidecars: OnceLock::new(),
            root,
            trace,
            exec_info: exec_info.map(|(info, _)| info),
            memory_payload_contract,
            allocations,
            #[cfg(feature = "campaign")]
            sampling: None,
            stats,
            validation,
            payload_by_path: BTreeMap::new(),
            raw_payload_retained: false,
            threadblock_by_path,
        };
        trace.verify_direct_generations(&reader, &generations)?;
        Ok(trace)
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
            None,
            allocations_json,
            stats_csv,
            threadblock_payload.by_path,
        )
        .map(|mut trace| {
            trace.sampling = sampling;
            trace
        })
    }

    pub(super) fn from_loaded_bytes(
        root: PathBuf,
        dynamic_trace_pb: Vec<u8>,
        exec_info: Option<ExecInfo>,
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
            lineage: TraceLineage::new(),
            async_sidecars: OnceLock::new(),
            root,
            trace,
            exec_info,
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

    /// Parse native async sidecars once for this loaded trace, then reuse the
    /// compact per-launch observations. Reopening a trace never reuses a
    /// process-global path cache. Dropping the trace frees the index.
    pub fn async_sidecars(
        &self,
    ) -> Result<&crate::async_sidecars::AsyncSidecars, crate::async_sidecars::AsyncSidecarError>
    {
        self.async_sidecars
            .get_or_init(|| {
                let mut streams: BTreeMap<(i64, i32), BTreeSet<i64>> = BTreeMap::new();
                for loc in self.kernels() {
                    streams
                        .entry((loc.device_id(), loc.kernel().id))
                        .or_default()
                        .insert(loc.stream_id());
                }
                crate::async_sidecars::AsyncSidecars::load(&self.root, &streams)
            })
            .as_ref()
            .map_err(Clone::clone)
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
    /// Map fields are BTreeMaps (see pbgen), and launch IDs are the producer's
    /// per-device sequence numbers.  The protobuf repeated-field order is not
    /// an identity guarantee: a merge/re-encoder can legally reorder kernels.
    /// Sort each stream by its unique launch ID so this reader agrees with the
    /// Python admission path (`_ordered_launches`) and stream-L2 replay even
    /// when the wire vector is out of order.
    pub fn kernels(&self) -> impl Iterator<Item = KernelLoc<'_>> {
        let mut launches = Vec::new();
        for (&device_id, device) in &self.trace.gpu_device {
            for (&stream_id, stream) in &device.streams {
                let mut stream_launches: Vec<_> = stream
                    .kernels
                    .iter()
                    .map(|kernel| KernelLoc {
                        device_id,
                        stream_id,
                        kernel,
                    })
                    .collect();
                stream_launches.sort_by_key(|loc| loc.kernel.id);
                launches.extend(stream_launches);
            }
        }
        launches.into_iter()
    }

    pub(super) fn require_kernel_loc(&self, loc: &KernelLoc<'_>) -> Result<(), TraceError> {
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

    /// Project a launch's validated per-lane GLOBAL memory identities using
    /// the same per-lane memory occurrence convention as the NVBit value
    /// capture (`0` for the first global-memory record of a lane). The launch
    /// component is the trace kernel id; a sidecar producer must bind its
    /// launch_id to that same identity. This is an identity/topology
    /// projection only: it does not prove L1 residency,
    /// ownership, or writeback completion.
    pub fn dynamic_memory_join_locations(
        &self,
        loc: &KernelLoc<'_>,
    ) -> Result<BTreeMap<DynamicMemoryJoinIdentity, DynamicMemoryJoinLocation>, TraceError> {
        self.require_kernel_loc(loc)?;
        let launch_id =
            u64::try_from(loc.kernel.id).map_err(|_| TraceError::TraceCountOverflow {
                counter: "kernel launch identity",
            })?;
        let blocks = self.validated_threadblocks_in_launch_order(loc)?;
        let mut locations = BTreeMap::new();
        for (_, block) in blocks {
            let cta = block.block_id();
            for warp in block.warps() {
                let warp_id =
                    u32::try_from(warp.warp_id()).map_err(|_| TraceError::TraceCountOverflow {
                        counter: "trace warp identity",
                    })?;
                let mut occurrence_by_lane = [0_u32; 32];
                for instruction in warp.instructions() {
                    let Some(memory) = instruction.memory() else {
                        continue;
                    };
                    if !matches!(warp.memory_route(memory), Ok(ValidatedMemoryRoute::Global)) {
                        continue;
                    }
                    let mask = instruction.predicate_on_mask();
                    if mask == 0 {
                        continue;
                    }
                    for lane in 0..32_usize {
                        if mask & (1_u32 << lane) == 0 {
                            continue;
                        }
                        let occurrence = occurrence_by_lane[lane];
                        occurrence_by_lane[lane] =
                            occurrence
                                .checked_add(1)
                                .ok_or(TraceError::TraceCountOverflow {
                                    counter: "per-lane global memory occurrence",
                                })?;
                        let key = DynamicMemoryJoinIdentity {
                            launch_id,
                            function_unique_id: instruction.function_unique_id(),
                            cta,
                            warp: warp_id,
                            lane: u32::try_from(lane).unwrap_or_else(|_| unreachable!()),
                            pc: instruction.pc(),
                            occurrence,
                        };
                        let location = DynamicMemoryJoinLocation {
                            key,
                            block_id: cta,
                            warp: warp_id,
                            instruction_index: instruction.instruction_index(),
                            launch_index: block.launch_index(),
                            warp_slot: instruction.warpid_sm(),
                            access_kind: memory.replay_meta().access_kind,
                            sm_id: instruction.sm_id(),
                        };
                        if locations.insert(key, location).is_some() {
                            return Err(TraceError::DynamicMemoryJoinIdentityDuplicate);
                        }
                    }
                }
            }
        }
        Ok(locations)
    }

    pub fn dynamic_memory_join_identities(
        &self,
        loc: &KernelLoc<'_>,
    ) -> Result<BTreeSet<DynamicMemoryJoinIdentity>, TraceError> {
        Ok(self
            .dynamic_memory_join_locations(loc)?
            .into_keys()
            .collect())
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
