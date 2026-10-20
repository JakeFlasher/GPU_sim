// Lexical implementation section 3/5; included by the parent facade.
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

    fn from_loaded_bytes(
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

