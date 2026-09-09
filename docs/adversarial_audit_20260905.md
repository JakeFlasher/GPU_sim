# Adversarial simulator audit

Updated 2026-09-08.

This record covers the NVBit → enhanced protobuf → Rust trace reader → SM86
scheduler/memory model → NCU-shaped result path. It records confirmed
correctness fixes, bounded performance changes, representative execution
evidence, and remaining architecture-card limits.

## Current retained inputs

The runnable regenerated cohorts are:

- `results/captures/sm86_regenerated_workloads_small_20260905_v1` (8 workloads)
- `results/captures/sm86_regenerated_workloads_large_20260905_v1` (7 workloads)

The active capture tree retains the current/best comparison and diagnostic
records needed for simulator and measurement work.

## Confirmed fixes

### Input and metadata boundaries

- Rust and Python protobuf preflight reject over-wide uint32, signed-int32, and
  enum varints before generated decoding; map keys, embedded warp IDs,
  dimensions, IDs, and repeated capability values cannot silently narrow.
- Duplicate protobuf map keys and duplicate map-entry value fields are rejected.
- Shared JSON preflight rejects duplicate members, invalid UTF-8, raw or decoded
  NULs, trailing bytes, and excessive nesting before materialization.
- Recursive DOM duplicate-key validation handles zero-length string storage
  without constructing from a null pointer.
- Architecture cards, P-chase cards, launch manifests, static execution
  metadata, and the direct simulator wire consumer share the strict JSON helper.
- Static and dynamic launch identities are checked before payload joins; kernel
  IDs, register counts, PCs, lane masks, address spaces, dimensions, and
  memory widths are range-checked.
- Direct replay binds a canonical `<benchmark>.control_bits.json` to the
  recorded control-bits binary basename, accepting the native `_run` wrapper
  form. Shape-specific or otherwise transformed sidecar names remain valid
  compatibility inputs.
- Trace launch iteration is numerically ordered by producer kernel ID, so
  reordered protobuf vectors cannot change stream-L2 selection.

### Enhanced C++ parser

- SASS guards accept the complete canonical `@P0..@P6`, `@PT`, and uniform
  forms. Reserved operand tokens (`RZ`, `URZ`, `PT`, `UPT`, `PR`, `UPR`)
  require exact spelling; suffixed aliases are rejected.
- Serialized execution guards are limited to predicate slots 0–7. Whole
  predicate-register sentinels remain available for operand-use accounting.
- MREF and CBANK shapes require balanced, nonempty components. Register,
  immediate, modifier, tensor-shape, encoded-word, and control-bit parsing now
  fails explicitly on malformed input.
- Instruction/kernel/execution deserializers build complete candidates before
  publication. Duplicate PCs or kernel names, invalid nested records, and
  failed retries cannot poison an already valid object.
- Explicit and implicit PC insertion advances from the highest published PC and
  never overwrites an occupied entry.
- Public lookup paths use non-mutating access. Missing entries throw typed
  `std::out_of_range` exceptions or return `false`/null according to the API;
  they do not insert null map values in release builds.
- Conflicting duplicate no-binary registrations return `false` in all build
  modes. Null operands and absent tensor-shape metadata fail explicitly.
- Tensor register-use accounting follows the PTX fragment layout: dense IMMA
  and BMMA A/B inputs are not halved, while sparse integer IMMA halves only
  the structured-sparse A fragment. This fixes dense `IMMA.16816.S8.S8` from
  1/0 to 2/1 A/B registers without changing sparse A/B counts.

### Rust replay and topology

- Compact backward-branch promotion requires the adjacent producer to write the
  branch’s consumed predicate; unconditional aliases and unrelated producers
  remain ordinary target-resolution intervals.
- Predicate sources/destinations reject data modifiers, repeated negation,
  subselectors, and unsupported uniform aliases.
- LDGSTS source/destination pairing preserves executing `ignore-src` lanes as
  destination zero-fill work, rejects missing paired lanes, and keeps bypass
  copies in async-group bookkeeping.
- Compact LDGSTS topology conversion returns a typed out-of-range error instead
  of panicking when a u32 projection cannot fit its compact representation.
- Address compression uses signed checked differences; negative non-sentinel
  addresses are rejected.
- L1/L2 observers, response IDs, immediate-constant ordering, instruction-cache
  sequencing, and instruction-front-end transactions commit atomically on
  failure.

### Native tracer and performance

- NVBit fallback parser results, allocation/channel buffers, CUDA synchronization,
  stats CSV, sidecars, protobuf writes, and shell-constructed paths are checked
  before use or publication.
- Launch products and counters use checked 64-bit arithmetic; RFU and uniform
  register conversions require complete tokens and architecture bounds.
- The tracer Makefiles rebuild CUDA objects when enhanced headers change, include
  architecture in object names, emit dependency files, and link with unresolved
  symbols rejected.
- Protobuf binding modules are reused across threadblocks and the complete
  binding audit runs at the load boundary, removing repeated admission work
  without changing the contract.
- Hash maps replace repeated wide-batch line/sector scans where ordering and LRU
  semantics are preserved.
- CUDA launch callbacks preserve distinct stream ordinals through dynamic
  protobuf, threadblock filenames, stats, sampling scope, and accepted-launch
  metadata. Forced stream-zero attribution is gone, so replay can reject or
  isolate multi-stream traces instead of inventing same-stream cache history.
- The active stream/kernel snapshot is now stored per CUDA context, while
  launch IDs remain globally unique per device and `cuStreamGetId` supplies a
  program-lifetime driver stream ID. Concurrent receiver threads therefore
  cannot relabel each other's batches.
- Cohort validation compares the captured stream ID against stats instead of
  assuming every launch uses stream 0.
- `_ptsz` launches with a null handle resolve `CU_STREAM_PER_THREAD` through
  `cuStreamGetId_ptsz`; ordinary launches retain legacy/null semantics.
- Stream-ordered async-free retirement now keys pending frees by the resolved
  driver stream ID, so a per-thread/default stream or recycled handle cannot
  retire another stream’s allocation.
- Allocation and VMM interval endpoints are checked before publication or
  teardown; VMM release cleanup removes any intersecting live mapping, and
  allocation/context IDs stop at the positive 32-bit boundary instead of
  wrapping into the unresolved sentinel.
- Stream-ordered pool allocations now use a process-wide UVA map and record the
  allocation device from `cuPointerGetAttribute`, so async frees from another
  context retire the correct allocation. Each accepted launch snapshots live
  generations before receiver release; delayed channel records therefore keep
  old/new same-base reuse identities distinct.
- Launch snapshots use sorted context-bound and context-independent interval
  vectors with logarithmic lookup. A 200,000-query host benchmark improved from
  1.11 s to 23 ms at 10,000 live allocations while preserving overlap and
  unresolved results.
- Sequential CUDA contexts now retain all accumulated protobuf kernels and
  de-duplicate repeated capability fields; a two-context kernel probe emits
  both stream 7 and stream 14 kernels in the final trace. Context-bound
  allocations are marked dead before sidecar publication at context teardown.
- The two-context trace passes admission with two launches, 38 dynamic records,
  two threadblock payloads, and two distinct stream IDs. A direct CUDA probe
  confirms `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)` is
  valid immediately after `cuMemAllocAsync`, including before stream sync.
- Context teardown now skips receiver joins for contexts that never initialized
  an NVBit channel. An allocation-only two-context CUDA probe exits zero,
  completes `nvbit_at_ctx_term`, and frees the cross-context async
  allocation successfully. The same-base async-reuse probe exits zero with no
  unmapped or cross-allocation exclusions.
- `cuCtxSynchronize_v2(NULL)` now retires pending async frees against the
  callback's current context. A real injected probe changed the allocation
  sidecar from live to freed after the null-context synchronization, with zero
  exit status.
- A sequential-context protobuf containing two kernels passes trace admission
  and the Rust machine-local full walk with 2 launches, 2 blocks, 20
  instructions, and 2 memory records; no consumer contract change was needed.
- The metric extractor no longer caches stream IDs by reusable opaque handles.
  It resolves CUDA Driver context/stream IDs per launch, preserves the
  per-thread-stream entry point, and keys pending graph state by `(context,
  stream)`.
- The metric JSONL records their `cuda_driver` identity namespace explicitly;
  the wall-clock join uses only singleton launch signatures across separate
  runs and never compares process-local Driver/CUPTI numbers. Repeated
  signatures are ambiguous, while stream/context IDs remain diagnostics.
- The metric-extractor Makefile exposes only its runnable build and clean
  targets; the active capability inventory contains only existing entry
  points.
- The active capture budget dependency is now a concise parser at
  `tools/cloud/trace_budgets.py`, and SM86/SOL campaign imports resolve.
- The budget loader classifies invalid UTF-8 as its typed `SchemaInvalid`
  error instead of leaking a decoder exception.
- The NVBit runtime loader no longer performs repeated SHA-256 admission or
  copy checks, and a rebuild does not get rejected because the descriptive
  tracer-size field changed. Manifest hash fields remain compatibility
  metadata; path, inode, DSO-size, bounded-read, and loader-resolution checks
  remain active.
- Dispatch planning records the solution identity once for cache metadata;
  the redundant post-run SHA-256 immutability scan was removed.
- Calibration sidecar loading now treats ledger and split digests as inert
  provenance fields; it no longer rehashes those files or rejects a load on a
  digest mismatch. Schema, architecture, metric-family, and anchor validation
  remain active.
- Surface cache identity construction now propagates a failed
  surface-specific dispatch composition instead of silently hashing the base
  command and risking reuse of evidence from a different invocation shape.
- Campaign result and error emission no longer reparses and rehashes an already
  closed wire document; one seal computation and one canonical serialization
  remain.
- Admitted trace source reads now rely on the retained descriptor generation
  and size checks instead of rehashing every payload read; initial admission
  metadata remains available to downstream consumers.
- The CUPTI/NVBit wall-clock joiner now rejects invalid UTF-8, malformed, or
  non-object JSONL lines with a specific input error instead of silently
  undercounting a surface.
- The static/dynamic Blackwell bucket cross-checker now applies the same
  strict JSONL rule; malformed, non-object, and invalid-UTF-8 launch records
  fail with a line-specific error instead of being dropped from totals.
- Static and dynamic cross-check inputs are read once; the retained artifact
  digest and decoded content share the same byte buffer, avoiding duplicate
  file I/O without changing the recorded metadata.
- Host-tool subprocess captures in the active surfaces, preflight, native
  dispatch, and calibration wrapper decode diagnostic bytes with replacement;
  non-UTF-8 CUDA-tool output therefore becomes a typed result instead of
  aborting the harness with `UnicodeDecodeError`.
- The native capture wrapper applies the same replacement decoder to its
  `Popen` stream, so a child diagnostic cannot abort capture cleanup during
  `communicate()`.
- Direct SOL harness/scaffold entrypoints now resolve their current package
  locations under `tools/sol_baselines_*`; `report.py`, `run.py`,
  `validate.py`, and `scaffold.py` execute their help paths successfully.
- Harness defaults now point at the current `tools/sol_execbench` data,
  examples, source, ledger, and report paths. An empty-root report run exits
  zero and writes a valid empty report using those defaults.
- The scaffold manifest CLI now reads the current evaluator tree and emits all
  235 mapped entries without relying on the removed `scripts/` layout.
- Native dispatch now resolves the in-tree
  `tools/sol_execbench/.venv/bin/sol-execbench` executable (or an explicit
  override/PATH fallback), so direct runs no longer depend on an activated
  shell environment.
- The surface dispatch plan reuses that same executable resolver, and malformed
  cache records are treated as misses instead of raising during tuple
  conversion.
- The surface CLI and orchestrator now use the current bundle and evaluator
  paths; `list` and Stage-1 `dry-run` both complete successfully with six
  registered surfaces.
- The SOL score helper now imports from the active evaluator source tree and
  resolves successfully in a direct module probe.
- The harness package exports compatibility aliases at package import time, so
  direct imports of dispatch-plan and surface modules resolve without relying
  on entrypoint bootstrapping.
- Identity-bearing native-environment exceptions are shared across qualified
  and compatibility module names, preventing callers from missing a typed
  degraded-environment error due to duplicate module loading.
- Native-environment fingerprints launched from system Python now probe the
  evaluator's managed venv for Python, Torch, CUDA, and cuDNN versions; system
  and managed invocations produce the same complete component set.
- Managed version-probe caching is keyed by the selected interpreter, so an
  explicit evaluator override cannot reuse canonical-venv values in a
  long-lived process.
- Cache evaluator identity now fingerprints Git-visible working-tree content,
  including uncommitted evaluator edits and non-ignored additions; a temporary
  mutation probe changed and then restored the fingerprint as expected.
- Live cache-predictor and predictor-design loaders now resolve the retained
  calibration and 235-row host-architecture manifests under
  `tools/sol_baselines_harness`; the sm_86 pilot variant resolves its sidecar
  from the same current tree instead of the retired `scripts/` layout.
- Predictor-design manifest loading treats invalid UTF-8 as an unavailable
  optional extension instead of leaking a decoder exception during import.
- Scaffold package documentation and generators now describe only the current
  `tools/` layout; manifest generation emits 235 mapped entries from the active
  evaluator tree without references to removed package locations.
- The deployed raw cache predictor no longer imports scikit-learn eagerly;
  calibration reconstruction loads that optional dependency only when a fitted
  sidecar is requested and reports `calibrator_dependency_missing` when the
  optional package is unavailable.
- Cache derivation and record matching now reject an unavailable evaluator or
  native-environment identity instead of allowing two `unknown` values to
  match; empty and non-string supplied identities are rejected as well.
- B200 preflight now resolves the active NVBit metric-extractor source and
  include paths; a local preflight reports both `sm_100a_compile_proof` and
  `nvbit_metrics_tool` as ready. The nonexistent native cache-replay probe was
  removed from the active readiness manifest; Python/Rust replay is canonical.
- Both local-smoke and phase-2 preflight modes now complete with
  no blockers; only phase-2 manifests set `ready_for_phase_2=true`, while
  local-smoke manifests set the separate `ready_for_local_smoke` flag.
- The live `nvbit_metrics` surface now points at the same active
  `tools/nvbit_tracer/metric_extractor/metric_extractor.so` path as preflight,
  eliminating a false unavailable result during an orchestrated run.
- The CUPTI activity surface now locates the active evaluator timing module,
  so its probe reports `ok` instead of missing the renamed source directory.
- The SM86 coverage-row emitter now writes current `tools/` trace paths in its
  report records, keeping generated evidence links executable from this
  checkout.
- Deferred-ledger lint treats `harness_commit` as opaque provenance text
  rather than a digest-format gate. The current 23-entry ledger now supports
  an empty-root report run that exits zero and writes JSON/Markdown output.
- Optional KB-routing tests now skip when their removed routing source tree is
  unavailable; the remaining scaffold/harness contract tests pass without
  restoring that source tree.
- The optional KB-manifest/FTS tests use the same source-presence guard, so
  the focused scaffold/harness suite now reports 9 passed and 8 explicit
  skips instead of failing during collection.
- The combined focused Python run reports 41 passed and 14 explicit skips;
  all skips correspond to unavailable external fixtures or removed optional
  source trees.
- Graph capture initialization is capture-safe: channel allocation is deferred
  until capture ends, and zero-initialized device counters are seeded on the
  first post-capture callback. The tracer now fails closed at `cuGraphLaunch`
  because the available NVBit surface has no per-node channel identity/flush
  path; this prevents an empty graph trace from being mistaken for a valid
  replay input.
- Capture bookkeeping commits only after a successful begin-capture callback,
  and inactive launches guard the uninitialized stop-report buffer. A failed
  capture followed by an ordinary kernel exits zero and still records one
  kernel.
- Context initialization now uses the typed context lookup rather than a
  mutating map access, so an unregistered handle cannot create a null state
  and continue into channel setup.
- The injected tracer payload now value-initializes optional register arrays,
  captures all four words of a `REGULAR_4_REGS` operand, and makes the
  stop-report early exit warp-uniform so later shuffles never name an exited
  lane.
- After rebuilding the CUDA tracer, a fresh three-kernel task-done capture
  exits zero; trace admission accepts 3 launches, 3 threadblocks, 81,433
  instructions, and 296,236 records, and `trace_printer` opens the result
  without diagnostics.
- The capture-environment validator accepts the explicit
  `allow_multiple_launches` field emitted by multi-kernel probes while keeping
  single-launch records backward compatible.
- The capture output contract now checks unique trace-launch IDs, contiguous
  accepted ordinals, module-mapping cardinality, and channel-record totals;
  a tampered multi-launch mapping is rejected before publication.
- A temporary `uint4` vector-copy CUDA probe exercises the rebuilt tracer's
  vector-register path; capture and admission both succeed with one launch,
  11 instructions, and 19 records.
- Its generated SM86 control sidecar decodes 24 instructions with no unknown
  encodings; two `sim-run` replays both exit zero with empty stderr and produce
  identical 13,214-byte JSON reports.
- The capture campaign's trace-budget dependency is now a concise current
  parser at `tools/cloud/trace_budgets.py`; imports for the SM86 and SOL
  capture paths resolve without restoring unrelated cloud preflight tools.
- The SOL timing demangler now treats `__cxa_demangle` output as owned native
  memory, copies it before decoding, and frees it on success, status failure,
  or UTF-8 decode failure. The deallocator is resolved from the process-wide
  symbol scope so allocator interposition remains compatible. This keeps
  repeated CUPTI activity processing from accumulating native allocations.
- The public CUDA-event timing helper supplies a no-op setup callback when
  callers omit setup, and invalid timing methodology names return a typed
  `ValueError` before entering the timing loop.
- Timing and allocator entry points reject negative warmup, non-positive rep,
  and non-positive pool iteration counts before allocating CUDA state.

## Executable evidence

- `cargo fmt --all -- --check` passes.
- `cargo test -p sim-trace --lib`: 129 passed, including typed LDGSTS
  topology overflow, duplicate JSON/wire members, identity joins, and
  transaction failures.
- `cargo test -p sim-trace --lib --features campaign`: 169 passed, including
  rejection of sparse dynamic and threadblock payloads above their default
  bounds before materialization.
- `cargo test -p sim-sm sass_operands --lib`: 14 passed.
- The release native tracer build succeeds with CUDA 13.2, GCC 15, and the
  pinned protobuf prefix.
- The resumed release sweep reran all 8 small and 7 large retained workloads;
  every run exited zero with empty stderr and matched its retained JSON
  result byte-for-byte.
- A final retained HMMA small replay produced 15,042 bytes and a final
  feedback-enabled SwiGLU large replay produced 15,173 bytes; both matched
  the retained comparison JSON exactly with empty stderr.
- The runtime-environment CLI accepts the rebuilt tracer with the retained
  manifest, and a temporary manifest containing invalid SHA fields still
  loads, confirming those fields are inert metadata.
- The active graph probe now exits with a typed unsupported-graph failure
  before publishing an empty trace; the same graph used outside the active
  profiler region exits zero without a tracer crash.
- `cargo build --release -p sim-run-campaign` succeeds with the campaign
  snapshot path and the new finite capture defaults.
- The campaign package suite passes 68 library tests, 8 CLI tests, 20 retained
  transport tests, 19 end-to-end tests, and 1 registry-lockstep test after the
  emission-path simplification.
- A full `cargo test --workspace` rerun passes across the current crates,
  including 48 architecture, 9 event-engine, 125 memory, 433 SM, and 169
  campaign-enabled trace library tests plus all campaign integrations and
  doctests; only machine-local GPU fixtures remain explicitly ignored.
- A freshly captured three-launch task-done trace replays successfully in
  both launch-scoped and stream-L2 modes; the stream-L2 report is deterministic
  across two runs and records the expected cross-launch cache session.
- The campaign runner completes its retained canary fixture with zero exit and
  empty stderr after the bound change.
- Rebuilding after the final DOM guard accepts all 15 regenerated execution
  documents and rejects the malformed metadata probe corpus.
- Assertions-disabled C++ probes pass for exact predicate/register tokens,
  malformed operands, transactional metadata, missing lookups, null operands,
  and absent tensor metadata.
- A focused release C++ probe reports PTX-consistent tensor fragment counts for
  dense IMMA, sparse IMMA, BMMA `.b1`, and HMMA BF16. A rebuilt tracer captures
  the portable INT8 IMMA workload successfully; admission and `trace_printer`
  both exit zero.
- A two-stream CUDA probe captures two launches on distinct driver stream IDs;
  launch-scoped replay reports both launches, while stream-L2 mode rejects the
  mixed-stream trace before replay as required.
- A default-stream `gemm_imma_int8_sm80` capture still admits one launch and
  replays twice deterministically after the stream-ID change.
- A CUDA program compiled with `--default-stream per-thread` captures two
  null-handle launches as distinct driver IDs and replays both deterministically.
- An async-allocation/free probe retires its allocation only after stream
  synchronization; admission and two simulator replays pass with no exclusion
  records.
- A host-only registry probe covers overflowing ranges, duplicate/overlapping
  registrations, pending same-base reuse, partial unmaps, and VMM intersection
  cleanup; all boundary assertions pass.
- The metric-extractor `sm_86` shared-library build succeeds with the Driver
  identity ABI. An injected two-stream probe exits zero and emits two bounded
  records with distinct stream IDs; reordered synthetic CUPTI records match
  2/2 by singleton signatures, while repeated signatures become ambiguous.
- The metric extractor's per-thread-default-stream probe exits zero and emits
  two distinct Driver stream IDs (13 and 14) through the `_ptsz` path.
- Synchronous `cuMemcpyHtoD_v2` and its per-thread-default variant now resolve
  the selected default stream through `cuStreamGetId`/`cuStreamGetId_ptsz`
  before entering the protobuf stream map. On the pinned host the legacy probe
  returns Driver stream ID 7, so a literal zero would have separated
  memcpy ordering from default-stream kernels.
- A fresh injected CUDA probe that performs `cuMemcpyHtoD` followed by one
  kernel publishes a single protobuf stream with ID 7 and two ordered events;
  the stream identity and event count are visible in `trace_printer`.
- The tracer now covers legacy/v2 synchronous and asynchronous HtoD callback
  variants, including PTDS/PTSZ default-stream entry points. A sync+async+
  kernel probe emits the sync event on stream 7 and the async event before its
  kernel on stream 13; a per-thread-default null-stream probe preserves its
  distinct stream identity.
- A 10,000-call long-symbol FFI probe now changes live native allocation by
  only 784 bytes, versus about 10.4 MB before ownership was fixed; malformed
  and non-mangled names retain their original fallback behavior.
- A real one-iteration CUDA-event call with `setup=None` returns timing data,
  and an invalid methodology call returns `ValueError("Unknown timing
  methodology: bad")`.
- Invalid-count probes cover both timing methods, `time_runnable`, and the
  shifting allocator; each fails with the expected `ValueError`, while a
  valid one-iteration event run succeeds.
- The allocator's storage-byte source view no longer allocates a second full
  tensor. A 64 MiB CUDA input now adds one 64 MiB pool allocation at peak
  instead of a 128 MiB pool-plus-source spike, while alias and stride probes
  remain correct.
- CUPTI timing teardown now propagates cleanup failure on an otherwise
  successful run, while preserving the original benchmark exception when both
  execution and teardown fail. A fake-CUPTI probe covers both paths.
- The metric-extractor README and Makefile describe its current build path,
  which succeeds from the repository checkout.
- The complete retained release replay sweep covers 8 small and 7 large
  workloads: 15/15 exit zero, stderr is empty, and the current response set is
  byte-stable against the current comparison outputs except the documented
  corrected unmodeled-DRAM gauges.
- The rebuilt direct runner rejects a canonical canary sidecar whose recorded
  binary is changed to `unrelated_rebuilt_kernel` (exit 1, no report), while
  the intact canary still exits zero with 54 issued instructions.
- The current direct-run library and CLI suites pass 20 and 8 tests,
  respectively, including malformed candidate documents, scope conflicts,
  stale launch keys, and closed-output handling.
- The manifest-driven `sim-run-campaign` CLI returns structured usage and
  manifest-admission errors for zero-argument and missing-manifest invocations
  without panicking.
- SM launch-evidence admission now applies the same canonical sidecar binary
  basename binding as direct replay. Its regression fixture rejects a canary
  sidecar naming an unrelated executable; the focused admission suite passes
  all 25 tests.
- The full configured replay sweep remains
  byte-stable: 8/8 small outputs and 7/7 large feedback-enabled outputs match
  the current comparison responses, with zero stderr.
- All 15 retained NVBit directories pass
  `trace_admission.py` with zero exit and empty stderr.
- Current representative replays are deterministic:
  HMMA small emits 15,042-byte JSON twins; SwiGLU large with admission feedback
  emits 15,173-byte JSON twins. Both runs exit zero with empty stderr.
- The release-binary replay confirms the latest baselines: two HMMA
  runs are byte-identical to the latest reference output, and the SwiGLU
  feedback run is byte-identical to its retained regenerated output. No new
  simulator regression was observed.
- A full release sweep covers all 15 retained workloads and matches
  the current response set exactly: 8 small and 7 large, zero exit, empty stderr,
  and no output changes.
- A fresh pinned-host SM86 HMMA capture produces 422,400 recorded instructions,
  passes metadata deserialization and admission, and replays twice with
  identical 15,034-byte JSON and empty stderr. Capture stderr contains only the
  known 462-byte NVBit initialization/fallback warning.
- Fresh captures can vary in dynamic protobuf bytes while `stats.csv` remains
  stable; timing differences are therefore capture-input variation, not fixed
  trace nondeterminism. Seven-repeat locked-clock NCU regeneration shifted
  medians by at most 5.33% (small) and 1.18% (large); indicative strict maxima
  remain 8.44% and 8.72%.
- The timing allocator now groups tensor arguments by PyTorch `UntypedStorage`
  and reconstructs each view's shape, stride, storage offset, `requires_grad`,
  conjugate, and negative metadata. A CPU alias probe confirms that overlapping
  input views still share mutations and that successive iterations use distinct
  storage offsets.
- CUPTI teardown is exception-safe: enabled activity kinds are disabled and
  `finalize()` is attempted when setup, synchronization, or the benchmark
  callable raises. A fake-CUPTI failure probe observes all five disables and one
  finalization after the callable raises.
- The joiner now recognizes only CUPTI `KERNEL=3` and
  `CONCURRENT_KERNEL=10`; runtime API records (`RUNTIME=5`) are excluded even
  when their `cbid` is absent. A retained-corpus scan found no local enum
  remapping, and synthetic kind probes pass.

## Primary research

Bounded source clones are isolated at
`/tmp/gpu_sim_external_research_20260905`: Accel-Sim `d930ad6d`,
GPGPU-Sim `a4ce3fea`, MGPUSim `d44e03fd`, NVBit `73092be6`, and ACALSim
`45f4b745`. Related papers are retained beside the clones.

- [Analyzing and Improving Hardware Modeling of Accel-Sim](https://arxiv.org/abs/2401.10082)
  motivates explicit front-end, result-bus, memory-pipeline, predicate,
  uniform-register, tensor, and register-bank checks.
- [Exploring Modern GPU Memory System Design Challenges](https://arxiv.org/abs/1810.07269)
  supports counter-by-counter validation and treats cycle noise separately from
  hardware counters.
- [MGSim/MGMark](https://arxiv.org/abs/1811.02884) supports event-driven
  wakeups and explicit component communication.
- [ACALSim](https://arxiv.org/abs/2605.22936) supports deterministic event
  ordering, fast-forwarding, and explicit backpressure.
- The [NVBit MICRO 2019 paper](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_NVBit.pdf)
  supports active-mask, predicate, and register capture boundaries.
- The vendored NVBit `instr_count_cuda_graph` sample was checked for graph-node
  callback ordering; it confirms that graph kernels require explicit
  `cuGraphAddKernelNode`/`cuGraphLaunch` handling beyond ordinary launch hooks.
- NVIDIA’s [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
  and [CUDA-GDB register definitions](https://raw.githubusercontent.com/NVIDIA/cuda-gdb/nvidia-gdb-16.3/gdb/cuda/cuda-tdep.h)
  define the predicate/register forms used by the parser.
- The cloned Accel-Sim/NVBit instrumentation path and NVBit MICRO paper both
  rely on warp-wide active-mask collectives; the tracer now keeps stop-report
  control flow uniform before any masked shuffle, preserving that collective
  precondition.
- NVIDIA’s [CUDA Driver stream API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  defines program-lifetime `cuStreamGetId` identities, while the [Programming
  Guide’s per-thread entry-point section](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html)
  defines the `_ptsz` default-stream semantics used by the producer fix.
- The [MGPUSim source](https://github.com/sarchlab/mgpusim) and its
  [memory-system paper](https://arxiv.org/abs/1810.07269) likewise keep event
  ordering explicit across memory requests and responses; the queue audit found
  no corresponding simulator ordering defect.
- NVIDIA’s [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
  defines activity `streamId` values and `cuptiGetStreamIdEx` as
  context-unique, with `cuptiGetContextId` supplying a process-unique context
  identity; those values remain namespace-tagged diagnostics and are not
  compared with the NVBit process's Driver IDs.
- The [Itanium C++ ABI demangler contract](https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler)
  and [libstdc++ `__cxa_demangle` documentation](https://gcc.gnu.org/onlinedocs/libstdc++/latest-doxygen/a00026.html)
  support treating a null-output-buffer result as caller-owned storage; the
  timing FFI now releases that storage explicitly.
- NVIDIA’s [stream-ordered memory allocator guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
  permits memory-pool reuse after stream-ordered frees, which is why exact-base
  pending-free reuse remains an explicit registry case.
- NVIDIA’s [CUDA async-allocation API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)
  states that allocations are returned before the queued allocation operation
  completes, that a specified pool may belong to another device than the
  stream, and that stream synchronization establishes the completion boundary.
  The [unified-addressing contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html)
  defines pointer uniqueness for the 64-bit tracer process.
- NVIDIA’s [CUDA context API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  specifies that a null `cuCtxSynchronize_v2` context synchronizes the current
  context; the tracer now follows that rule for pending-free retirement.
- PyTorch’s [tensor view contract](https://docs.pytorch.org/docs/2.14/tensor_view.html#tensor-views)
  requires view mutations to share the underlying storage; the local PyTorch
  source clone at `/tmp/gpu_sim_external_research_20260908_pytorch_storage_views`
  follows this by rebuilding tensors with `set_` from untyped storage while
  retaining shape, stride, and storage offset.
- NVIDIA’s [CUPTI activity-kind definitions](https://docs.nvidia.com/cupti/13.0.0/api/group__CUPTI__ACTIVITY__API.html)
  assign `KERNEL=3`, `DRIVER=4`, `RUNTIME=5`, and
  `CONCURRENT_KERNEL=10`; the joiner uses these exact serialized values.
- NVIDIA’s [PTX ISA matrix-fragment tables](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-i8-f8)
  specify dense IMMA A/B/C/D fragments as 2/1/4/4 `.b32`/`.s32` registers
  for `m16n8k16`, and the [sparse IMMA tables](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-sparse-mma-16864-u8s8-fp8)
  specify the structured-sparse A reduction with an unchanged B fragment.

## Latest transactional-boundary finding

The SM issue-cycle review found that CTA retirement could remove the final
warp and then fail while admitting the next launch wave (for example, when the
arrival-stamp or unfinished-warp counter overflowed), leaving a partially
mutated actor. `preflight_issue_accounting` now projects successor residency,
physical slots, static PC joins, arrival stamps, and unfinished-warp cardinality
before any retirement mutation. Focused corruption probes and the full
`sim-sm` library suite pass (405 tests). The rebuilt release binary also
replays all eight regenerated small and seven regenerated large SM86 workloads
deterministically. All 15 outputs match the latest post-cleanup comparison
cohort; HMMA BF16 and feedback-enabled SwiGLU remain byte-identical to their
retained regenerated results.

Direct trace loading now captures and retains `enhanced_execution_info.json`
through the same anchored reader as the dynamic trace and payload tree. Its
generation is included in the final direct-loader comparison, so scheduling
metadata cannot be reopened from a different file generation during replay.
The direct and streaming loader paths remain output-equivalent on the canary
fixture and all 15 regenerated HMMA/attention/normalization replays. Synthetic
traces that omit execution metadata remain available to the low-level reader;
the direct runner reports a typed execution-info error when a runnable trace
lacks it.

The direct runner now applies the same bounded, no-follow anchored read and
post-read generation check to architecture cards, P-chase cards, and control
sidecars before parsing them. This closes the remaining direct-input rewrite
window without adding SHA-based admission work. A current-directory filename
edge in that loader was also normalized to `.` and covered by a focused
regression test.

The calibration diagnostic entry point still used the removed `scripts/`
bootstrap and manifest paths, so its direct `--help` invocation failed before
loading the package. The bootstrap, imports, default ledger/split paths, and
git-tree module path now use `tools/sol_baselines_harness`; the direct help
smoke exits successfully.

The trace-layout parser accepted both `sm86` and `sm_86` (and the sm120 forms)
but compared the raw suffix against canonical manifest identities, falsely
rejecting the underscore spellings. Suffix parsing now normalizes both forms;
malformed manifest host shapes also resolve to typed missing-architecture
outcomes instead of raising `AttributeError`.

The companion `accuracy_diagnostic_cli.py` had the same removed-package
bootstrap and failed before parsing `--help`. Its direct import now uses the
active `tools` package path; help exits cleanly and a missing report returns
the documented typed exit code 2.

The retained dossier and corpus-expansion CLIs had the same stale `scripts/`
bootstrap and `baselines_harness` imports. Their direct help surfaces now use
the active `tools/sol_baselines_harness` package and manifest paths; both
commands import and parse arguments successfully.

The deployed `solar_analysis` module also imported PyYAML at module import time,
making score/report imports fail on hosts without the optional SOLAR stack.
YAML is now loaded only when precision probing or SOLAR analysis is requested;
missing PyYAML returns the existing `unknown_t_sol` path while package imports
remain usable.

The isotonic calibration module had the same eager optional-dependency failure
for scikit-learn. Its import is now lazy at fit time, returning the typed
`calibration_dependency_missing` error when the package is unavailable while
keeping the predictor package importable.

Active SM120 coverage tools also carried stale `scripts/baselines_harness`
documentation and default-output paths after the package move. Those paths now
resolve through `tools/sol_baselines_harness`; all six active SM120 CLIs pass
direct `--help` and compile checks. Retired SM86/expanded artifact paths remain
unchanged as historical data.

The retained path-relocation test also encoded an obsolete assumption that all
five SM86 seed slugs must be excluded. The current host-architecture manifest
admits those seed rows, so the test now checks the actual contract: seed records
and seed exclusions are disjoint and cover the complete seed set. The retained
path and source-boundary tests pass (**16 total**).

The AC9 closeout emitter also retained a removed `scripts/baselines_harness`
output root. Its active calibration root now resolves to
`tools/sol_baselines_harness`; the emitter builds and runs with exit code 0
without changing its retained report schema.

The cache-key review reproduced acceptance of arbitrary nonempty hash strings
as a valid cache identity. Cache records and caller-supplied native identities
now require canonical 64-character lowercase digest text; malformed records
become cache misses without any extra content rehash.

The bundle-name regex used `\d`, which admits Unicode decimal digits despite
the ASCII slug contract. It now uses `[0-9]`; ASCII names remain accepted and
Unicode-digit names are rejected before routing.

The harness row collector also ignored malformed JSONL lines when other rows
were valid. Parse errors now always produce a failure reason, preventing a
partial evaluator stream from being accepted as a complete workload report.

The aggregate report had a matching cache path flaw: an identity-matched trace
with parse errors could still be replayed or used as fresh-run fallback when
its parsed rows happened to be nonempty. Candidate row shapes are now checked,
clean replay requires a structurally clean trace, and fallback rejects parser,
row-count, evaluator, and zero-row failures while preserving complete honest
non-PASSED evidence.

The clean-trace helper also treated malformed present fields such as boolean
`expected_workload_count` or non-list `gate_reasons` as legacy omissions. It now
rejects those shapes while preserving genuinely legacy traces that omit the
fields entirely.

The fallback structural checker now applies the same malformed-field rejection
to present `gate_reasons` and `expected_workload_count`, closing the last path
where malformed metadata could be treated as a legacy omission.

The same guard rejects a present zero workload count paired with nonempty rows,
an impossible current-orchestrator state that could otherwise look like valid
legacy metadata.

Manifest lint also accepted malformed `surface_set` values (strings, duplicate
names, unknown names, and unsorted lists). It now requires a sorted unique list
drawn from the locked surface vocabulary before publication or reuse.

Manifest lint now also enforces the closed `build_mode` and
`workload_selection_policy` vocabularies, preventing malformed strings from
reaching cache or resume decisions.

That vocabulary check exposed a producer mismatch in report stub manifests,
which emitted the obsolete policy `all`. The writer now emits `full-fanout`, and
a patched stub-manifest probe passes lint and preserves the intended policy.

A bounded manifest type-fuzz found additional top-level shape gaps. Lint now
validates nullable hash text, config/dispatch objects, timeout and reference
timing types, NVTX booleans, and invocation-order entries before publication.

Per-surface fuzzing found equivalent gaps in command, exit-code, artifact,
waiver, size, substatus, and cache-identity fields. Status entries now enforce
their declared scalar/list/object shapes and status-specific required values;
713 retained manifests still pass the stricter lint.

Surface-cache resume also treats non-dict prior entries or cache identities as
misses instead of raising during `.get()` access.

The no-op surface-resume path now additionally requires the prior evaluator
trace to be all-PASSED. Complete non-PASSED traces remain reportable evidence,
but trigger a fresh evaluator attempt instead of being preserved indefinitely.

The same fuzz found host identity fields accepting arbitrary JSON types. Host
device, compute-capability, driver, CUDA-runtime, and capture-time fields now
require nonempty strings alongside the existing host-kind check.

Collection-field fuzzing also caught Python booleans being accepted as workload
indices and non-string native-environment components. Both now fail lint; all
713 retained manifests still pass.

Identity-field fuzzing found arbitrary `slug` and `language` values accepted by
lint. Slugs now require nonempty path-safe text, and language is restricted to
the two supported C++ variants.

The surface orchestrator’s resume path had a weaker prior-trace check than the
aggregate report: any nonempty `rows` list could enable surface-cache reuse.
Resume validation now requires object-shaped rows, rejects parser/evaluator/
row-count/zero-row failures and malformed metadata, and still accepts complete
honest non-PASSED traces.

The cache-status probe found a retry starvation edge: a surface that timed out
or was blocked by host policy was eligible for reuse when its identity matched,
so a normal resume could preserve the transient failure forever. Only stable
`ok` and deterministic `unavailable` entries are reusable now; `blocked`,
`skipped`, and `failed` entries rerun.

Direct replay also had three optional candidate-document loaders that bypassed
the anchored input reader. Memory-service candidates, admission candidates,
and timing-active feedback now use the same bounded no-follow generation check
as cards and control metadata, so a path swap cannot alter timing inputs during
replay.

Active campaign documentation and the coverage-row augmenter now resolve the
`tools/sol_baselines_harness` and `tools/sol_baselines_scaffold` entry points;
the augmenter emits a direct prerequisite error when its aggregate report has
not yet been generated.

Trace-layout consumers had an asymmetry where the strict evidence-key helper
rejected a bundle-name/manifest architecture disagreement but the convenience
helper returned the explicit suffix. The convenience path now returns `None` on
the same disagreement, so callers cannot silently route contradictory evidence.

The host contract also documented `local_sm_86` while the live probe emits
`local_sm_8_6`. Active host, trace-layout, and campaign documentation now use
the emitted token consistently, preserving the `__sm86` routing decision.

Cache-record fuzzing found that malformed `native_env_hash` text could survive
the persisted-record filter even though derived keys require canonical digest
text. Cache loading now rejects that shape before it can be retained or
considered for replay.

Native-environment and host probes were reducing `nvcc --version` to only its
major/minor release, so a patch-level compiler change could retain an old cache
identity. Both probes now retain the full `V<major>.<minor>.<patch>` token when
the banner provides it, while preserving release-only fallback parsing.

The host fingerprint had a separate labeling defect: `HostInfo.cuda_runtime_version`
was populated from the compiler banner rather than the loaded CUDA runtime.
Host probing now calls `cudaRuntimeGetVersion` through libcudart and reports
`unknown` when that API is unavailable, so the field and cache axis describe the
runtime they claim.

Manifest lint accepted arbitrary nonempty text for top-level evaluator and
native-environment identities. It now requires lowercase hexadecimal digest
shape (`40` or `64` characters for evaluator tree identities, `64` for native
environment identities); all 713 retained manifests still pass.

Evaluator JSONL ingestion also accepted a solution-bearing `Trace` with no
evaluation object as a clean row-count-complete run, yielding `status=None`.
`run.py` now gates any parsed row missing an evaluation status while preserving
known non-PASSED statuses as reportable evidence.

The causal-frontier prototype merge was reviewed against the standalone
`/home/jakeshea/GPU_prototypes` crate. Its descriptor/event/scoreboard/mini-GPU
modules remain a separate dependency-free experiment because their state model
does not match the production replay types; the compatible schedule bridge and
observation-only NCU surface are merged. The causal analyzer now canonicalizes
each program by sequence before both exact and collapse passes, preventing an
arrival-order permutation from changing candidate-region accounting.

The merged NCU CSV reader had accepted `nan`/`inf` and truncated fractional
integer metrics. It now rejects non-finite values and fractional or negative
integer fields, with focused parser coverage; NCU observations remain strictly
comparison data and never configure replay.

The same integer parser had a precision trap: large decimal/exponent spellings
could pass through `f64` and round before conversion. NCU integer fields now
accept only exact decimal `u64` text, preserving values above the floating-point
exactness range without lossy conversion.

A retained raw NCU export also left `launch__kernel_name` empty while its
`Kernel Name` column carried the real kernel signature. The observation parser
now falls back to that column and retains the prototype's separate FMA/LSU
instruction observations without inventing a total-issue count.

NVIDIA's current [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
defines .sum as the total across hardware-unit instances and .avg as their
average. The reader no longer uses smsp__inst_executed.avg as a fallback total;
it exposes that average separately and only feeds the exact .sum into
resource-dependency totals.

The NCU CLI also scales each CSV metric by its per-column units row when
--print-units auto is selected. Importing one retained report reproduced the
bug: gpu__time_duration.avg changed from 9664 ns in base output to 9.664 us in
auto output, while the old parser returned 9.664 for a field named gpu_time_ns.
The reader now normalizes ns/us/ms/s and SI-scaled cycle/instruction/warp/sector
units, rejects unknown units, and accepts exact integral --print-fp dimensions.

The Python NCU path already strips a UTF-8 BOM, but the direct Rust reader did
not; a BOM-prefixed otherwise-valid export failed header detection. The Rust
reader now strips the leading BOM and the release HMMA replay remains silent
with a parsed launch observation.

Cross-format replay comparison shows one representation caveat: NCU base
output rounds smsp__inst_executed.avg to 140.29 while auto output retains
140.285714. Normalized time, total instruction, and sector observations match;
the average remains an observation with source-format precision, so byte
identity is not claimed for that field.

An installed-NCU `--print-fp --print-units base` export was replayed directly;
decimal launch dimensions `256.00` and `8.00` are accepted as exact integers,
and the normalized observation remains 9664 ns and 47136 instructions.

Architecture-card resource deserialization had the same `f64` rounding hazard:
values just above a `u32` or exact-JSON-integer boundary could round down into
the accepted range. Card integer fields now parse their original decimal or
exponent token exactly; valid integer-valued decimals remain accepted.

Latency values with an explicit unit are now required to describe cycles,
preventing a card that labels a latency in seconds or nanoseconds from being
silently consumed as cycle timing.

Consumed card blocks also accepted whitespace-only provenance strings. Latency,
L2 geometry, and DRAM bandwidth inputs now require non-empty provenance text so
their measured or documented origin cannot disappear while values remain
executable.

The P-chase resolver similarly ignored an explicit property unit, allowing a
capacity or latency labeled with another dimension to be consumed as bytes or
cycles. It now rejects mismatched units while preserving legacy cards that omit
the optional unit field.

The Python pchase resolver had the same explicit-unit gap as its Rust twin.
Capacity and latency resolvers now reject mismatched labels in both languages;
the retained SM86 card resolves to 98,304 bytes and 33.0016/243.6152 cycles.

Its JSON loader also narrowed decimal integers to binary64 before `_card_int`,
so large capacities could be accepted under a rounded value. It now loads
decimal tokens as `Decimal` until integer validation; a `9007199254740993.0`
probe is rejected using the original value rather than a rounded substitute.

The Python resolver also accepted integers beyond the Rust `u64`/`u32` bounds,
which could turn a malformed capacity or MSHR declaration into oversized cache
state. It now enforces the same width limits before deriving line capacity or
MSHR count.

Python pchase card loading also lacked Rust's bounded-file read and could ingest
an unbounded JSON document before validation. It now caps card input at 16 MiB,
rejecting oversized or non-UTF-8 files before parsing while preserving the
retained card path.

The initial size check still read the entire file before measuring it, so the
bound did not constrain peak memory. Loading now reads at most 16 MiB plus one
byte; a sparse 128 MiB card under a 96 MiB address-space limit fails with the
typed size error rather than `MemoryError`.

The bounded Decimal integer check also materialized `int(raw)` before testing
the upper bound, making a huge exponent an avoidable allocation hazard. Bounds
are now checked before integer conversion; a `1e1000000` capacity is rejected
without materializing a million-digit integer.

Latency conversion had the dual edge case: a positive Decimal such as `1e-400`
passed the Decimal check but became `0.0` when converted to f64. The resolver
now rechecks positivity and finiteness after conversion while retaining the
smallest representable positive f64 latency.

The same loader followed a symlinked card leaf or parent directory. It now
rejects symlink components before reading the card, keeping Python and anchored
Rust card identity tied to the requested file.

Carded L2 geometry also allowed a capacity whose per-slice set count was
fractional, silently flooring away part of the declared cache. The carded
constructor now rejects non-integral per-slice set geometry and capacities with
slice remainders; the generic analytical configuration keeps its documented
flooring behavior.

The causal-frontier event-reduction counter used saturating subtraction, which
could hide an internal accounting mismatch. Region reduction and its arithmetic
now use checked operations and return a typed completion-overflow error instead.

The causal report also exposed wall-clock `planner_time_ns` in its serialized
shape, making repeated analyses differ even when their event and cycle results
matched. Planner timing remains available in memory for telemetry but is now
excluded from serialized scientific output.

The L1 due-fill recency preflight is retained for the near-limit boundary,
but now uses the maintained minimum-fill timestamp to skip map scans whenever
no timed fill is due. Exact counting remains only at a due-fill boundary, while
boundary tests retain the transactional u64::MAX behavior.

The campaign's causal-probe validator compared the two final-cycle fields
directly, so a malformed output with both fields absent passed the equality
check. It now requires all checked region and frontier values to be
nonnegative integers before accepting the integration result.

The post-merge simulator stage also recorded NCU output artifacts without
requiring the simulator JSON to contain its observation-only join. It now
fails a campaign row when a matching NCU CSV exists but ncu_observation is
missing, and records the presence bit in the stage result.

The same contract initially accepted an empty observation object. Campaign
validation now requires a nonnegative launch ID plus nonempty kernel and device
identity fields; all 8 small and 7 large retained simulator outputs satisfy
that semantic check.

The raw observation was then bound to the normalized NCU record by launch ID
and kernel identity. A stale or substituted CSV now fails before the simulator
row is accepted; all 15 retained raw/normalized pairs match.

The same binding now compares the raw CSV content identity recorded by
normalized.json. A byte mutation that preserves launch and kernel labels is
rejected, and all 15 retained CSV artifacts match their normalized identities.

Campaign resume also checked declared inputs without checking the repository
revision, allowing a later commit to append records to an older manifest. An
existing manifest now has to match the current branch and commit before any
stage can append results; a stale-revision resume probe exits with a typed
campaign error.

The resume contract also lacked a manifest-level simulator executable binding,
so a same-revision rebuild could mix binary generations. New manifests record
the simulator binary identity and resume rejects a changed executable. The
retained 9df campaign manifests intentionally predate this field and are
therefore treated as historical evidence rather than resumable state.

Workload executables had the same unbound substitution path across correctness,
NCU, NVBit, and static stages. New manifests also record the identity of every
declared workload binary; a changed workload set now blocks resume before any
stage runs.

Resume now recursively revalidates the path/content identity of artifacts
already recorded in prior stage results. A temporary byte mutation is rejected;
the retained 9df manifests correctly fail this check only because the current
simulator binary was rebuilt later.

That recursive check initially omitted the top-level causal-frontier probe
artifacts and tolerated a non-list records field. Resume now validates the
probe binary/output too and reports malformed records as a typed campaign
error; a temporary probe-output mutation is rejected.

Resume also rejects duplicate stage/workload records and non-object records,
preventing a malformed manifest from presenting repeated or structurally
invalid evidence as a complete campaign.

Each persisted record must now also name one known stage and a nonempty
workload ID; a single empty record no longer bypasses the duplicate check.

Malformed manifest inputs are now rejected with the same typed campaign error
instead of escaping as an AttributeError during resume validation.

Artifact validation also followed symlinks through is_file(), allowing a
same-content symlink to masquerade as a retained result. New and persisted
artifacts now reject symlink paths; all retained campaign artifact paths are
regular files.

The top-level campaign input identities had the same follow-symlink behavior.
Manifest inputs now reject symlink paths before JSON loading; all retained
campaign input paths are regular files.

Resolving CLI and declared paths before that check could bypass the leaf
restriction, including through a symlinked parent directory. Campaign path
resolution now rejects symlink components before canonicalization; leaf and
parent symlink probes both fail.

The no-follow boundary is exercised from outside the repository as well:
absolute normal paths still produce a valid campaign plan after component
validation.

Post-merge validation rebuilt sim-run, passed 29 direct-run feature-library
tests, 127 sim-mem tests, 440 campaign-enabled sim-sm tests, 169
campaign-enabled sim-trace tests, and the dependency-free prototype cargo
check. A retained HMMA small replay and a
SwiGLU large feedback replay both exited silently; the large replay matched its
retained 15,173-byte JSON exactly, while two fresh HMMA runs were byte-identical.
The raw `/tmp/gpu_sector_probe_s1.csv` export also joined successfully and
reported its quoted kernel signature, RTX 3090 Ti device, LSU count, and global
load/store sectors.


`ManifestBuilder.lint_and_write` now honors its atomic-write contract by
serializing to a same-directory temporary file and replacing the destination
only after the full document is written; interrupted writes cannot leave a
truncated manifest.

## Remaining high-value work

The NVBit capture output contract accepted symlinked required artifacts because
its `is_file()` check followed links; trace admission caught this only later.
The capture contract now rejects symlinked required outputs immediately, while
all 15 retained NVBit directories still pass the contract.

The same contract also counted regular payloads while silently ignoring a
symlinked `threadblocks/*.pb` beside them. It now rejects any symlink payload
explicitly; retained payload files remain regular files.

The payload directory itself could also be replaced by a symlink before
validation, causing the recursive scan to inspect another tree. The contract
now rejects a symlinked `threadblocks/` directory before traversal.

The same traversal could skip a symlinked nested directory and thereby hide
payloads while leaving other files valid. The contract now rejects any symlink
entry below `threadblocks/` before counting payloads.

NCU wide CSV identity parsing also accepted a negative launch ID until the
simulator join rejected it later. The campaign parser now rejects negative IDs
at metric ingestion, keeping malformed profiler identity failures local to the
NCU stage.

The shared NCU CSV reader only recognized lines beginning with a quote, so a
valid unquoted CSV export was reported as empty. Header discovery now accepts
both quoted and unquoted CSV fields while still skipping profiler preambles.
The text API also strips a leading UTF-8 BOM, matching file-based loading.

The campaign's wide-export identity join still filtered out unquoted rows even
after metric parsing was generalized. It now locates the quoted or unquoted
`ID`/`Kernel Name` header and validates the same row shape; both encodings join
to the retained launch identity.

The Python admission contract rejected regular non-`.pb` files below
`threadblocks/`, but Rust's anchored tree reader silently skipped them. Strict
Rust tree readers now reject the same non-protobuf entries, keeping direct
replay and admission behavior aligned.

Capture destination setup also followed an empty symlinked directory and could
write a new trace outside the requested output root. Destination resolution now
rejects leaf and parent symlink components before `mkdir`; the normal retained
capture roots remain regular paths.

The simulator-stage NCU join had a smaller time-of-check/time-of-use gap: it
probed and hashed `ncu.csv` through the raw path after command construction.
The stage now applies the same no-follow resolver before every probe and hash,
so a symlink introduced between setup and replay is rejected consistently.

An attempted cross-component instruction-count check also exposed an important
frontier distinction: NVBit `total_reported_insts` is the dynamic trace-record
population, while simulator `instructions_issued` and NCU
`smsp__inst_executed.sum` count executed instructions after predication. The
latter two match across all 15 workloads; the trace-record total must not be
used as their oracle.

The campaign previously retained NCU and simulator values without a joined
comparison record. Simulator-stage results now include diagnostic per-metric
absolute errors and coverage; the retained data identifies large FlashAttention
backward DRAM-sector error (95.15%) without using NCU values to configure replay.

The comparison initially dropped zero-baseline rows from its maximum error even
when a simulator value was nonzero. It now records exact-zero matches and
zero-baseline mismatches separately; all 60 retained comparable rows have no
zero-baseline mismatch.

Comparison numeric validation also rejected booleans explicitly; Python's bool
subclassing of int had otherwise allowed malformed true/false values into the
diagnostic error calculation.

An admission sweep initially used system Python and failed uniformly because
protobuf 7.34.1 was unavailable. Repeating the same sweep with the pinned
`.venv` passed all 15 roots with zero stderr, separating an environment
prerequisite failure from trace validity.

Uniform and tensor execution resources remain an explicit `DispatchOnly`
boundary until an SM86 card-backed cadence is measured. The retained corpus has
four `UISETP` sites, no `BRA.U`, and 53 admitted HMMA/IMMA sites; no unsupported
form is silently assigned a timing interval. Destination service timing for
source-only or bypass LDGSTS remains an explicit unmodeled tail where the
retained card has no destination-coupling measurement.

The continuous audit remains active; future work should target new
architecture-backed discriminators or newly observed replay failures.

The primary-source review of [Analyzing and Improving Hardware Modeling of
Accel-Sim](https://arxiv.org/abs/2401.10082) confirms that modern SMs expose
independent sub-core pipelines and tensor units, supporting the simulator's
per-SMSP tensor resource seam. It does not supply a validated SM86 cadence for
the remaining uniform or SFU forms, so assigning them a guessed interval would
violate the card-backed timing boundary.
