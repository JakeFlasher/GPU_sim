# Testing and executable evidence

Tests are bounded engineering checks. They do not replace NCU measurements,
do not create approval gates, and do not turn a model-relative result into
silicon validation.

The Rust workspace currently passes its full test suite. A repository-wide
`pytest tools` sweep is not an admissible simulator gate: the retained
SOL-ExecBench/FlashInfer reference tests require optional packages such as
`torch`, `flashinfer`, and `click` that are not installed in the base host.
Those collection errors are environment/dependency gaps outside the SM86
simulator and native differential paths; run their own provisioned environment
before making SOL/campaign claims.

## Current handoff smoke

From the repository root, the current consolidation is exercised with:

```bash
cargo test --manifest-path sim/Cargo.toml --workspace --quiet
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet
cargo build --release --manifest-path sim/Cargo.toml -p sim-run -p sim-arch
sim/target/release/custom_simt_its \
  arch_profiles/specs/simt_its_dynamic.json \
  arch_profiles/programs/simt_its_dynamic.json --trace

# Bounded PTX-like functional lowering with input-varying memory.
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin ptx_subset_compile -- \
  example_its arch_profiles/programs/simt_its_subset.ptx /tmp/simt_its_subset.json \
  64 arch_profiles/programs/simt_its_subset_memory_mixed.json

# Custom alternative-resource timing experiment.
cargo run --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run -- \
  arch_profiles/specs/simt_u32_route_alternatives.json \
  arch_profiles/programs/simt_u32_route_alternatives.json

# Conservative fast-only scalar-region analysis; does not run native timing.
sim/target/release/sim-run \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --mode flat --fast-frontier-only

# Opt-in bounded dirty-sector writeback; default carded replay is unchanged.
sim/target/release/sim-run \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card PCHASE_CARD.json --mode l1-l2 --carded-dram \
  --dirty-writeback-capacity-sectors 64

# Opt-in ordered-L2 directory service (hypothetical, not protocol timing).
sim/target/release/sim-run \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card PCHASE_CARD.json --mode l1-l2 \
  --coherence-service-cycles 2 --coherence-capacity-sectors 64

# Explicit typed L1 observation/invalidation event path; bounded experiment.
sim/target/release/sim-run \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card PCHASE_CARD.json --mode l1-l2 \
  --coherence-service-cycles 2 --coherence-capacity-sectors 64 \
  --l1-visibility-protocol

# Also select resident-hit L1 dirty tracking. This requires the lower
# writeback queue and remains a bounded policy experiment.
sim/target/release/sim-run \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card PCHASE_CARD.json --mode l1-l2 --carded-dram \
  --dirty-writeback-capacity-sectors 64 \
  --coherence-service-cycles 2 --coherence-capacity-sectors 64 \
  --l1-resident-writeback
```

The custom ITS output above is validated against
`arch_profiles/schema/simt_its_trace.schema.json`. A retained native canary or
representative SM86 trace should still be replayed with a bounded timeout after
native metric changes. Do not treat the schema check as proof of native replay
compatibility: `custom_its_dynamic_v3` is explicitly a separate frontend.
The fast-only output is validated by its contract fields and refusal reasons;
it is not a substitute for the ordinary replay path.

For the bounded PhaseFuse analyzer scaling probe (synthetic public
descriptors, not a native trace or timing prediction):

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-sm --bin phasefuse-scale
python workloads/research_probes/phasefuse_scaling/benchmark.py \
  --binary sim/target/release/phasefuse-scale \
  --repeats 3 --output /tmp/phasefuse_scale.jsonl
```

The probe reports sorted and deterministic-shuffled `N={1k,2k,4k,8k,16k,32k,64k}`
cases over region length, residency and refusal dimensions. The wrapper adds
child wall/user/system time, peak RSS, timeout/exit status and raw JSONL plus
median summaries. It measures descriptor construction and canonicalization
only; it does not claim trace parsing, native replay or silicon speedup. Each
child object follows [`phasefuse_scale.v1.schema.json`](../sim/run/schema/phasefuse_scale.v1.schema.json).

## Focused checks

From the repository root:

```bash
cargo test --manifest-path sim/Cargo.toml -p sim-arch --lib
cargo test --manifest-path sim/Cargo.toml -p sim-mem --lib
cargo test --manifest-path sim/Cargo.toml -p sim-sm --lib
cargo test --manifest-path sim/Cargo.toml -p sim-run --lib
cargo test --manifest-path sim/Cargo.toml --workspace --all-targets
python3 -m py_compile tools/custom_simt_differential.py
pytest -q workloads/research_probes/sm86_ncu_demo_benchmark/test_campaign.py
```

For custom changes, also run the relevant Rust/Python differential and a
`custom-program-run` producer/consumer or disjoint-group example. For native
changes, run a retained canary or representative trace with a bounded timeout.
Inspect exit status, modeled progress, event/frontier counts, and typed tails.

Machine-local trace/sidecar tests may remain ignored when the capture is absent;
that is an environment limitation, not evidence of success. A full workspace
green run is useful for regression confidence but is not a numerical baseline
or hardware-validation result.

## 2026-09-27 Phase-0 bounded campaign checks

The following focused checks exercise the current refusal and mechanism seams
without running a large external simulator workload:

```bash
cargo test --manifest-path sim/Cargo.toml -p sim-mem --lib --quiet
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --lib --quiet
cargo test --manifest-path sim/Cargo.toml -p sim-run --lib --quiet
```

The memory suite contains the queue reference/litmus cases for invalid masks,
capacity, stale plans, zero-service unrelated lines, and pending visibility.
The SM campaign suite covers sampled-coverage refusal, missing-mask trace
admission, texture-domain refusal, cross-SM directory transfer, and separate
frontier counters. The direct-run suite checks that fast-only mode rejects
native memory/DRAM/writeback/coherence options it does not consume.

For a retained canary, use the actual option names shown by `sim-run --help`:

```bash
TRACE=sim/run/tests/fixtures/addr_carry_canary_bundle/trace
CONTROL=sim/run/tests/fixtures/addr_carry_canary_bundle/control/canary_sm86.control_bits.json
CARD=sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86.json
PCHASE=sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86_static_card.json
sim/target/release/sim-run --trace "$TRACE" --control "$CONTROL" \
  --arch-card "$CARD" --mode flat --cache-scope launch
sim/target/release/sim-run --trace "$TRACE" --control "$CONTROL" \
  --arch-card "$CARD" --pchase-card "$PCHASE" --mode l1-l2 \
  --coherence-service-cycles 2 --coherence-capacity-sectors 8
sim/target/release/sim-run --trace "$TRACE" --control "$CONTROL" \
  --arch-card "$CARD" --pchase-card "$PCHASE" --mode l1-l2 \
  --coherence-service-cycles 2 --coherence-capacity-sectors 8 \
  --l1-visibility-protocol
sim/target/release/sim-run --trace "$TRACE" --control "$CONTROL" \
  --arch-card "$CARD" --mode flat --fast-frontier-only
```

Record input identity, accepted/refused counts, refusal reasons, output digest,
and wall/RSS when available. The fast-only canary currently has no certificate
because its dynamic records lack the static register-projection evidence
required by the strengthened PhaseFuse contract; that is an explicit refusal,
not a speed or accuracy result.

The locked-clock prerequisite can be checked without mutating the GPU:

```bash
python3 tools/clock_lock_probe.py
```

`status=ready` only means noninteractive privilege is available;
`target_clocks_not_stable` is a refusal when samples drift. The probe never
runs a clock-lock command and never establishes a timing claim.

The enhanced NVBit tracer/protobuf build uses the pinned protobuf runtime and
requires a CUDA-supported host compiler on this image:

```bash
make -C tools/nvbit_tracer/tracer_tool -j2 NVCC_CCBIN=/usr/bin/g++-15
```

This validates compilation only; it does not capture a GPU trace or establish
register-value/SASS-regeneration evidence.

On the verified local RTX 3090 Ti, the bounded counter-domain development rerun can
be regenerated in a fresh output directory with:

```bash
python workloads/research_probes/sm86_metric_domains/run.py \
  --output results/captures/sm86_independent_holdout_$(date +%Y%m%d_%H%M%S)_v1 \
  --size small --suite domains
python workloads/research_probes/sm86_metric_domains/run.py \
  --output results/captures/sm86_independent_holdout_$(date +%Y%m%d_%H%M%S)_writeback_v1 \
  --size small --suite writeback
```

This path records device identity, NVBit trace, NCU report/CSV, replay input
hashes and a comparison CSV. It reuses the existing metric-domain source/case
matrix and is deliberately non-scoring unless clocks,
metric domains, and dirty-flush/ownership contracts are independently closed;
see [the holdout report](archive/research_20260930/sm86_independent_holdout_20260927.md).

The strict tiny external-input probe is exercised without a large simulator:

```bash
PYTHONPATH=. .venv/bin/python -m tools.traceg_adapter_probe.adapter \
  --trace DYNAMIC_TRACE.pb --threadblock ONE_BLOCK.pb \
  --control CONTROL_BITS.json --out-dir /tmp/traceg_probe \
  --exec-info ENHANCED_EXECUTION_INFO.json
.venv/bin/python -m pytest tools/traceg_adapter_probe/test_adapter.py -q
```

The input must contain an actual dynamic `EXIT`; EOF and synthetic termination
are refused. Unsupported opcodes,
register files, masks, address encodings and missing joins return typed
refusals.

Do not add hash/seal, receipt, reviewer, freeze, or mandatory test-pyramid
machinery around the simulator. Retained manifests/checksums are data formats,
not authority over code or agents.
## Accel-Sim literature-floor manifest

The literature lane is a target manifest, not a ground-truth data source. Check
its JSON shape and content-bound extraction identity before using any row:

```bash
python3 -m json.tool docs/accelsim_literature_lower_bound_manifest_20260928.json >/dev/null
python3 -m json.tool docs/accelsim_literature_lower_bound_manifest.schema.json >/dev/null
sha256sum docs/archive/research_20260930/gpu_simulator_sota_matrix_20260924.md \
  docs/gpu_simulator_sota_matrix_20260924.csv
.venv/bin/python -m pytest -q tools/test_accelsim_literature_manifest.py
.venv/bin/python -m pytest -q tools/test_evaluate_accelsim_literature_floor.py
```

Accuracy ceilings require local NCU measurements for the declared workload and
endpoint. Throughput/speedup floors require a matched host/toolchain/denominator
protocol; NCU is not silently used as a simulator-wall-time denominator.

The bounded compiler lane can emit a content-bound manifest:

```bash
.venv/bin/python tools/ptx_subset_compile_manifest.py \
  --compiler sim/target/debug/ptx_subset_compile \
  --architecture example_its \
  --source arch_profiles/programs/simt_its_subset.ptx \
  --output /tmp/ptx_subset_program.json \
  --manifest /tmp/ptx_subset_manifest.json
```

The manifest records source/generated-program hashes and explicitly marks
arbitrary PTX/ISA, CUDA ABI, SASS generation, and timing as unsupported.

The evaluator can be run against the retained development table:

```bash
.venv/bin/python tools/evaluate_accelsim_literature_floor.py \
  --manifest docs/accelsim_literature_lower_bound_manifest_20260928.json \
  --holdout docs/sm86_holdout_replay_20260926.csv
```

The current output is intentionally `scoring: false` and
`not_evaluable_*`; the table has incomplete tail semantics and the active clock
campaign is not admitted.

## Native value-dependent differential

The SM86 value-differential probe is a separate functional lane:

```bash
ptxas -arch=sm_86 -o /tmp/value.cubin workloads/research_probes/sm86_value_differential/value.ptx
# Build/run the pinned NVBit capture and CUDA-driver runner as documented in
# workloads/research_probes/sm86_value_differential/README.md.
.venv/bin/python workloads/research_probes/sm86_value_differential/validate_capture.py \
  workloads/research_probes/sm86_value_differential/native_capture_delta_2pow31_seed17.out
.venv/bin/python -m pytest -q workloads/research_probes/sm86_value_differential/test_validate_capture.py
```

The retained SM86 capture checks 32 lanes, 13 dynamic PCs, load-return values,
predicate outcomes, store registers, and final output arithmetic on the local
RTX 3090 Ti. It is functional differential evidence only: it does not claim
static SASS regeneration, timing, or physical-pipe reconstruction.

Convert the same capture into a strict value-bearing payload sidecar:

```bash
.venv/bin/python workloads/research_probes/sm86_value_differential/native_capture_to_payload.py \
  workloads/research_probes/sm86_value_differential/native_capture_delta_2pow31_seed17.out
```

This sidecar records known store bytes and line/sector identity while refusing
to claim initial memory, cache ownership, coherence, or flush completion.

`sim_mem::MemoryOrderLedgerV1` is the corresponding value-bearing ordering
admission seam. It requires explicit read-from IDs and release/acquire scope
for cross-requester visibility, and refuses missing values, unknown read-from
edges, scope mismatch, and sequence regressions. It is a reference contract,
not a claim that proprietary hardware fences have been reconstructed.
Its `observe_qualified_read` API additionally rejects a plain cacheable
cross-requester read and requires an explicit volatile/proxy/atomic qualifier,
matching the independent L1-control litmus.
`L1VisibilityQueue::preview_qualified` exposes the same admission rule at the
L1 ownership/invalidation seam.

The independent hardware litmus is reproducible with:

```bash
/opt/cuda/bin/nvcc -O2 -arch=sm_86 \
  workloads/research_probes/sm86_coherence_litmus/message_passing.cu \
  -o /tmp/message_passing
/tmp/message_passing 1000
```

The retained RTX 3090 Ti run reports 1000 successful message-passing
iterations with zero stale-read failures and zero timeouts. This is a silicon
value observation for the declared fence/atomic litmus, not proof of simulator
coherence completion.

The L1 control run records the ordinary non-volatile path as a negative
visibility control (stale reloads), while volatile and consumer-fence reloads
both have zero failures over 1000 iterations. The ordinary path is not treated
as a valid CUDA synchronization contract; this result motivates explicit
volatile/proxy/read-from metadata rather than an unconditional invalidation
claim.

The native capture can also be compared against the finite interpreter on the
same compiled SASS:

```bash
PYTHONPATH=tools/nvbit_tracer .venv/bin/python \
  workloads/research_probes/sm86_value_differential/compare_native_interpreter.py \
  workloads/research_probes/sm86_value_differential/value.sm_86.sass.txt \
  workloads/research_probes/sm86_value_differential/native_capture_delta_2pow31_seed17.out
```

The comparison is functional only and does not validate timing, pipe routing,
or static SASS regeneration.

The Rust memory layer admits the same sidecar schema through
`sim_mem::load_native_payload_sidecar`. It validates generation, allocation,
sector, byte-range, known-mask, and sequence identity before constructing
`SectorPayloadV1`; the ordinary replay path does not load it implicitly.
`sim_mem::coalesce_payloads` then merges ordered partial stores and refuses any
sector that is not fully known before a `PayloadCommitLedgerV1` transaction.

The real `sim-run` path accepts the opt-in `--native-payload-sidecar FILE`
argument. It reports the admitted record count and leaves replay scheduling
unchanged; the current canary smoke admitted 32 records and completed one flat
launch. This is admission evidence, not a claim that native caches consumed
the payloads.
