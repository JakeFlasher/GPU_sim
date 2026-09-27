# Testing and executable evidence

Tests are bounded engineering checks. They do not replace NCU measurements,
do not create approval gates, and do not turn a model-relative result into
silicon validation.

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
```

The custom ITS output above is validated against
`arch_profiles/schema/simt_its_trace.schema.json`. A retained native canary or
representative SM86 trace should still be replayed with a bounded timeout after
native metric changes. Do not treat the schema check as proof of native replay
compatibility: `custom_its_dynamic_v2` is explicitly a separate frontend.
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
  --arch-card "$CARD" --mode flat --fast-frontier-only
```

Record input identity, accepted/refused counts, refusal reasons, output digest,
and wall/RSS when available. The fast-only canary currently has no certificate
because its dynamic records lack the static register-projection evidence
required by the strengthened PhaseFuse contract; that is an explicit refusal,
not a speed or accuracy result.

On the verified local RTX 3090 Ti, the bounded independent counter holdout can
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
hashes and a comparison CSV. It is deliberately non-scoring unless clocks,
metric domains, and dirty-flush/ownership contracts are independently closed;
see [the holdout report](sm86_independent_holdout_20260927.md).

The strict tiny external-input probe is exercised without a large simulator:

```bash
PYTHONPATH=. .venv/bin/python -m tools.traceg_adapter_probe.adapter \
  --trace DYNAMIC_TRACE.pb --threadblock ONE_BLOCK.pb \
  --sidecar ENHANCED_EXECUTION_INFO.json --out-dir /tmp/traceg_probe \
  --allow-eof
.venv/bin/python -m pytest tools/traceg_adapter_probe/test_adapter.py -q
```

`--allow-eof` emits a parser-only non-scoring artifact; explicit `EXIT` is
required for any future common-endpoint timing attempt. Unsupported opcodes,
register files, masks, address encodings and missing joins return typed
refusals.

Do not add hash/seal, receipt, reviewer, freeze, or mandatory test-pyramid
machinery around the simulator. Retained manifests/checksums are data formats,
not authority over code or agents.
