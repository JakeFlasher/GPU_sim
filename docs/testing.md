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
```

The last output is validated against
`arch_profiles/schema/simt_its_trace.schema.json`. A retained native canary or
representative SM86 trace should still be replayed with a bounded timeout after
native metric changes. Do not treat the schema check as proof of native replay
compatibility: `custom_its_dynamic_v1` is explicitly a separate frontend.

## Focused checks

From the repository root:

```bash
cargo test --manifest-path sim/Cargo.toml -p sim-arch --lib
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

Do not add hash/seal, receipt, reviewer, freeze, or mandatory test-pyramid
machinery around the simulator. Retained manifests/checksums are data formats,
not authority over code or agents.
