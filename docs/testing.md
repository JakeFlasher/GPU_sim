# Development checks

The alpha branch uses direct, bounded checks while developing the simulator.
They are not approval gates and do not establish hardware accuracy. Run the
smallest command that answers the question, inspect the output, and continue
iterating. NCU measurements remain an independent comparison target.

## Predictive configuration

The predictive profile is an input contract. Load it and, when needed, fit a
profile from independent probe observations:

```bash
python3 -m json.tool docs/sm86_predictive_closure_config_20261001.json >/dev/null
python3 -m json.tool docs/sm86_service_rate_profile_20261001.json >/dev/null
PYTHONPATH=tools python3 tools/calibrate_sm86_service_profile.py \
  PROBE_SERVICE_OBSERVATIONS.json /tmp/sm86_service_profile.json
```

For a declared model-local service intervention, compare profile variants
against one detailed replay without joining NCU values:

```bash
PYTHONPATH=tools python3 tools/evaluate_reification_design.py \
  --sim-run sim/target/release/sim-run \
  --profile docs/sm86_service_rate_profile_20261001.json \
  --trace sim/run/tests/fixtures/addr_carry_canary_bundle/trace \
  --control sim/run/tests/fixtures/addr_carry_canary_bundle/control/canary_sm86.control_bits.json \
  --arch-card sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86.json \
  --pchase-card sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86_static_card.json \
  --scale 1 --scale 2 --scale 4
```

The report ranks only the declared simulator interval, checks native replay
invariance, and labels the result model-local; it rejects NCU/CUPTI inputs.
Pass a separately generated `--fast-frontier-only` JSON with
`--fast-frontier-report` to add balanced candidate/reified/uncovered coverage
and compression metrics.

After predictions are materialized, join independent NCU targets in a separate
step:

```bash
PYTHONPATH=tools python3 tools/evaluate_prediction_splits.py \
  PREDICTIONS.json NCU_OBSERVATIONS.json
```

For raw NCU CSV output, create the observation document first with
`tools/ncu_csv_to_observation.py`; this adapter has no access to predictions.

The evaluator rejects observed fields inside predictions and reports split
coverage, interval width, error, refusal/OOD rate, runtime, memory, and
compression metrics.

For multiple current trace families, use `tools/run_prediction_manifest.py` to
materialize the calibration/development/held-out prediction document before
running the NCU CSV adapter and join.

The loader checks shape and numeric domains only. Supplied rates are examples,
not cycle-accuracy evidence.

## Rust workspace

Run focused crates first, then the workspace when a change crosses crate
boundaries:

```bash
cargo test --manifest-path sim/Cargo.toml -p sim-arch --lib
cargo test --manifest-path sim/Cargo.toml -p sim-mem --lib
cargo test --manifest-path sim/Cargo.toml -p sim-sm --lib
cargo test --manifest-path sim/Cargo.toml -p sim-run --lib
cargo test --manifest-path sim/Cargo.toml --workspace --quiet
cargo build --release --manifest-path sim/Cargo.toml -p sim-run -p sim-arch
```

These are implementation checks. They do not replace an NCU run or turn a
model-relative result into a silicon claim.

## Direct replay canaries

Use a retained trace with a timeout after native replay changes:

```bash
TRACE=sim/run/tests/fixtures/addr_carry_canary_bundle/trace
CONTROL=sim/run/tests/fixtures/addr_carry_canary_bundle/control/canary_sm86.control_bits.json
CARD=sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86.json
PCHASE=sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86_static_card.json
timeout 60 sim/target/release/sim-run --trace "$TRACE" --control "$CONTROL" \
  --arch-card "$CARD" --mode flat --cache-scope launch
timeout 60 sim/target/release/sim-run --trace "$TRACE" --control "$CONTROL" \
  --arch-card "$CARD" --pchase-card "$PCHASE" --mode l1-l2 \
  --coherence-service-cycles 2 --coherence-capacity-sectors 8
```

Inspect termination, frontiers, event counts, and typed error output. The
`simt_its_v1` and `simt_u32_v1` commands are functional/custom experiments,
not native SASS timing:

```bash
sim/target/release/custom_simt_its \
  arch_profiles/specs/simt_its_dynamic.json \
  arch_profiles/programs/simt_its_dynamic.json --trace
cargo run --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run -- \
  arch_profiles/specs/simt_u32_route_alternatives.json \
  arch_profiles/programs/simt_u32_route_alternatives.json
```

## Python and probe checks

Use the repository environment for a focused Python test or syntax check:

```bash
python3 -m py_compile tools/custom_simt_differential.py
.venv/bin/python -m pytest -q \
  workloads/research_probes/sm86_ncu_demo_benchmark/test_campaign.py
```

Run the research-probe Makefile or script for the specific question. Missing
optional CUDA, NCU, NVBit, or third-party packages are environment limitations;
report them plainly instead of creating a replacement artifact or skipped gate.

## Measurement boundary

Compare simulator output with NCU only after the replay path has completed.
Never use a scored workload's NCU result as a simulator input, tuning feature,
or model-selection signal. Report prediction error, assumptions, and runtime
together. A passing local check is implementation evidence, not independent
silicon evidence.

The repository permanently excludes M5/M6 authority, receipt, audit,
provenance, hashing, sealing, admission, approval, freeze, and validation
workflow machinery. Do not add a wrapper, manifest, digest, or receipt merely
to run a command or commit a change.
