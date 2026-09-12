# Rust simulator

The workspace contains the executable replay path:

- `sim-trace`: protobuf trace and control-sidecar readers.
- `sim-arch`: architecture cards and capability selection.
- `sim-sm`: warp, CTA, dependency, barrier, and issue scheduling.
- `sim-mem`: L1/TEX, L2, and memory completion models.
- `sim-engine`: shared event ordering.
- `sim-engine::QueueStats`: virtual clock-gap telemetry kept separate from
  handler-event counts and host wall time.
- `sim-semantics`: portable event and partial-order certificate types.
- `sim-semantics::ObservationContract`: explicit boundary/per-cycle/omitted
  observations for model-relative exactness claims.
- `sim-trace`'s `semantic-probe`: SM86 NVBit semantic adapter and canonical
  event round-trip probe.
- `sim-run-campaign`: manifest driven CLI and JSON output.
- `sim-run`: direct replay convenience CLI.
- `sim-sm::causal_frontier`: opt-in causal-frontier analysis for explicit
  descriptor streams; it is observational until a producer supplies dependency
  and completion facts.
- `causal-frontier-probe`: bounded release-build integration probe used by the
  full campaign to verify safe collapse and contended reification.

Build the runner with:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
cargo build --release --manifest-path sim/Cargo.toml -p sim-run-campaign
cargo build --release --manifest-path sim/Cargo.toml -p sim-sm --bin causal-frontier-probe
cargo build --manifest-path sim/Cargo.toml -p sim-trace --bin semantic-probe
```

The SM86 semantic probe reads one validated trace and its canonical control
sidecar. It reports exact lane-address evidence and static-only constant-cache
gaps, plus sidecar stall/barrier/wait/reuse evidence, without changing native replay. Add `--roundtrip OUT_DIR` to emit a new
canonical trace for a focused equivalence check; the original capture is never
overwritten. `tools/semantic_events/sm86_roundtrip_compare.py` automates the
probe, native replay, round-trip replay, and comparison of counters, timing,
and coverage.

For a retained campaign directory, `tools/semantic_events/sm86_reference_compare.py`
can run every recorded simulator command in both modes and emit a JSON
comparison:

```bash
python tools/semantic_events/sm86_reference_compare.py \
  --command-dir results/captures/sm86_postmerge_campaign_small_20260909_v1/evidence/raw/simulator
```

Add `--trace-dir DIR` for post-event boundary projections; the comparator
checks their event ordering and snapshots in addition to the final replay JSON.

For a bounded differential run, build the optional reference feature and set
`SIM_SM_REFERENCE_MODE=dense`. The default event mode keeps the existing
next-transition advancement; dense mode visits each locally idle cycle so its
serialized replay output can be compared with the accelerated run. The feature
also prints `SIM_REFERENCE_PERF` diagnostics with local cycle steps, idle jumps,
and virtual cycles skipped. These diagnostics are engineering measurements,
not hardware-speed claims.

Set `SIM_SM_REFERENCE_TRACE=/path/to/boundary.jsonl` alongside the reference
mode to emit an optional event-order and canonical SM boundary trace. Records
retain the legacy `state_projection_complete: false` field for V1 consumers,
while the V2 `typed_boundary` record carries typed L1/L2, front-end, pipeline,
async, queue, ownership, and fixed-clock activity-power state. Its
`completeness` field is the contract-relative result used by new consumers.

For an observation-only NCU join, append `--ncu-csv PROFILE.csv` and, for a
multi-launch export, `--ncu-launch ID`. The values appear under
`ncu_observation` and never affect replay timing or architecture parameters.

Run it with a retained NVBit trace and control sidecar as described in
`docs/runbook_host.md`. Architecture values belong in `arch_profiles/cards`,
not in workload-specific code.
