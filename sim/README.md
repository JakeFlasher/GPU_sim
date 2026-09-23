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
- `sim-semantics::transform`: contract-aware typed boundary proof/checker
  interfaces; only its zero-length identity rule certifies, and it makes no
  speed claim.
- `sim-semantics::ObservationContract`: explicit boundary/per-cycle/omitted
  observations for model-relative exactness claims.
- `sim-arch::spec` and `sim-arch::simt`: experimental, fail-closed custom
  architecture definitions and a bounded `simt_u32_v1` functional executor;
  the executor includes a declared-latency async-store/wait-group timing
  probe; these do not alter the SM86 production replay.
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

Repeat `--command-dir` to compare multiple cohorts in one run; every selected
`*/command.txt` is reported and receives a unique trace filename.

Add `--trace-dir DIR` for a bounded post-event boundary projection; the
comparator checks its event ordering and snapshots in addition to the final
replay JSON. The trace stores one full typed boundary followed by lossless JSON
deltas with exact array splices for changed-length arrays, and the reader
compares it as a stream, so memory use is bounded by a constant number of
boundaries. Full-workload differential runs should omit
`--trace-dir`; retain a small canary trace when boundary-level diagnosis is needed.

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
while the V15 `typed_boundary` record carries typed L1/L2, front-end, pipeline,
async, queue, ownership, and fixed-clock activity-power state. Its
`completeness` field is the contract-relative result used by new consumers.
Set `SIM_SM_REFERENCE_MAX_RECORDS` to bound diagnostic trace generation before
running a large workload; full typed JSON traces can consume many gigabytes and
are not part of the production replay path. The default bound is 8,192 records;
raise it explicitly only when the retained trace size is understood. The bound
covers data records; a terminal footer is emitted separately so an exact-limit
trace remains readable. If a run fails before a valid footer, the comparator
removes its partial trace automatically, including timeout termination.

For an observation-only NCU join, append `--ncu-csv PROFILE.csv` and, for a
multi-launch export, `--ncu-launch ID`. The values appear under
`ncu_observation` and never affect replay timing or architecture parameters.

Run it with a retained NVBit trace and control sidecar as described in
`docs/runbook_host.md`. Architecture values belong in `arch_profiles/cards`,
not in workload-specific code.
