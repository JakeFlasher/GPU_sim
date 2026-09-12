# Current status

Updated 2026-09-12.

The end to end SM86 path is operational: NVBit traces and control sidecars are
read by `sim-trace`, replayed by the SM and memory models, and emitted through
`sim-run`. The current working tree contains active scheduler, cache, memory
completion, trace, and result-schema changes; those sources remain the live
implementation.

The current SM86 comparison set is:

| Evidence | Role | Result |
| --- | --- | --- |
| `results/captures/sm86_adversarial_current_campaign_20260904_v11` | Current simulator campaign | 15 workloads; maximum scored error 9.704% |
| `results/captures/sm86_fresh_locked_ncu_validation_20260904_v84` | Best locked-clock comparison | strict median maximum 8.671%; raw SM-domain maximum 9.428% |
| `results/captures/sm86_fresh_locked_ncu_validation_20260904_v85` | Latest locked-clock comparison | strict median maximum 9.147%; raw SM-domain maximum 9.378% |
| `results/captures/sm86_fresh_batch_sensitivity_20260904_v33` | Repeat comparison | zero strict pass/fail flips between batches |
| `results/captures/sm86_fresh_locked_ncu_small_20260904_v10` | Small NCU input set | 8 workload shapes, seven repeats |
| `results/captures/sm86_fresh_locked_ncu_large_20260904_v10` | Large NCU input set | 7 workload shapes, seven repeats |
| `results/captures/sm86_regenerated_workloads_small_20260905_v1` | Fresh full small replay | 8 correctness, NCU, NVBit, static, and simulator stages completed |
| `results/captures/sm86_regenerated_workloads_large_20260905_v1` | Fresh full large replay | 7 correctness, NCU, NVBit, static, and simulator stages completed |
| `results/captures/sm86_postmerge_campaign_small_20260909_v1` | Post-merge clock-qualified small campaign | 8/8 all stages; NCU observation joined; causal probe recorded |
| `results/captures/sm86_postmerge_campaign_large_20260909_v1` | Post-merge clock-qualified large campaign | 7/7 all stages; NCU observation joined; causal probe recorded |

The simulator reports separate issue, modeled-event, completion, and launch
frontiers. An incomplete retirement tail is represented as typed state rather
than an invented fixed delay. Physical cache counters remain diagnostics until
there is an independent NCU mapping.

The active capture tree retains current/best comparison inputs and the runnable
small/large regenerated cohorts. Superseded capture roots are preserved under
`/data1/archive/gpu_captures_20260908`; non-runnable debug and failed staging
trees are under `/data1/archive/results_misc_20260908`. After this cleanup, all
15 regenerated NVBit captures still pass admission, and the complete release
8-small/7-large replay sweep remains zero-exit, silent, and byte-identical.

Campaign snapshots bind dynamic traces and threadblock payloads before
materialization. The release campaign runner and direct replay path both pass
their focused package checks, and the retained canary fixture remains runnable.

The post-merge clock-qualified full campaigns at revision `9df50c47` are
retained under `results/captures/sm86_postmerge_campaign_small_20260909_v1`
and `results/captures/sm86_postmerge_campaign_large_20260909_v1`. All 8 small
and 7 large rows completed correctness, seven-repeat NCU, NVBit, static SASS,
and simulator stages with zero clock-contract mismatches. Simulator commands
now attach the matching raw NCU CSV through the observation-only `--ncu-csv`
surface; 15/15 outputs contain the parsed launch observation. Each manifest
also records the native causal-frontier probe (26 exact events, 19 causal,
one collapsed region, two reified regions, unchanged final cycle 9).

The same-input post-merge differential replay preserved every scored counter
and raw issue/terminal frontier for all 8 small and 7 large retained traces.
Fresh NCU medians are diagnostic because capture/profile sessions are not the
frozen scorer inputs; no per-workload timing or traffic constants were fitted.

The SM86 novelty slice now includes a pure `sim-semantics` event and
partial-order certificate crate, an SM86 NVBit adapter, and the
`semantic-probe` executable. The adapter preserves lane-byte addresses, alias
multiplicity, widths, optional masks, allocation IDs, cache operators, and
placement identity. `tools/semantic_events/sm86_roundtrip_compare.py` runs a
canonical event round-trip and then the existing cycle-accurate replay. All
fifteen retained rows round-trip with identical counters, timing, and coverage;
the sidecar classifier leaves no opcode unresolved. Static constant-cache
`LDC/ULDC` records still have explicit dynamic memory-class gaps because the
producer omits that tag; the adapter preserves them as static-only events and
never routes them through global memory. This is a metadata gap, not a timing
default.

The first exact-acceleration implementation slice is now present: a versioned
`sim-semantics::ObservationContract` distinguishes boundary and per-cycle
claims, `sim-engine::QueueStats` reports virtual clock gaps separately from
handler events, and schedule-derived causal inputs reject barriers, memory
roles, and uncharacterized dispatch-only operations before certification.
The optional dense reference differential now matches all 8 retained small
SM86 workloads. The event lane recorded 201,105 local idle jumps covering
3,789,118 virtual cycles across that set; dense mode remains the differential
reference and no hardware-speed claim is attached. The comparison is
reproducible with `tools/semantic_events/sm86_reference_compare.py`. The
retained 7-workload large cohort also matches 7/7; across all 15 rows the event
lane took 3,793,743 local steps and skipped 6,992,182 virtual cycles, while
dense mode took 10,785,925 local steps.

The optional `SIM_SM_REFERENCE_TRACE` emits post-event ordering plus canonical
warp/block boundary projections. The overall projection remains deliberately
marked incomplete while legacy debug-formatted duplicate fields are retained,
but L1, shared L2 ownership, async-group state, memory/scalar pipeline state,
instruction-front-end state, and fixed-clock activity power are now typed. Each
event carries versioned cache geometry/set/way state, canonical SM response
owners, pending-fill ready times, typed pending-event payloads, and an
activity-only power state. The typed L1/L2/async/pipeline/front-end/power root
record matches between event and dense replay, is complete at quiescent canary
boundaries, and explicitly reports incomplete ownership timing while an
outstanding response has no exact due time. The old aggregate projection flag
is retained as a conservative legacy field for V1 consumers.

## Run the live path

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
```

Use the retained NVBit trace and control sidecars from a current capture with
`sim/target/release/sim-run`; the exact command form is in
[runbook_host.md](runbook_host.md). For a quick numerical comparison:

```bash
python -m eval.direct_score --observed observed.json --predicted predicted.json
```

## Next engineering slice

1. Keep the current SM86 scheduler and memory changes small and measurable.
2. Replay one representative current trace after each change.
3. Profile event throughput and allocation cost when a replay exposes a real
   bottleneck.
4. Keep new behavior SM86-scoped until a separate architecture scope is
   explicitly reopened.
