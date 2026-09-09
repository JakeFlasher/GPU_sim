# Current status

Updated 2026-09-08.

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
4. Add SM120 behavior only from SM120 traces and cards.
