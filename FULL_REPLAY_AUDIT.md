# Full-replay merge audit

This audit reruns the copied real capture corpora in full mode with the unified
binary. Every invocation used `--workers 12`; no one-worker replay was used.
Fast replay was intentionally skipped.

## Final runs

| Lane / mode | Captures | Statuses | Launches | Workers | Output |
|---|---:|---|---:|---:|---|
| SM120, DRAM disabled | 339 | 339 `ok` | 11,558 | 12 | `full_replay_runs/fix3_sm120_dram_disabled/full.jsonl.zst` |
| SM120, carded DRAM enabled | 339 | 339 `ok` | 11,558 | 12 | `full_replay_runs/fix3_sm120_dram_enabled/full.jsonl.zst` |
| SM86 / RTX 3090 Ti | 356 attempted | 286 `ok`, 62 `control_error`, 8 `replay_error` | 12,392 | 12 | `full_replay_runs/fix6_sm86_full/full.jsonl.zst` |

The status/launch totals exactly match the retained baselines in
`final_results/dram_disabled`, `final_results/dram_enabled`, and
`final_results/3090ti`.

## Root causes fixed during the loop

1. **SM120 L1 lookup drift:** the SM86 line-allocation lookup formula had been
   applied to SM120. Added an architecture-scoped projection: SM120 retains
   pending-merge/four-sector semantics; SM86 retains the card-backed formula.
2. **SM120 store lookup drift:** the SM86 store-hit observer had been applied to
   SM120. Store lookup projection now dispatches by architecture.
3. **Shared wavefront doubling:** the merge had retained two aggregate
   `bank_wavefronts` increments. Removed the duplicate increment.
4. **SM86 branch-count drift:** the SM120 extended branch/control opcode list had
   been applied to SM86. SM86 uses the original direct-branch list; SM120 keeps
   the extended list.
5. **SM86 shared-wavefront drift:** SM120 phase-local/width-16 observer logic had
   been applied to SM86. SM86 now uses an exact compatibility projection and
   suppresses the SM120-only width-16 correction; SM120 retains the newer
   effective observer.
6. **SM120 global-load scope drift:** the SM86 narrow `memory_scope` count had
   been applied to SM120. SM120 restores the baseline broad logical memory
   count; SM86 retains the narrow static-SASS scope.

## Comparison result

`FULL_NUMERIC_COMPARISON.json` compares all common per-capture/per-launch
numeric fields against the retained final-result JSONL, excluding only known
nondeterministic wall-clock fields and explicitly additive schema observers.
The final result is **zero numeric discrepancies** for all three campaigns:

* SM120 DRAM disabled: 0 differing rows / 0 differing numeric fields.
* SM120 carded DRAM enabled: 0 differing rows / 0 differing numeric fields.
* SM86 full: 0 differing rows / 0 differing numeric fields.

The remaining output-schema differences are additive observer fields or
semantics text introduced by the unified merge; they do not change the
baseline numeric replay behavior.
