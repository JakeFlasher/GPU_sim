# Source-branch comparison

This is the concise result of the exhaustive comparison performed before the
merge. The read-only source root was `/home/jakeshea/5090_branch/`; research was
limited to `final_results/`, `GPU_simulator/`, `GPU_simulator_5090/`,
`sm86_rtx_3090_capture/`, and `sm120_rtx_5090_capture/`. Paper and manuscript
trees were excluded.

## Common execution path

Both branches use the Rust pipeline:

```text
capture protobuf + static/control sidecars
    -> sim/trace -> sim/sm + sim/mem -> sim/engine -> sim/direct_run
    -> compressed replay JSONL -> optional independent NCU join
```

The merge keeps one implementation and dispatches architecture-specific
behavior through the architecture card and validated capture metadata.

## Branch differences

| Concern | SM86 branch | SM120 branch | Unified rule |
| --- | --- | --- | --- |
| Card | RTX 3090 Ti / SM86 card and P-chase inputs | RTX 5090 / SM120 effective card | Card identity is mandatory and capture metadata must agree. |
| Memory | Card-backed L1/L2 and carded DRAM hypothesis | Effective conditional L1/L2, TMA/async roles, optional carded DRAM | No cross-generation fallback; candidate documents are SM86-only until qualified. |
| Trace input | Native SM86 traces plus labeled portable controls | SM120 traces with TMA/async sidecars | Shared reader with typed architecture/sidecar validation. |
| Replay | 356 attempted captures, 286 successful full rows | 339 successful full rows | Same twelve-worker full runner and compressed JSONL schema. |
| Fast mode | Trace-derived compatibility frontier | Trace-derived flat frontier | Fast mode is separate and not used for merge correctness. |
| NCU | Independent comparison targets | Independent comparison targets | Never a replay/model input. |

## Interfaces examined

* `sm86_rtx_3090_capture/config/target.json`, controls, replay metadata, and
  the `final_results/3090ti` full baseline define the SM86 contract.
* `sm120_rtx_5090_capture/config/target.json`, dynamic captures, profiles,
  sidecars, and the `final_results/dram_disabled` and `dram_enabled` full
  baselines define the SM120 contract.
* `final_results` supplies reference rows; it is not copied into the simulator
  and its profiler observations are not used to select parameters.

## Resolved conflicts

The final full replay exposed six non-random merge drifts: SM120 L1 lookup and
global-load scope had inherited SM86 projections; SM120 store lookup had
inherited SM86 hit accounting; shared wavefronts were incremented twice; SM86
used the extended SM120 branch list; and SM86 used SM120 phase-local shared
corrections. The unified code now has explicit architecture dispatch for each
case. The complete comparison is summarized in `docs/RESULTS.md` and
`FULL_NUMERIC_COMPARISON.json`.

## Reproducibility boundary

Raw captures, baseline result trees, and large generated replay outputs are
external/ignored inputs. Use `docs/REPLAY.md` with explicit paths. Historical
branch plans, patch bundles, profiler dumps, and per-probe diaries were
removed from the active tree; Git history retains their provenance without
making them current instructions.
