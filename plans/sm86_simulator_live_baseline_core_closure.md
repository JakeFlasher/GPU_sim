# SM86 simulator live baseline and core-closure milestone

## Goal Description

Re-establish the current event-driven GPU simulator as a reproducible,
evidence-backed `sm_86` tool on the local RTX 3090 Ti, then close one
high-value simulator execution seam chosen from fresh evidence. Run the
matched physical-GPU path (correctness, NCU, NVBit, static SASS, and
`sim-run`) for a representative six-workload cohort, refresh or explicitly
reclassify stale replay outputs, and compare the simulator's causal
diagnostics with live observations. A null result or a typed incomplete
frontier is an acceptable scientific outcome; do not fit a per-workload
constant merely to produce a number.

The current focus is the GPU simulator itself. JADE/HETR, CPU-GPU co-simulation,
and unrelated host-integration work are explicitly deferred. `sm_120` remains
out of this milestone until the documented Ampere transition criterion is met.

## Current closure state (2026-08-30)

The retained six-workload baseline and the follow-on LDGSTS burst assay remain
non-promoting. The smaller store-free async-source slice first removed global
store/barrier tails. Its route-controlled destination-layout successor now
holds source addresses and traffic exact while changing only shared topology.
Across depth 4 and held-out depth 8, the cache-all model predicts the live
layout delta within 1.01%; bypass shows a repeat-positive 4.58%/6.39% live
effect while current replays remain layout-identical behind the typed
`async_copy_shared_visibility` boundary. The detailed records are
[`docs/sm86_async_source_only_milestone.md`](../docs/sm86_async_source_only_milestone.md)
and
[`docs/sm86_async_source_visibility_milestone.md`](../docs/sm86_async_source_visibility_milestone.md).

The follow-on three-frontier phase assay now separates those mechanisms. All
108 compact/conflict issue deltas are zero; all 108 post-issue drain deltas are
positive. Cache-all scales linearly with depth as the current model predicts,
whereas L2-hit bypass remains near 11–14 cycles per group over depths 1–8.
The detailed result is
[`docs/sm86_async_visibility_phase_milestone.md`](../docs/sm86_async_visibility_phase_milestone.md).

This closes issue replay as the cause and rejects a group-aggregate bypass
constant. Per-copy source replies, payload work, and projected destination
wavefronts are now preserved in cold replay state and exposed by readiness
schema 14, with no timing change. The retained result is
[`docs/sm86_async_per_copy_observation_milestone.md`](../docs/sm86_async_per_copy_observation_milestone.md).

The trace projection separates the layout pair by 2× while live `.cg`
wavefronts separate it by 4×. The additional severe geometry is now complete:
cache-all transfers exactly, while bypass conflict/severe have equal NCU
wavefront counts but a repeat-stable +4.013-cycle drain difference. See
[`docs/sm86_async_bypass_projection_milestone.md`](../docs/sm86_async_bypass_projection_milestone.md).

This falsifies both scalar projection choices. Keep the typed lower bound and
scored defaults unchanged. Any later bypass timing slice must preserve
source-cohort/destination-line fanout separately from bank multiplicity and
apply it per copy; do not fit another scalar to these three consumed layouts.

The orthogonal skew slice and schema-15 implementation now preserve those
dimensions separately. The trace-derived proxy matches all compact/conflict/
skew/severe NCU topology counts, but the timing-neutral local candidate fails
depth transfer. See
[`docs/sm86_async_bypass_multidim_milestone.md`](../docs/sm86_async_bypass_multidim_milestone.md).
No bypass timing or tail removal is promoted. A future core slice requires an
arrival-spacing discriminator for a pipelined per-copy/cohort service; the
consumed local-latency candidate must not be activated or retuned.

## Acceptance Criteria

- AC-1: A current-host preflight and release build are recorded for the RTX
  3090 Ti (`sm_86`), CUDA/NVBit/NCU tool versions, clock state, and repository
  revision. A small live CUDA correctness smoke and a fresh NVBit trace both
  complete with bounded commands and inspectable progress/errors.
  - Positive evidence: the selected workload exits correctly, the trace has
    full required payload/metadata files, and release `sim-run` consumes it.
  - Negative evidence: a missing permission, parser limitation, or stale
    binary is reported as a typed blocker rather than silently bypassed.

- AC-2: The six-workload Ampere cohort has matched, provenance-bound stages:
  correctness, NCU timing/traffic observations, NVBit traces, static control
  data, and simulator replays. NCU and NVBit are separate executions; NCU
  values remain comparison targets and never become simulator inputs.
  - The initial live slice may be a bounded subset, but the plan records the
    exact remaining cells and a resumable command for the full cohort.
  - Every retained cell records workload/arguments, GPU/SM, tool versions,
    binary identity where available, trace/control/card paths, and salient
    output metrics.

- AC-3: Current release `sim-run` replays the fresh cohort and the active
  retained corpus deterministically. Instruction/traffic/sector invariants,
  coverage, modeled event times, readiness causes, and typed incomplete tails
  are summarized alongside live NCU values. Any freshness failure is resolved
  by a minimal identity-safe regeneration or a documented validator correction;
  no broad workflow gate is introduced.

- AC-4: One highest-value simulator-core gap is selected from the fresh
  residual/diagnostic table. Either:
  - implement the smallest architecture-local change that explains a
    discriminating live/held-out effect and validate it on an untouched cell;
    or
  - record a falsified/no-go result with the next discriminating experiment.
  No candidate is promoted from an in-sample fit, a self-authored test alone,
  or an NCU value copied into a card/configuration.

- AC-5: The final handoff states the causal mechanism (or why it remains
  unknown), commands and observed results, reproducibility metadata, claim
  limits, and next GPU-simulator action. No JADE/HETR file, CPU-GPU bridge,
  `sm_120` behavior, or unrelated campaign is changed.

## Path Boundaries

### Upper Bound

The six current `sm_86` production/probe workloads represented by the existing
cycle-residual cohort: cp.async canary, BF16 HMMA, FP16 HMMA, INT8 IMMA, sparse
BF16 HMMA, and BF16 forward attention. Their direct live artifacts, static
sidecars, deterministic `sim-run` replays, one residual analysis, and one
minimal simulator-core patch or terminal no-go.

### Lower Bound

One clean correctness → NVBit → static-control → `sim-run` → NCU comparison
for a representative memory/async workload, plus a fresh release build and
an explicit report of which cohort stages remain. The lower bound must still
produce a useful simulator diagnostic and a reproducible next action.

### Allowed Choices

- Use existing research-probe Makefiles, `tools/nvbit_tracer`,
  `tools/local_labels/collect_ncu.py`, current architecture cards, and direct
  `sim-run` interfaces.
- Use the existing six-workload retained corpus as a comparison baseline, but
  label it historical and refresh outputs when current source identity makes
  them stale.
- Add bounded logs/counters or a small analyzer/report under simulator or
  research-probe paths when they directly expose the selected gap.
- Use fixed clocks only when already available or explicitly authorized and
  always record/restorable state. `--clock-control none` is valid when a lock
  cannot be obtained.
- Keep current typed `complete`/incomplete frontier semantics when live data
  do not identify a safe elapsed-cycle mapping.

### Out of Scope

- JADE, JADE_HETR, CPU-GPU co-simulation, G1/M3/M4 tracks, and host command
  coupling.
- `sm_120` timing or inherited Ampere constants.
- CounterTrace-GPU formal checker implementation, Vortex/Ventus pilots, or a
  new research-direction literature program.
- Reopening rejected LDGSTS/front-end, scheduler, queue, hash, or completion
  candidates without a new live discriminator.
- Large speculative test suites, per-workload calibration constants, broad
  refactors, workflow/approval gates, or cryptographic authorization schemes.

## Dependencies and Sequence

1. **Preflight and baseline execution**
   - Record host/GPU/toolchain/revision state and inspect the current release
     build, traces, cards, and retained replay freshness.
   - Run one correctness smoke, one bounded NVBit capture, one static-control
     inspection, one NCU target launch, and one `sim-run` replay.
2. **Matched live cohort**
   - Build or reuse the six `sm_86` probe binaries.
   - Run correctness, NCU, NVBit, and static stages serially with resumable
     per-cell outputs and bounded subprocess timeouts.
   - Replay each admitted trace with cold `l1-l2` state and retain the causal
     readiness/event diagnostics; preserve a separate warm/stream-L2 run only
     when the launch chain is explicitly present.
3. **Freshness and residual table**
   - Validate schema, coverage, instruction/traffic conservation,
     deterministic twin bytes, and output freshness against the current
     release binary and inputs.
   - Compare the simulator's last-issue/event/completion frontiers with NCU
     `sm__cycles_active.max`/`gpc__cycles_elapsed.max` and report typed gaps
     instead of collapsing them into one scalar.
4. **One core-closure decision**
   - Select the largest repeat-stable, mechanism-local residual with a cheap
     held-out discriminator.
   - Implement and rerun only that seam, or document the falsification/no-go.
   - Regenerate only affected replay outputs first, then expand to the full
     active cohort if the change is byte- and behavior-safe.
5. **Handoff and next slice**
   - Update the relevant simulator status/roadmap document, retain commands and
     results, commit the bounded change, and state the next GPU-only milestone.

## Validation Commands

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
cargo test --manifest-path sim/Cargo.toml --workspace --all-targets
make -C tools/nvbit_tracer -j"$(nproc)"
python3 workloads/research_probes/validate_active_replay_pairs.py
```

Use the smallest affected Python analyzer/tests and a real GPU smoke before
considering the broad commands. Record failed commands and their first
concrete error; do not hide an unavailable NCU metric, parser, or completion
tail behind a fabricated value.

## Evidence and Reproducibility

Every live claim names the command, workload and arguments, architecture
profile, GPU/driver/tool versions, seed/configuration, source/binary identity
when available, output path, and salient result. NCU observations are held out
from simulator inputs and tuning. A self-authored test is supporting evidence,
not an independent oracle; executable behavior, invariants, and differential
comparisons decide promotion.
