# SM86 predictive-closure implementation plan

**Status:** active implementation plan, 2026-10-01.
**Authority:** `AGENTS.md`, `docs/STATUS.md`, `docs/PROJECT_MEMORY.md`, and
[`docs/PREDICTIVE_CLOSURE.md`](../docs/PREDICTIVE_CLOSURE.md).
**Supersedes:** older exact-cycle, ownership-certification, cross-architecture,
and campaign-expansion plans. Those files remain historical source context.

## Goal

Turn the existing unknown-preserving SM86 replay into an NCU-targeted
predictive lane without claiming proprietary mechanism closure.

Completion means the model can emit a declared NCU prediction, simulator-native
frontiers, route/cache uncertainty, and a family-held-out score report. It does
not mean recovering physical pipes, cache owners, arbitrary SASS/ABI generation,
or a shared silicon/simulator clock.

## Non-goals

- no physical SM/SMSP/pipe/slot reconstruction;
- no unrestricted SM86 assembler, scheduler, linker, or CUDA ABI;
- no cache-owner/read-from/DRAM receipt claim;
- no NCU values as same-workload features;
- no new hash, approval, audit, reviewer, or mandatory workflow machinery;
- no rerun of completed full campaigns unless the measurement contract changes.

## Workstreams

### 1. Active contract and configuration

- Keep `sim-semantics::latent` as the unknown-preserving boundary.
- Add a versioned predictive-closure config containing resource-pool names,
  calibration provenance, endpoint policy, score metric, and OOD behavior.
- Keep historical committee paths in the archive; active docs point only to the
  compact contract and this plan.

### 2. Resource-demand quotient

- Add validated abstract resource-pool/service-demand types to
  `sim-semantics`.
- Extend `sim-sm::TraceMetrics` with demand counts and candidate-pool intervals
  separate from selected/scheduled diagnostic assignments.
- Convert existing physical-pipeline candidate classes and operand-bank
  pressure into the new demand envelope without assigning silicon routes.
- Add unit tests for empty pools, overflow, impossible intervals, and JSON
  round trips.

### 3. Calibration profile

- Define a small JSON profile for positive service rates, initiation intervals,
  register-bank capacity, memory queue capacity, and endpoint nuisance terms.
- Add a parser/validator plus the offline
  `tools/calibrate_sm86_service_profile.py` fitter; fitting consumes probe
  service observations and never score-workload NCU rows.
- Reject negative, zero-capacity, inverted, or unknown-unit parameters.

### 4. Cache and value semantics

- Retain current sector/reuse and transactional payload contracts.
- Add candidate read-from/write-epoch summaries to the predictive envelope.
- Keep model-local forwarding and writeback completion separate from hardware
  completion.
- Reuse current identity and store-before-event checks; do not infer residency.

### 5. NCU observer projection

- Add a typed observer descriptor for cache mode, serialization, clock policy,
  profiler start/end semantics, and included/excluded work.
- Project simulator frontiers into an NCU-targeted output without converting
  CUDA-event/CUPTI/device-timer values into simulator cycles.
- Preserve the existing ordinary output when the opt-in projection is absent.

### 6. Validation

- Probe cohort: independent synthetic kernels for resource/cache calibration.
- Development cohort: separate workloads used to freeze model choices.
- Score cohort: family-held-out NCU labels loaded only after prediction.
- Report cycle/duration error, interval coverage, counter error, ranking, and
  unsupported/OOD behavior.
- Treat self-authored tests as implementation evidence, not independent silicon
  validation.

## First vertical slice

Implement workstreams 1–3 first:

1. add `ResourceDemandEnvelope` and `ServiceRateProfile` to `sim-semantics`;
2. expose a `TraceMetrics::resource_demand_envelope()` adapter;
3. add a compact predictive-closure config and schema validation;
4. add unit tests and one retained canary projection;
5. update active docs and the supersession index.

This slice is useful without a new hardware endpoint and does not alter default
native replay.

## Later slices

- richer endpoint projection over `TimingReport` and calibrated observer
  padding;
- cache candidate-set refinement over per-event memory counters;
- independent probe corpus generation and family-held-out score harness;
- performance/interval report integration after a new independent corpus exists.

## Stop/research rule

If a proposed implementation requires a physical route, cache owner, arbitrary
compiler artifact, or synchronized endpoint, stop that seam and return to the
focused public-source/local-contract research before widening the code change.
