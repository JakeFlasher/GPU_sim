# Continuation memory

**Updated 2026-10-01.** Read this file, then [`STATUS.md`](STATUS.md), then
[`PREDICTIVE_CLOSURE.md`](PREDICTIVE_CLOSURE.md), and only the one focused
archive report needed for the next seam. `AGENTS.md` governs execution.
Historical reports and plans are opt-in context, not queues or gates.

## Scope

- Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
- Hardware anchor: RTX 3090 Ti / GA102 / SM86, UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09.
- Locked campaign point: SM 1350 MHz, memory 10501 MHz, P0.
- Native path: `NVBit/SASS -> enhanced protobuf -> sim-trace -> sim-sm /
  sim-mem / sim-engine -> sim-run`.
- Custom `simt_u32_v1` and functional `simt_its_v1` are separate model-relative
  paths, not native SASS timing.
- NCU is the declared comparison target, never a same-workload feature.

## Decision

Exact mechanism closure remains **blocked**. Public tooling still does not
provide physical pipe/SMSP/slot traces, unrestricted independent SM86
compiler/ABI/linker generation, cache-owner/read-from/DRAM completion
telemetry, or a synchronized simulator/hardware endpoint.

The active replacement objective is **predictive closure**: a route/cache latent
quotient plus an explicit NCU observer projection, evaluated on held-out NCU
labels. See [`PREDICTIVE_CLOSURE.md`](PREDICTIVE_CLOSURE.md) and the active
implementation plan in `../plans/sm86_predictive_closure_implementation_20261001.md`.

## Final campaign evidence

| Campaign | Rows | NCU policy | Median NCU | Median GPC |
|---|---:|---|---:|---:|
| `full_3090ti_20260930_rm_cache_cold_live` | 15/15 | clock none/cache all | 10,080 ns | 13,572 cycles |
| `full_3090ti_20260930_rm_cache_warm_live` | 15/15 | clock none/cache none | 8,704 ns | 11,709 cycles |

These are independent score records, not mechanism labels. Do not rerun the
full campaigns unless the endpoint contract changes.

## Current implementation surfaces

- Native replay is runnable; issue, modeled-event, launch-envelope, typed-tail,
  visibility, writeback, and endpoint frontiers remain separate.
- `sim-semantics::latent` and `sim-run --latent-envelope` preserve route,
  compiler, cache, and timing unknowns with candidate domains and checked
  frontier/endpoint descriptors.
- `TraceMetrics` exposes candidate pipeline classes and operand-bank pressure
  as model bounds, never physical route observations.
- Payload sidecars, residency witnesses, store-bound event ingress, and
  forwarding ledgers are opt-in model contracts; they do not certify silicon
  residency or DRAM completion.
- The finite SM86 value lane covers selected arithmetic, predicate, branch,
  memory, `LOP3.LUT`, funnel-shift, and bounded `IMAD` forms. It is functional
  regeneration, not arbitrary SASS timing.

## Exact boundaries still open

| Boundary | Current response | Missing evidence for exact closure |
|---|---|---|
| Physical route | Candidate pools, class intervals, pressure envelope | Dynamic `(SM,SMSP,pipe,slot,cycle,PC,warp)` trace |
| Compiler/ABI | Trace capsule and finite independent families | Complete encoder/scheduler/relocations/linker/device ABI |
| Cache/coherence | Sector replay, reuse, causal candidate writes, model queues | Owner/read-from/dirty publication/DRAM acknowledgement |
| Endpoint | Separate simulator and NCU descriptors | Shared synchronized completion identity |

## Next code seam

The first predictive slice is implemented: validated abstract resource-pool
and service-rate types, profile loading, a `TraceMetrics` resource-demand
adapter, cache candidate summaries, an opt-in NCU observer interval, and an
offline probe-profile fitter. The checked-in profile is abstract/pending
independent probe calibration. The next slice is independent probe-corpus
calibration and family-held-out score reporting;
default native replay remains unchanged.

## Evidence and supersession

- Final external seam: `archive/research_20260930/sm86_final_external_seam_audit_20260930.md`
- Public endpoint recheck: `archive/research_20260930/sm86_public_endpoint_recheck_20260930.md`
- Physical-pipe recheck: `archive/research_20261001/sm86_physical_pipe_endpoint_recheck_20261001.md`
- Predictive replacement research: `archive/research_20261001/sm86_predictive_closure_replacement_20261001.md`
- Capability history: [`HISTORY.md`](HISTORY.md)

Do not delete retained campaign data, schemas, tests, source cards, or dated
evidence while trimming context. Do not add hash/seal/reviewer/approval
machinery. Passing self-authored tests are implementation evidence only.
