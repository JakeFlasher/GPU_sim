# Current status

**Updated 2026-10-01.** This is the compact capability boundary. Read it after
[`PROJECT_MEMORY.md`](PROJECT_MEMORY.md). Historical reports are indexed by
[`HISTORY.md`](HISTORY.md) and are not active plans.

## Decision

The native SM86 replay lane and NCU comparison lane are runnable. Exact
proprietary-mechanism closure is **blocked**, not complete. The active software
objective is predictive closure: model hidden mechanisms as route/cache/endpoint
latent domains and predict the declared NCU metric on held-out workloads.

## Runnable surfaces

| Surface | Status |
|---|---|
| Native replay | SM86 trace-driven issue/memory/event replay; frontiers remain separate |
| Custom model | `simt_u32_v1` and functional `simt_its_v1`; model-relative only |
| NCU projection | Candidate point/interval outputs; NCU labels remain score-only |
| Latent envelope | Opt-in route/compiler/cache/timing candidates and endpoint descriptors |
| Payload path | Opt-in identity-bound sidecar, residency witness, event ingress, and model ledger |
| Value lane | Finite value-dependent functional forms; unsupported forms refuse |
| Predictive closure | First resource-demand/service-rate slice is implemented in the opt-in latent report; calibration, cache quotient, and NCU prediction remain |

## Final NCU comparison records

Hardware: RTX 3090 Ti, SM86, driver 615.71.09, locked 1350/10501 MHz, P0.

| Campaign | Rows | Cache policy | Median NCU | Median GPC |
|---|---:|---|---:|---:|
| `full_3090ti_20260930_rm_cache_cold_live` | 15/15 | `all` | 10,080 ns | 13,572 |
| `full_3090ti_20260930_rm_cache_warm_live` | 15/15 | `none` | 8,704 ns | 11,709 |

These measurements are independent comparison targets. They do not identify
pipes, owners, compiler decisions, or a shared simulator endpoint.

## Four unresolved boundaries

| Boundary | Observable now | Exact closure would require |
|---|---|---|
| Physical pipe/issue | Aggregate PMA/NCU, scheduler/resource bounds | Dynamic physical route and slot trace |
| Compiler/ISA/ABI | Finite no-template SM86 families and trace capsules | Unrestricted encoder, scheduler, relocations, linker, device ABI |
| Cache ownership/flush | Protocol, reuse, topology, model queues | Owner/read-from/dirty-sector/DRAM completion telemetry |
| Unified timing | Separate simulator/NCU/CUPTI/event descriptors | Synchronized completion endpoint |

## Predictive-closure contract

The active replacement is documented in
[`PREDICTIVE_CLOSURE.md`](PREDICTIVE_CLOSURE.md). Its rules are:

- route identity becomes abstract candidate resource demand;
- compiler/ABI becomes an opaque trace/semantic input boundary;
- cache ownership becomes sector/reuse plus causal candidate read-from state;
- timing becomes simulator frontiers plus an explicit NCU observer projection;
- NCU labels are loaded only after prediction for the score cohort;
- OOD inputs widen intervals or refuse; they do not receive fabricated precision.

## Immediate implementation

See [`../plans/sm86_predictive_closure_implementation_20261001.md`](../plans/sm86_predictive_closure_implementation_20261001.md). The first slice adds validated resource-demand/service-rate contracts and a `TraceMetrics` adapter. It must preserve ordinary replay and existing output schemas.

## Authoritative entry points

- Predictive config: `docs/sm86_predictive_closure_config_20261001.json`
- Resume: `docs/PROJECT_MEMORY.md`
- Active replacement: `docs/PREDICTIVE_CLOSURE.md`
- Implementation plan: `plans/sm86_predictive_closure_implementation_20261001.md`
- Build/capture/replay: `docs/runbook_host.md`
- Architecture/configuration: `docs/ARCHITECTURE.md`, `docs/archprofile.md`
- Final seam audit: `docs/archive/research_20260930/sm86_final_external_seam_audit_20260930.md`
- Latest endpoint recheck: `docs/archive/research_20261001/sm86_physical_pipe_endpoint_recheck_20261001.md`

No repository test, hash, verification, audit, or approval workflow is part of
this status.
