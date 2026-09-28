# Current status

Updated 2026-09-28. This is the current capability boundary, not a declaration
that every research objective is complete. Read [PROJECT_MEMORY](PROJECT_MEMORY.md)
first; use [HISTORY](HISTORY.md) for dated context.

## Alpha policy

The active branch is fast and unbound. Alpha results do not require hashes,
freshness manifests, verification gates, correctness gates, or scoring gates.
The optional `--bound` forms are promotion tooling only and were not used for
the retained alpha campaign.

## Runnable surfaces

| Surface | Current implementation | Hard boundary |
|---|---|---|
| Native SM86 | NVBit/enhanced protobuf through `sim-run` with L1/L2/event replay | Fixed captured values/control/addresses; native retirement and cache tails remain |
| Custom `simt_u32_v1` | Functional dependency/resource scheduling, groups, shared memory, barriers, atomics, collectives | Fixed custom ISA; not native timing |
| Custom `simt_its_v1` | Single-group per-lane CFG, indirect u32 memory, `custom_its_dynamic_v3` trace | Functional only; not native SASS |
| `ptx_subset_v2` | Bounded text frontend with arithmetic, predicates, branches, indirect memory, deterministic atomic-add | No general PTX/ISA, ABI, calls, fences, vectors, SASS, or timing |
| Writeback/coherence | Explicit opt-in queues and typed L1 visibility/resident-dirty experiments | No proprietary ownership, eviction, payload, host flush, or native fence semantics |
| PhaseFuse | Conservative scalar frontier/refusal analysis | No memory, barriers, pipes, or silicon timing certification |
| `.traceg` probe | Strict ALU-only v2 adapter plus optional external parser smoke | Representation evidence only; no NCU or elapsed-time claim |

## Latest campaign

`results/alpha/full_3090ti_20260928_r3/performance.csv` contains the latest
15-workload RTX 3090 Ti alpha campaign: 15/15 completed. Companion metric and
writeback outputs are under `results/alpha/` and are intentionally ignored by
Git because they contain large raw traces and profiler exports.

Representative medians are recorded in `docs/PROJECT_MEMORY.md`. Native timing,
NCU timing, simulator event frontiers, and GPC envelopes are separate domains;
NCU used `--clock-control none`, so no drift-free timing score is admitted.

## Open blockers

- exact hardware physical-pipe/issue attribution;
- unrestricted ISA/compiler generation and CUDA ABI;
- native cache ownership/eviction/payload/host-flush semantics;
- drift-free locked-clock NCU timing;
- fully matched external timing domains;
- general value-dependent SASS regeneration and complete memory ordering.

Bounded substitutes remain useful development mechanisms but must not be renamed
as closure of these blockers.

## Review and handoff

- [Adversarial review and simplifications](code_adversarial_review_20260928.md)
- [Requirement audit](remaining_limitations_audit_20260928.md)
- [Dependency-ordered plan](remaining_limitations_plan_20260927.md)
- [SM86 campaign plan](../plans/sm86_next_experiment_campaign_20260927.md)
- [Host runbook](runbook_host.md)

No test suite or verification gate was run for the 2026-09-28 review. The
affected Rust path received only a build-only compile check.
