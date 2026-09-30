# Current status

**Updated 2026-09-30.** This is the authoritative capability boundary. Read it
after [`PROJECT_MEMORY`](PROJECT_MEMORY.md). Historical reports under
`archive/research_20260930/` are evidence records, not active plans.

## Decision

The simulator has a runnable native SM86 replay lane and a retained fast NCU
projection lane. The full RTX 3090 Ti cold/warm campaigns are complete. The
research goal is **blocked** under the external-endpoint rule, not complete:
exact dynamic pipe/slot identity, unrestricted independent compiler/ABI/linker
generation, native cache ownership/read-from/DRAM completion, and simulator
completion synchronization remain unavailable through current public tooling.

## Execution surfaces

| Surface | Status |
|---|---|
| Native replay | Runnable SM86 alpha; issue/event/completion/launch/typed-tail frontiers remain separate |
| Custom model | Runnable and model-relative; not native SASS timing |
| NCU projection | Candidate point plus uncertainty interval for current contract; related-cohort evidence only |
| CUPTI/NAK/PMA/RM probes | Independent observations; no proprietary-state closure |
| Independent ABI/control ledger | Finite current ABI-v8 families execute without ptxas control words; general closure open |

## Final performance

Hardware: RTX 3090 Ti UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, SM86,
driver 615.71.09, locked 1350/10501 MHz, P0.

| Campaign | Rows | NCU policy | Median NCU | Median GPC | Median native | Median replay |
|---|---:|---|---:|---:|---:|---:|
| `full_3090ti_20260930_rm_cache_cold_live` | 15/15 | clock none/cache all | 10,080 ns | 13,572 | 0.009448 ms | 0.629196 s |
| `full_3090ti_20260930_rm_cache_warm_live` | 15/15 | clock none/cache none | 8,704 ns | 11,709 | 0.009435 ms | 0.644798 s |

See [`sm86_final_performance_handoff_20260930.md`](archive/research_20260930/sm86_final_performance_handoff_20260930.md).

## Evidence corrections

- Use `NV20_SUBDEVICE_0` for GR issue controls. Static rate values are
  available; device-handle `NOT_SUPPORTED` was a targeting error.
- Use SM70+/SM86 control fields: delay, yield, write barrier, read barrier,
  wait mask, reuse mask. Earlier SM120-style stall/rbar/misc labels are
  superseded by [`sm86_control_word_decode_correction_20260930.md`](archive/research_20260930/sm86_control_word_decode_correction_20260930.md).
- Raw-QMD and Driver timer text differs in tail control words; timing claims are
  semantic-body-level, not exact SASS-image identity.

## Four unresolved boundaries

| Boundary | Observable now | Required external closure |
|---|---|---|
| Physical pipe/issue | aggregate PMA/NCU, scheduler partition, operand pressure, control sensitivity | dynamic physical pipe/subunit/slot plus SM/SMSP trace |
| Compiler/ISA/ABI | finite no-template ABI-v8/control families | all-form SM86 encoder, scheduler, resources, relocations, linker, device ABI |
| Cache ownership/flush | public protocols, PTX operators, reuse/topology/WT observations | line owner/read-from/dirty-sector/DRAM completion telemetry |
| Unified timing | long raw/Driver/CUPTI/event/NCU crosswalk under explicit clock policy | synchronized simulator completion and endpoint identity |

## Current entry points

- Model config: `docs/sm86_refined_model_config_20260929.json`
- Final seam audit: `archive/research_20260930/sm86_final_external_seam_audit_20260930.md`
- Codegen matrix: `archive/research_20260930/sm86_independent_codegen_capability_matrix_20260930.md`
- Final performance: `archive/research_20260930/sm86_final_performance_handoff_20260930.md`
- Build/capture/replay: `runbook_host.md`

No repository test, hashing, or verification workflow is part of this status.
