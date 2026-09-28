# SM86 feature and unsupported coverage — first cohort

> **2026-09-28 interpretation correction.** This dated report preserves its executable/overlap observations, but any `non-scoring`, `matched baseline`, or hardware-mismatch wording applies only to a direct executable cycle score. NCU on the local RTX 3090 Ti is detailed ground truth; published Accel-Sim aggregates are active accuracy-ceiling/throughput-speed-floor targets, registered in `docs/accelsim_literature_lower_bound_manifest_20260928.json`, not per-kernel ground truth.

Updated 2026-09-23 (historical cohort snapshot). This is an inventory of the retained eight-row
SM86/SM80-labelled development cohort, not a universal support claim.

The producer-side provenance gap identified by this snapshot was addressed for
fresh runtime-launched GEMM/RMSNorm captures on 2026-09-26; see
[the module-origin follow-up](sm86_module_origin_followup_20260926.md).

## Inputs and coverage

The cohort contains eight full-coverage NVBit traces, from 512 down to 8
blocks, and 2,000 distinct static instruction records across the eight kernels.
The static sidecars contain 100 distinct opcode spellings.

| Feature family | Cohort observation | Current replay status |
| --- | --- | --- |
| compute/scalar/vector instructions | 1,390 static records | modeled through trace/control schedule |
| global/shared/constant memory | 343 static records | modeled in the selected L1/L2 path where classified |
| control/synchronization | 247 static records | modeled through control schedule/barrier paths where supported |
| tensor instructions | 20 static records | model-specific tensor issue/readiness paths |
| dynamic block coverage | 8/8 rows full | supported for this cohort |
| unsupported memory-path records | empty in all 8 unsupported_memory_path.json files | no observed producer gap in this cohort |
| unmapped memory ranges | empty in all 8 unmapped_memory_range.json files | no observed address-map gap in this cohort |
| static module identity | origin_unresolved in all 8 traced-module records | provenance gap; static sidecar is separately bound to the workload binary |

Representative opcode families include HMMA, IMMA, LDGSTS, LDS/STS, STG,
ULDC, MUFU, BAR.SYNC, BSSY/BSYNC, DEPBAR, BRA, and uniform/predicate
operations. This is useful coverage evidence for the selected cohort only.

## Unsupported and trace-safety boundary

The empty producer-side unsupported arrays do not prove arbitrary SASS support.
They only say that these eight captures did not emit those two diagnosed input
failure classes. The simulator still reports typed unmodeled tails for dirty
writeback/coherence on every current-source row.

The current M2 issue-port override is trace_safe for this cohort under the
declared contract: it changes modeled issue/resource service while preserving
the captured instruction stream, addresses, predicates, synchronization
participants, and values. Changing ISA semantics, warp width, branch behavior,
atomic order, or address-generation behavior is requires_execution and must
not be scored by fixed-trace replay.

## Holdout and larger-cohort expansion

The same parser inventory now covers two additional retained cohorts:

| Cohort | Rows | Static records | Distinct opcodes | Class counts (compute/memory/control/tensor/collective) | Unsupported memory paths | Unmapped ranges | Static module status |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| timing-crosswalk holdout | 3 | 856 | 75 | 559/198/84/15/0 | 0 | 0 | 3 origin_unresolved |
| regenerated large | 7 | 3,536 | 130 | 2,625/581/296/33/1 | 0 | 0 | 7 origin_unresolved |

All ten rows retained exact active-launch contracts and full replay coverage.
The larger cohort adds collective behavior and broader tensor/control
spellings, while the producer-side unsupported arrays remain empty.

The persistent `origin_unresolved` values below remain historical observations
for the 18 inventoried rows (8 small + 3 holdout + 7 large); they are not
rewritten retroactively.  Fresh captures now resolve exact module/library
owners without same-name guesses, while JIT, deleted, opaque, and graph-only
origins remain explicitly unavailable.

No unsupported reason is inferred from an opcode spelling alone. New typed
unsupported categories should be added only when a concrete trace or replay
path produces one. The fair direct executable-cycle comparison remains
deferred; published Accel-Sim aggregate floors are evaluated separately against
local NCU ground truth.
