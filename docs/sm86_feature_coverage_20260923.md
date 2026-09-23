# SM86 feature and unsupported coverage — first cohort

Updated 2026-09-23. This is an inventory of the retained eight-row
SM86/SM80-labelled development cohort, not a universal support claim.

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

## Next inventory expansion

Run the same inventory over the retained timing-crosswalk holdout and the
larger regenerated cohort. Add a typed unsupported reason only when a concrete
input or replay path produces it; do not infer support from opcode names alone.
