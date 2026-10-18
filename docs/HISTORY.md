# Merge history

The repository combines the SM86/RTX 3090 Ti lane and the SM120/RTX 5090
lane from the read-only source checkout. The merge keeps one Rust trace → SM →
memory → event-engine → direct-run pipeline and selects architecture-specific
behavior from the card and validated capture metadata.

## Resolved merge drifts

The full-corpus loop exposed and fixed six non-random differences:

1. SM86 L1 lookup projection was incorrectly reused for SM120.
2. SM86 store lookup accounting was incorrectly reused for SM120.
3. Shared bank-wavefront totals were incremented twice after aggregation.
4. The SM120 extended branch list was incorrectly applied to SM86.
5. SM86 received SM120 global-load scope and phase-local shared corrections.
6. SM120 and SM86 shared-wavefront projections were separated explicitly.

The final three-campaign comparison is recorded in `docs/RESULTS.md` and the
machine-readable `FULL_NUMERIC_COMPARISON.json`.

The adversarial review also made the verification surface fail closed: the
runner strips inherited `SIM_*` behavior overrides, enforces complete-corpus
status counts, rejects architecture-free candidate lanes on both public and
direct replay paths, and refuses malformed/non-regular preferred sidecars.
The comparator detects duplicate/missing IDs, missing status-specific schema
fields, non-finite values, and baseline numeric fields dropped by a unified row.

## Repository cleanup

The active tree no longer treats branch-specific plans, H/S milestone diaries,
generated patch bundles, transient profiler dumps, or copied replay outputs as
execution instructions. Those artifacts remain recoverable through Git history
when historical archaeology is needed. Current behavior is documented only by
the contract, architecture, replay, results, and session-memory files linked
from [`README.md`](../README.md).
