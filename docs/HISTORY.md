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

The final typed L2 client-scope observer is deliberately a report-only seam.
Its refinement tags write/atomic/reduction and async-bypass requests where
their static request kind is known, while preserving `Unknown` for
unqualified bypasses. The twelve-worker full replay comparison records no
SM86 common numeric differences and only the already-approved SM120 async-L1
projection differences; no physical hierarchy, DRAM, or timing path was
changed.

The 2026-10-19 Blackwell deep-research review is recorded in
[`BLACKWELL_RESEARCH.md`](BLACKWELL_RESEARCH.md). It confirms that the TMA
sidecars already contain decoded descriptor/coordinate/address evidence, but
adjudicates all proposed new active projections against full-corpus results:
TMA-shadow DRAM substitution, unconditional write hits, LTC128B promotion,
default 96-slice/carveout changes, and post-hoc IPC scaling remain rejected or
experiment-only. The validated replay state is unchanged.

The subsequent SM86 native unified-L1 correction is the one accepted follow-up:
it reuses the validated form-qualified global lookup endpoint, leaves SM120's
alpha path unchanged, and reduced joined SM86 unified-L1 APE from 29.04% to
12.15%. It does not yet satisfy the <10% objective.

## Repository cleanup

The active tree no longer treats branch-specific plans, H/S milestone diaries,
generated patch bundles, transient profiler dumps, or copied replay outputs as
execution instructions. Those artifacts remain recoverable through Git history
when historical archaeology is needed. Current behavior is documented only by
the contract, architecture, replay, results, and session-memory files linked
from [`README.md`](../README.md).
