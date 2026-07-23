# M6 Determinism Contract v2

This document describes the offline M6 determinism contract for the local
`sm_86` simulator campaign.  It is a producer/consumer contract, not evidence
that a hardware campaign has completed.  Collection remains disabled by the
offline-source gate; `hardware_done` and `authorized_for_handoff` remain false,
and M7 remains blocked.

## Scope and authority

M6 consumes a verified, externally sealed M4 archive and the v2 schedule,
journal, and analysis artifacts produced by
`tools/local_labels/m6_determinism_campaign.py`.  NCU and hardware values are
labels and scoring targets only.  They do not enter schedule features,
simulator configuration, architecture constants, calibration, or feature
selection.  The canonical publication metric remains
`gpc__cycles_elapsed.max`; M6 itself does not make a publication claim for that
metric.

The active v2 schemas are:

- `m6_determinism_schedule.v2.schema.json`;
- `m6_determinism_observation_journal.v2.schema.json`;
- `m6_determinism_analysis.v2.schema.json`;
- `m6_determinism_done.v2.schema.json`.

The v1-named schemas are frozen historical authorities.  They may be inspected
through explicit legacy dispatch, but a v1 artifact cannot authorize an active
M6 publication.  Schema paths, `$id` values, and SHA-256 bytes are pinned by
the controller.

## Schedule identity and coordinates

The frozen sm_86 matrix contains 344 cells, of which 122 are runnable.  The
schedule has six configurations and a single global slot coordinate:

- Tier 1 ordinals are `0..8539` (8,540 slots);
- Tier 2 candidate ordinals are `8540..9759` (1,220 candidates);
- the maximum schedule is 9,760 slots;
- every ordinal is unique and globally contiguous;
- Tier 2 candidates use the reference configuration and are not active until a
  causal trigger decision authorizes them.

A slot identity hashes its ordinal, tier, sweep/repeat index, cell identity,
configuration identity, and fresh-process isolation.  The schedule ID is
independently derived from the source, configuration, metric, and seed
identities.  Validation rederives matrix semantics, source paths, registry and
sidecar authorities, metric projections, and all seals from canonical repository
bytes.  Byte-identical copies at caller-selected paths are not substitutes for
the closed canonical authority set.

`rehash_sources=False` is structural inspection only.  It cannot publish a
journal or analysis, bind complete M4, or create `M6_DONE.json`.

## Invocation snapshots

One invocation-local snapshot cache binds each declared source path to its
canonical path, declared digest and size, verified bytes, and pre/post stable
seven-field identity:

`device`, `inode`, `mode`, `nlink`, `size`, `mtime_ns`, and `ctime_ns`.

Repeated reads use the bound payload.  A changed inode, byte payload, path
alias, or mixed generation fails closed with a typed snapshot error.  Publication
parents are pinned and revalidated through the final no-replace operation;
parent-directory fsync is part of the durability proof.

## Causal observation journal

The active journal is a hash-chained causal stream.  Its records are:

1. `epoch_open`;
2. `attempt_started`;
3. `attempt_terminal` (observed or typed exclusion);
4. `tier1_closed`;
5. `trigger_decision`;
6. `tier2_activated`;
7. `checkpoint`;
8. `epoch_closed`.

Replay, rather than caller summaries, derives attempt linkage, ordinal
consumption, costs, pending work, trigger state, and epoch transitions.  The
replay checks predecessor hashes, no gaps or regressions, one terminal per
attempt, balanced epochs, Tier-1 closure before a trigger, a trigger before
Tier-2 activation, and exactly ten candidates for each activated cell.  A
later ordinal cannot close while an earlier eligible ordinal is pending.

An operator stop is a typed exclusion and requires a matching 64-hex authority
receipt.  Failed attempts retain their start linkage and a typed outcome/cost.
Finite out-of-range observations remain row classifications; they are not
silently converted into slot exclusions.  Diagnostic partial journals may be
retained, but they cannot produce a DONE seal.

## Analysis and scientific gates

M6 analysis distinguishes diagnostic publication, incomplete evidence, M6
DONE, the unreachable M6-to-M7 handoff candidate, and M7 DONE.  It also keeps
simulator scoreability and publication verdicts machine-distinct.

A complete analysis must provide:

- exact closure of the 6 x 55 selected metric buckets;
- accounting identities for slots, attempts, rows, and typed exclusions;
- at least 30 valid cells per metric, or an explicitly operator-authorized
  fallback receipt matching that metric;
- finite, nonnegative available noise floors, including a legal zero floor;
- no pending denominator;
- positive absolute resolution plus exact-match semantics for deterministic
  denominators;
- the reference out-of-range-zero proof;
- sealed limitations covering censoring, finite-N behavior, within-session
  effects, multiplicity, and historical reconciliation.

A missing floor, pending denominator, unresolved attempt, insufficient inventory,
legacy top-level schema, or unsealed limitation refuses DONE rather than
becoming a vacuous pass.  The handoff predicate remains false because no
versioned production M7 consumer exists.

## M4 sealing boundary

The capture producer emits per-chunk writable evidence.  Before `bind-m4`, the
operator-gated `tools/local_labels/seal_m4_archive.py` performs the additive
transition:

1. require one external absolute archive root;
2. require exactly the 18 canonical chunk directories;
3. verify every `chunks/<chunk_id>/DONE.json` with
   `campaign_done_manifest.verify_seal` and require an empty problem list;
4. reject symlinks, hard-linked regular files, non-regular entries, missing
   chunks, extra chunks, and unstable trees;
5. change only modes: regular files to `0444` with `nlink == 1`, directories to
   `0555`;
6. fsync files/directories and re-inventory the tree; and
7. publish one immutable no-replace seal receipt outside the archive.

The sealer never renames, rewrites, truncates, deletes, or cleans evidence.
A second invocation verifies the current bytes and returns the existing sealed
receipt without rewriting it.  M6 independently pins and re-verifies the
archive; a sealer receipt is not itself a hardware result.

Interrupted M6 staging is handled separately by
`tools/local_labels/m6_recover_staging.py`.  It supports only byte-exact
`EXACT_PREFIX_COMPLETE` and `EXACT_PREFIX_PARTIAL` recovery.  Contradictory
bytes, inode identity, mode, type, or topology are preserved and recorded in
an immutable recovery receipt; recovery never turns an unresolved attempt into
an observation.

## DONE inventory and state boundaries

The dependency-closed DONE inventory binds schedule, journal, analysis, M4
receipt/archive root, all 18 M4 DONE hashes, retained evidence, matrix/roster/
registry/sidecar authorities, controller/statistics/immutable-helper sources,
active schemas, recovery artifacts when used, and the invocation source
snapshot.  Each inventory path is safe-relative and each payload is re-read
through the pinned snapshot before it is admitted.

The following states are intentionally distinct:

- `diagnostic_publication`: inspectable output, not completion;
- `M6_INCOMPLETE_EVIDENCE`: evidence or causal work remains;
- `M6_DONE`: all active gates and inventory dependencies close;
- `M6_M7_HANDOFF_CANDIDATE`: a predicate that is currently unreachable;
- `M7_DONE`: owned by a future, versioned M7 consumer.

No state in this document authorizes hardware capture, privileged commands,
publication of a cycle prediction, or migration to `sm_120`.
