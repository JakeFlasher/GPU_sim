# Immutable publication contract

`tools/local_labels/immutable_artifact.py` is the shared same-directory
publication boundary for retained M5/M6 artifacts. It is a tamper-detection
and fail-closed contract, not a same-UID tamper-prevention mechanism.

## Anchored reads

Pinned reads transition from `CLOSED` to `CHAIN_PINNED`, `LEAF_PINNED`,
`CONSUMED`, and `REVERIFIED`, then return to `CLOSED`. Ancestor components are
opened descriptor-relatively with no-follow directory flags and bind each name
to device, inode, and file type. The leaf is a single-link regular file; its
content is consumed through the originally pinned descriptor, with exact-size
and EOF checks. The leaf's seven-field snapshot (device, inode, mode, nlink,
size, mtime, and ctime) must remain stable through consumption. Every failure
closes owned descriptors in reverse order.

Live-parent reads intentionally bind ancestor identity rather than requiring
ancestor timestamp stability: unrelated sibling activity can legitimately
change a live directory's metadata. Frozen published-tree verification applies
full entry snapshots and closure checks to the retained tree.

## Publication and retention

Publication uses deterministic staging/partial names, exclusive creation,
cooperative locks, stable hash/size checks, read-only modes, and
`RENAME_NOREPLACE`. Producer failure and recovery paths never unlink partials;
zero/short writes, copy failures, foreign replacement, and incomplete crash
remnants remain available for typed recovery or operator intervention. The
only producer-side name removal is the rename of an exact-complete,
identity-verified partial into an absent final name.

Public material names are separate from transaction names. Public components
cannot begin with a dot, contain control bytes, use reserved staging/partial
suffixes, or create file/directory prefix conflicts. Whole-material namespace
validation happens before a staging directory is created.

Regular-file and tree publication perform ordered file, directory, and parent
fsync operations. A failed fsync is `DURABILITY_UNPROVEN`; committed evidence
is not rolled back. Winner acceptance repeats durability proof, closure
verification, and caller authorization. Ordinary success also reauthorizes
after the final expensive verification and before its cheap final closure.

Large retained trees may use snapshot-closure verification with an iterative,
O(depth) descriptor budget: each file is hashed once, metadata is retained in
memory, the tree is reinventoried, and named entries are reopened
no-follow for identity and snapshot closure. Metadata closure detects races;
it does not prevent a same-UID writer from changing a writable parent.

## Threat boundary

`flock` is advisory. It serializes cooperating publishers that lock the same
inode but does not constrain a same-UID process that ignores the lock or
replaces the lock name. Modes `0444` and `0555` are discretionary point-in-time
attributes, not immutable protection. Pinned descriptors, inode/snapshot
identity, ctime-sensitive closure, and `RENAME_NOREPLACE` provide detection
and typed failure. Strong prevention would require a distinct owner or
privileged filesystem immutability controls and is outside this contract.

## Evidence boundaries

`evidence/path_aliases.v1.json` and `v2.json`, historical ADR bytes, and
point-in-time external snapshots remain frozen. Any successor alias authority
is inactive until its schema, decision record, clean/tracked state, hash
bindings, and explicit operator acceptance are all verified. M7 remains
upstream-blocked until its M4/M5/M6 receipts and downstream consumer contract
exist; this document does not activate M7 or claim hardware readiness.

`evidence/manifests/external_capture.sha256` is a point-in-time snapshot, not
an active whole-root inventory. Supplemental archives, preparation-failure
ledgers, M5 DONE bundles, M6 analysis roots, and frozen-label outputs close
only their owner-specific namespaces. The currently omitted future-M7
categories are out-of-band M5 trust-anchor distribution, durability receipts,
a whole-root closure authority, M4 receipts inside the M6 bundle, and an M7
DONE/verifier/consumer contract. These omissions remain typed
`M7_BLOCKED_UPSTREAM`; they are not silently filled by this contract.
