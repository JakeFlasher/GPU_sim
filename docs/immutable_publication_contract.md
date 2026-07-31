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
is not rolled back. Winner acceptance repeats durability proof, metadata/snapshot
closure, and caller authorization. Ordinary success also reauthorizes after the
final expensive verification and before its final metadata/inode closure. The
post-authorization closure does not rehash content, but it may traverse the
retained tree; it is bounded by descriptor depth rather than a constant-time
guarantee.

Regular-file publication accepts a caller-supplied `pinned_parent` only when it
is an exact `PinnedDirectoryReader`. Before creating or recovering any partial,
an independent reader binds the declared `parent` path to the same device/inode
as that capability; a type or inode mismatch leaves both the partial and final
leaf absent. Both bindings remain revalidated across the namespace commit. An
independent rollback-parent descriptor remains live through pinned-reader
monitor and directory closure, followed by a raw no-hook lexical rebind before
return. Renaming and recreating the declared parent during close therefore
fails publication, preserves the committed bytes through the exact displaced
parent, and leaves any recreated-path winner untouched.

`publish_retained_regular_file` keeps an independently opened, no-follow
capability to the exact committed inode after the publisher's writable handle
closes. Callers reauthenticate its payload hash, seven-field snapshot, named
path, and mutation monitor through downstream closure. On failed closure,
`preserve_retained_regular_file` retires that exact generation when it still
owns the name; if another publisher replaced the name, the replacement remains
active and the retained inode is salvaged through its descriptor instead.
Same-inode mutations are moved out of the active namespace as failed evidence.

`close_retained_regular_files_authenticated` closes one or more locators as a
single publisher-locked set. It first duplicates every exact inode and ancestor
chain into independent rollback guards. Mutation monitors remain armed while
the original `_PinnedRegular` capabilities and optional source/lock callbacks
close. A caller that owns a publication-lock reader transfers its sole reader
or reader-stack close callback into this phase; activation transfers its
lock-source stack, and roster detaches its aggregate reader owner before the
handoff. If setup fails before the callback is invoked, the detached owner
closes against that active error. Once invoked, no caller-owned reader close
remains after the last locator observation. The guards then reauthenticate
every path, snapshot, size, and payload
before and after monitor closure. Each authenticated guard is duplicated once
more into a final sentinel before the ordinary guards cross their hookable
`_close_descriptors` release boundary. Guard descriptor ownership is detached
before that close attempt: a reported close failure may already have released
an fd, so its uncertain numeric value is never retried and cannot close an
unrelated descriptor that reused the number. The complete sentinel set is
reauthenticated after the hookable release. Any pinned-close, monitor-close,
callback,
guard-release, or final-identity failure transfers every current recovery set
back to its caller, so the concurrent winner and all original
activation/aggregate evidence remain preservable. Only a fully authenticated
final-sentinel set is released, in one raw libc `close(2)` pass that attempts all
owned descriptors and exposes no later Python cleanup callback able to mutate a
locator after its last authentication.

Retained trees may use snapshot-closure verification with an iterative,
O(depth) descriptor budget: each file is hashed once, metadata is retained in
memory, the tree is reinventoried, and named entries are reopened no-follow for
identity and snapshot closure. Metadata closure detects races; it does not
prevent a same-UID writer from changing a writable parent.

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
