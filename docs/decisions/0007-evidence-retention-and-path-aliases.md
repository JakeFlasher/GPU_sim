# ADR 0007 — Proposed evidence-retention path-alias successor

- **Status:** Proposed; inactive pending explicit operator acceptance
- **Date:** 2026-07-21
- **Authority domain:** `round28_preparation_failure_adjudications`
- **Successor artifact:** `evidence/path_aliases.v3.json`
- **Supersedes:** `evidence/path_aliases.v1.json` only after acceptance and exact SHA-256 verification

## Decision

The existing `evidence/path_aliases.v1.json` and `evidence/path_aliases.v2.json`
remain byte-frozen retained evidence. This proposal does not rewrite, rename,
repurpose, or activate either authority. Historical ADR 0005 remains the
problem-039 capture-envelope decision and is not repurposed as an alias
successor decision.

The proposed v3 authority is scoped to the explicit authority domain
`round28_preparation_failure_adjudications`. Consumers must declare both the
domain and schema version. Unknown versions and wrong domains fail closed.
The v3 root carries the exact v1 supersession path and SHA-256, and every row
requires historical-to-canonical SHA-256 equality.

Rows support exact and prefix matching. A matching exact row takes precedence
over every prefix row. Otherwise the longest matching historical prefix wins.
More than one surviving exact row or equal-length prefix row is
`PREP_ARCHIVE_EVIDENCE_ALIAS_AMBIGUOUS`. No resolution is accepted without
regular-file, no-symlink endpoint checks and byte-exact SHA-256 equality.

## Activation gate

The v3 file is proposal-only and declares
`inactive_operator_acceptance_required`. The runtime dispatcher continues to
use the existing v1 path and its typed missing-authority behavior. Activation
requires explicit operator acceptance of both this ADR and the v3 artifact,
tracked and clean repository state, exact authority hashes, and a separate
consumer migration decision. No hardware or M7 milestone is implied.

## Consequences

Until the gate is satisfied, consumers must retain typed failures rather than
silently falling back to v3 or treating the proposal as an active authority.
The v1/v2 bytes and ADR 0005 are audited as immutable historical evidence.
