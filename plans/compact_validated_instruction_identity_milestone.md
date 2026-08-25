# Compact validated-instruction identity milestone

**Completed:** 2026-08-20  
**Predecessor:** [compact direct trace lifetime](../docs/simulator_compact_trace_lifetime_milestone.md)  
**Goal:** reduce the remaining direct-replay peak without weakening trace
validation or changing replay-visible instruction identity

## Research result

The predecessor proposed one-file-at-a-time payload validation. A complete
descriptor-relative roster implementation was tested first. It binds every
payload path, size, inode, and generation before semantic decoding, then
reopens through the retained root and rejects replacement or mutation.

A three-round 12-case pilot falsified streaming as the main optimization:
gap-heavy RSS improved only 4.6%, while deep work and feedback cases were
essentially unchanged. The remaining peak was therefore the validated compact
representation itself. The gap trace contains 6,804,000 records, each retaining
a 32-byte `ValidatedInstruction`.

## Retained representation

- Function identity is already validated against the root launch and is stored
  once per owning warp.
- Threadblock warp identity is already validated against the protobuf map key
  and is reconstructed from that warp key.
- Per-record `sm_id` and `warpid_sm` remain exact signed i32 values.
- One explicit byte carries their presence bits, so `None`, zero, negative,
  `i32::MIN`, and `i32::MAX` remain distinct.
- PC and the memory-arena index remain per record.

This reduces `ValidatedInstruction` from 32 to 20 bytes without changing its
public borrowed accessor surface. Streaming roster support remains available
only as an opt-in performance-build experiment; it is not the default loader.

## Acceptance and outcome

The seven-round matched 12-case comparison requires byte-identical normal JSON,
no case above 3% paired timing regression, no memory regression above 1%, and
more than 10% RSS reduction on the gap-heavy case.

All criteria pass. RSS falls 7.8–22.4% across the cohort; gap-heavy falls 18.4%
and the worst paired timing regression is 1.98%. The final validation includes
the full Rust workspace and feature suites, formatting, affected Python tests,
all analysis reducers, and a regenerated 5,519-pair active replay corpus.

See the completed
[milestone report](../docs/simulator_compact_identity_milestone.md).

