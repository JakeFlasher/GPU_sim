# ADR 0001: Continue Task7/H3 with pinned evidence preserved

- Status: Accepted
- Decision date: 2026-07-13
- Decision code: `CONTINUE_PRESERVE_PINNED`

## Context

The previous Task7/H3 bootstrap attempt completed some GPU correctness work but
did not produce a terminal journal, DONE marker, accepted launch, admitted
NCU/NVBit/control evidence, or a campaign row. It therefore cannot be treated
as admitted evidence. The attempt and its surrounding artifacts may still be
unique and must not be discarded or silently reclassified.

## Decision

Continue the Task7/H3 live campaign outside RLCR under the existing campaign,
provenance, and hardware-evidence gates.

The incomplete prior attempt is historical, non-authoritative, and
non-reusable. No matrix, preflight result, compiled module, trace, label,
sidecar, journal fragment, or other output from that attempt may be used to
admit a new campaign row or satisfy a current gate.

Its preserved identity is:

- source HEAD: `9bfc1fc27fd24f062c7a1691e3a7d7b8b0b963c2`;
- chunk: `sol_execbench-000-bootstrap`;
- cell: `sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation`;
- journal inventory: 32 files, 3,888,602 bytes, inventory-domain SHA-256
  `b6569f61caf98a29e2b1a7711208f88938658f3d27582b4062fac1bf95fcbc5a`;
- whole staged chunk: 33 files, 3,889,540 bytes, inventory-domain SHA-256
  `2bf29fafa773052fc2c2bb5b4b982effb7c1f6ff16813c0d80af03eb41ab4d6f`.
- matrix file SHA-256:
  `e725709cdd7f4302befd123d391dcdffa233b40c37120b3226e39f56062c412d`;
- matrix seal SHA-256:
  `dabb26187c9617d5fd08c7ba6737af969f9202d7f7d3d164ad0a83407726ea5d`;
- preflight SHA-256:
  `b918531281b48ac3358b6a1e866cd4fa3da4c26babdcc4cf29412b4c833af898`;
- session: `88f01256-168a-4880-ac52-204014fe0040`;
- admission run:
  `e76313a5791db2c911a99113e9ca917dc354e2b32ee619d6e6d5660297dd3f35`.

The inventory domain is the SHA-256 of C-locale path-sorted rows encoded as
`<relative_path>\t<size_bytes>\t<file_sha256>\n`. Continuation must verify these
identities before preservation. Before any current-HEAD preflight or GPU work,
the single historical `journals` parent must be atomically moved into the
campaign's dedicated external `capture_defect_attempts` archive. The archive
copies and pins the matrix, preflight, and chunk-run journal context, seals the
moved bytes, and appends one exclusive ledger entry. It is separate from the
preparation-failure archive. The source is never deleted, overwritten, or
rolled back after the atomic move.

The typed reason is
`CAPTURE_DRIVER_INTEGRITY_FALSE_POSITIVE_BEFORE_ROI`: capture mode intentionally
bound `time_runnable` to `None`, while the integrity snapshot still required
the canonical timing function. The sealed result records one correctness call,
zero warmup calls, zero candidate calls inside the ROI, and zero profiler-start
calls. The archive is permanently non-authoritative and reuse-forbidden and
contributes exactly zero campaign rows. A new current-HEAD journal for the same
cell must carry a distinct `prior_capture_defect_attempts` ledger
back-reference. Any unarchived or reappeared foreign active journal is a
`CROSS_HEAD_ACTIVE_JOURNAL_UNRESOLVED` refusal before preflight or GPU work.

All pinned artifacts associated with the prior attempt remain preserved at
their existing byte identities. They must not be rebuilt, repaired, repinned,
overwritten, or deleted as part of continuation. Fresh work must use fresh
outputs produced under the current pinned contracts and must pass the current
gates on its own evidence.

## Consequences

- Live Task7/H3 campaign work is authorized to resume outside RLCR.
- The old incomplete attempt remains available only in the closed external
  capture-defect archive for historical diagnosis.
- Current accounting and publication claims receive no credit from the old
  attempt.
- Continuation must create new, fully validated evidence without mutating or
  repinning the preserved artifacts.
- Missing prerequisites, hash drift, or gate failures remain typed failures;
  this decision does not weaken any evidence or safety check.
