# ADR 0004: Uplift the SOL problem-027 capture envelope

- Status: Accepted
- Decision date: 2026-07-14
- Decision code: `SOL_PROBLEM_027_CAPTURE_ENVELOPE_10800`

## Context

The first current-HEAD capture of
`problem_027_grouped_query_attention_with_yarn_rope_and_qk_norm` completed its
NCU label, full NVBit trace, lossless AC9 route, and post-trace control. NVBit
admitted all 763,068,416 records across 1,536 threadblocks with zero drop. The
9,913,480,075-byte raw trace correctly entered the required AC9 route, which
produced a verified 750,034,333-byte retained trace.

The final `control_complete` gate ran 7,767.558450 seconds after the cell
started. The declared 7,200-second cell envelope therefore expired before the
result inventory and cell-complete journal could be published. The runner
refused with `cell_deadline_exhausted`; it did not publish a chunk `DONE.json`
or alter campaign accounting. The partial attempt remains non-authoritative
and must be preserved with its stopped campaign revision.

## Decision

Set
`problem_027_grouped_query_attention_with_yarn_rope_and_qk_norm.max_wall_time_s`
to 10,800 seconds. Keep its 1-GiB retained-trace ceiling and
`sampling_policy = "off"` unchanged. Keep every other SOL row at 7,200
seconds.

The 10,800-second envelope leaves 3,032.441550 seconds beyond the observed
control-complete time. This is a measured, cell-specific correction; it does
not relax trace integrity, retained-size admission, lossless routing,
preflight, sealing, accounting, or publication gates.

## Consequences

- The trace-budget authority and deterministic capture matrix receive new
  hashes and seals.
- Live capture resumes from a new source commit, authority, matrix, and
  preflight. Evidence sealed or staged under the prior source commit is
  preserved but is not credited to the new campaign generation, except for
  the exact immutable portable bootstrap exemption already enforced by
  accounting.
- The failed cell's partial raw and retained artifacts are never reused.
- Any later deadline failure remains typed and fail-closed. A further uplift
  requires new measured evidence and an explicit contract edit.
