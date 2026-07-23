# ADR 0006: Uplift the SOL problem-040 capture envelope

- Status: Accepted
- Decision date: 2026-07-16
- Decision code: `SOL_PROBLEM_040_CAPTURE_ENVELOPE_18000`

## Context

The first current-HEAD capture of
`problem_040_altup_predict_correction_cycle_backward` completed its NCU label,
full NVBit trace, lossless AC9 route, and post-trace control. NVBit admitted
all 431,751,168 records across 221,184 threadblocks with zero drop. The
4,936,756,640-byte raw trace contained 221,192 files and correctly entered the
required AC9 route, which produced a verified 459,105,421-byte retained trace
containing 221,194 files.

The final `control_complete` gate ran 12,646.174374 seconds after the cell
started. The declared 7,200-second cell envelope therefore expired before the
result inventory and cell-complete journal could be published. The runner
refused with `cell_deadline_exhausted`; it did not publish a chunk `DONE.json`
or alter campaign accounting. The partial attempt remains non-authoritative
and must be preserved with its stopped campaign revision.

## Decision

Set
`problem_040_altup_predict_correction_cycle_backward.max_wall_time_s` to
18,000 seconds. Keep its 1-GiB retained-trace ceiling and
`sampling_policy = "off"` unchanged. Keep problem-027 and problem-039 at their
separately measured 10,800-second and 21,600-second envelopes, and every other
SOL row at 7,200 seconds.

The 18,000-second envelope leaves 5,353.825626 seconds beyond the observed
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
