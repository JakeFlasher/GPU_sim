# ADR 0003: Uplift the atomic-histogram capture envelope

- Status: Accepted
- Decision date: 2026-07-13
- Decision code: `ATOMIC_HISTOGRAM_CAPTURE_ENVELOPE_5400`

## Context

The first current-HEAD capture of `ac4_global_atomic_histogram` completed its
NCU label and NVBit target process successfully. NVBit admitted all
411,254,784 records across 256 threadblocks with zero drop in 1,976.030
seconds. The 6,307,671,733-byte raw trace correctly entered the required AC9
lossless-compression route, which produced a verified 3,404,105-byte retained
trace.

End-to-end route publication completed about 3,637.221 seconds after the cell
started. The declared 3,600-second cell envelope therefore expired before the
separate control and inventory phases could run. The runner refused with
`cell_deadline_exhausted`; it did not publish a cell-complete journal or chunk
DONE marker. The complete and incomplete journals were atomically preserved
through the stale-preflight supersession ledger and are non-authoritative.

## Decision

Set `ac4_global_atomic_histogram.max_wall_time_s` to 5,400 seconds. Keep its
2-GiB retained-trace ceiling and `sampling_policy = "off"` unchanged. Keep the
other sixteen portable-LCD rows at 3,600 seconds.

The 5,400-second tier already exists in the budget authority. Against the
observed route-completion time it leaves about 1,763 seconds, including at
least 1,463 seconds after reserving the control phase's maximum 300-second
allowance. This is a measured, cell-specific correction; it does not relax
trace integrity, retained-size admission, preflight, sealing, accounting, or
publication gates.

## Consequences

- The trace-budget authority and deterministic capture matrix receive new
  hashes and seals.
- Live capture resumes from a new source commit, authority, matrix, and
  preflight. Evidence sealed under the prior source commit is preserved but is
  not credited to the new campaign generation, except for the exact immutable
  portable bootstrap exemption already enforced by accounting.
- Any later deadline failure remains typed and fail-closed. A further uplift
  requires new measured evidence and an explicit contract edit.
