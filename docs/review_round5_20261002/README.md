# R9.1 production witness-dispatch attempt (2026-10-02)

The reference feature now has an optional typed-witness input and a guarded
`SmWake` dispatcher. This stage deliberately reports the negative result:
ordinary native traces do not supply a complete independent witness, so no
macro is selected.

## Bounded canary result

The repaired SM86 `l1-l2` canary was rerun with
`SIM_SM_REFERENCE_MAX_RECORDS=100000` and no witness. It completed with:

- 10,033 records including the footer;
- zero complete typed boundaries;
- zero native macro dispatches;
- exact replay/fallback for every boundary.

The witness/dispatcher path therefore has no positive production fixture yet.
The test-only issue-model helper is not promoted to native evidence.

## Required falsifier for the next loop

A separately produced complete typed entry/effect/exit witness must match a
real event, cause exactly one dispatcher selection, and produce the same
detailed continuation state. Perturbing one entry/effect/pending field must
fallback before mutation. Until that intervention exists, the native result is
the explicit negative boundary above.
