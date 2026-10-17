# SM86 LTC128B probe boundary — 2026-10-16

H61 attempted a direct PTX form-specific probe in the notes project using:

```ptx
ld.global.LTC128B.u32
```

CUDA 13.2 `ptxas` rejects the modifier deterministically:
`Unknown modifier '.LTC128B'`. The form is a SASS-level encoding not exposed by
the supported PTX frontend. The existing H48 corpus traces and cuobjdump
observations remain the valid LTC128B evidence; forcing an inline opcode or
binary patch would invalidate functional provenance and is not attempted.

Decision: retain LTC128B as an observer-only form, defer an independent
LTC128B NCU sweep until a supported SASS-generation/injection path is available,
and do not infer timing/cache semantics from the toolchain failure.
