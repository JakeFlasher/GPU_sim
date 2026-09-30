# SM86 causal/evidence corrections — handle targets, control fields, and timer identity

**Date:** 2026-09-30

This correction pass audits recent probes for field-layout and object-handle
causality without deleting the original artifacts.

## RM issue controls

The first issue-control probe targeted `NV01_DEVICE_0`; GR controls require
`NV20_SUBDEVICE_0`. Correct subdevice reads return static rate modifiers:
legacy `fmla32=1`, `imla4=1`; v2 indices 0..8 are
`0,0,1,0,0,0,0,0,1`. Public throttle and internal/static KGR variants remain
unsupported. The corrected result is
`sm86_rm_issue_modifier_subdevice_20260930.md`.

## SM86 control-code layout

SM70+/SM86 control fields are delay 0..3, yield 4, write barrier 5..7, read
barrier 8..10, wait mask 11..16, and reuse mask 17..20. Earlier mutations were
misnamed using an SM120-style layout:

* `0x207f1 -> 0x7f1`: reuse-mask bit 2;
* `0x27ed -> 0x7ed`: composite range crossing read-barrier/wait-mask bits;
* `0xfed -> 0xfe1`: delay nibble 13 -> 1.

The observed functional/timing outputs remain retained, but isolated
stall/rbar/misc claims are withdrawn. The temporary rule-driven synthesizer used
the wrong packing and is not promoted.

## Timer image identity

The long-body raw-QMD header and Driver cubin have equal text length and equal
prefix through PC `0x0f0`, but differ in control words at PCs `0x100` and
`0x110`. The short-body retained raw header likewise is not byte-identical to
the Driver text. The timer crosswalk is therefore semantic-body/operation-level,
not exact SASS-image identity. The in-body timing results remain observations,
but their causal claim is narrowed accordingly.

Artifacts from the offline correction analysis are under
`results/alpha/sm86_causal_corrections_20260930/`.
