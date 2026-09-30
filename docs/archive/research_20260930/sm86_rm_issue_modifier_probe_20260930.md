# SM86 RM issue-modifier probes — handle-target history and correction

**Date:** 2026-09-30  
**Hardware/driver:** RTX 3090 Ti / SM86, NVIDIA driver 615.71.09.

## Initial probe (superseded handle target)

The first probe sent GR issue controls to `NV01_DEVICE_0`. Those controls are
owned by `NV20_SUBDEVICE_0`; the resulting `NV_ERR_NOT_SUPPORTED` is retained
as a handle-targeting negative only.

## Correct subdevice probe

The corrected read-only query targets `NV20_SUBDEVICE_0` and uses enumeration
size zero where supported:

* legacy issue-rate modifier: `imla0=0, fmla16=0, dp=0, fmla32=1, ffma=0,
  imla1=0, imla2=0, imla3=0, imla4=1`;
* v2 issue-rate modifier: indices `0..8` with data
  `0,0,1,0,0,0,0,0,1`;
* public issue-throttle control: `NV_ERR_NOT_SUPPORTED`;
* internal/static KGR v2 rate and throttle controls:
  `NV_ERR_NOT_SUPPORTED`.

Artifacts:
`results/alpha/sm86_rm_issue_modifier_probe_20260930/rm_subdevice_output.json`
and `results/alpha/sm86_causal_corrections_20260930/rm_subdevice_output.json`.

## Boundary

The static values are instruction-class speed-select priors. They do not expose
dynamic per-instruction pipe, subunit, scheduler, issue ordinal, or physical
SMSP identity. The corrected result is summarized in
`sm86_rm_issue_modifier_subdevice_20260930.md`.
