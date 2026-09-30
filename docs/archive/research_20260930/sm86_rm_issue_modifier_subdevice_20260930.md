# SM86 RM issue-modifier query — corrected subdevice target

**Date:** 2026-09-30  
**Hardware:** RTX 3090 Ti / SM86, driver 615.71.09.

## Correction

The earlier probe targeted `NV01_DEVICE_0`; these GR controls are subdevice
controls and must target `NV20_SUBDEVICE_0`. The device-targeted
`NV_ERR_NOT_SUPPORTED` result is retained as a handle-targeting negative, not as
a statement that the public controls are universally disabled.

The corrected read-only probe uses the `Device.subdevice` handle and enumeration
size zero:

* legacy `NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER` returns:
  `imla0=0, fmla16=0, dp=0, fmla32=1, ffma=0, imla1=0, imla2=0,
  imla3=0, imla4=1`;
* v2 `NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER_V2` returns nine entries:
  indices `0..8` with values `0,0,1,0,0,0,0,0,1`;
* public throttle control returns `NV_ERR_NOT_SUPPORTED`;
* internal/static KGR v2 rate and throttle controls return
  `NV_ERR_NOT_SUPPORTED`.

Artifacts:
`results/alpha/sm86_rm_issue_modifier_probe_20260930/rm_subdevice_output.json`
and `results/alpha/sm86_causal_corrections_20260930/rm_subdevice_output.json`.

## Interpretation

The corrected result supplies **static instruction-class speed-select priors**
for the card/RM. It does not provide dynamic per-instruction route, pipe,
subunit, issue ordinal, or physical SMSP identity. The static values are safe
model inputs only when kept separate from dynamic route observations.
