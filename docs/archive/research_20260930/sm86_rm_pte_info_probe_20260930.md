# SM86 RM PTE cache/aperture-information probe

**Date:** 2026-09-30  
**Hardware/driver:** RTX 3090 Ti / SM86, NVIDIA driver 615.71.09.

## Probe

The public `NV0080_CTRL_CMD_DMA_GET_PTE_INFO` interface documents PTE fields
including aperture, GPU-cached status, shader access, kind, validity, and
atomic/readonly attributes. A fresh RM-backed probe queried three allocations:

* local device memory, expected cached;
* mapped host memory, expected system aperture;
* local system/uncached allocation.

All three control calls returned
`NV_ERR_TEST_ONLY_CODE_NOT_ENABLED` (status 126) on the production RM. No PTE
fields were returned; zero-filled structures are not promoted as observations.

Artifacts:
`results/alpha/sm86_rm_pte_info_probe_20260930/`.

## Interpretation

This is a concrete finite negative for a plausible cache-ownership seam: the
public header advertises the fields, but the installed production RM disables
the control path. It does not alter the existing cache model and does not
provide line owner/read-from/DRAM completion evidence.
