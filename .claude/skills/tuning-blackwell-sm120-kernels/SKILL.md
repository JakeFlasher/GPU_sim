---
name: tuning-blackwell-sm120-kernels
description: Handle an explicitly Blackwell-targeted CUDA capability or tuning question using exact target documentation. The target is the SM120/RTX 5090 deployment lane; do not use it to reinterpret the SM86 lane.
---

# Blackwell-targeted work

Only select a Blackwell target when the task requires it. The active Blackwell target is SM120 on the RTX 5090; the SM86 lane remains a separate baseline and no cross-generation fallback is allowed.
Use the tracked `references/NVIDIA_Manuals/blackwell_tuning/`,
`blackwell_compatibility/`, and `ptx_isa/` material, and verify version-sensitive
claims with current NVIDIA sources.
Do not assume SM100 and SM120 expose identical instructions or cluster limits.

The optional `.claude/knowledge/` card tree is absent here. Consult
`docs/STATUS.md` before interpreting any local simulator output as a Blackwell
capability or timing claim.
