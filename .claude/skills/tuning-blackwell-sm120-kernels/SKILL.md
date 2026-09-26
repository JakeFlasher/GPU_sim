---
name: tuning-blackwell-sm120-kernels
description: Handle an explicitly Blackwell-targeted CUDA capability or tuning question using exact target documentation. Not the default for this SM86 project.
---

# Blackwell-targeted work

Only select a Blackwell target when the task requires it. The project's
current validation anchor is SM86; no sibling-generation fallback is allowed.
Use `references/NVIDIA_Manuals/blackwell_tuning/`, `blackwell_compatibility/`
and `ptx_isa/`, and verify version-sensitive claims with current NVIDIA sources.
Do not assume SM100 and SM120 expose identical instructions or cluster limits.

The inherited [historical notes](references/historical-notes.md) are opt-in
context, not verified device capabilities or a default cloud target. The
optional `.claude/knowledge/` card tree is absent here. Consult `docs/STATUS.md`
before interpreting any local simulator output as Blackwell validation.
