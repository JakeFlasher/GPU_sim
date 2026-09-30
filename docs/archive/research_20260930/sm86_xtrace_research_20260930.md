# Xtrace/G-Watch SM86 research lead

The September 2026 Xtrace paper describes binary-level SASS splicing for 19
NVIDIA/AMD architectures, including SM86, with globaltimer or per-SM clock
records, per-warp SMID anchors, and dead-register/hazard-aware insertion
([arXiv:2609.28769](https://arxiv.org/abs/2609.28769)). The public G-Watch
wheel `0.0.35` was installed in a local uv environment and inspected.

Its SASS schema records site ID, optional register value, timestamp, and (for
warp traces) SMID/clock anchors. It does **not** record a physical pipe,
sub-unit, issue-slot, or dynamic issue ordinal. Thus it can improve PC/time
fidelity but cannot by itself close the exact-pipe blocker.

A bounded SM86 launcher/cubin probe was attempted against the local RTX 3090 Ti.
G-Watch parsed the SM86 cubin and produced a spliced image, but the instrumented
launch returned CUDA error 715 (`CUDA_ERROR_ILLEGAL_INSTRUCTION`) during
synchronization. No trace records were accepted as hardware evidence. Artifacts are under
`results/alpha/sm86_xtrace_probe_20260930/`.

This is a useful external lead for future matched PC/timestamp work, not a
resolved blocker or a repository test/verification workflow.
