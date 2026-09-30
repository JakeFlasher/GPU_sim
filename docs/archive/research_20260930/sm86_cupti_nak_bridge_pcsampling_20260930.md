# CUPTI decoded PC sampling on NAK CUDA ELF bridge (2026-09-30)

The standalone NAK runtime was invisible to the public CUPTI injection path,
but the bounded NAK-to-CUDA-ELF bridge is a normal CUDA Driver launch. The
NVIDIA `pc_sampling_continuous` sample was attached to that bridge executable
with serialized PC sampling (`sampling-period=5`).

Observed output:

* **8 decoded PC records**;
* **34 total samples**;
* **0 dropped samples**;
* PC offsets `0, 16, 48, 80, 128, 208, 384, 448`;
* decoded aggregate reasons included long-scoreboard, instruction-cache miss,
  short-scoreboard, wait, branch-resolving, selected, and no-instruction.

The result is a new public decoded aggregate stall stream for an independently
generated NAK body inside NVIDIA ELF metadata. It still has no physical
execution-pipe, sub-unit, issue-slot, or exact issue-cycle field. The NAK body
and metadata bridge remain bounded; this is not unrestricted CUDA ABI closure.

Artifacts:

* `results/alpha/sm86_nak_bridge_pcsampling_goal_20260930/decoded.txt`
* `results/alpha/sm86_nak_bridge_pcsampling_goal_20260930/summary.json`
* `tools/sm86_cupti_pc_sampling_summary.py`

No tests, hashing, or verification workflow was run.
