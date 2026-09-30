# Repeated bridged NAK CUPTI PC sampling (2026-09-30)

The bounded NAK-to-CUDA-ELF bridge was launched 256 times through the CUDA
Driver API under NVIDIA's public `pc_sampling_continuous` CUPTI sample. The
serialized output decoded:

* **256 launch buffers**;
* **1,751 PC records**;
* **4,927 total samples**;
* **0 dropped samples**;
* **29 distinct PC offsets**.

Each serialized launch produced 18–35 total samples (median 19) with no
buffer drops, so the result also exposes launch-to-launch sampling variability.

Aggregate decoded stall buckets include long-scoreboard (2,088 issued/not-issued
pairs), instruction-cache miss (1,718 pairs), short-scoreboard (168 pairs),
wait (296 pairs), branch-resolving (59 pairs), drain (135 pairs), misc (89
pairs), and selected/no-instruction observations.

This is substantially stronger decoded aggregate evidence than the one-launch
bridge (8 PCs / 34 samples), and it demonstrates serialized launch correlation
through the public CUPTI path. It still does not expose physical execution
pipe, sub-unit, issue-slot, or exact issue-cycle identity; the ELF/ABI bridge
remains template-bounded.

Artifacts:

* `tools/sm86_nak_bridge_repeat_launch.cc`
* `tools/sm86_cupti_pc_sampling_summary.py`
* `results/alpha/sm86_nak_bridge_repeat_pcsampling_goal_20260930/`

No tests, hashing, or verification workflow was run.
