# SM86 non-greedy L0C candidate scan — 2026-10-15

## Mechanism

H23 completes the public Huerta SM86 issue-loop behavior after H22's bounded
four-cycle greedy L0C window. When the held warp's ordinary frontiers are ready
and its miss-switch countdown is exhausted, the scheduler probes that greedy
warp first, then scans remaining ordinary-ready siblings in deterministic
youngest order. A non-greedy IMC miss is staged in the cloned per-SMSP cache and
the scan continues; a hit becomes the effective issue pick. Memory-admission
feedback is checked against the final effective candidate, while skipped miss
probes do not consume an instruction admission.

Huerta's released `are_l1c_operands_ready()` allocates an L0C miss request and
leaves it in the MSHR/miss queue, so the Rust scan preserves all staged line
reservations and parks every probed miss atomically at the cycle boundary. This
is an effective model of the published behavior, not a proprietary cache claim
or an NCU fit.

References are the same public MICRO'25 source and SM86 configuration used for
H22: [Huerta §5.1.1](https://arxiv.org/html/2503.20481v1#S5.SS1.SS1),
[released issue loop](https://github.com/upc-arco/modern-gpu-simulator-micro-2025/blob/main/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L433-L465),
and [SM86 L0C configuration](https://github.com/upc-arco/modern-gpu-simulator-micro-2025/blob/main/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3090/gpgpusim.config#L1346-L1369).

## Implementation and safeguards

- Added a bounded per-SMSP candidate scan after a non-greedy IMC miss.
- Added an explicit expired-held greedy probe so the held warp is always
  examined before the youngest sibling, including when the countdown is zero.
- Staged cache fills, `imc_wait_until` updates, diagnostics, and countdown state
  transactionally; only the final effective issue pick participates in memory
  admission and instruction accounting.
- Added `non_greedy_scan_attempts`, `non_greedy_scan_misses`, and
  `non_greedy_scan_successes`. Rows with scan activity use readiness schema 28;
  rows with only the H22 surface remain schema 27.

## Real-trace evidence

The post-ordering representative batch completed 10/10 discovered real roots
(2,728 full launches), including the high-IMC adapted-L2, NVFP4, snapshot-SAM,
and GeGLU holdouts. Artifact:
`../sm86_replay_results/h23_l0c_scan_20261015_v2/full.jsonl.zst`.

## Complete-corpus evidence

The authoritative full-first replay completed before the complementary fast
stage with exactly twelve workers:

- 356 roots; 286 `ok`, 62 mixed-code controls, 8 replay/input failures.
- 12,392 full launches; full wall time 40.715 s.
- Artifact:
  `../sm86_replay_results/full_corpus_h23_l0c_scan_20261015_v2/full.jsonl.zst`.
- Post-replay report:
  `../sm86_replay_results/metric_report_h23_l0c_scan_20261015_v2/`.
- NCU was post-replay only; all generated JSONL is compressed `.jsonl.zst`.

Across the corpus the scan emitted 21,574,287 probe attempts, 21,348,688
probe misses, and 225,599 probe successes. IMC miss-wait events rose to
111,823,998 because the public cache access probes all ordinary-ready candidates
and pending merges overlap heavily. Tag misses changed only 1,174,632→1,174,635.

## Exact H22 comparison

| Quantity | H22 | H23 | Delta |
|---|---:|---:|---:|
| L1 hits | 6,240,192 | 6,240,088 | −104 |
| L1 misses | 8,795,747 | 8,795,851 | +104 |
| L1 merges | 6,800,604 | 6,800,838 | +234 |
| L2 hits | 2,793,478 | 2,793,478 | 0 |
| L2 misses | 7,320,523 | 7,320,523 | 0 |
| DRAM read sectors | 4,824,390 | 4,824,390 | 0 |
| GPC elapsed frontier sum | 86,121,963 | 86,122,355 | +392 |
| terminal frontier sum | 64,014,244 | 64,014,636 | +392 |

Only 74 of 12,392 common launches change GPC/terminal frontiers (−147 to
+505 cycles). The post-replay NCU reduction remains bit-for-bit neutral at its
reported aggregate precision: L1 lookup hit/miss 30.71%/58.44%, L2 28.24%,
DRAM 40.91%, SM active 18.01%, elapsed IPC 12.31%, and GPC 12.91%.

## Residual and next seam

The scan is now source-aligned and defaulted in full replay, but its large
overlap diagnostics and small positive aggregate frontier delta do not close
IPC. The next mechanism should target the remaining causal residual (typed
terminal/writeback/coherence tails or scheduler issue-resource contention), not
turn probe counts into a fitted delay. The static 46-cycle ERRBAR/post-EXIT
special remains deferred because no dynamic `ERRBAR` executes in the corpus.
