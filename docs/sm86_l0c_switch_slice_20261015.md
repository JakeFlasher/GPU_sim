# SM86 L0C miss-switch scheduling — 2026-10-15

## Mechanism

H22 activates the public Ampere/SM86 immediate-constant-cache scheduling
window that was already carded as `switch_after_miss_cycles = 4` but was not
previously consumed by the Rust scheduler. Huerta's released remodeled source
checks the constant-cache operand at issue, retains the greedy warp while its
ordinary control/dependency/resource conditions are ready, decrements a
four-cycle miss countdown, and only then permits the youngest ready sibling.
The released `SM86_RTX3090` configuration uses the same four-cycle value.

References:

- [Huerta et al., MICRO'25, §5.1.1](https://arxiv.org/html/2503.20481v1#S5.SS1.SS1)
- [released `subcore.cc` countdown](https://github.com/upc-arco/modern-gpu-simulator-micro-2025/blob/main/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L433-L465)
- [released SM86 RTX3090 configuration](https://github.com/upc-arco/modern-gpu-simulator-micro-2025/blob/main/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3090/gpgpusim.config#L1346-L1369)

This is a bounded effective scheduler model, not a proprietary L0C topology or
an NCU-fitted delay. It is restricted to SM86 full replay and is kept separate
from `imc_wait_until`, the line-response frontier. Ordinary readiness is
computed with the IMC frontier removed before the bounded greedy hold is
considered; memory-admission feedback can still reject the candidate.

## Implementation

- Added a per-SMSP `immediate_constant_switch_remaining` countdown.
- Added ordinary-readiness-without-IMC-frontier decomposition to avoid
  forcing a warp blocked by unrelated dependencies, front-end work, or a
  pipeline/resource condition.
- Added the countdown-aware CGGTY pick path to both normal and memory-admission
  scheduling.
- Reloaded the four-cycle window after every effective issue and consumed one
  opportunity for each parked greedy L0C miss.
- Added `greedy_switch_block_cycles` to IMC diagnostics. The readiness schema is
  now 27 when the IMC surface is present; all counters remain observational.

## Real-trace evidence

Representative full-first replay completed all four discovered roots (485 full
launches): `simbench_l1_register_spill_local_mem`,
`simbench_l2_geglu_ffn_fused_bf16`,
`sm86_adapted_l2_072_region_attention_backward_sm120`, and
`sm86_adapted_l2_060_chunk_delta_rule`.

Artifact:
`../sm86_replay_results/h22_l0c_switch_20261015_v2/full.jsonl.zst`.

The high-pressure roots exercise the arm: register-spill and GeGLU show
additional bounded miss-wait opportunities, while the region-attention and
chunk-delta traces expose multi-warp overlap. No synthetic fixture was used.

## Complete-corpus evidence

The authoritative run was full-first, followed by the complementary fast mode,
with exactly 12 workers and compressed artifacts:

- 356 roots; 286 `ok`, 62 mixed-code controls, 8 replay/input failures.
- 12,392 full launches.
- Full wall time: 37.195 s.
- Artifact:
  `../sm86_replay_results/full_corpus_h22_l0c_switch_20261015_v2/full.jsonl.zst`.
- Post-replay report:
  `../sm86_replay_results/metric_report_h22_l0c_switch_20261015_v2/`.
- NCU was not used as replay input; all JSONL artifacts are `.jsonl.zst`.

Across full launches, IMC tag misses remain exactly 1,174,632 and all L1/L2/
DRAM counters remain bit-for-bit unchanged from H20. The new scheduler surface
reports 3,125,368 miss-wait events and 1,489,412 greedy-switch-block cycles.
These are overlapping issue-opportunity diagnostics, not elapsed-cycle claims.

## Accuracy delta versus H20

The post-replay NCU reduction is neutral at reported precision:

| Metric | H20 mean APE | H22 mean APE | Delta |
|---|---:|---:|---:|
| L1 global-load lookup hits | 30.71% | 30.71% | 0.00 pp |
| L1 global-load lookup misses | 58.44% | 58.44% | 0.00 pp |
| L2 TEX source sectors | 28.24% | 28.24% | 0.00 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| SM active cycles | 18.01% | 18.01% | 0.00 pp |
| elapsed IPC | 12.31% | 12.31% | 0.00 pp |
| GPC elapsed cycles | 12.91% | 12.91% | 0.00 pp |

Comparing common full launches, only six terminal/GPC frontier rows move, by
one or two cycles (aggregate terminal/GPC delta −6 cycles). The arm is
therefore source-backed and live but not a corpus-level IPC closure mechanism.

## Residual and next seam

After the countdown expires, the current Rust pick path still probes only the
single youngest candidate. If that non-greedy candidate itself has an IMC miss,
the scheduler parks it rather than continuing through the remaining ready
priority list as the released C++ loop does. This is a bounded scheduler
residual, not a reason to widen the four-cycle arm. The next slice should add a
transactional non-greedy IMC candidate scan (with explicit bounded diagnostics)
and evaluate whether it changes critical frontiers or only overlapping waits.

The public 46-cycle `STALL=0 + YIELD=1` post-EXIT/ERRBAR special remains
deferred: the 356-root dynamic corpus has no executed `ERRBAR` records, so a
global promotion would be unsupported.
