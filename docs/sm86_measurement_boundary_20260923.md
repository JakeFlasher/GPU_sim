# SM86 measurement boundary — first M0/M1 probe

Updated 2026-09-23.

## Workload and inputs

* Workload: workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run
* Shape: timing_shape, warmup 0, reps 1
* Kernel: gemm_hmma_bf16_sm80_kernel
* Grid/block: (16, 32, 1)/(32, 1, 1)
* NVBit trace: results/captures/sm86_ncu_accuracy_recovery_20260901_final_v1/evidence/raw/nvbit/gemm_hmma_bf16_sm80
* Control sidecar: results/captures/sm86_ncu_accuracy_recovery_20260901_final_v1/evidence/raw/static/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80.control_bits.json
* Simulator command: the retained command in the simulator evidence directory,
  using sm_86.json, the SM86 pchase card, l1-l2 mode, launch cache scope, and
  NCU-shaped metrics.

The trace has one launch and full coverage: 512 observed/population blocks and
197,120 records/instructions.

## Independent NCU setup

The retained NCU command used:

    ncu --launch-skip 1 --launch-count 1 --replay-mode application \
      --cache-control all --clock-control none \
      --metrics gpc__cycles_elapsed.max,sm__cycles_active.max,\
gpu__time_duration.sum,smsp__inst_executed.sum,...

The host record identifies an NVIDIA GeForce RTX 3090 Ti, compute capability
8.6, driver 610.57.04, UUID
GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c, with 1350 MHz graphics/SM and
10501 MHz memory clocks. The campaign retained seven repeats.

## Observed values

| Observable | Median | Range over seven repeats |
| --- | ---: | ---: |
| gpc__cycles_elapsed.max | 13,440 cycles | 13,350–13,796 |
| sm__cycles_active.max | 11,658 cycles | 11,588–12,027 |
| gpu__time_duration.sum | 10,016 ns | 9,952–10,272 |
| smsp__inst_executed.sum | 197,120 | constant |

The simulator replay is deterministic across the seven retained replays:

| Frontier | Value |
| --- | ---: |
| diagnostic_last_issue_cycles | 8,915 |
| terminal_modeled_event_time | 8,915 |
| modeled completion lower bound | 8,915 |
| coverage | full |
| unmodeled tails | dirty writeback and coherence |

## Interpretation

The instruction-count observable is an exact trace-accounting match, not an
independent prediction. The 8,915-cycle model frontier is approximately 33.7%
below the median GPC elapsed-cycle label and 23.5% below the median SM-active
label. Those differences cannot yet be called simulator error because the
clock-domain, launch envelope, and unmodeled retirement-tail mapping are not
closed.

The first measurement-boundary hypothesis is therefore **blocked, not scored**:

* NCU setup and paired input identity are available.
* NCU repeats are stable enough to define an observation range.
* The current BindingCycles/frontier output is not semantically equivalent to
  either NCU cycle observable.
* Dirty-writeback/coherence tails are explicit reasons the modeled completion
  is a lower bound.

## Next discriminating probe

Keep this workload as the control row and choose one of two next probes:

1. add a simulator-side launch/tail projection only if its causal meaning can be
   stated from existing modeled state; or
2. choose a workload without dirty-writeback/coherence tails and test whether the
   remaining gap tracks the same launch/clock envelope.

Until one probe establishes a stable mapping, report model-relative timing and
NCU labels separately. Do not tune a constant to force 8,915 toward 13,440.



## Current-source replay follow-up

The current release binary was rebuilt from the active repository and replayed
all eight retained small-cohort traces. Each row completed with full block
coverage. The architecture-scoped gpc observation frontier is materially closer
to the NCU GPC label than the raw BindingCycles value:

| Workload | BindingCycles | GPC observation frontier | NCU GPC median | Frontier error | Replay wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| embedding_gather_int32_sm80 | 11,939 | 13,738 | 13,025 | +5.47% | 0.064 s |
| flash_attn_fwd_bf16_sm80 | 15,232 | 17,039 | 15,947 | +6.85% | 0.089 s |
| gemm_hmma_bf16_sm80 | 11,102 | 12,949 | 13,440 | −3.65% | 0.795 s |
| gemm_imma_int8_sm80 | 9,071 | 10,918 | 11,582 | −5.73% | 0.593 s |
| gemm_sparse_2to4_bf16_sm80 | 4,110 | 5,941 | 5,764 | +3.07% | 0.083 s |
| layernorm_fwd_bf16_sm80 | 20,487 | 22,310 | 21,769 | +2.49% | 0.676 s |
| rmsnorm_fwd_bf16_sm80 | 17,121 | 18,944 | 19,147 | −1.06% | 0.364 s |
| silu_fwd_bf16_sm80 | 3,787 | 5,642 | 5,741 | −1.72% | 0.448 s |

This is a promising **candidate** observation projection, not an accepted
accuracy result. All eight rows still expose dirty-writeback and coherence
tails, and the projection is explicitly labeled architecture-scoped rather
than a validated hardware crosswalk. The next check is to determine whether
the same projection remains stable on a second retained cohort/control set
without fitting a workload-specific offset. Until then, report both frontiers
and do not replace BindingCycles with the candidate value.



## Current host refresh attempt

A fresh NCU run was attempted on 2026-09-23 with NCU 2026.1 and driver
615.71.09. The host was idle at 210 MHz SM / 405 MHz memory, unlike the
retained campaign's 1350/10501 MHz contract. The command used
\`--clock-control base\`, which produced a 446.884 ms workload-reported mean for
the timing-shape GEMM rather than the retained ~10 ms profile scale. This run
is not joined to the retained crosswalk and is not used as a simulator label.

A final hardware comparison therefore remains pending a declared clock policy:
either temporarily lock/restore the retained operating point, or accept the
current NCU-controlled clock behavior and establish a new paired calibration.



## Fresh locked-clock paired row

After externally locking persistence, SM, and memory clocks, a fresh NCU run used:

    .venv/bin/python tools/local_labels/collect_ncu.py \
      --output results/milestones_20260923/ncu_locked_gemm \
      --launch-skip 1 --launch-count 1 --replay-mode application \
      --clock-control none --cache-control all \
      --metrics gpc__cycles_elapsed.max,sm__cycles_active.max,\
gpu__time_duration.sum,smsp__inst_executed.sum,... \
      -- workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run \
      --timing_shape --warmup 1 --reps 1

The fresh row observed: GPC cycles 13,438, SM-active cycles 11,665,
GPU duration 9,984 ns, and 197,120 executed instructions. The simulator
replayed the retained matching trace with the observation-only NCU CSV attached
and reported: BindingCycles 11,102, GPC observation frontier 12,949,
SM-active terminal frontier 11,101, full coverage, and replay wall time about
0.89 s.

The candidate frontier errors for this fresh row are −3.64% against GPC cycles
and −4.84% against SM-active cycles. This improves the measurement boundary
from host-clock-blocked to a live candidate crosswalk, but the projection still
has the declared dirty-writeback/coherence tails and is not promoted to a
hardware-equivalent metric.



## Retained timing-crosswalk holdout check

The current release binary was also replayed against the three retained timing
holdout traces. Existing three-repeat NCU labels were compared to the current
architecture-scoped frontiers:

| Workload | GPC frontier | NCU GPC median | Frontier error |
| --- | ---: | ---: | ---: |
| FP16 HMMA GEMM M224 | 11,570 | 12,318 | −6.07% |
| INT8 IMMA GEMM M192 | 8,982 | 8,181 | +9.79% |
| BF16 FlashAttention BH2 SQ96 SK96 | 13,959 | 12,754 | +9.45% |

All three rows had full block coverage and retained dirty-writeback/coherence
tail flags. This independent retained cohort supports the candidate projection
as a cross-workload observation, but not yet as a hardware-equivalent timing
metric: the NCU rows are historical captures, and the current host's driver
version differs from the capture metadata.



## Fresh live trace/NCU set

A fresh NVBit capture and three NCU application-replay repeats were then
collected under the locked 1350/10501 MHz clocks for three workloads. The
current binary replay used each fresh trace and its generated control sidecar:

| Workload | GPC frontier | NCU GPC median | GPC error | SM frontier | NCU SM median | SM error | Replay wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 HMMA GEMM | 12,949 | 13,440 | −3.65% | 11,101 | 11,730 | −5.36% | 0.903 s |
| BF16 RMSNorm | 18,944 | 18,883 | +0.32% | 17,120 | 17,152 | −0.19% | 0.404 s |
| BF16 SiLU | 5,576 | 5,685 | −1.92% | 3,720 | 3,914 | −4.96% | 0.459 s |

All three fresh rows have full block coverage and retain dirty-writeback/
coherence tail flags. The crosswalk is now demonstrated on fresh paired
inputs, not only historical captures, but it remains an architecture-scoped
candidate projection. Raw BindingCycles remains separate and substantially
lower for these rows.
