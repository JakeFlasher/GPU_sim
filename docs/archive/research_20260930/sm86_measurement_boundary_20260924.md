# SM86 same-launch measurement boundary — 2026-09-24

## Closure status

The first same-launch NCU/NVBit/simulator boundary is now closed for the
declared instruction-count observable:

`smsp__inst_executed.sum` ↔ simulator `counters.instructions_issued`

The three NCU repeats all measured `197,120`; the matching simulator replay
also emitted `197,120` for the same grid `(16,32,1)`, block `(32,1,1)`, and
single target launch. This is a count mapping, not a claim that the simulator
already predicts elapsed cycles.

The timing crosswalk remains explicitly separate: NCU
`gpc__cycles_elapsed.max` measured `13,475–13,529` cycles, while raw simulator
`BindingCycles` was `11,102`, the modeled GPC observation frontier was `12,949`,
and dirty-writeback/coherence tails remained incomplete. No cycle-accuracy
claim is promoted from this probe.

## Host and tools

| field | observed |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| UUID | `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c` |
| compute capability | 8.6 |
| driver | 615.71.09 |
| clocks | SM/graphics 1350 MHz; memory 10501 MHz |
| NCU | 2026.1.0 (`/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu`) |
| simulator | release `sim-run`, SM86 card, `l1-l2`, launch cache scope |

The live profile was updated to this observed driver identity in
`workloads/research_probes/sm86_ncu_demo_benchmark/host_profile.json`.

## Workload and commands

Workload: `gemm_hmma_bf16_sm80`, timing shape, warmup 1, reps 1. The retained
raw campaign evidence is under:

`results/captures/sm86_measurement_boundary_20260924_v1/`

The campaign ran correctness, NCU, NVBit, static-control, and simulator stages
with the repository virtualenv. The first attempted run with system Python was
rejected during post-replay protobuf validation; it was not used as evidence.

NCU collection command shape:

```text
.venv/bin/python tools/local_labels/collect_ncu.py \
  --ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu \
  --launch-skip 1 --launch-count 1 --replay-mode application \
  --clock-control none --cache-control all \
  --metrics gpc__cycles_elapsed.max,smsp__inst_executed.sum,... \
  -- workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run \
  --timing_shape --warmup 1 --reps 1
```

NVBit capture used the same timing shape with warmup 0 and reps 1. Simulator
replay consumed that trace and its generated SM86 control sidecar; NCU output
was joined only after replay and never passed to simulator configuration.

## Repeated NCU rows

| repeat | GPC cycles | SM active cycles | DRAM read sectors | instructions |
|---:|---:|---:|---:|---:|
| 1 | 13,513 | 11,706 | 8,364 | 197,120 |
| 2 | 13,529 | 11,722 | 8,416 | 197,120 |
| 3 | 13,475 | 11,654 | 8,448 | 197,120 |

Simulator row: instructions `197,120`; global-load sectors `196,608`; global
store sectors `16,384`; DRAM read sectors `8,262`; replay wall time `0.769 s`.

The count mapping is therefore closed and reproducible. The cycle row is a
development crosswalk only until retirement/coherence tails and a stable timing
frontier are modeled.
