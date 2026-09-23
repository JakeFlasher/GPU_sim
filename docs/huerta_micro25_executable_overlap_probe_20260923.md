# Huerta MICRO'25 executable overlap probe

Updated 2026-09-23. This is an external-baseline feasibility record, not a
numeric accuracy ranking.

## Build

The retained source is the Huerta *Dissecting and Modeling the Architecture of
Modern GPU Cores* tree:

| Component | Value |
| --- | --- |
| Source | `/home/jakeshea/GPU_sims_notes/upstream_research_20260901/modern-gpu-simulator-micro-2025` |
| Source revision | `117f9dc` |
| Architecture configuration | `SM86_RTX3090` |
| Host compiler | GCC 14.3.1 |
| CUDA toolkit used for compilation | CUDA 12.8.93 |
| Protobuf runtime used for the final link | host protobuf 36.1 + system Abseil |

The upstream tree has two host-toolchain compatibility defects on this host:
an old RapidJSON `GenericStringRef::operator=` assigns `const` members, and
`ldst_unit_sm.h` omits `<functional>`. A temporary, compile-only source shim
was used for those defects and reverted; the external Git worktree is clean.
The final executable is the external binary at
`simulator-remodeled/gpu-simulator/bin/release/accel-sim.out`.

## Same enhanced trace

The existing fresh locked-clock SM86 capture was supplied directly:

    results/milestones_20260923/live/gemm_hmma_bf16_sm80/trace/dynamic_trace.pb

It contains the same 512-block, 32-thread/block GEMM launch used by the local
Rust replay and the NCU row. Huerta's RTX 3090 configuration consumed the
enhanced protobuf/threadblock sidecars, not a regenerated workload or a
different application trace. The replay command was:

    accel-sim.out \
      -config .../gpgpu-sim/configs/tested-cfgs/SM86_RTX3090/gpgpusim.config \
      -config .../configs/tested-cfgs/SM86_RTX3090/trace.config \
      -trace .../trace/dynamic_trace.pb

Two successful bounded runs reported identical model counters:

| Run | Wall time | Sim cycles | Sim instructions | Issued CTAs | IPC | Exit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.680 s | 16,708 | 5,242,880 | 512 | 313.7946 | 0 |
| 2 | 8.233 s | 16,708 | 5,242,880 | 512 | 313.7946 | 0 |

The retained external artifacts are under
`/home/jakeshea/GPU_sims_notes/huerta_overlap_20260923/compat3/run-{1,2}/`.
The local Rust replay of the same enhanced trace is stable at approximately
0.916 s process wall time across three runs, with a 12,949-cycle GPC
observation frontier and 11,101-cycle SM-active terminal frontier.

These wall times are descriptive engineering measurements only. The models,
configuration details, and output frontiers are not identical enough to claim
that one simulator is faster or more accurate than the other.

## Why this remains non-scoring

1. Huerta's card is RTX 3090 (82 configured clusters), while the independent
   NCU ground truth and this project's card are RTX 3090 Ti (84 SMs). Same
   compute capability is not identical device identity.
2. Huerta reports its own `gpu_tot_sim_cycle` model counter. The NCU target is
   `gpc__cycles_elapsed.max`; these are not a declared common observable or
   clock-domain mapping.
3. The upstream protobuf map implementation is fragile with the host's modern
   runtime: repeated invocations can fail to materialize the GPU/stream map
   (`map.h:1429`, missing key 0 or 7). Two successful runs are retained, but
   this instability prevents a reproducible M6 score until the baseline is
   built in its supported environment or the source/runtime pair is fixed by
   its maintainers.

Therefore no Huerta error percentage, speed ratio, or SOTA ranking is
reported. The valid M6 status is `executable_overlap_non_scoring`; the fair
numeric cell remains unsupported.
