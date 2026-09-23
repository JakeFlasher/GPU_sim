# Accel-Sim executable overlap probe

Updated 2026-09-23. This note records a runnable external-baseline probe; it
does **not** establish a cross-simulator accuracy ranking or a SOTA claim.

## What closed

The locally retained Accel-Sim 2.0 source was built without changing its
simulator model sources. The build used a user-local CUDA 12.8 redistributable
toolchain and GCC 14:

| Component | Revision / version |
| --- | --- |
| Accel-Sim framework | `d930ad6d02c09bb56867132583735aba0389cff4` |
| Accel-Sim GPGPU-Sim dependency | `91880c53383d5a6a6742bfb1be2c5f34e39c7871` |
| CUDA toolkit used for build | 12.8.93 (`nvcc`) |
| Host compiler | GCC 14.3.1 (`g++-14`) |
| NVBit | 1.8 |
| Host GPU for the traced launch | RTX 3090 Ti, SM86 |

The executable was produced at the external research path:

    /home/jakeshea/GPU_sims_notes/upstream_research_20260901/
      accel-sim-framework/gpu-simulator/build-local-cuda128/accel-sim.out

The CUDA 12.8 toolchain is outside this repository and did not alter the
host's `/usr/local/cuda-13.2` installation.

## Same-workload overlap run

The portable BF16 HMMA GEMM was run at `M=256,N=256,K=256`, `warmup=0`,
`reps=1`, with its native Accel-Sim NVBit tracer. The legacy trace was then
post-processed to `.traceg.xz` and replayed with the native
`SM86_RTX3070/gpgpusim.config` card. The capture reported a 16x32 grid (512
blocks, 32 threads/block) and 213,504 reported trace instructions.

The three bounded replay runs all completed successfully:

| Run | Process wall time | Peak child RSS | Accel-Sim cycles | Accel-Sim instructions | Exit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.943 s | 489,980 KiB | 14,174 | 5,621,216 | 0 |
| 2 | 4.869 s | 489,980 KiB | 14,174 | 5,621,216 | 0 |
| 3 | 4.870 s | 489,980 KiB | 14,174 | 5,621,216 | 0 |

Artifacts are retained outside the repository at
`/home/jakeshea/GPU_sims_notes/accel_overlap_20260923/`:

* `gemm_timing_trace2/traces/traces/` — native legacy trace and post-processed
  `kernelslist.g`;
* `gemm_timing_sim/run-{1,2,3}/` — simulator stdout, counters, and measured
  process metrics.

## Why this is not a valid numeric baseline cell yet

The run proves that an external simulator can be built and exercised on the
same application shape, but two independent mismatches remain:

1. The upstream card is explicitly `SM86_RTX3070`, whereas the ground-truth
   machine and this project card are RTX 3090 Ti. The 3070 result therefore
   cannot be compared to the 3090 Ti NCU row as hardware accuracy.
2. Accel-Sim consumed its native legacy `.traceg.xz` representation (213,504
   reported instructions), while this project consumed the enhanced protobuf
   capture for the paired NCU row. The two traces came from the same workload
   launch shape but are not byte-identical trace contracts; preprocessing and
   predication accounting differ.

Consequently, the 14,174-cycle result and the 4.87 s process time are retained
as **executable overlap evidence only**. No error percentage, speed ratio, or
ranking is reported from this probe. The current project's fresh enhanced
capture remains the only NCU-bound evidence. A valid M6 cell still requires a
3090 Ti-compatible external card/configuration and a declared common trace and
observable boundary.

## Reproduction commands

Build (from the external Accel-Sim checkout):

    export CUDA_INSTALL_PATH=/home/jakeshea/GPU_sims_notes/toolchains/cuda-12.8.1
    export GPGPUSIM_ROOT=$PWD/gpgpu-sim
    cmake -S gpu-simulator -B gpu-simulator/build-local-cuda128 -G Ninja \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
      -DCUDAToolkit_ROOT="$CUDA_INSTALL_PATH"
    cmake --build gpu-simulator/build-local-cuda128 \
      --target accel-sim.out -j8

Replay (from a fresh run directory):

    accel-sim.out \
      -config "$GPGPUSIM_ROOT/configs/tested-cfgs/SM86_RTX3070/gpgpusim.config" \
      -trace /path/to/traces/kernelslist.g \
      -specialized_unit_1 1,4,4,4,4,BRA \
      -specialized_unit_2 1,4,200,4,4,TEX \
      -specialized_unit_3 1,4,32,4,4,TENSOR \
      -specialized_unit_4 1,4,4,4,4,UDP \
      -trace_opcode_latency_initiation_int 2,2 \
      -trace_opcode_latency_initiation_sp 2,1 \
      -trace_opcode_latency_initiation_dp 64,64 \
      -trace_opcode_latency_initiation_sfu 21,8 \
      -trace_opcode_latency_initiation_tensor 32,32

## M6 status

Status changes from “source/build blocked” to “external executable overlap
closed, fair numeric cell still unavailable.” Do not use this probe to claim
that the Rust simulator defeats Accel-Sim. Keep the M6 matrix's cell marked
`unsupported_pending_architecture_and_trace_alignment` until the two boundary
mismatches are resolved.
