# Baseline feasibility — first M6 probe

Updated 2026-09-23. No cross-simulator numeric claim is made by this note.

The machine-readable M6 matrix is
[`baseline_matrix_20260924.json`](baseline_matrix_20260924.json). It records
the local NCU anchor, executable external overlaps, and unsupported cells
without collapsing them into a ranking.

## Valid comparison cells

None have been scored yet. A valid cell requires a runnable baseline, the same
workload/trace scope, comparable modeled detail, and an identical observable
definition.

## Huerta MICRO'25 source probe — initial host attempt

The locally retained modern-gpu-simulator-micro-2025 source was initially configured for
CUDA 13.2 and a temporary makedepend shim, then built for 300 seconds. The
build did not produce an executable.

The first host blocking compiler error is in the retained RapidJSON headers:

    assignment of read-only member
    rapidjson::GenericStringRef<CharType>::length

The source also exposes an old header dependency requiring std::function. This
is an environment/source compatibility blocker, not evidence about simulator
accuracy. No source in that external baseline was modified by this initial
attempt. A later CUDA 12.8/system-protobuf compatibility build is recorded
below.

A second attempt used a clean Ubuntu 22.04 container, the CUDA 13.2 toolkit,
the system protobuf headers, and a temporary makedepend shim. Compilation
advanced substantially but stopped on older assumptions in the baseline's
CUDA/protobuf interface (`cudaDeviceProp::clockRate` and missing
protobuf runtime_version.h). No executable was produced.

## Accel-Sim 2.0

The Accel-Sim 2.0 source is available in the external research workspace. A
user-local CUDA 12.8 redistributable toolchain was sufficient to build its
`accel-sim.out` executable, and its native NVBit tracer/postprocessor produced
a legacy trace for the same `M=256,N=256,K=256` BF16 HMMA workload shape. Three
native replays completed at 14,174 cycles / 5,621,216 instructions and a median
4.870 s process wall time (peak child RSS 489,980 KiB).

This closes an **executable overlap probe**, not a valid comparison cell. The
replay used Accel-Sim's `SM86_RTX3070` card, while the ground-truth host and
this project card are RTX 3090 Ti. It also consumed Accel-Sim's legacy
`.traceg.xz` representation rather than this project's enhanced protobuf
capture. The exact commands, revisions, and artifacts are in
`docs/accelsim_executable_overlap_probe_20260923.md`; no accuracy percentage
or speed ranking is inferred from it.

The official Ubuntu 24.04/CUDA 12.8 container was also attempted from the
published package tag. Most layers downloaded, but two registry layers entered
repeated network retries and the pull was terminated without a usable image.
This is an external environment failure; no baseline number is inferred from
it.

## Huerta MICRO'25 modern-GPU simulator

The retained Huerta source includes a native `SM86_RTX3090` card and now has an
executable overlap probe using the same enhanced protobuf capture as the local
Rust replay. Two successful runs reported 16,708 modeled cycles, 5,242,880
instructions, 512 issued CTAs, and 7.680/8.233 s process wall time. The exact
build, command, and outputs are recorded in
`docs/huerta_micro25_executable_overlap_probe_20260923.md`.

This is still not a valid numeric cell: the card is RTX 3090 rather than the
ground-truth RTX 3090 Ti, its counter is not the NCU GPC-cycle observable, and
repeated invocations can fail in the upstream protobuf map materialization
(`map.h:1429`). The result is therefore executable overlap evidence only.

## M6 decision

Do not report a numerical baseline ranking yet. Accel-Sim 2.0 and Huerta
MICRO'25 now have executable overlap records, but the observed cells remain
`unsupported_pending_architecture_and_trace_alignment` or
`executable_overlap_non_scoring`. Continue with the local accuracy/speed and
design-study evidence while seeking a 3090 Ti-compatible external card and a
stable common trace/observable boundary.
