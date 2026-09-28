# Baseline feasibility — first M6 probe

Updated 2026-09-28. This dated report retains its executable-probe facts, but
its interpretation is superseded by the three-lane contract below.

## Interpretation correction (2026-09-28)

NCU measurements on the local RTX 3090 Ti are the detailed ground truth.
Published/previously collected Accel-Sim aggregates are not ground truth, but
they are active accuracy ceilings and throughput/speedup floors that this
project must meet or surpass; see
[`accelsim_literature_lower_bound_manifest_20260928.json`](accelsim_literature_lower_bound_manifest_20260928.json).
The pinned executable probes below are parser/representation evidence only and
are not required to be a same-card timing baseline.

The machine-readable M6 matrix is
[`baseline_matrix_20260924.json`](baseline_matrix_20260924.json). It records
the local NCU anchor, executable external overlaps, and unsupported cells
without collapsing them into a ranking.

## Comparison lanes

* **Local NCU lane:** score simulator accuracy against NCU on the RTX 3090 Ti
  with a content-bound workload, endpoint, and clock protocol.
* **Literature-floor lane:** evaluate the manifest's per-domain error ceilings
  and speed floors; different published hardware prevents per-kernel transfer
  but does not erase the target.
* **Executable-smoke lane:** validate parser/trace compatibility only. Its
  cycle counter is not an NCU label.

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
`docs/accelsim_executable_overlap_probe_20260923.md`; no local NCU accuracy or
literature-floor attainment is inferred from it.

A follow-up derived profile changed only SM count, memory partitions, and
locked clock domains toward the RTX 3090 Ti anchor. Three replays completed at
12,588 cycles, 5,635,808 instructions, and median 6.337 s wall time. Because
the remaining cache/scheduler/memory fields still come from RTX 3070, this
remains `derived_profile_non_scoring`; details are in
`docs/accelsim_derived_sm86_overlap_20260924.md`.

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

This is still not a direct executable NCU cell: the card is RTX 3090 rather than the
ground-truth RTX 3090 Ti, its counter is not the NCU GPC-cycle observable, and
repeated invocations can fail in the upstream protobuf map materialization
(`map.h:1429`). The result is therefore executable overlap evidence only.

## M6 decision

Do not report a direct executable-cycle ranking. Accel-Sim 2.0 and Huerta
MICRO'25 executable overlaps remain parser/representation evidence, while the
published Accel-Sim aggregate rows are registered accuracy-ceiling and
throughput/speed-floor targets. Continue
with local NCU accuracy-ceiling and throughput/speed-floor evidence and report attainment against the
manifest; a 3090 Ti-compatible external card is not required for the
literature-floor lane.
