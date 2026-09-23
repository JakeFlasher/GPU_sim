# Baseline feasibility — first M6 probe

Updated 2026-09-23. No cross-simulator numeric claim is made by this note.

## Valid comparison cells

None have been scored yet. A valid cell requires a runnable baseline, the same
workload/trace scope, comparable modeled detail, and an identical observable
definition.

## Huerta MICRO'25 source probe

The locally retained modern-gpu-simulator-micro-2025 source was configured for
CUDA 13.2 and a temporary makedepend shim, then built for 300 seconds. The
build did not produce an executable.

The first blocking compiler error is in the retained RapidJSON headers:

    assignment of read-only member
    rapidjson::GenericStringRef<CharType>::length

The source also exposes an old header dependency requiring std::function. This
is an environment/source compatibility blocker, not evidence about simulator
accuracy. No source in that external baseline was modified.

## Accel-Sim 2.0

The Accel-Sim 2.0 source is available in the external research workspace, but
no local executable or matching SM86 baseline run has been established in this
probe. Its official release supports modern Hopper features and reports broad
H100 validation; those published values remain literature context, not a local
comparison.

## M6 decision

Do not report a numerical baseline ranking yet. Continue with the local
accuracy/speed and design-study evidence. The next baseline action is to use a
supported baseline environment/toolchain or an already-built executable, then
select one overlapping SM86 trace cell and record unsupported cells explicitly.
