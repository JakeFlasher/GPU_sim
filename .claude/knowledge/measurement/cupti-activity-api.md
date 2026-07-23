---
Use when: implementing or reviewing the non-root CUPTI Activity collector; diagnosing `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`; checking which CUPTI activity kinds are admissible in cloud containers.
Primary raw source: NVIDIA CUPTI Usage Guide 13.x, https://docs.nvidia.com/cupti/main/main.html
Supporting sources:
- NVIDIA CUPTI Release Notes: https://docs.nvidia.com/cupti/release-notes/release-notes.html
- NVIDIA CUPTI `ERR_NVGPUCTRPERM` guidance: https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
- `CLAUDE.md` and `.claude/knowledge/measurement/nonroot-cloud-contract.md`.
Repo touchpoints:
- `workloads/simbench_bw/cupti_activity_collector.{h,cu}` - current collector implementation.
- `tools/cloud/preflight.sh` - CUPTI Activity capability probe.
- `tools/cloud/preflight_manifest.py` - manifest validation.
Out of scope: NCU counter capture (first-class, but owned by `measurement/pinned-protocol-v21.md`); CUPTI Events/Metrics, CUPTI Profiling, Range Profiler, PM Sampling (removed/redundant CUPTI surfaces — counters come from NCU).
Version family: CUDA/CUPTI 13.x.
---

# CUPTI Activity API

CUPTI Activity is a corroboration tool alongside first-class NCU: it records
kernel and transfer timestamps in un-replayed runs, cross-checking NCU-derived
timing collected under the pinned protocol
(`measurement/pinned-protocol-v21.md`). It works without admin access to GPU
hardware counters.

## Allowed Activity Kinds

| Activity kind | Role |
|---|---|
| `CUPTI_ACTIVITY_KIND_KERNEL` | Kernel start/end timestamps and launch metadata. |
| `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` | Kernel timestamps when streams overlap. |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | Transfer timing and byte counts. |
| `CUPTI_ACTIVITY_KIND_MEMSET` | Bulk initialization timing. |
| `CUPTI_ACTIVITY_KIND_DRIVER` | Driver API call timing. |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | Runtime API call timing. |
| `CUPTI_ACTIVITY_KIND_OVERHEAD` | CUPTI overhead diagnostics. |
| `CUPTI_ACTIVITY_KIND_DEVICE` / `CONTEXT` | Device and context metadata. |

The collector should degrade gracefully if a deployment blocks an activity kind,
and should surface that failure in the sidecar or preflight manifest.

## Removed or Redundant CUPTI Surfaces

Do not build counter collection on these CUPTI paths — counter ground truth
comes from NCU under the pinned protocol (`measurement/pinned-protocol-v21.md`),
and these surfaces are removed or redundant:

- CUPTI Events and Metrics.
- CUPTI Profiling API.
- Range Profiler.
- PM Sampling.
- PC sampling.
- SASS metrics.
- Instruction execution activity kinds.

In CUDA 13.0+, CUPTI Events and Metrics are removed for modern targets. NCU is
the project's counter instrument on all targets; CUPTI stays timestamp-only.

## Collector Pattern

The current collector under `workloads/simbench_bw/` should:

1. Enable only allowed Activity kinds.
2. Register buffer request/completion callbacks.
3. Flush on synchronization and process exit.
4. Emit sidecar records with kernel name, correlation id, launch index,
   timestamps, stream/context ids, grid/block dimensions, registers, and shared
   memory.
5. Include preflight-manifest provenance so timing can be joined with the
   environment state.

## Coexistence With NVBit

Collect CUPTI Activity and NVBit in separate runs. NVBit uses CUDA driver
interposition and can conflict with other injected/profiling tools in the same
process. Join offline by kernel name, launch index, and a static signature.

## See Also

- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- `workloads/simbench_bw/cupti_activity_collector.cu`
- `workloads/simbench_bw/README.md`

## Escalate to Raw When

- CUPTI changes the Activity callback buffer state machine.
- MIG, MPS, vGPU, WSL, or confidential-computing mode changes Activity support.
- A new CUDA release changes CUPTI removal/deprecation behavior.
