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
Related context: NCU is the ordinary counter tool;
`measurement/pinned-protocol-v21.md` describes one historical fixed-clock
dataset. Other CUPTI surfaces may be used when supported and relevant.
Version family: CUDA/CUPTI 13.x.
---

# CUPTI Activity API

CUPTI Activity records kernel and transfer timestamps in un-replayed runs and
can corroborate NCU or CUDA-event timing. It often works without admin access
to GPU hardware counters.

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

## Other CUPTI surfaces

NCU is usually the simpler counter instrument. Availability of these CUPTI
surfaces varies by CUDA version and target:

- CUPTI Events and Metrics;
- CUPTI Profiling API and Range Profiler;
- PM and PC sampling;
- SASS metrics; and
- instruction execution activity kinds.

In CUDA 13.0+, CUPTI Events and Metrics are removed for modern targets. Check
the installed CUPTI documentation before choosing a lower-level counter path.

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
