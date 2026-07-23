---
name: validating-nonroot-gpu-measurement
description: RETIRED historical reference. The former non-root/no-NCU measurement contract is not current policy and must not be applied. Use root CLAUDE.md, active label schemas, tools/local_labels/, and the host runbook for current measurement authority; retain this skill only to interpret pre-pivot artifacts.
---

# Validating Non-Root GPU Measurement — RETIRED (historical)

> **RETIRED at the 2026-07-02 re-founding.** Every rule below describes the
> previous project era and is historical context only. Do not infer current NCU,
> clock, privilege, driver, or simulator permissions from this file. Current
> measurement authority is root `CLAUDE.md`, the active label schemas,
> `tools/local_labels/`, and the host runbook; NCU values remain targets/labels,
> never simulator features or same-cell calibration inputs. Read this file only
> to interpret pre-pivot artifacts.

This skill owned every measurement-contract question for the RETIRED
non-root measurement and prediction direction. The contract was enforced
by a static-guard test
(`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`) —
now superseded.

## Cards (read in this order for unfamiliar tasks)

1. `.claude/knowledge/measurement/nonroot-cloud-contract.md` — what's allowed, what's forbidden, the static-guard test
2. `.claude/knowledge/measurement/cupti-activity-api.md` — CUPTI Activity is the only counter-free wall-clock telemetry surface (MIG/MPS/vGPU/CC caveats, CUDA 13.x deprecations)
3. `.claude/knowledge/measurement/nvbit-injection-playbook.md` — `LD_PRELOAD` / `CUDA_INJECTION64_PATH`, NVBit 1.8 / sm_120 / driver 595.x compat, tool-combination constraints
4. `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` — what `%clock64` and `%globaltimer` PTX promise, what's undefined, cycle-conversion machinery under DVFS variance
5. `.claude/knowledge/measurement/nvarchsim-vs-this-project.md` — NVArchSim positioning (HPCA 2021, Villa et al.) as prior art

## Hard rules (CLAUDE.md Critical Rules 1–4)

| # | Forbidden | Why |
|---|---|---|
| 1 | NCU, `ncu`, `nv-nsight-cu-cli`, `.ncu-rep`, `.ncu-repz` | Cloud GPU containers deny `RestrictProfilingToAdminUsers` toggle; NCU returns `ERR_NVGPUCTRPERM`. |
| 2 | `nvidia-smi -lgc/-lmc/-rgc/-rmc/-pm/-c/-pl/-mig/-ac`, `--lock-gpu-clocks`, `--lock-memory-clocks`, `modprobe nvidia`, `NVreg_RestrictProfilingToAdminUsers`, DKMS, MOK, kernel-module rebuild | All require GPU-driver-level root the cloud target denies; project never installs host drivers. |
| 3 | CUPTI Events API (`cupti_events.h`), CUPTI Metrics API (`cupti_metrics.h`), CUPTI Range Profiler, CUPTI PM Sampling | All gated behind profiling-permission the cloud denies. **Only the CUPTI Activity API is allowed.** |
| 4 | `/etc/apt/sources.list.d/` modifications, DKMS, reboots, host-driver installs | Container `apt-sudo` is allowed only for in-container build-environment setup. |

## Allowed surfaces (the only telemetry the contract permits)

- **CUPTI Activity API** — kernel timestamps, memcpy events, runtime/driver tracing.
- **NVBit 1.8** — host-built tracer at `tools/nvbit_tracer/`. Works via `LD_PRELOAD` + `CUDA_INJECTION64_PATH` (no driver-root needed).
- **`%clock64` injection via NVBit** — per-kernel cycle counter, reconciled against wall-clock under perturbation envelope.
- **Static SASS / PTX introspection** — `cuobjdump`, `nvdisasm`, `cu++filt`, `nvprune` are read-only and allowed.
- **Compile-path inspection** — `nvcc -keep`, ptxas verbose output.

## Operator workflow

```
tools/cloud/container_bootstrap.sh      # apt deps + toolchain check
tools/cloud/preflight.sh                # emit preflight manifest
                                        #   probes NVBit injection, CUPTI Activity availability, cluster-attribute API
workloads/simbench_bw/                  # main bandwidth + tensor-core suite
```

## Current Research Package

- `README.md` — active project scope and hard non-root contract.
- `CLAUDE.md` — agent-facing guardrails.
- `MANIFEST.md` — active repository scope and cleanup record.
- `workloads/simbench_bw/README.md` — benchmark/evidence harness.
- `tools/cloud/README.md` — non-root cloud workflow.
- `tools/nvbit_tracer/README.md` — NVBit tracer build and runtime notes.
- `references/papers/knowledge_base/problem_playbook.md` — prior-art routing.

## When the user asks "why can't I just use NCU?"

Open `.claude/knowledge/measurement/nonroot-cloud-contract.md` and explain:
1. Cloud target denies `NVreg_RestrictProfilingToAdminUsers` toggle.
2. NCU returns `ERR_NVGPUCTRPERM` immediately.
3. The active contract was specifically built to recover accurate timing and cache-behavior prediction under that constraint.
4. Bypassing the rules also fails CI (the static-guard test scans active code paths).

Do NOT propose workarounds that violate Rules 1–4.

## Prediction Target

The active target is accurate wall-clock and cache-behavior prediction
from non-root signals. Historical simulator gates may be useful as prior
art, but they are not active project direction.

## Cross-references to other skills

- PTX `%clock64`/`%globaltimer` syntax → `interpreting-ptx-isa`
- NVBit unknown-opcode SASS handling → `analyzing-sass-nvbit-traces`
- Simulator prior-art context → `designing-gpu-simulator`
- Blackwell-specific telemetry quirks → `tuning-blackwell-sm120-kernels`

## Cite consulted files

End every reply with `## Knowledge Consulted` listing the cards / raw sources read.
