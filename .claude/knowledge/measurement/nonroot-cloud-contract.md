---
Use when: deciding what a measurement run must record under the pinned NCU protocol (CLAUDE.md Rule 2); setting up root capabilities (clock lock, persistence, counter permissions) on a new host; diagnosing `ERR_NVGPUCTRPERM` / `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` on a freshly provisioned cloud instance; understanding what the retired non-root contract was and why pre-2026-07 artifacts look the way they do.
Primary raw source: `CLAUDE.md` Rules 1-4 (re-founded 2026-07-02) and `plans/micro2027_ncu_sim_execution_plan.md` §1A/§7.
Supporting sources:
- NVIDIA `ERR_NVGPUCTRPERM` page: https://developer.nvidia.com/ERR_NVGPUCTRPERM
- NVIDIA CUPTI permission page: https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
- `eval/guards/test_targets_not_features_guard.py` - the surviving Rule-1 static guard.
Repo touchpoints:
- `tools/local_labels/` - the NCU label factory (capture -> repeatability gate -> admit).
- `tools/nvbit_tracer/` - SASS/address tracing (the simulator's input).
- `tools/cloud/` - provisioning/preflight skeleton, repurposed for GB202 bring-up.
Out of scope: the retired quarantine rules themselves (do not re-apply them).
Version family: CUDA 13.x, NCU 2026.1.0 pinned; RTX 3090 Ti sm_86 (dev host), RTX 5060 laptop GB206 + cloud RTX PRO 6000 GB202 (sm_120 targets).
---

# GPU Measurement Contract (post-2026-07-02 re-founding)

The old non-root/counter-less deployment thesis is **retired**. NCU is a
first-class ground-truth instrument on every target (local and cloud), root
operations are allowed and preferred, and the one surviving prohibition is
Rule 1: **targets-not-features** - NCU/counter values never enter simulator
inputs, trace features, or calibration constants fitted on the cells being
scored. Enforced by `eval/guards/test_targets_not_features_guard.py`.

## The pinned capture protocol (Rule 2)

Every NCU capture records, and label admission REQUIRES:

| Pin | Detail |
|---|---|
| NCU version | 2026.1.0 on the 3090 Ti host; upgrades are deliberate, recorded protocol bumps |
| `--clock-control` | Explicit, never default (the default changed `base`->`boost` in NCU 2026.1) |
| `--cache-control` | Explicit (labels default `all`; the simulator emits cold-cache parity) |
| `--replay-mode` | Explicit |
| Driver + CUDA toolkit | e.g. 610.43.02 / CUDA 13.2 (legacy sweeps were 595.71.05 - record the drift) |
| Clock-lock state | `nvidia-smi -lgc` (+`-lmc` where supported), persistence mode |
| NVML sidecar | clocks/power/temp/throttle time-series per launch |

Root state changes (clock locks, persistence, counter permissions) must be
recorded in the run sidecar.

## Tool-combination rules (still binding - interposition, not policy)

- NVBit uses CUDA driver interposition: **never co-run NVBit with NCU** in the
  same process. Trace runs and NCU label runs are separate, identically
  configured runs joined offline by kernel identity + launch index + static
  signature.
- Nsight Systems may run in separate runs; do not attach it to an NVBit process.

## New-host / cloud-instance bring-up (GB202 checklist seed)

1. Counter access: run as root, or set `NVreg_RestrictProfilingToAdminUsers=0`
   (module reload), or grant CAP_SYS_ADMIN / the R610 capabilities path.
   Symptoms of missing setup: `ERR_NVGPUCTRPERM` (NCU),
   `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` (CUPTI).
2. `nvidia-smi -pm 1`; verify `-lgc <ref>` sticks. On laptops a lock can act as
   a *limit* under thermal pressure - keep nvbench-style noise gates (≥10
   samples, 0.5% rel-stddev stop, kernels ≥20 µs where possible).
3. Rebuild all compiled artifacts on-host (`tracer_tool.so`, metric_extractor);
   binaries never transfer (CLAUDE.md session-handoff rule).
4. Record the full Rule-2 pin set in a first sidecar before any capture at
   scale. `tools/cloud/preflight.sh` is the probe skeleton to extend (its
   non-root gating is obsolete; its structure is reusable).

## Reading legacy-era artifacts

Artifacts produced before 2026-07-02 (sweeps, closures, the sm_120 refutations)
were captured under the old contract: no clock locking, CUPTI dual-source
timing, NCU quarantined to `tools/local_labels/`. Their DVFS spread (15.34x on
the laptop) and `cross_arch_precondition: false` pins are era artifacts, not
current constraints. The tri-state relax (2026-07-06) lets new closures carry
`true`/`"pending"` without invalidating frozen historical records.

## See Also

- `.claude/knowledge/measurement/cupti-activity-api.md` (CUPTI Activity is now
  corroboration, not the timing label - D8 retired dual-source as ground truth)
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md`
- `plans/micro2027_ncu_sim_execution_plan.md` §1A (capture protocol v2), §7 (M0 gate)

## Escalate to Raw When

- A new NCU release changes replay/clock-control semantics (re-pin deliberately).
- A cloud deployment mode changes NVBit injection, MPS, MIG, vGPU, or
  confidential-computing behavior.
- A new CUDA release changes CUPTI availability or NVBit compatibility.
