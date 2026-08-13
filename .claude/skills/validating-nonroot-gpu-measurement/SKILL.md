---
name: validating-nonroot-gpu-measurement
description: Diagnose and run GPU measurement on hosts with or without profiling privileges. Use for NCU permissions, CUPTI Activity, CUDA-event timing, NVBit capture, clock-state interpretation, or old non-root campaign artifacts.
---

# Validate GPU measurement

Choose tools from the capabilities of the current host. The old repository-wide
ban on NCU, root operations, clock controls, and privileged CUPTI is retired.
This skill is technical guidance, not a permission system.

## Useful cards

- `.claude/knowledge/measurement/nonroot-cloud-contract.md` — current host
  capability choices and interpretation of older non-root captures.
- `.claude/knowledge/measurement/cupti-activity-api.md` — CUPTI Activity
  timestamps and deployment caveats.
- `.claude/knowledge/measurement/nvbit-injection-playbook.md` — build,
  dependency, and injection troubleshooting.
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md` — device timer
  semantics and clock variation.
- `.claude/knowledge/measurement/pinned-protocol-v21.md` — historical v2.1
  settings and measured clock-regime effects; reuse only when reproducing that
  particular dataset.

## Working method

1. Identify the measurement needed and the current GPU/tool availability.
2. Use NCU, CUPTI, CUDA events, NVBit, static SASS/PTX, or NVML as appropriate.
3. Run NVBit and NCU in separate processes because both use instrumentation or
   replay mechanisms that can interfere.
4. Record enough tool, device, workload, and clock context to interpret the
   number. Exact historical settings are needed only for a historical
   reproduction claim.
5. Repeat noisy measurements and report dispersion. A failed probe is a
   technical result to diagnose, not a repository freeze.

NCU values remain evaluation targets. Do not feed a scored workload's measured
counter values back into that workload's simulator features or parameters.

## Direct entry points

```bash
.venv/bin/python tools/local_labels/collect_ncu.py --help
make -C tools/nvbit_tracer
bash tools/nvbit_tracer/smoketest/no_op_kernel_idempotency.sh
```

For the retained legacy campaign formats, opt in through `run-sol-campaign`.
