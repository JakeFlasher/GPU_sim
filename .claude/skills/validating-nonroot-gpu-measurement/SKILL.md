---
name: validating-nonroot-gpu-measurement
description: "Run or diagnose GPU measurements with the privileges and tools actually available, including NCU, CUPTI, CUDA events and NVBit. Interpret legacy non-root captures without inheriting their old bans."
---

# GPU measurement

The historical name does not prohibit profiling, clock controls or authorized
native tools. Root `AGENTS.md` owns permissions. Check device name, UUID,
compute capability, driver and index independently before a new host claim.

Use `docs/runbook_host.md`, `tools/local_labels/collect_ncu.py`, and
`tools/nvbit_tracer/README.md`; profiler semantics are in
`references/NVIDIA_Manuals/ncu_guide/`. Missing optional knowledge cards do
not block measurement.

Keep CUDA-event, CUPTI and NCU intervals distinct. Record clocks/cache/replay
settings and avoid scoring different counter domains. NCU is an independent
validation target, never a same-workload simulator feature or tuning input.
Historical pinned host profiles describe their capture dates, not the machine
forever. On permission failure, report the concrete error and choose a valid
available measurement—not an invented substitute counter.
