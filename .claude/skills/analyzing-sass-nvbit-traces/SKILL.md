---
name: analyzing-sass-nvbit-traces
description: "Inspect SASS/cubin and NVBit enhanced traces, diagnose capture or replay failures, and join static/control and dynamic instruction evidence."
---

# SASS and NVBit

Start with `tools/nvbit_tracer/README.md` and `docs/runbook_host.md`.
Use `cuobjdump --dump-sass BINARY`, `nvdisasm CUBIN`, or
`tools/nvbit_tracer/tracer_tool/trace_printer TRACE_DIR` as applicable.
Pinned producer environment: `tools/nvbit_tracer/pinned_build_env.sh`.

Consumer seams: `sim/trace/src/reader.rs`, `sim/sm/src/program.rs`, and
`tools/sass_control/`. Wire schemas are under
`tools/nvbit_tracer/traces_enhanced/dynamic_trace/`; generated bindings must
remain consistent when a schema changes. Trace completeness/masks/addresses
are not inferred from a workload name or missing field.

Static opcode text is not dynamic physical-pipe selection. Native replay
cannot invent untaken branches or new values. Active graph launches are
explicitly rejected by the current producer. Stable module-owner identity is
not guaranteed for JIT/deleted/opaque code.

Reference: `references/NVIDIA_Manuals/cuda_binary_utility/`. Optional
`.claude/knowledge/` cards are absent in this checkout; do not retry them.
