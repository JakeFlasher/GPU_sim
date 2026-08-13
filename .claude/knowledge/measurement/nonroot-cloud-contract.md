# GPU measurement capabilities

Use this card when bringing up measurement on a new host or interpreting older
non-root captures. The former repository-wide non-root and no-NCU restrictions
are retired.

## Current approach

Use the tools available on the selected host:

- NCU for hardware-counter ground truth;
- CUDA events and CUPTI Activity for elapsed-time observations;
- NVBit for dynamic SASS/address traces;
- NVML or `nvidia-smi` for useful clock, power, and thermal context;
- `cuobjdump`, `nvdisasm`, and `ptxas` for static inspection; and
- authorized administrative commands when host configuration needs them.

NCU measurements are validation targets. Do not feed a scored workload's
measured counters into simulator features or tune per-workload constants to
that same cell.

## Host capability troubleshooting

`ERR_NVGPUCTRPERM` or `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` means the requested
counter surface is unavailable to the current process. Depending on the host,
run with appropriate privileges or configure NVIDIA profiling permissions.
This is a host setup issue; it does not disable other tools or freeze the
project.

Clock locks and persistence mode may reduce variance when supported. On mobile
or thermally constrained systems, a requested lock can behave as a ceiling, so
observe actual clocks and temperature. Repeat short or noisy kernels and report
their dispersion.

## Tool interaction

Run NCU and NVBit in separate matched processes. Both alter execution through
replay or interposition, and combining them can invalidate the result. Nsight
Systems is also easiest to interpret in its own run.

## Historical captures

Pre-July-2026 datasets include non-root, counter-free, and fixed-regime
experiments. Their restrictions describe how those bytes were produced, not
what current research may do. See `pinned-protocol-v21.md` when reproducing the
specific RTX 3090 Ti v2.1 regime.

Current direct entry points are `docs/runbook_host.md`,
`tools/local_labels/collect_ncu.py`, and `tools/nvbit_tracer/README.md`.
