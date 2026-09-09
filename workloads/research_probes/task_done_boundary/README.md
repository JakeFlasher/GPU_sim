# SM86 task-done boundary probe

This probe separates producer completion, consumer completion, host copy, and
memory quiescence for a small multi-kernel workload. It is diagnostic input for
simulator endpoint work; it does not tune the simulator.

Current outputs:

- `results/captures/sm86_task_done_boundary_20260904_v2`
- `results/captures/sm86_task_done_boundary_ncu_20260904_v2`
- `results/captures/sm86_task_done_boundary_ncu_sensitivity_20260904_v1`

Run the producer capture and NCU collection independently, then replay the
captured trace with `sim-run` using the command recorded beside the trace.
The result separates issue, modeled terminal event, SM-active terminal, and
host-side endpoints. A fixed retirement tail is not inferred from this probe.
