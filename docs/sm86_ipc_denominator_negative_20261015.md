# SM86 SM-frontier IPC denominator probe — 2026-10-15

## Hypothesis

Use the explicit SM-active observation frontier rather than the GPC launch
frontier for `sm__inst_executed.avg.per_cycle_elapsed` on SM86. This was a
mechanism-domain probe, not NCU fitting: the two frontiers represent different
observation semantics and the probe tested whether the SM-level endpoint could
be substituted directly.

## Evidence

Four real roots completed full replay 4/4 in
`../sm86_replay_results/h7_sm_ipc_denominator_20261015_v1/full.jsonl.zst`.
The post-replay report was
`../sm86_replay_results/metric_report_h7_sm_ipc_denominator_20261015_v1/`.
The SM-frontier denominator increased elapsed-IPC mean APE to 51.81% on the
four exact joins, versus 12.53% for the retained GPC-envelope denominator.
The hypothesis is rejected and the implementation was reverted before any
complete-corpus promotion. The GPC denominator remains the active effective
endpoint, with its assumptions explicitly reported.

This negative result is preserved as evidence that an SM-active observation
frontier is not automatically equivalent to NCU's elapsed denominator.
