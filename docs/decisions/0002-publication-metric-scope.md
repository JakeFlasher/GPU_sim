# ADR 0002: Bind the publication headline to elapsed GPC cycles

- Status: Accepted
- Decision date: 2026-07-13
- Decision code: `CYCLES_ONLY_BINDING`

## Context

The metric registry contains 61 entries and defines the maximum label-capture
surface. The simulator currently implements seven secondary NCU-shaped
descriptors, while its intended primary cycles output remains subject to its
silicon-semantics and evaluation gates. Registry membership and capture do not
by themselves establish that the simulator implements or accurately predicts a
metric.

## Decision

The sole publication-binding headline metric is
`exec_cycles_gpc_max` / `gpc__cycles_elapsed.max`.

Its engineering accuracy target is at most 10% cycle MAPE. At most 15% cycle
MAPE is the minimum publication floor. The headline may not be claimed unless
the applicable evidence, aggregation-semantics, coverage, and evaluation gates
pass; the issue-frontier field formerly named `KernelReplayResult.max_cycles`
(now `diagnostic_last_issue_cycles`) and the separate raw
`terminal_modeled_event_time` are not substitutes for an admitted prediction.

The following seven existing simulator descriptors remain reported secondary
outputs and are ungated for publication:

- `smsp_inst_executed_sum`
- `traffic_gmem_ld_sectors`
- `traffic_gmem_st_sectors`
- `cache_hit_rate_l1tex_read_sector`
- `cache_hit_rate_l1tex_sector`
- `cache_hit_rate_l2_sector`
- `traffic_dram_read_sectors`

The 61-entry metric registry remains the capture ceiling and metric-identity
authority. It is not a commitment to implement, gate, or publish 61 simulator
outputs.

## Consequences

- Publication success or failure is determined by the admitted
  `gpc__cycles_elapsed.max` result at the stated target and floor.
- The seven secondary descriptors must be reported honestly, including typed
  unavailable or not-applicable outcomes, but they do not gate the publication
  headline.
- Capturing the registry superset remains authorized and useful for diagnosis,
  future metric promotion, and repeatability studies without widening the
  current publication claim.
- A secondary metric can become publication-binding only through a later
  explicit operator decision with its own implementation, evidence, and gate
  contract.
