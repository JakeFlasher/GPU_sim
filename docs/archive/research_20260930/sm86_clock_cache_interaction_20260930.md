# Clock/cache interaction surface (2026-09-30)

The four paired temporal campaigns permit a post-hoc decomposition without
using any values for fitting or routing:

* low-clock cold/warm;
* boost-requested high-clock cold/warm.

For each workload and endpoint, define

```text
interaction = (high_warm / high_cold) / (low_warm / low_cold)
```

Unity would support a separable clock factor and cache-phase factor. The
observed interaction ratios are:

| endpoint | median | p10--p90 | min--max |
|---|---:|---:|---:|
| GPC cycles | 0.988 | 0.929--1.109 | 0.775--1.173 |
| duration | 0.987 | 0.928--1.107 | 0.778--1.171 |

The median is close to separable, but the broad per-workload spread requires a
bounded interaction interval rather than a single clock multiplier. This
supports the regime committee's endpoint-specific intervals and retains cache
history as a latent state.

Report: `results/alpha/sm86_temporal_phase_interaction_goal_20260930/interaction.json`.

This bounded interaction treatment follows the piecewise cache/bandwidth
methodology in the [Ampere GPU memory-model source](https://github.com/brucelee503jo3/gpu-memory-model)
and the modular analytical-model direction of [AMALi](https://github.com/ShihengCao/AMALi):
retain separable mechanisms where evidence supports them, but expose residual
uncertainty when a hidden state changes the response. The local SM86 model is
weaker because cache ownership and timing endpoints remain externally hidden.

Source inspection of AMALi found an explicit separation between opcode/unit
latency (`ISA_parser/sass_parser.py`), piecewise cache hit-rate estimation from
reuse profiles (`src/memory_model.py::get_hit_rate_analytical`), and
throughput/occupancy slopes (`src/accelerators.py`). The SM86 committee borrows
that separation while keeping unavailable cache-owner and endpoint fields as
intervals or unknowns.
