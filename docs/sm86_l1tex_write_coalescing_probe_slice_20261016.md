# SM86 L1TEX/XBAR write-coalescing probe — H75 — 2026-10-16

## Matched real RTX 3090 Ti probe

A new CUDA 13.2/g++-14 `sm_86` probe was built in
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/store_coalescing_h75/`.
The same binary, 84 blocks, four warps per block, depth 64, and locked clocks
was profiled with NCU for three address/spacing arms:

| arm | L2 TEX `op_write` sectors | `op_write_lookup_hit` | `op_write_lookup_miss` | L1TEX→XBAR write sectors | L1 global-store sectors |
|---|---:|---:|---:|---:|---:|
| distinct 128B lines | 86,016 | 86,016 | 0 | 86,016 | 86,016 |
| four warps, same 32B sector | 5,699 | 6,213 | 0 | 5,699 | 21,504 |
| same sector with arithmetic spacing | 6,651 | 5,888 | 0 | 6,651 | 21,504 |

The same-sector arms demonstrate a large L1TEX-to-XBAR/L2 write reduction, but
`op_write_lookup_hit` can exceed `op_write` sectors and changes non-monotonically
with spacing. Thus the NCU lookup-hit endpoint is not a direct L2 tag-residency
counter, and a simple present-sector or fixed-window formula would be unsafe.

## Decision

H75 is probe evidence for a write-coalescing mechanism, not a promoted timing
model. No simulator source, L2 lookup policy, DRAM service, or completion
frontier changed; the H74 full-first corpus artifact remains the baseline.
A future P3 arm requires dynamic same-binary NVBit write-request identity and a
crosswalk among L1 global-store sectors, XBAR write sectors, L2 `op_write`, and
lookup hits/misses, including cold/warm residency and cross-warp spacing.
