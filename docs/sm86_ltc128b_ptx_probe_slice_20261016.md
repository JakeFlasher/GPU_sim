# SM86 `.L2::128B` / LTC128B probe — 2026-10-16

## Result

The H61 PTX blocker is resolved for a supported frontend path. CUDA 13.2
`ptxas -arch=sm_86` accepts inline PTX
`ld.global.ca.L2::128B.u32` and `cuobjdump` reports
`LDG.E.LTC128B.STRONG.SM`. The rebuilt probe executed successfully on the
RTX 3090 Ti host (CC 8.6, driver 615.71.09). Sources, SASS, and run outputs are
under:
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/ltc128b_scope_h65/`.

## Matched NCU observation

A same-binary probe used 84 blocks, eight warps/block, 64 iterations, eight
active lanes, and a 5.5 MiB streaming allocation. Plain `.ca` and
`.ca.L2::128B` forms produced the same 43,008 global-load sectors and zero L1
lookup hits:

| Form | L2 TEX read-miss sectors | promoted sectors | DRAM read sectors |
|---|---:|---:|---:|
| plain `.ca` | 43,008 | 0 | 43,176 |
| `.L2::128B` / LTC128B | 172,032 | 129,024 | 172,672 |

The LTC form therefore promotes three additional sectors for each one-sector
L2 read miss in this control. NCU values are independent observations only;
no scored workload or simulator parameter used them.

## Decision and remaining work

Retain H48 static form observers and add this as independent evidence for a
future typed `promoted_fill_sectors` observer. Do not yet enable timing-active
promotion in the simulator: the probe has no NVBit capture/holdout trace, the
full form matrix (plain/U16/constant/bypass/working-set) is incomplete, and
promotion must preserve L2 source scope and DRAM read-client accounting. The
single H40/H41 64× corpus scope outlier is not automatically attributed to
LTC128B by this probe.
