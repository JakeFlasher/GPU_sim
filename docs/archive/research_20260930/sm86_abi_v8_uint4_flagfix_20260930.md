# SM86 current ABI-v8 uint4 correction — vector KPARAM_INFO size flag

**Date:** 2026-09-30  
**Hardware:** NVIDIA GeForce RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Result

The earlier pointer/uint4/pointer attempt used the ptxas-derived 16-byte layout
(offsets 0, 16, 32 and a 40-byte parameter block) but encoded the vector
`EIATTR_KPARAM_INFO` size word as `0x21f000`, the 64-bit-pointer form. Its output
was `0,3,0,0`, so that run is **not** evidence of successful uint4 consumption.

The corrected no-template emitter uses the ptxas shape exactly:

* `PARAM_CBANK`: `c[0]` base `0x160`, extent `0x28` (40 bytes);
* parameter offsets/sizes: pointer `(0, 8)`, uint4 `(16, 16)`, pointer `(32, 8)`;
* reverse KPARAM records;
* vector size word `0x41f000` for the 16-byte parameter;
* the body reads all four vector components at `c0[0x170]`, `0x174`,
  `0x178`, and `0x17c` and uses them in four IMAD operations.

Five 1,000-launch CUDA-Driver runs at the locked clocks returned the expected
first words `0,3,6,9`:

| run | CUDA-event envelope |
|---:|---:|
| 1 | 1.762304 ms |
| 2 | 1.747968 ms |
| 3 | 1.763328 ms |
| 4 | 1.793024 ms |
| 5 | 1.752064 ms |

Median: **1.762304 ms per 1,000 launches**.

Artifacts: `results/alpha/sm86_abi_v8_param_campaign_20260930/uint4_flagfix_consumed/`.

## Boundary interpretation

This is a real, semantically consumed 16-byte vector point in the current
ABI-v8 family and removes an emitter defect from the evidence ledger. It does
not establish arbitrary vector/struct layouts, relocation or linker semantics,
resource metadata synthesis, control-word generation, or all-form SM86
instruction coverage. The correct conclusion remains **bounded code-generation
capability, not unrestricted compiler/ISA closure**.
