# Mixed NAK-to-CUDA-ELF bridge: live NCU observation

This is a live performance observation, not a test, hash, or verification
workflow. A fused NAK body combining integer multiply-add, XOR, and integer
maximum was inserted into a larger NVIDIA-generated CUDA-ELF template and
launched through the CUDA Driver API.

## Results

The Driver API output prefix was `7,11,8,13`.

| NCU cache control | duration | GPC cycles | instructions | ALU aggregate | LSU aggregate | XU aggregate |
|---|---:|---:|---:|---:|---:|---:|
| `all` | 2.464 us | 3,284 | 26 | 18 | 3 | 0 |
| `none` | 2.336 us | 3,090 | 26 | 18 | 3 | 0 |

Warm/cold ratios are 0.94805 for duration and 0.94093 for GPC cycles. The
instruction and aggregate pipe populations are unchanged. This demonstrates a
real CUDA-loadable body spanning multiple logical operation families and adds
cache-domain endpoint sensitivity for a mixed body.

It does not identify which dynamic PC used which physical sub-unit or issue
cycle. The two-parameter metadata remains template-derived, so this is not
general CUDA ABI/linker closure. Raw profiles and disassembly are retained in
`results/alpha/sm86_nak_cuda_elf_bridge_mixed_ncu_goal_20260929/`.

Five-repeat follow-up profiles are retained under `repeats/`:

| control | median duration | duration range | median GPC cycles | GPC range | stable counts |
|---|---:|---:|---:|---:|---|
| `all` | 2.464 us | 2.432–2.496 us | 3,271 | 3,247–3,332 | 26 / 18 ALU / 3 LSU / 0 XU |
| `none` | 2.336 us | 2.304–2.528 us | 3,089 | 3,082–3,355 | 26 / 18 ALU / 3 LSU / 0 XU |

The stable populations and varying timing further separate aggregate
instruction evidence from hidden route and cache-state behavior.

The first PM-sampling invocation omitted the launcher's cubin argument and
returned `CUDA_ERROR_INVALID_VALUE`. A corrected live invocation is retained at
`results/alpha/sm86_nak_cuda_elf_bridge_mixed_pmsampling_goal_20260929/`.
It passes the mixed cubin explicitly and records 2.336 us / 3,085 GPC cycles,
26 instructions, aggregate ALU/LSU/XU 18/3/0, and PM-sampling long-scoreboard
2.205 with no buffer overflow. The short body produces zero PC samples at the
32-cycle interval, so this remains aggregate stall evidence rather than a
physical route or issue-slot witness.

The matching cold-cache PM profile records 2.464 us / 3,265 GPC cycles and
long-scoreboard 2.747 versus warm 2.336 us / 3,085 cycles and 2.205. Aggregate
instruction/resource populations remain 26 / 18 ALU / 3 LSU / 0 XU. This adds
cache-phase endpoint sensitivity to the bridged ABI observation.

A repeated-launch follow-up is retained at
`results/alpha/sm86_nak_cuda_elf_bridge_mixed_pmsampling_repeat_goal_20260929/`.
The launcher issued the same body 256 times per replay pass, but NCU still
reported the selected 26-instruction kernel row and zero PC samples. Repeating
the body therefore did not manufacture a public dynamic route/issue stream.
