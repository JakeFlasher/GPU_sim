# NAK-to-CUDA-ELF template sensitivity

The same retained NAK mixed body was bridged into two independently NVIDIA-
compiled CUDA-ELF templates:

| template | parameters | cold NCU | warm NCU | instructions | ALU/LSU/XU |
|---|---|---:|---:|---:|---:|
| prior large template | two-parameter layout | 2.464 us / 3,265 cycles | 2.336 us / 3,085 cycles | 26 | 18/3/0 |
| shifted template | extra/out/in layout | 2.464 us / 3,274 cycles | 2.432 us / 3,198 cycles | 26 | 18/3/0 |

The cold endpoints agree closely, but warm duration/GPC differ materially while
SASS instruction and aggregate resource populations are identical. This is a
live demonstration that metadata/ABI layout participates in the measured
endpoint. It narrows the ABI blocker but does not provide a general relocatable
CUDA ABI, QMD/control-word generator, or physical route witness.

Artifacts:
`results/alpha/sm86_nak_cuda_elf_bridge_mixed_alt_template_20260929/`.
