# SM86 constant-pressure clock64 probe — 2026-10-15

## Probe

The retained constant-load path kernel was rebuilt for `sm_86` and run on the
local RTX 3090 Ti without NCU. A dependent read-only chain (mode 0) and four
independent reads per iteration (mode 1) used 128 iterations, 256 threads, and
1/4/16 blocks; every run passed the functional/checksum oracle.

| Blocks | Dependent min–max cycles | Four-independent min–max cycles |
|---:|---:|---:|
| 1 | 9,691–9,726 | 12,369–12,419 |
| 4 | 9,691–9,920 | 12,371–12,932 |
| 16 | 9,690–9,943 | 11,577–12,814 |

The four-independent arm adds roughly 2.7k cycles at 128 iterations, but the
differential is an issue/throughput mixture rather than a uniquely identifiable
L0C return frontier. It does not justify changing the existing source-backed
SM86 `11 + 39` pressure arm or adding a generic L1TEX dependency multiplier.

## Verdict

This is negative mechanism evidence: pressure is real and broad, but the
clock64 differential cannot separate operand return, scheduler overlap, and
constant-cache queue service. H28's 6,336,451-instruction pressure observer
remains the active default diagnostic; no new timing arm is promoted and no NCU
value is used as a fit. Compact report only:
`docs/sm86_constant_dependency_probe_slice_20261015.md`.
