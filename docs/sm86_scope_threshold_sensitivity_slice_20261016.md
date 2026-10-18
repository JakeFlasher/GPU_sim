# SM86 scope-warning threshold sensitivity — 2026-10-16

H56 stress-tests the fixed 4× scope-warning ratio on H41 exact joins. For L2
source-sector and DRAM read-sector operation domains:

| threshold | L2 exclusions | DRAM exclusions | DRAM secondary mean APE |
|---:|---:|---:|---:|
| 2× | 6 | 70 | 19.15% |
| 4× (retained) | 3 | 49 | 26.84% |
| 8× | 3 | 29 | 32.81% |

The 4× threshold is retained as a fixed gross-scope diagnostic, not an
optimization knob. The H40 global opcode warning uses a distinct opcode-domain
relationship and is not balanced against sector endpoints. Excluded rows remain
unresolved and raw/all-row APE remains authoritative.
