# sm_86 GA102 address-to-LTS mapping

This campaign establishes an exact collision oracle, but does **not** promote an address hash into the simulator.

## Live evidence

- 196,096 one-pass Nsight Compute kernel actions across 9 official reports.
- 97,920 exhaustive pair observations across three allocation phases; every phase yields 48 transitive LTS classes.
- 4,096 weighted/encoded and 2,400 wide/encoded overlap records agree exactly.
- Dense fit: 65,535 lines, 48 classes, 1363–1368 lines/class.
- Strict high holdout: 32,768 lines over regions 65,535–262,139.

The weighted oracle uses 105 anchor CTAs plus binary CTA weights. The encoded oracle is stronger and cheaper: an anchor executes 48 RED generations while the 48 representatives execute 0–47, so one LTS maximum identifies one class.

## Hash transfer

| candidate | fit recall | high-holdout recall | high-holdout specificity | balanced |
|---|---:|---:|---:|---:|
| line_mod | 2.11% | 2.07% | 97.92% | 49.99% |
| sector_mod | 8.28% | 8.31% | 91.67% | 49.99% |
| xor_folded | 2.08% | 2.09% | 97.92% | 50.00% |
| knuth_mult | 2.13% | 2.07% | 97.92% | 49.99% |
| shift_mod | 2.16% | 2.11% | 97.92% | 50.01% |
| bit_reverse | 2.02% | 2.47% | 97.49% | 49.98% |
| ipoly64_line | 2.35% | 2.32% | 97.66% | 49.99% |
| splitmix64 | 2.08% | 2.09% | 97.92% | 50.00% |

A deliberately overfit 65,536-entry low-bit lookup is exact on fit but has only 13.37% collision recall and 0.00% direct-label accuracy on holdout.

Region bits [0, 2] are invariant in the dense fit. Three nonzero linear parity invariants exist locally, but only one survives when the high-address holdout is included. This is direct evidence against treating the local pattern as a transferable linear slice hash.

## Decision

No hash is integrated. Scheduling remains conservative and the GA102 slice hash stays an explicit research gap. The next bounded discriminator is a CUDA VMM alias experiment: map one physical allocation at multiple virtual addresses and test whether the LTS class follows the physical allocation or the virtual address.
