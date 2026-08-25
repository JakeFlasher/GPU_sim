# sm_86 GA102 address-to-LTS mapping milestone

**Completed:** 2026-08-20  
**Goal:** recover a transferable LTS collision map or establish why virtual
trace addresses are insufficient

## Acceptance

1. Hold cubin, allocation size, operation count, and cache-line granularity
   constant while changing only runtime-selected addresses.
2. Cover every pair in a discovery window and require a complete transitive
   48-class relation.
3. Cross-check two independently encoded collision oracles.
4. Exhaustively classify at least 65,535 fit lines and preserve a disjoint
   high-address holdout.
5. Score collision recall and specificity, not imbalanced raw accuracy.
6. Promote no hash unless strict holdout reaches 95% collision recall and 99%
   specificity.

## Outcome

All measurement and internal-consistency criteria pass. Nine official NCU
reports contain 196,096 one-pass kernel actions. The two oracles agree exactly
where they overlap, and every allocation phase recovers 48 classes.

No address-only candidate transfers. The strongest deliberately overfit lookup
retains only 13.37% of true high-address collisions, while the simulator hashes
and IPOLY remain at chance. No timing or slice hash is integrated. The next
milestone is a same-physical-allocation/multiple-virtual-alias discriminator.

See the completed
[milestone report](../docs/sm86_atomic_lts_address_mapping_milestone.md).
