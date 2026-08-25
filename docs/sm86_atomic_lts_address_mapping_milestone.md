# Completed milestone: sm_86 GA102 address-to-LTS mapping

**Status:** complete 2026-08-20; exact oracle retained, address hash withheld  
**Predecessor:** [many-region atomic/LTS scheduling](sm86_atomic_many_region_lts_milestone.md)  
**Evidence:**
`results/captures/sm86_atomic_lts_mapping_20260820_v1/analysis/summary.json`

## Question

The predecessor proved that maximum RED occupancy on one LTS slice explains
the local atomic timing split, but its pair oracle was too expensive to infer a
transferable address map. This milestone asks whether a fixed-allocation,
address-only GA102 slice function can be recovered without feeding profiler
counters into replay.

Nsight Compute's Python report interface reports zero instances for
`lts__t_sectors_op_red.{sum,max,min,avg}` in the retained GA102 reports. This is
consistent with the recent-GPU limitation described by
[Jin et al.](https://people.ece.ubc.ca/aamodt/publications/papers/realgpu-noc.micro2024.pdf):
aggregate L2 counters remain available while per-slice identities are not.
The campaign therefore uses exact aggregate maxima as a collision oracle.

## Two exact live oracles

The weighted probe assigns 105 of 168 CTAs to one anchor and assigns six
candidate addresses binary CTA weights 1, 2, 4, 8, 16, and 32. The anchor is
always the busiest slice, so the observed LTS maximum minus 105 encodes all six
anchor-collision answers. Exhaustive all-pairs discovery over 256 adjacent
128-byte cache lines produces 48 transitive equivalence classes independently
at three allocation prefixes.

The second probe reduces classification to one launch. It executes 49 CTAs:
the anchor performs 48 RED generations and the 48 known-distinct
representatives perform 0 through 47. The maximum is therefore
`32 * (48 + class)`, while the total is exactly 37,632 sectors. A selective
49-word clear kernel prevents high-address reset traffic and is excluded from
the profiled kernel filter.

The official corpus contains:

- 196,096 single-pass Nsight Compute kernel actions in nine reports;
- 97,920 exhaustive pair observations;
- 65,535 dense fit addresses covering every low-16 line-offset value except
  the bit-zero duplicate;
- 32,768 untouched high-address holdout values through region 262,139;
- 4,096 same-allocation wide holdout addresses; and
- exact agreement on 4,096 weighted/encoded and 2,400 wide/encoded overlap
  records.

No sudo command was required.

## Findings

All 48 classes are nearly balanced in dense fit (1,363–1,368 lines per class).
Region bits zero and two do not change the class in the full dense window.
Three independent-looking local parity relations exist in the low window, but
only one remains invariant when high addresses are included.

That collapse is the decisive transfer result. Every ordinary simulator hash,
IPOLY64, and SplitMix64 scores at chance on pair equivalence. A deliberately
overfit 65,536-entry low-bit lookup is exact on fit but obtains only 13.37%
collision recall, 98.16% specificity, 55.76% balanced accuracy, and 0.003%
direct-label accuracy on the strict high-address holdout.

The data is compatible with hidden physical-page placement participating in
the LTS choice. CUDA traces expose virtual addresses, so fitting a table to one
allocation would encode allocator state rather than a portable architecture
rule. This sharpens the older warning in the focused research notes: a virtual
address pattern must not be mistaken for a physical slice hash.

## Simulator decision

Retained:

- arbitrary-count fixed-allocation region mapping;
- the six-way weighted collision oracle;
- the one-launch 48-class encoded oracle; and
- exact LTS occupancy as an evaluation label.

Rejected:

- all existing address-only hash candidates;
- the locally exact low-bit/parity rules;
- a 65,536-entry architecture lookup;
- counter-fed replay correction; and
- activation of cross-SM atomic timing.

No simulator behavior changes in this milestone. The next bounded experiment
uses CUDA virtual-memory management to map one physical allocation at multiple
virtual aliases. If the class follows the allocation rather than the alias,
physical-page placement—not a virtual trace-address hash—is the missing
component. That successor is now complete and confirms the physical domain;
see the [CUDA-VMM alias discriminator](sm86_atomic_vmm_alias_milestone.md).
