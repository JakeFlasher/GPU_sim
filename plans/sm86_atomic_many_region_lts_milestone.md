# sm_86 many-region atomic/LTS scheduling milestone

**Completed:** 2026-08-20  
**Goal:** determine whether the many-region atomic residual is explained by a
shared LTS frontier and whether its address mapping transfers

## Acceptance

1. Fix the allocation size and report the device-address phase for every run.
2. Hold SASS, CTA shape, generations, request count, and sector count constant.
3. Map every pair in a 12-region window with the direct per-LTS maximum.
4. Test every three-region subset and require a non-overlapping timing
   discriminator before considering a service.
5. Transfer any selected address rule through at least 128 regions and score
   balanced accuracy plus collision recall.
6. Run full NCU, SourceCounters, NVBit, static decode, and deterministic replay
   on representative fast, slow, and transfer cells.

## Outcome

All measurement and integrity criteria pass. Maximum LTS-slice occupancy
perfectly separates 160 fast from 60 slow triplets and yields 0.623% median-
class CLOCK MAPE. The locally exact mapping has zero transfer collision recall,
and all retained ordinary hashes fail. The probe and causal diagnosis advance;
the timing queue does not.

See the completed
[milestone report](../docs/sm86_atomic_many_region_lts_milestone.md).

