# sm_86 CUDA-VMM alias discriminator milestone

**Completed:** 2026-08-20  
**Goal:** distinguish physical-allocation from virtual-pointer control of the
GA102 atomic LTS class

## Acceptance

1. Use one `cuMemCreate` physical handle with at least seven `cuMemMap` aliases.
2. Respect CUDA alias coherence: no kernel may write through multiple proxies.
3. Validate any aggregate read oracle with a one-pointer control.
4. Give every RED report a complete in-report 48-class reference self-check.
5. Test at least eight physical offsets and seven independent reports each.
6. Promote no trace-address hash unless virtual aliases change exact RED class.

## Outcome

The read maximum is rejected because one pointer spreads read sectors across
LTS instances. The sequential RED oracle passes all 3,080 launch checks. All
392 virtual-alias observations preserve physical class, while eight physical
offsets span six classes. A dense 16,384-offset successor then transfers the
complete 2 MiB collision partition exactly between VMM and `cuMemAlloc` up to
class relabeling. Extended evidence shows one exact permutation per 2 MiB
granule, not one per allocation; a whole 8 MiB range retains only 55.51% of
collisions under one phase. Eight distinct handles then produce eight stable
maps, four with only 32 classes for the canonical representatives. Allocation
ID plus relative offset is not universal, so no hash or timing service is
integrated. The resulting two-family phase evaluator is balanced four/four
across handles; its phase arms score 100%/75% pair collision recall and both
classify all 220 local triplets, but the latent phase is unavailable to replay.
The full workspace/feature tests pass and all 5,519 active replay
pairs are fresh and byte-identical after the timing-neutral trace change.

See the completed
[milestone report](../docs/sm86_atomic_vmm_alias_milestone.md).
