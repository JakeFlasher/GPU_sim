# Completed milestone: sm_86 CUDA-VMM alias discriminator

**Status:** complete 2026-08-20; physical address domain established  
**Predecessor:** [GA102 address-to-LTS mapping](sm86_atomic_lts_address_mapping_milestone.md)  
**Evidence:**
`results/captures/sm86_atomic_vmm_alias_20260820_v1/analysis/summary.json`

## Question and API contract

The predecessor found a deterministic but non-transferring mapping from CUDA
device pointers to 48 LTS classes. This milestone asks whether the true atomic
slice identity follows the trace-visible virtual pointer or its translated
physical allocation.

CUDA's official
[Virtual Memory Management guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
states that repeated `cuMemMap` calls can create multiple virtual proxies for
one physical allocation. It also warns that aliases are incoherent during a
writing operation. The probe therefore uses two stages:

1. a read-only alias diagnostic, for which simultaneous proxies are valid; and
2. a sequential RED discriminator that accesses exactly one proxy per kernel
   and synchronizes before switching aliases.

The device reports a 2 MiB minimum VMM granularity. Seven aliases occupy a
contiguous 14 MiB virtual range while mapping the same 2 MiB physical handle.

## Live campaign

- 133 direct correctness executions;
- 121 one-pass cache-flushed read profiles;
- 56 sequential-RED profiles, each containing 55 selected RED launches;
- one 16,432-launch dense allocation-relative RED report;
- two extended relative reports containing 82,015 RED launches;
- seven cross-allocation reports containing 3,024 RED launches;
- 104,672 official NCU kernel actions across 187 reports; and
- exact 37,632-sector conservation in all 3,080 RED actions.

Three read-total outliers are retained and robust medians are used. No sudo
command was required.

## Read-path rejection

The read-only probe cannot serve as a slice identity oracle. Even when all 168
CTAs use one identical pointer, the busiest LTS instance contains only 8.05% of
the observed read sectors. LTS read traffic is replicated or merged across
instances. A matched test with byte-identical virtual pointers and different
physical layouts changes the maximum, but that difference is diagnostic only;
it cannot be interpreted as one address-to-one-slice routing.

## Exact sequential RED result

Each RED report first targets all 48 non-aliased representatives. The decoded
maximum must be exactly `0..47`, establishing an in-report class basis. The
next seven launches target the same physical word through aliases separated by
2 MiB. Each launch completes before the next alias is used.

| physical offset | decoded class | alias observations |
|---:|---:|---:|
| 0 | 5 | 49 |
| 128 | 5 | 49 |
| 512 | 5 | 49 |
| 4,096 | 13 | 49 |
| 65,536 | 42 | 49 |
| 524,288 | 3 | 49 |
| 1,048,576 | 39 | 49 |
| 2,096,128 | 40 | 49 |

All 392 of 392 alias observations agree. Classes are stable over seven
independent reports per physical offset, despite the 14 MiB virtual span, and
change only when the physical offset changes.

## Allocation-relative transfer

The dense successor rotates 16,384 physical offsets across the seven aliases,
covering every 128-byte line in the complete 2 MiB VMM allocation. Its 48-way
collision partition is exactly identical to the independent `cuMemAlloc`
partition over the same relative offsets, up to a one-to-one relabeling: 100%
collision recall and 100% specificity.

This distinction matters. Absolute slice labels depend on hidden physical
placement, but within-allocation collision equivalence is a transferable
function of allocation-relative offset only within one physical granule and its
phase.

The exhaustive 2–8 MiB extension preserves exact partition transfer inside
each of three 2 MiB granules, but each granule needs a different class
permutation. Treating the complete range as one phase gives only 55.51%
collision recall. A size-matched 32 MiB high-address holdout transfers exactly,
showing that allocation layout determines the phase sequence.

## Cross-allocation phase

Seven further reports allocate eight distinct 2 MiB physical handles and
classify the same 48 relative representatives against one common basis. The
eight maps are repeat-stable and all distinct. Their representative coverage is
`[48, 32, 48, 32, 32, 48, 32, 48]`; allocations one through seven have zero
same-offset class matches with allocation zero.

Thus allocation-relative offset is not a universal collision key. Physical
allocation/granule phase can merge pairs that are distinct in another
allocation.

## Two-family phase candidate

All observed granules reduce to two canonical partition families. The four
standard-allocation granules alternate `[0, 1, 0, 1]`; the eight distinct
handles use `[0, 1, 0, 1, 1, 0, 1, 0]`, a balanced four/four split. Phase zero
and one obtain 100% and 75% collision recall on the retained 183-pair corpus,
while both reproduce the maximum-occupancy class of all 220 local triplets.

This two-family model is retained as an evaluator. Its latent physical phase is
not present in the trace, and the observed cross-allocation same-offset class
match rate is zero, not the independent-uniform `1/48` assumption. It is not
activated as simulator state.

## Simulator decision

GA102 atomic LTS identity is a property of translated physical allocation plus
offset, not the virtual pointer alone. NVBit traces do retain allocation IDs
and VA ranges, so they can derive relative offsets; they cannot recover the
absolute cross-allocation slice phase.

No hash or cross-SM atomic timing queue is promoted. Allocation ID and relative
offset are necessary inputs, but the cross-allocation experiment proves they
are insufficient without physical granule phase. The retained alternatives are
explicit physical-page/phase tracing or a separately validated statistical
phase abstraction. Deterministically hashing either absolute trace VAs or
relative offsets alone would model the wrong domain.

One timing-neutral prerequisite is integrated: compact global-atomic records
now preserve their already-validated allocation ID in a route-disjoint word
that was previously unused for atomics. `AllocationRegistry` exposes checked
relative byte/sector offsets. `ValidatedMemoryInstruction` remains 24 bytes or
smaller, cache behavior and replay order are unchanged, and no slice class is
assigned.

The complete Rust workspace, campaign-enabled trace tests, performance-counter
SM tests, 51 research-probe Python tests, formatting, and both campaign
reducers pass. The release simulator was rebuilt and all 5,519 active replay
pairs were regenerated; the final validator reports fresh schema/configuration
identity and byte-identical twins.
