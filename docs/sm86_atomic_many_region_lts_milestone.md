# Completed milestone: sm_86 many-region atomic/LTS scheduling

**Status:** complete 2026-08-20; LTS occupancy identified, timing withheld  
**Predecessor:** [adversarial atomic topology](sm86_atomic_adversarial_milestone.md)  
**Evidence:**
`results/captures/sm86_atomic_many_region_20260820_v1/analysis/README.md`

## Question and controlled probe

The pairwise predecessor established an exact aligned-256-byte RED contention
region but could not explain why many distinct regions have strongly irregular
B84 throughput. This milestone asks whether those regions share a downstream
L2-slice service frontier.

The extended driver holds constant:

- one RED cubin and register allocation;
- 84 one-warp CTAs and 1,024 generations;
- one full-warp same-word RED per CTA/generation; and
- one fixed 32 MiB backing allocation whose actual device address is reported.

Only an explicit runtime region-id list changes. This removes the predecessor's
allocation-size/address-phase confound and supports arbitrary pairs and
triplets without changing SASS.

## Live campaign

- 220 three-region cells and 1,540 correct long executions;
- 183 single-pass NCU pair profiles collecting only LTS RED sectors;
- 74 full NCU profiles after expanding eight multimodal cells to seven repeats;
- four SourceCounters reports;
- 98 trace-length correctness executions;
- 14 NVBit captures containing 825,552 tracer records; and
- 14 byte-identical simulator replay pairs containing 331,632 warp records.

The full-profile GPC label is multimodal: maximum CV is 6.283% and robust MAD
4.562%. The in-kernel seven-trial CLOCK result is the primary timing label for
the mechanism test.

## Exact local relation

NCU expands `lts__t_sectors_op_red.sum` into per-LTS average/minimum/maximum/sum
reductions. Equal pair work therefore reveals whether both regions land on one
LTS slice. In the fixed 12-region window the exact equivalence classes are:

```text
(0, 2) (1, 3) (4, 6) (5, 7) (8, 10) (9, 11)
```

This relation explains all 220 triplets with no classification error:

| maximum regions on one LTS | cells | median CLOCK | class boundary |
|---:|---:|---:|---:|
| 1 | 160 | 956,641 | maximum 958,949 |
| 2 | 60 | 2,204,900 | minimum 2,054,865 |

The clusters do not overlap. An oracle using only their two medians has 0.623%
MAPE and 7.301% maximum error. SourceCounters preserve the predecessor's causal
location: loop-branch long-scoreboard not-issued samples rise from 37,009 in a
slice-distinct sentinel to 69,867 in a same-slice sentinel.

This agrees with the
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
which defines LTS as an L2 cache slice, and with
[Maucher et al.](https://publikationen.bibliothek.kit.edu/1000188141), who place
GPU atomic units at L2. It does not imply that NCU counters may be used as
simulator inputs.

## Hash transfer and simulator decision

The local relation can be described by dropping one low region bit, but anchor
transfer through region 128 finds collisions at offsets 2, 57, and 59. The
local rule scores:

- 100% local balanced accuracy and collision recall;
- 49.1% transfer balanced accuracy; and
- 0% transfer collision recall.

The six existing L2 hash candidates also miss every local positive except the
weak bit-reversal arm, which fails broadly. Raw accuracy is misleading because
transfer contains only two positive collision pairs; balanced accuracy and
collision recall govern the decision.

Retained:

- the fixed-allocation arbitrary-region probe; and
- maximum per-LTS RED-sector occupancy as the causal explanation of the local
  triplet classes.

Rejected:

- region count alone;
- ordinary cache hashes as the GA102 address-to-LTS mapping;
- the locally exact low-bit rule as a transferable hash;
- any counter-fed, per-spacing, or address-table timing correction; and
- activation of a cross-SM atomic timing queue.

Exact hardware LTS occupancy is evaluation evidence, not trace-derived state.
Until a transferable address hash is identified, atomic timing remains typed
incomplete. The full Rust workspace, 63 affected Python tests, predecessor
atomic reducers, static decode, and the 5,519-pair active simulator corpus all
pass. No sudo command was required.
