# sm_86 atomic completion and sector-serialization milestone

## Outcome

> Superseded timing note: the later
> [adversarial atomic milestone](sm86_atomic_adversarial_milestone.md) rejects
> the offline 2-cycle/sector ATOMG and 9/8-cycle/sector RED candidate. The exact
> sector projection promoted here remains valid and is extended with exact L2
> request topology and CLOCK-bounded replay diagnostics.

The live campaign validates the existing distinction between returning
`ATOMG` and non-returning `RED`, fixes an atomic address-projection defect, and
rejects a generic same-address timing penalty. Atomic timing service remains
inactive pending the adversarial partial-alias successor.

The promoted behavior is exact transaction formation. On the measured sm_86
GPU, a warp atomic with 32 lanes targeting one word generates 32 serialized
32-byte atomic sector transactions. A warp whose lanes target 32 distinct
words inside one 128-byte line generates four sector transactions. Replay had
collapsed the first case to one unique sector; it now retains same-word alias
multiplicity and reproduces all campaign ATOMG and RED sector counts exactly.

## Experiment

The probe crosses returning `ATOMG` and non-returning `RED` with four layouts:

| Layout | Lane-address relation | L1TEX sectors / warp instruction | L2 requests / warp instruction |
|---|---|---:|---:|
| `lane_line` | one distinct 128-byte line per lane | 32 | 32 |
| `lane_word` | one distinct word per lane in one line | 4 | 1 |
| `warp_word` | all lanes in one warp target one word | 32 | 32 |
| `global_word` | all threads target one word | 1 | 1 |

The last shape is not a primary contention treatment. CUDA 13.2 recognizes its
uniform address and emits warp aggregation (`VOTEU`, `FLO`/`UFLO`, `POPC`, and
one predicated ATOMG/RED). It uses 18 registers and 48 instructions in the
returning case instead of the primary 14-register/40-instruction shape.

Widths 1, 4, 8, and 16 separate single-warp latency, one warp per SMSP, and
latency-hiding/service transfer. Long cubins execute 1,024 generations; trace
cubins execute 64. Final memory sums validate every atomic update. Returning
variants additionally validate the order-independent sum of all pre-increment
values.

The retained result tree is
`results/captures/sm86_atomic_completion_serialization_20260818_v1` and its
machine-readable reduction is `analysis/summary.json`.

## Campaign integrity

- 32 cells;
- 224 long and 224 trace-length correctness executions;
- 96 general NCU profiles and eight W16 SourceCounters profiles;
- 32 NVBit captures containing 101,587 records;
- 32 byte-identical simulator replay pairs;
- 16 spill-free cubins and 656 decoded controls;
- 82 local operand projections with zero gaps; and
- 169 retained sidecars, 49,440 records, and 18,508 projections with zero gaps.

Maximum NCU active-cycle CV is 2.095%; maximum robust MAD is 0.077%. Every NCU
profile has an issue ratio of 1.0, exact ATOMG-versus-RED classification, exact
warp-instruction request counts, and the expected sector, wavefront, and L2
request counts.

## Findings

### Returning completion is real and typed

At W1, returning ATOMG is much slower than RED. SourceCounters attribute the
returning delay to the dependent `IADD3` whose wait mask consumes ATOMG's write
barrier. This agrees with replay's existing typed `MemoryOperation::Atomic`,
L1-bypass/L2 path, response token, and dependence-counter completion. It does
not justify treating every atomic as a returning load: RED remains a distinct,
non-returning reduction role.

### RED exposes service through its read barrier

Sector-heavy RED has no returned register, but SourceCounters place its W16
long-scoreboard samples on the backward `BRA` waiting on RED's read barrier.
LG and MIO throttle are effectively zero. A future timing queue therefore must
drive dynamic read-barrier completion; a generic LSU throttle or a fabricated
write response would encode the wrong causal relation.

### Intra-warp address equality is not an extra scalar penalty

`lane_line` has 32 distinct lines while `warp_word` aliases one word within
each warp. Both generate 32 atomic sectors and are cycle-for-cycle equivalent
through W16 for both ATOMG and RED. The evidence rejects an additional
intra-warp same-word or same-line penalty once transaction formation is held
equal. It does not imply that different warps/CTAs avoid shared-address
contention; the adversarial successor identifies that scope separately.

### Sector service is the surviving timing locus

Current replay has no atomic sector service/backpressure queue, so its
trace-length last-issue timing underpredicts the 18 primary W4/W8/W16 cells by
56.451% MAPE (95.613% maximum). An offline deterministic queue uses W1 as the
per-static-variant latency baseline, fits W4/W8, and holds out W16. The selected
rates are two cycles per returning ATOMG sector, 9/8 cycle per RED sector, and
a 268-cycle return tail. W16 holdout MAPE is 1.715% (7.526% maximum).

That candidate is not active. Alias degrees 2/4/8/16 are unmeasured, the
global-word shape is compiler-transformed, cross-SM sharing is unmeasured, and
hardware CLOCK bounds do not yet match an exact replay interval. Activating the
queue now would turn a strong but incomplete fit into an architecture default.

## Implementation

`sim-trace` now projects address-carrying global atomics onto alias-aware
32-byte transaction sectors. For each touched sector, it repeats that sector by
the maximum exact-address multiplicity among active lanes. This gives:

- distinct lines: 32 distinct sector entries;
- distinct words in one line: four entries;
- one shared word: the same sector repeated 32 times; and
- a compiler-elected single lane: one entry.

The projection happens while exact lane addresses are still available. The
compact validated trace then discards the full lane vectors as before. Exact L2
request count is reconstructed from the compact sector multiplicities, so later
work distinguishes four sectors in one request from four serialized requests.

## Next adversarial experiment

Use runtime-loaded address tables so all variants retain the same SASS and
register allocation. Sweep alias degrees 1, 2, 4, 8, 16, and 32; permute lane
assignments; cover 32- and 64-bit ATOMG/RED; expose replay issue timestamps at
the start and stop CLOCK PCs; and test whether the W4/W8 rates transfer to W16
without layout-specific correction. Only then should the service queue,
dynamic RED read-barrier completion, and ATOMG response tail be considered for
activation.
