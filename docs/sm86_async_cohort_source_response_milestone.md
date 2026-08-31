# SM86 async-copy cohort source-response milestone

**Status:** complete 2026-08-31; matched-marginal assay invalid, no timing or
tail promotion

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti

**Evidence root:**
`results/captures/sm86_async_cohort_pairing_20260830_v2/`

**Machine report:** `analysis/report.json`

## Outcome

Schema 17 now retains exact source-sector-to-cohort response membership and a
timing-neutral per-cohort completion candidate. It does not alter production
readiness, modeled event time, or the typed `async_copy_shared_visibility`
tail. The 16 new pairing traces and 22 retained issue-spacing traces replay in
deterministic twins; every retained spacing output normalizes to its schema-16
predecessor after removing only the schema-17 observation.

The live matched-marginal assay is a closed invalid/no-go result. All 72 direct
runs are correct, every median issue delta is zero, and every paired wait
median is negative. The parameter-free cohort candidate instead predicts
`+15.750` cycles/group for all eight comparisons. However, six of eight paired
NCU endpoints violate the frozen DRAM-source tolerance, so the variants cannot
be relabeled as an independently established identical source mix. No timing
or tail change is authorized from either the sign disagreement or the
candidate error.

## Schema-17 observation

The trace projection forms flexible source cohorts in increasing source-address
order. Each cohort spans at most 128 bytes and retains its payload bytes,
destination-line fanout, shared-bank multiplicity, and cohort-local bypass
projection. For every compacted 32-byte source sector, the trace reader stores
an exact `u32` mask of every cohort whose lane transfer overlaps that sector.
One response may therefore update more than one cohort without duplicating or
guessing the source event.

Replay uses the mask to count pending responses and record each cohort's first
and last source-response times. At `wait_group 0`, the observational candidate
is:

```text
P_c = 4 * (bypass_wavefronts_c - 1)
    + 2 * (bank_wavefronts_c - 1)

candidate_c = max(
    ldgsts_completion_ready(last_response_c, payload_work_c),
    first_response_c + P_c
)

group_candidate = max_c(candidate_c)
```

Schema 17 reports cohort counts, response memberships and spans, topology
penalty, candidate completion/exposure, and signed distances from the existing
per-copy, group-aggregate, and production frontiers. These are observations
only: the group still completes through the existing production mechanism and
retains its typed visibility tail.

## Dependence-barrier compatibility

The cohort probe exposed the compiler form
`DEPBAR.LE SB1, 0x0, {3,2}`. The SM86 parser now admits the exact ordinary
forms `DEPBAR.LE SB1..SB5, 0x0[, {n,...}]`, where the optional brace set is one
canonical nonempty set of unique scoreboard ordinals 0--5. Readiness waits for
the pending counter and clear time of every named ordinary scoreboard. Invalid
names, nonzero thresholds, duplicate or malformed sets, and unsupported
layouts remain fail-closed.

SB0 remains a separate protocol. `LDGDEPBAR` commits an async-copy group and
`DEPBAR.LE SB0, N` waits on the permitted committed-group prefix. If that SB0
instruction's encoded control word also contains an ordinary wait mask, the
instruction waits for both frontiers and is released by the later one. The
ordinary brace-set path never commits, consumes, or clears an SB0 async group.

## Assay repair and construction

The v1 precursor mixed lane addresses from separate CUDA allocations. All 16
NVBit endpoints produced nonempty `cross_allocation_lane_span` sidecars, so v1
is retained as `invalid_precursor_cross_allocation_lane_span` and excluded
from final evidence.

The v2 probe places the hot footprint and two independent 128 MiB cold rings
in aligned subranges of one 268,443,648-byte CUDA allocation. Its 1 KiB cold
step and 8,192-group bound avoid ring wrap. Each copy retains four prewarmed
source cohorts, four cold-ring source cohorts, four compact destination
cohorts, and four severe destination cohorts. `slow_expensive` pairs cold
sources with severe destinations; `slow_cheap` swaps the pairing while
preserving the static source vector and destination multiset. Direct repeats
alternate `ab`/`ba` execution order. The one-warp kernel has no CTA barrier,
and all global result stores follow the issue and wait measurement intervals.

Static evidence covers eight functions and 1,368 decoded instructions with
zero unknown encodings, stack use, or spills. The retained source-pool repair
gives all 16 v2 NVBit endpoints one binary identity and empty cross-allocation,
unmapped-range, and unsupported-path sidecars.

## Evidence coverage

| lane | retained evidence |
|---|---:|
| paired direct live-GPU comparisons | 8 |
| correct direct executions | 72 |
| independent NCU endpoints | 16 |
| complete v2 NVBit endpoint traces | 16 |
| schema-17 pairing replay twins | 16 cells / 32 outputs |
| retained spacing schema-17 twins | 22 cells / 44 outputs |

The host identity is pinned to device 0, GPU UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
610.57.04. NCU observations remain comparison targets and are never simulator
inputs.

## Direct live result

The delta is `slow_expensive - slow_cheap` wait time in cycles/group. All nine
repeat deltas in every row are negative; all rows have an exactly zero median
issue delta.

| depth | spacing | median issue delta | median wait delta |
|---:|---:|---:|---:|
| 1 | 0 | 0.000 | -10.322 |
| 1 | 4 | 0.000 | -10.468 |
| 2 | 0 | 0.000 | -5.460 |
| 2 | 4 | 0.000 | -7.150 |
| 4 | 0 | 0.000 | -11.227 |
| 4 | 4 | 0.000 | -10.172 |
| 8 | 0 | 0.000 | -17.694 |
| 8 | 4 | 0.000 | -11.674 |

This reverse sign is a repeat-stable live observation. It is not yet an
identified pairing effect because the independent NCU marginal gate below
fails.

## NCU admission failure

All 16 endpoints have exact LDGSTS instruction count, global source sectors,
and destination wavefront work. Every endpoint also has nonzero L2-hit,
L2-miss, and DRAM-read populations. The frozen paired-source tolerance is
`max(4, ceil(1% * expected source sectors))`. Six of eight pairs fail it due
to the DRAM-read difference:

| depth | spacing | tolerance | paired DRAM difference | admitted |
|---:|---:|---:|---:|---|
| 1 | 0 | 21 | 304 | no |
| 1 | 4 | 21 | 4 | yes |
| 2 | 0 | 41 | 24 | yes |
| 2 | 4 | 41 | 48 | no |
| 4 | 0 | 82 | 164 | no |
| 4 | 4 | 82 | 272 | no |
| 8 | 0 | 164 | 292 | no |
| 8 | 4 | 164 | 69,880 | no |

The exact instruction/source-sector totals establish equal requested work, but
they do not override the unequal observed DRAM service. In particular, the
depth-8 spacing-4 endpoint is not clipped or silently repaired. The assay is
therefore reported as `invalid_cohort_pairing_assay_no_timing_promotion`; its
hot/cold labels are not promoted into simulator timing evidence.

## Candidate and retained-spacing decision

The parameter-free cohort candidate predicts `+15.750` cycles/group for each
pair while the live medians range from `-5.460` to `-17.694`. It has 0/9
correct repeat signs for every comparison, 23.641 cycles/group identification
MAE, and 27.147 holdout MAE. The corresponding maximum absolute errors are
26.072 and 33.444 cycles/group. These numbers document disagreement; the
failed NCU admission gate prevents interpreting them as a clean causal
falsification of source/destination pairing.

Schema 17 was also replayed over the 22 retained issue-spacing traces. On the
four predeclared structural-calibration comparisons the cohort candidate has
1.042 MAE and 2.068 maximum error. On the remaining 11 transfer comparisons it
has 3.568 MAE and 6.979 maximum error. It therefore does not rescue the earlier
uniform-cohort timing proposal, and the retained spacing corpus is not timing
authority for the mixed-cohort assay.

No production wait, modeled completion frontier, or tail flag changes. The
typed `async_copy_shared_visibility` tail remains required.

## Next discriminating question

If a strictly paired recapture confirms the reverse sign, the next mechanism
to discriminate is destination service that starts and overlaps or competes
before the slow source cohort's final response. Such overlap could make
expensive destination work paired with a slow source less exposed at the
group wait, unlike the current candidate that adds topology after source
arrival. This is a hypothesis, not a conclusion from the present assay.

Before implementing that mechanism, the live experiment must first solve
strict variant pairing and DRAM attribution so the independent NCU source
marginals pass their frozen tolerance. Another scalar bypass latency or a
relabeling of the current endpoints would not answer the question.

## Reproduction

```bash
make -C workloads/research_probes/async_source_only async_cohort_pairing

CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_cohort_pairing \
  --groups 8192 --depth 2 --spacing-samples 0 --order ab --markers off

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/capture_nvbit.py \
  --output /tmp/async-cohort-pairing \
  --cell async_cohort_pairing_slow_expensive_d2_s0_g64 \
  --binary workloads/research_probes/async_source_only/async_cohort_pairing \
  --mode ncu --gpu 0 --timeout 300 --target-args \
  --groups 64 --depth 2 --spacing-samples 0 \
  --variant slow_expensive --markers on

cargo test --manifest-path sim/Cargo.toml --workspace --lib
cargo build --release --manifest-path sim/Cargo.toml -p sim-run

sim/target/release/sim-run --trace /tmp/async-cohort-pairing \
  --control results/captures/sm86_async_cohort_pairing_20260830_v2/static/control_bits.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json --mode l1-l2

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_cohort_pairing.py \
  --root results/captures/sm86_async_cohort_pairing_20260830_v2 \
  --output results/captures/sm86_async_cohort_pairing_20260830_v2/analysis/report.json
```

The analyzer exits successfully for this scientifically closed no-go and
reports `invalid_cohort_pairing_assay_no_timing_promotion`, zero missing
artifacts, and zero validation errors. An invalid admission result remains a
no-promotion decision rather than a tool failure.

## Content identities

- Schema-17 simulator commit: `5c568305`
- Probe and reducer commit: `d67a9e3e`
- Scientific no-go exit-status commit: `0579fe6d`
- Probe binary:
  `dded4cb03868f5cc757f86103d3ec45d4c8adf66c580b39ac90e3c4034f94f4f`
- Release `sim-run`:
  `530c2c0a278565030159e7154b7a4e1b25b3f19683e5e915d2b10bf816ed4f79`
- Static control sidecar:
  `74852c2cc4d0e45986b3487ec9fc29e76aa44d9bf2f84400a4394769dea902f4`
- Machine report:
  `36a6fbb53c41ecd017fc05c30ed29db60b5114391d3b56b56704362075c80157`
