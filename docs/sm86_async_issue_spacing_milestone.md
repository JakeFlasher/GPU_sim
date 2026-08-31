# SM86 async-copy issue-spacing milestone

**Status:** complete 2026-08-30; pipelined per-copy candidate falsified, no
timing promotion

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti

**Evidence root:** `results/captures/sm86_async_issue_spacing_20260830_v1/`

**Machine report:** `analysis/report.json`

## Outcome

The issue-spacing campaign is a completed no-go. None of the predeclared
pipeline initiation candidates `Q={1,2,4,8,16}` is identified by the training
slice, passes its absolute training bounds, and transfers across the held-out
spacing/topology/depth comparisons. Schema 16 therefore reports the five
candidates only as observations. Production wait readiness, modeled time, and
the typed `async_copy_shared_visibility` tail remain unchanged.

This result closes the schema-15 proposal of a single pipelined server driven
by each copy's maximum source-completion time. It does not show that bypass
delivery lacks pipelining. The next narrow discriminator is a per-cohort
source-response arrival/service observation; fitting another scalar local
latency to the consumed layouts would not resolve the missing event order.

## Assay construction

`async_issue_spacing` retains the store-free, barrier-free three-clock
measurement shape: pre-issue, post-last-issue, and post-wait. Compile-time
special-register clock samples and retained XOR dependencies are placed
between copies at spacing settings `0/1/2/4/8/16`. The resulting manipulation
changes the dynamic instruction spacing while preserving each route, source
footprint, destination layout, copy count, and measured post-wait interval.
This is an **issue-spacing** assay; it does not directly observe the individual
live source-response arrival times.

The direct matrix contains L2 compact/conflict/skew/severe at depth 1 with
spacing 0, depth 2 with all six spacings, and depth 8 with spacings 0/1/4/8.
The cache-all positive control contains compact/conflict at depth 4 and
spacings 0/1/4/8. Every cell has nine 8,192-group direct repeats.

Static evidence covers 192 functions and 21,568 decoded instructions with
zero unknown encodings, 26--30 registers, and zero spills. All 192 spacing
contracts pass. The filler has no SB0 asynchronous-copy wait and the only
global result store follows all measured intervals.

## Evidence coverage

| lane | retained evidence |
|---|---:|
| direct live-GPU matrix | 52 cells / 468 executions |
| NCU comparison endpoints | 12 |
| complete marker-scoped NVBit traces | 22 |
| schema-16 deterministic replay twins | 22 cells / 44 outputs |

All 12 NCU endpoints preserve the expected LDGSTS instruction count, global
source-sector count, and destination-wavefront topology across the selected
spacing endpoints. NCU remains an independent comparison target and is not a
simulator input.

## Live spacing response

The table reports median treatment-minus-compact wait time in cycles per
group. `--` means that spacing was not part of that direct slice.

| route / treatment | depth | s0 | s1 | s4 | s8 | s16 |
|---|---:|---:|---:|---:|---:|---:|
| cache-all conflict | 4 | 63.996 | 27.000 | 15.999 | 15.999 | -- |
| L2 conflict | 2 | 11.396 | 11.919 | 12.069 | 12.052 | 12.030 |
| L2 severe | 2 | 13.393 | 13.932 | 14.048 | 14.037 | 14.010 |
| L2 conflict | 8 | 11.700 | 13.512 | 2.782 | 2.797 | -- |
| L2 skew | 8 | 12.979 | 7.854 | 2.421 | 2.414 | -- |
| L2 severe | 8 | 16.602 | 18.444 | 6.775 | 6.790 | -- |

The cache-all control drains strongly as spacing is introduced, reaching the
approximately 16-cycle final-copy residual at spacing 4. L2 depth 2 instead
retains a nearly flat layout residual even at the widest spacing. L2 depth 8
has a distinct transition: its larger zero/one-spacing residual falls sharply
at spacing 4 and then remains stable at spacing 8. Raw scheduler/system pauses,
including one paired negative depth-2 spacing-1 outlier, are retained; the
reported medians are not clipped.

## Schema-16 candidate and compatibility fix

For each bypass copy, schema 16 begins with the existing source/payload arrival
`a_i` and schema-15 topology penalty `P_i`. Copies are stably ordered by
`(a_i, instruction)`. For each candidate `Q`, the timing-neutral recurrence is:

```text
start_i = max(a_i, next_accept)
finish_i = start_i + P_i
next_accept = start_i + ceil(P_i / Q)
```

The group observation is the maximum `finish_i`, with exposure measured from
the time the wait becomes current. These fields do not gate readiness or
consume/clear an async-copy group.

The retained clock reads compile to `S2UR` producers followed by
`DEPBAR.LE SB1..SB5, 0x0`. The SM86 parser now treats these exact forms as
ordinary compiler-scoreboard waits, enforces their existing dependence-barrier
readiness, and never routes them through the SB0 async-copy group protocol.
Other barrier names, nonzero SB1--SB5 thresholds, and unsupported forms remain
fail-closed. The predecessor binary's rejection of the spaced trace is
recorded as structural incompatibility rather than a timing difference.

Six spacing-zero traces remain normalized-equal to their schema-15 predecessor
outputs after removing only the schema-16 observational fields and restoring
the prior schema number. This establishes that the compatibility extension and
new observations do not alter predecessor timing or traffic.

## Candidate decision

The frozen training slice is L2 depth-2 conflict/severe at spacing 1 and 4.
Every Q ties at training MAE 3.008 cycles/group with maximum absolute error
4.068. Thus the training data do not identify Q, and every candidate already
fails the 1.0-MAE/2.0-maximum training gate. The machine report records
`training_q_identified=false` and `selected_q=null`.

| Q | training MAE | held-out MAE | held-out maximum | transfers |
|---:|---:|---:|---:|---|
| 1 | 3.008 | 6.764 | 19.021 | no |
| 2 | 3.008 | 4.135 | 9.607 | no |
| 4 | 3.008 | 3.498 | 5.607 | no |
| 8 | 3.008 | 3.407 | 5.579 | no |
| 16 | 3.008 | 3.407 | 5.579 | no |

Q8/Q16 have the smallest held-out aggregate error, but neither is an
identified or transferable mechanism. Q1's held-out maximum error reaches
19.021 cycles/group. No Q is selected, no candidate is promoted to timing,
and the typed tail remains.

## Reproduction

```bash
make -C workloads/research_probes/async_source_only async_issue_spacing

CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_issue_spacing \
  --groups 8192 --depth 2 --route l2 --layout conflict \
  --spacing-samples 4 --markers off

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/capture_nvbit.py \
  --output /tmp/async-issue-spacing --cell spacing_l2_conflict_d2_s4 \
  --binary workloads/research_probes/async_source_only/async_issue_spacing \
  --mode ncu --gpu 0 --timeout 300 --target-args \
  --groups 64 --depth 2 --route l2 --layout conflict \
  --spacing-samples 4 --markers on

cargo test --manifest-path sim/Cargo.toml --workspace --lib
cargo build --release --manifest-path sim/Cargo.toml -p sim-run

sim/target/release/sim-run --trace /tmp/async-issue-spacing \
  --control results/captures/sm86_async_issue_spacing_20260830_v1/static/control_bits.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json --mode l1-l2

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_spacing.py \
  --root results/captures/sm86_async_issue_spacing_20260830_v1 \
  --output results/captures/sm86_async_issue_spacing_20260830_v1/analysis/report.json
```

The report status is `pass_issue_spacing_no_timing_promotion`.

## Content identities

Current retained identities, computed from the completed evidence root:

- Probe binary: `dc40ed57520acdb0ec3dc87169faae6e94966f4728a431f90104b5e83e149125`
- Release `sim-run`: `707bb49d48785bfe843d326d77a8cfb9dd2fcea4a15aae68de857be68c2ce4aa`
- Static control sidecar: `1829b7d6cd19b95033d4d0014f7bedc3ebd643da0698e357b228fe2726330d83`
- Machine report: `0a96efc25585197aaef9fe899f0e8821b3fc68a1e6d9b1577aade527c0573786`
