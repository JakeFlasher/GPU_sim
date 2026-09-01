# SM86 asynchronous-copy cross-warp queue-overlap milestone

**Status:** complete 2026-08-31 as
`complete_cross_warp_queue_control_no_go`

## Outcome

This milestone implements a timing-neutral schema-20 cross-SMSP LDGSTS issue
observation and runs a fresh exact-work queued-versus-drained live-GPU smoke.
The CUDA manipulation, route attribution, NCU work, NVBit completeness, and
deterministic replay gates all pass. The simulator's cache-all positive control
does not: queued and drained traces reconstruct identical issue-overlap
populations and differ by only `-1.3125` modeled cycles/group, below the
prospectively required 16-cycle magnitude.

The ten-pair identification and W8 holdout were therefore not captured or
read. No simulator time, queue capacity, service rate, architecture card, or
typed-tail state changes. `async_copy_shared_visibility` remains required for
bypass.

## Exact-work live manipulation

One fixed 256-thread CTA uses four active warps and a 32 KiB destination arena
plus bounded timestamp/result state (33,112 static shared bytes). Warp 3 is the
measured target. Each warp owns eight disjoint 512-byte source/destination
positions and executes eight full-warp 16-byte LDGSTS instructions per epoch.

Background warps partition their eight copies into four committed two-copy
groups; the target uses one eight-copy group. Total source sectors,
destination wavefronts, and copied bytes are identical per warp and across the
two arms:

- `queued`: backgrounds signal after commit and retain all four groups until
  the target completes;
- `drained`: backgrounds wait before signaling the target.

Shared timestamps are outside the target clocks. In both L1 and L2 smoke
launches, every queued epoch has all three background intervals overlapping
the target (`192/192` total), while every drained epoch has zero overlap.
Target issue and wait are measured separately.

The retained smoke contrasts, `queued - drained`, are:

| Route | Issue cycles/group | Wait cycles/group |
|---|---:|---:|
| cache-all L1 | +0.046875 | +4.156250 |
| L2-hit bypass | +0.046875 | +31.468750 |

These smoke values establish that the live manipulation is nonvacuous. They
are not a fitted timing model or an authorized full identification label.

## Schema-20 observation

`readiness.async_issue_overlap` is allocated only when the modeled scheduler
selects at least two effective async copies together or an issued copy sees
committed groups owned by another live warp. It reports:

- coissue cycles and copies, split cache-all/bypass;
- mixed-policy cycles;
- structural Q1 excess and maximum issue width; and
- copies seeing other-warp committed groups, with summed/maximum population.

It reads the immutable four-SMSP pick transaction before issue mutation and
never changes selection, requests, async groups, destination service,
readiness, event time, or tails. Checked aggregation and overflow remain
transactional.

Each W4 endpoint reports 512 three-copy coissue cycles, 1,536 coissued copies,
1,024 structural excess copies, and maximum width three. Queued and drained
also both report 1,152 copies seeing other-warp committed groups, summed
population 4,608, and maximum population six. Route splits are exact: all
1,536 copies are cache-all in L1 cells and bypass in L2 cells.

The equality is the decisive result. NVBit retains each warp's dynamic program
order, but the trace contains no global cross-warp issue/service order or
timestamps. Replay independently reschedules the warp streams and reconstructs
the same overlap populations for both live orderings. Schema 20 exposes that
loss; it cannot manufacture the missing order.

## Structural evidence

The smoke runs on device 0, NVIDIA GeForce RTX 3090 Ti, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
610.57.04, CUDA 13.2.51, and Nsight Compute 2026.1.0.0.

Every endpoint NCU report has exact work:

| Metric | Value |
|---|---:|
| LDGSTS warp instructions | 2,048 |
| LDGSTS source sectors | 32,768 |
| shared LDGSTS wavefronts | 8,192 |

Queued/drained metric maps are identical within each route. Cache-all has a
98.4375% L1-hit ratio. Bypass has a 100% L1-miss ratio and 98.4375% attributable
LTS-hit ratio, with zero LTS misses.

The four NVBit traces contain one complete 256-thread CTA and exact reported
instruction counts:

```text
l1 queued    143863 / 143863
l1 drained   124363 / 124363
l2 queued    143890 / 143890
l2 drained   124345 / 124345
```

All eight release replay outputs form byte-identical twins. Cache-all cells
retain no async visibility tail; both bypass cells retain it. A retained
one-warp schema-18 predecessor remains byte-identical after schema 20:

```text
61338a8d446d7fd699d2462ff3817908736323bc294cfcedf94997f0427c555e
```

## Evidence identity

The canonical root is:

```text
results/captures/sm86_async_cross_warp_queue_overlap_20260831_v1/
```

It contains two paired direct smoke runs, four NCU reports, four complete NVBit
traces, and eight deterministic simulator outputs. No full-identification or
W8 holdout path exists. Salient identities are:

```text
CUDA binary   9944a80c075e823f914a9adb1416acc34be6bb30f6e7e1dab2d03849379d2dff
CUDA source   7b37af95e1e4c5e9660e692f8731b17de24b04b6092716b452bfd142528efd1d
sim-run       9f2281f4959250feec6cecbd967b50e92de820d70c9d130570b9673b985d625a
control       ef231b9c47ccdc8449873b51651fb90de88c3b008bd81ca292c040abcf7a26ce
report        31562b8fc2788d8735762387f766be4b53c2f3e95fd964695ec7a54e0d9a3b36
```

The prospective plan began at commit `812f5a2c`; construction-only amendments
were committed before the retained final smoke. The exact implementation and
tracked decision are commit
`d6ee290e0aca02849382b067ad1ec98542b8fa58`.

## Reproduction

From an empty canonical root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cross_warp_queue.py \
  --phase static --timeout 900
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cross_warp_queue.py \
  --phase direct --smoke --timeout 300
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cross_warp_queue.py \
  --phase nvbit --timeout 900
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cross_warp_queue.py \
  --phase replay --timeout 1800
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_cross_warp_queue.py \
  --phase ncu --timeout 900
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_cross_warp_queue.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_cross_warp_queue.py \
  --validate-only
```

## Decision and next boundary

Retain schema 20: it is parameter-free, timing-neutral, and reveals real
cross-SMSP demand that prior readiness output hid. Do not run the full matrix,
open W8, or fit the live smoke residual under this trace contract.

A future cross-warp service-order milestone needs a prospective tracer field
that binds a globally comparable per-SM order/timestamp at selected LDGSTS
issue, commit, and wait events. Until that independent input exists, replaying
per-warp streams cannot validate silicon queue order, regardless of how many
additional queued/drained timing cells are collected.
