# SM86 asynchronous-copy wait-prefix ordering milestone

**Status:** complete 2026-08-31 as
`complete_wait_prefix_identification_no_go`

## Outcome

This milestone adds the first live and simulated Ampere path in the repository
that keeps multiple nonempty `cp.async` groups committed across a partial
`wait_group N`. The existing FIFO mechanism gets the live phase directions
right, but the prospective quantitative and cancellation gates fail before
holdout. No latency, coefficient, queue width, architecture-card value, or
readiness time is changed. `async_copy_shared_visibility` remains required.

The work also closes one concrete simulator bug found by the real replay:
NVCC emits `LDGDEPBAR` with its reserved SB0 write barrier composed with
ordinary SB1--SB5 wait-mask bits. The parser previously rejected that valid
layout even though the scheduler already implements the ordinary scoreboard
wait. SM86 admission now permits those ordinary bits while still rejecting an
SB0 wait bit, a read barrier, or the wrong write barrier.

## Prospective assay

Each 8,192-epoch identification run commits one repeated cache-all `.ca` copy
and one L2-hit bypass `.cg` copy to disjoint compact shared destinations:

- `l2_old`: commit `.cg`, commit `.ca`, `wait_group 1`, then `wait_group 0`;
- `l1_old`: commit `.ca`, commit `.cg`, `wait_group 1`, then `wait_group 0`.

Source and destination identities stay attached to their routes. Only FIFO age
changes. Ordered warp clocks measure issue/commit, the partial wait, one shared
consume of the oldest destination, the draining wait, and the complete window.
The single kernel function contains both runtime-selected paths, so registers,
shared memory, and static control identity are common.

The frozen holdout would commit three groups and use `wait_group 2`; it was not
captured because identification failed.

## Simulator observation

Readiness schema 19 adds a cold, timing-neutral
`readiness.async_wait_prefix` object for programs containing a nonzero wait
immediate. Its `partial` and `draining` classes separately report:

- wait count and immediate sum;
- committed groups before the wait;
- consumed prefix groups and groups remaining; and
- prefix-completion constraint exposure from when the wait became current.

The observation does not feed scheduler readiness, completion time, the
destination service, or typed tails. A retained schema-18 coordinate replay
still hashes byte-for-byte to its predecessor after this change:

```text
61338a8d446d7fd699d2462ff3817908736323bc294cfcedf94997f0427c555e
```

Both new 64-epoch replay cells report 128 groups committed and waited. Each
partial class observes 64 `wait_group 1` instructions, 128 committed groups,
64 consumed groups, and 64 retained groups; each draining class consumes the
remaining 64 groups.

## Smoke evidence

The complete smoke runs on device 0, NVIDIA GeForce RTX 3090 Ti, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
610.57.04, CUDA 13.2.51, and Nsight Compute 2026.1.0.0.

The two NVBit traces have one complete 32-thread CTA each and exact reported
instruction streams. Release `sim-run` consumes each trace twice with
byte-identical outputs and readiness schema 19. The two NCU reports are exact
and paired:

| Metric | Both variants |
|---|---:|
| LDGSTS instructions | 128 |
| LDGSTS source sectors | 4,096 |
| Shared LDGSTS wavefronts | 1,024 |
| Local-device LTS reads | 2,080 |
| LTS read hits | 2,080 |
| LTS read misses | 0 |

Thus each epoch has exactly two effective LDGSTS instructions, 64 source
sectors, and 16 shared wavefronts. NCU remains comparison-only and is not a
simulator input.

## Identification result

Ten pairs were executed in alternating AB/BA order. Every value below is
`l2_old - l1_old`, normalized per epoch:

| Phase | Median cycles/epoch | Repeat sign | Frozen gate | Result |
|---|---:|---:|---:|---|
| Issue and commit | +5.000000 | 10 positive | median absolute ≤ 1 | fail |
| Prefix wait | +187.529724 | 10 positive | median ≥ +64; ≥8 positive | pass |
| Oldest consume | 0.000000 | exact tie | median absolute ≤ 1 | pass |
| Remainder wait | −47.541199 | 10 negative | median ≤ −64; ≥8 negative | fail magnitude |
| Total | +144.988525 | 10 positive | cancellation in ≥8 pairs | fail, 0/10 |

The silicon therefore confirms FIFO age direction but not the frozen
two-phase cancellation model. This is not permission to weaken the issue
gate, lower the 64-cycle threshold, subtract the observed issue delta, or fit
the remainder.

Schema 19 predicts `+205.718750` cycles/epoch for the prefix and
`−112.218750` for the draining wait. The prefix absolute error is 18.189026,
within its 28.129459-cycle tolerance. The remainder error is 64.677551,
outside its 16-cycle tolerance. Directional agreement is therefore not an
absolute transfer.

## Evidence identity

The canonical root is:

```text
results/captures/sm86_async_wait_prefix_order_20260831_v1/
```

It contains 20 full identification runs, two NCU reports, two complete NVBit
traces, and four deterministic simulator outputs. Salient identities are:

```text
CUDA binary   23be48f1175ab92d6354035b37c1f66b1bea41ea10473d7170e4c0d0c7e77560
CUDA source   dd2abaa5363fa672cf310b4227210c21170f81b756076c79ee4e9afcba420ac7
sim-run       09e7138e66a727368501a9c8ee9790b50d2172e2d667b87c69870c50cfb4fce8
control       df3e87a0235b9772ad4ea33a50f77c456e808f255d7909c1e733fc8f831d44af
report        6d68779a03d25565682f18d3a4d65bbc404e48f032bfd6e920afa87053a85bfb
```

The captured binary's exact source is committed by this milestone; the live
runs were based on prospective-plan commit `4c36b672` plus that bounded
implementation diff. The final implementation commit is recorded in the Git
history adjacent to this document.

## Reproduction

From an empty canonical root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_wait_prefix.py \
  --phase smoke --timeout 900

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/capture_wait_prefix.py \
  --phase direct --repeats 10 --timeout 300

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_wait_prefix.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_wait_prefix.py \
  --validate-only
```

The NVBit launcher uses a delegated user scope on this host so its mandatory
cgroup-v2 descendant containment remains active. The analyzer fails closed on
missing or mismatched static, direct, NCU, NVBit, replay, schema-19, tail, or
holdout state.

## Decision

Retain the parser fix and timing-neutral schema-19 observation: both close real
simulator capability gaps and are independently validated. Do not promote a
wait-prefix timing correction or capture the holdout under this plan. The
partial-wait experiment demonstrates that route age affects more than a
simple redistribution between `wait_group 1` and `wait_group 0`; a successor
would need a new independent observable for the uncancelled issue/overlap
component rather than a fitted subtraction.
