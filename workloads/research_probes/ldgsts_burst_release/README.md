# LDGSTS barrier-free burst assay

This prospective `sm_86` probe separates the pre-commit LDGSTS issue interval
from the post-issue completion interval. It crosses depths around the published
five-position boundary, but does not assume that software async groups expose a
physical queue or that a dependency latency releases an entry.

## Current boundary

Round 0 provides the fixed target/campaign scaffold and pure analyzer logic.
Only synthetic fixtures are consumed by `analyze.py`; it does not discover a
capture tree, invoke CUDA, open an NCU report, or infer a result. Run the local
checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/analyze.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/campaign.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/campaign.py direct --dry-run
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/campaign.py ncu --dry-run
```

The NCU command is prospective only in Round 0. Its exact metric, application-
replay, cache/clock-control, detailed-instance, preflight, and repeat commands
are available through `ncu --dry-run`; a non-dry `ncu` action exits clearly
until the later static/NCU implementation is added.

The Round-0 analyzer fixtures check the exact matrix, paired-row schema,
process-level effects, achieved-gap matching, and hard threshold boundaries.
NCU work/route/SMSP admission and the staged W1/W4/L2 decision table are
reserved for the later live-analysis phase. There is no fitted timing
constant.

## Frozen matrix and reduction

Every logical cell has enabled and empty-reference off arms, 16,384 measured
bursts after fixed warmup, and seven raw process repetitions. The current
compiler-stable recovery uses compile-time depth-specialized contiguous
prefixes; it does not claim a fixed predicate-ignorant eight-instruction
count until a later source-level construction proves that invariant.

| Role | Cells |
|---|---:|
| W1 L1/DRAM identification, depths 1/4/5/6/8, gap 0 | 10 |
| W1 L1/DRAM spacing, depths 4/5/6, gaps 16/64 | 12 |
| unopened W1 L2 route holdout, depths 1/4/5/6/8 | 5 |
| W4 L1/L2/DRAM placement diagnostic, depths 4/5/6 | 9 |
| independent-LDG clock assay, depths 4/5/6/8 | 4 |

For process `j`, burst `b`, and warp slot `s`, the primary effects are:

```text
dI = issue_on - issue_off
dC = drain_on - drain_off
```

`drain_*` is already the interval from issue-end to drain-end. Bursts are
reduced to one median per process and slot before `P=E5-E4`, `S=E6-E5`,
`K=S-P`, and `R=Edram-El1` are formed within matched repetitions. Seven
processes are the independent units. W4 slots remain four separate vectors;
neither bursts nor slots are pooled into extra replicates.

The frozen predicates are `MAD <= 0.5`, operational equality at absolute
median `<= 1`, positive material response at median `> 2`, and route ordering
at median `> 0`. W1 L1/DRAM alone locks identification. W4 is then a placement
diagnostic and L2 is opened last; neither can relabel W1.

## Live-stage safety

Build and static-dump commands are available. The static action emits an
observational SASS/resource/control-bit report; it is evidence for the next
round, not a permission token or workflow gate. Correctness and direct actions
remain explicit campaign commands, never analyzer side effects. When used,
they run serially on one selected GPU with bounded subprocesses,
selected-device binding, fixed-clock restoration, and atomic complete-only
bundle publication. NCU execution is not implemented in Round 0; only its
dry-run command surface exists. Do not run a live stage concurrently with
another GPU campaign.

The current CUDA 13.2 build has zero-gap compact variants with contiguous
depth-specific LDGSTS prefixes, retained issue/drain clocks, 32 KiB shared
storage, and no spills. The gapped variants compile and pass small correctness
smokes, but their generated control sidecar still shows compiler scheduling
wait masks on some inter-position clock reads; spacing interpretation remains
a technical follow-up rather than a claim from this scaffold.

## Claim ceiling

Even a fully measured result may describe only a GA102 burst issue/spacing
relation through depth eight for the measured route-dominant and placement
cells. It cannot identify physical queue size, entry release, backpressure
causality, or pipeline stage location; validate an Ampere pipeline; justify a
production timing model; pass P3; or advance Jade M3/M4/G1. Null, unstable, or
mixed evidence is a terminal result rather than permission to fit a model.
