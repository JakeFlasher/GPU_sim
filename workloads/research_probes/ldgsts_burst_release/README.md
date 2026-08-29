# LDGSTS barrier-free burst assay

This prospective `sm_86` probe separates the pre-commit LDGSTS issue interval
from the post-issue completion interval. It crosses depths around the published
five-position boundary, but does not assume that software async groups expose a
physical queue or that a dependency latency releases an entry.

## Current boundary

Round 0 provides the fixed target/campaign scaffold and pure analyzer logic.
The generated executable is intentionally ignored by the sibling `.gitignore`.
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
bursts after fixed warmup, and seven raw process repetitions. The zero-gap
enabled arm now uses one eight-position body with seven distinct runtime
predicates; position zero is unpredicated because SM86 exposes seven user
predicate registers, and the host restricts depth to the frozen domain
`{1,4,5,6,8}`. The off arm remains an empty reference, so NCU instruction
counts are interpreted for the enabled target work rather than as a claim that
two eight-instruction bodies execute.

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
The direct scheduler records balanced forward/reverse requests. For the route
bundle, reverse means DRAM→L1 followed by the L2 holdout; for placement, it
reverses within the W1 block and within the W4 block so the W1→W4 rendezvous is
preserved. This keeps holdout/placement sequencing explicit instead of
pretending that every phase is globally reversed.

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
If a live subprocess fails, its partial raw directory is retained as a
`<cell>.failed*` sibling with a small failure record; only directories with a
complete marker for the current binary mtime/size are resumed as successful
results. Pass `--force` after a rebuild when replacing an older capture is
intentional; no cryptographic hash gate is involved.

The current CUDA 13.2 build has compact and specialized-direct zero-gap
variants with eight contiguous LDGSTS positions, seven distinct depth
predicates, retained issue/drain clocks, and 32 KiB shared storage; the
observed `ptxas` build reports zero spills. In the chosen binary, predicate
setup is before the clock in both compact and direct entries; the static report
records that placement without attributing a causal role to the compiler’s
lane-local algebra. Three predicated-off filler slots appear
before the first target instruction; they are part of the fixed timed body and
are not silently treated as LDGSTS work. Position zero is unpredicated because
SM86 exposes seven user predicate registers; the empty off arm is a separate
paired baseline by design.
The gapped variants compile and pass small correctness smokes, but their
generated control sidecar still shows compiler scheduling wait masks on some
inter-position clock reads; spacing interpretation remains a technical
follow-up rather than a claim from this scaffold.
The report separates the start-clock scheduler mask from the issue/drain
endpoint masks: current zero-gap issue and drain endpoints and all target rows
decode with wait mask zero, while a compact start clock may carry a pre-target
scheduler dependency whose producer is not classified here.
Direct raw records use a streaming-hint global store after each paired burst;
the scaffold does not yet fence that output path, so any live timing reduction
must treat output-store overlap as an explicit confound until it is measured.

## Claim ceiling

Even a fully measured result may describe only a GA102 burst issue/spacing
relation through depth eight for the measured route-dominant and placement
cells. It cannot identify physical queue size, entry release, backpressure
causality, or pipeline stage location; validate an Ampere pipeline; justify a
production timing model; pass P3; or advance Jade M3/M4/G1. Null, unstable, or
mixed evidence is a terminal result rather than permission to fit a model.
