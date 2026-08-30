# LDGSTS barrier-free burst assay

The execution contract for this GPU-only discriminator is tracked at
[`plans/sm86_ldgsts_burst_release_next_discriminator.json`](../../../plans/sm86_ldgsts_burst_release_next_discriminator.json).
It pins the source/binary, enabled-versus-off arms, 40-cell geometry, bounded
correctness/NVBit/NCU/static/sim-run commands, output paths, independent issue
and drain/%smid observables, and explicit pass/falsify/unknown rules. The live
run is retained outside the scored simulator cohort at
`results/captures/sm86_ldgsts_burst_release_20260830_v4/analysis/`.

This prospective `sm_86` probe separates the pre-commit LDGSTS issue interval
from the post-issue completion interval. It crosses depths around the published
five-position boundary, but does not assume that software async groups expose a
physical queue or that a dependency latency releases an entry.

## Current boundary and live result

The target, campaign, strict offline reducer, and streaming live adapter are
implemented. `campaign.py ncu` performs the frozen three-repeat trusted-NCU
profiles with application replay and comparison-only metrics; `live_analysis.py`
consumes the retained direct/NCU trees without loading multi-gigabyte JSONL into
memory.
The generated executable is intentionally ignored by the sibling `.gitignore`.
Run the local checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/analyze.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/campaign.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/campaign.py direct --dry-run
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/campaign.py ncu --dry-run
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/ldgsts_burst_release/live_analysis.py \
  --root results/captures/sm86_ldgsts_burst_release_20260830_v4 --force
```

The terminal live observation is a no-cliff issue result: zero-gap LDGSTS
`P/S/K` and all L1-versus-DRAM issue contrasts are flat and repeat-stable,
whereas the independently retained post-issue completion contrasts are large
and route/depth ordered. The independent LDG clock control has `P=4`, `S=4`,
`K=0`, so the required positive clock-assay knee is absent. The run also retains
eight raw achieved-gap scheduling outliers and three matched NCU LTS conservation
deviations; these keep the strict experiment admission closed. The external
1350/10501 MHz lock could not be applied because non-interactive `sudo` requires
a password. The full 16,384-burst NVBit capture hit its bounded timeout before
producing a complete trace. A marker-scoped 64-burst NCU-mode smoke did
produce a complete trace and two byte-identical simulator replays; those are
pipeline-only evidence, not matrix timing results.

The analyzer fixtures check the exact matrix, timing-schema-2 paired rows,
process-level effects, P/S/K/R contrasts, achieved-gap matching, staged
W1/W4/L2 status precedence, and hard threshold boundaries. There is no fitted
timing constant. Raw captures from the earlier post-issue drain encoding are
intentionally incompatible with timing schema 2 and are not silently mixed.

The pure reducer surface is `reduce_pskr()` (or `reduce_slot_pskr()` for one
W4 slot), followed by `classify_identification()`, `classify_placement()`,
`classify_l2()`, and `classify_staged()`. These functions accept only validated
process reductions and explicit admission flags; they do not read files,
invoke NCU, or infer live evidence.

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
dC = (drain_on - issue_on) - (drain_off - issue_off)
```

`drain_*` is the endpoint duration from the same arm start as `issue_*`, so the
post-issue interval is made explicit once. Bursts are reduced to one median per
process and slot before `P=E5-E4`, `S=E6-E5`, `K=S-P`, and `R=Edram-El1` are
formed within matched repetitions. Seven processes are the independent units.
W4 slots remain four separate vectors; neither bursts nor slots are pooled into
extra replicates.

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
bundle publication. NCU profiles use the trusted executable recorded in each
cell's metadata and are comparison-only. Do not run a live stage concurrently with
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
The gapped variants use one inline-PTX target/clock region, so the current
control sidecar reports zero wait masks on all gapped issue, gap, and drain
clocks. The 32-bit `%clock` spin keeps nominal short/long achieved delays in
the frozen windows in normal samples; raw scheduling pauses remain visible in
the retained stream and are never clipped in the scientific record.
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
