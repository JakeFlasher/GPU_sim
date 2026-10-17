# Thesis and GPU-simulator integration study

**Date:** 2026-10-17
**Scope:** active HKUST thesis reconstruction, the SM86/RTX 3090 Ti simulator
checkout, the SM120/RTX 5090 checkout, retained replay/NCU artifacts, and the
research corpus in `/home/jakeshea/GPU_sims_notes`.

## Executive decision

The active thesis is a clean HKUST-formatted three-paper monograph.  Its
technical spine is **Generate** (MEMSYS), **Retain** (VitaBeta), and
**Select/Transfer** (IAPS-ECO).  GPU simulation is currently an appendix-level
observability boundary and is not yet presented as a paper chapter.  The
appropriate update is therefore a fourth technical chapter and a linked
arXiv-style paper, not a replacement of the existing three chapters or a
pooling of their metrics.

The GPU paper can make a defensible contribution claim, but not the historical
claims that were rejected by the SOTA audit:

* native NVIDIA SASS tracing is established by Accel-Sim and related systems;
* event-driven GPU simulation is established by MGPUSim/Akita and NVArchSim;
* modern scheduler/control reconstruction has close prior work in Huerta et
  al.; and
* profiler agreement is not, by itself, a new simulator or a cycle-accuracy
  result.

The defensible claim is a **trace-to-frontier replay contract**: a deterministic
Rust replay stack joins dynamic warp/lane evidence with static SASS control,
models sectored L1/L2 and a typed memory-service envelope, emits separate issue,
completion, observation, and typed-tail frontiers, and evaluates those outputs
against independent NCU observations over complete real-workload corpora.  The
contribution is the integration and explicit evidence boundary, not a claim to
recover proprietary topology.

## 1. Current thesis state

### Active source and structure

The authoritative source is
`../Thesis/thesis-paper-integrated-v3/manuscript/`.  `STATUS.md` records R0--R19
complete, a clean 80/81-page thesis PDF, four review variants, and a final
visual review.  The source hierarchy is:

1. MEMSYS accepted source and Tables 1--9;
2. VitaBeta editable manuscript/assets;
3. current IAPS-ECO editable TCAD revision;
4. thesis-native connective prose and figures.

The clean thesis intentionally excludes paper pagination, submission-only
metadata, visible provenance labels, and inserted paper-PDF pages.  Any GPU
paper integration must preserve this rule: the paper is retypeset as an HKUST
chapter, with a paper-derived block and thesis extensions, rather than pasted as
a facsimile.

### Existing narrative and gap

The current title, abstract, introduction, related-work chapter, synthesis,
conclusion, publication appendix, and GPU boundary appendix all describe a
three-lane thesis.  The introduction already supplies the right vocabulary:
compact object, side information, authority, denominator, and terminal.  Those
terms transfer directly to GPU replay:

| Thesis term | GPU replay instantiation |
|---|---|
| Compact object | Dynamic trace plus static SASS-control sidecar and architecture card |
| Side information | PC, masks, lane sectors, placement, launch geometry, cache/service assumptions |
| Authority | Independent NCU metric for a matching capture/profile directory |
| Terminal | Predicted metric, typed unavailable tail, scope mismatch, or replay error |

The missing link is a technical chapter that explains why an instruction trace
is useful without being a full hardware state capture, how the simulator keeps
NCU outside the replay input, and which metric families are currently strong
enough to report.

## 2. Existing simulator state and evidence

### SM86 / RTX 3090 Ti

The final sequential full replay is
`../final_results/3090ti/full.jsonl.zst`: 356 capture roots, 286 successful
roots, 62 typed control dispositions, 8 replay/input failures, and 12,392
successful launches.  The run used one simulator worker; generated JSONL is
compressed.  The latest accepted H116 metric report uses the same 356-root
scope, 107 exact identity joins, and 106 compatible global-load rows for most
L1 endpoints.

The current scored subset is:

| Metric family | Mean APE | Compared | Treatment |
|---|---:|---:|---|
| Global-load sectors | 0.210% | 106 | report |
| Global-store sectors | 0.000% | 107 | report |
| Global L1 lookup hits | 9.155% | 106 | report |
| Global L1 lookup misses | 9.828% | 106 | report |
| L2 TEX lookup hits | 3.862% | 104 | report |
| L2 TEX lookup misses | 0.000% | 104 | report |
| L2 read-operation lookup hits | 5.436% | 104 | report |
| L2 read-operation lookup misses | 0.000% | 104 | report |
| Shared-load wavefronts | 5.623% | 107 | report |
| Shared-store wavefronts | 2.418% | 107 | report |
| Active IPC (`inst/cycle`) | 14.487% | 107 | report as <15%, not <10% |
| Elapsed IPC (`inst/cycle`) | 12.307% | 107 | report as <15%, future <10% work |

The table deliberately excludes metric families whose current residual is over
the field's 15% publication bar or whose scope is not established.  In
particular, DRAM-read, GPC/SM-cycle, aggregate L2-source-hit, and certain
uniform/atomic tails remain future-work obligations rather than favorable
headline numbers.  The authoritative raw/all-row view remains available in the
machine-readable H116 report; the table above is a publication filter, not a
replacement for that report.

### SM120 / RTX 5090

The latest full artifacts are in `/home/jakeshea/5090_branch/final_results`:

* `dram_disabled/full.jsonl.zst`: 339/339 roots, 11,558 launches, 511.80 s
  external wall;
* `dram_enabled/full.jsonl.zst`: 339/339 roots, 11,558 launches, 512.99 s
  external wall, with the explicitly marked prototype DRAM-bandwidth card.

The profiles are in `../sm120_rtx_5090_capture/profiles`.  Because the NVBit
kernel names and NCU names are not byte-identical, the comparison must join by
capture/profile directory and launch order after trying exact identity.  A
The correct full-only reduction uses the 339-directory profile corpus at
`/home/jakeshea/sm120_rtx_5090_capture/profiles`.  It compares all 11,558
launches after exact identity plus same-directory ordered fallback: 98 are
exact identity and 11,460 are ordered fallback.

The publication-safe SM120 subset is:

The durable reduction is retained at
`../final_results/metric_report_sm120_enabled_full_only_correct/`.

| Metric family | Mean APE | Compared | Treatment |
|---|---:|---:|---|
| Global-load sectors | 0.010% | 11,558 | report |
| Global-store sectors | 0.474% | 11,558 | report |
| Global L1 lookup misses | 6.094% | 11,558 | report |
| L2 TEX lookup misses | 2.297% | 11,558 | report |
| L2 read-operation lookup misses | 0.031% | 11,556 | report |
| Shared-load wavefronts | 7.250% | 11,558 | report |
| Shared-store wavefronts | 0.084% | 11,558 | report |
| Active IPC | 9.455% | 11,558 | report |
| Elapsed IPC | 13.602% | 11,558 | report as <15%, future <10% work |

Global L1 lookup hits, unified cache hit rate, aggregate L2 lookup hits, DRAM
read sectors, and GPC elapsed cycles are not included in the headline table
because their current errors exceed the 15% bar or their physical scope is not
established.  They remain explicitly listed as future work in the paper.

### Runtime evidence

Runtime quantities have different meanings and are never pooled into a single
speedup claim:

| Campaign | Workloads | Total |
|---|---:|---:|
| SM86 bare workload process, no NCU | 286 | 242.530 s |
| SM86 simulator full external wall | 356 capture roots | 509.05 s |
| SM86 simulator driver sum, successful rows | 286 | 344.819 s |
| SM86 NVBit capture artifact estimate | 286 | 2,640.820 s |
| SM86 NCU profiling-command artifact estimate | 286 | 902.797 s |
| SM120 simulator full external wall, DRAM disabled | 339 | 511.80 s |
| SM120 simulator full external wall, DRAM enabled | 339 | 512.99 s |
| SM120 NVBit capture artifact estimate | 339 | 13,689 s |
| SM120 NCU profiling-command artifact estimate | 339 | 33,159 s |

The 509.05 s versus 242.530 s comparison is a real sequential process-wall
comparison on the 3090 Ti.  The 902.797 s and 33,159 s values are NCU command
campaign estimates reconstructed from retained artifact timestamps; they are
not bare GPU execution time.  The paper therefore reports both labels and does
not claim a direct Accel-Sim speedup.  A same-host, same-workload, same-output
Accel-Sim measurement would be required for that claim.

## 3. SOTA and novelty boundary

The notes under `GPU_sims_notes/gpu_simulator_presentation/`,
`gpu_sim_breakthrough_research/`, and `gpu_simulation_frontier_study/` establish
the following positioning:

* **GPGPU-Sim and Accel-Sim:** execution-driven and native-SASS trace-driven
  GPU simulation are established baselines; Accel-Sim is the required public
  comparator for native NVIDIA traces.
* **MGPUSim/Akita and NVArchSim:** event/protocol-driven simulation and
  variable-fidelity modeling are established.  The project must not claim
  “first event-driven GPU simulator.”
* **Huerta et al. (MICRO 2025):** modern NVIDIA scheduler, control-word,
  dependency, register-file, and memory findings are architectural lineage,
  not newly discovered mechanisms here.
* **Sim-FA, GPU-Tile-Sim/GTSim, PipeWeave, and TileSight:** modern alternatives
  raise the empirical bar with specialized asynchronous-pipeline, tile, or
  analytical/learned models.  Their reported errors are not directly pooled
  with this instruction/event-level replay because the abstractions and targets
  differ.
* **Trace-sufficiency and causal replay literature:** Netrace, SynchroTrace,
  CausalSim, Correct Wrong Path Simulation, observational determinacy, and
  endpoint-sufficiency work rule out generic claims that dependencies,
  ambiguity checking, or omitted-event repair are new in isolation.

The paper's contribution sentence is consequently narrow and testable:

> We provide an evidence-bounded, deterministic trace-to-frontier replay stack
> whose dynamic/SASS join, sector-aware hierarchy, typed frontier separation,
> and post-replay NCU comparator are evaluated over complete modern workload
> corpora on Ampere and Blackwell-class targets.

This is a systems integration contribution with a reproducible evaluation
contract.  It does not claim proprietary RTL recovery, universal architecture
support, or a new formal theory of trace sufficiency.

## 4. Figure-generation and writing study

The thesis figure study (R10--R19) and the GPU publication-figure package use
the same rules:

1. one mechanism question per figure;
2. concrete artifacts and numbered transformations rather than generic boxes;
3. fixed gutters and equal panel geometry;
4. axes with quantity and unit, direct legends, and color plus marker/line
   redundancy for grayscale and color-vision accessibility;
5. captions that state population, conditions, what the reader should learn,
   and the measurement boundary;
6. no visible “Source Figure,” “retypeset,” or submission-editorial labels in
   the clean thesis; and
7. SVG/PDF/vector output whenever possible, with source/rights bookkeeping in
   ledgers rather than inside the figure.

The paper reuses the mechanism-first vocabulary from
`gpu_simulator_presentation/publication_figures/` but regenerates its result
figures from current architecture and metric data: contribution stack,
concrete sim-trace/sim-sm/sim-engine/sim-mem/sim-run pipeline, causal timeline,
SOTA matrix, and results dashboard.  The older
15-workload figure is retained as historical evidence only and is not used as
the current accuracy headline.

The writing style follows the existing thesis: define the object, state the
authority and denominator, separate observation from model, report unfavorable
boundaries, and use “supports,” “is consistent with,” and “is unavailable”
rather than claiming silicon truth.  The arXiv paper uses a shorter systems
paper structure, while the thesis chapter adds connective prose and a detailed
validity boundary.

## 5. Thesis integration plan executed in this slice

The new material is organized as:

* `paper/gpu_simulator_arxiv/`: standalone arXiv-format manuscript, references,
  data tables, figures, and build instructions;
* `docs/thesis_gpu_simulator_integration_study.md`: this study and claim ledger;
* `../Thesis/thesis-paper-integrated-v3/manuscript/chapters/gpu-simulator.tex`:
  retypeset GPU paper chapter with thesis extensions;
* thesis metadata, abstract, introduction, related-work, synthesis, conclusion,
  publication, and source-status updates linking the fourth paper; and
* thesis-native vector figures for the workflow and current metric subset.

The thesis remains a coherent monograph: the GPU chapter is not a pooled score
with MEMSYS/VitaBeta/IAPS-ECO.  It is a fourth authority contract that broadens
the dissertation from selective workload/physical evaluation to evidence-bound
architectural replay.

## 6. Open scientific gaps

The current paper should state these plainly:

* DRAM-read traffic and aggregate L2-source-hit semantics require additional
  GCC/maintenance evidence;
* GPC and SM elapsed-cycle denominators remain above the preferred 10% bar on
  at least one target and are not promoted as closed timing accuracy;
* Blackwell carded DRAM values are prototype hypotheses, not locked hardware
  measurements;
* NCU and NVBit totals are campaign-tool times, not bare-kernel execution
  times; and
* no direct same-host Accel-Sim speed benchmark has been run.

These are future-work statements, not exclusions used to manufacture a claim.
The raw compressed replays, profile directories, joins, and unavailable metric
inventory remain the authoritative audit trail.

## 7. SOTA comparison used by the new figures

The figure matrix and paper table use the following local-audit evidence.  The
reported numbers are reproduced only as context because the systems use
different devices, workloads, abstraction levels, and denominators.

| System/family | Abstraction and scope | Reported result in the local audit | Comparison to this replay stack |
|---|---|---|---|
| Accel-Sim | Native NVIDIA SASS plus detailed model | Validated public framework | Closest workflow baseline; our claim is not “first SASS replay” |
| MGPUSim/Akita | Event/protocol GPU components | ISCA validation and active AMD ecosystem | Event-driven precedent; different vendor/trace contract |
| NVArchSim | Industrial event-driven NVIDIA model | Trustworthy variable-fidelity lessons | Event-driven precedent; not a public same-corpus baseline |
| Modern GPU core (MICRO 2025) | SASS/core reconstruction | 13.45% mean MAPE | Modern core-fidelity bar; different cohort and projections |
| Sim-FA | Hopper asynchronous pipeline | 5.7% MAPE on H800 FA3 | Specialized TMA/WGMMA scope |
| GPU-Tile-Sim/GTSim | Warp-level tile graph | 1.22–8.71% on A100/H100 | Coarser tile abstraction and different workload set |
| PipeWeave | Analytical plus learned interactions | 6.1% kernel; 8.5% end-to-end | Prediction model, not causal per-sector replay |
| TileSight | First-principles tile/core/cache/network | 12.35% single-GPU; 13.52% serving | Tile/network abstraction and serving scope |
| LPGSim | Compressed SASS/local clocks | 21.4–23.3% cycle error; 9.97× single-thread Accel-Sim speedup | Relaxed/local-clock policy; no same-host comparison here |

The project is strongest when it makes the integration boundary explicit: the
dynamic/SASS identity join, sector-qualified L1/L2 state, four separately named
frontiers, directory-scoped NCU joining, and complete 286/339-root evidence.
The matrix intentionally avoids a radar-score or “SOTA” scalar because such a
score would erase the differences between sampling, analytical prediction,
tile modeling, and native trace replay.
