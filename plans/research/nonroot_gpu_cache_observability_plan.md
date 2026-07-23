# Counter-Free GPU Cache Observability Research Plan

Date: 2026-05-03

## Purpose

This plan defines the research direction for predicting cache behavior on GPU
cloud/server environments where the process has CUDA access but does not have
admin control over the host GPU driver.

The target deployment environment has these hard constraints:

- No Nsight Compute / NCU profiling on server GPUs.
- No GPU hardware performance counters on server GPUs.
- No privileged CUPTI profiling, range profiling, PM sampling, events, or
  metrics on server GPUs.
- No `nvidia-smi` clock locking, power-limit changes, persistence-mode changes,
  MIG changes, or other host-driver mutation.
- Root inside a container does not count as host-driver privilege.

The local machines are the only labeled environments:

- Local RTX 3090 Ti: NCU is available and may be used for labels.
- Local RTX 5060 laptop: NCU is available and may be used for labels.

Server/cloud GPUs are permanently unlabeled for cache metrics. They can only be
evaluated through allowed timing, tracing, static, and read-only covariate
signals.

## Central Claim

The project should not claim that it measures cache hit ratio on server GPUs.
It should claim:

> We predict NCU-equivalent L1/TEX and L2 sector hit-rate metrics from
> non-root user-space evidence. Prediction accuracy is validated against local
> NCU labels on the 3090 Ti and 5060 laptop, then the same counter-free
> predictor is deployed on unlabeled server GPUs with label-free consistency
> checks.

This reframes the work as a non-privileged GPU cache-observability and metric
distillation method, not as another GPU cache simulator or direct cache-counter
measurement tool.

## Novelty Positioning

The crowded prior-art area is "reuse-distance modeling for GPU caches." That is
not enough for novelty by itself.

Known adjacent prior art includes:

- Nugteren et al. 2014, "A Detailed GPU Cache Model Based on Reuse Distance
  Theory": GPU reuse-distance model with warp/threadblock hierarchy, coalescing,
  associativity, MSHRs, latency, and divergence.
- RDGC 2019: reuse-distance-based GPU cache hierarchy hit-ratio and transaction
  analysis.
- PPT-GPU-Mem 2020: NVBit trace extraction and reuse profiles for L1/L2 GPU
  cache memory modeling.
- MDM 2020: NVBit-collected per-warp memory addresses, memory divergence, DPKI,
  MSHR pressure, request batching, and queueing effects.
- MLCRP 2025: ML-based GPU cache performance modeling with reuse profiles.
- Sieve, STEM, GCL-Sampler, and related sampled GPU simulation work: trace and
  kernel sampling, clustering, and learned kernel similarity.

The defensible novelty is the combination of the following:

1. NCU metric distillation under a deployment environment where server labels
   never exist.
2. Prediction of NCU-equivalent sector hit-rate metrics rather than an abstract
   cache hit ratio.
3. Warp-coalesced, conditional hierarchical reuse distance adapted from HRD.
4. GPU-specific sharing locality adapted from Shalom.
5. Condens-style adaptive NVBit trace acquisition for locality proxies.
6. Timing-only cache-behavior tomography on unlabeled server GPUs.
7. Explicit uncertainty and domain-shift reporting for cloud/server deployment.

## Research Questions

RQ1. Can local NCU L1/TEX and L2 sector hit-rate labels be accurately predicted
from only signals that are also available in non-root cloud environments?

RQ2. Does conditional hierarchical reuse distance improve prediction over flat
reuse-distance profiles and simple coalescing features?

RQ3. Does GPU sharing-locality decomposition improve L2 prediction over
per-warp or per-CTA reuse alone?

RQ4. Can cache-policy and working-set perturbations provide enough timing-only
constraints to retune the model on unlabeled server GPUs?

RQ5. Can uncertainty estimates identify when predictions on server GPUs are
out-of-domain relative to local 3090 Ti and 5060 laptop calibration?

RQ6. Does adding predicted cache behavior improve wall-clock prediction beyond
static SASS, launch metadata, CUDA event timing, and CUPTI Activity timing
alone?

## Allowed Evidence Surface

### Local and Server

These signals are allowed on both local and server GPUs:

- CUDA events for elapsed time.
- CUPTI Activity API records for kernel, concurrent-kernel, memcpy, memset,
  runtime, driver, overhead, device, and context activity where available.
- NVBit SASS instrumentation in separate matched runs.
- Static SASS/PTX from `cuobjdump`, `nvdisasm`, `nvcc`, and `ptxas -v`.
- Inline `%clock64` or `%globaltimer` only as diagnostic phase signals with
  caveats.
- Read-only NVML covariates such as GPU name, driver version, temperature,
  current clocks, power draw, power limits, P-state, and throttle reasons.
- Read-only `nvidia-smi` only as fallback/debug. Prefer NVML for production
  covariates because `nvidia-smi` text output is not a stable API.

### Local Only

NCU is allowed only on local GPUs to produce labels:

- RTX 3090 Ti.
- RTX 5060 laptop.

NCU outputs must be treated as labels for model validation and distillation,
not as part of the server deployment workflow.

### Forbidden in Server Deployment

- NCU / Nsight Compute.
- CUPTI Events and Metrics.
- CUPTI Profiling API.
- CUPTI Range Profiler.
- CUPTI PM Sampling.
- CUPTI SASS Metrics API.
- CUPTI PC Sampling as a required baseline. It may be explored only as an
  optional availability probe, but the core method must not depend on it.
- Nsys GPU Metrics.
- Clock locking or power-limit mutation.
- Any host driver or kernel-module setting change.

## Metric Target

The target is not "cache hit ratio" in the generic sense. The target is:

- NCU-equivalent L1/TEX sector hit rate.
- NCU-equivalent L2 sector hit rate.
- Optional supporting outputs:
  - sector/request pressure,
  - sector miss pressure,
  - coalescing efficiency,
  - memory-divergence class,
  - MSHR-pressure proxy,
  - prediction uncertainty.

Nsight Compute's memory workload model treats a sector as an aligned 32-byte
chunk. L1 and L2 cache lines are four sectors, i.e. 128 bytes. L1 sector hit
rate is not the same as tag hit rate. The model should therefore be trained and
reported in terms of NCU-equivalent sector accounting.

## High-Level Architecture

The system should be organized as a two-level model:

```text
NVBit/static features -> architecture-invariant cache evidence
timing perturbations + microbenchmarks -> device latent parameters
local NCU labels -> metric distillation target
```

At deployment time on server GPUs:

```text
server allowed evidence + local-trained model + timing retuning
    -> predicted NCU-equivalent sector hit rates + uncertainty
```

No server NCU labels are assumed or required.

## Core Method Components

### 1. NCU Metric Distillation

Treat NCU as a local teacher and train a student model using only features that
are available without privileged counters.

Teacher labels:

- L1/TEX sector hit-rate labels.
- L2 sector hit-rate labels.
- Optional memory workload metrics if available locally and stable across NCU
  versions.

Student inputs:

- Static SASS/PTX features.
- NVBit memory-reference features.
- Conditional HRD features.
- GPU sharing-locality features.
- Timing perturbation features.
- Timing-only microbenchmark retuning features.
- Read-only device covariates.

Student outputs:

- predicted L1/TEX sector hit rate,
- predicted L2 sector hit rate,
- confidence interval,
- out-of-domain score.

### 2. Warp-Coalesced Conditional HRD

Use Maeda's hierarchical reuse distance as a conditional hierarchy, not as
independent flat reuse-distance histograms.

The proposed hierarchy:

1. 32-byte sector.
2. 128-byte line/fill group.
3. allocation region or virtual page bucket.
4. optional larger region bucket for streaming/kernels with regular tiling.

The key rule is conditional:

- Profile reuse at the finest granularity first.
- Only when the finer layer sees infinity/cold reuse does the model consult the
  next coarser layer.

This preserves fine-grained sector locality while retaining larger locality
information useful for L2 and allocation-level behavior.

Before HRD, lane addresses must be converted into coalesced warp-level memory
requests:

- apply active/predicate mask,
- group touched 32-byte sectors,
- group sectors into 128-byte line/fill groups,
- record sectors per request,
- record uniform/broadcast/overlap cases,
- record strided/scattered cases.

### 3. Schedule-Marginalized Reuse Ordering

Reuse distance needs an ordered stream. GPU execution order is not uniquely
observable under non-root tracing and can vary by scheduling, residency, and
latency.

Instead of assuming one order, compute several plausible order projections:

- lane-order projection,
- warp-local projection,
- CTA-local projection,
- SM-local phase projection using NVBit CTA/warp/SM identity where available,
- kernel-level L2 projection,
- sequential-kernel allocation projection.

The model should learn from a distribution or feature bundle over these order
projections. The final prediction should include uncertainty when projections
disagree.

### 4. GPU-Shalom Sharing Locality

Adapt Jiang/Shalom conceptually, but do not reuse CPU thread-block-section
fractions.

GPU sharing classes:

- intra-lane or same-lane reuse,
- intra-warp reuse,
- inter-warp same-CTA reuse,
- inter-CTA same-SM reuse,
- inter-CTA cross-SM reuse visible mostly through L2,
- sequential-kernel reuse on persistent allocations,
- private/cold accesses.

For L1/TEX prediction, same-SM and same-CTA locality should matter most. For L2
prediction, inter-CTA and sequential-kernel allocation reuse should matter more.

Required trace metadata:

- kernel identity,
- launch index,
- CTA coordinates,
- warp id,
- SM id where available,
- function-relative SASS PC,
- memory op type,
- lane addresses,
- active/predicate masks,
- allocation/range identity.

### 5. Memory-Divergence And MSHR-Pressure Proxies

MSHR counts and replacement policy are not directly observable in server
deployment. Use proxies:

- unique 32-byte sectors per warp memory instruction,
- unique 128-byte lines per warp memory instruction,
- sectors/request,
- memory divergent loads per kilo instruction,
- outstanding load density by phase,
- PC-level repeated miss-likely streams,
- stride entropy,
- pointer-chase timing sensitivity,
- throughput saturation under increasing resident warps,
- timing response to thrash/working-set sweeps.

These proxies connect HRD hit-rate prediction to wall-clock impact. The same
sector hit rate can produce different wall-clock behavior depending on
divergence, request batching, and queueing.

### 6. Causal Cache Tomography

On server GPUs, no NCU labels exist. Use matched perturbations to constrain
latent cache behavior through timing deltas.

Candidate perturbations:

- `.ca` vs `.cg`: expose L1 sensitivity by bypassing/preventing normal L1 use
  where the compiler/SASS honors the policy.
- `.cs`: expose streaming/pollution behavior.
- `.cv`: use carefully; do not assume universal device-memory bypass semantics
  without local validation.
- L2 prefetch-size hints: `.L2::64B`, `.L2::128B`, `.L2::256B` where supported.
- L2 cache hints and eviction-priority policy where supported.
- L2 persisting access-policy window on compute capability 8.0+ where not
  disabled by MIG/MPS constraints.
- shared-memory/L1 carveout as a driver hint, not a deterministic control.
- working-set size sweeps for capacity knees.
- stride and misalignment sweeps for sector amplification.
- warm/cold paired runs.
- eviction-buffer sweeps.

Important interpretation rules:

- PTX cache operators are performance hints, not semantic controls.
- L2 `hitRatio` is not a measured hit rate. It is a policy probability/hint.
- `discard.global.L2` is destructive and must not be used as a benign eviction
  primitive for live application data.
- Perturbation deltas are constraints on latent cache behavior, not labels.

### 7. Condens-Style Adaptive NVBit Sampling

Full NVBit memory tracing can be very expensive and can perturb cache behavior.
Use adaptive trace acquisition:

1. Start with small trace windows.
2. Build HRD/MRC estimates for sector and line levels.
3. Continue tracing until the chosen locality proxy converges.
4. Stop when max change across selected MRC quantiles is below a threshold for
   several consecutive windows.
5. Report trace convergence confidence.

This adapts Condens from statistical simulation into evidence acquisition.
The convergence guarantee applies only to the chosen HRD/MRC proxy, not to
wall-clock or NCU labels directly.

### 8. Server Domain Adaptation

Server GPUs are unlabeled. Adapt only latent device parameters using allowed
signals:

- CUDA device attributes,
- compute capability,
- static SASS features,
- timing microbenchmark results,
- cache-perturbation deltas,
- read-only NVML covariates,
- observed noise level and thermal/clocks state.

Do not hard-code one architecture. Treat cache geometry, effective capacity,
replacement behavior, L1/shared carveout behavior, and L2 policy behavior as
latent or probed parameters.

The 5060 laptop should be probed at runtime for compute capability and
available features rather than assumed from product name.

## Experimental Design

### Local Labeled Experiments

Use both local devices as labeled domains:

- RTX 3090 Ti.
- RTX 5060 laptop.

Core splits:

1. Train on 3090 Ti, test on 5060.
2. Train on 5060, test on 3090 Ti.
3. Train on both, hold out kernels.
4. Train on both, hold out shapes.
5. Train on both, hold out perturbation families.
6. Train on both, hold out memory-pattern classes.

Local NCU is label-only. The deployment feature set must not include any NCU
metric. The model must be able to run without NCU.

### Server Unlabeled Experiments

On server/cloud GPUs:

- collect CUDA event timing,
- collect CUPTI Activity timing,
- collect NVBit evidence in separate matched runs,
- collect static SASS/PTX,
- collect read-only NVML covariates,
- run microbenchmark retuning,
- run cache-tomography perturbations.

No server NCU accuracy claim should be made because no server labels exist.

Server evaluation should report:

- prediction outputs,
- uncertainty intervals,
- domain-shift score,
- timing consistency,
- perturbation consistency,
- cache-feature contribution to wall-clock prediction,
- stability under repeated runs.

## Validation Metrics

### Local Labeled Metrics

For NCU-equivalent sector hit-rate prediction:

- MAE for L1/TEX sector hit rate.
- RMSE for L1/TEX sector hit rate.
- MAE for L2 sector hit rate.
- RMSE for L2 sector hit rate.
- calibration error for confidence intervals.
- Spearman/Pearson correlation across kernel variants.
- rank accuracy across optimization variants.
- delta-sign accuracy under perturbations.

For wall-clock linkage:

- timing prediction MAE/RMSE,
- improvement over a static-only baseline,
- improvement over a timing-only baseline,
- improvement over flat reuse-distance baseline,
- improvement over non-hierarchical coalescing baseline.

### Server Unlabeled Metrics

For unlabeled deployment:

- perturbation direction consistency,
- capacity-knee alignment with timing sweeps,
- `.ca`/`.cg` L1-sensitivity consistency,
- sector-pressure consistency with stride/misalignment,
- repeated-run stability,
- uncertainty-width response to noisy or out-of-domain settings,
- wall-clock prediction improvement from cache features.

## Baselines

Required baselines:

- Static SASS + launch metadata only.
- Timing-only microbenchmark features only.
- Coalescing features only.
- Flat reuse distance at 32B sector only.
- Flat reuse distance at 128B line only.
- Independent multi-granularity RD without conditional HRD.
- HRD without GPU-Shalom sharing classes.
- HRD without cache-tomography perturbation features.
- Full proposed model.

If feasible, compare conceptually or experimentally to:

- Nugteren-style GPU RD features.
- PPT-GPU-Mem-style reuse profile features.
- MDM-style memory-divergence features.
- ML model without analytical HRD structure.

## Artifacts To Build

### Data Schema

Create explicit artifact schemas for:

- local NCU label sidecar,
- NVBit cache evidence sidecar,
- HRD feature sidecar,
- GPU-Shalom sharing sidecar,
- perturbation-run ledger,
- timing microbenchmark ledger,
- prediction output with uncertainty,
- server deployment report.

### NVBit Enhancements

Needed fields:

- function-relative PC,
- unique function id,
- SASS opcode string,
- parsed memory op class,
- parsed cache policy if visible,
- active mask,
- predicate mask,
- lane address vector,
- CTA coordinates,
- warp id,
- SM id if available,
- launch id,
- phase id,
- allocation/range id.

Cache policy should be a derived parsed static-SASS/PTX feature, not assumed to
be a first-class NVBit output.

### Feature Extractors

Implement:

- warp coalescer,
- sector stream generator,
- conditional HRD profiler,
- memory reference curve generator,
- sharing-locality classifier,
- divergence/MSHR-pressure proxy extractor,
- perturbation feature extractor,
- domain-shift feature extractor.

### Microbenchmarks

Build or standardize:

- pointer chase for latency bands,
- working-set sweep for capacity knees,
- stride/misalignment sweep for sector amplification,
- warm/cold cache pair,
- `.ca`/`.cg`/`.cs` variants,
- L2 persisting-window sweep where supported,
- carveout sensitivity probe,
- bandwidth saturation sweep,
- resident-warp pressure sweep.

## Expected Risks And Mitigations

### Risk: Prior Art Crowding

Mitigation:

- Do not claim generic GPU reuse-distance novelty.
- Emphasize non-root, permanently unlabeled server deployment.
- Emphasize NCU-equivalent sector metric distillation.
- Emphasize cache tomography and uncertainty-aware adaptation.

### Risk: Local Devices Do Not Represent Server GPUs

Mitigation:

- Use 3090 Ti and 5060 as two labeled domains.
- Report cross-device transfer explicitly.
- Use server microbenchmarks to retune latent device parameters.
- Report uncertainty and domain-shift scores.

### Risk: NVBit Perturbs Cache Behavior

Mitigation:

- Separate NVBit trace runs from native timing runs.
- Use adaptive trace sampling.
- Never use NVBit-instrumented wall-clock as application ground truth.
- Join runs offline by kernel signature, launch id, binary hash, and covariates.

### Risk: Cache Hints Are Not Deterministic

Mitigation:

- Treat cache operators and L2 policies as perturbation levers.
- Validate perturbation direction locally against NCU labels.
- Use timing deltas as weak constraints, not labels.

### Risk: Clocks And Thermals Are Uncontrolled

Mitigation:

- Do not lock clocks in the deployment-like path.
- Use repeated runs, medians, quantiles, and robust regression.
- Record read-only NVML covariates.
- Include noise level in uncertainty model.

### Risk: Server Labels Never Exist

Mitigation:

- Be explicit: server evaluation is label-free.
- Validate accuracy locally.
- Validate deployment by metamorphic/perturbation consistency and wall-clock
  prediction improvement.

## Proposed Paper Contribution Wording

Contribution 1:

> A counter-free cache-observability formulation that predicts
> NCU-equivalent L1/TEX and L2 sector hit-rate metrics from signals available
> in non-root GPU cloud environments.

Contribution 2:

> A warp-coalesced conditional HRD model with GPU sharing-locality classes that
> captures sector, line, CTA, SM, L2, and sequential-kernel reuse without
> hardware counters.

Contribution 3:

> A cache-tomography protocol using timing-only cache-policy and working-set
> perturbations to adapt a locally calibrated model to unlabeled server GPUs.

Contribution 4:

> An adaptive NVBit acquisition method that stops tracing when locality proxies
> converge, reducing instrumentation overhead while reporting uncertainty.

Contribution 5:

> A validation methodology that separates local NCU label accuracy from
> server-GPU unlabeled deployment consistency.

## Near-Term Milestones

1. Define exact NCU label set on the local 3090 Ti and 5060 laptop.
2. Freeze the deployment feature contract: no feature may require NCU or
   hardware counters.
3. Implement local NCU label sidecar.
4. Implement NVBit cache evidence extraction or extend the current tracer.
5. Implement warp coalescing and sector stream generation.
6. Implement conditional HRD for sector/line/allocation hierarchy.
7. Implement GPU-Shalom sharing classification.
8. Implement first microbenchmark retuning suite.
9. Run a pilot on 3090 Ti: NCU labels vs counter-free predictions.
10. Run cross-device pilot: train 3090 Ti -> test 5060, and reverse.
11. Run a server unlabeled pilot with perturbation consistency checks.
12. Write a novelty comparison table against Nugteren, RDGC, PPT-GPU-Mem, MDM,
    MLCRP, Sieve/STEM/GCL-Sampler, and cuThermo.

## Source Map

Local primary target papers:

- `references/papers/knowledge_base/inbox/maeda2017.pdf`
- `references/papers/knowledge_base/inbox/jiang2022.pdf`

Local related papers:

- `references/papers/knowledge_base/active/2014__a_detailed_gpu_cache_model_based_on_reuse_distance_theory.pdf`
- `references/papers/knowledge_base/active/2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus.pdf`
- `references/papers/knowledge_base/active/2020__mdm_the_gpu_memory_divergence_model.pdf`
- `references/papers/knowledge_base/active/2015__dissecting_gpu_memory_hierarchy_through_microbenchmarking.pdf`
- `references/papers/knowledge_base/active/2022__demystifying_the_nvidia_ampere_architecture_through_microbenchmarking.pdf`
- `references/papers/knowledge_base/active/2024__benchmarking_and_dissecting_the_nvidia_hopper_gpu_architecture.pdf`
- `references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf`

Useful local knowledge cards:

- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/cupti-activity-api.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- `.claude/knowledge/cuda/cuda-memory-hierarchy.md`
- `.claude/knowledge/best-practices/perf-coalescing-memory.md`
- `.claude/knowledge/ptx/ptx-isa-cache-hints.md`

External references:

- NVIDIA ERR_NVGPUCTRPERM:
  https://developer.nvidia.com/ERR_NVGPUCTRPERM
- CUPTI documentation:
  https://docs.nvidia.com/cupti/
- CUPTI Activity API:
  https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html
- CUDA Runtime API events:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- PTX ISA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Nsight Compute Profiling Guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- NVBit GitHub:
  https://github.com/NVlabs/NVBit
- NVBit research page:
  https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus
- PPT-GPU-Mem:
  https://laro.lanl.gov/esploro/outputs/conferenceProceeding/Fast-accurate-and-scalable-memory-modeling/9916369267303761
- RDGC:
  https://www.cai.sk/ojs/index.php/cai/article/view/2019_2_421
- GCL-Sampler:
  https://www.researchgate.net/publication/401470700_GCL-Sampler_Discovering_Kernel_Similarity_for_Sampled_GPU_Simulation_via_Graph_Contrastive_Learning

