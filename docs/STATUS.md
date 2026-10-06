# Current status

Updated 2026-10-06. The project develops analytical/statistical GPU performance
models and a runnable native SM86 replay path, with NCU as independent ground
truth for the declared measurement protocol.

## Active SM120 campaign — RTX PRO 6000 Workstation (2026-10-06)

The live SM120 target is the cloud NVIDIA RTX PRO 6000 Blackwell Workstation
Edition (`sm_120`, `rtx_pro_6000_ws`). The independent collection workspace is
`/home/jakeshea/sm120_rtx_pro_6000_capture`; cloud-side NCU/NVBit output is
transferred to the local RTX 3090 Ti for replay and simulator development.
The dedicated card is vendor-informed with typed cache/latency/issue gaps; the
historical RTX 5060 laptop lane remains separate and is not a Pro 6000 proxy.
The next bounded step is cloud identity/toolchain preflight followed by one
same-binary correctness → NVBit → full NCU vertical.

## Research scope

Strictly forbidden: pursuing goals or exact proprietary mechanisms beyond normal academic PhD-level research; improve accuracy through analytical or statistical modeling and treat unobservable internals as black boxes.

The alpha branch permanently excludes M5/M6 authority, receipt, audit,
provenance, hashing, sealing, approval, freeze, admission, and validation
workflow artifacts. They must not be restored or required for execution or
commits; only local runtime safeguards needed for meaningful execution remain.
The obsolete M5/M6 and CUTLASS-successor campaign surfaces and dedicated tests
are deleted rather than skipped.

## Latest representative SOL slices — 2026-10-06

Problems084/085/088/090/028, FIB/problems001/005/007/008/009/012, and
Quant/problem014 have representative screens/memcheck, NVBit/control, replay
and five NCU repeats. FIB006 has M=6 L1-L2 evidence plus a larger-row timeout;
L1/003 and L1/037 retain bounded capture timeouts; Quant/problem020 is an
explicit SM120-only disposition. L1/022 now replays all seven TF32 launches in
L1-L2 and flat modes after the exact SM86 HMMA.1688.F32.TF32 capability repair;
the retained INT8 IMMA row now replays after the exact IMMA.16832 capability repair;
five post-fix NCU sets preserve instruction/HMMA counts. Physical tails remain
open.

## Current M5 frontier — 2026-10-06

M5-F is a scoped async observer holdout (5/5 target coverage, point APE
0.0634%, empirical only). M5-H/I reconcile store-tail and async protocol
negatives without hardware completion tokens. M5-J preserves exact fallback for
the real two-warp continuation boundary. M5-K is a negative FIB005 M=80
within-family holdout (0/5 coverage, point APE 0.155%, no refit). M5-L/M
reconcile frontiers and blockers; the post-TF32 L1/037 retry reproduces the
capture timeout before `dynamic_trace.pb`; M5-N4 found no additional local SM86 seam. **Next: M5-N3**, retain the objective open until external-evidence
residuals change; no broad accuracy, physical completion, broad native continuation,
or hardware scheduling claim is made.

## Implemented and remaining work

| Surface | Current state |
|---|---|
| Native replay | SM86 instruction, memory, and event replay; separate frontiers |
| Custom frontends | Bounded functional/resource experiments, separate from native timing |
| Predictive extension | Opt-in resource/service intervals composed into a model-local frontier, cache-topology candidates, NCU candidate projection |
| Calibration | Profile loader and offline fitter exist; v2 requires explicit dependent latency and independent throughput, retains queue-depth/capacity interventions without fitting a queue equation, retains throughput interval endpoints, and records queue/occupancy/contention as explicitly unmodeled dimensions; supplied rates remain uncalibrated |
| Accuracy | R6 has a document-identity-consistent six-family holdout join labeled manifest-historical; live host identity is not independently verified; all 15 native intervals are open, so finite coverage is not applicable and no hardware-accuracy claim is made |
| Reification | Typed region contract and native-boundary comparator preserve only complete exact projections; incomplete canary boundaries fall back to replay |
| Selective coverage | Multi-warp/order and completion-multiplicity guards refuse conservatively; fast-frontier output includes full-trace candidate/reified/uncovered counts |
| Model macro seam | `sim-sm::ModelExecutor` consumes a certified scalar `MacroTransition`, emits explicit effects/pending completions, and resumes detailed descriptors; this is model-local |
| Native refinement | Narrow real-trace scalar continuation dispatch now works for a hand-authored SM86 probe, including a typed front-end/cache snapshot; broader front-end/readiness classes still refuse conservatively and hardware completion/native scheduling remain open |
| Service composition | Typed residual/replacement/independent-post-replay roles and a serial queue projection are explicit; grouped unknown tails keep queue uppers open and replacement admission is validated; full-trace demand is currently independent post-replay, so it does not silently alter the observer endpoint |
| Cache candidates | Warp-local instrumentation order can be observed when decoded identities are complete; exact history additionally requires temporal order, fence scope, line generation, owner/read-from, and hardware completion, while unresolved domains remain lower-only/unknown |
| Design evaluation | Fifteen-workload, six-family calibration/development/held-out NCU join runs post-prediction; v2 target contracts check metric/endpoint/clock/cache/launch/repetition and declared GPU identity plus family split semantics; predictions and manifest-built observations bind trace/control/card/launch identities when available; observations are labeled manifest-historical unless metadata is independently parsed; finite intervals, open refusals, accepted/refused coverage, model-local partial-order ties, and model-only step reduction are reported, while native front-end traces remain exact/refused |
| Payload/value work | Bounded model-local identity, ordering, and conservation checks |

## M4 / C7-S33BB closure — 2026-10-04

The remaining declared research-probe population is reconciled in
`docs/sm86_c7_research_probe_inventory_v1.json`, whose status is
`48_live_probe_stages_reconciled_m4c_verticals_complete`. Seven previously
uncaptured Makefile families and five source/preserved-binary families now have
same-binary correctness, Compute Sanitizer, NVBit, deterministic replay, and
five independent NCU vertical records. The load-priority and corrected LDGSTS
records are explicit lower-bound falsifiers; no timing parameter was fitted.
The LDGSTS process-start timeout and wrong-launch NCU attempts remain retained.
Python-only/host-only wrappers have explicit non-native dispositions rather
than synthetic native rows. M4 is complete; M5 remains open for completion-tail
semantics, calibration-only accuracy, and a real native continuation
differential.

The independent M4-E closeout recheck repaired three reproducibility gaps
without rerunning retained matrix workloads: it regenerated the missing
barrier/L1TEX declarative manifests (33/33 and 100/100 cell-key crosswalks),
corrected one stale bypass stdout path, and pinned the SFU Makefile to
`/usr/bin/g++-14` for CUDA 13.2 despite the ambient GCC 16. The fresh SFU v2
vertical passed seven correctness trials, full-shape Compute Sanitizer, NVBit,
tail-fact extraction, replay, and five NCU repetitions; its `38099` simulator
lower bound versus `34257` NCU median remains a falsifier, not a fitted timing
result. All 37 Makefile targets return success under the bounded rebuild loop,
all 42 inventory directories resolve, every live result path resolves, and 135
document/schema pairs validate. This confirms M4 population closure while
keeping completion tails and native dispatch open.

C7-S33BD then synchronized the retained build matrix with the current
inventory. Six current Makefile families missing from the old 31-row matrix
received bounded make checks returning zero; the matrix now covers all 37
Makefile probes, with five analysis-only rows and no separate source/preserved
rows. This repairs bookkeeping drift only.

C7-S34A then inventoried the 203 SOL baseline problem directories without
execution: FIB 25, L1 94, L2 82, Quant 2; all have CUDA/host source and
`solution.json`, 201 have an SM86 marker, two do not, and none has a local
prebuilt artifact. This is source inventory only.

C7-S34B source-reviewed one SM86-marked FIB candidate. The existing
`tools/sol_execbench/.venv` supplies Torch 2.12.0+cu130 and SM86 CUDA; explicit
`nvcc -ccbin=/usr/bin/g++-14` produced the extension, BF16 correctness batches
1/2 passed, and Compute Sanitizer reported zero errors. A profiler-delimited
NVBit/control/replay/NCU vertical then passed with 3824 simulator/NCU warp
instructions and a 5374-cycle NCU median; raw tracer units and the enhanced
JIT parser limitation remain explicit. Evidence:
`docs/sm86_c7_sol_candidate_vertical_v1.json`.

C7-S34D repeated the vertical on FIB/problem_002 at hidden size 4096. The
SM86 build/correctness/memcheck smoke, NVBit/control extraction, deterministic
4562-record replay, and five corrected NCU runs passed; NCU median was 7922
cycles with 4562 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib002_vertical_v1.json`.

C7-S34F repeated the custom CUDA vertical on FIB/problem_003 at hidden size
7168. Build/correctness/memcheck, NVBit/control extraction, deterministic
7202-record replay, and five corrected NCU runs passed; NCU median was 10966
cycles with 7202 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib003_vertical_v1.json`.

C7-S34J repeated the custom CUDA vertical on FIB/problem_021 at hidden size
128. Build/correctness/memcheck, NVBit/control extraction, deterministic
154-record replay, and five corrected NCU runs passed; NCU median was 4341
cycles with 154 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib021_vertical_v1.json`.

C7-S34K repeated the custom CUDA vertical on FIB/problem_022 at hidden size
512. Build/correctness/memcheck, NVBit/control extraction, deterministic
562-record replay, and five corrected NCU runs passed; NCU median was 4433
cycles with 562 instructions matching simulator records. Dirty-writeback and
coherence tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib022_vertical_v1.json`.

C7-S34L repeated the custom CUDA vertical on FIB/problem_023 at hidden size
1536. Build/correctness/memcheck, NVBit/control extraction, deterministic
1442-record replay, and five corrected NCU runs passed; NCU median was 6365
cycles with 1442 instructions matching simulator records. Dirty-writeback and
coherence tails remain open, and the enhanced JIT parser limitation plus raw
tracer unit remain explicit. Evidence:
`docs/sm86_c7_sol_candidate_fib023_vertical_v1.json`.

C7-S34M1 completed the source review for FIB/problem_024 RMSNorm at hidden
size 2048. The source contains an SM86 compile marker, a bounded
`main.cpp::run` entry point, explicit BF16/FP32/EPS contracts, and a separate
FP32 reference; the permitted Torch 2.12.0+cu130/GCC14 environment sees the
RTX 3090 Ti at SM86. Evidence:
`docs/sm86_c7_sol_candidate_fib024_source_v1.json`.

C7-S34M2 then passed the fresh SM86 build, batch-1/16 correctness smoke, and
Compute Sanitizer memcheck with zero errors. Evidence:
`docs/sm86_c7_sol_candidate_fib024_smoke_v1.json`. C7-S34M3 then completed
the bounded profiler-delimited capture, independent 152-instruction control
sidecar, deterministic 1762-record replay, and five target NCU repetitions.
The NCU median was 7413 cycles with 1762 instructions matching replay; tails
remain open. Evidence: `docs/sm86_c7_sol_candidate_fib024_vertical_v1.json`.

C7-S34N is now active for source review of FIB/problem_025 RMSNorm at hidden
size 4096. C7-S34N1 verified the SM86 marker, bounded entry point, distinct
EPS `1e-5`/shape contract, independent FP32 reference, and permitted
Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_candidate_fib025_source_v1.json`. C7-S34N2 then passed the
fresh SM86 build, batch-1/16 correctness smoke, and Compute Sanitizer memcheck
with zero errors; evidence is
`docs/sm86_c7_sol_candidate_fib025_smoke_v1.json`. C7-S34N3 then completed
the bounded profiler-delimited capture, independent 152-instruction control
sidecar, deterministic 3042-record replay, and five target NCU repetitions.
The NCU median was 11164 cycles with 3042 instructions matching replay; tails
remain open. Evidence: `docs/sm86_c7_sol_candidate_fib025_vertical_v1.json`.

C7-S34O is now active for source review of FIB/problem_026 RMSNorm at hidden
size 7168, the final listed FIB RMSNorm shape. C7-S34O1 verified the SM86
marker, bounded entry point, 512-thread/shape contract, independent FP32
reference, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_candidate_fib026_source_v1.json`. C7-S34O2 then passed the
fresh SM86 build, batch-1/16 correctness smoke, and Compute Sanitizer memcheck
with zero errors; evidence is
`docs/sm86_c7_sol_candidate_fib026_smoke_v1.json`. C7-S34O3 then completed
the bounded profiler-delimited capture, independent 152-instruction control
sidecar, deterministic 5426-record replay, and five target NCU repetitions.
The NCU median was 10537 cycles with 5426 instructions matching replay; tails
remain open. Evidence: `docs/sm86_c7_sol_candidate_fib026_vertical_v1.json`.

C7-S34P is now active for a distinct unexecuted L1 custom-CUDA family,
`problem_002_vae_conv3x3_groupnorm_silu_residual_fused`. Source review found
that row is actually pure ATen: `kernel.cu` is a stub and the only `__global__`
token is comment text. C7-S34P corrected the census from 73/116 to 72 custom
CUDA / 117 pure ATen and retained the row as an explicit source blocker
without duplicate tracing. Evidence:
`docs/sm86_c7_sol_l1_problem002_source_class_correction_v1.json`. C7-S34Q is
now active for the next L1 row with an actual custom CUDA kernel,
`problem_006_hyena_depthwise_conv1d_split_gate`. C7-S34Q1 verified its two
custom kernels, FP32 shape/oracle contract, and permitted Torch/GCC14 runtime;
evidence is `docs/sm86_c7_sol_l1p006_source_v1.json`. C7-S34Q2 is active for a
fresh build/correctness/memcheck smoke. That smoke passed batch/sequence
shapes `[1,512]` and `[4,2048]` exactly against the independent reference and
Compute Sanitizer reported zero errors; evidence is
`docs/sm86_c7_sol_l1p006_smoke_v1.json`. C7-S34Q3 then completed the bounded
profiler-delimited capture: both kernels were retained, 72 static instructions
per kernel decoded, and replay produced 679936 records total. Five NCU
repetitions selected both target kernels with matching per-kernel instruction
counts; per-kernel lower frontiers were 5242 and 4359 cycles, with tails open.
Evidence: `docs/sm86_c7_sol_l1p006_vertical_v1.json`. C7-S34R1 then verified
the custom RoPE kernel, custom position/frequency input contract, and permitted
Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p011_source_v1.json`. C7-S34R2 then passed custom-input
RoPE shapes `[1,256]` and `[4,1024]` with max BF16 errors `0.00098` and
`0.00391`; Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p011_smoke_v1.json`. C7-S34R3 then completed the bounded
profiler-delimited capture, independent 56-instruction control sidecar,
deterministic 22016-record replay, and five target NCU repetitions. The NCU
median was 4432 cycles with 22016 instructions matching replay; tails remain
open. Evidence: `docs/sm86_c7_sol_l1p011_vertical_v1.json`. C7-S34S1 then
verified the custom cos/sin kernel, FP32/BF16 shape contract, and permitted
Torch/GCC14 runtime; evidence is `docs/sm86_c7_sol_l1p012_source_v1.json`.
C7-S34S2 then passed scaled cos/sin shapes `[1,512]` and `[4,1024]` with max
errors up to `0.0078125`; Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p012_smoke_v1.json`. C7-S34S3 then completed the bounded
profiler-delimited capture, independent 40-instruction control sidecar,
deterministic 61440-record replay, and five target NCU repetitions. The NCU
median was 4172 cycles with 61440 instructions matching replay; tails remain
open. Evidence: `docs/sm86_c7_sol_l1p012_vertical_v1.json`. C7-S34T is now
active for `problem_013_fused_residual_rms_norm_backward`; no timing or NCU
value is transferred from prior rows. C7-S34T1 verified the two custom
backward kernels, hidden-size/shape contract, independent RMSNorm backward
oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p013_source_v1.json`. C7-S34T2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed backward-gradient shapes
`[1,256]` and `[2,512]` against the independent reference, with BF16 max
error `0.0078125` and Compute Sanitizer reporting zero errors; evidence is
`docs/sm86_c7_sol_l1p013_smoke_v1.json`. C7-S34T3 is active for a bounded
profiler-delimited capture; no timing or NCU value is transferred from prior
rows. C7-S34T3 then completed the two-launch capture, independent 552/320
instruction control sidecars, deterministic 2457600-record replay, and five
per-kernel NCU repetitions. The `grad_weight` lower-frontier falsifier is
retained (`113069` modeled versus `34972` NCU median); evidence is
`docs/sm86_c7_sol_l1p013_vertical_v1.json`. C7-S34U is now active for
`problem_016_rope_inverse_frequency_computation`; no timing or NCU value is
transferred from prior rows. C7-S34U1 verified its custom inverse-frequency
kernel, scalar/oracle contract, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p016_source_v1.json`. C7-S34U2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed all seven declared
`rope_theta` values with sub-`1e-7` max FP32 error and Compute Sanitizer
reported zero errors; evidence is
`docs/sm86_c7_sol_l1p016_smoke_v1.json`. C7-S34U3 then completed the bounded
capture, independent 24-instruction sidecar, deterministic 30-record replay,
and five target NCU repetitions. The NCU median was 2995 cycles with 30
instructions matching replay; tails remain open. Evidence:
`docs/sm86_c7_sol_l1p016_vertical_v1.json`. C7-S34V is now active for
`problem_018_fused_rope_with_qk_norm_and_kv_cache_update`. C7-S34V1 verified
the four custom kernels, cache/shape contract, independent custom-input
oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p018_source_v1.json`. C7-S34V2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed the bounded
`[B=1,S=1,cache_len=0]` custom-input case with zero output error and zero
Compute Sanitizer errors; evidence is
`docs/sm86_c7_sol_l1p018_smoke_v1.json`. C7-S34V3 then completed the
four-launch capture, independent 304-instruction sidecar, deterministic
48592-record replay, and five per-launch NCU repetitions; per-launch
instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p018_vertical_v1.json`. C7-S34W is now active for
`problem_019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward`.
C7-S34W1 verified the two custom kernels, FP32 Vision-RoPE backward contract,
independent oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p019_source_v1.json`. C7-S34W2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed Vision-RoPE backward
sequence lengths `[128,449]` against the independent oracle with FP32 errors
below `1e-5`, and Compute Sanitizer reported zero errors; evidence is
`docs/sm86_c7_sol_l1p019_smoke_v1.json`. C7-S34W3 then completed the
three-launch capture, independent 336-instruction sidecar, deterministic
764928-record replay, and five per-launch NCU repetitions; all per-launch
instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p019_vertical_v1.json`. C7-S34X is now active for
`problem_022_vision_mlp_gelu_backward`. C7-S34X1 verified the CUTLASS/Sm80
TensorOp source, MLP/GELU backward contract, independent oracle, and permitted
Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p022_source_v1.json`. C7-S34X2 is active for a fresh
build/correctness/memcheck smoke; replay may retain a CUTLASS-specific
crosswalk blocker rather than silently falling back. The repository CUTLASS
include path was required; build and memcheck passed, but independent FP32
correctness failed materially in TF32 GEMM-derived gradients, so no native
vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1p022_smoke_v1.json`. C7-S34Y is now active for
`problem_024_vision_rotary_position_embedding_generation_backward`. Source
review corrected this row from marker-only custom CUDA to pure ATen: its `.cu`
is a stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem024_source_class_correction_v1.json`. The census
is now 71 custom CUDA / 14 CUTLASS / 118 pure ATen; C7-S34Z is active for
`problem_025_video_latent_gelu_activation`. C7-S34Z1 verified its vectorized
and scalar-tail custom kernels, FP32 GELU contract, independent oracle, and
permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p025_source_v1.json`. C7-S34Z2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed vectorized/tail GELU
shapes `[2,128]` and `[4,541]` with max error `5.60e-6`; Compute Sanitizer
reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p025_smoke_v1.json`. C7-S34Z3 then completed the bounded
vector-kernel capture, independent 88-instruction sidecar, deterministic
368640-record replay, and five target NCU repetitions. The scalar tail kernel
is source-present but unreachable for fixed intermediate size 4096; evidence is
`docs/sm86_c7_sol_l1p025_vertical_v1.json`. C7-S34AA is now active for
`problem_029_mamba_conv1d_with_gating`. Source review corrected this row from
marker-only custom CUDA to pure ATen: its `.cu` is a stub and `__global__` is
comment text. Evidence:
`docs/sm86_c7_sol_l1_problem029_source_class_correction_v1.json`. The census
is now 70 custom CUDA / 14 CUTLASS / 119 pure ATen; C7-S34AB is active for
`problem_033_post_norm_residual`. C7-S34AB1 verified the custom post-norm
RMSNorm kernel, BF16/FP32 shape
contract, independent oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p033_source_v1.json`. C7-S34AB2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed post-norm residual shapes
`[1,128]` and `[2,512]` with max BF16 error `0.00390625`; Compute Sanitizer
reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p033_smoke_v1.json`. C7-S34AB3 is active for a bounded
profiler-delimited capture. That capture completed with an independent
160-instruction sidecar, deterministic 440576-record replay, and five target
NCU repetitions; instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p033_vertical_v1.json`. C7-S34AC is now active for
`problem_034_flux_multi_axis_rope_frequency_computation`. C7-S34AC2 passed all
16 declared workload shapes across three deterministic seeds (48 correctness
rows) at unchanged 1e-5 tolerances; max absolute error was `5.96e-7`, and
Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p034_smoke_v1.json`. C7-S34AC3 then completed the row-0
capture, independent 56-instruction sidecar, deterministic 160517-record
replay, and five target NCU repetitions. The NCU median was 4659 cycles with
160517 instructions matching replay; tails remain open. Evidence:
`docs/sm86_c7_sol_l1p034_vertical_v1.json`. C7-S34AD is now active for
`problem_035_flux_ada_layer_norm_zero_modulation_extraction`. C7-S34AD1
verified the custom AdaLN modulation kernel, 3072/18432 chunk contract,
independent matmul/chunk oracle, and permitted Torch/GCC14 runtime; evidence is
`docs/sm86_c7_sol_l1p035_source_v1.json`. C7-S34AD2 is active for a fresh
build/correctness/memcheck smoke. That smoke passed B=1/16 AdaLN matmul/chunk
correctness at the declared 0.0021 tolerance with max error `7.25e-5`; Compute
Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p035_smoke_v1.json`. C7-S34AD3 is active for a bounded
profiler-delimited capture. That capture completed with an independent
232-instruction sidecar, deterministic 8607744-record replay, and five target
NCU repetitions; instruction counts matched. The typed unsupported-memory-path
count (18432) remains explicit. Evidence:
`docs/sm86_c7_sol_l1p035_vertical_v1.json`. C7-S34AE is now active for
`problem_036_flux_output_norm_projection_chain`. Source/build/memcheck passed,
but the independent FP32 smoke failed the declared tolerance on B=2/S=293
(max error `0.0045166` versus absolute `0.0031`); no native vertical was
admitted. Evidence:
`docs/sm86_c7_sol_l1p036_smoke_v1.json`. C7-S34AF is now active for
`problem_038_flux_multi_head_rmsnorm_qk`.
 C7-S34AF1 verified the custom per-head Q/K RMSNorm kernel and 48×128
contract; evidence is `docs/sm86_c7_sol_l1p038_source_v1.json`. C7-S34AF2
passed all five declared shapes against the independent FP32 reference with
max Q/K error `1.91e-6`; Compute Sanitizer reported zero errors. Evidence is
`docs/sm86_c7_sol_l1p038_smoke_v1.json`. C7-S34AF3 is active for bounded
profiler/control/replay/NCU capture. That capture used the pinned CUDA 13.2 /
GCC14 extension, retained both launches, and produced a 64-target-instruction
sidecar, 5013504 replay instructions, and five per-kernel NCU repetitions;
instruction counts matched. Evidence:
`docs/sm86_c7_sol_l1p038_vertical_v1.json`. C7-S34AG is now active for
`problem_039_adaptive_instance_normalization_2d_backward`. Source review
corrected this row from marker-only custom CUDA to pure ATen: its `.cu` is a
stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem039_source_class_correction_v1.json`. The census
is now 69 custom CUDA / 14 CUTLASS / 120 pure ATen; C7-S34AH is active for
`problem_040_conv2d_residual_block`.
 C7-S34AH1 verified the two custom convolution kernels and 3/32/3 residual
contract; evidence is `docs/sm86_c7_sol_l1p040_source_v1.json`. C7-S34AH2
passed all five declared shapes with max error `0.00042725`; Compute Sanitizer
reported zero errors. Evidence is `docs/sm86_c7_sol_l1p040_smoke_v1.json`. C7-S34AH3 then completed the pinned
two-launch capture, independent 3024-instruction sidecar, deterministic
10155520-record replay, and five per-kernel NCU repetitions; instruction counts
matched. Evidence: `docs/sm86_c7_sol_l1p040_vertical_v1.json`. C7-S34AI is now
active for `problem_041_strided_conv2d_downsampling`. Source review corrected
this row from marker-only custom CUDA to pure ATen: its `.cu` is an explicit
stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem041_source_class_correction_v1.json`. The census
is now 68 custom CUDA / 14 CUTLASS / 121 pure ATen; C7-S34AJ is active for
`problem_042_moe_expert_load_balancing_and_token_capacity_backward`.
 Source review corrected this row from marker-only custom CUDA to pure ATen:
its `.cu` is a stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem042_source_class_correction_v1.json`. The census
is now 67 custom CUDA / 14 CUTLASS / 122 pure ATen; C7-S34AK is active for
`problem_043_mla_fused_qkv_rope_split`.
 Source review corrected this row from marker-only custom CUDA to pure ATen:
its `.cu` is a stub and `__global__` is comment text. Evidence:
`docs/sm86_c7_sol_l1_problem043_source_class_correction_v1.json`. The census
is now 66 custom CUDA / 14 CUTLASS / 123 pure ATen; C7-S34AL is active for
`problem_046_attention_softmax_with_softcapping_and_dropout`.

C7-S34AL completed the next genuine L1 custom-CUDA row. Source review recorded
the BF16 `[B,8,Q,K]` softcap/softmax contract and 16 declared workload shapes;
the pinned CUDA 13.2/GCC14 extension passed all 16 rows across three
deterministic seeds (48 correctness rows), with maximum absolute error
`0.005859375` and zero tolerance violations. Representative Compute
Sanitizer memcheck reported zero errors. A profiler-gated `[1,8,293,293]`
launch then produced 8,159,464 raw tracer instructions, an independent
248-instruction sidecar (232 target instructions plus a 16-instruction inert
CUB helper), deterministic `l1-l2` replay with 2,894,840 records/instructions,
and five NCU repetitions (cycle median `19158`, SM-active median `17492`,
2,894,840 instructions each). Simulator and NCU instruction counts match;
dirty-writeback/coherence tails and zero native macro dispatch remain explicit.
Evidence: `docs/sm86_c7_sol_l1p046_source_v1.json`,
`docs/sm86_c7_sol_l1p046_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p046_vertical_v1.json`. C7-S34AM is now active for
`problem_048_fused_gate_up_projection_with_swiglu`; no timing or NCU value is
transferred from prior rows.

C7-S34AM completed source review and a repaired build/smoke for
`problem_048_fused_gate_up_projection_with_swiglu`. The row is a genuine
custom elementwise kernel chained after two vendor cuBLAS BF16 matmuls. The
first independent smoke exposed that `--use_fast_math`'s float tanh saturated
the GELU tail. A multiply-only diagnostic did not repair the error and was
reverted; the source now calls double libdevice tanh for the narrow saturated
tail while retaining the BF16 activation and product path. The pinned build passed all 16 declared
rows across three deterministic seeds (48 rows), with exact output equality
in this run, and representative Compute Sanitizer reported zero errors.
Two bounded NVBit attempts then failed before trace emission on vendor
`HMNMX2.BF16_V2` static-SASS decoding; the same fatal remained with
`SKIP_ENHANCED_TRACER=1`, so this is not merely an enhanced-parser failure.
No control sidecar, replay, NCU repetition, custom-only substitute, or timing
claim is admitted. Evidence: `docs/sm86_c7_sol_l1p048_source_v1.json`,
`docs/sm86_c7_sol_l1p048_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p048_capture_blocked_v1.json`. C7-S34AN is now active for
`problem_052_altup_hidden_state_collapse_with_magnitude_normalization`.

C7-S34AN corrected the source census for `problem_052` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen/cuBLAS with no custom launcher. The corrected population is now
65 custom CUDA / 14 CUTLASS / 124 pure ATen. No build, trace, replay, or NCU
vertical was admitted for this row. Evidence:
`docs/sm86_c7_sol_l1_problem052_source_class_correction_v1.json`. C7-S34AO is
now active for `problem_053_gaussian_topk_sparse_activation`.

C7-S34AO completed the genuine L1 Gaussian top-k sparse activation row. The
pinned CUDA 13.2/GCC14 build passed all 12 declared shapes across three
deterministic seeds (36 rows; maximum absolute error `0.03125`, with zero
declared-tolerance violations), and representative Compute Sanitizer reported
zero errors. The `[1,512,12288]`, target-sparsity `0.2` capture passed the
profiler-gated NVBit path, independent 488-instruction control sidecar
(472 target plus 16 inert CUB helper), deterministic byte-identical replay,
and five application-replay NCU repetitions. Replay and NCU instruction
counts match (`2354176`); NCU cycle median is `58369`, SM-active median
`56623`; dirty-writeback/coherence tails, host-side `ndtri`, parser limitation,
raw tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p053_source_v1.json`,
`docs/sm86_c7_sol_l1p053_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p053_vertical_v1.json`. C7-S34AP is now active for
`problem_056_mamba_ssm_dt_projection_softplus_clamp_backward`.

C7-S34AP completed the genuine L1 Mamba softplus/clamp backward row. The
pinned CUDA 13.2/GCC14 build passed all 16 declared shapes across three
deterministic seeds (48 rows; maximum `grad_dt` absolute error
`0.0009765625`, with zero declared-tolerance violations), and representative
Compute Sanitizer reported zero errors. The `[2,1571,40]` capture passed the
profiler-gated NVBit path, independent 224-instruction control sidecar
(208 target plus 16 inert CUB helper), deterministic byte-identical replay,
and five application-replay NCU repetitions. Replay and NCU instruction
counts match (`111840`); NCU cycle median is `15654`, SM-active median
`13997`; the raw tracer unit (`310400`), FP32 bias reduction, parser
limitation, dirty-writeback/coherence tails, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_l1p056_source_v1.json`,
`docs/sm86_c7_sol_l1p056_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p056_vertical_v1.json`. C7-S34AQ is now active for
`problem_057_mtp_shifted_embedding_with_dual_rms_norm_fusion`.

C7-S34AQ corrected the source census for `problem_057` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen roll/embedding/RMSNorm/concatenation/cuBLAS with no custom launcher.
The corrected population is now 64 custom CUDA / 14 CUTLASS / 125 pure ATen;
no build, trace, replay, or NCU vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1_problem057_source_class_correction_v1.json`. C7-S34AR is
now active for `problem_058_moe_expert_token_radix_sort_with_prefix_sum`.

C7-S34AR corrected the source census for `problem_058` from marker-only custom
CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates to
stable argsort, bincount, cumsum, and destination copies with no custom
launcher. The corrected population is now 63 custom CUDA / 14 CUTLASS / 126
pure ATen; no build, trace, replay, or NCU vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1_problem058_source_class_correction_v1.json`. C7-S34AS is
now active for `problem_059_moe_group_score_aggregation_and_masking`.

C7-S34AS completed the genuine L1 MoE group-score aggregation/masking row.
The pinned CUDA 13.2/GCC14 build passed all 16 declared token counts across
three deterministic seeds (48 rows), with the declared matching-negative-
infinity exception explicit, and representative Compute Sanitizer reported
zero errors. The checked `[1163,256]` capture passed the profiler-gated NVBit
path, independent 312-instruction control sidecar (296 target plus 16 inert
CUB helper), deterministic byte-identical replay, and five application-replay
NCU repetitions. Replay and NCU instruction counts match (`711756`); NCU
cycle median is `11294`, SM-active median `9582`; the superseded inline-oracle
attempt is retained separately, while the checked official-reference capture
is authoritative. Dirty-writeback/coherence tails, parser limitation, raw
tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p059_source_v1.json`,
`docs/sm86_c7_sol_l1p059_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p059_vertical_v1.json`. C7-S34AT is now active for
`problem_061_tanh_gated_residual_add_backward`.

C7-S34AT completed the genuine L1 tanh-gated residual backward row. The
pinned CUDA 13.2/GCC14 build passed all 16 declared shapes across three
deterministic seeds (48 rows; maximum `grad_hidden_states` absolute error
`0.0625`, with zero declared-tolerance violations), and representative
Compute Sanitizer reported zero errors. The checked capture passed two
profiler-gated custom launches after fusing the identity residual write into
Kernel A and replacing the scalar `gate.to(float)` helper launch with a direct
`item<float>()` read. Independent control extraction decoded 400 instructions
(216 + 168 target plus 16 inert CUB helper), replay was byte-identical on a
repeat, and five application-replay NCU repetitions ran per kernel. Replay and
NCU instruction counts match (`231346` and `355`); cycle medians are `9795`
and `3287`, with SM-active medians `8115` and `1722`. Dirty-writeback/coherence
tails, parser limitation, raw tracer units, and zero native macro dispatch
remain explicit. Evidence: `docs/sm86_c7_sol_l1p061_source_v1.json`,
`docs/sm86_c7_sol_l1p061_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p061_vertical_v1.json`. C7-S34AU is now active for
`problem_062_kv_cache_update_with_rope_backward`.

C7-S34AU corrected the source census for `problem_062` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen index selection, BF16 RoPE algebra, reductions, clone/index-fill, and
destination copies with no custom launcher. The corrected population is now
62 custom CUDA / 14 CUTLASS / 127 pure ATen; no build, trace, replay, or NCU
vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1_problem062_source_class_correction_v1.json`. C7-S34AV is
now active for `problem_064_latent_kv_expansion_with_split`.

C7-S34AV corrected the source census for `problem_064` from marker-only
custom CUDA to pure ATen: `kernel.cu` is a stub and `main.cpp::run` delegates
to ATen RMSNorm, cuBLAS projection, view/transpose/split, and destination
copies with no custom launcher. The corrected population is now 61 custom CUDA
/ 14 CUTLASS / 128 pure ATen; no build, trace, replay, or NCU vertical was
admitted. Evidence:
`docs/sm86_c7_sol_l1_problem064_source_class_correction_v1.json`. C7-S34AW is
now active for `problem_066_masked_softmax_with_attention_dropout_backward`.

C7-S34AW initially retained `problem_066` as a correctness blocker from a
custom-input/all-elements diagnostic. C7-S34BB corrected that disposition:
the actual SOL generator, reference, negative-infinity handling, and
`required_matched_ratio=0.99` criterion pass all 16 declared rows at seed 223
(minimum matched ratio 1.0; maximum absolute error `5.96e-8`). The raw
diagnostic remains retained but is explicitly superseded; no NCU value entered
the audit. Evidence: `docs/sm86_c7_sol_l1p066_correctness_recheck_v1.json` and
the superseded raw record `docs/sm86_c7_sol_l1p066_smoke_v1.json`. C7-S34AX is
now active for `problem_068_fused_rope_application_backward`.

C7-S34BB3 then completed a corrected problem066 native vertical using the
official generator/reference path. One profiler-gated custom launch produced
592 decoded control instructions (576 target + 16 CUB helper), 24,707,072
replay/NCU instructions, byte-identical replay repeat, and five application-
replay NCU repetitions (cycle median `174414`, SM-active median `172696`).
The modeled GPC lower (`173682`) exceeds two measured repeats; that is retained
as a lower-bound falsifier, not fitted or called completion. Evidence:
`docs/sm86_c7_sol_l1p066_vertical_v1.json`. C7-S34BC is now the active
remaining-population/source-execution seam.

C7-S34AX completed the genuine L1 FP32 RoPE backward row. The pinned CUDA
13.2/GCC14 build passed all 14 declared shapes across three deterministic
seeds (48 rows; maximum absolute error `0.0000190735`, with zero declared-
tolerance violations), and representative Compute Sanitizer reported zero
errors. The `[1,32,8,293]` capture passed three profiler-gated custom launches,
independent 768-instruction control extraction (40 + 712 target plus 16 inert
CUB helper), deterministic byte-identical replay, and five application-replay
NCU repetitions per launch. Replay and NCU instruction counts match; cycle
medians are `17355`, `6081`, and `20834`; SM-active medians are `15611`,
`4392`, and `19181`. Dirty-writeback/coherence tails, parser limitation, raw
tracer units, and zero native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p068_source_v1.json`,
`docs/sm86_c7_sol_l1p068_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p068_vertical_v1.json`. C7-S34AY is now active for
`problem_069_rms_norm`.

C7-S34AY completed the genuine L1 1024-thread BF16x2 RMSNorm row. The pinned
CUDA 13.2/GCC14 build passed all 16 declared shapes across three deterministic
seeds (48 rows; maximum absolute error `0.125`, with zero declared-tolerance
violations), and representative Compute Sanitizer reported zero errors. The
`[1,131,8192]` capture passed one profiler-gated launch, independent
192-instruction control extraction (176 target plus 16 inert CUB helper),
deterministic byte-identical replay, and five application-replay NCU
repetitions. Replay and NCU instruction counts match (`1192624`); NCU cycle
median is `17285`, SM-active median `15399`; dirty-writeback/coherence tails,
parser limitation, raw tracer units, and zero native macro dispatch remain
explicit. Evidence: `docs/sm86_c7_sol_l1p069_source_v1.json`,
`docs/sm86_c7_sol_l1p069_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p069_vertical_v1.json`. C7-S34AZ is now active for
`problem_071_kv_cache_update_with_rope`.

C7-S34AZ completed the genuine L1 KV-cache update with RoPE row. The pinned
CUDA 13.2/GCC14 build passed all 16 declared shapes across three deterministic
seeds (48 rows; exact output equality), and representative Compute Sanitizer
reported zero errors. The `[B=2,H=10,new=512,current=0]` capture passed one
profiler-gated launch, independent 152-instruction control extraction (136
target plus 16 inert CUB helper), deterministic byte-identical replay, and
five application-replay NCU repetitions. Replay and NCU instruction counts
match (`4136960`); NCU cycle median is `21576`, SM-active median `19847`;
dirty-writeback/coherence tails, parser limitation, raw tracer units, and zero
native macro dispatch remain explicit. Evidence:
`docs/sm86_c7_sol_l1p068_source_v1.json`,
`docs/sm86_c7_sol_l1p068_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p068_vertical_v1.json`. C7-S34BA is now active for
`problem_073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection`.

C7-S34BA completed the problem073 source/build/correctness bar but retained a
full-chain NVBit blocker. The pinned CUDA 13.2/GCC14 mixed custom-RMSNorm plus
cuBLAS/layout implementation passed all 16 declared rows × 3 seeds using the
actual SOL generator and `required_matched_ratio=0.99`; strict all-element
diagnostics retain small mismatches, and representative memcheck reported zero errors.
The full profiler-gated chain then failed before trace emission on vendor/ATen
`DEPBAR.LE SB0 0x0 {2 1}` decoding. No RMSNorm-only substitute, replay, or
NCU vertical was admitted. Evidence:
`docs/sm86_c7_sol_l1p073_source_v1.json`,
`docs/sm86_c7_sol_l1p073_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p073_capture_blocked_v1.json`. The BA blocker is retained;
BB is complete and BC is now the active remaining-population/source-execution
seam.

C7-S34BC1 source-reviewed problem075 and corrected the prior marker census:
its `.cu` is a stub and `main.cpp` delegates the full grouped-query attention
chain to ATen/cuBLAS; no custom launcher exists. No build or native vertical
was admitted. The corrected population is now 60 custom CUDA / 14 CUTLASS /
129 pure ATen. Evidence:
`docs/sm86_c7_sol_l1_problem075_source_class_correction_v1.json`. C7-S34BD
is now active for `problem_078_group_norm_fusion`.

C7-S34BD1 source-reviewed problem078 as a genuine 32-group FP32 custom
GroupNorm kernel (not an ATen marker). The source contract and 16 declared
shape rows are recorded; build/correctness/memcheck is the next seam. Evidence:
`docs/sm86_c7_sol_l1p078_source_v1.json`.

C7-S34BD completed problem078. All 16 declared shapes × 3 seeds (48 rows)
passed the pinned CUDA 13.2/GCC14 build and independent FP32 reference
(`max_abs_error=2.861e-6`); representative memcheck reported zero errors.
The `[B=4,C=320,H=64,W=64]` capture passed one profiler-gated launch,
independent 448-instruction control extraction (432 target + 16 CUB helper),
byte-identical replay repeat, and five application-replay NCU runs. Replay and
NCU instruction counts match (`5542528`); NCU cycle median is `97482`,
SM-active median `95960`; retirement tails remain open. Evidence:
`docs/sm86_c7_sol_l1p078_source_v1.json`,
`docs/sm86_c7_sol_l1p078_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p078_vertical_v1.json`. C7-S34BE is now active for
`problem_080_adaptive_layernorm_continuous_with_modulation`.

C7-S34BF1 source-reviewed problem081 and corrected the marker census: its
`.cu` is a stub and its full grouped-query attention chain is ATen/cuBLAS, with
no custom launcher. No build or native vertical was admitted. The corrected
population is now 59 custom CUDA / 14 CUTLASS / 130 pure ATen. Evidence:
`docs/sm86_c7_sol_l1_problem081_source_class_correction_v1.json`. C7-S34BG
is now active for `problem_082_qk_norm_scaled_dot_product_attention`.

C7-S34BG1 source-reviewed problem082 and corrected the marker census: its
`.cu` is a stub and the full QK-normalized scaled attention chain is
ATen/cuBLAS with no custom launcher. No build or native vertical was admitted.
The corrected population is now 58 custom CUDA / 14 CUTLASS / 131 pure ATen.
Evidence: `docs/sm86_c7_sol_l1_problem082_source_class_correction_v1.json`.
C7-S34BH is now active for `problem_084_silu_activation_backward`.

C7-S34BE1 source-reviewed problem080 as a genuine two-kernel FP32 adaptive
LayerNorm/modulation path. The contract and 16 declared shape rows are
recorded; build/correctness/memcheck is the next seam. Evidence:
`docs/sm86_c7_sol_l1p080_source_v1.json`.

C7-S34BE completed problem080. All 16 declared shapes × 3 seeds (48 rows)
passed the pinned CUDA 13.2/GCC14 build and independent FP32 reference
(`max_abs_error=0.00036621`); representative memcheck reported zero errors.
The `[B=1,S=128]` capture passed two profiler-gated launches, independent
576-instruction control extraction (216 + 344 target plus 16 inert CUB helper),
byte-identical replay repeat, and five application-replay NCU runs per kernel.
Replay and NCU instruction counts match; cycle medians are `77856` and `9787`,
SM-active medians `76128` and `8109`; retirement tails remain open. Evidence:
`docs/sm86_c7_sol_l1p080_source_v1.json`,
`docs/sm86_c7_sol_l1p080_smoke_v1.json`, and
`docs/sm86_c7_sol_l1p080_vertical_v1.json`. C7-S34BF is now active for
`problem_081_joint_attention_context_projection`.

C7-S34G source-reviewed an L2 pure-ATen/cuBLAS candidate. It has an explicit
SM86/shape contract but delegates to multiple GEMM and elementwise kernels; no
duplicate NVBit attempt was made after the adjacent L1 pure-ATen bounded hang.
The row remains an explicit source blocker with no native replay/NCU claim.
Evidence: `docs/sm86_c7_sol_l2_pure_aten_source_blocked_v1.json`.

C7-S34H then built and sanitizer-checked a CUTLASS TensorOp FIB candidate and
captured a real NVBit/control sidecar. The `l1-l2` replay refused on an
invalid inferred bypass-copy cohort projection; flat-mode replay completed and
five independent NCU runs measured a 73800-cycle median with a retained
four-instruction simulator/NCU crosswalk difference. Async/writeback/coherence
tails remain open. Evidence:
`docs/sm86_c7_sol_candidate_fib004_flat_fallback_v1.json`.

C7-S34I initially classified the remaining 203 SOL rows from source markers;
C7-S34P1 corrected one comment-only `__global__` false positive, leaving 72
custom CUDA, 14 CUTLASS, and 117 pure ATen/cuBLAS-style rows. This census
changes candidate selection only; it does not claim compilation, replayability,
completion, timing, or SOL score. Evidence:
`docs/sm86_c7_sol_source_class_census_v1.json`.

C7-S34E source-reviewed an L1 pure-ATen candidate. It built and passed exact
analytic correctness plus memcheck, but a profiler-delimited NVBit attempt
hung before trace emission and was terminated after a bounded observation. No
replay/NCU vertical was admitted; the multi-kernel/static-module blocker is
retained explicitly. Evidence: `docs/sm86_c7_sol_l1_candidate_blocked_v1.json`.

C1-R-C/C6-M added a focused semantics regression: a structurally valid live
async-copy boundary is refused before scalar macro derivation/mutation. This
strengthens exact fallback; async visibility and hardware completion remain
open.

C3-F is closed only as a boundary-selection phase: the physical completion
upper remains unknown, and C4-A is the selected independent seam. The existing
async-group and destination-layout observer contracts are reconciled; the
endpoint remains model-local rather than physical completion.

C4-B is complete as one controlled observer slice: the dependent SFU warp
observer interval `34205.4..34521.3` covered all five warp-8 target repeats
with `0.150%` point APE, while its simulator lower bound remained separate at
`38126` cycles. C4-C, C4-D, and C4-E retain negative quad transfer, shape, and
block-count intervention results. C4-F's empirical batch feature covered four
of five 210-block target repeats, with one 12-cycle interval miss and
`0.028%` point APE; no designed-extrapolation coverage guarantee is claimed.
C4-G fixes the observer-domain boundary: broad hardware accuracy, semantic
completion, and native dispatch remain open.

C6-A made one fresh real-trace reference-feature attempt with a 10,000-record
bound. It reached the bound without a footer and returned the retained limit
error; this is not labeled complete fallback or a native witness. C6-B
preserves the prior footer-terminated exact fallback; C6-D/E now cover a
narrow default-front-end snapshot transition. C6-F preserves the multi-warp
boundary, and C6-G closes that boundary explicitly. C6-H now closes the
scoped real store/fence negative branch; broader completion remains open. The next implementation seam is C1-R/C3/C6: preserve the store/fence mismatch,
retain exact fallback, and seek independently typed ownership/visibility state
without synthetic promotion. The scoped C3-G/C6-I writeback audit is complete; C3-H/C6-J preserves the visibility exact-fallback boundary.

C6-B closes only the footer-terminated negative branch: a real reference run
has 33,290 typed boundaries, zero complete transitions, and 33,290 exact
fallbacks with zero native dispatch. C6-C now closes a narrow reference-only
positive real-trace dispatch; physical completion is still unknown.

C6-C now closes a narrow reference-only positive real-trace dispatch. The
memory-free scalar cubin produced a complete typed boundary after disabling
the reference-only front end; a separate detailed replay supplied six issue
effects and a terminal barrier/exit state, and the witness run dispatched one
typed macro with zero fallbacks while matching detailed counters, cycles,
frontiers, and SM activity. Readiness diagnostics differ because compressed
execution does not replay every issue-causal diagnostic increment. The same
witness falls back without the typed front-end snapshot; C6-D/E now provide a
narrow positive default-front-end snapshot transition.

C6-D/E now also have a narrow positive default-front-end transition: a typed
instruction-cache/stream exit snapshot enables one dispatch with zero
fallbacks, matching detailed counters, cycles, frontiers, terminal time, and
SM activity. Readiness diagnostics remain separate and differ by design. C6-F preserves the multi-warp boundary; C6-G closes the broader state
decision. C6-H now closes the scoped real store/fence negative branch. The snapshot
transition is not a general cache/front-end proof.
Hardware completion and native hardware scheduling remain open.

C6-H closes a scoped real store/device-fence negative branch. The one-block
SM86 probe passed correctness, NVBit capture, static control extraction, and
deterministic replay; tail extraction observed 17 stores and `MEMBAR.SC.GPU`
/ `ERRBAR` while fence scope and hardware completion remained unknown. The
reference-feature trace terminated with a footer after 34 reconstructed
boundaries (17 pre/post pairs); every boundary was incomplete for
`unmodeled_completion_tails`, exact replay was selected throughout, and native
macro dispatch remained zero. This is not a physical completion upper or a
native scheduling claim. Evidence: `docs/sm86_c6h_store_fence_completion_boundary_v1.json`.

C3-G/C6-I then audited the existing opt-in writeback/coherence service on the
same real probe. The writeback arm modeled completion for 512 dirty sectors
(16,384 bytes) at cycle 11,121; enabling the model-local L1 visibility/
resident-writeback path moved the modeled completion frontier to 11,146 while
the separate SM-active frontier stayed 11,140. `dirty_writeback`/`coherence`
remained unknown in both arms, so the typed boundary and native macro guards
retain exact fallback for incomplete tails. This is model-local service
progress, not a physical completion upper or NCU fit. Evidence:
`docs/sm86_c3g_optin_writeback_coherence_tail_audit_v1.json`.

C3-H/C6-J then ran the same visibility/resident-writeback path in reference
mode. A 256-record bound was retained as a limit failure; the 4096-record
run reached a footer with 1130 boundaries (565 pre/post pairs), including
1024 visibility accesses, two fences, two invalidations, and 32 L2 batches.
All boundaries remained incomplete for `unmodeled_completion_tails`, exact
replay was selected, and native dispatch stayed zero. Fence scope/order are
event observations, not a complete owner/generation/read-from witness.
Evidence: `docs/sm86_c3h_visibility_reference_boundary_v1.json`.

C3-I/C6-K then ran the independent raw-protobuf/static-SASS tail extractor
without simulator or NCU inputs. It confirmed the observed store/fence prefix
but found no owner/read-from, line-generation, global-order, fence-scope, or
hardware-completion relation. The visibility reference stream remains exact
fallback for all 1130 boundaries with zero native dispatch. This closes the
negative witness branch only; independent ownership/generation evidence remains
open. Evidence: `docs/sm86_c3i_owner_generation_negative_v1.json`.

C4-H then ran an independent constant-load dependent-chain observer protocol:
calibration blocks 8/16/32 produced a frozen `13805.3..13848.1` interval
before an untouched 64-block target. All five target repeats missed the
interval (`0/5`, point APE `0.434%`), so no refit or widening was performed.
C4-I synchronizes the domain: the positive SFU slice remains mechanism-local;
constant-load and quad transfers are falsified or partial, and broad accuracy
remains open. Evidence: `docs/sm86_c4i_observer_domain_reconciliation_v1.json`.

C5-A reports the corresponding held-out statistics: the frozen constant-load
interval covered `0/5` target repeats, width was `42.86` cycles, point APE was
`0.434%`, and target stdev was `64.67` cycles. The simulator lower bound
(`62287`) and open writeback/coherence tails remain separate. Broad C5
accuracy is still open. Evidence:
`docs/sm86_c5a_constant_load_holdout_evaluation_v1.json`.

C5-B then evaluated the disjoint independent-load mode-1 family. Calibration
blocks 8/16 produced a frozen `11003..11293` interval before an untouched
32-block target; all five target repeats missed (`0/5`, point APE `2.578%`).
The miss is retained without refit or widening, and simulator frontiers stay
separate. C5-C now synthesizes the positive/negative observer slices; broad
accuracy remains open. Evidence:
`docs/sm86_c5b_constant_load_mode1_reconciliation_v1.json`.

C5-C synthesizes the positive SFU slice with both constant-load transfer
falsifiers while keeping coverage, width, signed error, semantic lower bounds,
and completion status separate. No broad accuracy or completion claim is made.
The narrow scalar actor also gained a pre-mutation detailed-effect-span guard:
truncated or non-monotone supplied effects now refuse, while the model-local
empty-effect recurrence remains compatible.

M5-A through M5-D have now been executed as scoped phases. The matched
control/reduction-tail differential is in
`docs/sm86_m5a_native_tail_differential_launch_cycle_v1.json`: simulator GPC
lower-bound delta `671` cycles versus independent NCU median delta `740`, with
matching instruction counts, deterministic replay, and zero native macro
dispatches. The reduction arm retains reduction-RMW, dirty-writeback, and
coherence tails. The calibration-only async observer holdout is reconciled in
`docs/sm86_m5b_calibration_holdout_reconciliation_v1.json`: its pre-NCU
`37737..38655` interval covers five group-128 repeats and has 0.396% point APE,
but is not a semantic upper. The fresh bounded real-corpus continuation run
(`docs/sm86_m5c_native_continuation_fallback_v1.json`) has 10,032 exact
fallback boundaries, zero complete transitions, and zero native macro
dispatches. Claim synchronization and focused regressions are recorded in
`docs/sm86_m5d_claim_synchronization_v1.json`.

These phases close their scoped bars, not the physical program: C1/C3/C4/C5,
the positive C6 continuation witness, broader C7 coverage, and physical C8
mechanism closeout remain open.
No scored NCU value became a model feature or tuning input.

C8-A is now complete: `docs/sm86_c8a_interface_reconciliation_v1.json`
records clean active `sim-run`/workload-runner help surfaces, no retired
`--enable-cupti-activity` token in active source, resolved links for all 42
inventory entries and 48 retained stages, and retained historical SimBench
references. C8-B is complete for corrected-workload failure dispositions;
historical material is not deleted and no unfinished mechanism is relabeled.
The C8-B disposition record now retains six concrete blockers (external
tracer contract, no currently unresolved tensor-pipeline rows, and three vendor
static-module rows) without shape substitution; see
`docs/sm86_c8b_corrected_failure_dispositions_v1.json`. C8-C is complete as a
scoped final runbook synchronization. `docs/sm86_c8c_runbook_synchronization_v1.json`
now links the M5/C8 records from `docs/README.md` and confirms that the open
C1/C3/C4/C5/C6/C7 surfaces remain visible; C8 is complete as a scoped
runbook/claim phase, not as physical mechanism completion.

The residual **C1-R matched resident-memory fence/visibility suite** now has a
fresh four-arm capture in
`docs/sm86_c1r_store_fence_differential_live_v1.json`. Control/store ×
no/device-fence arms passed seven host oracles, sanitizer, NVBit, deterministic
replay, and five NCU repetitions. Simulator store deltas are `2886` cycles;
NCU medians are `6080`/`6228`, and the simulator's 12-cycle fence increment
misses NCU's 177/325-cycle increments. This is a retained fence-model
mismatch, not a fitted parameter; store completion tails remain open and C1-R
stays active for repair. The follow-up system-fence arms reproduce the prior
crosswalk blocker (`756` extra NCU instructions per system arm), so those rows
are retained but not scored. C1-R-B is complete as a negative reconciliation;
the next technical seam is C3-F physical-upper/accuracy boundary. C3-A's
focused model-local regression passed 13 targeted tests
covering feedback acknowledgements, checked completion arithmetic, async
refusal, and explicit writeback-tail propagation; C3-B then fixed the per-warp
acknowledgement readiness bug and
the SM-active projection bug, and the retained traces passed the `0` versus
`4096` model-only perturbation. C3-C independently extracted 1428 store facts
per arm and the device-fence opcode while retaining hardware completion,
ownership, generation, order, and fence scope as unknown. C3-D is complete
for async-copy/visibility tail reconciliation and records 8192 async lane
copies with 2048 committed/waited groups plus an unchanged carded-DRAM
negative perturbation. C3-E is complete for an independently witnessed
completion/visibility continuation and has now completed the async reference
branch with 33290 exact-fallback boundaries and zero complete transitions.
C3-F remains active because the physical completion upper is still unknown;
none of these local results close hardware completion.

Native replay is unchanged by the predictive flag. Service demand now yields an
explicit model-local service/queue composition; the live full-trace role is
independent post-replay, while only an explicit residual/replacement role may
feed an observer frontier. Unknown typed tails remove finite upper bounds. The
R6 corpus confirms that real native rows remain open/exact fallback and records
no macro dispatch or native step
reduction. Physical routing, compiler implementation, and internal cache state
are black-box variables, not milestones. Topology summaries are not causal
read-from histories. Endpoint projections require an explicit
statistical/modeling assumption rather than timer equality.

The P2 synthetic reference-feature fixture is a plumbing/differential result,
not native corpus compression or hardware speedup: its one dispatch is counted
in the reference event loop, while the retained SM86 corpus still reports zero
dispatches and exact fallback.

## Next work

The first-round history remains in the
[reification/refactor plan](../plans/sm86_reification_refactor_20261001.md).
The active second-round program is the
[second-round refinement plan](../plans/second_round_refinement_20261001.md),
with **R1 semantic closure and evidence hygiene complete**, **R2 executable
macro transition seam complete**, **R3 independent witness/order closure
complete**, **R4 service/queue semantics complete**, **R5 native state
refinement complete for the closed issue-model seam**, **R6
independent-corpus/observer evaluation complete**, and **R7 rebuttal/claim
discipline complete**. R1 removed unsound finite
bounds and evaluator ambiguities. R2 executes a model-local nonidentity
transition and exact continuation. R5 retains a closed native issue-model
helper transition, while P2 additionally exercises the production
reference-feature event loop on a synthetic complete scalar witness; real
front-end-enabled launches remain exact fallback. R6
expands the corpus and confirms that native compression, finite observer
uppers, and hardware accuracy remain unproven; no model-only step reduction is
relabeled as native compression.
Report finite intervals, open/refused cases, OOD behavior, actual execution
steps, memory, and runtime separately; a passing unit suite is not accuracy
evidence. The R6 corpus and third-round claim map are in
[`review_round3_20261002/README.md`](review_round3_20261002/README.md) and
[`third_round_rebuttal_20261002.md`](../plans/third_round_rebuttal_20261002.md).

The latest independent panel re-review is being addressed by the staged
[R8 refinement program](../plans/latest_review_refinement_20261002.md).
R8.0 review intake, R8.1 immediate soundness guards, R8.3 conservative native
eligibility, R8.4 service/queue identifiability, R8.5 measurement identity and
cost utility, and R8.6 synchronization are complete. R8.2 real-corpus native
dispatch remains **open**: a bounded native canary exposed a response-owner/
queue join failure before any real production macro could be consumed. P2
separately closes a synthetic complete-witness reference-feature event-loop
differential. The retained front-end-enabled SM86 corpus remains exact
fallback, with no native speedup or hardware causality claim. R8.3
exact-falls back for non-unit completion tails, pending tokens, or
untouched future-relevant state rather than calling the scalar recurrence
state-complete. R8.4 adds explicit residual/replacement service roles and a
model-local intervention test; the independent-post-replay default remains a
sidecar and queue-depth observations remain unmodeled.

R9.0 repaired the bounded reference-canary owner/queue projection failure by
widening unresolved joins to unknown ownership. The rerun terminates with
10,032 reference records and exact fallback throughout; it does not close R8.2
production macro dispatch.

R9.1 attempted the guarded witness/dispatcher path. The native canary has no
complete independent witness, so the run records zero native macro dispatches
and exact fallback throughout. P2 closes a separate synthetic reference
fixture, but does not promote it to a silicon or front-end completion claim.

R9.2 completed the two-pass witness audit negatively: no complete typed
pre/effect/post pair exists in the bounded canary, so self-derived witness
construction is rejected. The required positive falsifier is documented in the
R9 plan and remains open.

R10.0 completed a live 168-block SM86 development campaign: 10/10 cases ran
native correctness, NVBit trace, simulator replay, analyzer, and three fresh
post-prediction NCU repetitions. All intervals remained open, analyzer
candidate records and native macro dispatches were zero, and no speedup claim
is made. See [the live campaign result](review_round7_20261002/README.md).

R10.1 completed a live writeback campaign: 2/2 large dirty-stream cases ran
native correctness, NVBit trace, replay/analyzer, and three fresh NCU
repetitions. Writeback completion remains explicitly unavailable/open; no
parameter fit, native dispatch, or speedup claim is made. See
[the writeback result](review_round8_20261002/README.md).

R10.2 completed a live cache-policy contrast on two large cases with three
fresh `cache-control=none` repetitions after the all-cache measurements. The
contrast is observational only; no fitting, cache-causality, or speedup claim
is made. See [the contrast result](review_round9_20261002/README.md).

R11.0 completed detailed-versus-fast-frontier cost measurements on all 15
retained traces: 5,130,860 records, zero accepted candidates, and zero native
macro dispatches. The result reports wall/RSS and refusal coverage, not
speedup. See [the full cost result](review_round10_20261002/README.md).

The post-review P0–P6 completion program is synchronized and closed. Its final
real-path falsifier is the current bounded SM86 canary: 10,033 reference
records, exact fallback, and zero native dispatches. The synthetic
reference-feature fixture is a differential/plumbing result only. Full Cargo
workspace tests and all affected Python suites pass; repository-wide pytest is
not a valid check because the untracked review corpus and optional campaign
dependencies interrupt collection.

The accuracy-first NCU workstream has completed A0: the exact
metric/replay/cache/device/repetition contract, identity/build requirements,
artifact separation, and split rules are declared and schema-tested in
`docs/sm86_ncu_accuracy_protocol_v1.json`; no new measurements were collected.
A1 is complete: the metadata-only identity capture path validates GPU/driver/
clocks, NCU metric availability, tool/build/input identity, and replay/cache
contract. The current dirty checkout is labeled ineligible and no workload was
profiled. A2 is complete: the preflight/reporting path validates bounded
termination, five-repetition policy, dispersion, and prediction-before-
observation ordering without launching NCU. The current historical manifest is
correctly not ready for the new protocol. A3 is complete: source-path and
family/split closure pass, five expansion slots are declared, and missing
predictions keep preflight unready. A4 is complete: the protocol-aligned
source-bound 15-row prediction artifact is prepared before observations and
the service sidecar remains explicitly unvalidated/unmodeled where required.
No NCU observations were collected in A4. A5 is now complete: five fresh
application-replay/all-cache/no-clock-control repetitions ran for all 15
workloads after clean identity/preflight. The 75 observations are separate
from predictions; all intervals remain open, so finite accuracy is not
applicable (lower consistency 0.667/1.000/0.750 across the three splits). A6
is complete: cache `all→none` (15 workloads × five repetitions) and clock
`none→base` (two workloads × five repetitions) contrasts preserve separate
prediction/observation joins and matching mechanism counters. Simulator
projections were invariant; directions remain observational-only and all
uppers remain open. A7 is complete: the retained real SM86 trace produced
zero complete typed boundaries, zero native macro dispatches, and exact
fallback under the verified identity/endpoint contract. The primary outcome is NCU
accuracy; simulator wall-clock runtime remains secondary, with no
compression/speedup metric.

The A8 live-campaign foundation is complete under
[`accuracy_first_live_campaign_a8_20261002.md`](../plans/accuracy_first_live_campaign_a8_20261002.md).
A8.0–A8.3 are complete: the bounded runner classifies OOM/OOD/timeout and
launcher failures, all 15 portable-LCD binaries were rebuilt from source, and
all 15 correctness smoke rows passed without silent shape substitution. A8.4
and A8.5 are complete for one fresh `gemm_hmma_bf16_sm80` slice: native
correctness, NVBit trace, simulator replay, prediction-before-observation,
independently identified five-repetition NCU, and schema/contract validation
all pass. The prediction lower is 3023 cycles with an open upper; NCU target
cycles are 5684/5490/5514/5524/5517 (median 5517), so finite interval coverage
and point error are not applicable. Simulator dirty-writeback/coherence tails
remain open and native macro dispatch is zero. A8.6 handoff synchronization is
complete; A8 is complete and A9 is the next phase (not started).

## Completion program correction — 2026-10-03

The active [plan](../plans/sm86_completion_frontier_and_native_dispatch_20261002.md)
reopens C0/C1/C3. The previous tail-flag clearing used insufficient counters;
the 1600-cycle padding has no reproducible calibration; the async comparison
mixed 64- and 16384-group binaries; managed-memory message-passing timings
were not isolated coherence measurements. Previous finite-upper and C1
completion claims are withdrawn. Raw measurements remain historical data.
C0-R repairs soundness, then C1-R repairs matched live comparisons. Finite
statistical prediction and exact semantic completion are separate questions.
C4/C5 exploratory positive slice: calibration on group counts 16/32 predicted a finite 37737..38655 empirical NCU interval for a new group-128 binary before its NCU run; all 5 repeats (38353/38339/38348/38338/38395) were covered and point error was 0.396%. This is observer statistics only: simulator semantic lower 168898 and writeback/coherence remain open. C1-R matched async/control differential is a real negative model result: same 64-group binary target records match, NCU async-control delta is 13760 cycles while simulator lower-bound delta is 78852. C1-R matched async recapture is a real negative model result: same 64-group binary/arguments/control, target trace records equal NCU instruction counts, but simulator lower 84866 exceeds NCU median 20946. A resident-device message-passing recapture has simulator lower 1753 below NCU median 4321 and remains open. Both are retained as mismatch evidence; no finite upper or tuning claim. C0-R is verified on a live store trace; the same service configuration drains modeled queues but leaves dirty-writeback/coherence open. C1-R is active for matched async/fence recaptures. C4/C5 have one exploratory positive statistical slice: disjoint group16/32 calibration predicted group128 37737..38655 with 5/5 NCU coverage and 0.396% point error, while semantic completion remains open. C3 now refuses async timing lower bounds after the matched differential instead of emitting the false 84866-cycle lower. Full completion and real native selective dispatch remain unachieved.

The first A5 preflight correctly rejected the dirty checkout; after the
coherent implementation was committed and rebuilt, independent identity and
preflight passed. The first collection command retained a historical launch
skip and produced no profiled kernels; the corrected skip-zero campaign
completed all 75 repetitions and is the retained baseline.

### Completion-frontier continuation — 2026-10-03

The C3 async source-token repair is recorded in
`docs/sm86_completion_frontier_c3_async_source_token_repair_v1.json`. The
cacheable LDGSTS return ledger now charges one service token per coalesced
warp-level instruction rather than per 32-byte sector. A fresh post-change
group sweep gives model cycles 7,442/13,090/24,386/46,978 for groups
16/32/64/128 versus NCU medians 7,904/12,254/20,949/38,348; the mismatch is
reduced but remains. A fresh NVBit plus five-process NCU source-only capture is
retained in `docs/sm86_completion_frontier_c3_source_only_live_v1.json`: the
store-free case has finite model-local completion at 13,235 cycles versus NCU
median 12,616 (4.91% point error), while hardware completion tokens and fence
scope remain unobserved. This is a narrow positive model case, not a general
hardware-accuracy or writeback/coherence result.

C4/C5 now has a second declared holdout: compact/conflict calibration predicted
the severe destination-layout target at `24,476.2..24,902.2` cycles before
profiling. Five fresh severe NCU values are 24,822/24,855/24,875/24,891/24,840
(median 24,855); the interval covers the median and point APE is 0.667%. The
prediction and post-observation join are separated in
`docs/sm86_completion_frontier_c4_destination_layout_prediction_pre_ncu_v1.json`
and `docs/sm86_completion_frontier_c4_destination_layout_holdout_v1.json`.
This is one controlled layout family, not a semantic upper or global accuracy
claim.

C6 has a real same-shape scalar negative slice in
`docs/sm86_completion_frontier_c6_fixed_latency_live_v1.json`: the exact
64-generation trace cubin passes correctness and five NCU captures, but the
fast analyzer refuses all candidates for nonfull active/predicate populations
(zero native dispatches). Simulator endpoint 3,578 versus NCU median 6,812 is
47.5% low; prior 8,192-generation NCU rows are excluded.

C7’s portable population slice is complete: the full 15-row live campaign ran
native timing, NVBit tracing, two simulator replays, and five fresh NCU runs
per row in 389 seconds, with 15/15 completed and zero OOM/OOD/timeouts or
failed stages. Raw data is under
`results/alpha/c7_full_live_20261003_run/`; tracked summary:
`docs/sm86_c7_full_live_campaign_v1.json`. This is not yet the full repository
population: native probes, fixed-latency rosters, SOL/other suites, and
correctness-mode joins remain to be inventoried. Portable replay rows retain
open dirty-writeback/coherence tails; no native macro dispatch was requested.

The C7 inventory lists 15 portable rows, 31 SM86-capable rows among 92
SimBench rows, 42 research-probe directories, and 203 SOL baseline problems.
Portable timing paths and twenty-six direct SimBench slices have completed:
`fp32_fma_dep_chain`, `cp_async_shared_tile_copy`, `global_atomic_histogram`,
`global_vectorized_128b_copy`, `bf16_hmma_gemm_traceable`,
`global_scalar_32b_copy`, `global_atomic_return_used`,
`fp16_hmma_gemm_traceable`, `global_vectorized_64b_copy`,
`shared_bank_conflict_stride`, `l2_thrash_pointer_chase`,
`l2_persisting_window_sweep`, `l2_prefetch_hint_sweep`, and
`warm_cold_pair_relaunch`, and
`carveout_sensitivity_probe`, and `working_set_capacity_sweep`, and `dram_stream_stride_mix`, and `local_dynamic_index_array`, and `local_volatile_array`, `register_spill_local_mem`, and `global_atomic_max_histogram`, and `global_atomic_max_return_used`, and `rmsnorm_fwd_bf16`, and `global_atomic_and_histogram`. These check outputs at the captured shape; the FP32
direct slice is timing-only (its separate correctness-shape run does not
validate that captured shape). All twenty-six retain open tails and zero native
macro dispatch. The inventory is
`docs/sm86_c7_workload_population_inventory_v1.json`.

The full-shape `dram_streaming_dot_product` slice is additionally recorded in
`docs/sm86_c7_simbench_dram_streaming_dot_product_direct_live_v1.json`:
argument/launch/allocation guards and a per-block roundoff-bounded reference
check pass, Compute Sanitizer reports zero errors, NVBit/replay cover all 128
blocks with 297,600 warp instructions, and five NCU gpc-cycle values are
7714/7579/7548/7524/7562 (median 7562). Its simulator lower bound is 5345,
so lower-bound consistency is true; dirty-writeback/coherence and native
dispatch remain open.

The bounded `global_atomic_and_histogram` slice is recorded in
`docs/sm86_c7_simbench_global_atomic_and_histogram_direct_live_v1.json`. The
repaired driver checks the identity output and deterministic bitwise-AND
histogram, emits RED.E.AND, and avoids the old billion-operation timing shape
that could cause OOM or impractical traces. Replay has 42,874 warp records;
NCU cycles are 568244/568570/568083/567453/568196 (median 568196), while the
simulator lower is 436096. Dirty-writeback/coherence and native dispatch remain
open.

The full-shape `rmsnorm_fwd_bf16` slice is recorded in
`docs/sm86_c7_simbench_rmsnorm_fwd_bf16_direct_live_v1.json`. Its repaired
driver validates BF16 inputs element-by-element, uses one 256-thread block per
row, and has explicit OOM/OOD/launch checks. The `rows=1024, hidden=4096`
capture passes Compute Sanitizer; NVBit/replay cover 1,024 blocks with
4,235,264 warp instructions, and five NCU gpc-cycle values are
48453/48106/48931/50179/48481 (median 48481). The simulator lower is 43214,
so lower-bound consistency is true; dirty-writeback/coherence and native
dispatch remain open.

TF32 HMMA and INT8 IMMA rows now pass host correctness, NVBit, NCU, and exact
SM86 replay at their shape-specific capability seams; physical tails remain open. Ten memory/control rows remain real lower-bound falsifiers; three L2
examples are pointer chase (59,545 vs NCU median 49,580), persisting window
(31,173 vs 29,846 after the sink repair), and prefetch hint (46,874 vs
40,598). The prefetch binary emits
five static `CCTL.E.PF2` instructions; cache effect and completion remain
unobservable. The corrected shared-bank shape uses SM86 dynamic shared-memory
opt-in (65,536 bytes against default 49,152 and opt-in 101,376), superseding
its old false OOD disposition.

The research-probe population was re-enumerated from the current tree: 42 directories are present (37 Makefile-backed probes, no separate source/preserved-binary rows, and 5 analysis-only rows). The bounded inventory is `docs/sm86_c7_research_probe_inventory_v1.json`; the synchronized 37-row build matrix is `docs/sm86_c7_research_probe_build_matrix_v1.json`; all 37 Makefile probes build successfully with the pinned CUDA 13.2/GCC 14/sm_86 matrix in `docs/sm86_c7_research_probe_build_matrix_v1.json`. Matrix campaigns are complete for issue_port_control (68), async_barrier_overlap (36), store_ack_frontier (30 arms), memory_retirement_service (102 with eight crosswalk exclusions), barrier_pipeline (33), and l1tex_admission_backpressure (100, all lower-bound-consistent). An explicit warmup/target l2_queue_frontier vertical cell also passes correctness/NVBit/replay/five-NCU with lower-bound consistency in `docs/sm86_c7_research_probe_l2_queue_frontier_live_v1.json`; remaining probe families are not yet claimed.

The remaining BF16 cuBLAS control row is now also dispositioned in
`docs/sm86_c7_simbench_vendor_gemm_cublas_blocked_live_v1.json`: host BF16
reference validation and Compute Sanitizer pass, five NCU rows observe the
dynamic `cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_32x1_nn_align8`
kernel (1456 warp instructions; median 5347 cycles), but the repository
control sidecar has zero functions and NVBit refuses the tensor shape before
publishing a complete trace. All 31 SM86 SimBench roster rows therefore have
either a direct live artifact or an explicit replay blocker; this does not
close the separate research-probe or SOL populations.

Warm/cold multi-launch is complete with separate outputs/sinks, zero
Compute-Sanitizer errors, detailed three-launch replay, launch/stream-L2
separated simulator scopes, and cache-control=none NCU observations. First,
eviction, and repeat medians are 63,183 / 66,548 / 46,876 with large cache-state
dispersion; this is plumbing evidence, not point accuracy or completion proof.
See `docs/sm86_c7_simbench_warm_cold_pair_relaunch_direct_live_v1.json`.

Corrected active evidence:
- `docs/sm86_c7_simbench_shared_bank_conflict_stride_corrected_live_v1.json`
- `docs/sm86_c7_simbench_l2_persisting_window_sweep_corrected_live_v1.json`
- `docs/sm86_c7_simbench_l2_prefetch_hint_sweep_corrected_live_v1.json`
- `docs/sm86_c7_simbench_warm_cold_pair_relaunch_direct_live_v1.json`
- `docs/sm86_c7_simbench_carveout_sensitivity_probe_direct_live_v1.json`
Superseded artifacts remain historical and are not current evidence.

C7-S18 atomic-and-return is complete after repairing its shared-bin sink
race: disjoint return sinks and a deterministic final bitwise-AND histogram pass;
46,882 replay/NCU instructions match. C7-S19 carveout sensitivity is complete
and falsifies lower-bound consistency (211,002 simulator lower vs 178,521 NCU
median). C7-S20/S21 vendor SGEMM variants are explicit replay blockers: cuBLAS
kernels from libcublasLt.so have no stable static control sidecar, and traced
records cannot be silently crosswalked to NCU instruction counts. See
`docs/sm86_c7_simbench_vendor_sgemm_cublas_blocked_live_v1.json` and
`docs/sm86_c7_simbench_vendor_sgemm_strided_batched_cublas_blocked_live_v1.json`.

C7-S22 working-set capacity is complete: the repaired anchor identity checks
pass at a 1MiB working set, replay/NCU match at 789,888 instructions, and
simulator lower 143,555 exceeds NCU median 129,756 by 13,799 cycles (10.63%).
See `docs/sm86_c7_simbench_working_set_capacity_sweep_direct_live_v1.json`.

C7-S24 local dynamic-index is complete: STL/LDL local traffic is exposed, replay/NCU match at 72,655 instructions, and simulator lower 1,443,183 exceeds NCU median 1,243,714 by 199,469 cycles (16.04%). See `docs/sm86_c7_simbench_local_dynamic_index_array_direct_live_v1.json`.

C7-S25 local volatile-array is complete: STL local traffic is exposed, replay/NCU match at 29,017 instructions, and simulator lower 524,608 exceeds NCU median 422,474 by 102,134 cycles (24.18%). See `docs/sm86_c7_simbench_local_volatile_array_direct_live_v1.json`.

C7-S26 register-spill local memory is complete: -maxrregcount=32 exposes 424-byte STACK/STL traffic, replay/NCU match at 271,680 instructions, and simulator lower 30,010 exceeds NCU median 23,073 by 6,937 cycles (30.08%). See `docs/sm86_c7_simbench_register_spill_local_mem_direct_live_v1.json`.

C7-S27 atomic max histogram is complete: deterministic RED.MAX histogram checks pass, replay/NCU match at 42,874 instructions, and simulator lower 436,096 is below NCU median 568,408 while tails remain open. See `docs/sm86_c7_simbench_global_atomic_max_histogram_direct_live_v1.json`.

C7-S28 atomic max return is complete: disjoint return sinks and deterministic final max histogram checks pass, replay/NCU match at 46,882 instructions, and simulator lower 916,220 is below NCU median 1,060,648 while tails remain open. See `docs/sm86_c7_simbench_global_atomic_max_return_used_direct_live_v1.json`.

C7-S23 DRAM stream stride is complete: replay/NCU match at 1,716,864
instructions, while simulator lower 550,005 exceeds NCU median 419,040 by
130,965 cycles (31.25%). See
`docs/sm86_c7_simbench_dram_stream_stride_mix_direct_live_v1.json`.

The `async_visibility_phase` probe now independently recomputes and checks its
returned lane-0 token from the deterministic source initialization; it no
longer prints an unconditional correctness boolean. Fresh direct runs for
cache-all compact, cache-all conflict, and streaming severe all pass token and
interval-sum checks. Prior phase captures are historical because the binary
changed.

The corrected phase binary also completed a fresh NVBit/replay/five-NCU path;
its independently checked direct token is correct, simulator endpoint is
18,305 cycles, and NCU median is 16,658 (9.89% APE). The result-store’s
writeback/coherence tails remain open. See
`docs/sm86_completion_frontier_c3_async_phase_corrected_live_v1.json`.

The resident-device message trace has a same-input service differential in
`docs/sm86_completion_frontier_c3_message_service_differential_v1.json`:
explicit model-local queues complete 3 writeback sectors and 2 coherence
transactions, but both dirty-writeback/coherence tail flags remain true. This
demonstrates queue drain without manufacturing a semantic upper.

A matched one-warp `c00`/`c11` differential is retained in
`docs/sm86_completion_frontier_c3_async_endpoint_differential_warp1_v1.json`.
Five fresh NCU medians are 7,194 and 14,668 cycles (7,474-cycle increment),
while repaired simulator endpoints are 5,995 and 17,394 (11,399-cycle
increment); direct unprofiled `clock64` repeats give a 13,386-cycle
increment. This does not identify the missing async mechanism: the simulator
lies between the direct-clock and NCU differentials, so no scalar parameter is
changed from this comparison.

An independent cache-all source-only layout pair is now recorded in
`docs/sm86_completion_frontier_c3_async_destination_layout_pair_v1.json`.
Compact/conflict variants retain the same binary, depth, route, and launch;
destination wavefronts change 2,048→4,096. The simulator predicts a 4,096-cycle
delta and five fresh NCU medians show 4,070 cycles (26-cycle, 0.64% delta
error). Absolute endpoints remain 4.91%/3.87% high, so this is narrow
model-local differential evidence, not a general hardware upper or predictor.

## Current continuation — 2026-10-04

The system-fence crosswalk note was corrected: SourceCounters reports one
warp execution for each visible fence PC; zero values apply only to the
separate stall-sampling columns. The aggregate 12-instruction NCU delta is
still unresolved and remains excluded from scoring.

The L2 queue-frontier probe is hardened against malformed input, arithmetic
overflow, and OOM-prone allocations, and now validates its output checksum and
clock bounds independently. The fresh live artifact
`docs/sm86_c7_research_probe_l2_queue_frontier_live_v2.json` uses one rebuilt
binary for correctness, NVBit, replay, and five NCU runs. It reports 70,560
target warp instructions in both replay and NCU, NCU median 7,714 gpc cycles,
and simulator lower 5,751; dirty-writeback/coherence tails remain open.

The current C7 continuation has completed the L1TEX arrival/replay vertical
slice. Its checked `store,line,84,8,8,32,0,1` driver uses one rebuilt binary
across correctness, NVBit, replay, and NCU; the former separate trace-twin
interface is gone. Seven host trials, reduced Compute Sanitizer, full 84-block
trace, target replay, and five NCU captures pass the live contract. Replay and
NCU each report 809,088 warp instructions; NCU median gpc cycles are 31,748
and simulator lower is 8,486. Dirty-writeback/coherence tails remain open.
Artifact: `docs/sm86_c7_research_probe_l1tex_arrival_replay_service_live_v1.json`.

C7-S33U completed a live dependent-MUFU SFU slice after hardening its driver
with strict parsing, bounded iterations/allocation, sentinel output checks, and
timestamp validation. The same binary passed seven host trials, reduced Compute
Sanitizer, four-launch NVBit, target replay, and five NCU captures. Replay and
NCU agree at 1,389,024 target warp instructions; NCU median gpc cycles are
34,282, while simulator lower is 38,099. The lower-bound falsifier is retained
explicitly; no SFU timing is fitted or claimed accurate. Artifact:
`docs/sm86_c7_research_probe_sfu_frontier_live_v1.json`.

C7-S33V completed a same-binary result-queue collision slice. The stale
64-generation trace twin was removed; one patched 8,192-generation cubin now
serves correctness, NVBit, replay, and NCU. The hardened driver passed seven
host trials, Compute Sanitizer, full replay, and five NCU captures. Replay and
NCU agree at 524,416 warp instructions; NCU median gpc cycles are 307,646 and
simulator lower is 287,635. No result-queue latency is activated or fitted.
Artifact: `docs/sm86_c7_research_probe_fixed_latency_result_queue_live_v1.json`.

C7-S33W completed the matched result-queue collision/separated differential at
width 4 and 8,192 generations. Collision NCU median is 307,646 versus
simulator lower 287,635; separated median is 340,424 versus lower 353,171, a
retained lower-bound falsifier. The NCU delta is 32,778 cycles while the
simulator lower-bound delta is 65,536; no latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_pair_live_v1.json`.

C7-S33X completed the held-out width-8 result-queue collision/separated pair.
The live trace exceeded the old one-million-instruction preflight ceiling at
1,048,832 decoded instructions per block; the finite limit was raised to two
million and the compact route index widened by one bit, after which both exact
captures replayed. Both arms pass seven host trials, Compute Sanitizer, NVBit,
replay, and five NCU runs. Collision median is 447,092 versus lower 320,466;
separated median is 438,935 versus lower 369,618. Hardware separated is 8,157
cycles below collision while simulator lower delta is +49,152; no latency is
fitted. Artifact: `docs/sm86_c7_research_probe_fixed_latency_result_queue_width8_pair_live_v1.json`.

C7-S33Y completed the width-12 result-queue collision/separated pair. Each
block has 1,573,248 decoded instructions and replays under the two-million
finite decoder ceiling. Both arms pass seven host trials, Compute Sanitizer,
NVBit, replay, and five NCU runs. Collision median is 589,202 versus lower
527,551; separated median is 561,568 versus lower 533,490. Hardware separated
is 27,634 cycles below collision while simulator lower delta is +5,939; no
latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_width12_pair_live_v1.json`.

C7-S33Z completed the declared width-16 result-queue pair. Each block has
2,097,664 decoded instructions; the finite preflight ceilings were raised once
to 3,000,000 instructions and 4,000,000 decoded nodes with a 22-bit compact
route index, then both exact traces replayed. Both arms pass seven host trials,
Compute Sanitizer, NVBit, replay, and five NCU runs. Collision median is
708,805 versus lower 579,430; separated median is 693,935 versus lower
654,998. Hardware separated is 14,870 cycles below collision while simulator
lower delta is +75,568; no latency is fitted. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_queue_width16_pair_live_v1.json`.

C7-S33AA completed a hardened completion-distance split slice. One patched
8,192-generation cubin serves correctness, NVBit, replay, and NCU; the driver
now has strict parsing, bounded allocations, sentinel output and elapsed-sample
checks, and a sink oracle. The `distance=1, split, warps=4` cell passes seven
host trials, Compute Sanitizer, NVBit, replay, and five NCU runs. Replay and NCU
agree at 622,832 target warp instructions; NCU median is 586,172 versus lower
566,204. No latency is activated. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_live_v1.json`.

C7-S33AB completed the matched completion-distance bank3 arm. The same patched
8,192-generation cubin passes seven host trials, Compute Sanitizer, NVBit,
replay, and five NCU runs at 622,832 target warp instructions. Simulator lower
is 566,204; bank3 NCU median is 586,129 versus fresh split median 586,107, a
22-cycle differential. No latency is activated. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_pair_live_v1.json`.

C7-S33AC completed the held-out distance-2 completion pair. Both same-binary
arms pass seven host trials, Compute Sanitizer, NVBit, replay, and five NCU
runs with 622,832 matching target warp instructions. Simulator lower is 336,828
for both; split median is 340,415 and bank3 median 356,791. The 16,376-cycle
differential is retained without fitting latency. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance2_pair_live_v1.json`.

C7-S33AD completed the held-out distance-4 completion pair. Both arms pass
seven host trials, Compute Sanitizer, NVBit, replay, and five NCU runs at
622,832 target warp instructions. Simulator lower is 336,826; split median is
348,610 and bank3 median 414,145. The 65,535-cycle differential is retained
without fitting latency. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance4_pair_live_v1.json`.

C7-S33AE completed the held-out distance-8 completion pair. Both arms pass
seven host trials, Compute Sanitizer, NVBit, replay, and five NCU runs at
622,832 target warp instructions. Simulator lower is 336,826; split median is
356,797 and bank3 median 414,112. The 57,315-cycle differential is retained
without fitting latency. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_completion_distance8_pair_live_v1.json`.

C7-S33AF closed the completion-distance vertical family and reconciled the
probe inventory. All 23 live probe stages have schema-validated artifacts;
current manifests and focused selections pass; paired-binary campaigns no
longer require stale trace twins or predecessor-sidecar populations. Population
entries now link completed live families, while 26 current probe entries remain
uncaptured or analysis-only. Falsifiers, crosswalk exclusions, open tails,
unsupported rows, and VMM physical-identity limits remain explicit. Evidence:
`docs/sm86_c7_research_probe_inventory_v1.json`.

C7-S33AG completed an atomic completion serialization vertical slice. One
rebuilt 1,024-generation cubin per variant now serves correctness, NVBit,
replay, and NCU; stale 64-generation trace cubins are gone. The hardened
driver passes seven host trials, Compute Sanitizer, NVBit, replay, and five NCU
runs for `ret_lane_word, warps=4`. Replay/NCU agree at 24,660 warp
instructions; atomic requests are 4,096, NCU median 316,109, simulator lower
267,561. No atomic timing is fitted. Artifact:
`docs/sm86_c7_research_probe_atomic_completion_serialization_live_v1.json`.

C7-S33AH completed a multi-SM atomic service vertical slice. One rebuilt
1,024-generation cubin per operation serves all stages; the driver rejects
stale generation counts, bounds memory, validates data/returned sums and elapsed
samples, and cleans up resources. `ret32, blocks=4, spacing=0` passes seven host
trials, Compute Sanitizer, NVBit, replay, and five NCU runs. Replay/NCU agree at
24,692 instructions; NCU observes 4,096 atomic and 131,072 LTS requests, median
355,107, simulator lower 268,019. No routing/timing model is fitted. Artifact:
`docs/sm86_c7_research_probe_atomic_multism_service_live_v1.json`.

C7-S33AI completed an observational atomic alias-adversarial slice. One rebuilt
1,024-generation cubin per operation/width replaces trace twins; the driver
checks generation/pattern/alias/width, bounds memory, validates elapsed/data/
returned outputs, and cleans up. Ret32 dense alias-32, warps 4, passes seven
host trials, Compute Sanitizer, NVBit, replay, and five NCU runs. Replay/NCU
agree at 24,684 instructions; atomic requests 4,096, LTS requests 131,072,
NCU median 382,651, simulator lower 267,996. No proprietary hash or timing
model is claimed. Artifact:
`docs/sm86_c7_research_probe_atomic_alias_adversarial_live_v1.json`.

C7-S33AJ completed a CUDA VMM alias diagnostic. The rebuilt correctness
campaign passes 19/19 cells; the selected offset-0 alias cell has five NCU
profiles and an NVBit capture covering seven launches. NCU read-sector values
are observation only: VMM mappings/physical handles are outside ordinary trace
replay, so no simulator lower bound or hash/timing model is claimed. Artifact:
`docs/sm86_c7_research_probe_atomic_vmm_alias_live_v2.json`.

C7-S33AK completed a task-done multi-kernel boundary slice. The same
`store_device` binary passes seven host correctness trials (16-MiB footprint,
16-MiB thrash, four blocks, two warmups); NVBit captures all nine launches with
metadata enabled, sim-run replays all nine, and five NCU runs profile the same
nine-launch application with per-kernel instruction matches. Endpoints remain
separate; no fixed retirement upper is inferred. Artifact:
`docs/sm86_c7_research_probe_task_done_boundary_live_v1.json`.

C7-S33AM completed an SMSP scheduler-contention vertical slice. One rebuilt
8,192-generation binary serves all stages; `q20, phase=0, warps=8` passes seven
host trials, Compute Sanitizer, NVBit, replay, and five NCU runs. The 69-MiB
block payload required a finite anchored-reader increase from 64 to 128 MiB;
focused anchored tests pass. Replay/NCU match at 2,556,096 instructions; NCU
median is 1,271,033 versus lower 1,123,534. Phase labels are controls, not
physical scheduler-slot truth. Artifact:
`docs/sm86_c7_research_probe_smsp_scheduler_contention_live_v1.json`.

C7-S33AN completed a safe fixed-result bypass slice. One rebuilt 1,025-
generation cubin per safe variant replaces trace twins; the driver rejects stale
generations, validates elapsed/sink outputs, and cleans up. The fixed_retired
warps-4 arm passes seven host trials, Compute Sanitizer, metadata-enabled NVBit,
replay, and five NCU runs. Replay/NCU agree at 28,788 instructions; NCU median
34,930 and simulator lower 27,505. The intentionally illegal-address global
canary was excluded. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_bypass_live_v1.json`.

The active continuation is tracked in
`plans/sm86_completion_frontier_and_native_dispatch_20261002.md`; the old
C7-S33AO selection note is superseded by the completed AO/AP refreshes and the
AQ/AR stages below.

## Retained measurements

2026-09-30 RTX 3090 Ti campaigns used driver 615.71.09, 1350/10501 MHz, P0,
and NCU clock control `none`:

| Cache policy | Workloads | Median NCU duration | Median GPC cycles |
|---|---:|---:|---:|
| `all` | 15/15 | 10,080 ns | 13,572 |
| `none` | 15/15 | 8,704 ns | 11,709 |

These are measurements, not prediction-error results; campaign raw outputs remain under results/.

## Navigation

[Model contract](PREDICTIVE_CLOSURE.md) · [Configuration](archprofile.md) ·
[Runbook](runbook_host.md) · [Checks](testing.md).

C7-S33AO completed a fixed-result capacity live refresh. The active interface
now uses only 8,192-generation binaries; the W8/b9 arm passed seven correctness
trials, NVBit, deterministic replay, and five NCU runs. Replay/NCU match at
1,573,144 instructions; NCU median 851,475 cycles versus simulator lower
590,830. Capacity/pop width remain unidentified; no timing model is enabled.
Artifact: `docs/sm86_c7_research_probe_fixed_latency_result_capacity_live_v1.json`.

C7-S33AP completed an immediate-constant live anchor. The hand-authored PTX
cubin passed direct correctness, NVBit, sim-run replay, and one independent NCU
profile; 151,872 instructions match, simulator lower is 6,037 cycles, and NCU
is 10,709 GPC cycles. No IMC timing model is enabled.
Artifact: `docs/sm86_c7_research_probe_immediate_constant_frontier_live_v1.json`.

C7-S33AQ completed a hardened `constant_load_path` vertical slice. The source
now has strict parsing, bounded input sizing, sentinel result buffers, and a
separate functional output oracle. The same rebuilt binary passed seven
full-shape correctness runs, reduced Compute Sanitizer, NVBit, deterministic
replay, and five application-replay NCU captures. Replay and NCU match at
1,909,824 target warp instructions; the simulator lower bound is 123,727
cycles versus NCU median 24,016, a retained lower-bound falsifier. No
constant-load/cache parameter is fitted and dirty-writeback/coherence tails
remain open. Artifact:
`docs/sm86_c7_research_probe_constant_load_path_live_v1.json`.

C7-S33AR completed the L1TEX admission same-binary refresh. The campaign's
stale trace-only executable interface was removed; one rebuilt
`l1tex_admission_backpressure` binary served correctness, full-shape Compute
Sanitizer, NVBit, deterministic replay, and five application-replay NCU runs
for `store,line_unique,b84,w8,d32,l8,g0,all`. Replay and NCU match at
1,255,968 target warp instructions; simulator lower is 12,905 cycles and NCU
median is 29,879. No L1TEX/cache parameter is fitted and dirty-writeback/
coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_live_v2.json`.

C7-S33AS was active: capture the predeclared same-binary `control` versus
`store` contrast at the same 84-block/8-warp/depth-32 shape. Compare deltas
only as mechanism-local evidence; preserve falsifiers and do not fit a scalar
store or L1TEX latency from the pair.

C7-S33AS completed that pair. One rebuilt binary served both arms with seven
correctness trials, full-shape Compute Sanitizer, NVBit, deterministic replay,
and five NCU repetitions per arm. Hardware adds 12,823 cycles while the
simulator lower-bound delta is 556; this mismatch is retained without fitting.
Artifact:
`docs/sm86_c7_research_probe_l1tex_admission_backpressure_pair_live_v1.json`.

C7-S33AT was active: reconcile the live-probe inventory and stale-interface
claims, then select one uncaptured non-memory source with a bounded oracle and
native replay path. Do not count preserved binaries or old matrix rows as new
live stages until the current same-binary contract passes.

C7-S33AT completed that reconciliation. The inventory records the current
same-binary L1TEX refresh and control/store differential, and the active L1TEX
and Allocate interfaces no longer require their historical trace-only
executables. Historical matrix rows remain explicitly historical.

C7-S33AU completed a hardened fixed-Allocate same-binary slice. One rebuilt
8,192-generation cubin served seven correctness trials with elapsed/sink
oracles, full-shape Compute Sanitizer, NVBit, deterministic replay, and five
application-replay NCU runs for `w8_r10_native`. Replay and NCU match at
1,245,536 target warp instructions; simulator lower is 336,374 cycles and NCU
median is 593,194. No allocation/register-bank timing is fitted and
dirty-writeback/coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_allocate_live_v1.json`.

C7-S33AV was active: review `fixed_latency_front_end_footprint` next. Prefer a
bounded U1 same-binary cell; repair any short trace-twin interface before
capture and do not fit front-end timing from one cell.

C7-S33AV completed the front-end footprint same-binary refresh. The active
Makefile/campaign no longer requires the historical short trace pair; one
rebuilt 1,024-generation U1 fixed_retired cubin served seven correctness
trials with elapsed/sink oracles, full-shape Compute Sanitizer, NVBit,
deterministic replay, and five application-replay NCU runs. Replay and NCU
match at 32,852 target warp instructions; simulator lower is 32,403 cycles and
NCU median is 38,629. No front-end timing is fitted and dirty-writeback/
coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_front_end_footprint_live_v1.json`.

C7-S33AW was active: review `fixed_latency_result_order` as the next
same-binary non-memory source; preserve falsifiers and do not fit result-order
or issue-queue timing from one cell.

C7-S33AW completed the result-order same-binary refresh. The active interface
uses one rebuilt 8,192-generation cubin rather than the historical short trace
pair. The `w8_hf` cell passed seven correctness trials with elapsed/sink
oracles, full-shape Compute Sanitizer, NVBit, deterministic replay, and five
application-replay NCU runs. Replay and NCU match at 458,968 target warp
instructions; simulator lower is 197,490 cycles and NCU median is 257,751. No
result-order/issue-queue timing is fitted and dirty-writeback/coherence tails
remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_order_live_v1.json`.

C7-S33AX was active: review `fixed_latency_result_same_bank` next. Preserve
falsifiers and do not infer physical bank mapping or latency from one contrast.

C7-S33AX completed the same-bank result-order same-binary refresh. One rebuilt
8,192-generation cubin served the W8 HF cell through seven correctness trials
with elapsed/sink oracles, full-shape Compute Sanitizer, NVBit, deterministic
replay, and five application-replay NCU runs. Replay and NCU match at 458,960
target warp instructions; simulator lower is 197,489 cycles and NCU median is
233,169. No physical bank mapping or latency is inferred and dirty-writeback/
coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_result_same_bank_live_v1.json`.

C7-S33AY was active: review `fixed_latency_reuse` next; preserve falsifiers and
do not fit register-reuse or scoreboard timing from one cell.

C7-S33AY completed the register-reuse same-binary refresh. One rebuilt
8,192-generation variant-A cubin served the W8/U11-A cell through seven
correctness trials with elapsed/sink oracles, full-shape Compute Sanitizer,
NVBit, deterministic replay, and five application-replay NCU runs. Replay and
NCU match at 1,245,824 target warp instructions; simulator lower is 336,953
cycles and NCU median is 635,271. No reuse/scoreboard timing is fitted and
dirty-writeback/coherence tails remain open. Artifact:
`docs/sm86_c7_research_probe_fixed_latency_reuse_live_v1.json`.

C7-S33AZ completed a corrected front-end IBuffer same-binary slice. One rebuilt
baseline_s1 cubin passed seven timeline/sink-oracle trials, full-shape Compute
Sanitizer, NVBit, deterministic replay, and five application-replay NCU runs.
Replay and NCU match at 944 target warp instructions; simulator lower is 818
cycles and NCU median is 3,678. No IBuffer depth is fitted. Artifact:
`docs/sm86_c7_research_probe_front_end_ibuffer_depth_live_v1.json`.

C7-S33BA is historical and superseded: its stale analyzer/link reconciliation
was incorporated into the completed M3/M4 ledger without making historical
outputs execution prerequisites.

M1 correction completed: `docs/sm86_c7_research_probe_fixed_latency_result_capacity_live_v2.json` records strict parsing, initialized per-trial buffers, elapsed validation, seven correctness trials, Compute Sanitizer, same-binary NVBit/replay, and five NCU captures. Replay/NCU match at 1,573,144 instructions; simulator lower 590,830 and NCU median 851,446. Capacity remains unidentified.

M2 correction completed: `docs/sm86_c7_research_probe_immediate_constant_frontier_live_v2.json` records an independent modulo-u32 output oracle, timestamp bounds, correct 2,688 thread/84 warp accounting, executable-relative cubin lookup, seven correctness trials, reduced Compute Sanitizer, same-cubin NVBit/replay, and five NCU captures. Replay/NCU match at 151,872 instructions; simulator lower 6,037 and NCU median 10,315. IMC timing remains observational only.

C7-S33AZ completed a corrected front-end IBuffer same-binary slice. One rebuilt
`baseline_s1` cubin passed seven timeline/sink-oracle trials, full-shape
Compute Sanitizer, NVBit, deterministic replay, and five application-replay
NCU runs. Replay and NCU match at 944 target warp instructions; simulator lower
is 818 cycles and NCU median is 3,678. No IBuffer depth is fitted. Artifact:
`docs/sm86_c7_research_probe_front_end_ibuffer_depth_live_v1.json`.

M3 is complete: touched campaign analyzers use the active long-generation
populations, historical short-trace names are no longer execution prerequisites,
and inventory latest-artifact links point to corrected M1/M2 and AQ–AZ records.

C7-S33BB is complete after the M4-E independent closeout recheck: all current
source families have a live vertical or an explicit non-native disposition;
failing, unsupported, and physically unobservable rows remain preserved.
