# General GPU simulator: research and implementation roadmap

Recorded 2026-09-23; approved direction from the user discussion. This is the
canonical continuation plan. The current milestone is an executable custom
architecture seam alongside, not in place of, the SM86 replay.

## Outcome and research hypothesis

Build a configurable GPU research platform that users can extend to their own
architectures. Aim for ISCA/MICRO/HPCA-quality work and eventually a competitive
community simulator, without claiming universal superiority from one metric.

**Hypothesis, not established novelty:** a typed GPU architecture contract and
causality-preserving multi-resolution runtime can connect closed-vendor trace
replay, open/custom ISA execution, detailed event timing, tile-level models,
and optional RTL differential checking at lower extension cost and with
explicit accuracy limits.

The method must preserve lane/warp semantics, dependencies, memory visibility,
resource contention and asynchronous completion; it must enable a consequential
architecture result. A generic schema, Rust rewrite, extra validators, second
ISA, or event skipping alone is not a top-tier contribution.

## Baseline and what the evidence means

- Preserve NVBit/SASS -> sim-trace -> sim-sm -> sim-mem -> sim-engine -> sim-run.
- SM86 is the implemented silicon target; SM120 remains exploratory.
- The retained 15-workload comparisons are development evidence, not a new
  held-out timing study. Their ~9.7% maximum scored error is not comparable to
  another paper's elapsed-cycle MAPE. Trace-derived instruction equality is
  accounting correctness, not independent instruction prediction.
- The current runner emits a **model-relative binding-cycle diagnostic**;
  this does not establish hardware-equivalent elapsed-cycle fidelity.
- Event/dense replay equality is evidence for model-relative execution
  equivalence, not independent silicon accuracy or measured wall-clock speedup.
- The descriptor probe's 26 -> 19 count is not production queue reduction.
- Current source includes substantial pre-existing uncommitted work beyond
  HEAD `fec7e2f2`; preserve it and record the actual patch/configuration used.

## Scope and assumptions

Keep the SM86 backend and existing evidence-bound silicon card format intact.
Start a separate custom-model specification seam; do not weaken the legacy
card loader to accept arbitrary vendor identities or fill unknowns with
Ampere constants. User-designed models may execute with explicit hypothetical
parameters: lack of silicon calibration is a claim limit, not a ban on running.

Start with a small original SIMT example, followed by an actual Vortex/Ventus
adapter if the pilot is useful. An illustrative example is **not** Curry GPU
compatibility: the exact Zhihu article/source was not verified in the research
pass. Do not invent its ISA. AMD hardware-dependent validation remains deferred
until a device or suitable retained inputs are available; it does not block
the open/custom architecture work newly authorized by the user.

No external publication, license selection for other people's code, Git tag,
purchase, or privileged GPU reconfiguration is implied by this plan. Resolve
those choices when needed. Do not rewrite historical measurements or install
review/approval loops. Dated pointers update historical context without
changing old experimental records.

## Work sequence and practical completion evidence

| Stage | Change | Evidence required |
| --- | --- | --- |
| 0. Reproducible SM86 reference | Fix stale fixtures caused by known card changes; retain commands, build/input identities and representative replay results. Assess distribution license/dependency obligations before public release. | A representative current-source replay, affected tests, separate cycle/event/counter diagnostics; later absolute wall-time, KIPS and peak-memory measurements. No blanket claim that the dirty worktree is a released baseline. |
| 1. User-defined architecture seam | Add opaque architecture/ISA identity, SIMT execution shape, named resources and declared assumptions. Keep production card consumers unchanged. | Parse a non-SM86 example; execute the small supported path using the spec; changing SIMT width changes functional lane cardinality, while resource fields are preserved for a future timing backend; malformed/unsupported input is rejected. |
| 2. Open/custom functional execution | Define a small executable ISA backend and program/input format; later adapt an independently developed open GPU such as Vortex. Separate functional semantics from timing. | Correct outputs across legal scheduling/configuration variants; deterministic termination; compare to independent ISS/RTL where available. A toy functional model is only feasibility evidence, not cross-vendor validation. |
| 3. Shared semantics and one async mechanism | Extend/adapt existing sim-semantics rather than create a competing vendor-neutral IR. Choose one DMA/barrier or producer-consumer mechanism. Preserve native extensions and explicit information loss. | Functional/trace paths agree on represented effects; exact vs accelerated execution has the declared observation equivalence; contention/unknown effects use detailed execution. |
| 4. Independent performance validation | Separate calibration/development/held-out workloads and architecture configurations; run current relevant baselines. | Per-mechanism and held-out time/counter errors, coverage/failures, wall time, peak memory, throughput, ablations and extension cost. |
| 5. Architectural result and artifact | Study one scheduler/cache/async-resource design decision on a declared model family. Add tile/analytical/parallel modes only where measurements justify them. | A useful design conclusion checked independently, uncertainty/sensitivity reported, clean reproducible artifact and clear prior-art attribution. |

**Current progress:** the bounded `simt_u32_v1` executor now runs one
architecture-matched program using the spec's SIMT width, lane IDs, arithmetic
and per-lane stores. It is a functional feasibility oracle only; it makes no
timing, compiler, memory-consistency, or Curry/Vortex compatibility claim.

The positive example emits eight lane results and eight distinct stores. An
ISA identity mismatch exits nonzero with a typed diagnostic. The workspace
tests remain green after this addition.

The executor now also has one explicit asynchronous mechanism: `async_store`
records a pending group and `wait_group` resumes only at the memory resource's
declared completion cycle. This is a deterministic functional/timing probe,
not a claim about a physical GPU's DMA/barrier implementation.

`tools/arch_spec_design_sweep.py` now varies SIMT width and memory issue width
and records final-cycle/event/semantic-event outputs in
`docs/custom_architecture_design_sweep_20260923.json`. This is a reproducible
model-relative design probe, not silicon validation.

The public Vortex `VX_config.toml` shape was imported into an unvalidated spec
by `tools/vortex_config_to_spec.py`. The current executor correctly rejects a
Vortex program because its ISA/architecture adapter is not implemented; no
Vortex SimX/RTL build was silently substituted.

An attempted upstream Vortex configure on this CachyOS host exited
`Unsupported OS` before building, and the Vortex toolchain is absent. The
independent SimX/RTL comparison therefore remains an explicit external
environment blocker rather than an unreported missing experiment. A bounded
Ubuntu 22.04 container build subsequently installed the Vortex toolchain,
built SimX and rtlsim, and ran the same `vecadd.vxbin` on both; the functional
outputs matched. The ELF static adapter classified 6,101 instructions while
preserving 440 `.4byte` custom/unknown encodings. Cycle-trace comparison and a
Vortex binary-to-native semantic adapter remain open. A hand-authored
load/add/store custom program using the same vecadd inputs now matches all 16
SimX/rtlsim outputs; this is a cross-model functional differential, not proof
that arbitrary Vortex binaries execute in the custom backend. A narrow
`vortex_vecadd_elf_adapter.py` now recognizes the compiled Vortex kernel's
zero-offset register-consistent `lw/lw/add/sw` idiom and emits a checked custom
program with caller-supplied synthetic addresses; its output matches the
independent Python oracle. It does not decode ELF pointer semantics, and
general RISC-V decoding remains open.

The current-source SM86 replay check is recorded in
`docs/sm86_current_source_replay_20260923.md`; retained counters, coverage and
metric predictions remain equal except for the intentional
`required_unimplemented` -> model-relative `binding_cycles` transition; internal
readiness diagnostics also changed with the active scheduler revision.

Vortex's native regression `vecadd` now supplies a repeated timing parity
probe: SimX is deterministic at 112 instructions/455 cycles and rtlsim at
112/397 cycles over three runs. Both pass functionally; the 14.6% timing
difference is recorded as a future parity investigation, not silicon accuracy.

The narrow Vortex ELF path now projects its recognized integer idiom through
the Rust shared semantic-event representation. `docs/vortex_elf_shared_semantic_probe_20260923.json`
records six validated events, typed global memory families, width 16, and the
hypothetical final cycle 112. Synthetic address binding and unsupported
general-RISC-V semantics remain explicit limitations.

### Remaining completion gates

The software feasibility slice is exercised and regression-clean. The roadmap
remains active for three stronger gates that cannot be inferred from the
current results:

1. generalize the narrow real-Vortex integer pattern bridge into a real
   RISC-V/Vortex decoder with pointer/dataflow semantics; the current
   recognized ELF idiom already crosses the shared semantic-event path, but
   remains synthetic-address and pattern scoped;
2. compare SimX/RTL timing traces or counters after a stable Vortex trace
   interface is selected; and
3. perform held-out silicon validation and a design conclusion on an available
   physical GPU without using its scored observations as simulator inputs.

Do not add every ISA, a general compiler generator, full-system CPU coupling,
multi-GPU fabric, graphics, or all Hopper features at once. Move one real path
through the seam, run it promptly, then expand only where evidence demands it.

## Evaluation design

1. Calibrate mechanism parameters with independent microbenchmarks/documented
   inputs, never the scored workload's NCU/rocprof measurements. Keep scored
   rows out of features, parameter tuning and implementation selection.
2. Distinguish functional correctness, model-relative cycle equivalence and
   silicon timing fidelity. Compare the same observable, clock domain and
   launch scope; retain issue, event, completion, launch and tail evidence
   separately. Do not collapse them into a scalar selection score.
3. Compare modern NVIDIA behavior against applicable Accel-Sim 2.0,
   FlashGPU-sim and Huerta baselines; AMD against the supported MGPUSim/gem5
   target; custom/open designs against their native ISS/RTL. Same binaries,
   inputs, features and model detail where possible; report unsupported cells.
4. Report absolute throughput, wall time, RAM, trace/preprocessing cost, and
   serial/parallel resource use, not only virtual-cycle reduction. A same-code
   dense lane is an optimization check, not an independent hardware oracle.
5. Quantify extension effort with a real second backend and user modification,
   not just a count of configuration fields. Test changes that exercise
   scheduling, dependencies, memory and functional behavior.
6. Include a negative/control case: an unsupported instruction, unstable trace
   under changed execution, or an omitted timing effect must be reported, not
   converted to generic ALU or a workload-specific correction.

## Research sources and claim boundaries (checked in the 2026-09-23 discussion)

- [Accel-Sim 2.0 official release](https://accel-sim.github.io/) and
  [preprint](https://arxiv.org/abs/2608.22602): modern async NVIDIA modeling,
  hardware correlation and broad workloads; public release and paper scope
  differ for Blackwell/multi-GPU. Reported H100 errors are ~13.4–13.5%; these
  were not reproduced here.
- [FlashGPU-sim](https://arxiv.org/abs/2609.15311): author-listed to appear at
  MICRO 2026; PTX execution, modern async mechanisms and Triton. Reported 5.24%
  cycle MAPE on 131 configurations is a paper result, not a local replication.
- [MGPUSim ISCA 2019](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf)
  and [current project](https://github.com/sarchlab/mgpusim): modular AMD
  execution/multi-GPU simulation; the historical 5.5% average is GCN3/R9 Nano,
  not current MI300X validation.
- [NVArchSim HPCA 2021](https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf):
  hybrid trace/execution and selective detail are established prior art.
- [Huerta MICRO 2025 artifact](https://github.com/upc-arco/modern-gpu-simulator-micro-2025):
  modern sub-core/control-bit modeling and parallelization are prior art.
- [Vortex](https://github.com/vortexgpgpu/vortex) and
  [Ventus](https://github.com/THU-DSP-LAB/ventus-gpgpu): open GPU hardware,
  functional/timing models and software stacks provide independent targets.
- [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262),
  [TileSight](https://arxiv.org/abs/2607.22432),
  [LPGSim](https://doi.org/10.1145/3805642), and
  [Swift-Sim](https://doi.org/10.23919/DATE64628.2025.10992801): tile, analytical,
  hybrid and parallel simulation already occupy distinct speed/accuracy points.
- [Akita](https://github.com/sarchlab/akita),
  [VADL](https://arxiv.org/abs/2402.09087) / [OpenVADL](https://github.com/OpenVADL/openvadl),
  and [TAIDL](https://github.com/act-compiler/taidl): modular engines and ISA/tool
  generation are not novel by themselves. OpenVADL's released generator
  coverage is narrower than the full VADL vision.

## Milestone memory

**2026-09-23, first slice complete:** goal created; canonical plan recorded;
SM86 source and historical captures preserved. `sim-arch::spec`, the
`simt_u32_minimal.json` example, schema, and `architecture_spec` executable
now provide a bounded, fail-closed custom-shape path. The stale launch-
evidence seal was refreshed to the canonical value induced by the active card
schema (not by weakening checking). `cargo test --workspace --all-targets`
passes; one machine-local sidecar/trace test is intentionally ignored. No
independent silicon timing or SOTA performance claim has been established.

**2026-09-23, functional slice:** `sim-arch::simt` and
`custom_simt` now execute the matching four-instruction `simt_u32_v1` example.
The output demonstrates lane-specific arithmetic and stores; identity and
unsupported-ISA mismatches fail closed. This is a functional oracle feasibility
result only, not timing, compiler, memory-consistency, RTL, or silicon evidence.

**2026-09-23, roadmap vertical slice complete:** custom functional and
resource-timing execution, shared typed semantic events with explicit async
submit/wait effects, independent Python differential, six-cell design sweep,
Vortex SimX/rtlsim functional/timing evidence, static ELF summary, narrow real
ELF adapter, and current-source SM86 replay check are all recorded. General
RISC-V decoding and held-out physical-silicon validation remain future research
gates, not hidden assumptions.

**2026-09-23, adversarial review closure:** the cross-architecture seam was
reviewed for malformed-input behavior, arithmetic overflow, memory aliasing,
semantic round-trip integrity, adapter false positives, and evidence-claim
boundaries. Word-address overlap/wraparound, unchecked certificate timing
products, invalid static cache operators, broad ELF idiom matching, and
hard-coded cross-model output shape were fixed with negative tests. The full
workspace test suite and reviewed production-library clippy checks pass. The
bounded residuals and exact commands are recorded in
`docs/adversarial_code_review_20260923.md`; broad all-target clippy still has
historical test-style and `sim-sm` lint findings and is not presented as green.

The final resumed scheduler audit also removed an unsafe same-warp dual-issue
eligibility reset that erased control stalls, kept SFU readiness separate from
tensor readiness (schema 26 when present), and reran the release SM86 source
replay: counters/coverage remained equal and the model-relative binding-cycle
value was `11102`. These are simulator soundness fixes, not new silicon claims.
