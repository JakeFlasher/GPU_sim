# Adversarial review and research synthesis — 2026-09-24

Continuation state is summarized in [PROJECT_MEMORY.md](PROJECT_MEMORY.md);
current implementation/evidence status is in [STATUS.md](STATUS.md).

## Scope decision

The project is viable, but the current repository is **not** a SOTA or
universal GPU simulator. Its defensible scope is a software architectural
timing simulator:

```text
NVBit/SASS trace + architecture card/profile -> Rust trace/SM/memory replay
                                             -> NCU-shaped output
```

NCU/CUPTI observations on machines the team controls are independent labels
and validation targets. They are not simulator inputs. RTL, Verilog/Verilator,
FPGA, physical design, gate-level, and internal-silicon simulation are outside
this scope. The custom architecture path is now a bounded `sim-sm::custom`
dependency/resource backend, not an arbitrary ISA, native SASS backend, or
silicon timing model.

The right near-term paper thesis is therefore:

> A trace-driven GPU timing model can expose a real, user-editable resource
> intervention and improve the accuracy–speed/design-space trade-off on a
> declared SM86 workload boundary, while remaining explicit about fixed-trace
> limits and unsupported behavior.

That is narrower than “defeats every simulator,” but it is falsifiable and
compatible with the evidence the repository can actually collect.

## Blocker closure update

The three blockers identified in the first review are now closed at their
declared scope:

1. **Same-launch mapping:** the live RTX 3090 Ti pair closes
   `smsp__inst_executed.sum` ↔ `instructions_issued` at 197,120; cycle timing
   remains separately labeled as a crosswalk because tails are incomplete.
2. **Production custom seam:** `--architecture-spec` feeds a trace-safe SM86
   issue profile into `sim-run`, and `sim-sm::custom` executes bounded
   declarative groups through the shared event engine.
3. **Pipeline overlap:** dependency/resource scheduling now permits independent
   ALU/memory overlap, enforces producer readiness, tracks initiation interval
   and in-flight capacity, and emits completion events.

The fair numerical Accel-Sim/MGPUSim comparison remains intentionally deferred.

## What the review found

### Fixed in this pass

1. **Unmatched ITS `BSYNC` was accepted.**
   `sim/arch/src/its.rs` treated an unarmed barrier as fall-through, so a
   program containing only `BSYNC; HALT` exited successfully. It now returns
   `InvalidBarrierUse`, with a focused regression case.
2. **Direct replay ignored carded scheduler resources.**
   `sim/direct_run` reconstructed issue parameters from legacy latency values,
   ignoring `smsp.issue_ports`, occupancy, barrier resources, and dual-issue
   fields in a user card. It now consumes `IssueModelParams::from_card` when a
   card contains the resource blocks, matching the campaign path; the old
   resource-less fixture remains an explicit legacy compatibility path and
   partial resource blocks fail closed.
3. **Issue-port overrides could silently widen.**
   A card with dual issue enabled could still issue two transactions when the
   user requested `--hypothetical-issue-ports 1`. The override now disables the
   independent dual-issue admission path and validates the untouched card
   fields separately.
4. **NCU launch mapping is now closed for one count observable.**
   A fresh locked-clock campaign on the live RTX 3090 Ti completed correctness,
   NCU, NVBit, static-control, and replay stages. Three NCU application-replay
   rows and the matching simulator launch all report 197,120 executed
   instructions. The cycle crosswalk remains explicitly unpromoted because
   dirty-writeback/coherence tails are incomplete.
5. **The custom path now reaches the shared event engine.**
   `sim-sm::custom` and `custom-program-run` lower declarative `simt_u32_v1`
   programs into dependency/resource tasks, admit resident groups, emit
   completion events, and reject inter-group memory conflicts. This closes the
   prior “standalone toy only” integration gap without claiming arbitrary ISA
   semantics.
6. **Stale scope language was demoted.**
   The old code-review memo, the older conference memo, and the cross-project
   historical note now carry explicit archive/superseded banners and no longer
   advertise RTL/ISS differential or MI300X work as active gates. A stale skill
   link to a missing phase-2 plan now points to current status/speed evidence.
7. **Architecture-profile input is generation-anchored.**
   The new `--architecture-spec` path now uses the same no-follow,
   post-snapshot generation check as cards and control inputs; symlinked or
   replaced profiles fail before replay.
8. **Address-space semantics now have a bounded shared-memory slice.**
   `load_shared`/`store_shared` preserve separate functional state and
   semantic global/shared families, and their dependency domains cannot alias
   global addresses accidentally. The differential probe covers both state
   and timing projections.
9. **Group synchronization now has a bounded CTA slice.**
   Matching `group_barrier` generations add cross-group completion/release
   dependencies and require concurrent residency. Separated shared-memory
   producer/consumer values propagate through the shared map. Mismatched
   participation, insufficient residency, and unordered conflicts fail closed;
   inter-group atomics, arbitrary reductions, and weak-memory outcomes remain
   unsupported.
10. **Deterministic shared atomic RMW is covered.**
    `atomic_add_shared` performs lane-ordered u32 read-modify-write, preserves
    old values in destination registers, and emits an explicit atomic semantic
    direction. It does not claim hardware scope or throughput.
11. **One fixed lane-exchange collective is covered.**
   `broadcast_lane0` now has functional, dependency/timing, semantic, and
   differential coverage. It is intentionally not a general cross-lane or
   crossbar-throughput model.
12. **One fixed lane-reduction collective is covered.**
    `reduce_add` now has functional, dependency/timing, semantic, and
    differential coverage; arbitrary reductions and hardware throughput remain
    unsupported.
13. **The independent oracle had two blind spots.**
   It now compares shared-memory state, treats shared atomics as a separate
   address domain for pending-async checks, and mirrors explicit dependency,
   initiation-interval, and in-flight-capacity timing fields. Architecture
   specs also bound resident capacity to the 64-group event-engine limit.
14. **Initial address-space state was accidentally group-private without a barrier.**
   `sim-sm::custom` now seeds every group from common CTA-visible initial
   global/shared maps for both synchronized and conflict-free programs; a
   group can read words initialized by another group's input without a
   spurious missing-address failure.
15. **Phase conflict tracking could overwrite an earlier shared phase.**
   Accesses are now keyed by `(address-space, word, barrier phase)` while
   global conflicts remain cross-phase; same-phase writes cannot be hidden by
   a later barrier generation. Focused regressions cover both same-phase
   shared writes and global read/write conflicts separated by a barrier.

### Remaining correctness and claim risks

* `ArchitectureSpec` is not a general architecture description language. The
  production `ArchId` is a closed Rust enum, the custom executors hard-code
  `simt_u32_v1`/`simt_its_v1`, timing recognizes resources named `alu` and
  `memory` by default (with explicit operation bindings), and `instruction_set`
  is an allow-list rather than user-defined instruction semantics.
* The standalone ITS executor remains single-group, but the bounded
  `simt_u32_v1` `sim-sm::custom` backend now implements resident-group
  admission/completion for disjoint memory effects. Neither path is an
  arbitrary ISA or vendor-SASS frontend.
* Custom timing now tracks per-resource issue and register-ready frontiers, so
  independent ALU/memory operations overlap in the bounded probe. It still
  lacks multi-warp scheduling, arbitrary instruction semantics, and a
  production SM scoreboard; see `custom_pipeline_overlap_20260924.md`.
* The custom spec/program path is now consumed by `sim-sm::custom` and the
  `custom-program-run` binary. It remains a separate model-relative frontend;
  it does not silently replace native NVBit/SASS replay.
* The fixed SASS trace path can safely vary timing/resource service choices,
  but it cannot regenerate data-dependent control flow, addresses, atomics,
  synchronization participants, or values. Every new knob needs a
  `trace_safe`, `trace_bounded`, or `requires_execution` classification.
* A universal vendor-independent binary decoder and silicon-equivalent timing
  reconstruction are not completion obligations. Native adapters consume
  available NVBit/SASS evidence; NCU/CUPTI values remain independent labels.
* `BindingCycles`, event/dense equality, and frontier counts are
  simulator-relative diagnostics. They are not elapsed-time or silicon-cycle
  accuracy until a same-launch NCU observable and clock/cache/replay boundary
  are explicitly mapped.
* The current SM86-only result cannot be ranked globally against AMD MGPUSim,
  AMD NaviSim, Hopper Accel-Sim 2.0, or Blackwell/Hopper FlashGPU-sim. A
  baseline cell is valid only when hardware/profile, trace/program semantics,
  metric definition, and workload scope overlap.

## Prior-art boundary

### Accel-Sim and GPGPU-Sim

[Accel-Sim ISCA 2020](https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf)
establishes the modern NVIDIA trace baseline: NVBit native SASS traces retain
active masks, PCs, execution-unit information, and memory addresses, then feed
an ISA-independent IR and a detailed GPGPU-Sim timing model. Its correlator and
tuner produce hardware-counter comparisons. Trace replay avoids a functional
ISA implementation, but cannot answer interventions that change value-dependent
control flow or synchronization; execution mode is needed for those cases.

The current [official Accel-Sim 2.0 release](https://accel-sim.github.io/)
claims Hopper TMA, WGMMA, `mbarrier`, clusters, chiplet/uGPU memory, compressed
traces, vLLM/PyTorch hooks, and GPUVision; it reports 99% Pearson correlation
and 13.4% mean absolute cycle error over 34,000+ H100 kernel instances. These
are release/preprint claims and not local replications, but they define the
modern comparison boundary. A fixed SM86 result must not claim to supersede
them globally.

[GPGPU-Sim's manual](https://gpgpu-sim.org/manual/index.php/Main_Page)
remains useful historical/configurability prior art, but its classic model is
not a current commercial-GPU accuracy baseline.

### MGPUSim and NaviSim

[MGPUSim ISCA 2019](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf)
demonstrates modular event-driven AMD GCN3/multi-GPU simulation, extension
without modifying existing components, parallel simulation, and validation
against locked-clock R9 Nano hardware. Its reported 5.5% average error and
3.5×/2.5× functional/timing speedups are architecture/workload-specific. The
[current project](https://github.com/sarchlab/mgpusim) warns that stable support
is AMD GCN3; NVIDIA work is ongoing.

[NaviSim PACT 2022](https://sarchlab.org/navisim_pact_2022.pdf) is a useful
warning against knob-only generality: changing public GCN3 parameters to stand
in for RDNA produced errors as high as 318% and wrong trends. A general GPU
simulator therefore needs ISA/wavefront/cache semantics and component
connections, not just more scalar fields in a JSON card.

### Speed and fidelity frontiers

[NVArchSim HPCA 2021](https://d1qx31qr3h6wln.cloudfront.net/publications/Need_for_Speed_Experiences_Building_a_Trustworthy_System-Level_GPU_Simulator.pdf)
frames architectural simulation around three independent goals: accuracy on
existing hardware, trustworthiness for novel designs, and fast turnaround. It
uses selective fidelity and hybrid trace/execution rather than maximal cycle
detail everywhere.

[Huerta and González (2025)](https://arxiv.org/abs/2502.14691) report 5.8×
average and up to 14× speedup for deterministic 16-thread parallel Accel-Sim.
[HyFiSS](https://github.com/ConvolutedDog/HyFiSS) and [Photon](https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf)
show sampling/hybrid fidelity as separate speed research axes.

[FlashGPU-sim](https://arxiv.org/abs/2609.15311) is an emerging preprint for
modern Triton/async GPU workloads; it reports 5.24% cycle MAPE over 131
RTX-5090/H100/B200 configurations and 7.86× at 16 host threads. Treat those
numbers as contextual, not a locally verified ranking.

### User-defined architecture descriptions

[VADL](https://arxiv.org/abs/2402.09087) and
[OpenVADL](https://github.com/OpenVADL/openvadl) demonstrate that a real
architecture-description language separates ISA and microarchitecture and can
generate functional/cycle-approximate tools, compilers, and other artifacts.
This makes a schema plus a hand-written toy executor insufficient novelty by
itself. The GPU-specific contribution must be the trace/semantic contract,
SIMT scheduling/resource model, or an accuracy–speed mechanism demonstrated in
an architecture decision.

## Conference-quality evidence bar

MICRO and HPCA guidance accepts modeling/simulation and allows evaluation to be
proportional to a genuinely new area, but it does not make unsupported claims
credible. The minimum useful package here is:

1. exact GPU identity, driver/tool versions, clocks, cache/replay settings, and
   launch scope;
2. a matching NVBit trace/control input and card/profile;
3. one declared NCU observable with a documented simulator mapping (or an
   explicit mapping-blocked result);
4. per-workload error distribution plus outliers, not one tuned headline;
5. replay-only wall time, trace/preprocessing cost, host threads, RSS, modeled
   events/instructions, and virtual cycles;
6. one production resource intervention and an ablation of its mechanism;
7. explicit unsupported features and fixed-trace safety class; and
8. one architecture conclusion whose trend changes or becomes affordable.

Tests, hashes, receipts, and approval workflows are not scientific evidence.
Keep only small executable smoke/invariant checks that expose a concrete code
failure; do not turn them into completion gates. Content identities may be
retained when needed to bind a trace/card/result for reproducibility, but they
are not authority over the project.

## Recommended completion sequence

1. **Close one measurement boundary.** The instruction-count mapping is closed
   for one launch; keep the timing frontier separate until retirement/coherence
   tails are modeled.
2. **Keep the direct/campaign paths aligned.** The direct card fix in this pass
   removes one silent divergence; add no second configuration path without a
   named source label.
3. **Use one trace-safe production intervention.** Issue width is now wired;
   document its effect on issue/completion/event/launch frontiers and keep the
   override explicitly hypothetical.
4. **Measure speed honestly.** Separate trace loading, replay, and output
   serialization; report wall time/RSS. Retire event skipping as a speed claim
   if it does not improve process time.
5. **Only then expand the custom seam.** The production-consumable issue
   profile, dependency-aware overlap, bounded resident groups, shared-memory
   barriers, atomics, and two collectives are now present. Any future
   extension should be mechanism-driven (for example a concrete multi-warp
   scoreboard question), not another toy ISA or vendor layer.
6. **Compare baselines only on valid cells.** If no baseline shares the same
   hardware/trace/observable, publish context and a negative overlap result,
   not a ranking.

## Executable evidence from this review

The focused changes were checked with:

```text
cargo test --manifest-path sim/Cargo.toml -p sim-arch --lib --quiet       PASS (62)
cargo test --manifest-path sim/Cargo.toml -p sim-run --lib --quiet       PASS (34)
cargo test --manifest-path sim/Cargo.toml -p sim-run-campaign --quiet   PASS (68 + dependent groups)
cargo clippy --manifest-path sim/Cargo.toml -p sim-arch --lib --bins -- -D warnings PASS
cargo test --manifest-path sim/Cargo.toml --workspace --all-targets --quiet PASS (all workspace targets; machine-local tests remain ignored)
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt_its -- \
  arch_profiles/specs/simt_its8.json arch_profiles/programs/simt_its8_diamond.json PASS
release `sim-run` on the retained canary: PASS (1 launch, 54 instructions,
378 model-relative issue cycles, ~7 ms wall time on this host)
`custom-program-run` with `simt_u32_pipeline2.json` and two disjoint programs:
PASS (2 resident groups, final cycle 45, 10 completion events)
`custom-program-run` producer/consumer barrier: PASS (final cycle 90,
consumer register value 100); same-phase shared-write regression: PASS
(rejected); initial shared-state-without-barrier regression: PASS.
Independent functional/timing differentials for shared memory, atomics,
lane-0 broadcast, reduction, async overlap, and initiation/in-flight timing:
PASS.
```

The full-workspace run is retained as a bounded engineering signal, not a
scientific validation claim. A machine NCU/trace run remains required before
any final accuracy statement.
