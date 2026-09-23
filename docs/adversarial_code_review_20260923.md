# Adversarial code review — 2026-09-23

## Scope

This review covers the active cross-architecture additions and their seams into
the production SM86 replay path:

- `sim-arch::spec` and `sim-arch::simt` input boundaries, functional execution,
  timing projection, and typed asynchronous effects;
- `sim-semantics` custom ISA/width/async extensions and the SM86 NVBit adapter;
- the engine sequence-span and queue projection changes, carded issue-resource
  selection, L1/L2 typed-state additions, and model-relative binding-cycle
  emission;
- the independent Python probes, schemas, Vortex configuration/ELF adapters,
  and retained SimX/rtlsim evidence artifacts.

The review treats NCU/rocprof and open-GPU observations as independent
validation targets. It does not promote any model-relative result to a silicon
claim.

## Material findings fixed

1. **Word-lane aliasing and address wraparound.** The first feasibility
   fixtures used a one-byte lane stride while the executor emitted u32 words.
   That represented overlapping stores while the functional map treated each
   start address as an independent word. The retained probes now use a
   four-byte stride; the executor and Python oracle reject overlapping word
   ranges and checked u32 byte-address overflow. Regression tests cover both
   failures.
2. **Unchecked certificate arithmetic.** The compact interval certificate's
   closed-form timing offset used unchecked u128 multiplication/addition for
   adversarially large capacity, interval, and instruction counts. All timing
   products and frontier additions now use checked arithmetic and reify with
   `ArithmeticOverflow`; an extreme synthetic certificate test exercises the
   path.
3. **Static constant-cache metadata.** The SM86 semantic adapter now validates
   cache-operator enum values even when dynamic memory class is unavailable for
   `LDC`/`ULDC`; invalid static-only metadata cannot survive as a seemingly
   valid semantic event. Round-trip and negative tests cover this boundary.
4. **Real-ELF adapter false positives.** The Vortex adapter now requires an
   ordered register-consistent `lw/lw/add/sw` data idiom, rejects floating
   `flw/fadd.s/fsw` kernels explicitly, validates width/address bounds, and
   recognizes the two observed Vortex kernel-entry spellings. It remains a
   deliberately narrow adapter, not a general RISC-V decoder.
5. **Probe hard-coded output shape.** The Vortex cross-model checker exposes
   width and destination arguments (with checked defaults) and rejects output
   vectors of the wrong length instead of silently truncating or indexing a
   different shape.
6. **Schema/runtime integer agreement.** The program schema now bounds
   `add_imm.imm` to the Rust `i32` domain instead of accepting values that the
   runtime parser would reject.
7. **Stateful async proof boundary.** `validate_async_effects` now validates
   every event before checking submit/wait pairing, so malformed non-effect
   records cannot make a sequence proof pass accidentally.
8. **Source-level lint defects.** The descriptor-anchored trace reader,
   digest, lane decoder, and semantic/certificate test seams were cleaned up
   where strict clippy identified concrete issues. Production library clippy
   passes for the reviewed crates.
9. **Word memory model and token scope.** The toy executor now rejects
   unaligned u32 word addresses (including initial-memory entries), and async
   semantic tokens are keyed by launch/warp identity rather than a global group
   number. The deferred `async_store` is represented as a global store with a
   token effect, avoiding a false `cp.async`/global-to-shared interpretation.
10. **SFU readiness attribution.** The SM issue diagnostics had introduced a
    separate SFU readiness cause but accidentally accumulated it into the
    tensor counter. A dedicated serialized `pipeline_sfu` counter and checked
    aggregation path now keep the causal categories disjoint; the readiness
    schema advances to version 26 when that field is present.
11. **Card-resource provenance.** The new SM86 resource block is now explicitly
    documented as source-bound architecture input, not blanket measurement
    evidence; barrier latencies and occupancy ceilings remain reviewed
    hypotheses. This prevents the `architecture_card` wire label from being
    misread as independent silicon calibration.
12. **Dual-issue control-stall bypass.** The first same-warp dual-slot helper
    reset the warp's eligibility to the current cycle, erasing an encoded
    control stall. It was removed; the second carded slot now serves only an
    independently ready warp, with regression tests for both control stalls and
    independent pairing.

The independent timing oracle also now retains per-resource next-issue
frontiers, matching the Rust executor when an asynchronous memory operation is
interleaved with compute before a later memory operation, and the differential
compares every event's resource and cycle fields rather than only a final-cycle
scalar.

## Evidence run after the fixes

```text
cargo fmt --manifest-path sim/Cargo.toml --all -- --check       PASS
cargo test --manifest-path sim/Cargo.toml --workspace --all-targets --quiet  PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-arch --lib --bins -- -D warnings PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-semantics --lib -- -D warnings PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-engine --lib -- -D warnings PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-mem --lib -- -D warnings PASS
cargo clippy --manifest-path sim/Cargo.toml -p sim-trace --lib -- -D warnings PASS
/home/jakeshea/miniforge3/bin/python -m pytest ...             100 passed
```

The custom executable probes and their independent Python differential all
report `functional_match=true` and `timing_match=true`. The six-cell sweep
remains the declared hypothetical result (SIMT widths 4/8/16 × memory issue
widths 1/2; final cycles 26–40). Vortex SimX and rtlsim outputs match the
custom vecadd output, while their retained native regression timing remains
455 versus 397 cycles; that mismatch is recorded rather than hidden.

The retained SM86 current-source replay remains zero-exit with counters and
coverage equal to the retained result. The intentional model-relative
`binding_cycles` metric transition and active scheduler/front-end diagnostic
deltas are documented separately; byte identity with the historical output is
not claimed. The final release-run comparison is retained in
`docs/sm86_current_source_replay_20260923.json` (`binding_cycles=11102`,
`counters_equal=true`, `coverage_equal=true`).

## Bounded residuals (not silently promoted)

- Workspace `cargo clippy --workspace --all-targets -- -D warnings` still
  reports pre-existing assertion-style unwrap/style lints in broad SM/memory/
  trace test modules and large error-enum/argument-count lints in `sim-sm`.
  These are not production-library failures; the reviewed production crates
  above are clean. The ordinary full workspace test suite is green.
- General RISC-V/Vortex decoding, generated compiler/ISS support, cycle-trace
  parity against RTL, measured simulator speed, and held-out physical-silicon
  validation remain explicit future research gates. No current artifact claims
  Curry compatibility, universal configurability, SOTA accuracy, or silicon
  equivalence.

## Decision

The reviewed slice is safe to continue developing: malformed custom inputs
fail closed, functional/timing semantics have an independent oracle, the
SM86 production path remains card-bound, and evidence boundaries are explicit.
The project is ready for the next research gate—generated open-ISA semantics
plus RTL/ISS differential—not for a universal-SOTA claim.
