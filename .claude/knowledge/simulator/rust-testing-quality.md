---
Use when: reading nvasim testing and quality patterns as prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02).
Primary raw source: Effective Rust (Items 29-32) + NeXosim `references/Rust_refs/nexosim/.github/workflows/ci.yml`
Supporting sources:
- NeXosim `.github/workflows/ci.yml` (multi-platform CI with Loom dry-run + MIRI)
- NeXosim `.github/workflows/loom.yml` (dedicated Loom concurrency verification)
- Effective Rust Item 29 (listen to Clippy)
- Effective Rust Item 30 (write more than unit tests)
- Effective Rust Item 31 (use the tooling ecosystem)
- Effective Rust Item 32 (set up CI)
- Async Rust Ch11 (testing concurrent systems, deadlock-free verification)
Repo touchpoints:
- simulator/crates/event-engine/src/engine.rs (unit tests: dispatch, determinism, error paths)
- simulator/crates/sm-model/src/scheduler.rs (GTO/CGTY scheduling tests)
- simulator/crates/sm-model/src/register_file.rs (RF bank conflict tests)
- simulator/crates/memory/src/l1_cache.rs (sectored L1 tests)
- simulator/src/cluster_barrier.rs (barrier state machine tests)
- simulator/tests/ (Python/pytest integration suite)
- simulator/benches/event_engine.rs (criterion benchmark)
Out of scope: current test counts as statements of fact (see dev-patterns), test fixture implementations (see raw code), code changes to add new tests (separate plan).
Version family: Historical parent-repo nvasim SM120 / Blackwell snapshot, 2026-04-16.
---

# nvasim Testing and Quality Patterns

> **Current-scope note:** prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02). The nvasim source itself is historical — do not port it verbatim. NCU is a first-class ground-truth instrument now: its values are prediction TARGETS, never simulator inputs (Rule 1). The active implementation lives in `sim/` (see `plans/phase2_sim_core_progress.md`).

Testing strategy and quality tooling for the nvasim simulator. Documents
the current quality stack first, then recommended future tools. Each
future tool is labeled "future target" with specific nvasim application
targets. Verify current test counts with `cargo test --workspace -- --list`
(Rust) and `pytest simulator/tests/ --collect-only` (Python) — counts
change with every commit.

## Current quality stack (current pattern)

### Rust unit tests

Tests live in `#[cfg(test)] mod tests` at the bottom of each source file.
Key categories by module:

| Module | Coverage focus |
|--------|---------------|
| event-engine | Dispatch ordering, deterministic replay (hash equality), next-event advancement, infinite loop detection, handler error propagation, empty simulation |
| sm-model/scheduler | GTO and CGTY warp scheduling policies, stall handling |
| sm-model/register_file | Bank conflict detection, RF cache hypothesis |
| memory/l1_cache | Sectored cache hits/misses, MSHR allocation |
| cluster_barrier | Barrier state machine, late-wait patterns, multi-phase arrival/release, dual-counter coupling |
| simulator | End-to-end single-kernel runs, CTA activation, determinism hash regression |

Excellent existing patterns worth preserving:
- **Regression tests with pre-fix documentation**: tests that document
  the incorrect behavior before the fix, ensuring the bug cannot regress
- **Fixture builders**: helper functions like `handler_with_active_cta()`
  that construct complex test state without duplicating setup code
- **Determinism hash invariant tests**: run the same simulation twice,
  assert hash equality — catches nondeterminism from HashMap iteration,
  float instability, or ordering bugs

### Python/pytest integration tests

Located at `simulator/tests/`. Exercise the full CLI pipeline:
config loading, trace parsing, simulation, JSON report output, and
NCU comparison. Use `conftest.py` fixtures for shared setup.

### Criterion benchmark

`simulator/benches/event_engine.rs` benchmarks event throughput, queue
operations, and hasher performance. Uses
`criterion::black_box()` to prevent dead-code elimination.

### Clippy gate

`cargo clippy --workspace -- -D warnings` must be clean. This is a
local-only gate (no formal CI pipeline yet).

### Validation scripts

The historical parent snapshot contained shell scripts for single-CTA
build/trace/measure workflows. Its trace-audit mode belonged to that
parent-project pipeline and is not active in this fork.

## proptest — property-based testing (future target)

[proptest](https://crates.io/crates/proptest) generates random inputs
and checks that invariants hold. High-value targets in nvasim:

- **Event queue ordering**: insert events with random `(cycle, sm_id)`
  pairs, verify `pop()` always returns them in `(cycle, sm_id, seq)`
  order — catches Ord implementation bugs
- **Memory coalescing**: random address sets, verify coalesced sectors
  never exceed input address count — catches off-by-one in sector math
- **Determinism**: run simulation twice with random (but identical) config
  overrides, verify hashes match — catches nondeterminism from any source

Effective Rust Item 30: "property-based tests find edge cases that
example-based tests miss."

## insta — snapshot testing (future target)

[insta](https://crates.io/crates/insta) captures output and diffs against
stored snapshots. Primary target:

- **JSON report format**: the 60+ field `format!()` report in `main.rs`
  is exactly the high-value insta target. Any field rename, reorder, or
  format change would be caught by `cargo insta review`.

Secondary targets: config validation error messages, trace audit output.

## cargo-fuzz — coverage-guided fuzzing (future target)

[cargo-fuzz](https://crates.io/crates/cargo-fuzz) feeds random bytes to
a target function and monitors for panics/crashes. Highest-value target:

- **Trace parser** (`trace-parser` crate): the prost protobuf decoder
  processes untrusted binary input. Fuzzing catches malformed traces that
  trigger panics, infinite loops, or excessive memory allocation.

## Loom — concurrency verification (future target)

[Loom](https://crates.io/crates/loom) systematically explores all
thread interleavings for concurrent code. nvasim is currently
single-threaded, so Loom is not immediately applicable.

**When to adopt**: if nvasim adds multi-threaded execution (e.g.,
parallel SM simulation). NeXosim's CI uses `RUSTFLAGS='--cfg nexosim_loom'`
to gate Loom tests, running them on channel, task, broadcaster, and sync
container modules.

**NeXosim reference** (inspiration, not template — nvasim is
single-threaded): `ci.yml` includes a "Loom dry run" job that compiles
with the Loom cfg flag but does not run full exploration in CI (too slow).
A separate `loom.yml` workflow runs full Loom verification on demand.

## MIRI — memory safety verification (future target)

[MIRI](https://github.com/rust-lang/miri) is Rust's interpreter for
detecting undefined behavior, including but not limited to:
- Use-after-free, out-of-bounds access, uninitialized memory reads
- Strict provenance violations (pointer-integer casts)
- Data races (when applicable)

MIRI is broader than Loom — it checks memory safety in both
single-threaded and multi-threaded code. NeXosim runs MIRI with
`-Zmiri-strict-provenance -Zmiri-disable-isolation` at both 1 CPU and
4 CPUs, exercising 8+ example programs.

**nvasim application**: run `cargo miri test --workspace` to check unit
tests for UB. Note: `sim_config.rs` tests use `unsafe { std::env::set_var/
remove_var }` behind a mutex. The Rust standard library documents these
as fundamentally unsafe in multi-threaded programs (concurrent readers
outside the mutex are not protected). This is a known unsafe compromise
in the test suite, not a justified safe pattern — a future cleanup target
is to replace env-var mutation with a config-injection approach that
avoids `unsafe` entirely.

## Clippy pedantic lints (future target)

Current gate: `cargo clippy --workspace -- -D warnings`. Future
enhancement (Effective Rust Item 29):

```toml
# Cargo.toml or clippy.toml
[lints.clippy]
correctness = { level = "deny" }
suspicious = { level = "deny" }
cast_possible_truncation = { level = "warn" }   # numerical code
cast_sign_loss = { level = "warn" }              # numerical code
```

Cherry-pick `clippy::pedantic` lints relevant to numerical simulation
code rather than enabling the full pedantic set.

## CI pipeline design (future target)

nvasim currently has local-only quality gates. NeXosim's CI (inspiration,
not template — nvasim is single-threaded and has no WASM target) provides
a reference pipeline:

| Stage | NeXosim | nvasim equivalent |
|-------|---------|-------------------|
| Check | `cargo check` on stable + MSRV | `cargo check --workspace` |
| Test | `cargo test` on Linux/Windows/ARM | `cargo test --workspace` |
| Loom dry run | `--cfg nexosim_loom` compile-only | Skip until MT added |
| MIRI | Strict provenance, 1 + 4 CPUs | `cargo miri test --workspace` |
| Clippy | `cargo clippy --tests --all-features` | `cargo clippy --workspace -- -D warnings` |
| Format | `cargo fmt --all -- --check` | Same |
| Docs | `cargo doc --no-deps` | Same |
| Benchmark | Not in CI | `cargo bench` (regression detection) |

## Benchmarking practices (current pattern + future target)

See `rust-performance.md` for the full profiling workflow, criterion
benchmark details, and future targets (divan, flamegraph, regression
detection). The CI pipeline table above references benchmarking as a
future CI stage.

## See also

- `.claude/knowledge/simulator/nvasim-dev-patterns.md` — current test conventions and CI gate.
- `.claude/knowledge/simulator/rust-type-safety.md` — type safety patterns that proptest invariants should verify.
- `.claude/knowledge/simulator/rust-performance.md` — profiling workflow and allocation discipline.

## Escalate to raw when

- You need the full NeXosim CI pipeline — see `references/Rust_refs/nexosim/.github/workflows/ci.yml`.
- You need the Loom verification setup — see `references/Rust_refs/nexosim/.github/workflows/loom.yml`.
- You need proptest strategy examples — see Effective Rust Item 30.
- You need the current test fixture implementations — see `simulator/crates/event-engine/src/engine.rs` tests and `simulator/src/cluster_barrier.rs` tests.
- You need the pytest integration suite structure — see `simulator/tests/conftest.py`.
