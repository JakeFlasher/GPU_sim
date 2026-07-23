---
Use when: reading parent-repo nvasim Rust conventions as PRIOR-ART reference for the ACTIVE simulator build (re-founded 2026-07-02).
Primary raw source: simulator/ codebase analysis (Cargo.toml, src/main.rs, crate lib.rs files)
Supporting sources:
- simulator/src/sim_config.rs (config patterns)
- simulator/crates/sm-model/src/subcore.rs (crate organization example)
- simulator/tests/ (Python integration suite)
Repo touchpoints:
- simulator/Cargo.toml (workspace deps)
- simulator/src/main.rs (CLI, error handling, report output)
Out of scope: porting nvasim source verbatim (design from cards instead; this card is PRIOR-ART reference for the ACTIVE simulator build, re-founded 2026-07-02), nvasim architecture overview (see simulator/nvasim-architecture.md), CUDA/PTX semantics (see ptx/ cards).
Version family: Historical parent-repo snapshot as of 2026-04-16.
---

# nvasim Development Patterns

> **Current-scope note:** prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02). The nvasim source itself is historical — do not port it verbatim. NCU is a first-class ground-truth instrument now: its values are prediction TARGETS, never simulator inputs (Rule 1). The active implementation lives in `sim/` (see `plans/phase2_sim_core_progress.md`).

Conventions and patterns observed in the historical nvasim codebase. Items
marked "future target" were future work in that parent-repo context, not tasks
for this fork.

## Error Handling

Two primary patterns, separated by crate boundary:

| Layer | Crate | Pattern | Example |
|-------|-------|---------|---------|
| Top-level binary | `nvasim` (main.rs) | `anyhow::Result` with `.context()` for human-readable error chains | `toml::from_str(&config_str).context("failed to parse config TOML")?` |
| Event handler trait | event-engine | `Result<(), Box<dyn std::error::Error + Send + Sync>>` — handler errors are opaque to the engine | `EngineError` wraps handler errors via manual `impl std::error::Error` |
| Crate-internal errors | trace-parser, event-engine | Manual `impl std::fmt::Display` + `impl std::error::Error` structs (e.g., `TensorSuffixError`, `EngineError`) | See `event-engine/src/engine.rs:67-94`, `trace-parser/src/tensor_suffix.rs:119-129` |

**Note**: `thiserror` is declared in `[workspace.dependencies]` but the current
crate error types use hand-written impls, not `#[derive(thiserror::Error)]`.
Adopting thiserror for existing types is a future cleanup, not current practice.

**Convention**: functions in `main.rs` return `anyhow::Result<()>`. Crate-internal
functions return typed errors. Never `unwrap()` in production paths; use
`.context("explanation")` or `.expect("invariant: ...")` for truly unreachable paths.

**Known gap**: `simulator.rs:52-56` uses `panic!()` for uncalibrated tensor
precision — future target is `anyhow::bail!()`.

## Configuration (serde + TOML)

Config is deserialized via `serde::Deserialize` with `#[serde(default)]`
for optional fields. The flow:

1. Read TOML file → `toml::from_str::<SimulatorConfig>()`
2. Apply `NVASIM_*` env-var overrides via `sim_config::apply_env_overrides()`
3. Call `config.validate()` to catch invalid combinations

**Env-var override mechanism**: nvasim uses an explicit allow-list of
`NVASIM_*` keys (defined in `sim_config.rs:KNOWN_ENV_OVERRIDE_KEYS`). Only
three keys override `SimulatorConfig` fields: `NVASIM_SCHEDULER_POLICY`,
`NVASIM_IBUFFER_ENTRIES`, `NVASIM_REGISTER_FILE_BANKS`. Two additional keys
(`NVASIM_KERNEL_REGISTERS_JSON`, `NVASIM_KERNEL_REGISTERS_ALLOW_NO_MATCH`)
are consumed in `main.rs` for the per-kernel register sidecar. Any other
`NVASIM_*` key is a hard error (prevents silent typos).
Overrides are logged to stderr unless `--report-json` is active.

**Historical convention**: all config fields were documented with inline
comments in `simulator/configs/rtx_3090_ti.toml`. New historical config fields
cited provenance such as deviceQuery, profiler output, or paper references.

## CLI (clap derive)

```rust
#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    config: PathBuf,
    // ...
}
```

**Convention**: use clap derive, not builder. Short flags for frequently-used
options (`-c`, `-t`, `-v`), long flags for everything else. Boolean flags
use `--flag` (not `--flag=true`).

## Testing

All commands below assume `cd simulator/` (where the workspace
`Cargo.toml` lives).

| Suite | Location | Verify count | Runner |
|-------|----------|--------------|--------|
| Unit tests | `#[cfg(test)]` modules in each crate | `cargo test --workspace -- --list` | `cargo test --workspace` |
| Integration tests | `tests/` | `pytest --collect-only tests/` | `pytest tests/` |

**Convention**: unit tests live alongside the code they test. Integration
tests exercise the full CLI pipeline (config → trace → report → compare).
No snapshot testing yet (future target: `insta`). See
`rust-testing-quality.md` (sibling card) for proptest, insta, cargo-fuzz
strategy and specific nvasim targets.

**CI gate**: `cargo clippy --workspace -- -D warnings` must be clean.
`cargo test --workspace` must pass. No formal CI pipeline yet (local-only).

## Dependency Management

> **Historical nvasim pins.** The ACTIVE workspace's pins live in `sim/Cargo.toml` `[workspace.dependencies]` (prost 0.14, thiserror 2, edition 2024, clippy::unwrap_used warned).

Workspace-level pinning in `Cargo.toml`:

```toml
[workspace.dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
sha2 = "0.10"
prost = "0.13"
```

**Convention**: all shared dependencies are declared in `[workspace.dependencies]`
and consumed in crates via `.workspace = true`. Direct version pins in
individual crates are avoided.

**Known upgrade targets**: `sha2` 0.10 → 0.11, `prost` 0.13 → 0.14,
`criterion` 0.5 → 0.8 (or switch to `divan`).

## Code Organization

Each crate follows a consistent pattern:

- `src/lib.rs` — public API re-exports
- `src/<module>.rs` — one concern per file (e.g., `scheduler.rs`, `register_file.rs`)
- `src/config.rs` — crate-local config structs (if needed)
- Tests in `#[cfg(test)] mod tests` at the bottom of each file

**Convention**: types are named descriptively (`WarpState`, `L1Cache`,
`SimulatorConfig`). No newtype wrappers yet for IDs — `u32` is used directly
for SM IDs, warp IDs, register indices (future target: `SmId(u32)`,
`Cycle(u64)`, `Addr(u64)`). See `rust-type-safety.md` (sibling card) for why
newtypes matter, which 5 specific gaps exist, and NeXosim reference patterns.

## JSON Report Output

The `--report-json` flag emits per-kernel metrics for the validation pipeline.
Currently constructed via `format!()` string building.

**Future target**: Replace with `#[derive(Serialize)]` struct +
`serde_json::to_string_pretty()`. See `rust-type-safety.md` (sibling card)
for the derive-Serialize pattern as a type safety improvement.

## Logging

`tracing` and `tracing-subscriber` are in `Cargo.toml` but minimally used
(one `warn!` call in the entire codebase). Controlled via `RUST_LOG` env var.

**Future target**: `#[instrument]` on key simulation functions with
structured debug fields. See `rust-testing-quality.md` (sibling card) for
tracing as part of the observability and CI pipeline strategy.

## Modeling context (papers)

- [NVArchSim](references/papers/knowledge_base/active/2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim.pdf) — the design philosophy (event-driven, deterministic, SHA-hash provenance) that nvasim follows.

## See also

- `.claude/knowledge/simulator/nvasim-architecture.md` — historical workspace layout, SM timing model, trace format.
- `.claude/knowledge/measurement/nvarchsim-vs-this-project.md` — current positioning of simulator work as prior art.

## Escalate to raw when

- You need an exact historical detail from the parent simulator snapshot.
- You need to explain why a restored knowledge card mentions nvasim.
