# sim/ — the event-driven GPU simulator (Rust workspace)

Phase-0 scaffold (2026-07-06). Design: `plans/micro2027_ncu_sim_execution_plan.md` §6,
grounded in `references/papers/notes/draft_new_gpu_simulator.md` (NVArchSim-inspired)
and the `.claude/skills/designing-gpu-simulator/` routing. Build order = design-note
phases A→E (engine → memory → trace → modern features → accuracy tuning).

| Crate | Role (target fidelity) |
|---|---|
| `sim-arch` | Typed deployment/build/SKU identities and reviewed capability states, plus the fail-closed architecture-card loader and provenance boundary for per-architecture latencies/throughputs. `sm_86` capabilities are admitted; `sm_120` is an explicit template with typed research gaps/unsupported control paths. Cards are simulator inputs; NCU labels remain targets and never enter the feature/configuration path. |
| `sim-engine` | Event queue (binary-heap → timing-wheel hybrid later), tagged-enum dispatch, current event tuple `{time, seq, target, kind}`, per-SM actor mini-loop batching. Throughput target ≥3–10 M events/s. |
| `sim-sm` | Generic 4-sub-core issue/replay algorithms. Static control decoding is capability-gated: the checked-in Ampere sidecar path is active for `sm_86`; no unproven Blackwell decoder is implied. |
| `sim-mem` | Generic sectored L1/L2 state machines and warp-pool interleave. Geometry, cache-op defaults, MSHR approximations, and latency ladders enter through typed architecture/card resolvers; an `sm_120` gap cannot inherit an `sm_86` hypothesis. |
| `sim-trace` | NVBit protobuf trace ingestion through the committed Rust/prost bindings for the existing `traces_enhanced` wire format. Wire decoding remains architecture-neutral. |
| `sim-run` | Manifest-only production composition, `sim_run::emit` metric boundary, and one-argument `sim-run` binary: strict `sim_run_manifest.v2` admission, architecture-local configuration resolution, full-coverage replay, checked launch-set aggregation, seven implemented secondary descriptors plus blocked `exec_cycles_gpc_max`, and sealed canonical `sim_result.v2` output. Stdout is success-result-only, stderr is one sealed `sim_error.v2` document on failure, and no replay setting comes from flags or environment variables. |

## Architecture scope

The active development lane is Ampere `sm_86` on the RTX 3090 Ti. The
Blackwell `sm_120` card and enum values are retained as a typed integration
scaffold only: missing L1/cache-policy, scheduler, latency, DRAM operating
point, and control-layout evidence produces a typed gap or unsupported error.
No simulator constructor may substitute an `sm_86` value for one of those
states. Adding a future architecture requires a new card-backed capability
set and producer/consumer parity evidence rather than extending a string map.

`ArchId` (deployment), `BuildArch` (fatbin/compiler token), and `Sku` (device
card) are deliberately separate. This prevents build tokens such as
`sm_120a` from being mistaken for a deploy target and prevents one SKU's
architecture facts from being reused for another.

Hard rules: targets-not-features (CLAUDE.md Rule 1; enforced by
`eval/guards/test_targets_not_features_guard.py` which scans this tree);
undocumented microarchitecture = tunable hypotheses behind fidelity knobs; no
per-cycle global ticking hidden inside "event-driven" naming; no per-instruction
events (SM mini-loop batches issue cycles).

Build: `cargo build` from this directory (workspace).

Run the self-contained offline canary from the repository root:

```bash
cargo run --quiet --manifest-path sim/Cargo.toml -p sim-run -- \
  sim/run/tests/fixtures/addr_carry_canary_bundle/run_manifest.v2.json
```

`sim_run_manifest.v2` is the only production manifest version. It carries a
closed nine-field launch key for each launch, expected launch/card/P-chase
digests, architecture and SKU, scope, and one typed memory mode. It has no
numeric configuration overrides or target-value fields. A successful invocation
emits one sealed canonical `sim_result.v2` document plus a line feed on stdout;
each failure emits one of the nine sealed `sim_error.v2` arms on stderr while
stdout remains empty. `sm_120` remains a typed disabled prerequisite at this
boundary. The fixture result confers no hardware or scientific credit.
