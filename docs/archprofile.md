# Configuration contracts

[Input index](../arch_profiles/README.md) lists cards/specs/programs/schemas.
Native cards are architecture-bound inputs, but not every mechanism is carded:
`sim/arch/src/lib.rs` also contains architecture-local capability definitions.
Do not silently substitute values across architectures or treat every value as
an independently measured hardware fact.

## Separate four kinds of input

1. **Native card:** `arch_profiles/cards/sm_86.json` plus p-chase data for
   L1/L1-L2. Keep provenance and historical operating points unchanged when
   reorganizing docs. The retained SM120 card is exploratory, not this run's target.
2. **Hypothetical resources:** `ArchitectureSpec` under `arch_profiles/specs`.
   Custom execution consumes width/latency/initiation/capacity and bindings.
   The native `trace_sass_v1` bridge accepts only documented matching issue widths;
   unused fields reject. Functional custom programs cannot replace SASS inputs.
3. **Captured launch metadata:** trace resources and an optional independent
   CUPTI resource observation. Driver-selected carveout is not the preference
   percentage. The join validates symbol/geometry/resources; it does not alter
   replay timing/cache state.
4. **Validation labels:** NCU CSV data is joined after replay, never used to
   tune the same scored workload's inputs or model selection.

The carded 25/8-cycle barrier values and occupancy/resources include declared
architecture-local hypotheses. `architecture_card` identifies a source, not
scientific certification. Unavailable capability fields stay typed gaps.

## Operational configuration

- Rust version/components: `rust-toolchain.toml`; dependencies: `sim/Cargo.lock`.
- Native producer build: `tools/nvbit_tracer/pinned_build_env.sh`, which checks
  CUDA 13.2, GCC 14, protoc 34.1 and protobuf 34.1.0 on this setup.
- Python test imports: `pytest.ini`; use the existing `.venv` when needed.
- Agent routing: root `AGENTS.md`; project memory/status are concise entrypoints.
  Global Codex model/hooks/settings affect other projects and are not rewritten
  for repository cleanup. Project memory mirrors are outside Git.
- Local state/captures/builds remain ignored by `.gitignore`; no new lifecycle
  hooks, mandatory gates or duplicate `.codex` runtime defaults are introduced.

[Runbook](runbook_host.md) owns the commands; [history](HISTORY.md) owns dated
GPU UUID/driver/clock observations, which are not permanent host identity.
