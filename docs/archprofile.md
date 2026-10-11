# Configuration contracts

[`arch_profiles/README.md`](../arch_profiles/README.md) indexes cards, specs,
programs, and schemas. Native cards are architecture-bound inputs; architecture
local capability code in `sim/arch` may still add behavior. Never substitute a
value across architectures or present a hypothesis as an independent hardware
fact.

## Input classes

1. **Native cards:** `sm_86.json` is the established RTX 3090 Ti baseline;
   `sm_120-rtx-5090.json` is the active RTX 5090 exploratory/alpha card;
   `sm_120-laptop.json` remains a separate historical tier. Research gaps stay
   explicit and do not inherit SM86 values.
2. **Hypothetical resources:** `ArchitectureSpec` under `arch_profiles/specs`
   feeds bounded custom execution. It is not native SASS timing.
3. **Captured launch metadata:** trace resources and optional independent CUPTI
   observations. Driver-selected carveout is not a preference percentage.
4. **NCU measurements:** independent observations joined after replay; they do
   not tune the same scored workload or select model constants.

The SM120 card records live runtime limits and unresolved cache/issue/timing
fields. A card identifies the input source, not scientific certification.

## Operational configuration

- Rust components and dependencies: `rust-toolchain.toml`, `sim/Cargo.lock`.
- NVBit producer environment: `tools/nvbit_tracer/pinned_build_env.sh`.
- Runbook: `docs/runbook_host.md`.
- Continuation entrypoints: `docs/PROJECT_MEMORY.md`, `docs/STATUS.md`, and the
  external active plan.
- Local captures/builds/toolchains are ignored by `.gitignore`; no lifecycle
  hook or mandatory gate is part of the project contract.
