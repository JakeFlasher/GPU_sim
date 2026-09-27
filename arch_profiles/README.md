# Simulator input/configuration index

| Input | Consumer | Contract |
| --- | --- | --- |
| `cards/sm_86.json` | Native Rust/Python profile loaders | SM86 identity/geometry/resources and dated calibration provenance; not an all-hardware truth claim |
| `cards/sm_120-laptop.json` | Retained exploratory loaders | Unavailable fields stay gaps; no SM86 fallback or new validation claim |
| `aliases.json`, `build_arch_families.json` | Python loader | Runtime/deployment tokens differ from compile targets |
| `../data/pchase/sm_86_static_card.json` | Native L1/L1-L2 replay | Measured latency plus declared geometry/MSHR hypotheses |
| `specs/*.json` | `sim-arch::spec` | Hypothetical named resources; see [specs](specs/README.md) |
| `programs/simt_u32_*.json` | `custom-program-run`, custom functional/timing CLI | Fixed u32 semantics; see [programs](programs/README.md) |
| `programs/simt_its*.json` | `custom_simt_its` | Single-group functional ITS, not SASS/PTX |
| `programs/*.ptx` admitted subset | `ptx_subset_compile` | `ptx_subset_v1` functional lowering only; not CUDA/PTX/SASS compatibility |

[Configuration guide](../docs/archprofile.md) separates cards, overrides and
observations. Do not merge native and hypothetical numerical defaults, rewrite
dated driver observations as current, or change measured values during docs cleanup.

## Schemas

- `schema/arch_profile.schema.json`: native profile syntax; loaders add semantics.
- `schema/architecture_spec.schema.json`: resource/spec syntax.
- `schema/simt_program.schema.json`: `simt_u32_v1` input only, not ITS.
- `schema/simt_its_trace.schema.json`: opt-in ITS **output shape**, version 2 /
  `custom_its_dynamic_v2`. Its nested state/instruction fields are permissive;
  JSON-schema validation alone does not establish execution correctness.
  `sim/arch/src/its.rs` is the ITS input validator.

Toolchain: root `rust-toolchain.toml` and `sim/Cargo.lock`; local NVBit helper
`tools/nvbit_tracer/pinned_build_env.sh`; Python test root `pytest.ini`.
Ignored `.codex/`, toolchains and local captures are not runtime model inputs.
