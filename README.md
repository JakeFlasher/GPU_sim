# GPU simulator

Software architectural simulation with two executable frontends:

- **Native NVIDIA replay:** NVBit SASS/protobuf traces and architecture cards
  feed the Rust SM, cache/memory, and event models. The active hardware anchor
  is Ampere SM86 (RTX 3090 Ti); other retained cards are not validated by this work.
- **Custom models:** bounded `simt_u32_v1` programs and declarative resources
  feed `sim-sm::custom` and the shared dependency/event engine; resource
  alternatives are selected with explicit model timing. The separate
  Curry-inspired `simt_its_v1` example is functional-only, with a bounded
  `ptx_subset_v2` frontend and dynamic CFG trace.

```text
NVBit -> traces_enhanced -> sim-trace -> sim-sm / sim-mem / sim-engine -> sim-run
custom program + ArchitectureSpec -> sim-sm::custom -> sim-engine -> custom-program-run
NCU observations -------------------------------------> independent comparison
```

NCU values are validation targets, never same-workload tuning inputs. The
project does not require RTL, FPGA, physical design, universal proprietary
binary decoding, or silicon-equivalent internals. A vendor-neutral event
representation is not a universal binary decoder.

## Start here

- [Status and remaining boundaries](docs/STATUS.md)
- [Build, capture, and replay commands](docs/runbook_host.md)
- [Architecture](docs/ARCHITECTURE.md) and [configuration contracts](docs/archprofile.md)
- [Research roadmap](plans/cross_architecture_gpu_simulator_research_plan_20260923.md)
- [PhaseFuse fast SM86 prototype and SOTA study](docs/sm86_fast_exact_prototype_20260927.md)
- [Project memory for agent continuations](docs/PROJECT_MEMORY.md)
- [Documentation and dated evidence](docs/README.md)
- [History and stale-context map](docs/HISTORY.md)

From the repository root:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
sim/target/release/sim-run --help
cargo run --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run -- \
  arch_profiles/specs/simt_u32_pipeline2.json \
  arch_profiles/programs/simt_u32_barrier_producer.json \
  arch_profiles/programs/simt_u32_barrier_consumer.json
```

The custom example produces model-relative timing, not a hardware prediction.
Instruction-count mappings exist for retained SM86 examples; native elapsed
timing, coherence ownership/visibility, resident dirty flush, physical-pipe
attribution and independent evaluation remain open. The opt-in bounded
writeback queue, route alternatives and PTX subset are executable seams, not
silicon claims.
[Diagnostics](docs/experimental_diagnostics.md) records these limits.
No same-card executable Accel-Sim/MGPUSim cycle score has been established.
The retained published Accel-Sim aggregates are tracked separately as
accuracy-ceiling and throughput/speed-floor targets in
`docs/accelsim_literature_lower_bound_manifest_20260928.json`; NCU on the
local RTX 3090 Ti remains the detailed ground truth.

## Repository map

| Path | Role |
| --- | --- |
| `sim/` | Rust frontends, scheduler, memory model, event engine, runners |
| `arch_profiles/` | Native cards, hypothetical specs, bounded programs, schemas |
| `tools/nvbit_tracer/`, `tools/local_labels/` | Separate trace and profiler collection |
| `workloads/`, `eval/` | Executable inputs and comparison tools |
| `docs/`, `plans/` | Current contracts, dated observations, research direction |
| `data/`, `references/` | Retained datasets, source documents, and prior art |

Root [AGENTS.md](AGENTS.md) governs project execution. Historical experiment
records and retained SOL/campaign capability are not workflow gates.
