# GPU simulator

A software GPU simulator with two deliberately separate execution lanes:

- **Native NVIDIA replay:** NVBit SASS/protobuf traces and architecture cards
  feed the Rust SM, cache/memory, and event models. The active hardware anchor
  is Ampere SM86 (RTX 3090 Ti).
- **Custom models:** bounded `simt_u32_v1` programs and declarative resources
  feed `sim-sm::custom`; the `simt_its_v1` path is a single-group functional
  example, not native SASS timing.

```text
NVBit -> traces_enhanced -> sim-trace -> sim-sm / sim-mem / sim-engine -> sim-run
custom program + ArchitectureSpec -> sim-sm::custom -> sim-engine -> custom-program-run
NCU observations -------------------------------------------------> comparison only
```

NCU values are independent comparison targets, never same-workload tuning
inputs. Unobservable vendor internals remain black-box model variables; the
project uses analytical/statistical methods rather than pursuing them as
research goals. A vendor-neutral event schema is not a universal binary
decoder.

## Start here

- [Current status and open boundaries](docs/STATUS.md)
- [Continuation memory](docs/PROJECT_MEMORY.md)
- [Predictive replacement contract](docs/PREDICTIVE_CLOSURE.md)
- [Build, capture, and replay](docs/runbook_host.md)
- [Architecture and components](docs/ARCHITECTURE.md) · [configuration](docs/archprofile.md)
- [Documentation map](docs/README.md)
- [Review/refactor history](docs/REVIEW_HISTORY.md)
- [Active reification/refactor plan](plans/sm86_reification_refactor_20261001.md)

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
Native replay, the fast NCU-metric projection, and independent CUPTI/NAK/PMA
probes have different contracts; see `docs/STATUS.md` before comparing them.
Unmatched shapes in the projection receive a documented candidate fallback and
interval rather than an empty row, but no tool can guarantee accuracy for every
future workload without a defined distribution and independent observations.

The retained published Accel-Sim manifest
(`docs/accelsim_literature_lower_bound_manifest_20260928.json`) is a separate
literature accuracy/throughput reference. No same-card executable Accel-Sim or
MGPUSim cycle score is asserted here; local RTX 3090 Ti NCU remains the detailed
comparison target.

## Repository map

| Path | Role |
| --- | --- |
| `sim/` | Rust frontends, scheduler, memory model, event engine, runners |
| `arch_profiles/` | Native cards, hypothetical specs, bounded programs, schemas |
| `tools/nvbit_tracer/`, `tools/local_labels/` | Trace and profiler collection |
| `tools/sm86_*.py` | Fast projection and independent research alternatives |
| `workloads/`, `eval/` | Executable inputs and comparison tools |
| `docs/`, `plans/` | Active contracts and the predictive implementation plan |
| `data/`, `references/` | Retained datasets, source documents, and prior art |

Root [`AGENTS.md`](AGENTS.md) governs project execution. Historical reports
and plans are preserved for context; they are not workflow gates.
