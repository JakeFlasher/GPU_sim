# Rust simulator

The workspace contains the executable replay path:

- `sim-trace`: protobuf trace and control-sidecar readers.
- `sim-arch`: architecture cards and capability selection.
- `sim-sm`: warp, CTA, dependency, barrier, and issue scheduling.
- `sim-mem`: L1/TEX, L2, and memory completion models.
- `sim-engine`: shared event ordering.
- `sim-run-campaign`: manifest driven CLI and JSON output.
- `sim-run`: direct replay convenience CLI.
- `sim-sm::causal_frontier`: opt-in causal-frontier analysis for explicit
  descriptor streams; it is observational until a producer supplies dependency
  and completion facts.

Build the runner with:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
cargo build --release --manifest-path sim/Cargo.toml -p sim-run-campaign
```

For an observation-only NCU join, append `--ncu-csv PROFILE.csv` and, for a
multi-launch export, `--ncu-launch ID`. The values appear under
`ncu_observation` and never affect replay timing or architecture parameters.

Run it with a retained NVBit trace and control sidecar as described in
`docs/runbook_host.md`. Architecture values belong in `arch_profiles/cards`,
not in workload-specific code.
