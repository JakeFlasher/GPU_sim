# Rust simulator

The workspace contains the executable replay path:

- `sim-trace`: protobuf trace and control-sidecar readers.
- `sim-arch`: architecture cards and capability selection.
- `sim-sm`: warp, CTA, dependency, barrier, and issue scheduling.
- `sim-mem`: L1/TEX, L2, and memory completion models.
- `sim-engine`: shared event ordering.
- `sim-run-campaign`: manifest driven CLI and JSON output.
- `sim-run`: direct replay convenience CLI.

Build the runner with:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
cargo build --release --manifest-path sim/Cargo.toml -p sim-run-campaign
```

Run it with a retained NVBit trace and control sidecar as described in
`docs/runbook_host.md`. Architecture values belong in `arch_profiles/cards`,
not in workload-specific code.
