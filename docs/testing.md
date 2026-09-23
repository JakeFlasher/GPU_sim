# Running the live path

The project is judged by the final machine-grounded NCU comparison and
simulator wall time. Use one representative trace as the smallest executable
check before a larger run:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
sim/target/release/sim-run --help
```

For a GPU run, use [runbook_host.md](runbook_host.md), record the exact device,
clock/cache/replay settings, and inspect the result JSON plus stderr. Focused
unit or parser checks are optional engineering aids; they are not scientific
validation and do not create a project gate. Do not add process-only approval,
manifest, digest, hash, or documentation checks around a simulator replay.
