# Running the live path

Use a release build and one representative trace as the default smoke path:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
sim/target/release/sim-run --help
```

For a retained trace, follow [runbook_host.md](runbook_host.md) and inspect the
result JSON plus stderr. Python utilities can be compiled without running a
campaign:

```bash
python -m compileall -q eval workloads/research_probes tools/nvbit_tracer
```

Keep checks close to the changed path. Do not add process-only approval,
manifest, or digest checks around a simulator replay.
