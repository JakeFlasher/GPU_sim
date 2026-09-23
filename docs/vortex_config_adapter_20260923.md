# Vortex configuration adapter — 2026-09-23

The custom architecture path can ingest a public Vortex `VX_config.toml`
shape without pretending to understand Vortex ISA semantics. The adapter maps
literal thread/warp counts, requires callers to supply an already-resolved
issue width, leaves timing unknown, emits an empty instruction set, and marks
the result `unvalidated`.

## Command

```bash
python tools/vortex_config_to_spec.py \
  /path/to/vortex/VX_config.toml \
  --resolved-issue-width 1 \
  --output /tmp/vortex_config_spec.json
cargo run --manifest-path sim/Cargo.toml -p sim-arch \
  --bin architecture_spec -- /tmp/vortex_config_spec.json
```

During the 2026-09-23 probe, the public Vortex checkout was inspected at
`/tmp/vortex`. A separate Ubuntu-container build produced SimX and rtlsim and
ran a hostless vecadd parity probe; details are in
`vortex_vecadd_simx_rtlsim_20260923.md`.

The documented Vortex `configure` flow could not start directly on this
CachyOS host: the upstream script admits Ubuntu/CentOS/RHEL families and exits
`Unsupported OS`. The container route supplied the missing Ubuntu environment
and toolchain without changing the repository's host configuration.

Passing the generated `vortex_config` spec to the current `simt_u32_v1`
executor intentionally fails with an architecture/ISA identity mismatch. This
is the required fail-closed boundary before a real Vortex ISA adapter exists.
