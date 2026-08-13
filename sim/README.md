# Rust event-driven simulator

The workspace contains:

| Crate | Role |
|---|---|
| `sim-engine` | Deterministic event scheduling |
| `sim-trace` | Enhanced NVBit protobuf decoding |
| `sim-sm` | Warp issue, dependency, replay, and execution state |
| `sim-mem` | Sectored L1/L2 and DRAM timing |
| `sim-arch` | Architecture and SKU configuration |
| `sim-run` (`direct_run/`) | Direct end-to-end composition and JSON output |
| `sim-run-campaign` (`run/`) | Optional retained manifest/evidence campaign interface |

The first target is `sm_86`. Unknown `sm_120` behavior remains explicit.

Build and test:

```bash
cargo build
cargo test --all-targets
```

Run a trace directly:

```bash
cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --mode flat
```

The default workspace members exclude the campaign runner. Use it only when a
historical campaign manifest is intentionally in scope:

```bash
cargo run -p sim-run-campaign -- RUN_MANIFEST.json
```

NCU metrics are used only for comparison after simulation. The simulator input
surface is traces, static program information, and architecture parameters.
