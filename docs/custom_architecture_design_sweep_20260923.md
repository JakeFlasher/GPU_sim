# Custom architecture design sweep — 2026-09-23

This is a model-relative feasibility result for the user-defined architecture
path. It is not a silicon validation result and does not use NCU/rocprof data.

## Command

```bash
cargo build --release --manifest-path sim/Cargo.toml \
  -p sim-arch --bin custom_simt_timing
python tools/arch_spec_design_sweep.py \
  --binary sim/target/release/custom_simt_timing \
  --spec arch_profiles/specs/simt_u32_minimal.json \
  --program arch_profiles/programs/simt_u32_async_store.json \
  --output docs/custom_architecture_design_sweep_20260923.json
```

The sweep varies SIMT width and memory issue width. The async store uses one
pending group and a wait, so the final cycle exposes the declared issue/latency
tradeoff. Event and semantic-event counts remain conserved across variants.
All custom-program memory addresses are byte addresses; timing parameters are
hypothetical resource inputs, not hardware calibration.

## Interpretation

This experiment establishes that the custom spec is executable and that a
resource parameter can change a timing result without changing the program.
It does **not** establish hardware accuracy, a compiler contract, memory-model
correctness, RTL equivalence, or Curry/Vortex compatibility.

## Observed model output

| SIMT width | Memory issue width | Final cycle |
|---:|---:|---:|
| 4 | 1 | 29 |
| 4 | 2 | 27 |
| 8 | 1 | 33 |
| 8 | 2 | 29 |
| 16 | 1 | 41 |
| 16 | 2 | 33 |

All six rows emitted five timing events and five validated semantic events.
