# Custom lane-exchange collective probe — 2026-09-24

The bounded custom SIMT frontend now includes `broadcast_lane0`, a fixed
lane-exchange operation that copies lane 0's register value to every active
lane. It exercises the semantic `LaneExchange` category and producer/consumer
register dependencies without claiming arbitrary collective semantics.

## Command

```bash
cargo build --manifest-path sim/Cargo.toml -p sim-arch \
  --bin custom_simt --bin custom_simt_timing
python tools/custom_simt_differential.py \
  --binary sim/target/debug/custom_simt \
  --timing-binary sim/target/debug/custom_simt_timing \
  --spec arch_profiles/specs/simt_u32_collective.json \
  --program arch_profiles/programs/simt_u32_collective.json
```

## Result

The independent differential reports `functional_match=true` and
`timing_match=true`; every lane receives lane 0's value plus five in its final
destination register. This is a functional collective seam, not a model of
warp shuffle latency or hardware crossbar throughput.
