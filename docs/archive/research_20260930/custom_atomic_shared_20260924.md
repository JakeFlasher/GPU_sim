# Custom shared atomic probe — 2026-09-24

The bounded SIMT frontend now includes deterministic lane-ordered
`atomic_add_shared`. Each active lane reads the current shared word, writes the
wrapped u32 sum, and receives the previous value in its destination register.
This is a functional ordering contract, not a claim about hardware atomic
throughput, scope, or weak-memory behavior.

## Command

```bash
cargo build --manifest-path sim/Cargo.toml -p sim-arch \
  --bin custom_simt --bin custom_simt_timing
python tools/custom_simt_differential.py \
  --binary sim/target/debug/custom_simt \
  --timing-binary sim/target/debug/custom_simt_timing \
  --spec arch_profiles/specs/simt_u32_atomic.json \
  --program arch_profiles/programs/simt_u32_atomic.json
```

## Observed result

```text
functional_match = true
timing_match     = true
shared[0x190]    = 38
semantic         = MemoryDirection::Atomic / MemoryFamily::Shared
final cycle      = 30
```

The retained program uses stride zero deliberately to exercise deterministic
lane ordering. General atomics, arbitrary reductions, fences, and inter-group
atomic consistency remain unsupported.
