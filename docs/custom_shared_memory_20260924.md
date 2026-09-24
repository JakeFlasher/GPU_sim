# Custom shared-memory semantic probe — 2026-09-24

The bounded `simt_u32_v1` frontend now distinguishes global and shared memory
spaces in both functional state and semantic events. The shared path is
deliberately functional/resource-model evidence; it does not model cache banks,
coherence, or silicon timing.

## Command

```bash
cargo build --manifest-path sim/Cargo.toml -p sim-arch \
  --bin custom_simt --bin custom_simt_timing
python tools/custom_simt_differential.py \
  --binary sim/target/debug/custom_simt \
  --timing-binary sim/target/debug/custom_simt_timing \
  --spec arch_profiles/specs/simt_u32_shared.json \
  --program arch_profiles/programs/simt_u32_shared.json
```

## Observed result

```text
functional_match = true
timing_match     = true
global memory    = {}
shared memory    = initial words plus values +5 at 200..228
final cycle      = 59
```

The semantic events classify `load_shared`/`store_shared` as
`MemoryFamily::Shared` with `AddressSpace::Shared`; global and shared address
ordering are kept in separate dependency domains. This is the next bounded
functional seam toward Curry-style address spaces, not a claim of arbitrary ISA
or cache behavior.
