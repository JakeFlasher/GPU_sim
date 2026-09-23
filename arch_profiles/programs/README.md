# Minimal functional SIMT programs

These JSON programs are feasibility inputs for `sim-arch::simt`. They execute
one bounded SIMT group using the `simt_u32_v1` instruction subset:

- `set_lane_id`
- `load`
- `add_imm`
- `add`
- `store`
- `async_store`
- `wait_group`
- `halt`

They are not SASS/PTX, not a compiler format, and not a timing trace. The
executor is intentionally isolated from SM86 replay so functional semantics
can mature before a scheduler or memory-timing backend is added.

Run the retained example with:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt -- \
  arch_profiles/specs/simt_u32_minimal.json \
  arch_profiles/programs/simt_u32_lane_store.json
```

The async timing probe is:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt_timing -- \
  arch_profiles/specs/simt_u32_minimal.json \
  arch_profiles/programs/simt_u32_async_store.json
```

`simt_u32_vecadd.json` adds initialized memory plus load/add/store operations
for the functional cross-model comparison against Vortex `vecadd`.
Loads and stores are 32-bit byte-addressed words; the retained lane probes use
a four-byte stride to avoid overlapping word ranges. Unaligned or overlapping
words and u32 address wraparound are rejected by the executor rather than
resolved by host-map order. `async_store` is a deferred global store with an
explicit submit/wait token; it is not a CUDA/Vortex `cp.async` compatibility
claim.

The Curry-inspired `simt_its_v1` diamond is a separate functional feasibility
program. It demonstrates per-lane PCs, a predicate branch, BSSY/BSYNC
reconvergence, and lane-specific stores:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt_its -- \
  arch_profiles/specs/simt_its8.json \
  arch_profiles/programs/simt_its8_diamond.json
```
