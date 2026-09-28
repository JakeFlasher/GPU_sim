# Minimal functional SIMT programs

These JSON programs are feasibility inputs for `sim-arch::simt`. They execute
one bounded SIMT group using the `simt_u32_v1` instruction subset:

- `set_lane_id`
- `load`
- `load_shared`
- `add_imm`
- `add`
- `store`
- `store_shared`
- `atomic_add_shared`
- `broadcast_lane0`
- `reduce_add`
- `async_store`
- `wait_group`
- `group_barrier`
- `halt`

They are not SASS/PTX, not a compiler format, and not a timing trace. The
executor is intentionally isolated from SM86 replay; the separate timing
probe and production `custom-program-run` frontend consume the same bounded
semantics through the shared dependency/resource engine. Operation bindings
may select declared resources (with `alu`/`memory` defaults), but arbitrary
resource names are not an ISA extension.

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

`simt_u32_route_alternatives.json` demonstrates the bounded hypothetical
resource-routing contract: an operation may name an alternative set, and the
dependency/resource scheduler chooses the earliest feasible resource using
latency, initiation interval and in-flight capacity (canonical resource-index
order breaks equal ties). The selected resource is reported in timing events.
This is an executable custom-model experiment, not native SM86 physical-pipe
attribution.

`simt_u32_vecadd.json` adds initialized memory plus load/add/store operations
for the functional cross-model comparison against Vortex `vecadd`.
Loads and stores are 32-bit byte-addressed words; the retained lane probes use
a four-byte stride to avoid overlapping word ranges. Unaligned or overlapping
words and u32 address wraparound are rejected by the executor rather than
resolved by host-map order. `async_store` is a deferred global store with an
explicit submit/wait token; it is not a CUDA/Vortex `cp.async` compatibility
claim.

`simt_u32_shared.json` exercises the separate shared-memory address space. The
functional result and semantic events retain global and shared memory maps and
families separately; no cache-bank or silicon timing claim follows.

`simt_u32_barrier_a.json` and `simt_u32_barrier_b.json` exercise matching
whole-CTA barrier generations through `custom-program-run`. Barrier ordering is
modeled and separated shared producer/consumer values propagate; atomics,
reductions, and weak-memory behavior remain unsupported across groups.

`simt_u32_atomic.json` exercises deterministic lane-ordered shared atomic RMW;
it is not a hardware atomic throughput or memory-ordering model.

`simt_u32_collective.json` exercises a fixed lane-0 broadcast collective. It
is a typed SIMT operation, not a general collective-language or compiler
contract.

`simt_u32_reduce.json` exercises a deterministic lane reduction whose wrapped
u32 sum is written to every lane; it is not a hardware reduction-throughput
model.

The Curry-inspired `simt_its_v1` diamond is a separate functional feasibility
program. It demonstrates per-lane PCs, a predicate branch, BSSY/BSYNC
reconvergence, and lane-specific stores:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt_its -- \
  arch_profiles/specs/simt_its8.json \
  arch_profiles/programs/simt_its8_diamond.json
```

`custom_simt_its ... --trace` enables an execution-generated dynamic record
for this bounded ITS frontend. `simt_its_dynamic.json` demonstrates values
computed at run time becoming indirect load/store addresses; the emitted
records contain the executed PC, active mask, lane addresses, grouped successor
masks, runnable successors, bounded reconvergence metadata, and explicit
barrier-blocked lanes. Capture is bounded at 100,000
records and 32 MiB of serialized record bytes. This is execution evidence for
the custom functional ISA only, not a regenerated native SASS trace or a
silicon timing model. The opt-in output is versioned by
`arch_profiles/schema/simt_its_trace.schema.json`.

`ptx_subset_compile` provides a deliberately bounded text-to-`simt_its_v1`
frontend for `mov %laneid`, integer add/multiply, compare-less-than, direct
branch labels, indirect u32 load/store, deterministic indirect atomic-add,
and exit. For example:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin ptx_subset_compile -- \
  example_its arch_profiles/programs/simt_its_subset.ptx /tmp/simt_its_subset.json
```

This is a functional compiler/assembler seam only: calls, ABI, negated
predicates, fences/memory-model forms, vector types, arbitrary PTX, and native
SASS generation are rejected rather than guessed. Deterministic functional
indirect atomic-add is admitted. The contract is `ptx_subset_v2` and remains
separate from SM86 replay.
