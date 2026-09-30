# Custom group-barrier probe — 2026-09-24

The custom event backend now supports a bounded whole-CTA barrier generation.
All resident groups must contain the same barrier ID exactly once. The engine
adds cross-group completion dependencies before the barrier and release
dependencies after it. All participating groups must be resident concurrently;
mismatched participation or insufficient residency fails closed.

This is a bounded synchronization/timing evidence slice. Shared-memory
accesses separated by a matching barrier propagate through the shared
functional map; global-memory conflicts and same-phase shared conflicts still
fail closed. It is not a general CUDA/PTX memory model.

## Command

```bash
cargo build --manifest-path sim/Cargo.toml -p sim-sm --bin custom-program-run
sim/target/debug/custom-program-run \
  arch_profiles/specs/simt_u32_pipeline2.json \
  arch_profiles/programs/simt_u32_barrier_a.json \
  arch_profiles/programs/simt_u32_barrier_b.json
```

## Observed result

```text
group 0 barrier: cycle 5
group 1 barrier: cycle 5
group 0 store:   cycle 6
group 1 store:   cycle 14 (shared resource serialization)
final cycle:     42
```

This models the ordering contract without claiming the full CUDA/PTX memory
model. NVIDIA’s [PTX barrier documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
describes CTA barriers as both rendezvous and memory-ordering mechanisms; weak
memory, inter-group atomics, arbitrary reductions, and partial participation
remain unsupported.

A producer/consumer variant is also retained in
`simt_u32_barrier_producer.json` / `simt_u32_barrier_consumer.json`: the
consumer reads the producer’s shared store only after the barrier generation,
receives value `100`, and both groups finish with `final_cycle=90`.
