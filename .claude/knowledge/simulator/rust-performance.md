---
Use when: reading nvasim Rust performance patterns as prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02).
Primary raw source: Effective Rust Item 20 ("Optimize carefully — and measure") + Programming Rust 3rd ed Ch1-2
Supporting sources:
- Programming Rust 3rd ed Ch1 (iterator optimization, Vec::with_capacity, thread::scope)
- Programming Rust 3rd ed Ch2 (slice references, pre-allocation, checked arithmetic)
- Programming Rust 3rd ed Ch3 (move semantics, swap_remove, mem::replace)
- NeXosim `references/Rust_refs/nexosim/nexosim/src/executor/st_executor.rs` (single-threaded execution model)
- NeXosim `references/Rust_refs/nexosim/nexosim/src/channel.rs` (RecycleBox for allocation reuse)
Repo touchpoints:
- simulator/src/simulator.rs (SM mini-loop hot path, Vec allocation, struct destructuring, CTA activation)
- simulator/crates/event-engine/src/lib.rs (Event struct layout, min-heap)
- simulator/crates/event-engine/src/engine.rs (next-event advancement, event dispatch loop)
- simulator/crates/event-engine/src/hasher.rs (SHA-256 per-event hashing)
- simulator/crates/memory/src/l1_cache.rs (sectored cache, MSHR tracking)
- simulator/benches/event_engine.rs (criterion benchmark)
Out of scope: architecture decisions about event dispatch or queue design (see rust-des-patterns.md), numeric overflow safety (see rust-type-safety.md).
Version family: Historical parent-repo nvasim SM120 / Blackwell snapshot, 2026-04-16.
---

# Rust Performance Patterns for nvasim

> **Current-scope note:** prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02). The nvasim source itself is historical — do not port it verbatim. NCU is a first-class ground-truth instrument now: its values are prediction TARGETS, never simulator inputs (Rule 1). The active implementation lives in `sim/` (see `plans/phase2_sim_core_progress.md`).

How to identify, measure, and fix performance issues in the nvasim
simulator. Follows Effective Rust Item 20's guiding principle: "optimize
carefully — and measure." Every recommendation is grounded in specific
nvasim code and labeled as "current pattern" or "future target."

## Profiling workflow (current pattern + future target)

**Current**: criterion benchmark at `benches/event_engine.rs` defines
three bench functions:
- `event_engine_1M_events`: end-to-end throughput with realistic event
  mix (SM wake → L2 request → L2 response → SM wake chain)
- `event_queue_schedule_pop_1M`: raw queue schedule/pop throughput
- `event_hasher_1M_events`: SHA-256 hashing throughput

**Run**: `cd simulator && cargo bench --bench event_engine`. As of
2026-04-19 the bench compiles cleanly (verified with `cargo bench
--bench event_engine --no-run`); the prior `Event { sm_id, .. }`
construction sites in `benches/event_engine.rs` have been updated to
match the D2-era `Event` shape.

**Future targets**:
- **flamegraph**: `cargo flamegraph --bench event_engine` for visual
  hot-path identification. Requires `perf` on Linux (verify with
  `which perf` and install via your distro's package manager if missing)
- **Benchmark regression detection**: store criterion baseline
  (`cargo bench -- --save-baseline main`), compare on changes
  (`cargo bench -- --baseline main`), fail on >5% regression
- **Per-component benchmarks**: add benchmarks for L1 cache operations,
  coalescing, and trace parsing — currently only the event engine is
  benchmarked

Effective Rust Item 20: "Do not optimize without measuring first. Your
intuition about where time is spent is usually wrong."

## Allocation discipline (current pattern + future target)

### Hot-path avoidance (future target)

The SM mini-loop in `simulator.rs` (~1200 lines, executes per-cycle
per-SM) contains a hot-path `Vec` allocation:

```rust
let all_regs: Vec<u32> = inst.source_registers.iter()
    .chain(inst.dest_registers.iter()).copied().collect();
```

This creates a new heap-allocated `Vec` for every instruction dispatched
in the SM mini-loop — one allocation per instruction per cycle per SM.
The total count scales with trace size (verify via profiling). Fix
options:
- Pass `source_registers` and `dest_registers` as separate slices to the
  register file — no collection needed
- Use a small fixed-size array (`[u32; 16]` covers max register count)
  with a length field — stack allocation, no heap

### Unnecessary clone (future target)

`simulator.rs` CTA activation: `cta.warp_to_subcore.clone().iter().enumerate()`
could iterate by reference: `cta.warp_to_subcore.iter().enumerate()`.
Impact is small (activation is not a hot path) but the clone is
unnecessary.

### Pre-allocation (current pattern)

nvasim correctly uses `Vec::with_capacity()` for known-size collections:
- `cta_warp_indices` in CTA activation
- `EventQueue::with_capacity(1024)` in engine initialization

Programming Rust Ch1: "`Vec::with_capacity()` for pre-allocating event
queues to avoid reallocation during simulation." Also `vec![0; size]`
for single-allocation memory buffers.

## Memory layout (current pattern)

### Stack vs heap

nvasim's `Event` struct in `event-engine/src/lib.rs`:

```rust
pub struct Event {
    pub cycle: u64,    // 8 bytes
    pub sm_id: u32,    // 4 bytes
    pub seq: u64,      // 8 bytes
    pub event: SimEvent, // variable (enum)
}
```

Events are moved into the `BinaryHeap`'s backing storage. Since `Event`
contains inline scalar fields and an enum (no heap-allocated payload
behind pointers), a move copies the full struct value. The key benefit
is that no separate per-event heap allocation is needed — the
`BinaryHeap` manages a single contiguous `Vec` internally.

### Slice references (current pattern)

`&[T]` fat pointers (address + length) provide zero-copy views into
cache lines, register files, and instruction operands. Programming Rust
Ch2: slice references avoid copying data between components while
maintaining bounds checking.

### No interior mutability in hot path (current pattern)

All state mutation in the SM mini-loop uses explicit `&mut` references
obtained via struct destructuring. No `Cell`, `RefCell`, or `Mutex` in
the simulation hot path — these add runtime overhead (bounds checks for
`RefCell`, lock acquisition for `Mutex`).

## Iterator optimization (current pattern)

Programming Rust Ch1: "iterator method chaining compiles to same machine
code as manual loops." nvasim uses this throughout:
- Event queue `pop()` in a `while let` loop
- Register file iteration with `.iter().enumerate()`
- Address coalescing with `.map()` and `.collect()`

The compiler's iterator fusion eliminates intermediate allocations for
chains like `.iter().filter().map()`.

## Next-event time advancement (current pattern)

The engine in `engine.rs` jumps directly from the current cycle to the
next event's cycle:

```rust
if event.cycle > self.current_cycle {
    self.cycles_skipped += event.cycle - self.current_cycle - 1;
    self.current_cycle = event.cycle;
}
```

This makes simulation cost scale with event count rather than simulated
cycle count — idle cycles between events are skipped, not iterated. Per-
event runtime still depends on handler complexity and payload mix, but
the engine itself adds no overhead for skipped cycles. The
`cycles_skipped` counter tracks how many idle cycles were jumped over.

## SHA-256 determinism hashing (current pattern)

`hasher.rs` accumulates a SHA-256 hash over every dispatched event.
Implementation: `hash_bytes()` on `SimEvent` writes discriminant + field
bytes to a reusable `Vec<u8>` buffer, then feeds the buffer to the
SHA-256 hasher.

Overhead is small per-event (field bytes only, not full simulation state)
but accumulates over millions of events. The `--determinism-check` flag
controls whether the hash is printed — the hashing itself always runs
(it is part of the `Engine::run()` loop).

**Future consideration**: if hashing becomes a bottleneck, switch to a
faster non-cryptographic hash (xxHash, ahash) for determinism checking.
SHA-256 is not needed for security here — only for collision resistance
across ~100M events.

## Benchmarking practices (current pattern)

The criterion benchmark in `benches/event_engine.rs` follows best
practices:
- **`black_box()`** on return values prevents dead-code elimination
- **Realistic event mix**: the bench handler chains SM wakes → L2
  requests → L2 responses → more SM wakes, matching real workload patterns
- **Separate concerns**: throughput, queue operations, and hashing are
  benchmarked independently

**Future targets**:
- **divan** as a lighter-weight alternative (fewer dependencies, faster
  compilation, macro-based API)
- **iai-callgrind** for instruction-count-based benchmarks that are
  insensitive to system load (useful for CI)

## Allocation reuse patterns (future target)

NeXosim's `channel.rs` uses `RecycleBox` for allocation reuse in its
messaging layer — pre-allocated boxes are recycled rather than dropped
and reallocated. This pattern is relevant if nvasim's event dispatch
ever becomes allocation-bound.

Programming Rust Ch3: `std::mem::replace(&mut v[i], replacement)` for
swapping values without temporary allocation; `swap_remove(i)` for O(1)
removal from flat lists when order does not matter.

## Arena allocation (future target)

If profiling shows that per-event heap allocation is a bottleneck,
arena allocators like `bumpalo` can batch-allocate objects with O(1)
amortized cost. Events allocated in an arena are freed all at once when
the arena is reset (e.g., per-kernel or per-epoch).

**Note**: this is a significant architectural change. The current
`BinaryHeap<Event>` approach with move semantics is simpler and should
only be replaced if profiling data justifies the complexity.

## See also

- `.claude/knowledge/simulator/nvasim-architecture.md` — crate workspace layout and SM timing model.
- `.claude/knowledge/simulator/rust-des-patterns.md` — DES architecture rationale (priority queue design, dispatch, determinism).
- `.claude/knowledge/simulator/rust-type-safety.md` — checked arithmetic for cycle counters.
- `.claude/knowledge/simulator/rust-testing-quality.md` — benchmarking tools and CI regression detection.

## Escalate to raw when

- You need flamegraph setup for Rust — see `cargo-flamegraph` documentation.
- You need criterion API beyond basic benches — see `simulator/benches/event_engine.rs`.
- You need the SM mini-loop hot path — see `simulator/src/simulator.rs` (grep `sm_mini_loop` or the register Vec allocation).
- You need allocation-reuse patterns — see `references/Rust_refs/nexosim/nexosim/src/channel.rs` (RecycleBox).
- You need arena allocator guidance — see `bumpalo` crate documentation.
