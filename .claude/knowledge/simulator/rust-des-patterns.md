---
Use when: reading nvasim discrete-event-simulation design patterns as prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02).
Primary raw source: NeXosim `references/Rust_refs/nexosim/nexosim/src/simulation/scheduler.rs`, `references/Rust_refs/nexosim/nexosim/src/util/priority_queue.rs`, `references/Rust_refs/nexosim/nexosim/src/util/indexed_priority_queue.rs`, `references/Rust_refs/nexosim/nexosim/src/simulation.rs`
Supporting sources:
- Programming Rust 3rd ed Ch3 (ownership, moves, event loop patterns)
- Programming Rust 3rd ed Ch4 (references, lifetime parameters, split borrows)
- Programming Rust 3rd ed Ch5 (match exhaustiveness, tagged enum dispatch)
- Async Rust Ch3 (custom queue and scheduling concepts)
- Async Rust Ch9 (state machine pattern, circuit-breaker)
Repo touchpoints:
- simulator/crates/event-engine/src/lib.rs (Event struct, min-heap Ord reversal, (cycle, sm_id, seq) ordering)
- simulator/crates/event-engine/src/event.rs (SimEvent 15-variant tagged enum, discriminant(), hash_bytes())
- simulator/crates/event-engine/src/engine.rs (Engine, EventHandler trait, EventScheduler, EngineError, next-event advancement)
- simulator/crates/event-engine/src/hasher.rs (SHA-256 determinism proof)
- simulator/src/simulator.rs (SM mini-loop, struct destructuring for split borrows)
- simulator/src/cluster_barrier.rs (barrier state machine, dual-counter coupling)
Out of scope: what the crates do (see nvasim-architecture.md), current coding conventions (see nvasim-dev-patterns.md), performance tuning (see rust-performance.md).
Version family: Historical parent-repo nvasim SM120 / Blackwell snapshot, 2026-04-16.
---

# Rust Discrete-Event Simulation Patterns

> **Current-scope note:** prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02). The nvasim source itself is historical — do not port it verbatim. NCU is a first-class ground-truth instrument now: its values are prediction TARGETS, never simulator inputs (Rule 1). The active implementation lives in `sim/` (see `plans/phase2_sim_core_progress.md`).

Why specific DES architecture decisions were made in nvasim, and what
alternative approaches exist in the NeXosim production DES framework.
This card covers the *why* behind architectural choices; the *what*
is in `nvasim-architecture.md`.

NeXosim is an inspiration source, not a template. nvasim is
single-threaded; NeXosim references are qualified accordingly.

## Priority queue design (current pattern)

nvasim uses a `BinaryHeap<Event>` with reversed `Ord` to get min-heap
behavior. Events are ordered by `(cycle, sm_id, seq)`:

- `cycle`: simulation time — lowest first
- `sm_id`: SM affinity — deterministic ordering for multi-SM cluster runs
- `seq`: insertion sequence — FIFO among same-cycle, same-SM events

> **Active-reader note:** the ACTIVE `sim/engine` orders by `(time, seq)`, and SMs are now resumable `SmCore` actors driven by SmWake/L2Access/MemResponse events over that queue (`sim/sm/src/system.rs`); the engine's global order is the cross-SM interleaving at the shared L2. The driver enforces two fail-closed liveness/causality guards — `SystemError::Livelock` (consecutive zero-progress wakes at a frozen clock) and a per-SM "unpriced request" lower bound alongside `response_due`, so an SM never runs past a response whose L2 pricing hasn't happened yet (the gate's event-vs-constant differential caught exactly that hole).

The reversed `Ord` implementation in `event-engine/src/lib.rs`:

```rust
other.cycle.cmp(&self.cycle)
    .then_with(|| other.sm_id.cmp(&self.sm_id))
    .then_with(|| other.seq.cmp(&self.seq))
```

**NeXosim validation**: NeXosim's `util/priority_queue.rs` (153 LOC) uses
the same core pattern — a `BinaryHeap` wrapper with an epoch counter for
stable same-time ordering. Both frameworks solve the same problem: ensure
FIFO ordering among events at the same simulation time. nvasim's `seq`
counter serves the same role as NeXosim's `epoch` counter.

### Indexed priority queue (NeXosim reference)

NeXosim also provides an `IndexedPriorityQueue` at
`util/indexed_priority_queue.rs` (695 LOC) — a slab-based structure
supporting O(log N) deletion by key. This enables event cancellation
without scanning the queue.

**nvasim relevance**: currently not needed because nvasim does not
cancel scheduled events. If speculative execution or event invalidation
is added, the indexed variant becomes relevant. Trade-off: the indexed
queue stores additional per-entry bookkeeping (slab indices, reverse
map) on top of the heap entry, so it uses more memory per entry than the
simple `BinaryHeap` wrapper. See `indexed_priority_queue.rs` for the
actual field layout.

## Determinism guarantees (current pattern)

nvasim achieves deterministic simulation through three mechanisms:

1. **Single-threaded execution**: same inputs always produce the same
   event ordering — no thread scheduling variability
2. **Deterministic collection handling**: HashMap iteration is sorted
   before hashing or serialization (`main.rs`)
3. **SHA-256 determinism hash**: every dispatched event is hashed via
   `EventHasher` (`hasher.rs`). The `--determinism-check` flag prints
   the cumulative hash at exit. Identical inputs produce identical hashes.

**NeXosim's approach**: causal messaging with one-to-one ordering and
transitivity guarantees. This is stronger than nvasim's structural
ordering + hash verification, but also more complex (requires port/channel
abstractions). nvasim's approach is simpler and sufficient for
single-threaded execution.

Programming Rust Ch3: move semantics ensure no shared-mutable state in
the single-threaded event loop — events are owned values moving through
the queue, not shared references.

## Tagged enum dispatch (current pattern)

`SimEvent` in `event-engine/src/event.rs` has 15 variants dispatched via
`match`. This is a deliberate performance choice:

- **Low-overhead exhaustive dispatch**: the compiler guarantees every
  variant is handled; adding a new variant causes a compile error at every
  match site
- **No vtable indirection**: unlike trait objects (`dyn EventHandler`),
  enum dispatch has no pointer chase
- **Hashing without allocation**: `discriminant()` returns a `u8` for
  each variant; `hash_bytes()` serializes fields directly to a byte
  buffer without heap allocation

The event comment in the source explicitly notes: "Zero-cost dispatch via
match — no virtual function calls on the hot path."

**NeXosim's alternative**: actor model with typed ports — decoupled and
composable, better for multi-model simulations where components are
assembled at runtime. nvasim's tagged enum is the right choice for a
fixed-topology simulator where all event types are known at compile time.

## EventHandler trait (current pattern)

`engine.rs` defines the handler interface:

```rust
pub trait EventHandler {
    fn handle_event(
        &mut self,
        event: &Event,
        scheduler: &mut EventScheduler,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
```

Key design decisions:
- **`&mut self`**: the handler owns mutable simulation state
- **`&Event`**: events are borrowed, not consumed — the engine retains
  ownership for hashing before dispatch
- **`&mut EventScheduler`**: handlers can schedule new events, creating
  the feedback loop that drives the simulation forward
- **`Result` return**: handler errors abort the simulation cleanly via
  `EngineError::HandlerFailed`

This is a closed-loop callback pattern: the handler reads an event,
mutates state, and schedules new events — all within a single
synchronous call.

## Next-event time advancement (current pattern)

The engine skips idle cycles by jumping directly to the next event's
timestamp rather than incrementing cycle-by-cycle. See
`rust-performance.md` for the code, cost model, and `cycles_skipped`
counter details.

## Ownership patterns for simulation (current pattern)

Programming Rust Ch3-4 patterns observed in nvasim:

- **Events move through the queue**: `queue.pop()` returns `Option<Event>`
  (owned). The idiomatic loop: `while let Some(event) = self.queue.pop()`
- **Struct destructuring for split borrows**: `simulator.rs` destructures
  `PerSmState` to obtain `&mut` references to independent fields without
  conflicting borrows (Programming Rust Ch4)
- **Slice references for zero-copy views**: `&[T]` fat pointers into
  cache lines and register files avoid copying data between components
- **Ownership tree architecture**: the simulator owns SMs, which own
  SubCores, which own warps — no circular references, no `Rc`/`Arc`
  needed in the current single-threaded design

## Error taxonomy (current pattern + future target)

**Current**: nvasim uses `anyhow::Result` at the top level and manual
`impl Error` types in crates (`EngineError` in event-engine,
`TensorSuffixError` in trace-parser). `EngineError` has two variants:
`InfiniteLoop` and `HandlerFailed`.

**NeXosim reference design pattern**: `ExecutionError` is a structured
error enum with variants that carry diagnostic data and distinguish
resumable from terminal failures (e.g., halted is resumable; terminated
is not). `SchedulingError` handles scheduling-specific failures
separately.

**Future target**: consider a `SimulationError` enum that classifies
failure modes beyond "handler failed" — e.g., trace exhausted, config
invalid, infinite loop, completed normally. This would enable structured
error reporting in the JSON output and cleaner exit-code semantics.

## State machine patterns (current pattern)

`cluster_barrier.rs` implements a barrier state machine with:
- Dual independent counters (arrives + waits) for correct late-wait
  handling
- Phase advancement only when both counters reach expected count
- Per-warp stall via `pending_memory_ops` increment/decrement

Async Rust Ch9 describes the state machine pattern for component
lifecycle (idle, executing, completed, error). nvasim's barrier is a
concrete instance of this pattern applied to GPU synchronization.

## Future multi-threading reference (brief)

If nvasim adds multi-threaded SM execution, NeXosim's architecture
provides a reference:
- `executor/st_executor.rs` for deterministic single-threaded (current)
- `executor/mt_executor.rs` for work-stealing multi-threaded
- `std::thread::scope` for safe data parallelism (Programming Rust Ch1)

This is noted for future reference only — nvasim is single-threaded and
should remain so until there is a clear performance need.

## See also

- `.claude/knowledge/simulator/nvasim-architecture.md` — what the crates do (this card covers why).
- `.claude/knowledge/simulator/nvasim-dev-patterns.md` — current coding conventions.
- `.claude/knowledge/simulator/rust-type-safety.md` — type-driven design for simulation IDs and counters.
- `.claude/knowledge/simulator/rust-performance.md` — allocation discipline and hot-path optimization.

## Escalate to raw when

- You need the full NeXosim priority queue implementation — see `references/Rust_refs/nexosim/nexosim/src/util/priority_queue.rs` (simple) or `indexed_priority_queue.rs` (indexed).
- You need NeXosim's scheduler architecture — see `references/Rust_refs/nexosim/nexosim/src/simulation/scheduler.rs`.
- You need the NeXosim error taxonomy — see `references/Rust_refs/nexosim/nexosim/src/simulation.rs` (`ExecutionError`).
- You need the EventHandler trait implementation — see `simulator/crates/event-engine/src/engine.rs`.
- You need the barrier state machine — see `simulator/src/cluster_barrier.rs`.
