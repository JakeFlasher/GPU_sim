---
Use when: reading nvasim type-safety patterns as prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02).
Primary raw source: Effective Rust (Items 1-2, 9, 22-24) + Programming Rust 3rd ed (Ch2-4)
Supporting sources:
- NeXosim `references/Rust_refs/nexosim/nexosim/src/simulation/queue_items.rs` (EventId<T> phantom-typed IDs)
- NeXosim `references/Rust_refs/nexosim/nexosim/src/time.rs` (MonotonicTime newtype via tai-time)
- Programming Rust 3rd ed Ch2 (type aliases, checked arithmetic)
- Programming Rust 3rd ed Ch3 (ownership, move semantics, Option<T>)
- Effective Rust Item 9 (builder pattern)
- Effective Rust Items 22-24 (visibility, wildcard imports, re-exports)
Repo touchpoints:
- simulator/crates/event-engine/src/event.rs (SimEvent, MemReqType, DsMemOp enums)
- simulator/crates/event-engine/src/lib.rs (Event struct, cycle/sm_id/seq ordering)
- simulator/crates/event-engine/src/engine.rs (EventHandler trait, EngineError)
- simulator/src/simulator.rs (SM mini-loop, bit-packing, struct destructuring)
- simulator/src/main.rs (error context chaining, JSON report, named constants)
- simulator/src/sim_config.rs (config validation, ClusterTopology lifetime)
- simulator/src/cluster_barrier.rs (barrier state machine, CTA identity)
Out of scope: error handling conventions (see nvasim-dev-patterns.md), crate workspace layout (see nvasim-architecture.md), performance tuning (see rust-performance.md).
Version family: Historical parent-repo nvasim SM120 / Blackwell snapshot, 2026-04-16.
---

# Rust Type Safety Patterns for nvasim

> **Current-scope note:** prior-art reference for the ACTIVE simulator build (re-founded 2026-07-02). The nvasim source itself is historical — do not port it verbatim. NCU is a first-class ground-truth instrument now: its values are prediction TARGETS, never simulator inputs (Rule 1). The active implementation lives in `sim/` (see `plans/phase2_sim_core_progress.md`).

Type-driven design patterns for simulation code. Each recommendation is
grounded in specific nvasim audit findings and labeled as either a
"current pattern" (what nvasim does today) or a "future target" (what it
should adopt, with rationale from the reference sources).

## Newtype wrappers (future target)

The codebase uses raw primitives (`u32`, `u64`, `usize`) for semantically
distinct identifiers. Five specific gaps were identified in the audit:

| Gap | Current type | Risk | Recommended newtype |
|-----|-------------|------|-------------------|
| Cycle counter | `u64` | Mixing with address arithmetic or instruction counts; `cycle.saturating_add(offset)` silent if offset is a different unit | `struct Cycle(u64)` with saturating arithmetic methods |
| SM ID | `u32` | Cast to `usize` at 12+ indexing sites with manual bounds checks | `struct SmId(u32)` with `as_index() -> usize` |
| Partition ID | `u32` (bit-packed) | Manual `(sm_id << 24) \| req_id` at 3 encode/decode sites; variable naming confuses raw SM ID with packed value | `struct PartitionId { sm_id: u32, req_id: u32 }` with `encode()`/`decode()` |
| CTA identity | `u32` / `usize` | `cta_id` (cluster-scoped) vs `cta_idx` (Vec index) are both bare integers; comparison `cta.cluster_rank == wait.cta_id` is correct but semantically opaque | `struct ClusterRank(u32)` vs `struct CtaIndex(usize)` |
| Barrier index | `u8` with magic 255 | Value 255 means "no barrier" at 5+ locations, with manual `< 6` guard checks | `Option<u8>` or `enum BarrierSlot { Active(u8), None }` |

**NeXosim reference**: `EventId<T>(usize, PhantomData<fn(T)>)` in
`simulation/queue_items.rs` — phantom-typed event identifiers that prevent
scheduling an event with the wrong payload type at compile time.
`MonotonicTime` (re-exported from `tai-time`) wraps simulation timestamps
so they cannot be confused with durations or raw integers.

**Why newtypes over raw primitives**: Effective Rust Items 1-2 — the type
system catches ID swaps at compile time. Zero runtime cost in practice:
the compiler optimizes single-field struct operations to the same
instructions as the inner type. For guaranteed ABI-level layout identity
(e.g., FFI or transmute safety), add `#[repr(transparent)]`; without it,
the layout is unspecified by the Rust reference, even though current
compilers do produce identical codegen.

## Option vs magic constants (future target)

The barrier index magic constant (255 = "no barrier") at
`simulator.rs` and `execution_unit.rs` is a sentinel value pattern that
Rust's `Option<T>` eliminates. Replace `u8` with `Option<u8>`:

- No sentinel value to remember or miscompare
- `match` forces handling of the `None` case
- Programming Rust Ch3: "`Option<&T>` for optional references eliminates
  null-check classes of bugs" — the same principle applies to optional
  barrier slots

This pattern applies wherever nvasim uses a sentinel integer to mean
"absent" or "invalid."

## Checked arithmetic (future target)

Cycle counters use `u64` arithmetic throughout the simulator. Programming
Rust Ch2 documents the checked-arithmetic family:

- `checked_add()` returns `None` on overflow — use for safety-critical
  accumulations where silent wraparound would corrupt simulation state
- `saturating_add()` clamps at `u64::MAX` — appropriate for cycle
  counters where overflow means "simulation ran too long" rather than a
  logic error
- `wrapping_mul()` for intentional modular arithmetic (not applicable to
  cycle counters)

**Current state**: `event-engine/src/engine.rs` uses plain `+` for
`cycles_skipped += event.cycle - self.current_cycle - 1`. The subtraction
is safe because the engine guarantees `event.cycle > self.current_cycle`
in that branch, but documenting this invariant via a newtype's arithmetic
methods would make the guarantee self-enforcing.

## Type aliases as stepping stone (future target)

Programming Rust Ch2: `type Cycles = u64;` provides documentation value
with zero syntactic overhead. Less safety than a newtype (aliases are
transparent to the type checker) but useful as an interim step:

```rust
type Cycle = u64;    // Step 1: document intent
// Later: struct Cycle(u64);  // Step 2: enforce at compile time
```

This approach lets the team adopt semantic naming incrementally without
a large refactor.

## Explicit repr for small enums (future target)

`MemReqType` and `DsMemOp` in `event-engine/src/event.rs` derive
`Clone, Copy, PartialEq, Eq, Hash` but have no `#[repr]` attribute. The
compiler chooses layout freely, which is fine for in-process use but
matters for:

- Deterministic hashing: `discriminant()` and `hash_bytes()` manually
  assign `u8` values (e.g., `*req_type as u8`) — this cast is safe today
  but fragile if variants are reordered
- FFI or serialized traces: explicit `#[repr(u8)]` guarantees 1-byte
  representation

**Note**: no enums in the current codebase carry repr attributes. This
is a future target, not current practice.

## Struct destructuring for split borrows (current pattern)

`simulator.rs` destructures `PerSmState` into individual mutable bindings
to satisfy Rust's split-borrow rules:

```rust
let PerSmState { ref mut subcores, ref mut l1, ref mut shared_mem, .. } = per_sm[sm_id];
```

This avoids repeated `self.per_sm[sm_id as usize].field` indexing and
lets the borrow checker prove that `subcores` and `l1` are disjoint.
Programming Rust Ch4 explains why: shared `&T` vs mutable `&mut T`
references enforce read-write exclusivity — struct destructuring is the
idiomatic way to obtain multiple `&mut` references to different fields.

## Named constants with provenance (current pattern)

`main.rs` defines constants with inline provenance comments:

```rust
const SM120_REGISTER_ALLOCATION_GRANULARITY: u64 = 8;
// ^ deviceQuery confirms 8 on cloud RTX PRO 6000 Blackwell (sm_120)
```

Effective Rust Items 22-24: minimize visibility (`pub(crate)` by default),
avoid wildcard imports, and re-export types carefully. nvasim follows this
for constants — each constant cites its source (deviceQuery, NCU, or
paper reference).

## Error context chaining (current pattern)

`main.rs` uses `anyhow::Result` with `.with_context()` for
production-quality error messages that include file path, kernel index,
field name, and actual value:

```rust
.with_context(|| format!(
    "NVASIM_KERNEL_REGISTERS_JSON ({path}) kernels[{i}] '{name}': \
     num_registers must be a JSON integer (got {v})"
))
```

Effective Rust Item 18: avoid runtime panics. The one known exception
(`simulator.rs` tensor precision) is a documented future target to convert
from `panic!()` to `anyhow::bail!()`.

## Lifetime parameters for config references (current pattern)

`sim_config.rs` uses explicit lifetime parameters:

```rust
pub struct ClusterTopology<'a> { /* borrows SimulatorConfig */ }
```

Programming Rust Ch4: lifetime parameters make borrow relationships
explicit. nvasim uses this correctly for config references that must not
outlive the config they borrow.

## Move semantics for events (current pattern)

Events move through the priority queue — they are not cloned. The `Event`
struct in `event-engine/src/lib.rs` derives `Clone` (for testing) but the
engine's `pop()` returns owned values. Since `Event` contains inline
scalar fields and a `SimEvent` enum (no heap-allocated payload behind
pointers), a move copies the full struct value. The key ownership benefit
is that no separate per-event heap allocation is needed — the
`BinaryHeap` manages a single contiguous `Vec` internally.

Note: Programming Rust Ch3's "move transfers only the 3-word header"
describes heap-backed containers like `Vec` and `String`, not inline
structs like nvasim's `Event`.

The idiomatic event loop pattern is already in use:

```rust
while let Some(event) = self.queue.pop() { /* process owned event */ }
```

## Visibility discipline (current pattern + future target)

**Current**: nvasim uses `pub` for cross-crate types. `pub(crate)` is not
yet used anywhere in the codebase. The 4-crate workspace re-exports key
types through `lib.rs` (e.g., `event-engine/src/lib.rs` re-exports
`Engine`, `SimEvent`, `EventHasher`, `EventQueue`). One function-local
wildcard import exists (`use TensorPrecision::*` in `simulator.rs` for
match ergonomics) — this is an accepted pattern for bringing enum
variants into scope within a match block.

**Future target** (Effective Rust Items 22-24):
- Default to `pub(crate)` for all internal types
- Explicit `pub` only for types consumed by other crates
- Keep wildcard imports scoped to function-local match blocks only

## See also

- `.claude/knowledge/simulator/nvasim-dev-patterns.md` — current coding conventions and error handling patterns.
- `.claude/knowledge/simulator/nvasim-architecture.md` — crate workspace layout and SM timing model.
- `.claude/knowledge/simulator/rust-des-patterns.md` — DES-specific architecture rationale (priority queue design, determinism, dispatch).
- `.claude/knowledge/simulator/rust-performance.md` — allocation discipline and profiling workflow.

## Escalate to raw when

- You need the full `EventId<T>` phantom type implementation — see `references/Rust_refs/nexosim/nexosim/src/simulation/queue_items.rs`.
- You need the `MonotonicTime` newtype design — see `references/Rust_refs/nexosim/nexosim/src/time.rs`.
- You need the complete checked-arithmetic family — see Programming Rust 3rd ed Ch2 "Checked, Wrapping, and Saturating Arithmetic".
- You need builder pattern guidance for config structs — see Effective Rust Item 9.
- You need the PartitionId bit-packing sites — see `simulator/src/simulator.rs` (grep `<< 24`).
