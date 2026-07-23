# Rust Reference Material for nvasim

Source-material navigator for agents and humans working on the nvasim
simulator. Each source is described with its relevance to nvasim,
key files/chapters to start with, and completeness limits.

These references are the raw sources behind the distilled knowledge
cards at `.claude/knowledge/simulator/rust-*.md`. Start with the cards
for quick answers; escalate here when the cards say "Escalate to raw
when."

---

## NeXosim (`nexosim/`)

**What it is**: A production discrete-event simulation framework in Rust
(v1.0.0). Three-crate workspace: `nexosim` (core), `nexosim-macros`
(derive macros), `nexosim-util` (utilities). Uses epoch-based priority
queues, an actor model with typed ports, causal messaging for
determinism, single-threaded and multi-threaded executors with
work-stealing, and Loom + MIRI testing. MSRV 1.88.0, edition 2024.

**Relevance to nvasim**: NeXosim is an inspiration source, not a
template. nvasim is a single-threaded, fixed-topology GPU simulator;
NeXosim is a general-purpose, multi-model DES framework. The overlap is
in event-queue design, determinism guarantees, error taxonomy, and
testing infrastructure. Multi-threading patterns are noted for future
reference only.

**Key files to start with**:
- `nexosim/nexosim/Cargo.toml` — dependency overview, feature flags
- `nexosim/nexosim/src/util/priority_queue.rs` — simple BinaryHeap
  wrapper with epoch-based FIFO (153 LOC). Direct comparison to nvasim's
  event queue.
- `nexosim/nexosim/src/util/indexed_priority_queue.rs` — slab-based
  indexed queue with O(log N) deletion (695 LOC). Relevant if nvasim
  adds event cancellation.
- `nexosim/nexosim/src/simulation/queue_items.rs` — `EventId<T>` with
  `PhantomData<fn(T)>` for type-safe event registration
- `nexosim/nexosim/src/time.rs` — `MonotonicTime` newtype (via tai-time)
- `nexosim/nexosim/src/simulation.rs` — `ExecutionError` structured
  error enum, simulation lifecycle documentation
- `nexosim/nexosim/src/executor/st_executor.rs` — single-threaded
  executor (deterministic)
- `nexosim/nexosim/src/executor/mt_executor.rs` — multi-threaded
  executor with work-stealing (1000+ LOC)
- `nexosim/.github/workflows/ci.yml` — CI pipeline: check (stable +
  MSRV), test (Linux/Windows/ARM/WASM), Loom dry-run, MIRI (strict
  provenance, 1 + 4 CPUs), clippy, fmt, docs
- `nexosim/.github/workflows/loom.yml` — dedicated Loom concurrency
  verification

**Completeness limits**: Full source available. This is an external
project — version pinned at the commit fetched into `references/Rust_refs/`. Do not
assume API stability across NeXosim releases.

---

## Effective Rust (PDF)

**What it is**: "35 Specific Ways to Improve Your Rust Code" by David
Drysdale. Six parts: Basics (Items 1-7), Traits (8-13), Concepts (14-20),
Dependencies (21-26), Tooling (27-32), Beyond Standard Rust (33-35).

**Relevance to nvasim**: Best-practices guidance for type design,
error handling, visibility discipline, testing strategy, and CI. Most
directly applicable items:

| Items | Topic | nvasim application |
|-------|-------|--------------------|
| 1-2 | Type system | Newtypes, `#[repr(u8)]` for small enums |
| 9 | Builder pattern | Complex config construction |
| 17-18 | Parallelism, panics | Future MT, `panic!()` → `bail!()` |
| 20 | Optimization | Measure before optimizing |
| 22-24 | Visibility | `pub(crate)`, explicit re-exports |
| 29-32 | Tooling, testing, CI | Clippy, proptest, insta, cargo-fuzz |

**Key starting point**: Items 29-32 (tooling and testing) for immediate
CI improvements; Items 1-2 (type system) for the newtype migration.

**Completeness limits**: Full book available as PDF. All 35 items are
present.

---

## Async Rust (PDF)

**What it is**: "Unleashing the Power of Fearless Concurrency" by Maxwell
Flitton and Caroline Morton. Eleven chapters covering async/await,
Tokio, and concurrent systems design.

**Relevance to nvasim**: Mostly async/Tokio-focused, which is NOT
directly applicable to synchronous nvasim. Only three chapters have
transferable insights:

| Chapter | Transferable insight |
|---------|---------------------|
| Ch3 (Custom Queues) | Task queue and scheduling concepts validate nvasim's binary heap approach |
| Ch9 (Design Patterns) | State machine pattern for component lifecycle; circuit-breaker for fault handling |
| Ch11 (Testing) | Deadlock-free verification methodology; channel capacity testing |

**Skip**: Ch1 (async basics), Ch4 (Tokio runtime), Ch7 (streams),
Ch10 (advanced Tokio). These are async-specific and not applicable
to nvasim's synchronous architecture.

**Completeness limits**: Full book available as PDF. All 11 chapters are
present.

---

## Programming Rust 3rd ed (epub)

**What it is**: "Fast, Safe Systems Development" by Jim Blandy and Jason
Orendorff. **Early Release — only 5 of ~25 chapters are available.**

**Relevance to nvasim**: Deep-dive reference for Rust fundamentals
applied to simulation code. Available chapters:

| Chapter | Key concepts for nvasim |
|---------|----------------------|
| Ch1 (Tour of Rust) | `thread::scope`, `chunks_mut()`, `Vec::with_capacity()`, iterator fusion |
| Ch2 (Fundamental Types) | Type aliases, checked/saturating arithmetic, slice references, pre-allocation |
| Ch3 (Ownership and Moves) | Events move through queues (not clone), `Vec<Option<T>>`, `mem::replace`, `swap_remove` |
| Ch4 (References) | Shared vs mutable, lifetime parameters, fat pointers, ownership tree architecture |
| Ch5 (Expressions) | `match` exhaustiveness, `if let`, `let else`, `?` operator, closure patterns |

**Key starting point**: Ch3 (Ownership and Moves) for understanding why
nvasim's event dispatch uses move semantics.

**Completeness limits**: This is an Early Release. Only Ch1-5 are
present. Chapters on error handling, traits, generics, enums, concurrency,
unsafe, and macros are NOT available in this release. Do not cite
chapters beyond Ch5.

To extract text from the epub: it is a zip of XHTML files. Use
`unzip` + Python HTML-to-text extraction, or extract to
`/tmp/epub_extract/prog_rust_text/`.
