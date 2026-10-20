//! Trace → SM replay driver: flattens one traced kernel launch into per-SM
//! block replays using the REAL hardware placement recorded in the trace
//! (per-instruction `sm_id` / `warpid_sm`), joins each dynamic record to
//! the issue schedule (and, when an L1 setup is supplied, to the cache-op
//! manifest + per-record unique-sector expansion), and runs the issue
//! model on every SM.
//!
//! Placement policy v0: spatial placement is traced — each block runs on the
//! SM the hardware ran it on, and each warp uses its observed physical slot.
//! Traces without `sm_id` are a typed error (a synthetic placement model is a
//! future knob, not a silent fallback). Exact temporal block co-residency is not
//! present on the wire; launch-order overlap under the resident-warp and
//! architecture-card shared-memory ceilings plus disjoint traced physical slots is
//! explicitly typed as [`CoResidency::Modeled`].
//!
//! Memory-record classification (L1 mode): a global-class record takes
//! the [`MemRole`] its pc's SIDECAR OPCODE was classified to at join
//! time (`sim_mem::cache_op` port of the Python token rules) — eligible
//! load → modeled L1 sectors; bypass load → counted bypass; store head →
//! hit-tested without allocating; everything else → unified-denominator
//! count only. Sector expansion follows the `l1_trace_adapter` contract:
//! per-lane `addr>>5 ..= (addr+width-1)>>5`, deduplicated per INSTRUCTION
//! in first-seen order (active lanes are the non-sentinel slots — the
//! canonical positional guarantee).
//!
//! v0 limitation (fail-closed, not silent): every dynamic record of a
//! launch must carry the launch's own `function_unique_id` — traced device
//! function CALLs would need per-segment schedules and are rejected typed
//! until a trace exercises them.
//!
//! Memory discipline: threadblock payloads are decoded and semantically
//! validated once when `TraceDir` captures the immutable snapshot. The reader
//! retains compact scheduler/memory metadata plus a per-warp sector arena and
//! discards the address-heavy Prost tree. Extraction never performs a second
//! wire walk or lane-address reconstruction.

// Child modules declared by the original public surface.
mod aggregation;

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("replay/coverage.rs");
include!("replay/fast_warp.rs");
include!("replay/results.rs");
include!("replay/placement.rs");
include!("replay/session.rs");
