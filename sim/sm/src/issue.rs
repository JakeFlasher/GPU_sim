//! SMSP issue model (design §6.2): per-warp program replay where issue
//! timing is driven by the compiler's own dependence encoding — the SASS
//! control bits (stall counts, dependence-barrier set/wait, yield) — not by
//! simulated scoreboards. This is the 13–17% cycles-MAPE lever the UPC
//! MICRO'25 modern-core work identified `F6`.
//!
//! Model shape:
//! - 4 sub-cores (SMSPs) per SM; a warp lives on sub-core
//!   `warp_slot % 4` (the physical `warpid_sm` slot when traced).
//! - CGGTY-style pick per sub-core per cycle: keep issuing from the held
//!   warp while it stays ready (greedy; a set yield bit drops the hold),
//!   otherwise switch to the YOUNGEST ready warp (largest arrival stamp).
//! - Each issued instruction makes its warp eligible again at
//!   `t + max(stall, 1)`; a set write/read dependence barrier is stamped
//!   with a clear time; an instruction whose control word carries a wait
//!   mask cannot issue until every masked barrier has cleared.
//! - Admitted tensor instructions additionally wait for their SMSP-local
//!   Tensor Core initiation timeline. Other pipelines remain dispatch-only,
//!   so independent non-tensor work may issue while the tensor pipe is busy.
//! - Exact predicated stall-5/yield branches carry architecture-local
//!   readiness: ten cycles for the characterized forward construction and
//!   three target-resolution cycles for ordinary uniform backward loops. A
//!   separate compact-loop classifier admits the measured nine-cycle
//!   next-issue handoff when a fresh predicate immediately feeds a backward
//!   edge no more than 64 bytes away; longer/other forms retain the ordinary
//!   control path.
//! - `BAR.*` (classified by the sidecar's opcode text) is a block-wide
//!   barrier: generation semantics — it releases when every unfinished
//!   warp of the block is waiting at it (warps that already finished do
//!   not block release, matching Volta+ exited-thread semantics).
//! - Block residency: blocks are admitted to the SM in launch order while
//!   resident warps and resident shared-memory bytes stay within their
//!   architecture-local ceilings and concurrent CTAs use disjoint physical warp
//!   slots; a finished block frees both resources for a later CTA (wave execution).
//! - Memory dispatch uses the measured Ampere five-entry SMSP queue (four
//!   queued instructions plus the dispatch latch), operation/register-width
//!   WAR timing, one local address/operand acceptance every four cycles, and
//!   one SM-shared acceptance every two cycles. Cache mutation occurs at that
//!   shared-accept timestamp rather than scheduler dispatch.
//!
//! Memory timing comes in three modes ([`SmMemMode`]):
//! - **Flat**: no cache model; global-class barriers clear at the flat
//!   cold-DRAM latency (the pre-L1 v0 behavior).
//! - **L1Const**: a per-SM sectored L1 probed at memory-pipeline shared
//!   acceptance with fill times known there (constant L2-hit latency below the L1) — the
//!   sim-mem L1 v0 behavior, kept as the lockstep-differential surface.
//! - **L1Shared**: the request/response INVERSION for the shared L2
//!   (design §6.3): an L1 miss's fill time is NOT knowable at issue —
//!   the shared L2's state when the request arrives decides L2-hit vs
//!   DRAM pricing. Misses emit [`MemRequest`]s; ordinary load barriers
//!   and non-bulk async-copy groups retain typed completion targets on
//!   hits, fills, and MSHR retries. [`SmCore::on_response`] completes the
//!   fill and every target at response-event time. An explicit
//!   `LDGSTS.E.BYPASS` instead keeps the direct-L2 async response route.
//!   MSHR-full accesses park in a retry queue drained on the next
//!   completion (the event-mode analogue of the constant mode's
//!   retry-at-earliest-fill; the warp itself keeps issuing until a wait
//!   mask or async-group wait stops it, like the hardware's LSU replay
//!   path).
//!
//! [`SmCore`] is a resumable actor: `advance(horizon)` runs issue cycles
//! strictly BEFORE `horizon` (the event system passes the next queued
//! event's time so no cross-SM effect is overrun; a self-emitted request
//! additionally caps the run-ahead at the earliest time its response
//! could arrive). [`simulate_sm_with_residency`] is the synchronous wrapper for
//! the Flat/L1Const modes (horizon = ∞, no requests possible).
//!
//! The instruction COUNT produced by replay is placement- and
//! timing-invariant: every traced record issues exactly once, so
//! `instructions_issued` is exact regardless of the cycle model's fidelity.
//! The timing-active shared-L2 replay retains an unadjusted causal clock.  A
//! separate architecture-scoped SM-active observation projection can add its
//! measured launch boundary without contaminating event timestamps.

// Child modules declared by the original public surface.
mod candidates;
mod config;
mod memory_types;
mod readiness;

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("issue/diagnostics.rs");
include!("issue/requests.rs");
include!("issue/frontend.rs");
include!("issue/pipelines.rs");
include!("issue/async_copy.rs");
include!("issue/warp_state.rs");
include!("issue/core_init.rs");
include!("issue/core_mem.rs");
include!("issue/core_step.rs");
