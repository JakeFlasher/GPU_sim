//! Kernel-level event system: the per-SM issue actors ([`SmCore`]) over
//! the SHARED L2 model, coordinated by the deterministic engine queue
//! (design §6.1 + §6.3 — the request/response inversion).
//!
//! Event shape (tagged enum, no virtual dispatch):
//! - `SmWake` — resume an SM actor; it runs issue cycles strictly before
//!   the next queued event (plus its own emitted-request response bound,
//!   enforced inside [`SmCore::advance`]).
//! - `L2Access` — a request carrying 1..=4 sectors of one 128 B line
//!   arrives at the shared L2. The engine's `(time, seq)` order IS the
//!   cross-SM interleaving the timeless Phase-1 replay approximated
//!   with trace order. Every requested pending sector remains an L2 lookup
//!   miss but, under the architecture-gated sm_86 policy, joins the matching
//!   allocation-generation/sector's original lower-level fetch. A response
//!   cannot precede its L2-hit lower bound, any newly fetched sector's DRAM
//!   latency/bandwidth completion, or any joined sector's registered original
//!   ready time. The carded aggregate DRAM byte server ([`DramBw`]) can only
//!   push newly issued fetches later than the fixed latency arm.
//! - `L2Fill` — installs a mutation-only read fetch at its own data-ready
//!   time, independently of a returning SM response.
//! - `AtomicCompletion` — retains the intrinsic return-completion time of an
//!   ATOM whose value has no dependence-barrier observer.
//! - `MemoryWriteAck` — for an explicitly timing-active STORE/RED feedback
//!   candidate, drains the candidate's already-derived fence-visible frontier
//!   and only then publishes it to the issuing warp's `MEMBAR.SC.GPU` state.
//! - `MemResponse` — first completes any fill carried by the same request,
//!   then completes the originating SM's L1 fill (or clears a direct
//!   bypass/untagged barrier unit) at exactly the data-ready time and resumes
//!   that SM. Stale L2 installation never suppresses the SM response.
//! - `L1VisibilityComplete` — only when the typed replay configuration enables
//!   the L1 visibility protocol, completes a targeted per-SM L1 invalidation
//!   plan and resumes affected actors at its visibility frontier.
//! - `L1PayloadAdmission` — an explicit external producer event that validates
//!   value-bearing dirty sectors against a trace-bound store site and
//!   model-local residency witness before the existing transactional terminal
//!   writeback path; it is never inferred from a device timestamp or ordinary
//!   replay.
//!
//! Store and reduction mutation-only requests take no data response. Ordinary
//! replay changes L2 state in arrival order and nothing waits on them; the
//! opt-in feedback completion lane adds the typed acknowledgement above.
//!
//! Determinism: SM actors are created in ascending traced `sm_id` order,
//! all scheduling flows through the `(time, seq)` queue, and L2 lookup and
//! fill mutation occur only in their typed event handlers — two runs over the
//! same inputs produce bit-identical schedules and counters.

// Child modules declared by the original public surface.

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("system/queues.rs");
include!("system/dram.rs");
include!("system/observers.rs");
include!("system/execution.rs");
