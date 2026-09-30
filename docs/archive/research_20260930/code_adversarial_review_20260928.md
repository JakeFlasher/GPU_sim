# Alpha code adversarial review and simplification

Updated 2026-09-28. This review covers the alpha campaign runners, trace
adapters, payload/writeback seams, L1 visibility seam, and the SM event
counter changes in the current worktree. It is a source review, not a test
suite or a promotion/verification gate.

## Fixed findings

| Severity | Finding | Change |
| --- | --- | --- |
| High | A rejected `MemoryOrderLedgerV1` event advanced the requester's sequence before the value/order checks completed. One bad record could poison every later record. | Sequence state now advances only after a successful write/read admission. |
| High | The payload sidecar was read completely before its size limit was checked. A large file could allocate unbounded memory in the admission path. | Sidecar loading now streams at most the configured limit plus one byte. |
| High | Value-bearing writeback accepted duplicate sectors and mixed allocation generations when the line/mask happened to match. | Payload writeback requires unique sectors and one allocation identity. |
| High | L2 terminal-flush commit searched every set for a tag instead of deriving the owning set. Corrupt/duplicate tags could clear the wrong way. | Commit derives the set with the cache's signed slice/set mapping first. |
| Medium | Cacheable visibility qualification inspected every sector in a line, so an unrelated remote sector could reject an otherwise admissible access. | Qualification is restricted to the requested mask and validates request metadata first. |
| Medium | The fence classifier treated arbitrary `*.GPU`/`*.SYS` opcode heads as fences. | Only explicit `MEMBAR` forms are admitted by the observation seam. |
| Medium | Physical-pipe identifiability reported a one-solution result as truncated when the search limit was exactly one. | The search keeps one extra solution before deciding truncation; candidate input types are bounded. |
| Medium | The alpha metric campaign reused an existing binary without any identity binding, allowing stale source execution. | Alpha mode always recompiles the workload. |
| Medium | NCU CSV parsing assumed the header was row zero and treated an empty export as successful. | Header/unit rows are located defensively and empty launch tables become a recorded stage error. |
| Medium | A malformed SASS control parse escaped the stage ledger and aborted the workload before later observations. | Control parsing is recorded as a stage error and replay is skipped while native/NCU collection can continue. |
| Low | The campaign settings list used raw workload IDs even though base/large cohorts intentionally share IDs. | Settings now use the cohort-qualified campaign IDs. |
| Medium | The finite SASS interpreter accepted arbitrary `ISETP` forms and could silently ignore predicate operands, and it allowed out-of-range predicate registers to become implicit state. | The interpreter now admits only the explicit unsigned `ISETP.*.U32.AND`/`PT` form and bounds predicate indices. |
| Low | Visibility events were counted as ordinary coherence completions, conflating protocol observations with L2 directory completions. | Performance counters now expose a distinct `l1_visibility_event` count. |
| Low | Dead alpha identity/device helpers and an unused adapter digest helper added noise. | Removed unused metric-domain identity/digest code and the unused adapter helper. |

## Alpha identity simplification

The trace adapter, register-value adapter, and finite SASS value interpreter
keep their existing schema names for compatibility, but default alpha results
carry path/size identities only. Digest fields and manifest binding are
available only through the explicit `--bound` option. The full alpha campaign
and metric-domain campaign continue to emit no artifact hashes, freshness
checks, or scoring labels.

## Remaining high-risk boundaries

These are not silently promoted by this review:

1. Resident-hit L1 writeback is an opt-in model path. Its terminal queue
   service does not establish proprietary L1-to-L2 ownership, payload bytes,
   eviction ordering, or host-flush semantics.
2. The L1 visibility queue is a typed, model-local directory experiment. It
   requires explicit observations and remains a refusal boundary for missing
   qualifiers, stale fills, unsupported atomics/reductions, and pending
   invalidations.
3. Native timing remains split across warm-cache CUDA-event timing and cold-cache
   controlled NCU duration. NCU collection uses `--clock-control none`; no
   drift-free locked-clock result is claimed.
4. Candidate-pipe assignment, scheduled frontiers, compiler subset lowering,
   and payload ledgers remain bounded mechanisms rather than physical hardware
   witnesses or unrestricted NVIDIA compiler/ABI generation.

## Build-only observation

The affected Rust workspace path compiles with the performance-counter feature
after the review changes. No test suite, artifact hashing campaign, or
verification/promotion gate was run for this review.
