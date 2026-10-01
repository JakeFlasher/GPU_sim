# SM86 factorized latent mechanism contract

This implementation slice adds `sim-semantics::latent`, the
`sim-sm::TraceMetrics::latent_factors()` adapter, and the opt-in
`sim-run --latent-envelope` projection. It is a conservative model boundary,
not a new hardware oracle.

## Contract

For an event, the latent factors are kept independent:

```text
Z(e) = (route, compiler, cache, timing)
```

`CandidateDomain` represents either a finite nonempty candidate set or an
explicit unknown reason. Restriction intersects evidence and rejects an empty
feasible set; union preserves unknown if either explanation is unknown.

`LatentWorldSet::summarize()` computes:

```text
must(f) = intersection of f over surviving worlds
may(f)  = union of f over surviving worlds
```

An exact `must` value is exposed only for a singleton finite domain and is
tagged as hypothesis evidence for these finite worlds. No pressure-balancing
choice is promoted to a route observation.

`FrontierVector` retains issue, modeled-event, launch-envelope, typed-tail,
visibility, writeback-service, endpoint-start, and endpoint-end fields
separately. It checks only mechanically justified ordering edges; launch,
visibility, and writeback domains are not forced into one total order.

`EndpointDescriptor` requires clock domain, triggers, included/excluded work,
serialization, cache state, and clock policy. `same_contract_as()` is only a
necessary structural check; it does not prove synchronized hardware identity
or authorize an affine timing conversion.

## Current replay adapter

`TraceMetrics::latent_factors()` keeps dynamic route/subunit/slot identity
`Unknown`; compiler/ABI, cache ownership/read-from, and unified timing remain
unknown with explicit reasons. `PhysicalPipelineAttribution::class_count_envelope()`
provides independent lower/upper bounds for named candidate classes, with an
explicit unclassified remainder. Existing selected and scheduled pipe maps
remain diagnostics and are not replaced.
`PhysicalPipelineAttribution::resource_demand_envelope()` is the first
predictive-closure adapter: it exposes those candidate classes as abstract
service pools with lower/upper demand intervals. It does not assign a physical
route and is emitted only in the opt-in latent envelope.
`TraceMetrics::operand_bank_pressure` separately aggregates static SM86
regular-register source-bank reads and destination-bank writes for admitted
SASS forms. It is a pressure envelope, not a dynamic route or physical SMSP
observation.

`sim-run::TimingReport::latent_frontiers()` projects only the simulator's
issue/event/tail fields. `TimingReport::simulator_endpoint_descriptor()` names
the simulator event-queue endpoint and explicitly excludes host submission,
NCU front-end work, and observer tails. The descriptor is therefore not
compatible with an Nsight Compute descriptor merely because both report
cycles or nanoseconds.

When an observation-only NCU CSV is supplied, the opt-in envelope also retains
a separate `NsightCompute` endpoint descriptor. It records profiler
front-end/semaphore semantics only; it carries no duration and authorizes no
simulator-cycle conversion.

The envelope is deliberately absent unless `--latent-envelope` is supplied and
is not part of the versioned campaign result schemas. This keeps ordinary
native replay and NCU projection outputs byte-compatible while a retained
trace can opt into the contract.

## Validation

The module has unit coverage for candidate restriction, finite-world
`must`/`may` summarization, frontier contradiction/order checks, endpoint
identity, event-key validation, and JSON round trips. The full workspace run
passed all crates, including 96 `sim-semantics` tests and 507 `sim-sm` tests;
the memory crate passed 193 tests and the direct runner passed 57 tests.
Existing machine-local tests remain
ignored when their external capture environment is absent.

This contract narrows uncertainty representation; it does **not** close the
four external boundaries in `docs/STATUS.md`.

## Value-bearing sidecar boundary

When `--native-payload-sidecar` is supplied, the direct runner now coalesces
complete sectors and forwards them through a separate baseline-unknown logical
L1→L2→DRAM ledger. The report marks this as
`sidecar_only_logical_forwarding_not_native_cache_state`, retains
`pre_store_bytes_unknown`, reports identity completeness/join status, and
reports `hardware_completion=not_observed`.
This prevents a complete payload file from being mistaken for native cache
consumption or physical DRAM acknowledgement.

`L1Cache::preview_dirty_payload_admission()` and
`commit_dirty_payload_admission()` provide the next native-facing seam. They
require resident sectors, reject mask-only dirty state, preserve payload
identity, and commit an entire batch transactionally. Native replay does not
invoke them until a dynamic trace supplies the required residency and
allocation-generation witness. `NativePayloadJoinKey` now defines the
required launch/function/CTA/warp/lane/PC/occurrence identity, while historical
sidecars with missing fields are explicitly refused by the identity checker.
`join_native_payload_records()` matches complete records only against an
independently collected dynamic-key set and refuses unobserved keys before any
cache mutation. Admission plans also snapshot valid-sector masks, dirty masks,
and line-generation witnesses, so mask-only edits and same-tag reallocations
are rejected as stale rather than silently accepted.
`L1PayloadResidencyWitness` is the explicit missing proof input for the final
native-facing admission step: it binds payload allocation identity to the
current L1 line generation and resident sector mask.
`SmCore` now exposes this plan/commit seam to the shared system layer without
changing the compact replay instruction record or default replay behavior.
The public event ingress additionally requires the retained dynamic identity
to be a trace-declared GLOBAL `Store`, binds its CTA/warp/physical warp-slot/
instruction site, and rejects payloads before the modeled store acceptance or
after a newer store supersedes the watched line sector. This remains a
producer/causal contract; it does not expose silicon ownership or completion.
Terminal strict-flush runs carry complete admitted sectors through a separate
model-local forwarding ledger and report optional conservation counters and a
non-authoritative digest; the timing queue remains independent of any claim
that physical DRAM persisted the bytes.

`TraceDir::dynamic_memory_join_identities()` now projects per-lane GLOBAL
memory identities from the validated protobuf using the same per-lane memory
occurrence convention as the NVBit value capture. Its launch component is the
trace kernel ID; producers must bind sidecar `launch_id` to that value.
`dynamic_memory_join_locations()` additionally retains CTA, warp, and
instruction-index locations for event-time integration.
The retained address-carrying canary test exercises this projection on a real
validated trace; it does not imply cache residency or writeback completion.
When PC/CTA/warp/lane/occurrence fields are present but launch/function IDs
are absent, the direct runner enriches only those missing two fields from a
unique trace key; ambiguity and mismatch remain typed refusals.
