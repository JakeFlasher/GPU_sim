# User-defined architecture specifications

Files in this directory describe hypothetical or open/custom GPU shapes for
the experimental `sim-arch::spec` path. They are **not** replacements for
`arch_profiles/cards`, which remain the evidence-bound inputs for the
production SM86 replay.

Each specification declares an opaque architecture/ISA token, an admitted
instruction-name set, SIMT width, a resident-group bound, named resources,
optional timing values, and an evidence status. `unvalidated` means that the
values are legal experiment inputs but are not claims about physical silicon.
The bounded pipeline accepts at most 64 resident groups and 64 SIMT lanes.
The standalone ITS executor supports exactly one resident group; the bounded
`simt_u32_v1` backend can run multiple resident groups through
`sim-sm::custom`/`custom-program-run` when their memory effects are disjoint.
The timing backend consumes declared resources and operation bindings, while
instruction names still select the fixed `simt_u32_v1` semantics rather than
defining arbitrary new ISA operations. An absent latency is unknown, never an
invented default. Resource entries may also declare `initiation_interval` and
`max_in_flight`; these are consumed by the shared event pipeline rather than
treated as cosmetic metadata. `operation_resources` can bind admitted bounded
operations to named resources; unknown operations/resources fail closed.

Validate an example with:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin architecture_spec -- \
  arch_profiles/specs/simt_u32_minimal.json
```

The `simt_u32_minimal.json` example is a small resource-schedule fixture. The
`simt_its8.json` example is a Curry-inspired functional ITS fixture with
per-lane PCs and reconvergence barriers. Neither claims Curry compatibility,
physical accuracy, or production SM86 timing.

`simt_its_dynamic.json` is a matching bounded fixture for the opt-in
execution-generated trace demonstration; it exercises value-derived indirect
addresses and remains separate from native SM86 replay.

`sm86_trace_issue_profile.json` is the deliberately narrow bridge into
production `sim-run`: it uses `trace_sass_v1`, `simt_width=32`, one resident
group, and matching `alu`/`memory` issue widths. Passing it with
`--architecture-spec` changes only the hypothetical production issue width;
latency, initiation, in-flight, extra-resource, and operation-binding fields
are rejected because this bridge does not consume them. The fixed NVBit/SASS
trace and evidence-bound card remain unchanged.

`custom-program-run` also emits per-resource diagnostics for each declarative
run: task count, total work, service cycles, peak in-flight population, and
model-relative utilization. `ideal_work_capacity` accounts for both issue
width and initiation interval (`ceil(final_cycle / initiation_interval) *
issue_width`), while `peak_in_flight` follows completion events and therefore
matches the `max_in_flight` capacity domain. This makes resource-level
architecture studies reproducible instead of reducing a specification to one
final-cycle scalar; these values are not hardware utilization counters.
The JSON also carries schedule-level causal diagnostics: last issue and
completion frontiers, the explicit dependency-path bound, the largest
serialized resource-work bound, their conservative lower-bound gap, and the
declared bottleneck resource. The gap is an unattributed model interval, not
an NCU or elapsed-silicon measurement.
