# Rust event-driven simulator

The workspace contains:

| Crate | Role |
|---|---|
| `sim-engine` | Deterministic event scheduling |
| `sim-trace` | Enhanced NVBit protobuf decoding and alias-aware global-atomic sector projection |
| `sim-sm` | Warp issue, readiness schemas 1–13 including scalar-global work, operation-specific operand-read barriers, streaming services, finite hit/miss-aware admission candidates, feedback-coupled queue, per-warp MEMBAR completion, fixed and work-dependent completion tails, typed completion frontiers, ATOMG/RED topology, phase CLOCK diagnostics, branch readiness, replay, and sm_86 writeback priority |
| `sim-mem` | Sectored L1/L2 and DRAM timing, including architecture-local cache-token routing |
| `sim-arch` | Architecture/SKU configuration and typed sm_86 256-byte atomic contention topology |
| `sim-run` (`direct_run/`) | Direct end-to-end composition and JSON output |
| `sim-run-campaign` (`run/`) | Optional retained manifest/evidence campaign interface |

The first target is `sm_86`. Unknown `sm_120` behavior remains explicit.

Build and test:

```bash
cargo build
cargo test --all-targets
```

Run a trace directly:

```bash
cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --mode flat
```

Shared-L2 replay remains cold per launch by default. For a trace containing an
ordered prewarm/target chain on one device and stream, retain only fully
drained L2 state explicitly:

```bash
cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --pchase-card PCHASE_CARD.json \
  --mode l1-l2 \
  --cache-scope stream-l2
```

`stream-l2` consumes every traced launch in order and therefore cannot be
combined with `--kernel-id`. Per-SM L1, MSHR, dependency, scheduler, CTA,
pipeline, event-queue, and DRAM-server state is rebuilt at every launch.

Evaluate an explicit bounded memory-service hypothesis file without changing
issue scheduling or the completion frontier:

```bash
cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --pchase-card PCHASE_CARD.json \
  --mode l1-l2 \
  --memory-service-candidates CANDIDATES.json
```

The schema-1 document is bounded to 64 KiB and at most 16 candidates; duplicate
IDs and half-disabled rates fail before trace extraction. Results first appeared in
readiness schema 7; schema 8 additionally identifies scalar-global operand-read
admission barriers. This launch-scoped research option is incompatible with
`stream-l2` and does not make an incomplete STORE/RED frontier complete.

Evaluate finite STORE/RED operand-admission queues against the exact L2
outcomes produced by replay:

```bash
cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --pchase-card PCHASE_CARD.json \
  --mode l1-l2 \
  --memory-admission-candidates ADMISSION_CANDIDATES.json
```

This separate schema-1 file is also limited to 64 KiB and 16 candidates.
Readiness schema 9 reports finite SM/per-SMSP queue admission and an SM-shared
FIFO service priced from per-instruction L2 hit/miss sectors. It preserves the
observed issue/cache order and changes no timing; service and admission
candidate files are mutually exclusive on the direct CLI.

Run exactly one finite candidate with scheduler/memory feedback:

```bash
cargo run -p sim-run -- \
  --trace TRACE_DIR \
  --control CONTROL_BITS.json \
  --arch-card ARCH_CARD.json \
  --pchase-card PCHASE_CARD.json \
  --mode l1-l2 \
  --memory-admission-feedback ONE_CANDIDATE.json
```

The document must contain exactly one admission candidate. Schema 10 reserves
queue space at issue, yields the SM until exact L2 pricing is processed, and
uses queue release as a real readiness constraint. This timing-active lane may
change issue and cache order. Shared-capacity denial repicks other ready work
on the SMSP rather than creating a dispatch bubble. Its declared admission
latency must equal the issue model's read-barrier latency, and the complete
candidate is part of replay configuration identity. The lane remains opt-in
and mutually exclusive with the two observational candidate modes.

`l2_miss_service_width_sectors` is optional. Omitted/zero retains the legacy
fluid rational rate. A positive value batches misses into fixed-width quanta
and charges a full final quantum; values above 256 fail before replay. The
current sm_86 rate/width sweep retains this mechanism for research but promotes
no nonzero width or new rate as a default.

`membar_waits_for_service_completion: true` activates readiness schema 11.
Admission still releases at service start; `MEMBAR.SC.GPU` waits for the issuing
warp's own service-done frontier. The lane also emits first-to-second and
second-to-third CLOCK issue intervals. Omitting/false preserves schema 10.

`write_completion_latency_cycles` activates schema 12 when nonzero. This tail
starts after service done, occupies no service capacity, and affects only the
per-warp MEMBAR frontier. It is invalid unless MEMBAR completion feedback is
enabled. The sm_86 sweep promotes no nonzero latency.

`write_completion_request_cycles` and `write_completion_region_cycles`
activate schema 13 when either is nonzero. They charge exact L2 requests and
unique 256-byte regions, respectively, on an independent completion server
with `write_completion_parallel_lanes` lanes. A job starts no earlier than its
admission-service completion and occupies the earliest available lane for its
work; the optional schema-12 latency follows without occupying that lane.
Completion work requires MEMBAR feedback and one to 32 lanes. Omitted work and
lanes preserve schema 11/12 behavior.

For host-cost research, an explicit feature build emits exact event and phase
counters on stderr while leaving result JSON unchanged:

```bash
cargo build --release -p sim-run --features performance-counters
SIM_RUN_PERFORMANCE_COUNTERS=1 target/release/sim-run ...
```

The feature build remains quiet unless `SIM_RUN_PERFORMANCE_COUNTERS=1` is set.
The normal binary has no performance-counter state. Clean wall timing and the
feature-gated counting run must be measured separately. The retained benchmark
driver is under `workloads/research_probes/simulator_event_throughput/`.

The shared-system driver stores consecutive same-SM/same-time reply-less
STORE/RED requests from one emission as a single physical queue event, then
executes their existing L2 transactions in exact order. Logical event counters
remain request-counted; feature diagnostics separately report physical queue
events, batches, and batched requests. Response-bearing operations, ATOMG, and
timing-active feedback retain individual events.

Direct `sim-run` uses `TraceDir::open_direct_compact`: every threadblock is still
read, wire-preflighted, decoded, semantically validated, and compacted, but its
raw protobuf bytes are released before replay state is constructed. Explicit
`TraceDir::open_direct` and campaign snapshots retain raw bytes for diagnostic
decoding. A compact loader reports a typed error instead of reopening a raw
payload by path.

Library composition may use `sim_direct_run::PreparedReplay` for ordinary,
launch-cold replay. The facade owns one compact trace, validated schedules, and
cards; exposes preparation-scoped opaque kernel keys; and reconstructs borrowed
kernel locators only inside `replay_launch`. Its full-only admission mode
rejects sampling declarations before trace loading. It deliberately rejects
`stream-l2` and all candidate/feedback lanes, which remain supported by the
standalone direct runner. A prepared completion frontier is diagnostic output,
not a hardware-duration or external scheduling source.

Compact instructions store validation-proven function and threadblock-warp
identity once per owning warp; exact optional SM/physical-warp values remain
per record with presence bits. The retained record is 20 bytes instead of 32.
Performance builds may set `SIM_RUN_STREAM_TRACE_PAYLOAD=1` to exercise the
exact-generation rostered streaming research path; normal builds keep the
measured full-map compaction path.

Full-hierarchy counter JSON includes `l2_by_source` lanes (`l1_fill`, `bypass`,
`untagged`, and `store`) with sector and hit counts. These are causal replay
diagnostics, not NCU labels supplied as inputs.

Direct result JSON has root `schema_version: 1`, records the simulator package
version, memory mode, cache scope, whether carded DRAM bandwidth was active,
both candidate counts, and any timing-active feedback candidate id. The
optional historical `sim-run-campaign` v2 wire predates typed completion
frontiers and intentionally remains frozen; use direct JSON until a v3 wire is
defined. Rust crate APIs are pre-1.0 and source compatibility is not promised.
Campaign resume additionally compares the exact embedded candidate and launch
configuration and requires outputs to strictly postdate every replay input.

Each launch reports last issue, the last processed shared-system event, their
raw maximum, and a typed completion frontier. `complete` means no known
timing-bearing tail lies outside the event model;
`incomplete_unmodeled_retirement_tails` makes the raw maximum a lower bound.
Neither status includes Nsight Compute's launch envelope or untraced host gaps.

The default workspace members exclude the campaign runner. Use it only when a
historical campaign manifest is intentionally in scope:

```bash
cargo run -p sim-run-campaign -- RUN_MANIFEST.json
```

NCU metrics are used only for comparison after simulation. The simulator input
surface is traces, static program information, and architecture parameters.
