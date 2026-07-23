# Event-driven NVIDIA GPU simulator

This repository is being refocused around one goal: build an accurate, fast,
trace-driven simulator for modern NVIDIA GPUs. The simulator core is written in
Rust, consumes NVBit SASS/address traces plus static SASS control information,
and is validated against Nsight Compute (NCU) measurements from real silicon.

The first target is Ampere `sm_86`; Blackwell `sm_120` follows only after the
`sm_86` **M-SIM-1** gate closes and the operator explicitly approves migration.
The sole publication-binding headline is `exec_cycles_gpc_max` /
`gpc__cycles_elapsed.max`: cycle MAPE ≤10% meets the engineering target, >10%
and ≤15% is floor-only, and >15% is publication failure. The long-term speed
objective is 100–1000x faster than cycle-ticking simulators; no current
benchmark substantiates that speed objective yet.

NCU metrics are prediction targets and validation references. They are never
simulator inputs or per-cell fitted features.

CUDA authorities are role-specific: canonical C1 workload capture uses nvcc
13.3, supplemental SOL launch-set v3 uses nvcc 13.2, and the NVBit/tracer
artifact build is independently pinned to nvcc 13.2. None is a fallback for
another role.

## Authority and documentation

The active documentation map is [`docs/README.md`](docs/README.md). Evidence
manifests and versioned aliases govern byte identity; root `CLAUDE.md` governs
project policy; `AGENTS.md` is limited to the consultation protocol; current
state, design, inventory, runbook, and accepted ADRs live under `docs/`. The
root `ARCHITECTURE.md`, `HANDOFF.md`, `MANIFEST.md`, `MERGE_PROVENANCE.md`, and
`OPEN_RESEARCH_CHALLENGES.md` are retained historical records and are not
operational instructions.

The knowledge corpus contains 353 non-index content cards plus two physical
index/router files (`.claude/knowledge/INDEX.md` and `patents/INDEX.md`);
`_template.md` is excluded. The generated manifest indexes all 355 routable
Markdown rows.

## Current state

The repository has working Rust libraries for deterministic event scheduling,
NVBit trace decoding, SM replay, L1/L2/DRAM modeling, architecture-card
loading, and seven NCU-shaped secondary metric emitters. The current Rust
workspace gate reports 201 passing tests and three explicitly ignored
machine-local evidence tests.

This is not yet an end-to-end simulator application:

- there is no production `sim-cli` binary;
- no non-test consumer drives `replay_kernel` end to end;
- `gpc__cycles_elapsed.max` is required but not emitted;
- `KernelReplayResult.max_cycles` is only a directional diagnostic;
- no benchmark enforces the events-per-second or 100–1000x speed objective.

See [current status](docs/STATUS.md) for the campaign and cleanup state.

## Implemented data path

```text
NVBit tracer_tool
  -> traces_enhanced protobuf
  -> sim-trace (committed Prost bindings)
  -> sim-sm replay and issue model
  -> sim-mem L1 / shared L2 / DRAM model
  -> sim-engine deterministic event queue
  -> sim-emit NCU-shaped predictions
```

NCU capture is a separate label path. Prediction and label artifacts meet only
in evaluation; labels never flow back into trace, scheduling, memory, or
calibration inputs.

## Quick start

The default gates are CPU/offline and do not require a GPU:

```bash
cd sim
cargo fmt --all -- --check
cargo test --workspace --all-targets
cd ..
.venv/bin/python -m pytest -q
.venv/bin/python workloads/simbench_bw/validate.py --all
```

The Python default excludes tests marked `campaign_evidence` and
`hardware_evidence`. Those tests require machine-local artifacts and must run
through explicit gates; absence must never be interpreted as a pass. Run the
archive-integrity and frozen machine-local hardware gates separately:

```bash
tools/dev/campaign_evidence_gate.sh
tools/dev/hw_evidence_gate.sh
```

Archive-integrity, offline/frozen-hardware, live-campaign, campaign-M7
completion, and M-SIM-1 are independent gates; passing or authorizing one
satisfies none of the others. See [test boundaries](docs/testing.md) before
interpreting a green offline run.

## Repository map

| Path | Purpose |
|---|---|
| `sim/` | Rust discrete-event simulator workspace |
| `tools/nvbit_tracer/` | NVBit trace producer, protobuf wire format, and active instrumentation |
| `tools/local_labels/` | Core NCU capture, admission, provenance, and evidence machinery |
| `arch_profiles/` | Per-architecture cards and fail-closed gaps |
| `data/pchase/` | Microbenchmark-derived memory geometry and latency inputs |
| `tools/sass_control/` | Static SASS control-bit extraction |
| `tools/cache_predictor/`, `tools/cache_replay/` | Reference/parity models supporting the Rust memory model |
| `evidence/` | Tracked immutable fixtures, manifests, and historical path aliases |
| `workloads/` | Core capture workloads plus optional evaluation bundles |

The detailed boundaries are in [architecture](docs/ARCHITECTURE.md) and the
[component inventory](docs/COMPONENTS.md). Host, toolchain, NVBit, and capture
commands are in the [host runbook](docs/runbook_host.md).

The active label contract is the three-part v5 authority: the sidecar and
admitted schemas under `workloads/simbench_bw/schemas/`, plus
`tools/local_labels/metric_registry.json`. The active measurement-manifest v2
schema is a separate workload-provenance contract, not a timing label schema.

## Binding decisions

Architecture and evidence decisions are recorded under
[`docs/decisions/`](docs/decisions/). ADR 0001 accepted
`CONTINUE_PRESERVE_PINNED`: fresh Task7/H3 continuation may proceed outside RLCR,
while the historical defect attempt remains zero-credit and reuse-forbidden.
ADR 0002 accepted `CYCLES_ONLY_BINDING`: cycles are the sole publication
headline; the seven implemented secondary descriptors remain reported and
ungated.

Current canonical accounting is 344 cells: 122 runnable, 222 excluded, and
1/122 defensible runnable credit. The stopped 49-row revision is also
zero-credit and reuse-forbidden. The interrupted 53/201 SOL runnability attempt
is separately archived, canonical v1 is restored, and supplemental v3 remains
separate and live-disabled pending its pinned canary.

The current engineering state is recorded in [STATUS](docs/STATUS.md), and
binding cleanup/design choices are recorded in [ADRs](docs/decisions/). The
[simulator execution plan](plans/micro2027_ncu_sim_execution_plan.md) and the
[sm_86 capture plan](plans/sm86_comprehensive_live_capture_campaign_plan.md)
are the authoritative execution plans. Accepted ADRs amend them; where an older
plan statement conflicts with a later accepted ADR, the ADR controls.
