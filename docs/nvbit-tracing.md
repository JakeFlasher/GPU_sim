# NVBit Tracing

> **Current scope.** `tools/nvbit_tracer/` is an active simulator-input
> producer. It emits NVBit/SASS/address traces and sidecars; it does not itself
> read NCU counters. NCU capture, clock policy, and label admission belong to the
> separate `tools/local_labels/` path and follow root `CLAUDE.md` plus the active
> local-label plans and provenance schemas. The non-root/counter-free wording retained below is historical and
> must not override the current targets-never-features rule.

The suite ships **one** NVBit foundation — the vendored NVBit 1.8 SDK plus the
shared `isa_def/` opcode tables — and **two** injector tools built on top of it.
Both are NVBit *tools* (LD_PRELOAD / `CUDA_INJECTION64_PATH` targets); neither
is the NCU label factory. The trace path feeds the simulator architecture
[described here](ARCHITECTURE.md).

The two tools were merged from separate trees (Accel-Sim `tracer_tool` from the
profiler; `metric_extractor` from the dissolved SOL-ExecBench project) so they
could share the SDK and the ISA enums exactly once — see
[../MERGE_PROVENANCE.md](../MERGE_PROVENANCE.md).

Everything under `tools/nvbit_tracer/` is producer-side only. The honesty
posture carries through: sm_86 (Ampere) is a fully-classifiable target, while
sm_120 (Blackwell) opcodes the classifier does not recognize are **failed
closed** into typed exclusion sidecars rather than silently mis-tagged — see
[../OPEN_RESEARCH_CHALLENGES.md](../OPEN_RESEARCH_CHALLENGES.md).

## Shared foundation

```
tools/nvbit_tracer/
├── nvbit_release/          # vendored NVBit 1.8 SDK (libnvbit.a + core/ headers)
├── isa_def/                # opcode enums, shared by BOTH tools
│   ├── trace_opcode.h          # enum TraceInstrOpcode (append-only wire enum)
│   ├── operation_type.h        # enum uarch_op_t (ALU/SFU/TENSOR_CORE/…)
│   └── {ampere,blackwell,hopper,…}_opcode.h   # mnemonic → OpcodeChar maps
├── tracer_tool/            # VARIANT A — Accel-Sim SASS/address dumper
└── metric_extractor/       # VARIANT B — counter-free per-warp metric histograms
```

- **NVBit 1.8 SDK** (`nvbit_release/`) is the single vendored drop both tools
  link against (`core/libnvbit.a`, `core/nvbit.h`, `core/instr_types.h`, the
  GPU→host `core/utils/channel.hpp`). Its published `driver <=575.xx` limit is
  internally inconsistent with the CUDA 13.x driver matrix and has no formal
  maintainer correction. The custom tracer is locally smoke-tested on driver
  610.43.02, outside that published envelope.
- **`isa_def/`** holds the mnemonic→opcode maps (`Blackwell_OpcodeMap` in
  `isa_def/blackwell_opcode.h:41`, plus Ampere/Hopper/… peers) and two enums:
  `TraceInstrOpcode` (`isa_def/trace_opcode.h:11`) and `uarch_op_t`
  (`isa_def/operation_type.h:7`). `metric_extractor/classifier.h:24-28`
  includes the Blackwell/Hopper/Ampere maps directly, so **both** tools decode
  SASS through the same tables.

### Additive / append-only schema discipline

The wire enums are extended, never reshuffled. `TraceInstrOpcode` is guarded by
`test_trace_opcode_enum_append_only.py`, which pins a *frozen prefix*
(`isa_def/trace_opcode_frozen_prefix.json`): new members must be appended at the
tail so pre-existing numeric values stay stable across wire readers —
mid-enum insertion of `OP_LDT`/`OP_STT` was the exact regression it was written
to catch. Blackwell `(mnemonic, category)` pairs are pinned by
`test_blackwell_opcode_category.py` (a subset check — e.g. `UTCCP` must map to
`LOAD_OP`, not `UNIFORM_OP`). The `metric_extractor` JSONL wire is versioned in
`metric_extractor/JSONL_SCHEMA.md`: `schema_version` (currently **2**) bumps
only on a wire-incompatible change, and adding an optional field with a
documented default is explicitly *not* a bump.

## Two injectors, two consumers

```
                       CUDA_INJECTION64_PATH  (or LD_PRELOAD)
                                   │
              ┌────────────────────┴─────────────────────┐
              ▼                                           ▼
  ┌───────────────────────────┐             ┌───────────────────────────────┐
  │  VARIANT A: tracer_tool.so │             │ VARIANT B: metric_extractor.so│
  │  (Accel-Sim SASS + address │             │ (counter-free per-warp metric │
  │   dumper; full dyn trace)  │             │  histograms; NO addresses by  │
  │                            │             │  default)                     │
  └───────────────┬────────────┘             └───────────────┬───────────────┘
                  │ writes                                    │ writes
                  ▼                                           ▼
  ┌───────────────────────────┐             ┌───────────────────────────────┐
  │ TRACES_FOLDER/             │             │ metric_extractor_launches.jsonl│
  │  ├─ dynamic_trace.pb       │             │  (1 JSON obj / kernel launch)  │
  │  ├─ threadblocks/…         │             │ + opt-in Tier-2 sidecar        │
  │  ├─ extra_info/…json       │             │   (METRIC_EXTRACTOR_TIER2_     │
  │  └─ *_exclusion sidecars   │             │    ADDRESS_DUMP=1)             │
  └───────────────┬────────────┘             └───────────────┬───────────────┘
                  │ consumed by                               │ consumed by (historical/evaluation)
                  ▼                                           ▼
  ┌───────────────────────────┐             ┌───────────────────────────────┐
  │ active simulator/profiler │             │ SOL-ExecBench evaluation:      │
  │ sim/trace + run_suite.py  │             │ nvbit_metrics surface plugin   │
  │ (workloads/simbench_bw/)  │             │ launches_parser.py (strict)    │
  └───────────────────────────┘             └───────────────────────────────┘
```

Both tools decode SASS through the **same** `isa_def/` maps and the **same**
NVBit 1.8 channel; they diverge only in *what they emit* and *who consumes it*.

### Variant A — `tracer_tool` (SASS / address dumper)

Forked from an Accel-Sim / MICRO-2025 trace-tool tree (`tracer_tool/README.md`
"Provenance"). It captures a full dynamic trace — per-warp SASS, per-lane
addresses, threadblock geometry — into a protobuf (`dynamic_trace.pb`) plus a
`threadblocks/` tree and a static-metadata JSON sidecar.

The profiler drives it from `workloads/simbench_bw/run_suite.py`:
`NVBIT_TRACER_SO` resolves to `tools/nvbit_tracer/tracer_tool/tracer_tool.so`
and `handle_trace()` sets `CUDA_INJECTION64_PATH` with `TRACES_FOLDER` for the
output root after the shared loader preflight succeeds. Its paired baseline run
deliberately scrubs `CUDA_INJECTION64_PATH` so the un-instrumented reference is
clean.

Inspect a trace with `tracer_tool/trace_printer <dir>` (it takes the containing
directory, not the `.pb`). The per-instruction wire contract (every memory
instruction carries a non-`UNSPECIFIED` `memory_op_class` and a `data_width>0`)
is validated offline by `import_trace_contract.py`.

### Variant B — `metric_extractor` (counter-free histograms)

Purpose-built for the dissolved SOL-ExecBench framework as **historical Surface 1**
of its five counter-free analytic surfaces (`docs/sol_execbench/CLAUDE.md`). This
section documents the retained evaluation path; it is not a current root
measurement or simulator contract. Instead of dumping addresses, it accumulates
per-launch, per-warp **metric histograms** and emits one JSON-L record per kernel
launch (`metric_extractor_launches.jsonl`; `metric_extractor/JSONL_SCHEMA.md`). Each record carries the launch identity,
grid/block/cluster geometry, and an `instruction_class_histogram` over the AC-7
buckets (`ffma`, `mufu`, `ldg`, `stg`, `lds`, `sts`, `hmma`, `wgmma`,
`tcgen05`, `tma`, `dsmem`, `bra`, `barrier`, `warpsync`, `unknown`) plus sector
efficiency, bank conflicts and the branch active-mask histogram.

The dissolved SOL design injected it via
`CUDA_INJECTION64_PATH=…/metric_extractor.so` on the evaluator subprocess
(`docs/sol_execbench/CLAUDE.md` historical Surface-1 row). The eight surviving
SOL injector call sites remain owner-scoped historical/evaluation material; this
document does not claim universal runtime enforcement or authorize a new
injection. The strict historical consumer is
`tools/sol_baselines_harness/metric_extractor/launches_parser.py`, and
`tools/sol_baselines_harness/b200_preflight.py` lists the `.so` as a required
`nvbit_metrics_tool` probe.

**Opt-in Tier-2 address sidecar.** By default the extractor captures *no*
addresses and allocates *zero* managed memory for them (`metric_extractor/common.h`
Tier-2 notes). Setting `METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1` turns on an
ordered, compact per-sector-event sidecar — the env gate lives at
`metric_extractor/metric_extractor.cu:376-403` (`init_tier2_address_dump_locked`),
with `METRIC_EXTRACTOR_TIER2_PER_LANE` and
`METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES` as further knobs. Records use a fixed
`CompactAccessRecord` layout (524288 records/launch, 64 sectors/record) and can
be SHA-256-stamped for cross-surface joins (`tier2_sha256.h`,
`tier2_frame_serializer.h`).

## Exclusion taxonomy (fail-closed, typed)

When a tool cannot *truthfully* represent an instruction's memory shape on the
wire, it does **not** guess. It emits a typed exclusion sidecar and skips the
memory payload. Three typed reasons share one structural schema
(`import_nvbit_exclusion.py`):

| typed reason | meaning |
|---|---|
| `unsupported_memory_path` | classifier returned `kUnspecified`, or the op-class is one of the excluded-but-recognized classes (TEXTURE/SURFACE/ATOMIC/TMA/ASYNC_COPY/CLUSTER_DSMEM) |
| `unmapped_memory_range` | a lane's address falls outside every live allocation in the per-context registry |
| `cross_allocation_lane_span` | one warp instruction's active lanes span multiple distinct non-zero allocations |

Producer emission is centralized in `nvbit_tracer::aggregate(...)`
(`tracer_tool/tracer_tool.cu`), serialized one file per reason by
`exclusion_aggregator.cc`, and admitted by the consumer's `_SCHEMA_FILES` map in
`import_nvbit_exclusion.py`. All three layers are held in lockstep by
`test_exclusion_taxonomy_parity.py` — a producer that emits a reason the
consumer schema does not admit fails the parity test.

### Blackwell unknown-opcode → `kUnspecified` → fail-closed

This is the honesty pivot for sm_120. The two orthogonal classifiers
(`memory_op_class.h`, `memory_op_access_kind.h`) return the sentinel
`kUnspecified = 0` for any opcode head they do not recognize. `tracer_tool.cu`
routes both sentinel cases (and the six excluded-but-recognized classes) through
the `unsupported_memory_path` sidecar and **skips the entire memory-payload
emission** — no addresses, no `memory_op_class`, no `data_width`
(`tracer_tool/tracer_tool.cu:3219-3247`):

```
   opcode head ──▶ classify(class) ─┐
                                     ├─ either == kUnspecified? ──▶ aggregate(
   opcode head ──▶ classify(kind) ──┘        (unknown Blackwell op)   "unsupported_memory_path",
                                                                       path_resolved=false)
                                                                    ──▶ SKIP payload (fail closed)
```

The downstream `path_resolved` flag distinguishes the two: `true` for a
recognized-but-excluded class (promoted to `UNSUPPORTED_BLACKWELL_MEMORY_PATH`
in the headline-validity gate) versus `false` for a genuinely unknown opcode
head (kept on `TRACE_OPCODE_COVERAGE_INCOMPLETE`). Known Blackwell memory
opcodes are handled explicitly first: `LDT`/`STT`/`LDTM`/`STTM` route to
`kTmem` and `UTCCP`/`UTCSHIFT` to `kClusterDsmem`
(`memory_op_class.h:180-201`), so only the truly-unclassified tail fails closed.
`metric_extractor` mirrors this posture on its side with a `g_unknown_opcode_count`
that is merged into the terminal `filter_stats.json`
(`metric_extractor/metric_extractor.cu:599`).

## NVBit 1.8 binary-version fallback

`CU_FUNC_ATTRIBUTE_BINARY_VERSION` can return `CUDA_ERROR_INVALID_VALUE` on
both supported host families. The current CUDA 13.3-target smoke on the RTX
3090 Ti / sm_86 host returned `CUresult=400`; the cloud sm_120 path has also
exercised the same branch.
`tracer_tool.cu` falls back to deriving `binary_version` from
`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}` as `major*10 + minor`
(86 or 120 in those native-architecture smokes; cf.
`BLACKWELL_RTX_BINART_VERSION 120` in `isa_def/blackwell_opcode.h:36`). It fires
once per run with one `INFO:` line. Because it records the device capability,
the fallback is qualified for the repository's native-architecture or PTX-JIT
targets; it must not be treated as the original cubin version for an arbitrary
older-minor cubin running through same-major binary compatibility. The
architecture-conditional `sm_120a`/`sm_100a` variants (TMA, fifth-gen tensor
cores) are out of the current smoke-test scope and may need additional
binary-version handling.

## Building & running

```bash
bash tools/nvbit_tracer/pinned_build_env.sh --ensure
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer check-pinned-env
```

The custom-tool build is pinned to nvcc 13.2, g++-14, and exact protobuf 34.1.
This is repository-specific: NVBit 1.8 merely updated its bundled headers to
CUDA 13.2 and does not publish a CUDA 13.2 maximum for target applications.
The sourced environment validates every pin; `bc` is not required. Task7/H3
still holds: canonical Git worktrees refuse build, clean, regeneration,
replacement, and repin targets. An actual build check must use a non-Git
temporary copy with `NVBIT_TEMPCOPY_BUILD_OK=1`; the sentinel is invalid inside
every Git worktree and the protected artifact hashes must remain unchanged.

Python readers under `workloads/simbench_bw/_pb_generated` are tracked
authorities. Restore missing bytes with `git restore --source=HEAD --worktree
-- workloads/simbench_bw/_pb_generated`; do not generate replacements in the
canonical checkout. A missing Python protobuf runtime is a separate
prerequisite failure resolved from `requirements.txt`.

```bash
# Variant A (profiler path is run_suite.py --mode trace; manual form:)
REPO_ROOT="$(git rev-parse --show-toplevel)"
PB_LIB="$REPO_ROOT/.toolchain/pb/lib"
TRACE_ROOT="$REPO_ROOT/results/nvbit_capture_staging/manual_trace"
mkdir -p "$TRACE_ROOT"
LD_LIBRARY_PATH="$PB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
  USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$TRACE_ROOT" ./your_cuda_app

# Variant B is a historical SOL evaluation path; this example implies no invocation.
```

**Do not** run either tool in the same process as `ncu`, `nvprof`, or `nsys` —
that is a CUDA driver-injection collision, distinct from project policy. The active
tracer path itself never invokes NCU; NCU capture and label admission are owned by
`tools/local_labels/` under the root protocol. This separation does not make the
historical SOL injector contract current.

## Related docs

- [active architecture record](ARCHITECTURE.md) — where trace evidence enters the Rust simulator and its output contracts
- [../OPEN_RESEARCH_CHALLENGES.md](../OPEN_RESEARCH_CHALLENGES.md) — historical challenge framing for sm_120 classification and refuted predictor claims
- [../MERGE_PROVENANCE.md](../MERGE_PROVENANCE.md) — historical provenance for merging the two tool variants under one SDK
- [sol_execbench/CLAUDE.md](sol_execbench/CLAUDE.md) — historical SOL-ExecBench surfaces; `metric_extractor` was Surface 1
- `tools/nvbit_tracer/README.md` / `tools/nvbit_tracer/metric_extractor/JSONL_SCHEMA.md` — authoritative build + wire-format references
