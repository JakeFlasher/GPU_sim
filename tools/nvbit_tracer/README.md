# Standalone NVBit Tracer (host build)

Self-contained NVBit tracer tool, wired for **host-side** builds with the
repository-pinned CUDA 13.2 / NVBit 1.8 / libprotobuf 34.1 toolchain. CUDA 13.3
remains the host default for target workloads; it is not replaced system-wide.

This exists so the measurement stack can collect dynamic SASS, address, and
phase evidence — the simulator's input-feature side. Under the current charter
(root CLAUDE.md, re-founded 2026-07-02) NCU is the first-class ground-truth
instrument, collected in separate matched runs under pinned protocol v2.1
(`.claude/knowledge/measurement/pinned-protocol-v21.md`); the tracer and NCU
are never co-attached to one process.

The tracer is built and exercised on both local arches per the CLAUDE.md
"Two-Device Local Development Model": RTX 3090 Ti (sm_86, Ampere desktop,
GA102) and RTX 5060 laptop (sm_120, Blackwell consumer, GB206) are valid
live tracer targets. Trace contract evidence captured on one arch does not
carry to the other without an explicit live witness on the second arch (per
the cross-arch decoupling discipline). The tracer's wire format and
arch-aware fallback paths (notably the SM120 binary-version workaround in
"Known notes") are designed for cross-arch portability of producer-side
behavior; consumer-side tests that pin SASS layouts or specific PCs remain
arch-coupled and must be updated on session pivot.

## Layout

```
tools/nvbit_tracer/
├── Makefile                    # top-level entry (descends into tracer_tool/)
├── nvbit_release/               # local NVBit 1.8 drop used directly by the tracer
├── isa_def/                    # minimized opcode/type enums
│   ├── trace_opcode.h
│   ├── operation_type.h
│   └── {ampere,blackwell,hopper,kepler,pascal,turing,volta}_opcode.h
├── tracer_tool/                # NVBit tool source (tracer_tool.so + trace_printer)
│   ├── Makefile
│   ├── tracer_tool.cu
│   ├── inject_funcs.cu
│   ├── common.h
│   └── trace_printer.cc
└── traces_enhanced/            # enhanced trace protobuf/serialization library
    ├── Makefile
    ├── dynamic_trace/*.proto   # protobuf schemas (source of truth)
    ├── src/
    │   ├── rapidjson/          # vendored (document.h:326 patched for g++ 15+)
    │   ├── JSONBase.{cc,h}, JSONIncludes.h
    │   ├── control_bits.{cc,h}
    │   ├── register_usage.{cc,h}
    │   ├── string_utilities.{cc,h}
    │   ├── traced_constants.h
    │   ├── traced_execution.{cc,h}
    │   ├── traced_instruction.{cc,h}
    │   ├── traced_kernel.{cc,h}
    │   └── traced_operand.{cc,h}
    └── pb_trace/               # tracked C++ authority; temp-copy regeneration only
```

**Schema consumers.** Consumers of these .proto files: (1) tracked C++ pb_trace authorities, whose regeneration acceptance runs only in a host-gated external temporary directory; (2) tracked Python authorities under `workloads/simbench_bw/_pb_generated`; (3) COMMITTED Rust prost bindings sim/trace/src/pb/dynamic_trace.rs. Task7/H3 forbids canonical-checkout regeneration; a future schema change requires a separately authorized replacement workflow across all consumers.

## Provenance

The sources were forked from an Accel-Sim / MICRO 2025 trace-tool tree
(`modern-gpu-simulator-micro-2025/simulator-remodeled/`, removed from this
repo before this measurement-focused cleanup). For provenance only, the mapping
from this tree back to the upstream layout was:

| This tree | Upstream location |
|---|---|
| `tracer_tool/` | `util/tracer_nvbit/tracer_tool/` |
| `traces_enhanced/` | `util/traces_enhanced/` |
| `isa_def/{ampere,blackwell,...}_opcode.h` | upstream opcode enum headers |
| `isa_def/trace_opcode.h` | upstream trace opcode header |
| `isa_def/operation_type.h` | upstream operation-type enum header |

## Patches applied to make the host build work

1. **RapidJSON** (`traces_enhanced/src/rapidjson/document.h:326`) — the upstream
   vendored copy has a broken `GenericStringRef::operator=` that assigns through
   const members and never returns. g++ 15.2.1 rejects it. Replaced with
   `= delete` (class is non-assignable by design; matches upstream RapidJSON
   master).

2. **Dead include of upstream trace_parser.h** (`traces_enhanced/src/traced_instruction.cc:32`) —
   removed. Verified: no symbol from `trace_parser.h` is used anywhere in
   `traces_enhanced/src/`. The include was historical dead code that transitively
   dragged in a large unrelated dependency tree (`abstract_hardware_model.h`, `constants.h`,
   and friends).

3. **ISA_Def include paths** — rewritten from
   old upstream-relative enum paths
   to `../../isa_def/*.h` (from `traces_enhanced/src/`) and `../isa_def/*.h` (from
   `tracer_tool/`), and `operation_type.h` (from `isa_def/`, same dir).

4. **`bc` replaced with `awk`** (`tracer_tool/Makefile`) — version-comparison
   checks now use awk instead of bc, since awk is POSIX-required and bc is
   frequently absent on minimal Linux installs (including this host).

5. **Makefile relative paths** — `TRACES_ENHANCED_DIR` and
   `TRACES_ENHANCED_PROTOCOL_BUFFER_DIR` in `tracer_tool/Makefile` changed from
   `../../traces_enhanced/` to `../traces_enhanced/` to match the new tree depth.

6. **Pinned protobuf paths** (`traces_enhanced/Makefile`,
   `tracer_tool/Makefile`) — compile and link flags come from relocated
   `pkg-config` metadata under `.toolchain/pb`. Private include variables keep
   recursive make from clobbering the generated-header search path.

## Building

Provision the exact C++ protobuf toolchain, if absent, through its explicit
lock:

```bash
bash tools/nvbit_tracer/pinned_build_env.sh --ensure
```

Ordinary sourcing is verify/export-only: an absent prefix returns 2 and a
present mismatch returns 1 without repair. While Task7/H3 remains open, use the
canonical worktree only for the read-only pin check:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer check-pinned-env
```

Canonical Git worktrees refuse every build, clean, protobuf regeneration,
replacement, and repin target. Build feasibility may be tested only in a
non-Git temporary copy with `NVBIT_TEMPCOPY_BUILD_OK=1`; the sentinel is invalid
inside any Git worktree and cannot authorize changes to preserved bytes. The
sourced environment validates and
exports exact nvcc 13.2, g++-14, protoc/libprotobuf 34.1, CUDA include/link
paths, relocated pkg-config metadata, and the protobuf runtime loader path. An
unsourced non-Git temporary-copy build fails early with an actionable pin
error.

Do **not** pass `INCLUDES=...` or `PROTOBUF_INCLUDES=...` on the command line.
GNU make propagates command-line variables into `traces_enhanced`; the former
broken recipe replaced its private `-I./pb_trace/include` path and failed on
`address.pb.h`.

The Python readers used by fail-closed trace admission are tracked authorities
under `workloads/simbench_bw/_pb_generated`. Missing files are repository
corruption: restore their byte-exact committed content with `git restore
--source=HEAD --worktree -- workloads/simbench_bw/_pb_generated`. A missing
Python protobuf runtime is a separate prerequisite failure resolved by the
exact `requirements.txt` pin; consumers never generate bindings in place.

### Operator-only external build

Agents do not run this build. After committing every tracked and nonignored
change, an operator first materializes the exact protobuf lock directly into an
external, de-aliased prefix. `--ensure-external` is no-repair/no-replace and
requires an absent destination outside the repository. The external builder
also rejects `/usr/bin/g++-14` on hosts where it is a hard-linked alias. Copy
the driver and its matching GCC helper tree into one fresh external prefix so
the retained driver is single-link and can still resolve `cc1plus`:

```bash
mkdir -p /tmp/nvbit-protobuf /tmp/nvbit-tracer-candidates /tmp/nvbit-tracer-bundles
NVBIT_PB_PREFIX=/tmp/nvbit-protobuf/pb-34.1 \
  bash tools/nvbit_tracer/pinned_build_env.sh --ensure-external

HOST_CXX_ROOT=/tmp/nvbit-gcc14-UNIQUE
GCC14_LIBEXEC_DIR="$(dirname "$(/usr/bin/g++-14 -print-prog-name=cc1plus)")"
GCC14_LIBEXEC_RELATIVE="${GCC14_LIBEXEC_DIR#/usr/lib/gcc/}"
mkdir -p "$HOST_CXX_ROOT/bin" \
  "$HOST_CXX_ROOT/lib/gcc/$(dirname "$GCC14_LIBEXEC_RELATIVE")"
install -m 0755 /usr/bin/g++-14 "$HOST_CXX_ROOT/bin/g++-14"
cp -a "$GCC14_LIBEXEC_DIR" \
  "$HOST_CXX_ROOT/lib/gcc/$(dirname "$GCC14_LIBEXEC_RELATIVE")/"
test "$(stat -c %h "$HOST_CXX_ROOT/bin/g++-14")" -eq 1
```

The tracer itself uses a candidate/review/admit flow because `-g3` embeds the
fresh `/tmp` source root and therefore an exact output digest cannot be known
before the initial build. Phase 1 is a successful, no-replace candidate build;
it is not capture authority:

```bash
.venv/bin/python -m tools.nvbit_tracer.external_tracer_build build-candidate \
  --repo-root "$(git rev-parse --show-toplevel)" \
  --build-root /tmp/nvbit-tracer-build-UNIQUE \
  --bundle-dir /tmp/nvbit-tracer-candidates/sm86-UNIQUE \
  --git /usr/bin/git --bash /usr/bin/bash --make /usr/bin/make \
  --nvcc /usr/local/cuda-13.2/bin/nvcc \
  --ptxas /usr/local/cuda-13.2/bin/ptxas \
  --cuobjdump /usr/local/cuda-13.2/bin/cuobjdump \
  --host-cxx "$HOST_CXX_ROOT/bin/g++-14" \
  --protoc /tmp/nvbit-protobuf/pb-34.1/bin/protoc \
  --pkg-config /usr/bin/pkgconf \
  --protobuf-prefix /tmp/nvbit-protobuf/pb-34.1
```

Phase 2 is an independent review of that successful candidate. The reviewer
retains the exact `tracer_tool.so` SHA-256 outside the candidate; it must not be
copied from mutable `artifact_identity.json`, and a hash recovered from a failed
build is never admissible. Phase 3 supplies that reviewed digest to publish a
separate final bundle while preserving the candidate:

```bash
EXPECTED_TRACER_SHA256=64_LOWERCASE_HEX_RETAINED_BY_INDEPENDENT_REVIEW
.venv/bin/python -m tools.nvbit_tracer.external_tracer_build admit \
  --repo-root "$(git rev-parse --show-toplevel)" \
  --candidate-dir /tmp/nvbit-tracer-candidates/sm86-UNIQUE \
  --bundle-dir /tmp/nvbit-tracer-bundles/sm86-UNIQUE \
  --cuobjdump /usr/local/cuda-13.2/bin/cuobjdump \
  --expected-tracer-sha256 "$EXPECTED_TRACER_SHA256"

.venv/bin/python -m tools.nvbit_tracer.external_tracer_build verify \
  --repo-root "$(git rev-parse --show-toplevel)" \
  --bundle-dir /tmp/nvbit-tracer-bundles/sm86-UNIQUE \
  --cuobjdump /usr/local/cuda-13.2/bin/cuobjdump \
  --expected-tracer-sha256 "$EXPECTED_TRACER_SHA256"
```

Every destination parent must already be an external non-Git directory, and
candidate, build, handoff, and final names must be unused. Every supplied tool
and the complete protobuf handoff must be absolute, Make-safe, outside every
repository/worktree, symlink-free, and composed of single-link regular files.
The candidate and final bundle each contain four files: sealed source, build,
and artifact JSON plus `tracer_tool.so`. Only the final bundle has
`independent_reviewed_candidate_sha256.v1` authority; `verify` rejects candidate
receipts. The producer binds the exact wildcard/find source sets into argv,
executes every named tool through retained descriptor aliases, and supplies the
CUDA include directory derived from the retained nvcc path explicitly because a
descriptor alias cannot preserve nvcc's toolkit-root inference. It contains and
reaps the complete descendant process tree across timeout or cancellation,
continuously monitors extracted source and protobuf input generations, and
requires a target-free ELF64/EM_X86_64 shared object with only `sm_86` CUDA
cubins. Linux subreaper/procfs containment is mandatory; the producer never
falls back to an uncontained child. One absolute timeout covers authority
capture, hashing, probes, launch, build, inspection, publication, cleanup, and
final verification. Exit 1 denotes process/deadline failure; exit 2 denotes
source, tool, schema, identity, or publication authority failure.

### Host compiler selection (`NVCC_CCBIN`)

`pinned_build_env.sh` sets `NVCC_CCBIN` and `CXX` to `/usr/bin/g++-14`,
overriding this host's hazardous ambient g++-15 setting. To use another path to
a verified g++-14 installation, select it before sourcing:

```bash
NVBIT_TRACER_HOST_CXX=/path/to/g++-14 \
  source tools/nvbit_tracer/pinned_build_env.sh
```

This is **required on hosts where the default `g++` ships C++23-only
libstdc++ headers that the bundled CUDA `nvcc` cannot parse**. Concrete
trigger observed in R51-R52 of this repo: Arch Linux upgraded `g++` to
`16.1.1`; CUDA 13.2's `cudafe` frontend fails on `<functional>`,
`<memory>`, and `type_traits` headers that use C++23's explicit object
parameter syntax (`operator()(this _Self&& __self, ...)`). `--allow-
unsupported-compiler` does NOT help (the headers are unparseable, not
just from an unsupported compiler version), and using `clang++` as
`-ccbin` keeps the same libstdc++ headers.

The fix is to install a CUDA-compatible host compiler whose libstdc++
predates the C++23 lift. On Arch:

```bash
sudo pacman -S --noconfirm gcc14
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer check-pinned-env
```

`gcc14 14.3.1` ships its own libstdc++ at
`/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1/include/c++/14.3.1/`. Other
distros generally have an `apt install g++-14` or `dnf install gcc-14`
equivalent. Verify with `which g++-14 && g++-14 --version` before
invoking the build.

The repository build contract remains g++-14 even if another host compiler
happens to parse the sources. Change the pin only after a separate build and
live-trace qualification.

Build artifacts:

- `tracer_tool/tracer_tool.so` — the NVBit tool (LD_PRELOAD target)
- `tracer_tool/trace_printer` — a CLI that can read back `dynamic_trace.pb`
- `traces_enhanced/obj/*.o` — intermediate C++ objects
- `traces_enhanced/pb_trace/{include,src,obj}/*` — protoc-generated bindings

## Phase-injection runtime knobs

When the non-root measurement NVBit phase-injection path is enabled
(`SIMBENCH_NVBIT_PHASE_INJECT=1`), the per-launch flush sentinel and
the per-context teardown handshake are bounded by:

- `SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS` (default `10000000`) — iteration
  cap for both the post-launch sentinel spin and the per-context
  teardown spin. The runtime accepts only strictly positive integers;
  empty, non-numeric, negative, and zero values warn and fall back to
  the default so the bounded-spin guarantee always holds against a
  stalled recv thread. On post-launch exhaustion the tracer resets
  `skip_callback_flag`, snapshots whatever phase records the recv thread has
  already drained, and exits fatally before any canonical publication. If the
  snapshot is nonempty, its exact bytes are retained only as
  `.partial-phase-records_<repeat>_<launch>.bin`; that explicitly noncanonical
  name cannot match phase-ledger discovery and cannot authorize evidence. A
  failure while retaining those partial bytes is appended to the timeout
  diagnostic without replacing timeout as the primary fatal reason. On teardown
  exhaustion the tracer skips `phase_channel_host.destroy(false)`,
  `cudaFree(phase_channel_dev)`, and `delete ctx_state`, instead
  emitting a `WARNING:` line that reports `timed_join_attempted=false`
  and explains the intentional leak — keeping `CTXstate` alive avoids
  a use-after-free against the still-live phase recv thread, which
  closes over `ctx_state` and dereferences it on every iteration of
  its WORKING loop. Process teardown reclaims the leaked allocations
  as the OS tears the address space down.

The per-launch sentinel handshake is launch-indexed and uses a
generation-valued completed counter: the host stores the current
`launch_index` in the per-context `phase_expected_flush_launch_index`
atomic (for diagnostics) before pushing the flush sentinel, and waits
for `phase_completed_flush_launch_index == this_launch_id` after the
flush. The recv thread always stores the sentinel's `launch_index`
into `phase_completed_flush_launch_index` on every sentinel it
processes. The generation-valued handshake prevents any stale sentinel from
satisfying a different launch's wait under arbitrary recv-thread scheduling.
The recv thread also emits a stale-sentinel `WARNING:` when the sentinel's
`launch_index` does not match the current expected value; this is diagnostic
only and not load-bearing for safety. Record snapshots are likewise partitioned
by immutable `launch_index`, and earlier-generation records are quarantined as
defense in depth. Because a flush timeout is now fatal, no timed-out launch can
carry partial records forward into a later successful canonical capture.

Those canonical filenames are intentionally not device-qualified, so phase
injection has a strict one-GPU producer authority per process. It also permits
only one in-flight phase-enabled launch: `BeginLaunch` claims the immutable
`(device_id, launch_index)` before the process-global managed launch scalar is
updated, and a same-device overlap fails closed rather than overwriting that
identity. Launch exit and publication use the thread-local accepted launch ID,
never a later read of the mutable managed scalar.

During `nvbit_at_init` the tracer resolves `TRACES_FOLDER` and privately
duplicates an exact inherited `/proc/self/fd/<n>` directory authority (or opens
an ordinary path with `O_DIRECTORY|O_NOFOLLOW`), snapshots its device/inode
identity, and rejects a second producing CUDA device before its independent
launch counter can collide. `PhaseRecordWriter` is bound to its constructing
PID; every public operation checks PID ownership before acquiring the mutex, and
an inherited post-fork destructor also avoids that mutex, preventing child-side
deadlock or duplicated authority/counter use. Each `CTXstate` stores a device ID
only after `cudaGetDeviceCount` and `cuCtxGetDevice` succeed and both reported
bounds are validated. Callback dispatch never queries the current context
unconditionally: branches whose CUDA API contract requires one (notably memcpy
and kernel launch) perform a fresh status/bounds check and compare it with the
stored identity, while explicit-context and post-pop-safe paths use only the
stored bounds-checked ID. Receiver paths likewise validate stored identity
before any `kernel_id`, `current_stream_id`, or `stop_report` access.

Every nonempty complete phase snapshot is created relative to a revalidated
duplicate of the retained directory with `O_CREAT|O_EXCL|O_NOFOLLOW`;
preexisting regular files, symlinks, hardlinks, FIFOs, and other nodes are never
overwritten or followed. The publisher requires a single-link regular inode,
exact bounded writes, file and directory `fsync`, and checked closes. Any
authority, open, write, identity, or durability failure is process-fatal and
leaves uncertain partial evidence in place rather than unlinking it.

Two test-only deterministic stall hooks are compiled in but disabled
unless their env vars are set, used by the synthetic-stall regression
tests for the post-launch and teardown bounded-spin paths:

- `SIMBENCH_PHASE_TEST_STALL_LAUNCH=N` and
  `SIMBENCH_PHASE_TEST_STALL_ITERS=K` — when the recv thread sees the
  sentinel for launch `N`, it `sched_yield()`s `K` times before
  acknowledging. Combined with a small
  `SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS`, this exercises the post-launch
  timeout branch deterministically.
- `SIMBENCH_PHASE_TEST_STALL_TEARDOWN_ITERS=K` — when the recv thread
  exits its WORKING loop, it `sched_yield()`s `K` times before setting
  `FINISHED`. This exercises the teardown timeout branch.

Both hooks default off and are no-ops in production runs. The separate host-only
`phase_record_writer_test` injects a post-create hardlink, leaf and directory
`fsync` failures, checked-close failures (including cleanup precedence), a fake
post-fork PID, same-device launch overlap, and a fatal callback that throws in
process. It therefore exercises the live publisher and fatal wrapper without
launching CUDA, NVBit, a GPU, or a subprocess.

## Using the tracer

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PB_LIB="$REPO_ROOT/.toolchain/pb/lib"
TRACER="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so"
TRACE_ROOT="$REPO_ROOT/results/nvbit_capture_staging/manual_trace"
mkdir -p "$TRACE_ROOT"

LD_LIBRARY_PATH="$PB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$TRACER" \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$TRACE_ROOT" \
./your_cuda_app [args...]
```

`CUDA_INJECTION64_PATH` is preferred because frameworks may replace
`LD_PRELOAD`; use one injection mechanism, not both. The pinned protobuf path
is load-bearing: without it the loader may silently run the application
untraced. `USER_DEFINED_FOLDERS=1` is required for `TRACES_FOLDER` to be
honored.

Output:

- `results/nvbit_capture_staging/manual_trace/dynamic_trace.pb` — the protobuf-encoded dynamic trace
- `results/nvbit_capture_staging/manual_trace/extra_info/enhanced_execution_info.json` — static instruction metadata
- `results/nvbit_capture_staging/manual_trace/threadblocks/` — per-kernel threadblock data

## Inspecting a trace

```bash
.venv/bin/python tools/nvbit_tracer/trace_admission.py \
  results/nvbit_capture_staging/manual_trace
./tools/nvbit_tracer/tracer_tool/trace_printer \
  results/nvbit_capture_staging/manual_trace
```

The first command is the publication gate: file presence alone is insufficient.
It requires a root launch, instruction-bearing threadblocks, a valid contract,
and positive producer counters. `trace_printer` takes the containing directory
(not the `.pb` file) and opens an interactive menu over the admitted trace.

## Smoke test

A self-contained CUDA benchmark at `tools/nvbit_tracer/smoketest/backprop_enriched.cu`
acts as the host-side tracer validation target. It exercises 5 distinct
kernels (tiled matmul, fused ReLU+softmax, backward delta with atomicAdd,
gradient reduction, SGD+momentum weight update) across 12 kernel launches
per run, producing roughly 118k traced instructions and 108 per-CTA
threadblock files. The "backprop" name refers to the backpropagation
training algorithm exercised by the kernel; this binary has no external
source dependencies.

```bash
# Use the preserved smoke binary and tracer; do not build them in this checkout.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd tools/nvbit_tracer/smoketest
PB_LIB="$REPO_ROOT/.toolchain/pb/lib"
OUT="$REPO_ROOT/results/nvbit_capture_staging/manual_backprop_smoke"
mkdir -p "$OUT"
LD_LIBRARY_PATH="$PB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH=../tracer_tool/tracer_tool.so \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$OUT" \
./backprop_enriched
```

The host-side sanity check inside `backprop_enriched` verifies both the
forward output (all softmax rows finite and sum to 1.0 within 1e-3) and
that the optimizer step mutated the W2 weights meaningfully
(`||ΔW2||_1 > 0` with all entries still finite). If either fails, the
binary returns non-zero.

## Concurrency model

The tracer has two shared-state primitives:

1. **`pthread_mutex_t mutex`** (recursive) — serializes main-thread callback
   entries and all main-thread access to NVBit-callback-only state.

2. **`std::mutex recv_container_mutex`** — serializes multi-threaded access
   to the process-global trace containers: `dyn_trace` (protobuf),
   `threadblocks` (unordered_map), `remaining_injects_to_current_instruction`
   (unordered_map), and `pc_to_opcode` (map). Both main-thread callback
   handlers AND every recv thread acquire this mutex around their writes.

**Lock order** is always `mutex > recv_container_mutex`. The recv thread
only ever takes `recv_container_mutex` (it does not take `mutex` after its
one-shot ctx_state lookup at the top of `recv_thread_fun`), so the order is
consistent and non-cyclic.

**Handshake state** inside `CTXstate` is `std::atomic<RecvThreadState>` and
`std::atomic<bool>` with explicit release-store / acquire-load ordering —
the canonical C++ producer-consumer pattern, matching the one documented at
`NVIDIA_Manuals/cuda_guide/03-02-04-01-01_thread-scope-and-memory-ordering.md`
section 3.2.4.1.1. No `volatile`, no unsynchronized maps.

### Why the NVBit channel does NOT give us CPU→CPU ordering for free

The NVBit channel primitives at `nvbit_release/core/utils/channel.hpp`
provide **only GPU→host synchronization**, not CPU→CPU:

- `ChannelDev::flush()` (device side) issues `__threadfence_system()` before
  signaling the host via `*doorbell = nbytes`. This establishes happens-before
  between the GPU kernel's device-side writes into the channel buffer and
  the host's subsequent `recv()`.
- `ChannelHost::recv()` (host side) reads `*doorbell` (declared `volatile int*`),
  copies data out of pinned host memory, then clears the doorbell. No memory
  fences or atomics participate in host-side CPU→CPU ordering.

Concretely: the main thread's pre-launch writes to `kernel_id`, `pc_to_opcode`,
and `dyn_trace` happen on the **CPU** before the GPU kernel even starts. The
channel's synchronization does **not** carry those writes into the recv
thread's view. The only happens-before edge is the atomic
`recv_thread_receiving.store(release)` / `.load(acquire)` pair in
`tracer_tool.cu` — which is exactly why that atomic flag exists and why the
explicit `.load(memory_order_acquire)` calls on `kernel_id[dev]` and
`current_stream_id[dev]` immediately after `ch_host->recv()` returns are
**load-bearing, not redundant**. A future maintainer who thinks "the channel
already synchronized things" and deletes those acquire-loads will reintroduce
the race Codex originally flagged.

This invariant is not documented in the 2019 NVBit paper
(`papers/knowledge_base/active/2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus.pdf`)
— I had to read `channel.hpp` source to verify it.

### Why the custom STOP/FINISHED handshake exists

`ChannelHost::destroy()` already knows how to `pthread_join()` the recv
thread (see `channel.hpp:215-230`), so in principle the tracer could just
call `destroy()` at teardown and let the channel's built-in mechanism wait.
The tracer doesn't do that. Instead:

```cpp
// 1. Serialize dyn_trace to dynamic_trace.pb (under recv_container_mutex)
// 2. Run enhanced_tracer() (reads ELF/cubin, writes sidecar JSON)
// 3. ctx_state->recv_thread_done.store(STOP, release);
// 4. while (.load(acquire) != FINISHED) sched_yield();
// 5. ctx_state->channel_host.destroy(false);   // pthread_join on already-exited thread
```

Steps 1–2 must happen **while the recv thread is still observable and
cooperating**, because the tracer needs a consistent snapshot of dyn_trace
before any further writes. Step 3 is the explicit shutdown request; step 4
is the explicit acknowledgment; step 5 is the channel teardown. The
double-handshake exists so that the tracer can finalize its own state in a
controlled order relative to the recv thread's last write, which `destroy()`
alone would not allow.

### Known limitation

`nvbit_at_ctx_term` still holds the recursive `mutex` during its spin-wait
for the recv thread to signal FINISHED. This does not deadlock (the recv
thread does not need `mutex` on the path from its WORKING→STOP observation
to its FINISHED publication), but it does serialize other main-thread
callbacks behind the current context's teardown. For single-process
multi-threaded CUDA applications with several simultaneously-active
contexts, this manifests as a brief stall during the last context's
teardown.

Fixing this requires restructuring the mutex discipline around teardown
(releasing `mutex` before the spin, re-acquiring after) and has not been
done because:
(a) the tracer's primary use case is single-context CUDA validation kernels,
(b) the brief stall has no observable correctness implication,
(c) the fix has non-trivial interaction with `skip_callback_flag` and
    `pending_devices_to_finish` state and deserves its own focused change.

## Known notes

- **CUDA/NVBit compatibility scope** — NVBit 1.8's release notes say its
  bundled CUDA headers were updated to 13.2; that is an API snapshot, not a
  published maximum target-compiler version. Upstream lists CUDA `>= 12.0` and
  says precompiled targets are compiler-version independent. This repository
  still builds its custom tools with nvcc 13.2 because nvcc 13.3 cannot parse
  the pinned protobuf/Abseil translation units. CUDA 13.3 may build the target
  workload. NVBit's published `driver <= 575.xx` limit conflicts with CUDA
  13.x's driver matrix and has not been formally corrected by a maintainer.
  Driver 610.43.02 is therefore locally smoke-tested outside that published
  envelope, not officially certified. New CUDA 13.3 API, PTX-JIT, Tile/TMA, or
  SASS paths require their own representative contract-valid smoke.

- **Binary-version fallback** — `tracer_tool.cu` handles
  `CU_FUNC_ATTRIBUTE_BINARY_VERSION` returning `CUDA_ERROR_INVALID_VALUE` by
  deriving `binary_version` from
  `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}`. The fallback is not
  sm_120-only: the qualified CUDA 13.3-target smoke on the RTX 3090 Ti / sm_86
  host also returned `CUresult=400` and exercised it, producing
  `binary_version=86`. It emits one `INFO:` line per run. The `major*10 + minor`
  result is correct for the repository's native-architecture or PTX-JIT smoke
  targets. It is not a general proof of the original cubin target when an
  older-minor cubin runs through same-major binary compatibility. For sm_120,
  the current native-target result is also consistent with the fact that cubins
  are not binary-compatible across compute-capability major versions (see
  `NVIDIA_Manuals/cuda_guide/01-03-04-01_binary-compatibility.md` and
  `01-03-04-03_just-in-time-compilation.md`) — a cubin running on sm_120
  in this qualified path is native sm_120 or JIT-compiled to sm_120.

- **Architecture-conditional variants (`sm_120a`, `sm_100a`)** are outside
  scope of the current smoke test. The smoketest kernel is built with plain
  `-arch=sm_120`, not `sm_120a`. If a future measurement workload
  uses Blackwell-specific features (TMA, fifth-gen tensor cores, wgmma,
  etc.), the tracer may need additional binary-version handling — the
  `_a`-suffixed variants are "not forward or backward compatible" per
  `NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md`.

- **NVBit conflicts with other CUDA driver-injection/profiling tools in the
  same process**. Do not run the tracer alongside `ncu`, `nvprof`, or `nsys`
  in the same process — that constraint is about CUDA driver injection
  collisions, not about whether NCU is permitted at all in this repository.
  Nsight Systems can be used only as a separate matched run.

  Repository-level NCU policy (CLAUDE.md Rule 1, re-founded 2026-07-02):
  NCU is ALLOWED and is the first-class ground-truth instrument on all
  targets, captured under pinned protocol v2.1
  (`.claude/knowledge/measurement/pinned-protocol-v21.md`). The one surviving
  prohibition is targets-never-features: NCU metric values are prediction
  targets / validation labels for the simulator, never simulator inputs.
  The tracer itself never invokes NCU; the same-process injection collision
  rule above is a tracer-process-specific concern (collect NCU in a separate
  matched run), not a repository-wide NCU ban.

- **Host toolchain constraints**: requires relocated `pkg-config` metadata for
  exact protobuf 34.1 to resolve cleanly (the environment and Makefiles fail
  closed if it does not), exact nvcc 13.2, and g++ 14.x (specifically
  `/usr/bin/g++-14`) per the gcc-14
  install recipes documented earlier in the "Host compiler selection
  (`NVCC_CCBIN`)" section, because Arch's default `g++ 15.x` / `g++ 16.x`
  ship C++23-only libstdc++ headers that CUDA 13.2's `cudafe` frontend
  cannot parse. `bc` is NOT required; version checks use `awk` instead.
