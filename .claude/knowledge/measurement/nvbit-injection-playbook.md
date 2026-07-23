---
Use when: building or modifying the NVBit tracer; debugging NVBit injection failure; planning SASS instrumentation or `%clock64` phase probes in a non-root GPU environment.
Primary raw source: `tools/nvbit_tracer/README.md`.
Supporting sources:
- NVlabs/NVBit GitHub: https://github.com/NVlabs/NVBit
- NVBit MICRO 2019 paper: `references/papers/knowledge_base/active/2019__nvbit_a_dynamic_binary_instrumentation_framework_for_nvidia_gpus.pdf`
- Vendored NVBit 1.8 drop: `tools/nvbit_tracer/nvbit_release/`
Repo touchpoints:
- `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`
- `tools/nvbit_tracer/tracer_tool/inject_funcs.cu`
- `tools/cloud/preflight.sh`
- `workloads/simbench_bw/`
Out of scope: co-attaching NCU/nvprof/CUPTI-injection tools to the same process as NVBit (NCU itself is a first-class ground-truth instrument in this project — see `measurement/pinned-protocol-v21.md`).
Version family: NVBit 1.8, CUDA 13.x, sm_120 primary target.
---

# NVBit Injection Playbook

NVBit is a user-space dynamic binary instrumentation framework for NVIDIA GPUs.
It is allowed by this project because it does not require host-driver root or
GPU hardware-counter access.

## Build

```bash
bash tools/nvbit_tracer/pinned_build_env.sh --ensure
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer check-pinned-env
```

The sourced environment validates exact nvcc 13.2, g++-14, protoc/protobuf
34.1, and the checkout-local protobuf prefix. CUDA 13.3 remains the default
toolchain for target applications. While Task7/H3 remains open, the canonical
worktree may only verify these pins; its build, clean, regeneration,
replacement, and repin targets refuse. Build verification is allowed only from
a non-Git temporary copy with `NVBIT_TEMPCOPY_BUILD_OK=1`, with the preserved
artifact hashes unchanged. Do not use the sentinel inside any Git worktree.

Python `_pb2.py` readers are tracked authorities under
`workloads/simbench_bw/_pb_generated`. Restore missing bytes from Git exactly;
never generate replacements in the canonical checkout. Treat a missing Python
protobuf runtime as a distinct prerequisite failure.

The eight SOL injector call sites remain policy-quarantined until Stage 7.
These rules are binding at the guarded entry points; they are not a claim that
every repository injector is universally wired to the new runtime helpers.

## Invocation

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PB_LIB="$REPO_ROOT/.toolchain/pb/lib"
TRACE_ROOT="$REPO_ROOT/results/nvbit_capture_staging/manual_trace"
mkdir -p "$TRACE_ROOT"
LD_LIBRARY_PATH="$PB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$TRACE_ROOT" \
./your_cuda_app
```

Use one injection mechanism. `CUDA_INJECTION64_PATH` is preferred because
frameworks may rewrite `LD_PRELOAD`. `USER_DEFINED_FOLDERS=1` is required for
the custom output directory to be honored.

## Trap: `CUDA_INJECTION64_PATH` fails SILENTLY on unresolved deps

`CUDA_INJECTION64_PATH` injection fails SILENTLY when the tool `.so` has
unresolved shared-library dependencies — the app runs normally, no error, no
trace. The pinned tracer links against the repo-local protobuf and needs
`LD_LIBRARY_PATH=<repo>/.toolchain/pb/lib` in the environment. Use
`tools.nvbit_tracer.runtime_env.tracer_runtime_env()` to build the injection
environment instead of hand-rolling it. Diagnostic: if the NVBit startup
banner is missing from stderr, injection failed — do not trust the run. Before
publishing, run
`.venv/bin/python tools/nvbit_tracer/trace_admission.py results/nvbit_capture_staging/manual_trace`;
it rejects present-but-vacuous roots, empty threadblock trees, invalid records,
and zero producer counters.

## Current Project Uses

- Dynamic SASS instruction traces.
- Static/dynamic instruction fingerprints.
- Optional `%clock64` reads at kernel or phase boundaries.
- A/B overhead experiments for injected versus non-injected runs.
- Trace features that support wall-clock and cache-behavior prediction without
  profiler counters.

## Tool-Combination Constraint

Do not attach NVBit and another CUDA driver-injection/profiling tool to the
same process. In particular:

- NCU (and nvprof) are allowed project-wide as ground-truth instruments, but
  must NEVER be co-run with NVBit in one process — collect NCU labels in a
  separate matched run under the pinned protocol
  (`measurement/pinned-protocol-v21.md`).
- Nsight Systems is allowed only as a separate run, not co-attached with NVBit.
- CUPTI Activity timing should be collected in a separate matched run and joined
  offline.

## sm_120 Notes

The tracer carries an sm_120 binary-version fallback in
`tools/nvbit_tracer/tracer_tool/tracer_tool.cu`. Do not remove it unless a
replacement has been validated on the actual Blackwell cloud target.

## Expected Overhead

NVBit can be expensive. Per-instruction tracing can slow kernels by orders of
magnitude. Prefer bounded instrumentation, phase-level probes, sampling, or
short microbench runs when building prediction features.

Always record the tracer configuration and compare against an uninjected run
when the measurement is used for a quantitative claim.

## Output

```text
results/nvbit_capture_staging/manual_trace/dynamic_trace.pb
results/nvbit_capture_staging/manual_trace/extra_info/enhanced_execution_info.json
results/nvbit_capture_staging/manual_trace/threadblocks/
```

Use `tools/nvbit_tracer/tracer_tool/trace_printer` on the containing directory
to inspect a trace.

## See Also

- `tools/nvbit_tracer/README.md`
- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/cupti-activity-api.md`
- `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md`
- `.claude/knowledge/sass/sass-tooling.md`

## Escalate to Raw When

- NVlabs releases a new NVBit version.
- sm_120, sm_120a, or future Blackwell targets show changed binary-version or
  instruction-decoding behavior.
- Cluster, barrier, tensor, or async-copy instructions fail to decode cleanly.
