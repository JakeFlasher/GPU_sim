# NVBit cross-architecture runbook

Use this when a tracer or trace-consumer change should be exercised on both the
Ampere `sm_86` and Blackwell `sm_120` development machines.

This is an experiment checklist, not an approval gate. A result on one GPU is
still useful while the other GPU is unavailable.

## On each host

Record the ordinary environment facts:

```bash
git rev-parse --short HEAD
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
protoc --version
```

Build the current checkout directly:

```bash
# Optional when the repository-local CUDA/protobuf environment is useful.
source tools/nvbit_tracer/pinned_build_env.sh

make -C tools/nvbit_tracer -j"$(nproc)"
```

Run a small CUDA workload with the tracer:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT="$REPO_ROOT/results/nvbit-cross-arch/$(hostname)"
mkdir -p "$OUT"

CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$OUT" \
  /absolute/path/to/workload [arguments]
```

Inspect the result:

```bash
tools/nvbit_tracer/tracer_tool/trace_printer "$OUT"
```

## Compare the two hosts

Check the facts affected by the change:

- the producer starts and exits cleanly;
- the trace contains the expected launches and nonempty thread-block payloads;
- Python and Rust decoders accept the same wire fields;
- architecture-specific opcodes are classified for the correct GPU; and
- counts or timing observations being compared use the same workload arguments.

Keep architecture-specific differences explicit. Do not copy `sm_86`
assumptions into `sm_120` merely to make results match.

If one host fails, preserve the command and error, fix the local cause when it
is determinable, and continue any independent work. No frozen revision,
reviewer signoff, digest handoff, or clean-worktree ceremony is required.
