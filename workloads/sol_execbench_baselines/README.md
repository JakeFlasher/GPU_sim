# baselines/

The 235 *.cu CUTLASS C++ / CUDA C++ **template solutions** hand-ported
from NVIDIA SOL-ExecBench's official Python references. These are
**profiling targets** for the counter-free analytics framework, not
optimized kernels. SOL Scores in the **0.3 – 0.5+** band are accepted
when these eventually go through the online judge. See
[`../../AGENTS.md`](../../AGENTS.md) for the binding project contract.

### Landing contract (correctness-only)

A *.cu under `baselines/` is **landed** when:

1. Local correctness PASSED on every workload under the native
   `sol-execbench` evaluator (no Docker; the harness invokes
   `sol-execbench <problem_dir> --solution <path> --json` directly).
2. No reward-hack flag from the evaluator.

`speedup_factor` against the PyTorch reference is recorded as
diagnostic data only — a *.cu with `speedup_factor < 1.0` still
lands. The legacy `deferred_slow` status is retired.
`tools/sol_baselines_harness/deferred.json` now only records
`deferred_correctness` (workload correctness failure) and
`deferred_reward_hack` (evaluator-state exploit; should be empty).

## Layout

```
baselines/
  README.md                           # this file
  L1/                                 # 94 single-op kernels
    problem_NNN_<slug>/               # full markdown stem as slug
      README.md                       # per-problem notes (template below)
      cutlass/                        # CUTLASS C++ variant (optional)
        solution.json
        main.cpp                      # PyTorch binding wrapper
        kernel.cu                     # CUDA kernel + CUTLASS templates
        kernel.h                      # device-side decls (only if split)
      cuda_cpp/                       # CUDA C++ variant (optional)
        solution.json
        kernel.cu                     # may be single-file or split
  L2/                                 # 82 multi-op fused kernels
  Quant/                              # 33 FP8 / NVFP4 block-scaled kernels
  FIB/                                # 26 FlashInfer inference primitives
```

The four category directories pre-exist as a stable convention. Per-problem
subdirectories and their language variants are created when an actual
baseline lands — no scaffolding stubs.

### Slug convention

The per-problem **slug** is the full markdown stem from
`../references/NVIDIA_Manuals/solbench_problems/` (e.g.
`problem_004_gemm_n128_k2048`, not `p004` or `gemm_n128_k2048`). Problem
numbers repeat across spec groups; the markdown filename is the stable
external key.

### Language variants

A problem may carry **either or both** language variants (`cutlass/` and
`cuda_cpp/`). Keep them side-by-side under the same problem directory so
the SOL Score / latency comparison is one `git diff` away. Each variant
owns its own `solution.json` and source files.

File-shape conventions (don't force one shape):
- CUTLASS C++ typically uses `main.cpp` (PyTorch binding) + `kernel.cu`
  + `kernel.h` — see `../sol-execbench/examples/cutlass/gemm/`.
- CUDA C++ may be single-file (just `kernel.cu`) or split — see
  `../sol-execbench/examples/cuda_cpp/rmsnorm/` and
  `../sol-execbench/examples/cuda_cpp/flux_rope/`.

## Per-problem README template

Drop this at `baselines/<category>/<slug>/README.md` when starting a new
problem:

```markdown
# <problem_slug>

| Field                       | Value                                                          |
|-----------------------------|----------------------------------------------------------------|
| Problem                     | <one-line description from the markdown spec>                  |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/<file>.md`        |
| Benchmark problem_dir       | `data/SOL-ExecBench/benchmark/<category>/<dir>`                |
| Variant                     | `cutlass` \| `cuda_cpp`                                        |
| Evidence                    | `Compile-checked-sm_120` \| `Ran-local-sm_120` \| `Profiled-local-sm_120` \| `Profiled-B200-sm_100a` \| `Scored-B200-sm_100a` \| `Not-B200-validated` |
| Local speedup vs reference  | <diagnostic only; `<1.0` is acceptable>                        |
| SOL Score (online judge)    | <S value once submitted; expected 0.3 – 0.5+ band>             |
| Counter-free artifacts      | <which of {nvbit, cupti, nsys, static, sanitizer} emitted cleanly> |

## Approach

<algorithmic approach: tiling, schedule, fusion strategy, precision path,
TMA / async-copy / cluster / WGMMA usage, cuBLAS/cuDNN handshake notes>

## Architecture-coupling caveats

<anything that may differ between sm_120 local and sm_100a B200>
```

## Starting a new baseline

Copy from the official examples rather than maintaining stubs in this
tree:

- **CUTLASS C++** → `../sol-execbench/examples/cutlass/gemm/`
- **CUDA C++** → `../sol-execbench/examples/cuda_cpp/rmsnorm/`,
  `../sol-execbench/examples/cuda_cpp/flux_rope/`

The `solution.json` lives with our implementation (under
`baselines/<category>/<slug>/<language>/`), NOT inside the downloaded
benchmark directory. CLI invocation:

```bash
sol-execbench sol-execbench/data/benchmark/<category>/<problem_dir> \
    --solution baselines/<category>/<slug>/<language>/solution.json \
    --json
```

The `sol-execbench` CLI is on `PATH` once `uv sync` has installed the
`sol-execbench` package from `sol-execbench/pyproject.toml`. There is no
Docker shim in the dispatch path — the harness, the manifest, and the
cache all key on `native_env_hash` (see
`tools/sol_baselines_harness/native_env.py`).

The `<problem_dir>` argument supplies `definition.json` + `workload.jsonl`;
our `solution.json` supplies the kernel sources. The solution's
`definition` field must match the benchmark definition name.

## Rules digest

See [`../../AGENTS.md`](../../AGENTS.md) for binding rules. Highlights:

- No NCU; no `nvidia-smi` clock-locking on any device (including local 5090).
- No privileged CUPTI; no host-driver work.
- CUDA C++ / CUTLASS C++ / CuTe C++ only — no Python DSLs, no Triton.
  Submission `spec.languages` ∈ `["cutlass"]` or `["cuda_cpp"]`.
- No evaluator-state exploits (no `data_ptr` caching, no globals).
- Default-stream discipline (use `at::cuda::getCurrentCUDAStream()`).
- Precision is contractual — match problem dtype, document FP8 / NVFP4
  scale layout.
- sm_120 (local 5090) vs sm_100a (B200) are different targets — local
  pass proves compile / smoke viability only. Use evidence labels.
