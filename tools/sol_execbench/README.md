# SOL ExecBench

<div align="center" id="top">

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](/docs/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[HuggingFace Dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) | [Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench)
| [Technical Report](https://arxiv.org/abs/2603.19173) </div>

**Speed-Of-Light ExecBench** is a rigorous GPU kernel evaluation and benchmarking framework built to benchmark AI-generated kernel solutions written with the variety of DSLs that NVIDIA hardware supports.

Kernels are:
* Checked for various forms of reward hacking
* Tested against a reference solution for numerical correctness
* Timed under reproducible conditions

Leaderboard submissions are ranked based on [SOL-Score](/src/sol_execbench/sol_score.py): a metric that grades custom kernel performance based on the theoretical roofline of a NVIDIA B200 GPU (obtained analytically with [SOLAR](https://github.com/NVlabs/SOLAR)).  

Supported kernel languages: PyTorch, Triton, CUTLASS, cuDNN, CuTe DSL, cuTile, CUDA C++.

## Prerequisites

This fork runs `sol-execbench` as a native CLI on the host — the
upstream Docker workflow has been retired in this checkout. The host
must already provide:

- A working NVIDIA driver (580+ recommended) and a `nvidia-smi` that
  resolves the target GPU.
- A CUDA toolkit on `PATH` (`nvcc`, `cuobjdump`, `nvdisasm`, `ptxas`).
- `uv` (the package manager used by this repo's `pyproject.toml`).
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
  (`pip install huggingface-hub[cli]`) for benchmark-data download.

## Setup

### 1. Download benchmark data (one-time)

```bash
./scripts/download_data.sh
```

This downloads the [SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) and [FlashInfer Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) datasets into `data/`.

### 2. Install dependencies natively

```bash
uv sync --all-groups
```

This installs the `sol-execbench` package (and its dev / test groups)
into a project-local `.venv`. The `sol-execbench` CLI is on `PATH`
after activation (or via `uv run sol-execbench …`). There is no
container to build or enter.

## Evaluating a Solution

Use the `sol-execbench` CLI directly on the host:

```bash
# Evaluate using a problem directory (contains definition.json + workload.jsonl)
sol-execbench <problem_dir> --solution solution.json

# Or specify files explicitly
sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json
```

### Example

```bash
sol-execbench examples/cute_dsl/jamba_attn_proj \
  --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json
```

### CLI Options

| Flag | Description |
|---|---|
| `--compile-timeout` | Compilation timeout in seconds (default: 120) |
| `--timeout` | Evaluation timeout in seconds (default: 600) |
| `-o, --output` | Write JSONL traces to file |
| `--json` | Print traces as JSON to stdout |
| `--lock-clocks` | Lock GPU clocks for stable benchmarks |
| `--keep-staging` | Preserve staging directory after run |
| `-v, --verbose` | Show subprocess output |

### Persistent profiler capture mode

Reference-capture tooling can opt into a compile-once, one-workload execution
path without changing the normal application arguments. Set
`SOL_EXECBENCH_CAPTURE_REQUEST` to an absolute JSON request path while invoking
the same `sol-execbench` command used above. If the variable is absent, the
canonical judge path is unchanged.

A preparation request validates the selected workload before compilation and
publishes an immutable staging manifest with source, adapter, evaluation-driver,
runtime, and `benchmark_kernel.so` hashes:

```json
{
  "schema_version": 1,
  "action": "prepare",
  "stage_dir": "/absolute/capture/stage",
  "source_workload_index": 0,
  "environment_contract": {"NVCC_CCBIN": "g++-14"},
  "environment_contract_sha256": "d959a1a89d2f3aa3e7c351ba59788c016404bbc77e9da428123188ada775af74",
  "compile_timeout_s": 120
}
```

Label and trace invocations reuse that stage with separate selection and result
manifests:

```json
{
  "schema_version": 1,
  "action": "execute",
  "stage_dir": "/absolute/capture/stage",
  "source_workload_index": 0,
  "environment_contract": {"NVCC_CCBIN": "g++-14"},
  "environment_contract_sha256": "d959a1a89d2f3aa3e7c351ba59788c016404bbc77e9da428123188ada775af74",
  "capture_kind": "trace",
  "selection_manifest": "evidence/trace-selection.json",
  "result_manifest": "evidence/trace-result.json",
  "execution_timeout_s": 600
}
```

Execution rehashes every input, performs correctness and warmup outside the
profiler range, and uses CUDA events around one destination-passing candidate
call. It does not import or use the CUPTI timing path. Stage-owned artifact and
evidence paths are stored relative to the stage, so a completed chunk remains
verifiable after its parent directory is atomically renamed. The exact
environment contract and its digest are bound into the stage-definition hash;
reuse and execution refuse any drift. The stdlib-only Python API also exposes
`build_certified_env` and `environment_contract_sha256` alongside
`prepare_capture_stage`, `validate_capture_stage`,
`validate_capture_evidence`, and `execute_capture_stage` from
`sol_execbench.driver`.

## Running a Dataset

Use `scripts/run_dataset.py` to evaluate an entire dataset (or a single problem) in batch. By default it runs the definition's reference implementation as the solution unless `--solution-name` is specified. Saves to `./out/{subset}` by default.

```bash
# Run all problems in the benchmark.
# Auto builds solution.json from a single code file
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --solution-name solution.py

# Run specific categories with multiple solution code files
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --category L1 L2 --solution-name solution.json

# Run a single problem
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark/L1/my_problem

# Limit number of problems and workloads
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5 --max-workloads 3 -o ./results
```

Results (traces and a summary JSON) are written to `out/run_dataset/` by default (override with `-o`). Problems that already passed are skipped on subsequent runs unless `--rerun` is specified.

## Problem Format

A problem directory contains:

- **`definition.json`** — Kernel specification: function signature, tensor shapes, dtypes, reference implementation.
- **`workload.jsonl`** — One JSON object per line, each defining input shapes, values, and tolerance thresholds.

A solution is a separate JSON file referencing source files with the kernel implementation.

See the full schema docs:

- [Definition](docs/definition.md) — Kernel specification (function signature, tensor shapes, dtypes, reference code)
- [Workload](docs/workload.md) — Concrete input configurations and tolerance thresholds
- [Solution](docs/solution.md) — Source files and build specs for a kernel implementation
- [Trace](docs/trace.md) — Evaluation output (correctness and performance results)

## Citation

```bibtex
@misc{lin2026solexecbench,
      title={SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits}, 
      author={Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang, Zhifan Ye, Nestor Qin, Fengzhe Zhou, Yuan Zhang, Jingquan Wang, Sana Damani, Dheeraj Peri, Ouye Xie, Aditya Kane, Moshe Maor, Michael Behar, Triston Cao, Rishabh Mehta, Vartika Singh, Vikram Sharma Mailthody, Terry Chen, Zihao Ye, Hanfeng Chen, Tianqi Chen, Vinod Grover, Wei Chen, Wei Liu, Eric Chung, Luis Ceze, Roger Bringmann, Cyril Zeller, Michael Lightstone, Christos Kozyrakis, Humphrey Shi},
      year={2026},
      eprint={2603.19173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.19173}, 
}
```

## License

Apache-2.0. See [LICENSE](LICENSE). Contributions require DCO sign-off — see [CONTRIBUTING.md](CONTRIBUTING.md).
