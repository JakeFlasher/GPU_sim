---
Use when: looking up the structure of the official NVIDIA SOL-ExecBench evaluator-framework — how a solution-json submission is staged, compiled, and benchmarked by the native `uv run` dispatch (Docker was removed during solbench's docker_removal_plan 2026-05-13) — or auditing the local checkout's evaluation conventions before submitting a CUTLASS / CUDA C++ baseline.
Primary raw source: sol-execbench/
Supporting sources: sol-execbench/README.md; sol-execbench/CLAUDE.md; sol-execbench/docs/definition.md; sol-execbench/docs/solution.md; sol-execbench/docs/workload.md; sol-execbench/docs/trace.md
Repo touchpoints: sol-execbench/scripts/download_data.sh; sol-execbench/scripts/run_dataset.py; sol-execbench/src/sol_execbench/cli/; sol-execbench/src/sol_execbench/core/; sol-execbench/src/sol_execbench/driver/; sol-execbench/src/sol_execbench/core/bench/timing.py; sol-execbench/examples/cutlass/; sol-execbench/examples/cuda_cpp/
Patents anchored here: None
Out of scope: This stub is a navigation index for the upstream codebase, not a re-authored evaluator manual. Do not edit `sol-execbench/` — it is an upstream NVIDIA clone tracked separately. Solution-format authority lives in `sol-execbench/docs/solution.md`; this card cites it, it does not re-state it. The Python / Triton / cuDNN / cuBLAS / cuTile / cute_dsl example trees under `sol-execbench/examples/` are OUT of scope for our submission surface per project CLAUDE.md Rule 5; only `cutlass/` and `cuda_cpp/` are relevant.
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: sol-execbench; evaluator-framework; eval-harness; native-uv-dispatch; solution-json-schema; cupti-timing-divergence; problem-packager; reward-hack-detection; destination-passing-style; b200
---

# SOL-ExecBench Framework (Citation Stub)

## Use when

Looking up where a piece of the evaluator-framework lives in the local checkout, which schema doc to cite for `definition.json` / `solution.json` / `workload.jsonl` structure, or how the native `uv run` dispatch stages a problem and runs the benchmark subprocess before producing the per-workload trace (Docker was removed during solbench's docker_removal_plan 2026-05-13).

## Pointer

The upstream codebase lives at `sol-execbench/` (cloned at the repo root). This card mirrors the upstream README's table-of-contents and the upstream `sol-execbench/CLAUDE.md`; consult the upstream files for authoritative details.

## Setup

- Prerequisites — Hugging Face CLI (`pip install huggingface-hub[cli]`), NVIDIA driver 580+. (Docker is NO LONGER required after solbench's `docker_removal_plan` landed 2026-05-13; native `uv run` dispatch replaced the upstream Docker entrypoint.) See `sol-execbench/README.md` § Prerequisites for upstream details.
- Download benchmark data — `sol-execbench/scripts/download_data.sh` fetches the HuggingFace SOL-ExecBench plus FlashInfer Trace datasets into `data/`.
- Build and launch — solbench's `docker_removal_plan` (2026-05-13) replaced the upstream Docker entrypoint with native dispatch via `uv run` from the repo root. The upstream `sol-execbench/scripts/run_docker.sh` script is no longer used. The `sol-execbench` CLI is invoked directly: `uv run sol-execbench <problem_dir> --solution <path>`.

## Evaluating a Solution

- CLI entry point (native dispatch via `uv run`) — `sol-execbench <problem_dir> --solution solution.json`, or with explicit files `--definition def.json --workload wkl.jsonl --solution sol.json`.
- One-shot native invocation (post-docker-removal) — `uv run sol-execbench <problem_dir> --solution <path>` from the repo root.
- CLI options — see `sol-execbench/README.md` § CLI Options for `--compile-timeout` (default 120 s), `--timeout` (default 600 s), `--output`, `--json`, `--lock-clocks`, `--keep-staging`, `--verbose`.

## Running a Dataset

- `sol-execbench/scripts/run_dataset.py` evaluates a full dataset or a subset (`--category L1 L2 ...`) in batch; default output `./out/{subset}`.
- Single-problem dry run — `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark/L1/<problem>` from the repo root (no container).

## Problem Format

- `definition.json` — kernel specification: function signature, tensor shapes, dtypes, reference implementation.
- `workload.jsonl` — one JSON object per line, each defining input shapes / values / tolerance thresholds.
- `solution.json` — references source files with the kernel implementation.
- Schema docs — `sol-execbench/docs/definition.md`, `sol-execbench/docs/solution.md`, `sol-execbench/docs/workload.md`, `sol-execbench/docs/trace.md`.

## Architecture (three-layer)

Per `sol-execbench/CLAUDE.md` § Architecture:

- CLI — `sol-execbench/src/sol_execbench/cli/main.py` is the Click-based `sol-execbench` entry point. Parses args, invokes the driver, renders Rich tables with results.
- Driver — `sol-execbench/src/sol_execbench/driver/` orchestrates evaluation. `ProblemPackager` stages problem files into a temp directory; the eval driver template (`driver/templates/eval_driver.py`, ~800 lines) is the self-contained script that runs inside the compile and eval subprocesses.
- Core — `sol-execbench/src/sol_execbench/core/` has two sub-packages: `core/data/` (Pydantic v2 models for Definition / Solution / Workload / Trace) and `core/bench/` (CUDA-event timing, numerical correctness, GPU clock locking, reward-hack detection — detects timing manipulation, stream injection, monkey-patching).

## Examples

- `sol-execbench/examples/cutlass/` — pure C++ CUTLASS submission precedents (`spec.languages: ["cutlass"]`).
- `sol-execbench/examples/cuda_cpp/` — raw CUDA C++ submission precedents (`spec.languages: ["cuda_cpp"]`).
- `sol-execbench/examples/pytorch/`, `sol-execbench/examples/triton/`, `sol-execbench/examples/cute_dsl/`, `sol-execbench/examples/cudnn/`, `sol-execbench/examples/cublas/`, `sol-execbench/examples/cutile/` — additional submission-language example trees in the upstream codebase.

## Critical caveat — CUPTI-vs-paper timing divergence

The local `sol-execbench/` checkout's default benchmark path is **CUPTI-based** (`sol-execbench/src/sol_execbench/core/bench/timing.py`), while the published paper §5.1 describes **CUDA-event** timing; anchor each Tk claim to one authority (paper / leaderboard service / this local checkout) and call out which — see project `CLAUDE.md` Critical Rule 3.
